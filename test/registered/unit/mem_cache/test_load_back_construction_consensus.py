"""LOAD transfer construction must reach consensus before later collectives."""

import datetime
import socket
import time
import types
from unittest.mock import MagicMock

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.unified_cache.components.full_component import (  # noqa: E402
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import (  # noqa: E402
    SWAComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (  # noqa: E402
    ExternalCacheHitMarker,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

REPEATS = 20


class ReachedPrepare(Exception):
    """Raised if a test unexpectedly advances beyond the shared miss path."""


class FakeComponent:
    """Track component slots so every agreed abort must free them."""

    def __init__(self, name):
        self.name = name
        self.failure = None
        self.allocated = 0
        self.abort_error = None

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase == LinkerTransferPhase.LOAD
        if self.failure == "none":
            return None
        if self.failure == "raise":
            raise RuntimeError("build failed")
        if self.failure == "assert":
            raise AssertionError("tree invariant broken")
        self.allocated += 1
        return PoolTransfer(
            name=self.name, device_indices=torch.arange(4), keys=list(keys)
        )

    def update_external_linker_load(self, phase, req, full, transfer, prefix_len, **_):
        if phase == ExternalLinkerLoadPhase.PREPARE:
            raise ReachedPrepare()
        assert phase == ExternalLinkerLoadPhase.ABORT
        if self.abort_error is not None:
            raise self.abort_error
        self.allocated -= 1


def _cache(cp_group, tp_group, components, reductions):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.attn_cp_group = cp_group
    cache.attn_tp_group = tp_group
    cache.tp_world_size = 1
    cache.tree_core = types.SimpleNamespace(
        page_size=1,
        empty_match_result=types.SimpleNamespace(
            device_indices=torch.empty(0, dtype=torch.int64)
        ),
    )
    cache._components_tuple = tuple(components)
    all_reduce = cache._all_reduce_attn_groups

    def counted(tensor, op):
        reductions.append(int(tensor.item()))
        all_reduce(tensor, op)

    cache._all_reduce_attn_groups = counted
    return cache


def _wrapper(cache, reval_group, revalidations):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = types.SimpleNamespace()
    hit = ExternalCacheHitMarker(
        prefix_key=None, tail_hashes=["h0", "h1"], device_hit_len=0
    )
    wrapper.hit_markers = {"r": hit}

    def revalidate_load(transfers):
        revalidations.append(1)
        verdict = torch.ones(1, dtype=torch.int)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=reval_group)
        return False

    wrapper.cache_linker.revalidate_load = revalidate_load
    return wrapper


SCENARIOS = {
    "component_returns_none": {3: (1, "none")},
    "component_raises": {1: (0, "raise")},
    "component_asserts": {3: (0, "assert")},
    "all_constructed": {},
}


def _rank(rank, port, results):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        cp_groups = [dist.new_group(r, backend="gloo") for r in ([0, 1], [2, 3])]
        tp_groups = [dist.new_group(r, backend="gloo") for r in ([0, 2], [1, 3])]
        reval_group = dist.new_group([0, 1, 2, 3], backend="gloo")
        outcomes = {}
        for name, failures in SCENARIOS.items():
            components = [FakeComponent(PoolName.KV), FakeComponent(PoolName.SWA)]
            if rank in failures:
                index, failure = failures[rank]
                components[index].failure = failure
            reps = []
            for _ in range(REPEATS):
                reductions = []
                revalidations = []
                cache = _cache(
                    cp_groups[rank // 2], tp_groups[rank % 2], components, reductions
                )
                wrapper = _wrapper(cache, reval_group, revalidations)
                try:
                    indices, node = wrapper.load_back(
                        types.SimpleNamespace(rid="r", last_node="node")
                    )
                    result = "miss" if indices.numel() == 0 and node == "node" else "?"
                except (RuntimeError, AssertionError) as error:
                    result = f"raised:{error}"
                reps.append(
                    (
                        result,
                        [component.allocated for component in components],
                        len(revalidations),
                        len(reductions),
                    )
                )
            outcomes[name] = reps
        dist.barrier()
        results[rank] = outcomes
    finally:
        dist.destroy_process_group()


class TestConstructionConsensus(CustomTestCase):
    def test_all_ranks_abort_or_revalidate_together(self):
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with mp.Manager() as manager:
            results = manager.dict()
            context = mp.spawn(_rank, args=(port, results), nprocs=4, join=False)
            deadline = time.monotonic() + 180
            try:
                while not context.join(timeout=2):
                    if time.monotonic() > deadline:
                        self.fail("a rank hung before agreeing on LOAD construction")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.kill()
                    process.join(timeout=10)
            outcomes = [results[rank] for rank in range(4)]

        expected = {
            "component_returns_none": ("miss", [0, 0], 0, 1),
            "component_raises": ("miss", [0, 0], 0, 1),
            "component_asserts": ("miss", [0, 0], 0, 1),
            "all_constructed": ("miss", [0, 0], 1, 1),
        }
        raised = {
            ("component_raises", 1): ("raised:build failed", [0, 0], 0, 1),
            ("component_asserts", 3): (
                "raised:tree invariant broken",
                [0, 0],
                0,
                1,
            ),
        }
        for rank, outcome in enumerate(outcomes):
            for name, want in expected.items():
                want = raised.get((name, rank), want)
                with self.subTest(rank=rank, scenario=name):
                    self.assertEqual(len(outcome[name]), REPEATS)
                    for result in outcome[name]:
                        self.assertEqual(result, want)


class _TrackingAllocator:
    def __init__(self, first_id, *, alloc_none=False, available=1 << 20):
        self.next_id = first_id
        self.outstanding = set()
        self.alloc_none = alloc_none
        self.available = available
        self.freed = []

    def available_size(self):
        return self.available

    def alloc(self, num_tokens):
        if self.alloc_none:
            return None
        ids = list(range(self.next_id, self.next_id + num_tokens))
        self.next_id += num_tokens
        self.outstanding.update(ids)
        return torch.tensor(ids, dtype=torch.int64)

    def free(self, indices):
        for slot in indices.tolist():
            assert slot in self.outstanding, f"double or unknown free of {slot}"
            self.outstanding.remove(slot)
            self.freed.append(slot)


class TestFullSwaRollback(CustomTestCase):
    def _run(self, *, swa_alloc_none=False, swa_evict_raises=False):
        full_alloc = _TrackingAllocator(100)
        swa_alloc = _TrackingAllocator(
            500, alloc_none=swa_alloc_none, available=0 if swa_evict_raises else 1 << 20
        )

        def evict(params):
            if getattr(params, "swa_num_tokens", 0):
                raise RuntimeError("swa evict failed")

        cache = types.SimpleNamespace(
            page_size=1,
            is_swa_enabled=True,
            evict=evict,
            token_to_kv_pool_allocator=types.SimpleNamespace(
                full_attn_allocator=full_alloc, swa_attn_allocator=swa_alloc
            ),
            tree_core=types.SimpleNamespace(
                empty_match_result=types.SimpleNamespace(
                    device_indices=torch.empty(0, dtype=torch.int64)
                )
            ),
            _all_reduce_attn_groups=lambda tensor, op: None,
        )
        full = FullComponent.__new__(FullComponent)
        full.cache = cache
        swa = SWAComponent.__new__(SWAComponent)
        swa.cache = cache
        swa.sliding_window_size = 4
        cache._components_tuple = (full, swa)
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = types.SimpleNamespace(revalidate_load=lambda _: False)
        wrapper.hit_markers = {
            "r": ExternalCacheHitMarker(
                prefix_key=None, tail_hashes=["k0", "k1"], device_hit_len=0
            )
        }
        return wrapper, full_alloc, swa_alloc

    def test_swa_miss_frees_full_slots_once(self):
        wrapper, full_alloc, swa_alloc = self._run(swa_alloc_none=True)
        indices, node = wrapper.load_back(types.SimpleNamespace(rid="r", last_node="n"))
        self.assertEqual((indices.numel(), node), (0, "n"))
        self.assertEqual(full_alloc.outstanding, set())
        self.assertEqual(full_alloc.freed, [100, 101])
        self.assertEqual(swa_alloc.freed, [])

    def test_swa_build_error_frees_full_slots_then_raises(self):
        wrapper, full_alloc, swa_alloc = self._run(swa_evict_raises=True)
        with self.assertRaisesRegex(RuntimeError, "swa evict failed"):
            wrapper.load_back(types.SimpleNamespace(rid="r", last_node="n"))
        self.assertEqual(full_alloc.outstanding, set())
        self.assertEqual(full_alloc.freed, [100, 101])
        self.assertEqual(swa_alloc.outstanding, set())
