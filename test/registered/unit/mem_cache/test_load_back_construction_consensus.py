"""load_back agrees on LOAD transfer construction before any later collective.

Building a LOAD transfer allocates device slots and can fail on one rank
alone. Every rank must then abort together, each freeing what it built and
any claimed prefetch session, instead of one rank returning while the others
enter revalidate_load or the load itself.
"""

import datetime
import socket
import threading
import time
import types
import unittest
from unittest.mock import MagicMock

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (  # noqa: E402
    MooncakeDirectLinker,
)
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
SESSION = "__sglang_waiting_queue_prefetch__:r:1"


class BuildStop(BaseException):
    """A non-Exception interruption raised while building."""


class FakeComponent:
    """A persistent allocator: one unit per built transfer, ABORT frees it."""

    def __init__(self, name):
        self.name = name
        self.failure = None
        self.allocated = 0

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase == LinkerTransferPhase.LOAD
        if self.failure == "none":
            return None
        if self.failure == "raise":
            raise RuntimeError("build failed")
        if self.failure == "stop":
            raise BuildStop()
        self.allocated += 1
        return PoolTransfer(
            name=self.name, device_indices=torch.arange(4), keys=list(keys)
        )

    def update_external_linker_load(self, phase, req, full, transfer, prefix_len, **_):
        assert phase == ExternalLinkerLoadPhase.ABORT
        self.allocated -= 1


def _prefetch_backend(other_holder):
    """A real Mooncake linker holding one ready prefetch session for "r";
    only the native store calls are stubbed."""
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_lock = threading.Lock()
    linker.session_lock = threading.Lock()
    linker.host_prefetch_entries = {
        "r": {
            "state": "dfs_prefetched",
            "cancelled": False,
            "session_rid": SESSION,
            "reserved_bytes": 8 << 10,
        }
    }
    linker.prepared_load_sessions = {SESSION: ["k0"]}
    linker.session_refcounts = {"k0": 2 if other_holder else 1}
    linker.session_sources = {"k0": "dfs"}
    if other_holder:
        linker.prepared_load_sessions["other"] = ["k0"]
    ended = []
    linker.storage = types.SimpleNamespace(
        store=types.SimpleNamespace(
            batch_get_session_refresh=lambda keys: [0] * len(keys),
            batch_get_session_end=lambda keys: ended.append(list(keys)),
        )
    )
    return linker, ended


def _session_released(linker, ended, other_holder):
    if linker.host_prefetch_entries or "r" in linker.prepared_load_sessions:
        return False
    if SESSION in linker.prepared_load_sessions:
        return False
    if other_holder:
        return linker.session_refcounts == {"k0": 1} and ended == []
    return linker.session_refcounts == {} and ended == [["k0"]]


# scenario -> (path, {rank: (component index, failure)}, other session holder)
SCENARIOS = {
    "normal_second_component_none": ("normal", {3: (1, "none")}, False),
    "normal_first_component_raises": ("normal", {1: (0, "raise")}, False),
    "prefetch_one_rank_fails": ("prefetch", {2: (0, "none")}, False),
    "prefetch_second_component_raises": ("prefetch", {0: (1, "raise")}, False),
    "prefetch_fails_with_other_holder": ("prefetch", {2: (0, "none")}, True),
    "all_built": ("normal", {}, False),
}


def _cache(cp_group, tp_group, components, reductions):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.attn_cp_group = cp_group
    cache.attn_tp_group = tp_group
    cache.tp_world_size = 1
    # page_size is a property backed by tree_core.
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


def _wrapper(cache, backend, prefetch):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = backend
    hit = ExternalCacheHitMarker(
        prefix_key=None, tail_hashes=["h0", "h1"], device_hit_len=0
    )
    wrapper.hit_markers = {"r": hit}
    wrapper.host_prefetch_hits = {"r": hit} if prefetch else {}
    wrapper.pending_host_prefetch_submissions = {}
    return wrapper


def _rank(rank, world_size, port, results):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        cp = [dist.new_group(r, backend="gloo") for r in ([0, 1], [2, 3])]
        tp = [dist.new_group(r, backend="gloo") for r in ([0, 2], [1, 3])]
        # LayerSplit-style revalidation group: its MIN hangs if a rank skips it.
        reval = dist.new_group([0, 1, 2, 3], backend="gloo")
        out = {}
        for name, (path, failures, other_holder) in SCENARIOS.items():
            # One allocator pair for all repeats: a leak accumulates.
            components = [FakeComponent(PoolName.KV), FakeComponent(PoolName.SWA)]
            if rank in failures:
                index, failure = failures[rank]
                components[index].failure = failure
            reps = []
            for _ in range(REPEATS):
                reductions = []
                cache = _cache(cp[rank // 2], tp[rank % 2], components, reductions)
                revalidated = []
                if path == "prefetch":
                    backend, ended = _prefetch_backend(other_holder)
                else:
                    backend, ended = MagicMock(), None

                    def revalidate_load(transfers, _seen=revalidated):
                        _seen.append(1)
                        verdict = torch.ones(1, dtype=torch.int)
                        dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=reval)
                        return False  # stop right after it: a plain miss

                    backend.revalidate_load = revalidate_load
                wrapper = _wrapper(cache, backend, path == "prefetch")
                indices, node = wrapper.load_back(
                    types.SimpleNamespace(rid="r", last_node="node")
                )
                reps.append(
                    (
                        int(indices.numel()) == 0 and node == "node",
                        [c.allocated for c in components],
                        len(revalidated),
                        (
                            _session_released(backend, ended, other_holder)
                            if path == "prefetch"
                            else None
                        ),
                        len(reductions),
                    )
                )
            out[name] = reps
        dist.barrier()
        results[rank] = out
    finally:
        dist.destroy_process_group()


class TestConstructionConsensus(CustomTestCase):
    def test_every_rank_aborts_together_every_time(self):
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with mp.Manager() as manager:
            results = manager.dict()
            context = mp.spawn(_rank, args=(4, port, results), nprocs=4, join=False)
            deadline = time.monotonic() + 180
            try:
                while not context.join(timeout=2):
                    if time.monotonic() > deadline:
                        self.fail("a rank hung: construction was not agreed")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.kill()
                    process.join(timeout=10)
            outcomes = [results[rank] for rank in range(4)]

        # (miss, allocated, revalidations, session released, reductions)
        expected = {
            "normal_second_component_none": (True, [0, 0], 0, None, 1),
            "normal_first_component_raises": (True, [0, 0], 0, None, 1),
            # ready, valid, claimed, then construction.
            "prefetch_one_rank_fails": (True, [0, 0], 0, True, 4),
            "prefetch_second_component_raises": (True, [0, 0], 0, True, 4),
            "prefetch_fails_with_other_holder": (True, [0, 0], 0, True, 4),
            # Agreement passes, revalidate_load runs, then its miss frees all.
            "all_built": (True, [0, 0], 1, None, 1),
        }
        for rank, out in enumerate(outcomes):
            for name, want in expected.items():
                with self.subTest(rank=rank, scenario=name):
                    self.assertEqual(len(out[name]), REPEATS)
                    for rep, got in enumerate(out[name]):
                        self.assertEqual(got, want, f"repeat {rep}")


class TestInterruptDuringConstruction(CustomTestCase):
    """A non-Exception interruption still joins the agreement and frees what
    was built before it is re-raised."""

    def _load_back(self, prefetch):
        reductions = []
        first, second = FakeComponent(PoolName.KV), FakeComponent(PoolName.SWA)
        second.failure = "stop"
        cache = types.SimpleNamespace(
            page_size=1,
            tree_core=types.SimpleNamespace(
                empty_match_result=types.SimpleNamespace(
                    device_indices=torch.empty(0, dtype=torch.int64)
                )
            ),
            _components_tuple=(first, second),
            _all_reduce_attn_groups=lambda tensor, op: reductions.append(
                int(tensor.item())
            ),
        )
        backend = MagicMock()
        backend.get_host_prefetch_status.return_value = "dfs_prefetched"
        backend.revalidate_host_prefetch.return_value = True
        backend.claim_ready_host_prefetch.return_value = True
        wrapper = _wrapper(cache, backend, prefetch)
        with self.assertRaises(BuildStop):
            wrapper.load_back(types.SimpleNamespace(rid="r", last_node="node"))
        return reductions, first, backend

    def test_normal_path(self):
        reductions, first, backend = self._load_back(prefetch=False)
        self.assertEqual(reductions, [0])
        self.assertEqual(first.allocated, 0)
        backend.revalidate_load.assert_not_called()

    def test_prefetch_path(self):
        reductions, first, backend = self._load_back(prefetch=True)
        self.assertEqual(reductions, [1, 1, 1, 0])
        self.assertEqual(first.allocated, 0)
        backend.abort_prepared_load.assert_called_once_with("r")


class TestComponentRollback(CustomTestCase):
    class _Slots:
        def to(self, dtype):
            raise RuntimeError("conversion failed")

    def test_full_component_frees_slots_it_could_not_hand_over(self):
        component = FullComponent.__new__(FullComponent)
        allocator = MagicMock()
        allocator.available_size.return_value = 1 << 20
        slots = self._Slots()
        allocator.alloc.return_value = slots
        component.cache = types.SimpleNamespace(page_size=1, evict=MagicMock())
        component._full_allocator = lambda: allocator
        with self.assertRaises(RuntimeError):
            component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, ["k0", "k1"]
            )
        allocator.free.assert_called_once_with(slots)

    def test_swa_component_frees_slots_it_could_not_hand_over(self):
        component = SWAComponent.__new__(SWAComponent)
        allocator = MagicMock()
        allocator.available_size.return_value = 1 << 20
        slots = self._Slots()
        allocator.alloc.return_value = slots
        component.sliding_window_size = 4
        component.cache = types.SimpleNamespace(
            page_size=1,
            evict=MagicMock(),
            token_to_kv_pool_allocator=types.SimpleNamespace(
                swa_attn_allocator=allocator
            ),
        )
        with self.assertRaises(RuntimeError):
            component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, ["k0", "k1"]
            )
        allocator.free.assert_called_once_with(slots)


if __name__ == "__main__":
    unittest.main()
