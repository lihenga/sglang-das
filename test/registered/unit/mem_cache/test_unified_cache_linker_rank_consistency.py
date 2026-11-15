"""Direct-L3 failures must preserve a common TP/CP scheduler state."""

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache_linker import UnifiedCacheLinkerWrapper
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _reduce(tensor, op):
    # Exercise the same attention-group reduction used by the service.
    groups = NS(
        attn_cp_group=dist.group.WORLD,
        attn_tp_group=None,
        tp_world_size=dist.get_world_size(),
        tp_group=dist.group.WORLD,
    )
    UnifiedRadixCache._all_reduce_attn_groups(groups, tensor, op)


def _load_case(rank, fail_pool):
    allocated, freed, evictions, inserts, queued = [], [], [], [], []

    def alloc(n):
        if rank == 1 and fail_pool == "full":
            return None
        slots = torch.arange(n, dtype=torch.int64)
        allocated.append(slots)
        return slots

    allocator = NS(
        available_size=lambda: 0 if rank == 1 else 4096,
        alloc=alloc,
        free=freed.append,
    )
    root = NS(id=0)
    node = NS(id=7, parent=root, external_cache_stored=False)
    cache = NS(
        page_size=64,
        pp_size=1,
        evict=lambda params: evictions.append(params.num_tokens),
        _all_reduce_attn_groups=_reduce,
        tree_core=NS(
            empty_match_result=NS(device_indices=torch.empty(0, dtype=torch.int64)),
            collect_full_device_indices=lambda *_: allocated[0],
        ),
        insert=lambda params: inserts.append(params)
        or NS(
            last_device_node=7,
            mamba_exist=False,
            adopted_ranges={ComponentType.FULL: [(256, 384)]},
        ),
        inc_lock_ref=lambda _: NS(to_dec_params=lambda: "lock"),
        dec_lock_ref=lambda *_: None,
        resolve_node_handle=lambda _: node,
    )
    full = FullComponent.__new__(FullComponent)
    full.cache = cache
    full.component_type = ComponentType.FULL
    full._full_allocator = lambda: allocator
    cache._components_tuple = (full,)
    if fail_pool == "second":
        # Both ranks entered the second component, but only rank 0 allocated it.
        def second_build(*_):
            return (
                None
                if rank == 1
                else PoolTransfer(
                    name=PoolName.SWA, keys=["k0"], device_indices=torch.arange(64)
                )
            )

        def second_update(phase, req, full_transfer, transfer, prefix_len, **kw):
            assert phase == ExternalLinkerLoadPhase.ABORT
            freed.append(transfer.device_indices)

        cache._components_tuple += (
            NS(
                build_external_linker_transfer=second_build,
                update_external_linker_load=second_update,
            ),
        )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper._waiting_queue_prefetch_enabled = False
    wrapper.hit_markers = {
        "rid": NS(
            device_hit_len=256, tail_hashes=["k0", "k1"], prefix_key=list(range(384))
        )
    }
    wrapper.pending_loads = {}
    wrapper.cache_linker = NS(load=lambda rid, transfers: queued.append(rid) or True)
    req = NS(rid="rid", last_node=0, prefix_indices=torch.arange(256), kv=None)
    tail, endpoint = wrapper.load_back(req)
    assert evictions == [128], "even a rank with free capacity must evict consistently"
    if fail_pool:
        assert tail.numel() == 0 and endpoint == 0
        assert not inserts and not queued and not wrapper.pending_loads
        expected_frees = 0 if rank == 1 and fail_pool == "full" else 1
        if rank != 1 and fail_pool == "second":
            expected_frees = 2
        assert len(freed) == expected_frees
    else:
        assert tail.numel() == 128 and endpoint == 7
        assert len(inserts) == 1 and queued == ["rid"] and not freed


def _offload_case(rank):
    nodes = {n: NS(id=n, external_cache_stored=True) for n in (7, 8)}
    unlocked, retried = [], []
    refs = {7: 1, 8: 1}
    outcomes = iter([rank == 0, True])
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)

    def unlock(node_id, lock):
        unlocked.append(node_id)
        refs[node_id] -= 1

    def lock(node_id):
        refs[node_id] += 1
        return NS(to_dec_params=lambda: "lock")

    wrapper.cache = NS(
        resolve_node_handle=nodes.__getitem__,
        inc_lock_ref=lock,
        dec_lock_ref=unlock,
        _all_reduce_attn_groups=_reduce,
        _components_tuple=(
            NS(
                build_external_linker_transfer=lambda phase, node, keys: PoolTransfer(
                    name=PoolName.KV, keys=[str(node.id)]
                )
            ),
        ),
    )
    wrapper.cache_linker = NS(
        pop_completed_offload=lambda: next(outcomes),
        offload=lambda transfers: retried.extend(transfers[0].keys) or True,
    )
    wrapper.pending_offloads = [(7, "lock"), (8, "lock")]
    wrapper.drain_offloads(2)
    assert not nodes[7].external_cache_stored and nodes[8].external_cache_stored
    assert unlocked == [7, 8] and not wrapper.pending_offloads
    assert refs == {7: 0, 8: 0}
    wrapper.offload_nodes([7, 8])
    assert retried == ["7"] and wrapper.pending_offloads == [(7, "lock")]
    assert refs == {7: 1, 8: 0}
    # A globally empty completion batch must not issue a collective.
    wrapper.cache._all_reduce_attn_groups = lambda *_: pytest.fail("empty collective")
    wrapper.drain_offloads(0)


def _adder(capacity, linker=True, locked_capacity=None):
    current = [capacity]
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.tree_cache = NS(
        linker=object() if linker else None,
        disable=False,
        evictable_size=lambda: 0,
        _all_reduce_attn_groups=_reduce,
    )
    adder.token_to_kv_pool_allocator = NS(available_size=lambda: current[0])
    adder.is_all_swa = adder.is_hybrid_swa = adder.is_hybrid_ssm_cache = False
    adder.rem_total_token_offset = adder.cur_rem_token_offset = 0
    adder.rem_swa_token_offset = 0
    adder.rem_mamba_slots = None
    adder._linker_token_budgets = None
    adder.page_size = 64
    adder.rem_chunk_tokens = adder.rem_input_tokens = 16384
    adder.can_run_list = []
    adder.dsa_prefill_cp_in_seq_split = False
    adder.prefill_max_requests = adder.prefill_delayer_single_pass = None
    adder.dllm_config = None
    adder._mamba_slot_cost = 0
    adder._check_prefill_tile_budget = lambda _: None
    adder._req_inc_lock_ref = lambda _: None
    adder._update_prefill_budget = lambda *_, **kw: None

    @contextmanager
    def lock(_):
        if locked_capacity is not None:
            current[0] = locked_capacity
        yield
        current[0] = capacity

    adder._lock_node = lock
    return adder


def _req(tokens=128):
    req = NS(
        sampling_params=NS(ignore_eos=False, max_new_tokens=100),
        output_ids=[],
        full_untruncated_fill_ids=list(range(tokens)),
        prefix_indices=torch.empty(0),
        last_node=0,
        host_hit_length=0,
        swa_host_hit_length=0,
        storage_hit_length=0,
        retracted_stain=False,
        needs_host_load_back=lambda: False,
    )
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", NS(length=end - start)
    )
    return req


def _scheduler_case(rank):
    for after_lock in (False, True):
        capacity = 1024 if after_lock or rank == 0 else 200
        adder = _adder(
            capacity, locked_capacity=200 if after_lock and rank == 1 else None
        )
        assert adder.add_one_req(_req(), False, None) == AddReqResult.NO_TOKEN
        assert not adder.can_run_list
    adder = _adder(6000 if rank == 0 else 16368)
    req = _req(100000)
    assert adder.add_chunked_req(req) is req
    assert req.extend_range.length == 6000
    # Subsequent local reads must use the snapshot, without more collectives.
    adder.rem_total_token_offset += 64
    assert adder.rem_total_tokens == 5936
    adder = _adder(1024)
    assert adder.add_one_req(_req(), False, None) == AddReqResult.CONTINUE
    assert len(adder.can_run_list) == 1
    adder = _adder(0 if rank == 1 else 1024)
    assert adder.budget_state() == AddReqResult.NO_TOKEN


def _swa_case(rank):
    evictions = []
    swa = SWAComponent.__new__(SWAComponent)
    swa.sliding_window_size = 64
    swa.cache = NS(
        page_size=64,
        _all_reduce_attn_groups=_reduce,
        evict=lambda params: evictions.append(params.swa_num_tokens),
        token_to_kv_pool_allocator=NS(
            swa_attn_allocator=NS(
                available_size=lambda: 0 if rank == 1 else 4096,
                alloc=lambda n: torch.arange(n),
            )
        ),
    )
    transfer = swa.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["k0", "k1"]
    )
    assert evictions == [64] and transfer.keys == ["k1"]
    assert transfer.device_indices.numel() == 64


def _worker(rank, rendezvous, world_size):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        for failed_pool in ("full", "second", None):
            _load_case(rank, failed_pool)
        _offload_case(rank)
        _scheduler_case(rank)
        _swa_case(rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 8])
def test_rank_local_failures_preserve_common_state(tmp_path, world_size):
    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", world_size),
        nprocs=world_size,
        join=True,
    )


def test_non_linker_admission_keeps_local_behavior():
    for capacity, result in (
        (200, AddReqResult.NO_TOKEN),
        (1024, AddReqResult.CONTINUE),
    ):
        adder = _adder(capacity, linker=False)
        adder.tree_cache._all_reduce_attn_groups = lambda *_: pytest.fail(
            "non-linker collective"
        )
        assert adder.add_one_req(_req(), False, None) == result
