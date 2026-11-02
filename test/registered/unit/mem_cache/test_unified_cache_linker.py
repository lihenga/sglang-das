import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.cache_action import (
    ReplaceWriteThroughOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeLinker(UnifiedCacheLinker):
    def __init__(self):
        self.layer_done_counter = object()
        self.restorable = []
        self.queued_loads = {}
        self.queued_offloads = []
        self.load_landed = True
        self.completed_offloads = []
        self.reset_count = 0
        self.closed = False

    def lookup(self, rid, transfers):
        return list(self.restorable)

    def load(self, rid, transfers):
        self.queued_loads[rid] = list(transfers)
        return True

    def start_layer_wise_loading(self):
        return 3

    def cancel_queued_load(self, rid):
        if rid not in self.queued_loads:
            return False
        del self.queued_loads[rid]
        return True

    def finish_layer_wise_loading(self, counter_index):
        return self.load_landed

    def offload(self, transfers):
        self.queued_offloads.append(list(transfers))
        return True

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def pop_completed_offload(self):
        return self.completed_offloads.pop(0)

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class _MappingRecorder:
    def __init__(self):
        self.mapping = []

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def _cache_for_wrapper(**kwargs):
    defaults = {
        "tree_core": SimpleNamespace(enable_external_cache_linker=False),
        "write_through_threshold": 256,
        "pp_size": 1,
        "pp_group": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_cache_linker_attachment_is_backend_independent():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_external_cache_linker=False,
        write_through_threshold=256,
    )
    cache.linker = None
    linker = _FakeLinker()

    cache.init_cache_linker(linker)

    assert cache.linker.cache_linker is linker
    assert cache.tree_core.enable_external_cache_linker
    assert cache.write_through_threshold == 1
    assert cache.linker.layer_done_counter is linker.layer_done_counter


def test_restorable_prefix_intersects_sparse_rank_results():
    remote_mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int)

    def intersect_remote_mask(mask, op):
        assert op == torch.distributed.ReduceOp.MIN
        mask.copy_(torch.minimum(mask, remote_mask))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=intersect_remote_mask)
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    hit_pages = wrapper._sync_restorable_prefix([2, 4], num_pages=4, device_hit_pages=0)

    assert hit_pages == 2


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(
        id=node_id,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda node_ids, ack_id: (
                setattr(node, "write_through_pending_id", ack_id) or list(node_ids)
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
        resolve_node_handle=lambda value: node if value == node_id else None,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node_id])

    assert locks == [node_id]
    assert node.external_cache_stored
    assert not unlocks

    linker.completed_offloads.append(False)
    completed = wrapper.take_completed_offloads(finish_count=1)
    wrapper.commit_completed_offloads(completed)

    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


def test_layer_counter_reports_failure_after_the_forward():
    ready = []
    counter = LayerWiseLoadCounter(2, lambda index, layer: ready.append(layer))
    index = counter.update_producer()
    counter.set_consumer(index)
    counter.complete(index, 0)
    counter.fail(index, RuntimeError("storage read failed"))

    counter.wait_until(0)
    counter.wait_until(1)  # does not raise into the forward

    assert ready == [0, 1]  # an MLA dedup source still broadcasts layer 1
    assert counter.finish(index) is False


def test_layer_counter_finish_waits_for_layers_still_loading():
    counter = LayerWiseLoadCounter(1)
    index = counter.update_producer()
    loader = threading.Timer(0.05, counter.complete, (index, 0))
    loader.start()

    assert counter.finish(index) is True  # the forward never waited on layer 0
    loader.join()


def test_finish_loads_adopts_a_peer_rank_failure():
    reduce = MagicMock(side_effect=lambda value, op: value.fill_(0))
    wrapper = UnifiedCacheLinkerWrapper(
        _cache_for_wrapper(_all_reduce_attn_groups=reduce), _FakeLinker()
    )
    wrapper.inflight_load_rids = ["a", "b"]
    wrapper.start_layer_wise_loading()

    assert wrapper.finish_loads() == ["a", "b"]  # this rank landed, a peer did not
    assert wrapper.finish_loads() == []  # nothing loaded: no collective
    reduce.assert_called_once()


def test_release_request_cancels_queued_load():
    linker = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(_cache_for_wrapper(), linker)
    wrapper.hit_markers["rid"] = object()
    wrapper.inflight_load_rids = ["rid"]
    wrapper.private_load_indices["rid"] = torch.arange(4)
    linker.queued_loads["rid"] = [object()]

    wrapper.release_request("rid")

    assert wrapper.hit_markers == {}
    assert wrapper.inflight_load_rids == []
    assert wrapper.private_load_indices == {}
    assert "rid" not in linker.queued_loads


def test_failed_offload_rolls_back_split_fragments():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    child = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    parent = SimpleNamespace(
        id=8,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    nodes = {child.id: child, parent.id: parent}

    def mark_pending(node_ids, ack_id):
        for node_id in node_ids:
            nodes[node_id].write_through_pending_id = ack_id
        return list(node_ids)

    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=mark_pending,
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: lock_params),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
        resolve_node_handle=nodes.__getitem__,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes([child.id])

    parent.external_cache_stored = child.external_cache_stored
    parent.write_through_pending_id = child.write_through_pending_id
    wrapper.replace_pending_offload_node(child.id, child.id, [parent.id, child.id])
    linker.completed_offloads.append(False)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(finish_count=1))

    assert not parent.external_cache_stored
    assert not child.external_cache_stored
    assert parent.write_through_pending_id is None
    assert child.write_through_pending_id is None
    assert unlocks == [(child.id, lock_params)]


def test_split_action_retargets_pending_external_offload():
    calls = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        replace_pending_offload_node=lambda *args: calls.append(("linker", *args))
    )
    cache._replace_pending_write_through_node = lambda *args: calls.append(
        ("hicache", *args)
    )
    action = ReplaceWriteThroughOnNodeSplit(
        ack_id=7,
        old_node_id=7,
        new_node_id=8,
        new_child_node_id=7,
    )

    cache._apply_cache_action(action)

    assert calls == [
        ("hicache", 7, 7, [8, 7]),
        ("linker", 7, 7, [8, 7]),
    ]


def test_reset_quiesces_backend_before_releasing_pending_locks():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    events = []

    class _QuiescentFakeLinker(_FakeLinker):
        def reset(self):
            events.append("backend")
            super().reset()

    linker = _QuiescentFakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda node_ids, ack_id: (
                setattr(node, "write_through_pending_id", ack_id) or list(node_ids)
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=object),
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id)),
        resolve_node_handle=lambda node_id: node,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.inflight_load_rids = ["rid"]
    wrapper.private_load_indices["rid"] = torch.arange(4)
    wrapper.offload_nodes([node.id])

    wrapper.reset()

    assert events == ["backend", ("unlock", node.id)]
    assert wrapper.inflight_load_rids == []
    assert wrapper.private_load_indices == {}
    assert wrapper.pending_offloads == []
    assert not node.external_cache_stored
    assert node.write_through_pending_id is None


def test_close_quiesces_backend_before_releasing_pending_loads():
    events = []

    class _ClosingFakeLinker(_FakeLinker):
        def close(self):
            events.append("backend")
            super().close()

    linker = _ClosingFakeLinker()
    cache = _cache_for_wrapper(
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id))
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.inflight_load_rids = ["rid"]
    wrapper.private_load_indices["rid"] = torch.arange(4)

    wrapper.close()

    assert events == ["backend"]
    assert linker.closed
    assert wrapper.inflight_load_rids == []
    assert wrapper.private_load_indices == {}


def test_check_hicache_events_commits_common_rank_results():
    committed = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        num_completed_offloads=lambda: 3,
        take_completed_offloads=lambda count: [True] * count,
        commit_completed_offloads=committed.append,
    )

    reduce_calls = 0

    def reduce_to_common_state(value, op):
        nonlocal reduce_calls
        assert op == torch.distributed.ReduceOp.MIN
        reduce_calls += 1
        if reduce_calls == 1:
            value.copy_(torch.tensor([1]))
        else:
            value.fill_(0)

    cache._all_reduce_attn_groups = reduce_to_common_state

    cache.check_hicache_events()

    assert committed == [[False]]


@pytest.mark.parametrize("landed", [True, False])
def test_finish_loads_releases_private_index_references(landed, caplog):
    backend = _FakeLinker()
    backend.load_landed = landed
    wrapper = UnifiedCacheLinkerWrapper(
        _cache_for_wrapper(_all_reduce_attn_groups=lambda value, op: None), backend
    )
    wrapper.inflight_load_rids = ["rid"]
    wrapper.private_load_indices["rid"] = torch.arange(4)
    wrapper.start_layer_wise_loading()

    assert wrapper.finish_loads() == ([] if landed else ["rid"])
    assert wrapper.private_load_indices == {}
    assert wrapper.inflight_load_rids == []
    assert ("External linker load failed" in caplog.text) is (not landed)


@pytest.mark.parametrize("landed", [True, False])
def test_finished_load_marks_only_affected_requests(landed):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    backend = _FakeLinker()
    backend.load_landed = landed
    wrapper = UnifiedCacheLinkerWrapper(
        _cache_for_wrapper(_all_reduce_attn_groups=lambda value, op: None), backend
    )
    cache.linker = wrapper
    wrapper.inflight_load_rids = ["loading"]
    wrapper.start_layer_wise_loading()
    loading = SimpleNamespace(
        rid="loading", skip_radix_cache_insert=False, discard_output_reason=None
    )
    resident = SimpleNamespace(
        rid="resident", skip_radix_cache_insert=False, discard_output_reason=None
    )

    failed = cache.finish_external_linker_loads([loading, resident])

    assert failed == ([] if landed else [loading])
    assert loading.skip_radix_cache_insert is (not landed)
    if not landed:
        assert loading.discard_output_reason.status_code == 503
    assert resident.discard_output_reason is None
    assert not resident.skip_radix_cache_insert
    assert not cache.load_back_is_cache_owned()
    cache.linker = None
    assert cache.load_back_is_cache_owned()
    assert cache.finish_external_linker_loads([resident]) == []


@pytest.mark.parametrize("with_swa", [False, True])
@pytest.mark.parametrize(
    "outcome", ["queued", "refused", "exception", "allocation_failure"]
)
def test_load_back_keeps_tail_private_and_rolls_back_refusal(with_swa, outcome):
    events = []
    full_indices = torch.tensor([10, 11, 12, 13])
    swa_indices = torch.tensor([20, 21])
    full = PoolTransfer(name=PoolName.KV, keys=["a", "b"], device_indices=full_indices)
    swa = PoolTransfer(name=PoolName.SWA, keys=["b"], device_indices=swa_indices)
    mapping = _MappingRecorder()
    cache = _cache_for_wrapper(
        page_size=2,
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            empty_match_result=SimpleNamespace(
                device_indices=torch.empty(0, dtype=torch.int64)
            ),
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            full_attn_allocator=SimpleNamespace(
                free=lambda value: events.append(("free_full", value))
            ),
            swa_attn_allocator=SimpleNamespace(
                free=lambda value: events.append(("free_swa", value))
            ),
            set_full_to_swa_mapping=mapping.set_full_to_swa_mapping,
        ),
        insert=MagicMock(side_effect=AssertionError("load must not publish the tail")),
    )
    full_component = FullComponent.__new__(FullComponent)
    full_component.cache = cache
    full_component.component_type = ComponentType.FULL
    full_component._full_allocator = lambda: (
        cache.token_to_kv_pool_allocator.full_attn_allocator
    )
    full_component.build_external_linker_transfer = MagicMock(
        return_value=None if outcome == "allocation_failure" and not with_swa else full
    )
    swa_component = SWAComponent.__new__(SWAComponent)
    swa_component.cache = cache
    swa_component.component_type = ComponentType.SWA
    swa_component.sliding_window_size = 2
    swa_component.build_external_linker_transfer = MagicMock(
        return_value=None if outcome == "allocation_failure" else swa
    )
    cache._components_tuple = (
        (full_component, swa_component) if with_swa else (full_component,)
    )
    backend = _FakeLinker()
    if outcome == "refused":
        backend.load = MagicMock(return_value=False)
    elif outcome == "exception":
        backend.load = MagicMock(side_effect=RuntimeError("queue failed"))
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
        tail_hashes=["a", "b"], device_hit_len=2
    )
    req = SimpleNamespace(
        rid="rid",
        last_node=7,
        kv=SimpleNamespace(cache_protected_len=2, swa_evicted_seqlen=0),
        swa_branching_seqlen=4,
        prefix_indices=torch.tensor([0, 1]),
    )
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="queue failed"):
            wrapper.load_back(req)
    else:
        loaded, last_node = wrapper.load_back(req)
        assert last_node == 7
        assert loaded.tolist() == (full_indices.tolist() if outcome == "queued" else [])
    cache.insert.assert_not_called()
    assert req.kv.cache_protected_len == 2
    assert req.prefix_indices.tolist() == [0, 1]
    if outcome == "queued":
        assert not events
        assert wrapper.inflight_load_rids == ["rid"]
        assert torch.equal(wrapper.private_load_indices["rid"], full_indices)
        assert req.swa_branching_seqlen is None
        if with_swa:
            assert req.kv.swa_evicted_seqlen == 4
            assert len(mapping.mapping) == 1
            assert mapping.mapping[0][0].tolist() == [12, 13]
            assert mapping.mapping[0][1].tolist() == [20, 21]
    else:
        assert wrapper.inflight_load_rids == []
        assert wrapper.private_load_indices == {}
        assert req.swa_branching_seqlen == 4
        assert req.kv.swa_evicted_seqlen == 0
        assert not mapping.mapping
        if outcome == "allocation_failure":
            expected = ["free_full"] if with_swa else []
        else:
            expected = ["free_swa", "free_full"] if with_swa else ["free_full"]
        assert [event[0] for event in events] == expected


def test_no_load_does_not_enter_collective():
    reduce = MagicMock(side_effect=AssertionError("unexpected collective"))
    wrapper = UnifiedCacheLinkerWrapper(
        _cache_for_wrapper(_all_reduce_attn_groups=reduce), _FakeLinker()
    )
    assert wrapper.finish_loads() == []
    reduce.assert_not_called()


def test_started_load_cannot_be_canceled_before_finish():
    backend = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(_cache_for_wrapper(), backend)
    wrapper.inflight_load_rids = ["rid"]
    wrapper.private_load_indices["rid"] = torch.arange(4)
    wrapper.release_request("rid")
    assert wrapper.inflight_load_rids == ["rid"]
    assert "rid" in wrapper.private_load_indices


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
