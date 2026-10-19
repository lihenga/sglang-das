from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import EXTERNAL_KV_LOAD_ERR_TYPE
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.linker_fault_injection import (
    InjectedLinkerLoadFailure,
    arm_load_failure_injection,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_cache_linker import UnifiedCacheLinkerWrapper
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_layer_wise_load_failure_does_not_escape_wait_hook():
    counter = LayerWiseLoadCounter(num_layers=2)
    index = counter.update_producer()
    counter.set_consumer(index)
    counter.fail(index, ValueError("remote read failed"))

    counter.wait_until(0)
    counter.wait_until(1)

    assert index not in counter.futures


def test_load_thread_publishes_failed_completion():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.load_queue = Queue()
    linker.completed_loads = Queue()
    linker.layer_done_counter = LayerWiseLoadCounter(num_layers=1)
    linker.load_layer_wise = lambda _index, _transfers: False

    linker.load_queue.put(
        (0, {"rid-a": [object()]}, SimpleNamespace(synchronize=lambda: None))
    )
    linker.load_queue.put(None)
    linker.load_thread_func()

    assert linker.pop_completed_load() == (["rid-a"], False)


def test_linker_wrapper_commits_failure_and_detaches_chain():
    detached = []
    unlocked = []
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.failed_chains = {}
    wrapper.taken_loads = [(["rid-a"], False)]
    wrapper.pending_loads = {"rid-a": (7, "lock", 3)}
    wrapper.cache = SimpleNamespace(
        tree_core=SimpleNamespace(
            detach_external_load_chain=lambda endpoint, anchor: detached.append(
                (endpoint, anchor)
            )
            or [endpoint, anchor],
        ),
        dec_lock_ref=lambda node_id, lock_params: unlocked.append(
            (node_id, lock_params)
        ),
    )

    failed = wrapper.commit_completed_loads([False])

    assert failed == ["rid-a"]
    assert detached == [(7, 3)]
    assert unlocked == [(7, "lock")]
    assert wrapper.take_failed_chain("rid-a") == [7, 3]


def test_release_request_keeps_published_load_and_lock_until_completion():
    detached = []
    unlocked = []
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.hit_markers = {}
    wrapper._waiting_queue_prefetch_enabled = False
    wrapper.failed_chains = {}
    wrapper.pending_loads = {"rid-a": (7, "lock", 3)}
    wrapper.cache_linker = SimpleNamespace(cancel_queued_load=lambda _rid: False)
    wrapper.cache = SimpleNamespace(
        tree_core=SimpleNamespace(
            detach_external_load_chain=lambda endpoint, anchor: detached.append(
                (endpoint, anchor)
            )
            or [endpoint],
        ),
        dec_lock_ref=lambda node_id, lock_params: unlocked.append(
            (node_id, lock_params)
        ),
    )

    wrapper.release_request("rid-a")

    assert detached == []
    assert unlocked == []
    assert wrapper.pending_loads == {"rid-a": (7, "lock", 3)}
    assert wrapper.failed_chains == {}


def test_unified_radix_event_path_records_failed_load_rids():
    committed = []
    cache = SimpleNamespace()
    cache.linker = SimpleNamespace(
        num_completed_loads=lambda: 1,
        num_completed_offloads=lambda: 0,
        take_completed_loads=lambda count: [False] * count,
        commit_completed_loads=lambda successes: committed.extend(successes)
        or ["rid-a"],
        drain_offloads=lambda count: None,
    )
    cache._failed_linker_rids = []
    cache._all_reduce_attn_groups = lambda _tensor, _op: None

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache._collect_failed_linker_loads = (
        UnifiedRadixCache._collect_failed_linker_loads.__get__(cache)
    )
    UnifiedRadixCache.check_hicache_events(cache)

    assert committed == [False]
    assert cache._failed_linker_rids == ["rid-a"]


def test_scheduler_marks_failed_load_request_for_abort():
    trace = SimpleNamespace(abort=lambda **_kwargs: None)
    req = SimpleNamespace(
        rid="rid-a",
        last_node=7,
        external_load_retry_count=0,
        to_finish=None,
        finished_reason=None,
        skip_radix_cache_insert=False,
        time_stats=SimpleNamespace(trace_ctx=trace),
        finished=lambda: False,
        disagg_kv_sender=None,
        external_kv_cleanup_done=False,
        external_kv_abort_requested=False,
        external_kv_pending_chunk_cleared=False,
        external_kv_linker_released=False,
        external_kv_sender_abort_requested=False,
    )
    released = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.__dict__.update(
        _deferred_linker_rids=set(),
        _external_kv_abort_rids=set(),
        _external_kv_cleanup_rids=set(),
        disagg_prefill_pending_chunk_rids=set(),
        tree_cache=SimpleNamespace(
            drain_linker_loads=lambda: ["rid-a"],
            has_outstanding_failed_linker_chains=lambda: False,
            is_on_failed_linker_chain=lambda _node_id: False,
        ),
        running_batch=SimpleNamespace(is_empty=lambda: True),
        chunked_req=None,
        _release_aborted_request=released.append,
        enable_external_load_retry=False,
        external_load_retry_max=1,
    )

    Scheduler._mark_failed_linker_loads(scheduler, SimpleNamespace(reqs=[req]))

    assert released == ["rid-a"]
    assert isinstance(req.to_finish, FINISH_ABORT)
    assert req.to_finish.err_type == EXTERNAL_KV_LOAD_ERR_TYPE
    assert req.skip_radix_cache_insert


def test_scheduler_requeues_failed_load_when_retry_is_enabled():
    trace = SimpleNamespace(abort=lambda **_kwargs: None)
    req = SimpleNamespace(
        rid="rid-a",
        last_node=7,
        external_load_retry_count=0,
        external_kv_cleanup_done=False,
        time_stats=SimpleNamespace(
            trace_ctx=trace,
            reset_prefill_retry_time=MagicMock(),
        ),
        reset_for_external_load_retry=MagicMock(),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.__dict__.update(
        _deferred_linker_rids=set(),
        _external_kv_cleanup_rids=set(),
        tree_cache=SimpleNamespace(
            drain_linker_loads=lambda: ["rid-a"],
            has_outstanding_failed_linker_chains=lambda: False,
            is_on_failed_linker_chain=lambda _node_id: False,
        ),
        running_batch=SimpleNamespace(is_empty=lambda: True),
        chunked_req=None,
        enable_external_load_retry=True,
        external_load_retry_max=1,
        _release_aborted_request=MagicMock(),
        clear_pending_chunk_send=MagicMock(),
        grammar_manager=SimpleNamespace(
            process_req_with_grammar=MagicMock(return_value=False)
        ),
        _add_request_to_queue=MagicMock(),
    )

    with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
        Scheduler._mark_failed_linker_loads(
            scheduler, SimpleNamespace(reqs=[req])
        )

    release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    scheduler._release_aborted_request.assert_called_once_with("rid-a")
    req.reset_for_external_load_retry.assert_called_once_with()
    scheduler.clear_pending_chunk_send.assert_called_once_with(req)
    req.time_stats.reset_prefill_retry_time.assert_called_once_with()
    scheduler.grammar_manager.process_req_with_grammar.assert_called_once_with(req)
    scheduler._add_request_to_queue.assert_called_once_with(req)


def test_linker_wrapper_skips_lookup_for_retried_request():
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(page_size=1, pp_size=1)
    wrapper.hit_markers = {"rid-a": object()}
    wrapper.host_prefetch_hits = {}
    wrapper._waiting_queue_prefetch_enabled = False
    result = SimpleNamespace(device_indices=torch.tensor([1, 2]))
    req = SimpleNamespace(rid="rid-a", disable_external_linker=True)

    assert wrapper.match(object(), req, result) is result
    assert "rid-a" not in wrapper.hit_markers


def test_tree_core_reclaims_detached_external_load_leaf():
    component_types = (ComponentType.FULL,)
    root = UnifiedTreeNode(component_types)
    anchor = UnifiedTreeNode(component_types)
    endpoint = UnifiedTreeNode(component_types)
    root.children[1] = anchor
    anchor.parent = root
    anchor.children[2] = endpoint
    endpoint.parent = anchor
    for node, token in ((anchor, 1), (endpoint, 2)):
        node.key = SimpleNamespace(child_key=lambda _page_size, token=token: token)
        node.component_data[ComponentType.FULL].value = torch.tensor([token])

    core = UnifiedTreeCore.__new__(UnifiedTreeCore)
    core._node_arena = {node.id: node for node in (root, anchor, endpoint)}
    core._detached_roots = {}
    core.root_node = root
    core.page_size = 1
    core.components = ()
    core.full_host_duplicates = {}
    core.evictable_device_leaves = set()
    core.evictable_host_leaves = set()
    core.kv_events = SimpleNamespace(record_remove=lambda *_args, **_kwargs: None)
    core._update_evictable_leaf_sets = lambda _node: None

    assert core.detach_external_load_chain(endpoint.id, anchor.id) == [endpoint.id]
    assert core.is_on_detached_chain(endpoint.id)
    assert anchor.children == {}

    result = core.invalidate_external_load_chain(endpoint.id)

    assert result.is_dropped
    assert endpoint.id not in core._node_arena
    assert not core.holds_detached_node(endpoint.id)


def test_linker_fault_injection_is_disabled_by_default():
    with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(0.0):
        check = arm_load_failure_injection(0)
        check("kv", "layer=0")


def test_linker_fault_injection_probability_one_fails():
    with envs.SGLANG_TEST_LINKER_LOAD_FAILURE_PROB.override(1.0):
        check = arm_load_failure_injection(2)
        try:
            check("swa", "layer=3")
        except InjectedLinkerLoadFailure as error:
            assert "tp_rank=2" in str(error)
        else:
            raise AssertionError("failure injection did not fire")
