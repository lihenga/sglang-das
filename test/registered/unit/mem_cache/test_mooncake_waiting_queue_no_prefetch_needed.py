"""Promotion retains no queue-time session; admission obtains a fresh lease."""

import threading
import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.components import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Cache:
    """Records the admission reductions.

    ``peer_mins[i]`` is the other ranks' MIN for the i-th reduction (ready,
    valid, claimed); reductions beyond the list see no failing peer.
    """

    def __init__(self, peer_mins=()):
        self.peer_mins = list(peer_mins)
        self.reductions = []
        self.page_size = 1
        self.tree_core = types.SimpleNamespace(
            empty_match_result=types.SimpleNamespace(device_indices="empty")
        )
        # One component that cannot build a transfer stops load_back right
        # after the prefetch decision, before any device-side work. On the
        # prefetched path construction joins the claim reduction as
        # [claimed, constructed]; on the normal fallback it is its own
        # reduction, including when waiting-queue prefetch is disabled.
        component = MagicMock()
        component.build_external_linker_transfer.return_value = None
        self._components_tuple = (component,)

    def _all_reduce_attn_groups(self, tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        # A scalar or a vector verdict; peers are injected element-wise.
        local = tensor.tolist() if tensor.numel() > 1 else int(tensor.item())
        stage = len(self.reductions)
        peer = self.peer_mins[stage] if stage < len(self.peer_mins) else None
        if peer is not None:
            tensor.copy_(torch.minimum(tensor, torch.tensor(peer, dtype=tensor.dtype)))
        self.reductions.append(local)


def _wrapper(cache, cache_linker):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = cache_linker
    wrapper._waiting_queue_prefetch_enabled = True
    wrapper.hit_markers = {}
    wrapper.host_prefetch_hits = {}
    wrapper._update_load = MagicMock()
    hit = ExternalCacheHitMarker(prefix_key=None, tail_hashes=["h"], device_hit_len=0)
    wrapper.hit_markers["rid"] = hit
    wrapper.host_prefetch_hits["rid"] = hit
    return wrapper


class TestCancelWaitingQueuePrefetch(CustomTestCase):
    def _wrapper(self):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache_linker = MagicMock()
        wrapper._waiting_queue_prefetch_enabled = True
        wrapper.hit_markers = {"rid": object()}
        wrapper.host_prefetch_hits = {}
        wrapper.pending_host_prefetch_submissions = {}
        return wrapper

    def test_untracked_request_does_not_cancel_backend_prefetch(self):
        wrapper = self._wrapper()

        wrapper.cancel_waiting_queue_prefetch("rid")

        self.assertNotIn("rid", wrapper.hit_markers)
        wrapper.cache_linker.cancel_host_prefetch.assert_not_called()

    def test_tracked_host_prefetch_is_cancelled(self):
        wrapper = self._wrapper()
        wrapper.host_prefetch_hits["rid"] = object()

        wrapper.cancel_waiting_queue_prefetch("rid")

        self.assertNotIn("rid", wrapper.host_prefetch_hits)
        wrapper.cache_linker.cancel_host_prefetch.assert_called_once_with("rid")

    def test_pending_submission_is_cancelled(self):
        wrapper = self._wrapper()
        job = types.SimpleNamespace(cancelled=threading.Event())
        wrapper.pending_host_prefetch_submissions["rid"] = (job, object())

        wrapper.cancel_waiting_queue_prefetch("rid")

        self.assertTrue(job.cancelled.is_set())
        self.assertNotIn("rid", wrapper.pending_host_prefetch_submissions)
        wrapper.cache_linker.cancel_host_prefetch.assert_called_once_with("rid")


class _NoPrefetchMapAccess(dict):
    def get(self, key, default=None):
        raise AssertionError("disabled prefetch map must not be queried")

    def pop(self, key, default=None):
        raise AssertionError("disabled prefetch map must not be popped")


class TestDisabledPrefetchMapFastPath(CustomTestCase):
    def _wrapper(self, cache):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = MagicMock()
        wrapper._waiting_queue_prefetch_enabled = False
        wrapper.hit_markers = {}
        wrapper.host_prefetch_hits = _NoPrefetchMapAccess()
        wrapper.pending_host_prefetch_submissions = _NoPrefetchMapAccess()
        return wrapper

    def test_match_skips_host_prefetch_map_query(self):
        cache = types.SimpleNamespace(page_size=1, pp_size=1)
        wrapper = self._wrapper(cache)
        req = types.SimpleNamespace(rid="rid")
        result = types.SimpleNamespace(
            device_indices=types.SimpleNamespace(numel=lambda: 0)
        )

        self.assertIs(wrapper.match([], req, result), result)

    def test_normal_load_skips_host_prefetch_map_pop(self):
        cache = _Cache()
        first_component = MagicMock()
        transfer = object()
        first_component.build_external_linker_transfer.return_value = transfer
        cache._components_tuple = (first_component, *cache._components_tuple)
        wrapper = self._wrapper(cache)
        wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h0"], device_hit_len=0
        )
        wrapper._update_load = MagicMock()
        req = types.SimpleNamespace(rid="rid", last_node="node")

        indices, node = wrapper.load_back(req)

        self.assertEqual(indices, "empty")
        self.assertEqual(node, "node")
        first_component.build_external_linker_transfer.assert_called_once_with(
            LinkerTransferPhase.LOAD, None, ["h0"]
        )
        wrapper._update_load.assert_called_once_with(
            ExternalLinkerLoadPhase.ABORT, req, [(first_component, transfer)], 1
        )
        self.assertEqual(cache.reductions, [0])

    def test_normal_load_success_agrees_on_construction(self):
        cache = _Cache()
        cache.pp_size = 2
        transfer = types.SimpleNamespace(name=PoolName.KV, device_indices="loaded")
        component = cache._components_tuple[0]
        component.build_external_linker_transfer.return_value = transfer
        wrapper = self._wrapper(cache)
        wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h0"], device_hit_len=0
        )
        wrapper.cache_linker.revalidate_load.return_value = True
        wrapper._update_load = MagicMock(return_value=[transfer])
        wrapper._queue_load = MagicMock()
        req = types.SimpleNamespace(rid="rid", last_node="node")

        self.assertEqual(wrapper.load_back(req), ("loaded", "node"))

        self.assertEqual(cache.reductions, [1])
        wrapper.cache_linker.prepare_load.assert_not_called()
        wrapper.cache_linker.revalidate_load.assert_called_once_with([transfer])
        wrapper._update_load.assert_called_once_with(
            ExternalLinkerLoadPhase.PREPARE, req, [(component, transfer)], 1
        )
        wrapper._queue_load.assert_called_once_with(
            "rid", "node", [transfer], anchor="node"
        )

    def test_normal_load_build_error_propagates_after_consensus_and_abort(self):
        cache = _Cache()
        first_component = MagicMock()
        transfer = object()
        first_component.build_external_linker_transfer.return_value = transfer
        error = RuntimeError("construction failed")
        cache._components_tuple[0].build_external_linker_transfer.side_effect = error
        cache._components_tuple = (first_component, *cache._components_tuple)
        wrapper = self._wrapper(cache)
        wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h0"], device_hit_len=0
        )
        wrapper._update_load = MagicMock()
        req = types.SimpleNamespace(rid="rid", last_node="node")

        with self.assertRaises(RuntimeError) as raised:
            wrapper.load_back(req)

        self.assertIs(raised.exception, error)
        self.assertEqual(cache.reductions, [0])
        wrapper._update_load.assert_called_once_with(
            ExternalLinkerLoadPhase.ABORT, req, [(first_component, transfer)], 1
        )
        wrapper.cache_linker.revalidate_load.assert_not_called()

    def test_release_skips_prefetch_cancel_helper_but_keeps_normal_cleanup(self):
        wrapper = self._wrapper(None)
        wrapper.hit_markers["rid"] = object()
        wrapper.pending_loads = {}
        wrapper.cache_linker.cancel_queued_load.return_value = False
        wrapper.cancel_waiting_queue_prefetch = MagicMock(
            side_effect=AssertionError("disabled prefetch helper called")
        )

        wrapper.release_request("rid")

        self.assertNotIn("rid", wrapper.hit_markers)
        wrapper.cancel_waiting_queue_prefetch.assert_not_called()
        wrapper.cache_linker.cancel_queued_load.assert_called_once_with("rid")

    def test_disabled_wrapper_allocates_no_prefetch_maps_and_resets(self):
        cache = types.SimpleNamespace(
            tree_core=types.SimpleNamespace(),
            write_through_threshold=0,
            _all_reduce_attn_groups=MagicMock(),
            components=(),
        )
        cache_linker = MagicMock()
        cache_linker.waiting_queue_prefetch_enabled.return_value = False
        wrapper = UnifiedCacheLinkerWrapper(cache, cache_linker)

        self.assertIsNone(wrapper.host_prefetch_hits)
        self.assertIsNone(wrapper.pending_host_prefetch_submissions)
        self.assertEqual(
            wrapper.get_host_prefetch_admission_state("rid"), "not_tracked"
        )
        self.assertFalse(wrapper.prefetch_to_host(object()))
        cache_linker.get_host_prefetch_status.assert_not_called()
        cache._all_reduce_attn_groups.assert_not_called()

        wrapper.reset()

        cache_linker.reset.assert_called_once()


class TestPromotionAdmission(CustomTestCase):
    def test_promotion_and_normal_admission_share_one_session_path(self):
        for status in ("dfs_prefetched", "failed", None):
            with self.subTest(status=status):
                cache = _Cache()
                cache.pp_size = 2
                transfer = types.SimpleNamespace(
                    name=PoolName.KV, device_indices="loaded"
                )
                cache._components_tuple[
                    0
                ].build_external_linker_transfer.return_value = transfer
                linker = MagicMock()
                linker.get_host_prefetch_status.return_value = status
                linker.prepare_load.return_value = True
                wrapper = _wrapper(cache, linker)
                wrapper._update_load.return_value = [transfer]
                wrapper._queue_load = MagicMock()
                req = types.SimpleNamespace(rid="rid", last_node="node")
                self.assertEqual(wrapper.load_back(req), ("loaded", "node"))
                self.assertEqual(cache.reductions, [1])
                linker.prepare_load.assert_called_once_with("rid", [transfer])
                linker.revalidate_host_prefetch.assert_not_called()
                linker.revalidate_no_prefetch_needed.assert_not_called()
                linker.claim_ready_host_prefetch.assert_not_called()
                linker.revalidate_load.assert_not_called()

    def test_eviction_at_admission_becomes_a_collective_cache_miss(self):
        cache = _Cache()
        transfer = types.SimpleNamespace(name=PoolName.KV)
        cache._components_tuple[
            0
        ].build_external_linker_transfer.return_value = transfer
        linker = MagicMock()
        linker.prepare_load.return_value = False
        wrapper = _wrapper(cache, linker)
        req = types.SimpleNamespace(rid="rid", last_node="node")
        self.assertEqual(wrapper.load_back(req), ("empty", "node"))
        self.assertEqual(cache.reductions, [0])
        linker.cancel_host_prefetch.assert_called_once_with("rid")
        linker.revalidate_load.assert_not_called()

    def test_peer_failure_releases_local_fresh_session(self):
        cache = _Cache(peer_mins=[0])
        transfer = types.SimpleNamespace(name=PoolName.KV)
        cache._components_tuple[
            0
        ].build_external_linker_transfer.return_value = transfer
        linker = MagicMock()
        linker.prepare_load.return_value = True
        wrapper = _wrapper(cache, linker)
        req = types.SimpleNamespace(rid="rid", last_node="node")
        self.assertEqual(wrapper.load_back(req), ("empty", "node"))
        self.assertEqual(cache.reductions, [1])
        linker.abort_prepared_load.assert_called_once_with("rid")


class TestPromotionWorker(CustomTestCase):
    def make_linker(self, result):
        from queue import Queue

        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_enabled = True
        linker.host_prefetch_lock = threading.Lock()
        transfers = [object()]
        linker.host_prefetch_entries = {
            "rid": {"state": "queued", "transfers": transfers}
        }
        linker.host_prefetch_queue = Queue()
        linker.host_prefetch_queue.put(("rid", transfers))
        linker.host_prefetch_queue.put(None)
        linker.pool_group = MagicMock()
        linker._expanded_load_keys = MagicMock(return_value=["key"])
        linker.storage = types.SimpleNamespace(store=MagicMock())
        linker.storage.store.batch_prefetch_to_memory.return_value = result
        linker.prepare_load = MagicMock()
        linker._abort_prepared_load_now = MagicMock()
        return linker

    def test_success_retains_only_metadata(self):
        linker = self.make_linker([0])
        linker.host_prefetch_thread_func()
        self.assertEqual(linker.host_prefetch_entries["rid"]["state"], "dfs_prefetched")
        linker.storage.store.batch_prefetch_to_memory.assert_called_once_with(["key"])
        linker.prepare_load.assert_not_called()
        linker.storage.store.batch_get_session_prefetch.assert_not_called()
        linker._abort_prepared_load_now.assert_not_called()
        self.assertNotIn("session_rid", linker.host_prefetch_entries["rid"])

    def test_partial_or_failed_result_falls_back(self):
        for result in ([], [-1], [0, 0]):
            linker = self.make_linker(result)
            linker.host_prefetch_thread_func()
            self.assertEqual(linker.host_prefetch_entries["rid"]["state"], "failed")

    def test_cancel_during_promotion_discards_only_request_metadata(self):
        linker = self.make_linker([0])

        def promote(keys):
            linker.cancel_host_prefetch("rid")
            return [0]

        linker.storage.store.batch_prefetch_to_memory.side_effect = promote
        linker.host_prefetch_thread_func()
        self.assertNotIn("rid", linker.host_prefetch_entries)
        linker._abort_prepared_load_now.assert_not_called()


if __name__ == "__main__":
    unittest.main()
