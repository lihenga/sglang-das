import threading
import types
import unittest
from queue import Queue
from unittest.mock import Mock

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    ReadPlanLoadCounter,
)
from sglang.srt.mem_cache.unified_cache.components import ExternalLinkerLoadPhase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeDirectLinkerReadPlan(CustomTestCase):
    def test_read_plan_failure_is_deferred_until_after_forward(self):
        counter = ReadPlanLoadCounter(num_layers=2)
        index = counter.update_producer()
        counter.set_consumer(index)
        plan = Mock()
        plan.wait.side_effect = RuntimeError("range get failed: rc=-707")
        counter.bind(index, plan)

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="ERROR",
        ) as logs:
            counter.wait_until(0)
            counter.wait_until(1)

        self.assertEqual(len(logs.output), 1)
        self.assertNotIn(index, counter.plans)
        self.assertNotIn(index, counter.reported)

    def test_successful_read_plan_load_reports_success(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.read_plan_enabled = True
        linker.tp_rank = 0
        linker.load_with_read_plan = Mock()

        success = linker.load_layer_wise(7, [])

        self.assertIs(success, True)
        linker.load_with_read_plan.assert_called_once_with(7, [])

    def test_layout_expands_packed_layer_mapping(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.num_layers = 3
        linker.storage = types.SimpleNamespace(
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=lambda keys: [f"tagged:{key}" for key in keys],
        )

        pool = types.SimpleNamespace(
            layer_mapping={0: (0, 2), 1: 1},
            buffer_meta=[
                [(100, 10, 1), (110, 11, 2), (120, 12, 3)],
                [(200, 20, 4), (210, 21, 5), (220, 22, 6)],
            ],
            _component_offsets=[[7, 8, 9], [17, 18, 19]],
            packed=True,
            prepare_locations=lambda indices: [int(value) for value in indices],
        )
        linker.pools = {PoolName.KV: pool}

        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([4, 5]),
            keys=["page-0"],
        )
        layouts = linker._prepare_read_plan_layouts([("rid", [transfer])])

        self.assertEqual(len(layouts), 1)
        keys, locations, packed, layers = layouts[0]
        self.assertEqual(keys, ["tagged:page-0"])
        self.assertEqual(locations, [4, 5])
        self.assertTrue(packed)
        self.assertEqual(
            layers,
            [
                [
                    (100, 10, 1, 7),
                    (120, 12, 3, 9),
                    (200, 20, 4, 17),
                    (220, 22, 6, 19),
                ],
                [(110, 11, 2, 8), (210, 21, 5, 18)],
                [],
            ],
        )

    def test_host_prefetch_submission_does_not_prepare_inline(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_enabled = True
        linker.host_prefetch_limit = 8
        linker.host_prefetch_max_pages = 1024
        linker.host_prefetch_lock = threading.Lock()
        linker.host_prefetch_entries = {}
        linker.host_prefetch_queue = Queue()
        linker.prepare_load = Mock(
            side_effect=AssertionError(
                "session preparation must run on the prefetch worker"
            )
        )
        transfer = PoolTransfer(name=PoolName.KV, keys=["page-a"])

        self.assertTrue(linker.submit_host_prefetch("rid", [transfer]))
        linker.prepare_load.assert_not_called()
        self.assertEqual(linker.get_host_prefetch_status("rid"), "queued")
        queued_rid, queued_transfers = linker.host_prefetch_queue.get_nowait()
        self.assertEqual(queued_rid, "rid")
        self.assertEqual(queued_transfers, [transfer])
        linker.host_prefetch_queue.task_done()

    def test_host_prefetch_worker_batches_native_reads(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_lock = threading.Lock()
        linker.session_lock = threading.Lock()
        linker.host_prefetch_queue = Queue()
        linker.host_prefetch_batch_limit = 3
        linker.host_prefetch_limit = 3
        linker.host_prefetch_max_bytes = 1 << 20
        linker.host_prefetch_entries = {}
        linker.prepared_load_sessions = {}
        linker.session_sources = {}
        linker.session_refcounts = {}
        store = Mock()
        store.batch_get_session_prefetch.return_value = [0, 0]
        linker.storage = types.SimpleNamespace(store=store)
        linker._update_host_prefetch_reservation_metrics = Mock()
        linker._observe_host_prefetch_latency = Mock()

        transfers_by_rid = {
            "rid-1": [PoolTransfer(name=PoolName.KV, keys=["key-1"])],
            "rid-2": [PoolTransfer(name=PoolName.KV, keys=["key-2"])],
            "rid-3": [PoolTransfer(name=PoolName.KV, keys=["key-1"])],
        }
        for index, (rid, transfers) in enumerate(transfers_by_rid.items(), 1):
            key = transfers[0].keys[0]
            linker.host_prefetch_entries[rid] = {
                "state": "queued",
                "cancelled": False,
                "transfers": transfers,
                "session_rid": f"session-{index}",
                "object_sizes": {key: 4096},
                "reserved_bytes": 4096,
                "queued_at": 0.0,
            }

        def prepare(session_rid, transfers):
            key = transfers[0].keys[0]
            linker.prepared_load_sessions[session_rid] = [key]
            linker.session_sources[key] = "dfs"
            return True

        linker.prepare_load = Mock(side_effect=prepare)
        for rid, transfers in transfers_by_rid.items():
            linker.host_prefetch_queue.put((rid, transfers))
        linker.host_prefetch_queue.put(None)

        worker = threading.Thread(target=linker.host_prefetch_thread_func)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        linker.host_prefetch_queue.join()

        store.batch_get_session_prefetch.assert_called_once_with(["key-1", "key-2"])
        self.assertEqual(
            [
                linker.host_prefetch_entries[rid]["state"]
                for rid in ("rid-1", "rid-2", "rid-3")
            ],
            ["dfs_prefetched", "dfs_prefetched", "dfs_prefetched"],
        )

    def test_queued_host_prefetch_cancel_is_atomic(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_lock = threading.Lock()
        linker.host_prefetch_entries = {
            "queued": {"state": "queued", "reserved_bytes": 1},
            "reading": {"state": "reading", "reserved_bytes": 1},
        }
        linker._update_host_prefetch_reservation_metrics = Mock()

        self.assertTrue(linker.try_cancel_queued_host_prefetch("queued"))
        self.assertNotIn("queued", linker.host_prefetch_entries)
        self.assertFalse(linker.try_cancel_queued_host_prefetch("reading"))
        self.assertEqual(linker.host_prefetch_entries["reading"]["state"], "reading")
        linker._update_host_prefetch_reservation_metrics.assert_called_once_with()

    def test_host_prefetch_admission_distinguishes_queued_from_active(self):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache_linker = Mock()
        wrapper.host_prefetch_hits = {"rid": Mock()}

        for backend_state, admission_state in (
            ("queued", "queued"),
            ("preparing", "pending"),
            ("reading", "pending"),
        ):
            wrapper.cache_linker.get_host_prefetch_status.return_value = backend_state
            self.assertEqual(
                wrapper.get_host_prefetch_admission_state("rid"), admission_state
            )

    def test_best_effort_admission_atomically_cancels_queued_prefetch(self):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache_linker = Mock()
        wrapper.cache_linker.try_cancel_queued_host_prefetch.return_value = True
        wrapper.hit_markers = {"rid": Mock()}
        wrapper.host_prefetch_hits = {"rid": Mock()}

        self.assertEqual(
            wrapper.get_host_prefetch_admission_state("rid", cancel_queued=True),
            "cancelled",
        )
        self.assertNotIn("rid", wrapper.hit_markers)
        self.assertNotIn("rid", wrapper.host_prefetch_hits)
        wrapper.cache_linker.get_host_prefetch_status.assert_not_called()

    def test_rank_wide_queued_prefetch_remains_cancellable(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.linker = Mock()
        cache.linker.get_host_prefetch_admission_state.return_value = "queued"
        cache._all_reduce_attn_groups = Mock()

        self.assertEqual(
            cache.get_waiting_queue_prefetch_admission_states(["rid"]), ["queued"]
        )

    def test_rank_wide_queued_prefetch_cancellation_succeeds(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.linker = Mock()
        cache.linker.get_host_prefetch_admission_state.return_value = "cancelled"
        cache._all_reduce_attn_groups = Mock()

        self.assertEqual(
            cache.get_waiting_queue_prefetch_admission_states(
                ["rid"], cancel_queued=True
            ),
            ["cancelled"],
        )

    def test_rank_wide_partial_prefetch_completion_waits(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.linker = Mock()
        cache.linker.get_host_prefetch_admission_state.return_value = "queued"

        def add_completed_rank(counts, _op):
            counts[0, 4] += 1

        cache._all_reduce_attn_groups = add_completed_rank

        self.assertEqual(
            cache.get_waiting_queue_prefetch_admission_states(["rid"]), ["pending"]
        )

    def test_rank_wide_cancel_detects_worker_start_race(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.linker = Mock()
        cache.linker.get_host_prefetch_admission_state.return_value = "cancelled"

        def add_active_rank(counts, _op):
            counts[0, 2] += 1

        cache._all_reduce_attn_groups = add_active_rank

        self.assertEqual(
            cache.get_waiting_queue_prefetch_admission_states(
                ["rid"], cancel_queued=True
            ),
            ["pending"],
        )
        cache.linker.get_host_prefetch_admission_state.assert_called_once_with(
            "rid", cancel_queued=True
        )

    def test_no_prefetch_needed_revalidates_before_normal_load(self):
        """A no-op host prefetch owns no session and must use normal guards."""
        empty_indices = torch.empty((0,), dtype=torch.int64)
        transfer = PoolTransfer(name=PoolName.KV, keys=["page-a"])
        component = Mock()
        component.build_external_linker_transfer.return_value = transfer
        cache = types.SimpleNamespace(
            tree_core=types.SimpleNamespace(
                empty_match_result=types.SimpleNamespace(device_indices=empty_indices)
            ),
            _components_tuple=(component,),
            page_size=1,
            _all_reduce_attn_groups=Mock(),
        )
        backend = Mock()
        backend.get_host_prefetch_status.return_value = "no_prefetch_needed"
        backend.revalidate_load.return_value = False
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = backend
        wrapper.pending_loads = {}
        wrapper.failed_chains = {}
        wrapper.taken_loads = []
        wrapper.pending_offloads = []
        wrapper.hit_markers = {"rid": ExternalCacheHitMarker(Mock(), ["page-a"], 0)}
        wrapper.host_prefetch_hits = dict(wrapper.hit_markers)
        wrapper._update_load = Mock()
        last_node = object()
        req = types.SimpleNamespace(rid="rid", last_node=last_node)

        loaded, returned_node = wrapper.load_back(req)

        self.assertIs(loaded, empty_indices)
        self.assertIs(returned_node, last_node)
        backend.cancel_host_prefetch.assert_called_once_with("rid")
        backend.revalidate_load.assert_called_once_with([transfer])
        self.assertEqual(
            wrapper._update_load.call_args.args[0],
            ExternalLinkerLoadPhase.ABORT,
        )

    def test_reused_session_is_refreshed_before_refcount_increment(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        store = Mock()
        store.batch_get_session_refresh.return_value = [0]
        linker.storage = types.SimpleNamespace(
            store=store,
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=lambda keys: [f"tagged:{key}" for key in keys],
        )
        linker.host_prefetch_enabled = True
        linker.session_lock = threading.Lock()
        linker.prepared_load_sessions = {"old-rid": ["tagged:page-a"]}
        linker.session_refcounts = {"tagged:page-a": 1}
        linker.session_sources = {"tagged:page-a": "dfs"}
        transfer = PoolTransfer(name=PoolName.KV, keys=["page-a"])

        self.assertTrue(linker._prepare_expanded_load_batch([("new-rid", [transfer])]))

        store.batch_get_session_refresh.assert_called_once_with(["tagged:page-a"])
        store.batch_get_session_start_with_sources.assert_not_called()
        self.assertEqual(linker.session_refcounts["tagged:page-a"], 2)
        self.assertEqual(linker.prepared_load_sessions["new-rid"], ["tagged:page-a"])

    def test_failed_reused_session_refresh_does_not_acquire_reference(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        store = Mock()
        store.batch_get_session_refresh.return_value = [-1]
        linker.storage = types.SimpleNamespace(
            store=store,
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=lambda keys: [f"tagged:{key}" for key in keys],
        )
        linker.host_prefetch_enabled = True
        linker.session_lock = threading.Lock()
        linker.prepared_load_sessions = {"old-rid": ["tagged:page-a"]}
        linker.session_refcounts = {"tagged:page-a": 1}
        linker.session_sources = {"tagged:page-a": "dfs"}
        transfer = PoolTransfer(name=PoolName.KV, keys=["page-a"])

        self.assertFalse(linker._prepare_expanded_load_batch([("new-rid", [transfer])]))

        self.assertEqual(linker.session_refcounts["tagged:page-a"], 1)
        self.assertNotIn("new-rid", linker.prepared_load_sessions)


if __name__ == "__main__":
    unittest.main()
