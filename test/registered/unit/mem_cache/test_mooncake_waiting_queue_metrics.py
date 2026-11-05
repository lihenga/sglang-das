"""Metrics for --mooncake-enable-waiting-queue-dfs-prefetch only."""

import threading
import unittest
from functools import partial
from queue import Queue
from unittest.mock import Mock, patch

from prometheus_client import CollectorRegistry, Counter, Histogram

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestWaitingQueuePrefetchMetrics(CustomTestCase):
    def setUp(self):
        self.registry = CollectorRegistry()
        with patch.object(
            StorageMetricsCollector,
            "_counter_cls",
            partial(Counter, registry=self.registry),
        ), patch.object(
            StorageMetricsCollector,
            "_histogram_cls",
            partial(Histogram, registry=self.registry),
        ):
            collector = StorageMetricsCollector(labels={"tp_rank": "0"})
        self.linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        self.linker.storage_metrics_collector = collector
        self.linker.host_prefetch_enabled = True
        self.linker.host_prefetch_limit = 8
        self.linker.host_prefetch_generation = 0
        self.linker.host_prefetch_lock = threading.Lock()
        self.linker.session_lock = threading.Lock()
        self.linker.host_prefetch_entries = {}
        self.linker.host_prefetch_queue = Queue()
        self.linker.prepared_load_sessions = {}
        self.linker._abort_prepared_load_now = Mock()
        self.transfers = [PoolTransfer(name=PoolName.KV, keys=["page"])]

    def sample(self, suffix, **labels):
        return self.registry.get_sample_value(
            "sglang:waiting_queue_prefetch_" + suffix,
            {"tp_rank": "0", **labels},
        )

    def finish(self, outcome):
        entry = self.linker.host_prefetch_entries["rid"]
        self.linker._finish_host_prefetch(
            "rid", entry, str(entry["session_rid"]), outcome
        )

    def test_ready_claim_and_cleanup_are_distinct_from_completed_io(self):
        with patch("time.monotonic", return_value=10.0):
            self.assertTrue(self.linker.submit_host_prefetch("rid", self.transfers))
        entry = self.linker.host_prefetch_entries["rid"]
        session_rid = str(entry["session_rid"])
        self.linker.prepared_load_sessions[session_rid] = ["key"]
        with patch("time.monotonic", return_value=12.0):
            self.finish("dfs_prefetched")
        self.assertEqual(
            self.sample("duration_seconds_sum", result="dfs_prefetched"), 2
        )
        self.assertIsNone(self.sample("events_total", event="claimed"))
        with patch("time.monotonic", return_value=15.0):
            self.assertTrue(self.linker.claim_ready_host_prefetch("rid"))
            self.assertFalse(self.linker.claim_ready_host_prefetch("rid"))
        self.assertEqual(self.sample("events_total", event="claimed"), 1)
        self.assertEqual(self.sample("ready_lead_seconds_sum"), 3)
        self.linker.abort_prepared_load("rid")
        self.assertIsNone(self.sample("events_total", event="discarded_ready"))
        self.assertEqual(
            self.sample("duration_seconds_count", result="dfs_prefetched"), 1
        )

    def test_cancelled_queued_task_is_recorded_once(self):
        self.linker.submit_host_prefetch("rid", self.transfers)
        self.linker.cancel_host_prefetch("rid")
        self.linker.cancel_host_prefetch("rid")
        self.linker.host_prefetch_queue.put(None)
        self.linker.host_prefetch_thread_func()
        self.assertEqual(self.sample("duration_seconds_count", result="cancelled"), 1)
        self.assertIsNone(
            self.sample("duration_seconds_count", result="dfs_prefetched")
        )

    def test_worker_failure_no_dfs_and_inflight_cancel_are_separate(self):
        for outcome, cancelled, expected in [
            ("failed", False, "failed"),
            ("no_prefetch_needed", False, "no_prefetch_needed"),
            ("dfs_prefetched", True, "cancelled"),
        ]:
            with self.subTest(outcome=outcome, cancelled=cancelled):
                self.linker.host_prefetch_entries.clear()
                self.linker.submit_host_prefetch("rid", self.transfers)
                self.linker.host_prefetch_entries["rid"]["cancelled"] = cancelled
                self.finish(outcome)
                self.assertEqual(
                    self.sample("duration_seconds_count", result=expected), 1
                )
        self.assertIsNone(self.sample("events_total", event="claimed"))

    def test_ready_cancellation_and_disabled_switch(self):
        self.linker.submit_host_prefetch("rid", self.transfers)
        self.finish("dfs_prefetched")
        self.linker.cancel_host_prefetch("rid")
        self.assertEqual(self.sample("events_total", event="discarded_ready"), 1)
        self.linker.host_prefetch_enabled = False
        self.assertFalse(self.linker.submit_host_prefetch("other", self.transfers))
        self.assertEqual(self.sample("events_total", event="submitted"), 1)


if __name__ == "__main__":
    unittest.main()
