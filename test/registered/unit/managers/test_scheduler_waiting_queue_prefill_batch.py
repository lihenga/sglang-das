"""Bounded accumulation of partial Mooncake DFS prefill batches."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWaitingQueuePartialPrefillBatch(CustomTestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_waiting_queue_dfs_prefetch = True
        scheduler._waiting_queue_prefill_batch_size = 3
        scheduler._waiting_queue_partial_batch_idle_rounds = 2
        scheduler._waiting_queue_partial_batch_wait_ms = 0
        scheduler._waiting_queue_partial_batch_idle_count = 0
        scheduler._waiting_queue_partial_batch_deadline_ms = None
        scheduler._waiting_queue_partial_batch_last_queued_ids = None
        scheduler._waiting_queue_partial_batch_last_eligible_ids = None
        scheduler.chunked_req = None
        return scheduler

    def test_new_pending_request_resets_idle_rounds_and_partial_releases(self):
        scheduler = self._scheduler()
        eligible = {"a": "dfs_prefetched", "b": "no_prefetch_needed"}
        pending_arrival = {**eligible, "c": "pending"}
        with patch(
            "sglang.srt.managers.scheduler.get_schedule",
            return_value=SimpleNamespace(prefill_max_requests=None),
        ):
            # First observation and one unchanged round are both held.
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(eligible)
            )
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(eligible)
            )
            self.assertEqual(scheduler._waiting_queue_partial_batch_idle_count, 1)

            # A new request resets the counter even while that request's DFS
            # prefetch is pending and it cannot join the compute batch yet.
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(pending_arrival)
            )
            self.assertEqual(scheduler._waiting_queue_partial_batch_idle_count, 0)

            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(pending_arrival)
            )
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(pending_arrival)
            )
            self.assertEqual(scheduler._waiting_queue_partial_batch_idle_count, 0)

            # An empty eligible queue clears any accumulated partial-batch wait.
            scheduler._waiting_queue_partial_batch_idle_count = 1
            self.assertFalse(scheduler._should_delay_waiting_queue_partial_batch({}))
            self.assertEqual(scheduler._waiting_queue_partial_batch_idle_count, 0)

    def test_wall_clock_deadline_is_not_reset_by_progress(self):
        scheduler = self._scheduler()
        scheduler._waiting_queue_partial_batch_wait_ms = 10
        scheduler._waiting_queue_partial_batch_idle_rounds = 0
        scheduler.tree_cache = SimpleNamespace(
            last_waiting_queue_prefetch_admission_time_ms=100
        )
        with patch(
            "sglang.srt.managers.scheduler.get_schedule",
            return_value=SimpleNamespace(prefill_max_requests=None),
        ):
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"a": "dfs_prefetched"}
                )
            )
            self.assertEqual(
                scheduler._waiting_queue_partial_batch_deadline_ms, 110
            )

            # A queued request and a newly ready request do not restart the
            # original deadline.
            scheduler.tree_cache.last_waiting_queue_prefetch_admission_time_ms = 105
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"a": "dfs_prefetched", "b": "pending"}
                )
            )
            scheduler.tree_cache.last_waiting_queue_prefetch_admission_time_ms = 109
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"a": "dfs_prefetched", "b": "no_prefetch_needed"}
                )
            )
            self.assertEqual(
                scheduler._waiting_queue_partial_batch_deadline_ms, 110
            )

            scheduler.tree_cache.last_waiting_queue_prefetch_admission_time_ms = 110
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"a": "dfs_prefetched", "b": "no_prefetch_needed"}
                )
            )
            self.assertEqual(
                scheduler._waiting_queue_partial_batch_deadline_ms, 110
            )
            # A failed attempt to admit requests must not wait another 10 ms.
            scheduler.tree_cache.last_waiting_queue_prefetch_admission_time_ms = 111
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"a": "dfs_prefetched", "b": "no_prefetch_needed"}
                )
            )

            # An empty or full queue clears the old deadline; the next partial
            # batch starts a fresh interval.
            scheduler.tree_cache.last_waiting_queue_prefetch_admission_time_ms = 200
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"a": "dfs_prefetched"}
                )
            )
            self.assertFalse(scheduler._should_delay_waiting_queue_partial_batch({}))
            self.assertIsNone(scheduler._waiting_queue_partial_batch_deadline_ms)
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"a": "dfs_prefetched"}
                )
            )
            self.assertEqual(
                scheduler._waiting_queue_partial_batch_deadline_ms, 210
            )
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {
                        "a": "dfs_prefetched",
                        "b": "no_prefetch_needed",
                        "c": "dfs_prefetched",
                    }
                )
            )
            self.assertIsNone(scheduler._waiting_queue_partial_batch_deadline_ms)
