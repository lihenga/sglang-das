"""Fixed 27/27/10 accumulation for Mooncake DFS prefill batches."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWaitingQueueFixedPrefillBatch(CustomTestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_waiting_queue_dfs_prefetch = True
        scheduler._waiting_queue_dfs_prefetch_admitted_requests = 0
        scheduler.chunked_req = None
        return scheduler

    def test_stages_wait_for_27_27_10_and_disable_after_64(self):
        scheduler = self._scheduler()
        with patch(
            "sglang.srt.managers.scheduler.get_schedule",
            return_value=SimpleNamespace(prefill_max_requests=None),
        ):
            self.assertEqual(scheduler._waiting_queue_dfs_prefetch_batch_target(), 27)
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {f"r{i}": "dfs_prefetched" for i in range(26)}
                )
            )
            # 27 ready requests release the first batch without waiting for
            # all 64 requests to arrive.
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {f"r{i}": "dfs_prefetched" for i in range(27)}
                )
            )

            scheduler._waiting_queue_dfs_prefetch_admitted_requests = 27
            self.assertEqual(scheduler._waiting_queue_dfs_prefetch_batch_target(), 27)
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {f"r{i}": "no_prefetch_needed" for i in range(26)}
                )
            )

            scheduler._waiting_queue_dfs_prefetch_admitted_requests = 54
            self.assertEqual(scheduler._waiting_queue_dfs_prefetch_batch_target(), 10)
            self.assertTrue(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {f"r{i}": "dfs_prefetched" for i in range(9)}
                )
            )
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {f"r{i}": "dfs_prefetched" for i in range(10)}
                )
            )

            scheduler._waiting_queue_dfs_prefetch_admitted_requests = 64
            self.assertIsNone(scheduler._waiting_queue_dfs_prefetch_batch_target())
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {"r64": "dfs_prefetched"}
                )
            )

    def test_rank_wide_admission_target_respects_request_cap(self):
        scheduler = self._scheduler()
        with patch(
            "sglang.srt.managers.scheduler.get_schedule",
            return_value=SimpleNamespace(prefill_max_requests=8),
        ):
            self.assertFalse(
                scheduler._should_delay_waiting_queue_partial_batch(
                    {f"r{i}": "dfs_prefetched" for i in range(8)}
                )
            )
