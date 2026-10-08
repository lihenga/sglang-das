"""Unit tests for ReqTimeStats IPC serialization.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.
"""

import pickle
import unittest
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestScheduleTimeBatch(CustomTestCase):
    def test_merges_capacity_attrs_into_single_schedule_event(self):
        time_stats = mock.Mock()
        batch = mock.Mock()
        batch.reqs = [mock.Mock(time_stats=time_stats)]
        batch.forward_mode.is_decode.return_value = True
        batch.forward_mode.is_prefill.return_value = False
        batch.forward_mode.is_prebuilt.return_value = False

        with mock.patch.object(rts, "get_global_tracing_enabled", return_value=True):
            rts.set_schedule_time_batch(
                batch,
                attrs={
                    "l1_kv_available_slots_before": 100,
                    "l1_kv_available_slots_after": 90,
                },
            )

        time_stats.set_last_scheduled_time.assert_called_once()
        forward_mode, _, attrs = time_stats.set_last_scheduled_time.call_args.args
        self.assertIs(forward_mode, batch.forward_mode)
        self.assertEqual(attrs["batch_size"], 1)
        self.assertEqual(attrs["forward_mode"], "decode")
        self.assertEqual(attrs["l1_kv_available_slots_before"], 100)
        self.assertEqual(attrs["l1_kv_available_slots_after"], 90)


class TestSetstatePreservesUnsetTimeSentinels(CustomTestCase):
    def test_two_hop_round_trip(self):
        src = rts.SchedulerReqTimeStats()
        src.enable_metrics = True
        src.wait_queue_entry_time = 123.456
        src.prefill_finished_time = 0.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            hop1 = pickle.loads(blob)
            blob2 = pickle.dumps(hop1)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            hop2 = pickle.loads(blob2)

        self.assertEqual(hop2.prefill_finished_time, 0.0)
        self.assertAlmostEqual(hop2.wait_queue_entry_time, 123.456 - 9.0)


if __name__ == "__main__":
    unittest.main()
