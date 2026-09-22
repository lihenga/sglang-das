import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerWaitingQueuePrefetchWave(unittest.TestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._waiting_queue_prefetch_wave_by_rid = {}
        scheduler._waiting_queue_prefetch_wave_next_id = 0
        return scheduler

    def test_readiness_skew_waits_then_admits_wave_together(self):
        scheduler = self._scheduler()
        reqs = [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")]

        scheduler._register_waiting_queue_prefetch_wave(reqs, [True, True])
        blocked = scheduler._waiting_queue_prefetch_blocked_waves(
            reqs, ["dfs_prefetched", "pending"]
        )

        # The first request is ready, but it must not become a singleton
        # admission/H2D batch while its sibling is still in native DFS work.
        self.assertTrue(
            scheduler._waiting_queue_prefetch_request_is_blocked(
                "rid-1", "dfs_prefetched", blocked
            )
        )
        self.assertEqual(
            [
                req.rid
                for req, state in zip(reqs, ["dfs_prefetched", "pending"])
                if not scheduler._waiting_queue_prefetch_request_is_blocked(
                    req.rid, state, blocked
                )
                and state in {"dfs_prefetched", "no_prefetch_needed"}
            ],
            [],
        )

        blocked = scheduler._waiting_queue_prefetch_blocked_waves(
            reqs, ["dfs_prefetched", "dfs_prefetched"]
        )
        admitted = [
            req.rid
            for req, state in zip(reqs, ["dfs_prefetched", "dfs_prefetched"])
            if not scheduler._waiting_queue_prefetch_request_is_blocked(
                req.rid, state, blocked
            )
        ]
        self.assertEqual(admitted, ["rid-1", "rid-2"])

    def test_terminal_and_untracked_tail_do_not_deadlock_ready_sibling(self):
        scheduler = self._scheduler()
        reqs = [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")]
        scheduler._register_waiting_queue_prefetch_wave(reqs, [True, True])

        for tail_state in ("terminal", "cancelled", "not_tracked"):
            blocked = scheduler._waiting_queue_prefetch_blocked_waves(
                reqs, ["dfs_prefetched", tail_state]
            )
            self.assertEqual(blocked, set())

    def test_failed_submission_is_not_added_to_wave(self):
        scheduler = self._scheduler()
        reqs = [SimpleNamespace(rid="rid-1"), SimpleNamespace(rid="rid-2")]
        scheduler._register_waiting_queue_prefetch_wave(reqs, [True, False])

        blocked = scheduler._waiting_queue_prefetch_blocked_waves(
            reqs, ["dfs_prefetched", "pending"]
        )
        self.assertEqual(blocked, set())

    def test_scheduler_defers_h2d_until_wave_is_ready(self):
        class Request:
            def __init__(self, rid):
                self.rid = rid

            def init_next_round_input(self, _tree_cache):
                return None

        class FakePrefillAdder:
            instances = []

            def __init__(self, *args, **kwargs):
                self.can_run_list = []
                self.preempt_list = []
                self.new_chunked_req = None
                FakePrefillAdder.instances.append(self)

            def add_one_req(self, req, **kwargs):
                self.can_run_list.append(req)
                return scheduler_module.AddReqResult.CONTINUE

        class FakeBatch:
            def __init__(self):
                self.prepare_for_extend = Mock()
                self.decoding_reqs = None

        scheduler = self._scheduler()
        reqs = [Request("rid-1"), Request("rid-2")]
        scheduler.waiting_queue = reqs[:]
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = True
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.chunked_req = None
        scheduler.min_free_slots_delayer = None
        scheduler.get_num_allocatable_reqs = Mock(return_value=4)
        scheduler.policy = SimpleNamespace(calc_priority=Mock())
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(extend_attention_block_m=64),
                prefill_aware_swa=False,
            )
        )
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.page_size = 1
        scheduler.max_prefill_tokens = 1024
        scheduler.chunked_prefill_size = None
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 4
        scheduler.max_running_requests = 4
        scheduler.dllm_config = None
        scheduler.enable_lora = False
        scheduler.enable_hicache_storage = False
        scheduler.server_args = SimpleNamespace(
            mooncake_enable_waiting_queue_dfs_prefetch=True,
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="mooncake",
            mooncake_waiting_queue_dfs_prefetch_policy="wait_complete",
        )
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.schedule_policy = "fcfs"
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.truncation_align_size = 1
        scheduler.enable_priority_scheduling = False
        scheduler.spec_algorithm = None
        scheduler.enable_overlap = False
        scheduler.req_to_token_pool = object()
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.model_config = object()
        scheduler.load_inquirer = SimpleNamespace(
            _get_num_pending_tokens=lambda **kwargs: 0
        )
        running_batch = SimpleNamespace(
            reqs=[],
            batch_is_full=False,
            is_empty=lambda: True,
        )

        states = [["dfs_prefetched", "pending"]]
        h2d_batches = []

        def start_h2d():
            h2d_batches.append(
                [req.rid for req in FakePrefillAdder.instances[-1].can_run_list]
            )
            return len(h2d_batches)

        scheduler.tree_cache = SimpleNamespace(
            check_hicache_events=Mock(),
            get_waiting_queue_prefetch_admission_states=lambda rids, **kwargs: states[
                0
            ],
            record_waiting_queue_prefetch_event=Mock(),
            ready_to_load_host_cache=Mock(side_effect=start_h2d),
        )

        fake_memory = SimpleNamespace(enable_flexkv=False)
        fake_schedule = SimpleNamespace(prefill_max_requests=4)
        with (
            patch.object(scheduler_module, "PrefillAdder", FakePrefillAdder),
            patch.object(scheduler_module, "get_memory", return_value=fake_memory),
            patch.object(scheduler_module, "get_schedule", return_value=fake_schedule),
            patch.object(
                scheduler_module.ScheduleBatch,
                "init_new",
                return_value=FakeBatch(),
            ),
            patch.object(scheduler_module, "set_time_batch"),
            patch.object(scheduler_module.PrefillStats, "from_adder", return_value=object()),
        ):
            first_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=running_batch
            )
            self.assertIsNone(first_batch)
            self.assertEqual(h2d_batches, [])

            states[0] = ["dfs_prefetched", "dfs_prefetched"]
            second_batch, _ = scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=running_batch
            )

        self.assertIsNotNone(second_batch)
        self.assertEqual(h2d_batches, [["rid-1", "rid-2"]])
        self.assertEqual(scheduler.waiting_queue, [])


if __name__ == "__main__":
    unittest.main()
