"""Keep speculative native I/O collectives separate from scheduler/model groups."""

import datetime
import queue
import socket
import threading
import time
import unittest
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.mooncake_prefetch import (
    MooncakePrefetchSchedulerMixin,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import PreparedHostPrefetch
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


class _Backend:
    def __init__(self):
        self.submitted = []
        self.cancelled = []

    def submit_host_prefetch(self, rid, transfers):
        self.submitted.append(rid)
        return True

    def cancel_host_prefetch(self, rid):
        self.cancelled.append(rid)


def _round_worker(rank, port):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        main_group = dist.new_group([0, 1], backend="gloo")
        cp = dist.new_group([0, 1], backend="gloo")
        tp = dist.new_group([0, 1], backend="gloo")
        owner = MooncakePrefetchSchedulerMixin()
        owner._bg_completed_epoch = 0
        owner._bg_prefetch_jobs = queue.Queue()
        owner._bg_prefetch_acks = queue.Queue()
        owner._bg_pending_prefetch_jobs = {}
        owner._bg_prefetch_join_age = {}
        owner._bg_attn_cp_cpu_group = cp
        owner._bg_attn_tp_cpu_group = tp
        backend = _Backend()
        owner._bg_prefetch_jobs.put(
            PreparedHostPrefetch("shared", True, (), backend, threading.Event())
        )
        if rank == 0:
            owner._bg_prefetch_jobs.put(
                PreparedHostPrefetch(
                    "one-rank-only", True, (), backend, threading.Event()
                )
            )
        start = threading.Barrier(2)
        errors = []

        def background():
            try:
                for _ in range(5):
                    start.wait(timeout=20)
                    owner._run_bg_prefetch_round()
                    owner._bg_completed_epoch += 1
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=background, daemon=True)
        worker.start()
        for _ in range(5):
            start.wait(timeout=20)
            value = torch.tensor([rank + 1], dtype=torch.int)
            dist.all_reduce(value, group=main_group)
            assert value.item() == 3
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert not errors, errors
        assert backend.submitted == ["shared"]
        outcomes = {}
        while not owner._bg_prefetch_acks.empty():
            job, submitted = owner._bg_prefetch_acks.get_nowait()
            outcomes[job.rid] = submitted
        assert outcomes["shared"] is True
        if rank == 0:
            assert outcomes["one-rank-only"] is False
        dist.barrier(group=main_group)
    finally:
        dist.destroy_process_group()


class TestMooncakePrefetchScheduler(CustomTestCase):
    def test_idle_iteration_does_not_request_a_background_epoch(self):
        owner = MooncakePrefetchSchedulerMixin()
        owner.enable_waiting_queue_dfs_prefetch = True
        owner._drain_bg_prefetch_acks = Mock()
        owner._request_bg_prefetch_epoch = Mock()
        owner._begin_scheduler_iteration()
        owner._request_bg_prefetch_epoch.assert_not_called()

    def test_noncoordinated_empty_cleanup_does_not_enter_a_collective(self):
        owner = MooncakePrefetchSchedulerMixin()
        owner._coordinated_kv_cleanup = False
        owner._mooncake_cleanup_requests = {}
        owner._all_reduce_mooncake_cleanup = Mock()
        owner._process_mooncake_cleanup_requests()
        owner._all_reduce_mooncake_cleanup.assert_not_called()

    def test_forward_completion_wakes_an_empty_ingress_poll(self):
        owner = MooncakePrefetchSchedulerMixin()
        owner._forward_ingress_enabled = True
        future = Future()
        owner._forward_launch_executor = SimpleNamespace(submit=lambda *args: future)
        owner._bg_error = None
        owner._forward_deferred_reqs = []
        owner.forward_ct = 1
        owner.device_module = SimpleNamespace(
            StreamContext=lambda stream: nullcontext()
        )
        owner.schedule_stream, owner.forward_stream = Mock(), Mock()
        owner._trace_prefill_reqs = Mock()

        def receive(**kwargs):
            future.set_result("finished")
            return []

        owner.request_receiver = SimpleNamespace(
            ingress_sync_groups=lambda: [], recv_requests=receive, max_recv_per_poll=-1
        )
        with patch(
            "sglang.srt.managers.scheduler_components.mooncake_prefetch.time.sleep",
            side_effect=AssertionError("fixed sleep after forward completion"),
        ):
            self.assertEqual(
                owner._forward_with_waiting_queue_ingress(
                    SimpleNamespace(
                        forward_mode=SimpleNamespace(is_extend=lambda: True)
                    )
                ),
                "finished",
            )

    def test_cancelled_old_lifecycle_does_not_swallow_the_new_job(self):
        owner = MooncakePrefetchSchedulerMixin()
        owner._bg_completed_epoch = 0
        owner._bg_prefetch_jobs, owner._bg_prefetch_acks = queue.Queue(), queue.Queue()
        owner._bg_pending_prefetch_jobs, owner._bg_prefetch_join_age = {}, {}
        owner._bg_attn_cp_cpu_group = owner._bg_attn_tp_cpu_group = None
        backend = _Backend()
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        old, new = [
            PreparedHostPrefetch(
                linker.request_id(CacheRequestHandle("rid", 0, lifecycle)),
                True,
                (),
                backend,
                threading.Event(),
            )
            for lifecycle in (1, 2)
        ]
        old.cancelled.set()
        owner._bg_prefetch_jobs.put(old)
        owner._bg_prefetch_jobs.put(new)
        with patch("torch.distributed.get_rank", return_value=0):
            owner._run_bg_prefetch_round()
        self.assertNotEqual(old.rid, new.rid)
        self.assertEqual(backend.submitted, [new.rid])

    def test_forward_deferred_watermark_pauses_all_receive_peers_and_preserves_order(
        self,
    ):
        for peer_full in (False, True):
            with self.subTest(peer_full=peer_full):
                owner = MooncakePrefetchSchedulerMixin()
                owner._forward_ingress_enabled = True
                owner.FORWARD_DEFERRED_HIGH_WATER = 2
                owner.FORWARD_RECV_LIMIT = 1
                future = Future()
                owner._forward_launch_executor = SimpleNamespace(
                    submit=lambda *args: future
                )
                owner._bg_error = None
                owner._forward_deferred_reqs = []
                owner.forward_ct = 1
                owner.device_module = SimpleNamespace(
                    StreamContext=lambda stream: nullcontext()
                )
                owner.schedule_stream, owner.forward_stream = Mock(), Mock()
                owner._trace_prefill_reqs = Mock()
                received, limits = [], []
                paused = []

                def receive(**kwargs):
                    item = object()
                    received.append(item)
                    limits.append(owner.request_receiver.max_recv_per_poll)
                    if len(received) > 2:
                        future.set_result("overflow")
                    return [item]

                def reduce(active, **kwargs):
                    # A peer reaching its bound must pause this rank too.
                    if active.numel() == 2 and (peer_full or active[1].item()):
                        active[1] = 1
                        paused.append(True)
                        if not future.done():
                            future.set_result("finished")

                owner.request_receiver = SimpleNamespace(
                    ingress_sync_groups=lambda: [object()],
                    recv_requests=receive,
                    max_recv_per_poll=-1,
                )
                with patch("torch.distributed.all_reduce", side_effect=reduce):
                    result = owner._forward_with_waiting_queue_ingress(
                        SimpleNamespace(
                            forward_mode=SimpleNamespace(is_extend=lambda: True)
                        )
                    )
                self.assertEqual(result, "finished")
                self.assertTrue(paused)
                self.assertEqual(len(received), 0 if peer_full else 2)
                self.assertEqual(owner._forward_deferred_reqs, received)
                self.assertEqual(limits, [] if peer_full else [1, 1])
                self.assertEqual(owner.request_receiver.max_recv_per_poll, -1)

    def setUp(self):
        super().setUp()
        scope = get_context().override_server_args(
            enable_unified_cache_external_linker=True,
            mooncake_enable_waiting_queue_dfs_prefetch=True,
            disaggregation_mode="prefill",
            schedule_policy="fcfs",
        )
        scope.install()
        self.addCleanup(scope.restore)

    def test_startup_follows_native_capability(self):
        owner = MooncakePrefetchSchedulerMixin()
        owner.disaggregation_mode = DisaggregationMode.PREFILL
        owner.schedule_policy = "fcfs"
        owner.input_blocker = None
        owner.tree_cache = SimpleNamespace(waiting_queue_prefetch_enabled=lambda: True)
        self.assertEqual(owner._resolve_prefetch_startup_switches(None), (True, True))
        owner.tree_cache.waiting_queue_prefetch_enabled = lambda: False
        self.assertEqual(owner._resolve_prefetch_startup_switches(None), (False, False))

    def test_input_blocker_disables_forward_ingress_but_keeps_prefetch(self):
        owner = MooncakePrefetchSchedulerMixin()
        owner.disaggregation_mode = DisaggregationMode.PREFILL
        owner.schedule_policy = "fcfs"
        owner.input_blocker = object()
        owner.tree_cache = SimpleNamespace(waiting_queue_prefetch_enabled=lambda: True)
        self.assertEqual(owner._resolve_prefetch_startup_switches(None), (True, False))

    def test_decode_uses_direct_forward_without_ingress_collectives(self):
        owner = MooncakePrefetchSchedulerMixin()
        owner._forward_ingress_enabled = True
        owner.model_worker = SimpleNamespace(
            forward_batch_generation=lambda batch: batch
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
        )
        self.assertIs(owner._forward_with_waiting_queue_ingress(batch), batch)

    def test_rematch_cannot_restore_a_failed_external_admission(self):
        owner = MooncakePrefetchSchedulerMixin()
        hits = {"rid": object()}
        owner.tree_cache = SimpleNamespace(
            cancel_waiting_queue_prefetch=lambda rid: hits.pop(rid, None),
            prepare_external_linker_loads=lambda reqs: ["failed"],
        )
        # The final match finds keys even though the asynchronous refresh failed.
        req = SimpleNamespace(
            rid="rid",
            host_hit_length=64,
            swa_host_hit_length=32,
            mamba_host_hit_length=0,
        )
        self.assertTrue(owner._apply_external_linker_admission(req))
        self.assertEqual((req.host_hit_length, req.swa_host_hit_length), (0, 0))
        self.assertFalse(hits)

    def test_encoder_waiting_input_is_deferred_even_before_mm_inputs_arrive(self):
        req = SimpleNamespace(
            mm_inputs=None,
            sampling_params=SimpleNamespace(),
            need_wait_for_mm_inputs=True,
        )
        self.assertFalse(
            MooncakePrefetchSchedulerMixin._can_prepare_forward_ingress(req)
        )
        req.need_wait_for_mm_inputs = False
        self.assertTrue(
            MooncakePrefetchSchedulerMixin._can_prepare_forward_ingress(req)
        )

    def test_background_rounds_do_not_mix_with_foreground_collectives(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        children = mp.spawn(_round_worker, args=(port,), nprocs=2, join=False)
        deadline = time.monotonic() + 150
        while not children.join(timeout=2):
            if time.monotonic() >= deadline:
                for process in children.processes:
                    if process.is_alive():
                        process.kill()
                self.fail("Prefetch ranks did not finish their ordered rounds")


if __name__ == "__main__":
    unittest.main()
