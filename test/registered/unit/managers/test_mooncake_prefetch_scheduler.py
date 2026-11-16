"""Keep speculative native I/O collectives separate from scheduler/model groups."""

import datetime
import queue
import socket
import threading
import time
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.mooncake_prefetch import (
    MooncakePrefetchSchedulerMixin,
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
        owner.tree_cache = SimpleNamespace(waiting_queue_prefetch_enabled=lambda: True)
        self.assertEqual(owner._resolve_prefetch_startup_switches(None), (True, True))
        owner.tree_cache.waiting_queue_prefetch_enabled = lambda: False
        self.assertEqual(owner._resolve_prefetch_startup_switches(None), (False, False))

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
