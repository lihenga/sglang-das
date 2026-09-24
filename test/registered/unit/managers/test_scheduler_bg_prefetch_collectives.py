"""Waiting-queue DFS prefetch submission and admission coverage."""

import queue
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    PreparedHostPrefetch,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


class _FakeLinker:
    def __init__(self, *, fail_after_queue=False, submit_started=None, allow_submit=None):
        self.fail_after_queue = fail_after_queue
        self.submit_started = submit_started
        self.allow_submit = allow_submit
        self.submissions = []
        self.cancelled = []
        self.statuses = {}
        self.submit_thread_id = None

    def submit_host_prefetch(self, rid, transfers):
        self.submissions.append((rid, transfers))
        self.submit_thread_id = threading.get_ident()
        self.statuses[rid] = "queued"
        if self.submit_started is not None:
            self.submit_started.set()
        if self.allow_submit is not None and not self.allow_submit.wait(timeout=5):
            raise TimeoutError("test submit gate timed out")
        if self.fail_after_queue:
            raise RuntimeError("submit failed after queueing")
        return True

    def get_host_prefetch_status(self, rid):
        return self.statuses.get(rid)

    def cancel_host_prefetch(self, rid):
        self.cancelled.append(rid)
        self.statuses.pop(rid, None)


class _FakeTreeCache:
    def __init__(self, wrapper, job, admission_callback=None):
        self.wrapper = wrapper
        self.job = job
        self.admission_callback = admission_callback
        self.prepared = []
        self.completed = []

    def prepare_external_linker_prefetch(self, req):
        self.prepared.append(req.rid)
        return self.job

    def complete_external_linker_prefetch(self, job, submitted):
        self.completed.append((job.rid, submitted))
        return self.wrapper.complete_host_prefetch_submission(job, submitted)

    def cancel_waiting_queue_prefetch(self, rid):
        self.wrapper.cancel_waiting_queue_prefetch(rid)

    def get_waiting_queue_prefetch_admission_states(self, rids):
        if self.admission_callback is not None:
            return self.admission_callback(rids)
        return {}


class _FakeAdmissionLinker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def get_host_prefetch_admission_state(self, rid):
        if rid == "complete":
            return "dfs_prefetched"
        if rid == "pending":
            return "pending" if self.rank == 0 else "dfs_prefetched"
        if rid == "missing":
            return (
                "not_tracked"
                if self.rank == self.world_size - 1
                else "dfs_prefetched"
            )
        if rid == "failed":
            return "terminal" if self.rank == self.world_size - 1 else "dfs_prefetched"
        return "not_tracked"


def _assert_rank_wide_admission_states(rank, world_size, group):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.attn_cp_group = group
    cache.attn_tp_group = None
    cache.tp_world_size = world_size
    cache.tp_group = group
    cache.linker = _FakeAdmissionLinker(rank, world_size)

    scenarios = (
        ("complete", ["complete"], "dfs_prefetched"),
        ("pending", ["pending"], "pending"),
        ("missing", ["missing"] if rank == 0 else [], "terminal"),
        ("failed", ["failed"], "terminal"),
    )
    for rid, local_rids, expected in scenarios:
        states = cache.get_waiting_queue_prefetch_admission_states(local_rids)
        assert states == {rid: expected}, (world_size, rank, rid, states)


def _admission_worker(rank, world_size, port):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        two_rank_group = dist.new_group(ranks=[0, 1], backend="gloo")
        if rank < 2:
            _assert_rank_wide_admission_states(rank, 2, two_rank_group)
        dist.barrier()

        _assert_rank_wide_admission_states(rank, world_size, dist.group.WORLD)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestSchedulerWaitingQueuePrefetch(CustomTestCase):
    def _make_scheduler(
        self,
        *,
        fail_after_queue=False,
        locally_eligible=True,
        submit_started=None,
        allow_submit=None,
        start_worker=True,
        admission_callback=None,
    ):
        rid = "request-1"
        marker = object()
        linker = _FakeLinker(
            fail_after_queue=fail_after_queue,
            submit_started=submit_started,
            allow_submit=allow_submit,
        )
        job = PreparedHostPrefetch(
            rid=rid,
            locally_eligible=locally_eligible,
            transfers=("immutable-snapshot",),
            cache_linker=linker,
            cancelled=threading.Event(),
        )
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache_linker = linker
        wrapper.hit_markers = {}
        wrapper.pending_host_prefetch_submissions = {rid: (job, marker)}
        wrapper.host_prefetch_hits = {}
        tree_cache = _FakeTreeCache(wrapper, job, admission_callback)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_waiting_queue_dfs_prefetch = True
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = tree_cache
        scheduler._waiting_queue_prefetch_jobs = queue.Queue()
        scheduler._waiting_queue_prefetch_acks = queue.Queue()
        scheduler._waiting_queue_prefetch_stop = threading.Event()
        scheduler._waiting_queue_prefetch_worker = None
        if start_worker:
            self._start_worker(scheduler)

        class Request:
            def __init__(self):
                self.prefill_attempt_count = 0
                self.rid = rid

            def init_next_round_input(self, cache, cow_mamba=False):
                assert cache is tree_cache
                assert cow_mamba is False

        req = Request()
        scheduler.waiting_queue = [req]
        return scheduler, req, linker, wrapper, tree_cache

    @staticmethod
    def _start_worker(scheduler):
        scheduler._waiting_queue_prefetch_worker = threading.Thread(
            target=scheduler._waiting_queue_prefetch_worker_loop, daemon=True
        )
        scheduler._waiting_queue_prefetch_worker.start()

    @staticmethod
    def _stop_worker(scheduler):
        worker = scheduler._waiting_queue_prefetch_worker
        if worker is not None:
            scheduler._waiting_queue_prefetch_jobs.put(None)
            worker.join(timeout=10)
            assert not worker.is_alive(), "prefetch worker did not stop"

    @staticmethod
    def _wait_for_ack(scheduler):
        ack = scheduler._waiting_queue_prefetch_acks.get(timeout=5)
        scheduler._waiting_queue_prefetch_acks.put(ack)

    def test_enqueue_queues_snapshot_and_scheduler_drains_worker_ack(self):
        submit_started = threading.Event()
        allow_submit = threading.Event()
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            submit_started=submit_started, allow_submit=allow_submit
        )
        caller_thread = threading.get_ident()

        try:
            with (
                mock.patch.object(
                    dist,
                    "all_reduce",
                    side_effect=AssertionError("unexpected collective"),
                ),
                mock.patch.object(
                    dist,
                    "all_gather_object",
                    side_effect=AssertionError("unexpected collective"),
                ),
            ):
                scheduler._prefetch_kvcache(req)
                assert tree_cache.prepared == [req.rid]
                assert tree_cache.completed == []
                assert req.rid in wrapper.pending_host_prefetch_submissions
                assert submit_started.wait(timeout=5)
                assert linker.submissions == [(req.rid, ["immutable-snapshot"])]
                assert linker.submit_thread_id != caller_thread
                assert wrapper.get_host_prefetch_admission_state(req.rid) == "pending"

                allow_submit.set()
                self._wait_for_ack(scheduler)
                assert tree_cache.completed == []
                scheduler._drain_waiting_queue_prefetch_acks()

            assert tree_cache.completed == [(req.rid, True)]
            assert req.rid not in wrapper.pending_host_prefetch_submissions
            assert req.rid in wrapper.host_prefetch_hits
        finally:
            allow_submit.set()
            self._stop_worker(scheduler)

    def test_failed_local_submit_cancels_backend_session(self):
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            fail_after_queue=True
        )

        try:
            scheduler._prefetch_kvcache(req)
            self._wait_for_ack(scheduler)
            with mock.patch("sglang.srt.managers.scheduler.logger.error"):
                scheduler._drain_waiting_queue_prefetch_acks()

            assert linker.submissions == [(req.rid, ["immutable-snapshot"])]
            assert linker.cancelled == [req.rid]
            assert linker.statuses == {}
            assert req.rid not in wrapper.host_prefetch_hits
            assert req.rid not in wrapper.pending_host_prefetch_submissions
            assert tree_cache.completed == [(req.rid, False)]
        finally:
            self._stop_worker(scheduler)

    def test_cancel_before_worker_submit_skips_backend(self):
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            start_worker=False
        )

        scheduler._prefetch_kvcache(req)
        tree_cache.cancel_waiting_queue_prefetch(req.rid)
        self._start_worker(scheduler)
        try:
            self._wait_for_ack(scheduler)
            scheduler._drain_waiting_queue_prefetch_acks()

            assert linker.submissions == []
            assert linker.cancelled == [req.rid]
            assert req.rid not in wrapper.pending_host_prefetch_submissions
            assert tree_cache.completed == [(req.rid, False)]
        finally:
            self._stop_worker(scheduler)

    def test_cancel_during_submit_retires_queued_backend_session(self):
        submit_started = threading.Event()
        allow_submit = threading.Event()
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            submit_started=submit_started, allow_submit=allow_submit
        )

        try:
            scheduler._prefetch_kvcache(req)
            assert submit_started.wait(timeout=5)
            tree_cache.cancel_waiting_queue_prefetch(req.rid)
            allow_submit.set()
            self._wait_for_ack(scheduler)
            scheduler._drain_waiting_queue_prefetch_acks()

            assert linker.submissions == [(req.rid, ["immutable-snapshot"])]
            assert linker.cancelled == [req.rid, req.rid]
            assert linker.statuses == {}
            assert req.rid not in wrapper.host_prefetch_hits
            assert req.rid not in wrapper.pending_host_prefetch_submissions
        finally:
            allow_submit.set()
            self._stop_worker(scheduler)

    def test_stop_cancels_backlog_and_drains_inflight_ack(self):
        submit_started = threading.Event()
        allow_submit = threading.Event()
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            submit_started=submit_started, allow_submit=allow_submit
        )
        stop_thread = None

        try:
            scheduler._prefetch_kvcache(req)
            assert submit_started.wait(timeout=5)

            queued_job = PreparedHostPrefetch(
                rid="request-2",
                locally_eligible=True,
                transfers=("second-snapshot",),
                cache_linker=linker,
                cancelled=threading.Event(),
            )
            wrapper.pending_host_prefetch_submissions[queued_job.rid] = (
                queued_job,
                object(),
            )
            scheduler._waiting_queue_prefetch_jobs.put(queued_job)

            stop_thread = threading.Thread(
                target=scheduler._stop_waiting_queue_prefetch_worker
            )
            stop_thread.start()
            assert scheduler._waiting_queue_prefetch_stop.wait(timeout=5)
            assert queued_job.cancelled.wait(timeout=5)
            assert tree_cache.completed == [(queued_job.rid, False)]
            assert stop_thread.is_alive(), "stop should wait for the active submit"
            assert linker.submissions == [(req.rid, ["immutable-snapshot"])]

            allow_submit.set()
            stop_thread.join(timeout=10)
            assert not stop_thread.is_alive(), "stop did not finish after submit returned"

            assert scheduler._waiting_queue_prefetch_worker is None
            assert linker.submissions == [(req.rid, ["immutable-snapshot"])]
            assert tree_cache.completed == [
                (queued_job.rid, False),
                (req.rid, True),
            ]
            assert req.rid in wrapper.host_prefetch_hits
            assert req.rid not in wrapper.pending_host_prefetch_submissions
            assert queued_job.rid not in wrapper.pending_host_prefetch_submissions
        finally:
            allow_submit.set()
            if stop_thread is not None:
                stop_thread.join(timeout=10)
            self._stop_worker(scheduler)

    def test_ineligible_snapshot_is_not_submitted(self):
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            locally_eligible=False, start_worker=False
        )

        scheduler._prefetch_kvcache(req)

        assert linker.submissions == []
        assert req.rid not in wrapper.host_prefetch_hits
        assert req.rid not in wrapper.pending_host_prefetch_submissions
        assert tree_cache.completed == [(req.rid, False)]

    def test_ack_is_drained_before_admission_state_query(self):
        def check_ack_applied(rids):
            assert rids == [req.rid]
            assert tree_cache.completed == [(req.rid, True)]
            assert req.rid not in wrapper.pending_host_prefetch_submissions
            raise RuntimeError("admission query reached")

        scheduler, req, _linker, wrapper, tree_cache = self._make_scheduler(
            start_worker=False, admission_callback=check_ack_applied
        )
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        job = tree_cache.job
        wrapper.cache_linker.statuses[req.rid] = "queued"
        scheduler._waiting_queue_prefetch_acks.put((job, True, None))

        with mock.patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=SimpleNamespace(enable_flexkv=False),
        ), self.assertRaisesRegex(RuntimeError, "admission query reached"):
            scheduler._get_new_batch_prefill_raw(None, object())

        assert tree_cache.completed == [(req.rid, True)]

    def test_admission_is_consistent_for_two_and_eight_ranks(self):
        world_size = 8
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(
            _admission_worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
