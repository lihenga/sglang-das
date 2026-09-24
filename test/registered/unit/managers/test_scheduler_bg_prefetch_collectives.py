"""Waiting-queue DFS prefetch submission and admission coverage."""

import socket
import threading
import unittest
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
    def __init__(self, *, fail_after_queue=False):
        self.fail_after_queue = fail_after_queue
        self.submissions = []
        self.cancelled = []
        self.statuses = {}
        self.submit_thread_id = None

    def submit_host_prefetch(self, rid, transfers):
        self.submissions.append((rid, transfers))
        self.submit_thread_id = threading.get_ident()
        self.statuses[rid] = "queued"
        if self.fail_after_queue:
            raise RuntimeError("submit failed after queueing")
        return True

    def get_host_prefetch_status(self, rid):
        return self.statuses.get(rid)

    def cancel_host_prefetch(self, rid):
        self.cancelled.append(rid)
        self.statuses.pop(rid, None)


class _FakeTreeCache:
    def __init__(self, wrapper, job):
        self.wrapper = wrapper
        self.job = job
        self.prepared = []
        self.completed = []

    def prepare_external_linker_prefetch(self, req):
        self.prepared.append(req.rid)
        return self.job

    def complete_external_linker_prefetch(self, job, submitted):
        self.completed.append((job.rid, submitted))
        return self.wrapper.complete_host_prefetch_submission(job, submitted)


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
    def _make_scheduler(self, *, fail_after_queue=False, locally_eligible=True):
        rid = "request-1"
        marker = object()
        linker = _FakeLinker(fail_after_queue=fail_after_queue)
        job = PreparedHostPrefetch(
            rid=rid,
            locally_eligible=locally_eligible,
            transfers=("immutable-snapshot",),
            cache_linker=linker,
            cancelled=threading.Event(),
        )
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache_linker = linker
        wrapper.pending_host_prefetch_submissions = {rid: (job, marker)}
        wrapper.host_prefetch_hits = {}
        tree_cache = _FakeTreeCache(wrapper, job)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_waiting_queue_dfs_prefetch = True
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = tree_cache

        class Request:
            def __init__(self):
                self.prefill_attempt_count = 0
                self.rid = rid

            def init_next_round_input(self, cache, cow_mamba=False):
                assert cache is tree_cache
                assert cow_mamba is False

        return scheduler, Request(), linker, wrapper, tree_cache

    def test_enqueue_submits_snapshot_locally_before_returning(self):
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler()
        caller_thread = threading.get_ident()

        with (
            mock.patch.object(
                dist, "all_reduce", side_effect=AssertionError("unexpected collective")
            ),
            mock.patch.object(
                dist,
                "all_gather_object",
                side_effect=AssertionError("unexpected collective"),
            ),
        ):
            scheduler._prefetch_kvcache(req)

        assert tree_cache.prepared == [req.rid]
        assert linker.submissions == [(req.rid, ["immutable-snapshot"])]
        assert linker.submit_thread_id == caller_thread
        assert tree_cache.completed == [(req.rid, True)]
        assert req.rid in wrapper.host_prefetch_hits
        assert wrapper.get_host_prefetch_admission_state(req.rid) == "pending"

    def test_failed_local_submit_cancels_backend_session(self):
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            fail_after_queue=True
        )

        with mock.patch("sglang.srt.managers.scheduler.logger.exception"):
            scheduler._prefetch_kvcache(req)

        assert linker.submissions == [(req.rid, ["immutable-snapshot"])]
        assert linker.cancelled == [req.rid]
        assert req.rid not in wrapper.host_prefetch_hits
        assert req.rid not in wrapper.pending_host_prefetch_submissions
        assert tree_cache.completed == [(req.rid, False)]

    def test_ineligible_snapshot_is_not_submitted(self):
        scheduler, req, linker, wrapper, tree_cache = self._make_scheduler(
            locally_eligible=False
        )

        scheduler._prefetch_kvcache(req)

        assert linker.submissions == []
        assert req.rid not in wrapper.host_prefetch_hits
        assert req.rid not in wrapper.pending_host_prefetch_submissions
        assert tree_cache.completed == [(req.rid, False)]

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
