"""Worker session admission must agree before any rank commits device KV."""

import datetime
import pathlib
from queue import PriorityQueue, Queue
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import LinkerTransferPhase
from sglang.test.ci.ci_register import register_cpu_ci
from test_load_back_construction_consensus import (
    FakeComponent,
    _wrapper,
)
from test_mooncake_session_refresh_age import _linker

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class MetadataComponent(FakeComponent):
    def build_external_linker_transfer(self, phase, node, keys):
        if phase == LinkerTransferPhase.LOOKUP:
            return PoolTransfer(name=self.name, keys=list(keys))
        return super().build_external_linker_transfer(phase, node, keys)


def backend():
    linker = _linker(age_s=0)
    linker.host_prefetch_limit = 8
    linker.host_prefetch_queue = PriorityQueue(maxsize=16)
    linker.load_admissions = {}
    linker.pending_loads = {}
    linker.page_size = 1
    linker.pool_group = types.SimpleNamespace(
        resolve_transfers=lambda transfers, **_: transfers
    )
    return linker


def run_job(linker):
    _, _, (rid, transfers, admission) = linker.host_prefetch_queue.get_nowait()
    linker._run_load_admission(rid, transfers, admission)
    linker.host_prefetch_queue.task_done()


def _admission_rank(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=15),
    )
    try:
        for scenario in (
            "start_fail",
            "start_raise",
            "refresh_fail",
            "refresh_raise",
            "shared_order",
            "swa_subset",
            "same_prefix",
        ):
            linker = backend()
            component = MetadataComponent(PoolName.KV)
            cache = types.SimpleNamespace(
                page_size=1,
                _components_tuple=(component,),
                tree_core=types.SimpleNamespace(
                    empty_match_result=types.SimpleNamespace(
                        device_indices=torch.empty(0, dtype=torch.int64)
                    )
                ),
                _all_reduce_attn_groups=lambda tensor, op: dist.all_reduce(
                    tensor, op=op
                ),
            )
            wrapper = _wrapper(cache, linker, False)
            hit = wrapper.hit_markers["r"]
            keys = ["h0", "h1"]
            if scenario.startswith("refresh") or scenario in (
                "shared_order",
                "swa_subset",
            ):
                session_keys = (
                    list(reversed(keys)) if scenario == "shared_order" else keys
                )
                assert linker._prepare_expanded_load(
                    "session", [types.SimpleNamespace(keys=session_keys)]
                )
                linker.host_prefetch_entries["r"] = dict(
                    state="dfs_prefetched", session_rid="session"
                )
                wrapper.host_prefetch_hits["r"] = hit
                if scenario.startswith("refresh"):
                    for key in keys:
                        linker.session_lease_base[key] = time.monotonic() - 6
            if scenario == "same_prefix":
                assert linker._prepare_expanded_load(
                    "other", [types.SimpleNamespace(keys=["h1"])]
                )
            if rank == 1 and scenario.endswith("fail"):
                if scenario.startswith("start"):
                    linker.storage.store.batch_get_session_start_with_sources.return_value = (
                        [7, 7],
                        ["dfs", "dfs"],
                    )
                    linker.storage.store.batch_get_session_start_with_sources.side_effect = (
                        None
                    )
                else:
                    linker.storage.store.batch_get_session_refresh.side_effect = (
                        lambda keys: [7] * len(keys)
                    )
            if rank == 1 and scenario.endswith("raise"):
                method = (
                    linker.storage.store.batch_get_session_start_with_sources
                    if scenario.startswith("start")
                    else linker.storage.store.batch_get_session_refresh
                )
                method.side_effect = RuntimeError("master unavailable")
            req = types.SimpleNamespace(rid="r", last_node="node")
            assert not wrapper.prepare_load_admission(req)
            run_job(linker)
            assert wrapper.prepare_load_admission(req)
            if scenario.endswith(("fail", "raise")):
                indices, node = wrapper.load_back(req)
                assert indices.numel() == 0 and node == "node"
                assert component.allocated == 0
                assert not linker.prepared_load_sessions
            else:
                if wrapper.host_prefetch_hits:
                    assert linker.claim_ready_host_prefetch("r")
                actual = ["h1"] if scenario == "swa_subset" else keys
                if scenario == "shared_order":
                    actual = list(reversed(keys))
                assert linker.load("r", [PoolTransfer(name=PoolName.KV, keys=actual)])
                linker.storage.store.batch_get_session_refresh.reset_mock()
                assert linker._prepare_expanded_load(
                    "r", [types.SimpleNamespace(keys=actual)]
                )
                linker.storage.store.batch_get_session_refresh.assert_not_called()
                linker.abort_prepared_load("r")
                if scenario == "same_prefix":
                    assert linker.session_refcounts == {"h1": 1}
                    linker._abort_prepared_load_now("other")
                assert not linker.session_refcounts
        # A one-rank allocation failure must release ordinary admissions on
        # every rank, repeatedly beyond the admission limit.
        linker = backend()
        linker.host_prefetch_limit = 2
        for _ in range(6):
            component = MetadataComponent(PoolName.KV)
            cache._components_tuple = (component,)
            wrapper = _wrapper(cache, linker, False)
            req = types.SimpleNamespace(rid="r", last_node="node")
            assert not wrapper.prepare_load_admission(req)
            run_job(linker)
            assert wrapper.prepare_load_admission(req)
            if rank == 1:
                component.failure = "none"
            indices, _ = wrapper.load_back(req)
            assert indices.numel() == 0 and component.allocated == 0
            assert not linker.load_admissions and not linker.prepared_load_sessions
            assert not linker.session_refcounts
        component.failure = None
        wrapper = _wrapper(cache, linker, False)
        assert not wrapper.prepare_load_admission(req)
        run_job(linker)
        assert wrapper.prepare_load_admission(req)
        assert wrapper.load_admission_verdicts["r"]
        linker.abort_prepared_load("r")

        # Ordinary revalidation without a LayerSplit group also agrees.
        linker = backend()
        linker.revalidate_load = lambda transfers: rank == 0
        component = MetadataComponent(PoolName.KV)
        cache._components_tuple = (component,)
        wrapper = _wrapper(cache, linker, False)
        indices, _ = wrapper.load_back(types.SimpleNamespace(rid="r", last_node="node"))
        assert indices.numel() == 0 and component.allocated == 0
    finally:
        dist.destroy_process_group()


class TestLoadAdmission(unittest.TestCase):
    def test_over_capacity_requests_wait_and_all_retain_hits(self):
        linker = backend()
        requests = [str(index) for index in range(24)]
        transfers = [PoolTransfer(name=PoolName.KV, keys=["k"])]
        for rid in requests:
            self.assertEqual(linker.prepare_load_admission(rid, transfers), "pending")
        self.assertEqual(len(linker.load_admissions), 8)
        completed = set()
        for _ in range(4):
            while not linker.host_prefetch_queue.empty():
                run_job(linker)
            for rid in requests:
                if rid in completed:
                    continue
                state = linker.prepare_load_admission(rid, transfers)
                self.assertNotEqual(state, "failed")
                if state == "ready":
                    completed.add(rid)
                    linker.abort_prepared_load(rid)
        self.assertEqual(len(completed), 24)
        self.assertFalse(linker.load_admissions)
        self.assertFalse(linker.session_refcounts)

    def test_queue_and_cleanup_pressure_retry_without_losing_admission(self):
        linker = backend()
        transfers = [PoolTransfer(name=PoolName.KV, keys=["k"])]
        linker.pending_session_ends = 8
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        self.assertFalse(linker.load_admissions)
        linker.pending_session_ends = 0
        for index in range(16):
            linker.host_prefetch_queue.put((1, index, ("prefetch", [])))
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        linker.host_prefetch_queue.get_nowait()
        linker.host_prefetch_queue.task_done()
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        run_job(linker)  # Admission overtakes the remaining prefetch tasks.
        self.assertEqual(linker.prepare_load_admission("r", transfers), "ready")
        self.assertEqual(linker.host_prefetch_queue.qsize(), 15)

    def test_capacity_wait_has_total_timeout_in_wrapper(self):
        linker = backend()
        linker.host_prefetch_limit = 0
        component = MetadataComponent(PoolName.KV)
        cache = types.SimpleNamespace(
            _components_tuple=(component,),
            _all_reduce_attn_groups=lambda *args: None,
        )
        wrapper = _wrapper(cache, linker, False)
        req = types.SimpleNamespace(rid="r")
        self.assertFalse(wrapper.prepare_load_admission(req))
        started = req.external_load_admission_started
        with patch("time.monotonic", return_value=started + 20):
            self.assertFalse(wrapper.prepare_load_admission(req))
        self.assertEqual(req.external_load_admission_started, started)
        with patch("time.monotonic", return_value=started + 31):
            self.assertTrue(wrapper.prepare_load_admission(req))
        self.assertFalse(wrapper.load_admission_verdicts["r"])

    def test_cancelled_generation_cannot_end_replacement_session(self):
        linker = backend()
        entered, release = threading.Event(), threading.Event()

        def start(keys):
            if keys == ["old"]:
                entered.set()
                assert release.wait(5)
            return [0] * len(keys), ["dfs"] * len(keys)

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        old = [PoolTransfer(name=PoolName.KV, keys=["old"])]
        new = [PoolTransfer(name=PoolName.KV, keys=["new"])]
        linker.prepare_load_admission("r", old)
        old_session = linker.load_admissions["r"]["session_rid"]
        worker = threading.Thread(target=run_job, args=(linker,))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            linker.cancel_host_prefetch("r")
            linker.prepare_load_admission("r", new)
            self.assertNotEqual(old_session, linker.load_admissions["r"]["session_rid"])
            run_job(linker)
            self.assertEqual(linker.prepare_load_admission("r", new), "ready")
            self.assertTrue(linker.load("r", new))
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(linker.prepared_load_sessions, {"r": ["new"]})
        self.assertTrue(linker.load_admission_valid("r"))
        linker.abort_prepared_load("r")
        self.assertFalse(linker.session_refcounts)

    def test_fresh_prefetch_reuses_original_lease_deadline(self):
        linker = backend()
        transfers = [PoolTransfer(name=PoolName.KV, keys=["k"])]
        self.assertTrue(linker.prepare_load("session", transfers))
        base = linker.session_lease_base["k"]
        linker.host_prefetch_entries["r"] = dict(
            state="dfs_prefetched", session_rid="session"
        )
        linker.prepare_load_admission("r", transfers)
        run_job(linker)
        self.assertEqual(linker.prepare_load_admission("r", transfers), "ready")
        linker.storage.store.batch_get_session_refresh.assert_not_called()
        self.assertEqual(linker.load_admissions["r"]["deadline"], base + 5)

    def test_disabled_prefetch_empty_commit_and_cleanup_are_safe(self):
        linker = backend()
        linker.host_prefetch_enabled = False
        del linker.host_prefetch_lock
        wrapper = _wrapper(None, linker, False)
        wrapper._queue_load("r", "node", [], anchor="node")
        linker.cancel_host_prefetch("r")
        linker.abort_prepared_load("r")

    def test_two_rank_failures_and_valid_subsets(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _admission_rank,
                args=(str(pathlib.Path(directory) / "gloo"),),
                nprocs=2,
                join=True,
            )

    def test_blocked_master_does_not_block_admission_or_cancellation(self):
        linker = backend()
        entered, release = threading.Event(), threading.Event()

        def start(keys):
            entered.set()
            assert release.wait(5)
            return [0] * len(keys), ["dfs"] * len(keys)

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        transfers = [PoolTransfer(name=PoolName.KV, keys=["k"])]
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        worker = threading.Thread(target=run_job, args=(linker,))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            before = time.monotonic()
            self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
            wrapper = _wrapper(None, linker, False)
            wrapper.cancel_waiting_queue_prefetch("r")
            self.assertLess(time.monotonic() - before, 0.5)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(linker.prepared_load_sessions)
        self.assertFalse(linker.load_admissions)

    def test_cancellation_keeps_active_load_session_until_worker_drains(self):
        linker = backend()
        linker.pending_loads = {}
        linker.page_size = 1
        transfers = [PoolTransfer(name=PoolName.KV, keys=["k"])]
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        run_job(linker)
        self.assertTrue(linker.load("r", transfers))
        linker.cancel_host_prefetch("r")
        self.assertEqual(linker.session_refcounts, {"k": 1})
        linker.storage.store.batch_get_session_end.assert_not_called()
        linker.abort_prepared_load("r")  # The worker's terminal cleanup.
        self.assertFalse(linker.session_refcounts)
        linker.storage.store.batch_get_session_end.assert_called_once_with(["k"])

    def test_expired_readiness_refreshes_and_queue_is_bounded(self):
        linker = backend()
        transfers = [PoolTransfer(name=PoolName.KV, keys=["k"])]
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        run_job(linker)
        self.assertEqual(linker.prepare_load_admission("r", transfers), "ready")
        linker.load_admissions["r"]["deadline"] = 0
        linker.session_lease_base["k"] = time.monotonic() - 6
        linker.storage.store.batch_get_session_refresh.side_effect = lambda keys: [
            7
        ] * len(keys)
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        run_job(linker)
        self.assertEqual(linker.prepare_load_admission("r", transfers), "failed")
        self.assertFalse(linker.prepared_load_sessions)
        for i in range(100):
            self.assertEqual(
                linker.prepare_load_admission(str(i), transfers), "pending"
            )
        self.assertLessEqual(linker.host_prefetch_queue.qsize(), 16)
        self.assertLessEqual(len(linker.load_admissions), 8)

    def test_admission_refreshes_even_with_large_refresh_skip_age(self):
        linker = backend()
        linker.session_refresh_age_s = 9
        transfers = [PoolTransfer(name=PoolName.KV, keys=["k"])]
        self.assertTrue(linker.prepare_load("session", transfers))
        linker.host_prefetch_entries["r"] = dict(
            state="dfs_prefetched", session_rid="session"
        )
        linker.session_lease_base["k"] = time.monotonic() - 6
        self.assertEqual(linker.prepare_load_admission("r", transfers), "pending")
        run_job(linker)
        self.assertEqual(linker.prepare_load_admission("r", transfers), "ready")
        linker.storage.store.batch_get_session_refresh.assert_called_once_with(["k"])

    def test_prepared_subset_refreshes_only_read_keys_and_releases_all_refs(self):
        linker = backend()
        self.assertTrue(
            linker._prepare_expanded_load(
                "r", [types.SimpleNamespace(keys=["k2", "k1", "k3"])]
            )
        )
        linker.storage.store.batch_get_session_refresh.reset_mock()
        self.assertTrue(
            linker._prepare_expanded_load("r", [types.SimpleNamespace(keys=["k3"])])
        )
        linker.storage.store.batch_get_session_refresh.assert_called_once_with(["k3"])
        linker._abort_prepared_load_now("r")
        self.assertFalse(linker.session_refcounts)
        linker.storage.store.batch_get_session_end.assert_called_once_with(
            ["k2", "k1", "k3"]
        )

    def test_async_end_reserves_key_until_rpc_finishes(self):
        linker = backend()
        linker.session_end_queue = Queue()
        linker.pending_session_ends = 0
        transfers = [types.SimpleNamespace(keys=["k"])]
        self.assertTrue(linker._prepare_expanded_load("r", transfers))
        linker.abort_prepared_load("r")
        self.assertEqual(linker.pending_session_ends, 1)
        linker.storage.store.batch_get_session_end.assert_not_called()
        self.assertIn("k", linker.session_operations)
        worker = threading.Thread(target=linker.session_end_thread_func)
        worker.start()
        linker.session_end_queue.join()
        linker.session_end_queue.put(None)
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(linker.session_operations)
        self.assertEqual(linker.pending_session_ends, 0)
        linker.storage.store.batch_get_session_end.assert_called_once_with(["k"])


if __name__ == "__main__":
    unittest.main()
