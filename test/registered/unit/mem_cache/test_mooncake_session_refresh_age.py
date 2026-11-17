"""Admission may skip the lease refresh of a freshly leased prefetch session.

``SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S`` lets ``revalidate_host_prefetch``
trust a private session whose oldest key was leased less than that many
seconds ago. The age is measured from a monotonic timestamp taken before the
RPC that granted the lease, so it never understates the lease Mooncake sees.
"""

import threading
import types
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from unittest import mock
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ContendedLock:
    """A lock that signals when a thread starts waiting for it while held."""

    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self):
        if self._lock.locked():
            self.contended.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


class _Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def _linker(age_s=5.0, refresh_results=None, refresh_error=None):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.session_refresh_age_s = age_s
    linker.host_prefetch_enabled = True
    linker.host_prefetch_lock = threading.Lock()
    linker.host_prefetch_entries = {}
    linker.host_prefetch_limit = 8
    linker.host_prefetch_worker_count = 2
    linker._init_session_runtime()

    def start(keys):
        return [0] * len(keys), ["dfs"] * len(keys)

    def refresh(keys):
        if refresh_error is not None:
            raise refresh_error
        if refresh_results is not None:
            return list(refresh_results)
        return [0] * len(keys)

    store = types.SimpleNamespace(
        batch_get_session_start_with_sources=MagicMock(side_effect=start),
        batch_get_session_refresh=MagicMock(side_effect=refresh),
        batch_get_session_end=MagicMock(return_value=0),
    )
    linker.storage = types.SimpleNamespace(
        store=store,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
        _tag_keys=lambda keys: list(keys),
    )
    return linker


def _prefetched(linker, clock, rid="rid", keys=("k0", "k1"), session="s"):
    transfer = types.SimpleNamespace(keys=list(keys))
    with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
        assert linker._prepare_expanded_load(session, [transfer])
    linker.host_prefetch_entries[rid] = {
        "state": "dfs_prefetched",
        "session_rid": session,
    }


def _revalidate(linker, clock, rid="rid"):
    with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
        return linker.revalidate_host_prefetch(rid)


class TestSessionRefreshAge(CustomTestCase):
    def test_young_private_session_skips_refresh(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 4.9
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_old_session_refreshes_and_resets_its_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 5.0
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once_with(
            ["k0", "k1"]
        )
        self.assertEqual(linker.session_lease_base, {"k0": 105.0, "k1": 105.0})
        # Freshly refreshed, so the next admission can skip.
        clock.now += 1.0
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_disabled_always_refreshes(self):
        clock = _Clock()
        linker = _linker(age_s=0.0)
        _prefetched(linker, clock)
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_oldest_key_decides(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        linker.session_lease_base["k1"] = clock.now - 10.0
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_key_without_base_counts_as_old(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        del linker.session_lease_base["k0"]
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_shared_session_always_refreshes(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        # A second session reuses k1: its Mooncake session is no longer private.
        _prefetched(linker, clock, rid="other", keys=("k1",), session="s2")
        self.assertEqual(linker.session_refcounts["k1"], 2)
        self.assertTrue(_revalidate(linker, clock))
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 2)

    def test_reused_key_loses_its_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 3.0
        _prefetched(linker, clock, rid="other", keys=("k1", "k2"), session="s2")
        self.assertEqual(linker.session_lease_base, {"k0": 100.0, "k2": 103.0})

    def test_formerly_shared_key_stays_refresh_only(self):
        """Review P1-1: A shares k with B; A's refresh retires the native
        session and A is released, leaving B the sole holder of a young base."""
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock, rid="b", keys=("k",), session="sb")
        _prefetched(linker, clock, rid="a", keys=("k",), session="sa")
        linker.storage.store.batch_get_session_refresh.reset_mock()
        clock.now += 1.0
        linker.storage.store.batch_get_session_refresh.side_effect = lambda keys: [
            7
        ] * len(keys)
        self.assertFalse(_revalidate(linker, clock, rid="a"))
        linker._abort_prepared_load_now("sa")
        self.assertEqual(linker.session_refcounts, {"k": 1})
        # B is private again and young, but must still ask Mooncake.
        self.assertFalse(_revalidate(linker, clock, rid="b"))
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 2)
        # Even a successful refresh does not restore the skip for this generation.
        linker.storage.store.batch_get_session_refresh.side_effect = lambda keys: [
            0
        ] * len(keys)
        self.assertTrue(_revalidate(linker, clock, rid="b"))
        self.assertTrue(_revalidate(linker, clock, rid="b"))
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 4)
        self.assertEqual(linker.session_lease_base, {})

    def test_released_key_restarts_with_a_fresh_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock, rid="b", keys=("k",), session="sb")
        _prefetched(linker, clock, rid="a", keys=("k",), session="sa")
        linker._abort_prepared_load_now("sa")
        linker._abort_prepared_load_now("sb")
        linker.storage.store.batch_get_session_refresh.reset_mock()
        clock.now += 2.0
        _prefetched(linker, clock, rid="c", keys=("k",), session="sc")
        self.assertEqual(linker.session_lease_base, {"k": 102.0})
        self.assertTrue(_revalidate(linker, clock, rid="c"))
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_failed_refresh_is_invalid_and_drops_bases(self):
        clock = _Clock()
        linker = _linker(refresh_results=[0, 7])
        _prefetched(linker, clock)
        clock.now += 6.0
        self.assertFalse(_revalidate(linker, clock))
        self.assertEqual(linker.session_lease_base, {})

    def test_refresh_exception_is_invalid(self):
        clock = _Clock()
        linker = _linker(refresh_error=RuntimeError("master unreachable"))
        _prefetched(linker, clock)
        clock.now += 6.0
        with self.assertLogs(level="WARNING"):
            self.assertFalse(_revalidate(linker, clock))
        self.assertEqual(linker.session_lease_base, {})

    def test_not_prefetched_is_invalid_without_age(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        linker.host_prefetch_entries["rid"]["cancelled"] = True
        self.assertFalse(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_release_drops_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        _prefetched(linker, clock, rid="other", keys=("k1",), session="s2")
        linker._abort_prepared_load_now("s")
        # k1 was shared, so it lost its base already.
        self.assertEqual(linker.session_lease_base, {})
        self.assertEqual(linker.session_refcounts, {"k1": 1})
        linker._abort_prepared_load_now("s2")
        self.assertEqual(linker.session_lease_base, {})

    def test_unrelated_start_does_not_block_ready_admission(self):
        """Review P1-R1: another request's session start holds session_lock
        across its RPC; the admission check must age past that wait."""
        clock = _Clock()
        linker = _linker()
        linker.session_lock = _ContendedLock()
        _prefetched(linker, clock)
        clock.now += 1.0
        in_rpc = threading.Event()
        release = threading.Event()

        def slow_start(keys):
            in_rpc.set()
            if not release.wait(5):
                raise TimeoutError("the test never released the start RPC")
            return [0] * len(keys), ["dfs"] * len(keys)

        linker.storage.store.batch_get_session_start_with_sources.side_effect = (
            slow_start
        )
        transfer = types.SimpleNamespace(keys=["unrelated"])
        with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
            worker = threading.Thread(
                target=linker._prepare_expanded_load, args=("other", [transfer])
            )
            worker.start()
            self.assertTrue(in_rpc.wait(5))
            result = {}
            admission = threading.Thread(
                target=lambda: result.update(ok=linker.revalidate_host_prefetch("rid"))
            )
            admission.start()
            # The ready request must progress before the unrelated RPC returns.
            admission.join(2)
            progressed = not admission.is_alive()
            release.set()
            worker.join(5)
            admission.join(5)
            self.assertTrue(progressed)
        self.assertFalse(worker.is_alive())
        self.assertFalse(admission.is_alive())
        self.assertTrue(result["ok"])
        linker.storage.store.batch_get_session_refresh.assert_not_called()
        self.assertEqual(
            linker.session_lease_base, {"k0": 100.0, "k1": 100.0, "unrelated": 101.0}
        )

    def test_claim_wait_does_not_hold_host_prefetch_lock(self):
        linker = _linker()
        session_lock = _ContendedLock()
        linker.session_lock = session_lock
        linker.prepared_load_sessions = {"session": ["k0"]}
        linker.host_prefetch_entries["rid"] = {
            "state": "dfs_prefetched",
            "session_rid": "session",
        }
        linker.host_prefetch_entries["worker"] = {"state": "reading"}

        claim = {}
        claim_done = threading.Event()
        update_done = threading.Event()

        def claim_ready():
            try:
                claim["result"] = linker.claim_ready_host_prefetch("rid")
            except BaseException as exc:
                claim["error"] = exc
            finally:
                claim_done.set()

        def publish_ready():
            with linker.host_prefetch_lock:
                linker.host_prefetch_entries["worker"]["state"] = "dfs_prefetched"
            update_done.set()

        with session_lock:
            claimant = threading.Thread(target=claim_ready)
            claimant.start()
            claim_waiting_for_session = session_lock.contended.wait(5)
            worker = threading.Thread(target=publish_ready)
            worker.start()
            updated_while_session_locked = update_done.wait(5)

        claimant.join(5)
        worker.join(5)
        self.assertFalse(claimant.is_alive())
        self.assertFalse(worker.is_alive())
        self.assertTrue(claim_done.is_set())
        self.assertTrue(update_done.is_set())
        self.assertTrue(claim_waiting_for_session)
        self.assertTrue(
            updated_while_session_locked,
            "claim blocked host-prefetch state updates while waiting for session_lock",
        )
        self.assertNotIn("error", claim)
        self.assertTrue(claim["result"])
        self.assertEqual(linker.host_prefetch_entries["rid"]["state"], "claimed")
        self.assertEqual(linker.prepared_load_sessions, {"rid": ["k0"]})

    def test_end_waits_for_same_key_refresh(self):
        linker = _linker(age_s=0.0)
        clock = _Clock()
        _prefetched(linker, clock)
        in_rpc, release, ended = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def refresh(keys):
            in_rpc.set()
            if not release.wait(5):
                raise TimeoutError("refresh was not released")
            return [0] * len(keys)

        linker.storage.store.batch_get_session_refresh.side_effect = refresh
        linker.storage.store.batch_get_session_end.side_effect = lambda keys: (
            ended.set(),
            0,
        )[1]
        refresh_thread = threading.Thread(
            target=linker.revalidate_host_prefetch, args=("rid",)
        )
        refresh_thread.start()
        self.assertTrue(in_rpc.wait(5))
        end_thread = threading.Thread(
            target=linker._abort_prepared_load_now, args=("s",)
        )
        end_thread.start()
        self.assertFalse(ended.wait(0.1))
        release.set()
        refresh_thread.join(5)
        end_thread.join(5)
        self.assertFalse(refresh_thread.is_alive())
        self.assertFalse(end_thread.is_alive())
        self.assertTrue(ended.is_set())
        self.assertEqual(linker.session_lease_base, {})


class TestSharedSessionPreparation(CustomTestCase):
    def test_new_session_does_not_refresh_even_with_age_skip_disabled(self):
        linker = _linker(age_s=0.0)
        transfer = types.SimpleNamespace(keys=["k0", "k1"])
        self.assertTrue(linker._prepare_expanded_load("b", [transfer]))
        linker.storage.store.batch_get_session_start_with_sources.assert_called_once_with(
            ["k0", "k1"]
        )
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_expired_shared_session_is_renewed_before_read_and_keeps_other_holder(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        lease_deadline = clock.now + 8.0
        clock.now += 10.0

        def refresh(keys):
            nonlocal lease_deadline
            self.assertEqual(keys, ["k0", "k1"])
            lease_deadline = clock.now + 8.0
            return [0, 0]

        linker.storage.store.batch_get_session_refresh.side_effect = refresh
        transfers = [types.SimpleNamespace(keys=["k0", "k1"])]
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda *a, **k: transfers
        )
        with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
            self.assertTrue(linker.prepare_load("b", transfers))
        self.assertLess(clock.now, lease_deadline)  # native read now has a live lease
        self.assertEqual(linker.session_refcounts, {"k0": 2, "k1": 2})
        linker._abort_prepared_load_now("b")
        self.assertEqual(linker.prepared_load_sessions, {"s": ["k0", "k1"]})
        self.assertEqual(linker.session_refcounts, {"k0": 1, "k1": 1})
        linker.storage.store.batch_get_session_end.assert_not_called()
        self.assertTrue(_revalidate(linker, clock))
        linker._abort_prepared_load_now("s")
        linker.storage.store.batch_get_session_end.assert_called_once_with(["k0", "k1"])

    def test_mixed_batch_refreshes_all_reused_keys_only(self):
        linker = _linker()
        _prefetched(linker, _Clock())
        store = linker.storage.store
        store.batch_get_session_start_with_sources.reset_mock()
        transfer = types.SimpleNamespace(keys=["k0", "k1", "k2"])
        self.assertTrue(linker._prepare_expanded_load("b", [transfer]))
        store.batch_get_session_refresh.assert_called_once_with(["k0", "k1"])
        store.batch_get_session_start_with_sources.assert_called_once_with(["k2"])

    def test_missing_session_partial_failure_or_exception_cannot_reuse_old_refs(self):
        for result in ([0, 7], [], RuntimeError("session missing")):
            with self.subTest(result=result):
                linker = _linker()
                _prefetched(linker, _Clock())
                store = linker.storage.store
                if isinstance(result, BaseException):
                    store.batch_get_session_refresh.side_effect = result
                else:
                    store.batch_get_session_refresh.side_effect = lambda keys: result
                self.assertFalse(
                    linker._prepare_expanded_load(
                        "b", [types.SimpleNamespace(keys=["k0", "k1"])]
                    )
                )
                self.assertEqual(linker.prepared_load_sessions, {"s": ["k0", "k1"]})
                self.assertEqual(linker.session_refcounts, {"k0": 1, "k1": 1})
                self.assertEqual(linker.session_lease_base, {})
                store.batch_get_session_end.assert_not_called()
                self.assertEqual(
                    store.batch_get_session_start_with_sources.call_count, 1
                )
                self.assertFalse(_revalidate(linker, _Clock()))

    def test_ordinary_revalidation_exception_fails_closed(self):
        linker = _linker()
        transfer = types.SimpleNamespace(keys=["k"])
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda *a, **k: [transfer]
        )
        linker.storage._batch_exist = MagicMock(
            side_effect=RuntimeError("master failed")
        )
        self.assertFalse(linker.revalidate_load([transfer]))


class TestSessionOwnershipRaces(CustomTestCase):
    def test_new_start_cannot_overtake_last_end(self):
        linker = _linker()
        old = types.SimpleNamespace(keys=["key"])
        self.assertTrue(linker._prepare_expanded_load("old", [old]))
        in_end, release_end, new_started = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self.addCleanup(release_end.set)
        sessions = {"key": "old"}

        def end(keys):
            in_end.set()
            if not release_end.wait(5):
                raise TimeoutError("end was not released")
            sessions.clear()
            return 0

        def start(keys):
            sessions["key"] = "new"
            new_started.set()
            return [0], ["dfs"]

        linker.storage.store.batch_get_session_end.side_effect = end
        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        ending = threading.Thread(target=linker._abort_prepared_load_now, args=("old",))
        ending.start()
        self.assertTrue(in_end.wait(5))
        self.assertFalse(linker._abort_prepared_load_now("old"))
        self.assertFalse(linker.abort_prepared_load_async("old"))
        result = {}
        starting = threading.Thread(
            target=lambda: result.update(ok=linker._prepare_expanded_load("new", [old]))
        )
        starting.start()
        self.assertFalse(new_started.wait(0.1))
        release_end.set()
        ending.join(5)
        starting.join(5)
        self.assertFalse(ending.is_alive())
        self.assertFalse(starting.is_alive())
        self.assertTrue(result["ok"])
        self.assertEqual(sessions, {"key": "new"})
        self.assertEqual(linker.session_refcounts, {"key": 1})
        self.assertEqual(linker.session_key_locks, {})

    def test_cancel_inflight_start_rolls_back_before_publishing_ready(self):
        linker = _linker()
        in_start, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        sessions = set()

        def start(keys):
            in_start.set()
            if not release.wait(5):
                raise TimeoutError("start was not released")
            sessions.update(keys)
            return [0], ["dfs"]

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        linker.storage.store.batch_get_session_end.side_effect = lambda keys: (
            sessions.difference_update(keys),
            0,
        )[1]
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                ok=linker._prepare_expanded_load(
                    "rid", [types.SimpleNamespace(keys=["key"])]
                )
            )
        )
        worker.start()
        self.assertTrue(in_start.wait(5))
        self.assertFalse(linker._abort_prepared_load_now("rid"))
        release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(result["ok"])
        self.assertEqual(sessions, set())
        self.assertEqual(linker.prepared_load_sessions, {})
        self.assertEqual(linker.session_refcounts, {})
        self.assertTrue(linker._abort_prepared_load_now("rid"))

    def test_async_cancel_pending_start_keeps_bounded_credit_until_drained(self):
        linker = _linker()
        linker.host_prefetch_limit = 1
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers, **_: transfers
        )
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(linker.session_executor.shutdown, wait=True)
        self.addCleanup(release.set)

        def start(keys):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("start was not released")
            return [0], ["dfs"]

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        transfer = types.SimpleNamespace(keys=["key"])
        self.assertIn(linker.prepare_load_async("a", [transfer]), {"pending", "ready"})
        self.assertTrue(entered.wait(5))
        self.assertFalse(linker.abort_prepared_load_async("a"))
        self.assertEqual(linker.prepare_load_async("b", [transfer]), "failed")
        self.assertEqual(len(linker.session_jobs), 1)
        release.set()
        self.assertFalse(linker.session_jobs[("prepare", "a")].result(timeout=5))
        self.assertTrue(linker.abort_prepared_load_async("a"))
        self.assertEqual(linker.prepared_load_sessions, {})
        self.assertEqual(linker.session_jobs, {})

    def test_queued_load_cancel_returns_before_end_and_retains_cleanup_credit(self):
        linker = _linker()
        linker.host_prefetch_limit = 1
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers, **_: transfers
        )
        transfer = types.SimpleNamespace(keys=["key"])
        self.assertTrue(linker._prepare_expanded_load("a", [transfer]))
        prepared = Future()
        prepared.set_result(True)
        linker.session_jobs[("prepare", "a")] = prepared
        linker.host_prefetch_entries["a"] = {"state": "loading", "session_rid": "a"}
        linker.pending_loads = {"a": [transfer]}
        linker.pending_load_tokens, linker.pending_load_started = {"a": 1}, {"a": 1.0}
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(linker.session_executor.shutdown, wait=True)
        self.addCleanup(release.set)
        rpc_threads = []

        def end(keys):
            rpc_threads.append(threading.get_ident())
            entered.set()
            if not release.wait(5):
                raise TimeoutError("end was not released")
            return 0

        linker.storage.store.batch_get_session_end.side_effect = end
        caller = threading.get_ident()
        self.assertTrue(linker.cancel_queued_load("a"))
        self.assertTrue(entered.wait(5))
        self.assertNotEqual(rpc_threads, [caller])
        self.assertIn(("prepare", "a"), linker.session_jobs)
        self.assertEqual(linker.host_prefetch_entries["a"]["state"], "cleanup_pending")
        self.assertEqual(linker.cancelled_load_session_cleanup, {"a"})
        self.assertEqual(linker.prepare_load_async("b", [transfer]), "failed")
        linker._poll_cancelled_load_sessions()
        self.assertEqual(linker.cancelled_load_session_cleanup, {"a"})
        ended = linker.session_jobs[("end", "a")]
        release.set()
        self.assertTrue(ended.result(timeout=5))
        linker._poll_cancelled_load_sessions()
        self.assertEqual(linker.cancelled_load_session_cleanup, set())
        self.assertEqual(linker.session_jobs, {})
        self.assertEqual(linker.host_prefetch_entries, {})
        self.assertEqual(linker.pending_loads, {})

    def test_no_prefetch_load_consumes_each_admission_result(self):
        linker = _linker()
        linker.host_prefetch_enabled = False
        linker.host_prefetch_limit = 1
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers, **_: transfers
        )
        linker.pending_loads = {}
        linker.pending_load_tokens, linker.pending_load_started = {}, {}
        linker.page_size = 1
        transfer = types.SimpleNamespace(name="kv", keys=["key"])
        for index in range(4):
            rid = str(index)
            proof = Future()
            proof.set_result(True)
            linker.session_jobs[("refresh", rid)] = proof
            self.assertTrue(linker.load(rid, [transfer]))
            self.assertEqual(linker.session_jobs, {})
            self.assertIn(rid, linker.pending_loads)
            self.assertTrue(linker.cancel_queued_load(rid))

    def test_unknown_end_outcome_is_not_reported_as_success_or_reused(self):
        linker = _linker()
        transfer = types.SimpleNamespace(keys=["key"])
        self.assertTrue(linker._prepare_expanded_load("a", [transfer]))
        linker.storage.store.batch_get_session_end.return_value = -1
        with self.assertLogs(level="WARNING"):
            self.assertFalse(linker._abort_prepared_load_now("a"))
        self.assertFalse(linker._abort_prepared_load_now("a"))
        self.assertFalse(linker._prepare_expanded_load("b", [transfer]))
        self.assertEqual(
            linker.storage.store.batch_get_session_start_with_sources.call_count, 1
        )

    def test_request_id_distinguishes_reused_rid_lifecycles(self):
        linker = _linker()
        first = types.SimpleNamespace(rid="r", lifecycle_id=1, attempt_id=0)
        second = types.SimpleNamespace(rid="r", lifecycle_id=2, attempt_id=0)
        self.assertNotEqual(linker.request_id(first), linker.request_id(second))


class TestAsyncSessionAdmission(CustomTestCase):
    def test_ready_prefetch_cancel_retries_internal_end_after_quota_recovers(self):
        linker = _linker()
        linker.host_prefetch_limit = 1
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers, **_: transfers
        )
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(linker.session_executor.shutdown, wait=True)
        self.addCleanup(release.set)
        for rid, key in (("busy", "busy-key"), ("internal", "prefetch-key")):
            self.assertTrue(
                linker._prepare_expanded_load(rid, [types.SimpleNamespace(keys=[key])])
            )
        entry = {"state": "dfs_prefetched", "session_rid": "internal"}
        linker.host_prefetch_entries["public"] = entry

        def end(keys):
            if keys == ["busy-key"]:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("busy end was not released")
            return 0

        linker.storage.store.batch_get_session_end.side_effect = end
        busy = linker._session_job(
            "end", "busy", lambda: linker.abort_prepared_load("busy")
        )
        self.assertTrue(entered.wait(5))
        linker.cancel_host_prefetch("public")
        self.assertEqual(
            linker.host_prefetch_session_cleanup["public"], (entry, "internal")
        )
        self.assertIn("internal", linker.prepared_load_sessions)
        self.assertNotIn(("end", "internal"), linker.session_jobs)
        self.assertEqual(entry["state"], "cleanup_pending")
        release.set()
        self.assertTrue(busy.result(timeout=5))
        self.assertTrue(linker.abort_prepared_load_async("busy"))
        linker._poll_cancelled_load_sessions()
        with linker.session_lock:
            pending = linker.session_jobs.get(("end", "internal"))
        if pending is not None:
            self.assertTrue(pending.result(timeout=5))
        linker._poll_cancelled_load_sessions()
        self.assertEqual(linker.host_prefetch_session_cleanup, {})
        self.assertEqual(linker.host_prefetch_entries, {})
        self.assertEqual(linker.prepared_load_sessions, {})
        self.assertEqual(linker.session_jobs, {})
        self.assertEqual(linker.session_refcounts, {})

    def test_refresh_result_is_reused_briefly_then_renewed_while_peer_is_pending(self):
        for ttl in (10.0, 0.4):
            with self.subTest(ttl=ttl), envs.SGLANG_MOONCAKE_LEASE_TTL_S.override(ttl):
                linker = _linker(age_s=0.0)
                self.addCleanup(linker.session_executor.shutdown, wait=True)
                clock = _Clock()
                with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
                    _prefetched(linker, clock)
                    linker.revalidate_host_prefetch_async("rid")
                    first = linker.session_jobs[("refresh", "rid")]
                    self.assertTrue(first.result(timeout=5))
                    window = min(ttl / 2, 1.0)
                    clock.now += window / 2
                    self.assertEqual(
                        linker.revalidate_host_prefetch_async("rid"), "ready"
                    )
                    self.assertIs(linker.session_jobs[("refresh", "rid")], first)
                    clock.now += window
                    linker.revalidate_host_prefetch_async("rid")
                    renewed = linker.session_jobs[("refresh", "rid")]
                    self.assertIsNot(renewed, first)
                    self.assertTrue(renewed.result(timeout=5))
                    self.assertEqual(
                        linker.revalidate_host_prefetch_async("rid"), "ready"
                    )
                self.assertEqual(
                    linker.storage.store.batch_get_session_refresh.call_count, 2
                )

    def test_expired_prepare_refreshes_existing_lease_without_restarting_it(self):
        linker = _linker(age_s=5.0)
        self.addCleanup(linker.session_executor.shutdown, wait=True)
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers, **_: transfers
        )
        clock = _Clock()
        transfer = types.SimpleNamespace(keys=["key"])
        with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
            linker.prepare_load_async("rid", [transfer])
            first = linker.session_jobs[("prepare", "rid")]
            self.assertTrue(first.result(timeout=5))
            clock.now += 1.1
            linker.prepare_load_async("rid", [transfer])
            renewed = linker.session_jobs[("prepare", "rid")]
            self.assertIsNot(renewed, first)
            self.assertTrue(renewed.result(timeout=5))
            self.assertEqual(linker.prepare_load_async("rid", [transfer]), "ready")
        self.assertEqual(
            linker.storage.store.batch_get_session_start_with_sources.call_count, 1
        )
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 1)
        self.assertEqual(linker.session_refcounts, {"key": 1})

    def test_slow_rpc_cannot_publish_already_expired_admission(self):
        linker = _linker(age_s=0.0)
        self.addCleanup(linker.session_executor.shutdown, wait=True)
        clock = _Clock()

        def slow_refresh(keys):
            clock.now += linker.session_lease_ttl_s / 2 + 0.1
            return [0] * len(keys)

        linker.storage.store.batch_get_session_refresh.side_effect = slow_refresh
        with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
            _prefetched(linker, clock)
            linker.revalidate_host_prefetch_async("rid")
            self.assertFalse(linker.session_jobs[("refresh", "rid")].result(timeout=5))
            self.assertEqual(linker.revalidate_host_prefetch_async("rid"), "failed")


class TestSessionRefreshAgeConfig(CustomTestCase):
    def _config(self, **env):
        overrides = ExitStack()
        try:
            for name in (
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S",
                "SGLANG_MOONCAKE_LEASE_TTL_S",
            ):
                field = getattr(envs, name)
                overrides.enter_context(field.override(env.get(name, field.default)))
            return mooncake_direct_linker._session_refresh_age_config()
        finally:
            # close() restores descriptors even when configuration validation raises.
            overrides.close()

    def test_default_is_disabled(self):
        self.assertEqual(self._config(), 0.0)

    def test_age_below_ttl_is_kept(self):
        self.assertEqual(self._config(SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S="5"), 5.0)

    def test_age_not_below_ttl_disables(self):
        with self.assertLogs(level="WARNING"):
            self.assertEqual(
                self._config(
                    SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S="8",
                    SGLANG_MOONCAKE_LEASE_TTL_S="8",
                ),
                0.0,
            )

    def test_negative_age_is_rejected(self):
        with self.assertRaises(ValueError):
            self._config(SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S="-1")

    def test_non_finite_values_are_rejected(self):
        for env in (
            {"SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "nan"},
            {"SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "inf"},
            {
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "20",
                "SGLANG_MOONCAKE_LEASE_TTL_S": "nan",
            },
            {
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "20",
                "SGLANG_MOONCAKE_LEASE_TTL_S": "inf",
            },
            {"SGLANG_MOONCAKE_LEASE_TTL_S": "0"},
            {"SGLANG_MOONCAKE_LEASE_TTL_S": "-3"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self._config(**env)

    def test_group_semantics_disable_skipping(self):
        for grouped, expected in ((True, 0.0), (False, 5.0)):
            linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
            linker.session_refresh_age_s = 5.0
            linker.storage = types.SimpleNamespace(
                _can_use_group_semantics=lambda grouped=grouped: grouped
            )
            with self.subTest(grouped=grouped):
                if grouped:
                    with self.assertLogs(level="WARNING"):
                        linker._disable_refresh_skip_for_groups()
                else:
                    linker._disable_refresh_skip_for_groups()
                self.assertEqual(linker.session_refresh_age_s, expected)

    def test_class_defaults_disable_skipping(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        self.assertEqual(linker.session_refresh_age_s, 0.0)


if __name__ == "__main__":
    unittest.main()
