"""Admission may skip the lease refresh of a freshly leased prefetch session.

``SGLANG_MOONCAKE_SESSION_REFRESH_MAX_AGE_RATIO`` lets
``revalidate_host_prefetch`` trust a private session whose oldest key was
leased less than that fraction of Mooncake's lease TTL ago. The age is measured
from a monotonic timestamp taken before the RPC that granted the lease, so it
never understates the lease Mooncake sees.
"""

import inspect
import os
import threading
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock

from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.utils import common
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


def _linker(ratio=0.5, ttl_ms=10000, refresh_results=None, refresh_error=None):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.session_refresh_max_age_ratio = ratio
    linker.host_prefetch_enabled = True
    linker.host_prefetch_lock = threading.Lock()
    linker.host_prefetch_entries = {}
    linker.session_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}

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
        get_lease_ttl_ms=MagicMock(return_value=ttl_ms),
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
        linker = _linker(ratio=0.0)
        _prefetched(linker, clock)
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()
        linker.storage.store.get_lease_ttl_ms.assert_not_called()

    def test_skip_needs_age_strictly_below_the_limit(self):
        # q = 0.5 of a 10000 ms TTL: the limit is 5 s.
        for elapsed, skipped in ((4.999, True), (5.0, False), (5.001, False)):
            clock = _Clock()
            linker = _linker()
            _prefetched(linker, clock)
            clock.now += elapsed
            with self.subTest(elapsed=elapsed):
                self.assertTrue(_revalidate(linker, clock))
                refresh = linker.storage.store.batch_get_session_refresh
                self.assertEqual(refresh.called, not skipped)

    def test_ttl_changes_resize_the_window_and_are_logged(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        getter = linker.storage.store.get_lease_ttl_ms
        clock.now = 104.0
        with self.assertLogs(mooncake_direct_linker.logger, level="INFO") as logs:
            self.assertTrue(_revalidate(linker, clock))
            self.assertTrue(_revalidate(linker, clock))
            getter.return_value = 6000
            # 4 s is not below 0.5 * 6 s.
            self.assertTrue(_revalidate(linker, clock))
            getter.return_value = 10000
            clock.now = 115.0
            self.assertTrue(_revalidate(linker, clock))
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 2)
        self.assertEqual(
            [r.getMessage() for r in logs.records if "lease TTL" in r.getMessage()],
            [f"Mooncake lease TTL: {ttl} ms" for ttl in (10000, 6000, 10000)],
        )

    def test_raised_ttl_never_skips_a_session_with_a_shorter_lease(self):
        """Replays the 5 -> 15 -> 20 sequence of Mooncake's UpdateLeaseTtl test:
        every reply leases a session and moves the client's TTL as that rule
        does, so the getter and the Python bases follow the same events.
        Refresh replies carry 20 s here, which leaves the TTL as it is."""
        # (time s, session, its lease TTL in ms or None to check it, TTL in
        #  effect, skipped with q = 0.5)
        steps = (
            (0.0, "a", 5000, 5000, None),
            (14.9, "b", 15000, 5000, None),  # 15 s becomes the candidate
            (20.0, "c", 20000, 5000, None),  # another TTL restarts the wait
            (20.5, "c", None, 5000, True),
            (40.0, "d", 20000, 20000, None),  # seen alone for 20 s: raised
            (40.5, "b", None, 20000, False),  # 25.6 s old
            (40.5, "d", None, 20000, True),
            (41.0, "e", 5000, 5000, None),  # a smaller TTL applies at once
            (41.5, "d", None, 5000, True),  # 1.5 s old, below 2.5 s
        )
        clock = _Clock(0.0)
        linker = _linker()
        store = linker.storage.store
        leases = {}
        for at, name, lease_ms, ttl_ms, skipped in steps:
            clock.now = at
            if lease_ms is not None:
                _prefetched(linker, clock, rid=name, keys=(name,), session=name)
                leases[name] = lease_ms
                store.get_lease_ttl_ms.return_value = ttl_ms
                continue
            calls = store.batch_get_session_refresh.call_count
            with self.subTest(at=at, session=name):
                self.assertTrue(_revalidate(linker, clock, rid=name))
                did_skip = store.batch_get_session_refresh.call_count == calls
                self.assertEqual(did_skip, skipped)
                if leases[name] < ttl_ms:
                    self.assertFalse(did_skip)

    def test_unusable_ttl_refreshes_and_keeps_bases_to_the_refresh(self):
        for value in (0, -1, True, 10000.0, "10000", None, 2**64, RuntimeError()):
            clock = _Clock()
            linker = _linker()
            getter = linker.storage.store.get_lease_ttl_ms
            if isinstance(value, Exception):
                getter.side_effect = value
            else:
                getter.return_value = value
            _prefetched(linker, clock)
            clock.now += 1.0
            with self.subTest(value=value):
                self.assertTrue(_revalidate(linker, clock))
                linker.storage.store.batch_get_session_refresh.assert_called_once()
                self.assertEqual(linker.session_lease_base, {"k0": 101.0, "k1": 101.0})

    def test_missing_getter_refreshes_and_warns_once(self):
        mooncake_direct_linker.print_warning_once.cache_clear()
        clock = _Clock()
        linker = _linker()
        del linker.storage.store.get_lease_ttl_ms
        _prefetched(linker, clock)
        with self.assertLogs(level="WARNING") as logs:
            self.assertTrue(_revalidate(linker, clock))
            self.assertTrue(_revalidate(linker, clock))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 2)

    def test_log_failure_does_not_change_the_decision(self):
        kept = {"k0": 100.0, "k1": 100.0}
        renewed = {"k0": 101.0, "k1": 101.0}
        info = (mooncake_direct_linker.logger, "info")
        warning = (common.logger, "warning")
        # (getter TTL or None if missing, failing log call, refresh results,
        #  refreshes, bases afterwards)
        cases = (
            (10000, info, None, 0, kept),
            (None, warning, None, 1, renewed),
            (None, warning, [0, 7], 1, {}),
        )
        for ttl_ms, (log, level), results, refreshes, bases in cases:
            mooncake_direct_linker.print_warning_once.cache_clear()
            clock = _Clock()
            linker = _linker(ttl_ms=ttl_ms, refresh_results=results)
            if ttl_ms is None:
                del linker.storage.store.get_lease_ttl_ms
            _prefetched(linker, clock)
            clock.now += 1.0
            with self.subTest(ttl_ms=ttl_ms, level=level, results=results):
                with mock.patch.object(log, level, side_effect=RuntimeError("log")):
                    self.assertEqual(_revalidate(linker, clock), results is None)
                refresh = linker.storage.store.batch_get_session_refresh
                self.assertEqual(refresh.call_count, refreshes)
                self.assertEqual(linker.session_lease_base, bases)

    def test_age_includes_the_getter_wait(self):
        # 4.9 s old before the getter, 5.1 s once it returns: not below 5 s.
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now = 104.9

        def slow_getter():
            clock.now = 105.1
            return 10000

        linker.storage.store.get_lease_ttl_ms.side_effect = slow_getter
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
        linker.storage.store.batch_get_session_refresh.assert_called_once()

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

    def test_age_includes_session_lock_wait(self):
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
            # Admission blocks on the lock while the unrelated start RPC runs
            # past the skip threshold.
            self.assertTrue(linker.session_lock.contended.wait(5))
            clock.now += 10.0
            release.set()
            worker.join(5)
            admission.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(admission.is_alive())
        self.assertTrue(result["ok"])
        # Aged 11 s once it got the lock, so it refreshed instead of skipping.
        linker.storage.store.batch_get_session_refresh.assert_called_once()
        self.assertEqual(
            linker.session_lease_base, {"k0": 111.0, "k1": 111.0, "unrelated": 101.0}
        )

    def test_refresh_skips_base_of_key_ended_meanwhile(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 6.0

        def refresh(keys):
            # The session is released while the refresh RPC is in flight.
            linker._abort_prepared_load_now("s")
            return [0] * len(keys)

        linker.storage.store.batch_get_session_refresh.side_effect = refresh
        self.assertTrue(_revalidate(linker, clock))
        self.assertEqual(linker.session_lease_base, {})


_RATIO = "SGLANG_MOONCAKE_SESSION_REFRESH_MAX_AGE_RATIO"
# PR20's age and declared TTL, and v5's ratio: none of them is read any more.
_LEGACY = (
    "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S",
    "SGLANG_MOONCAKE_LEASE_TTL_S",
    "SGLANG_MOONCAKE_SESSION_REFRESH_MIN_REMAINING_RATIO",
)


class TestSessionRefreshAgeConfig(CustomTestCase):
    def _config(self, env):
        with mock.patch.dict(os.environ, env, clear=False):
            for name in (_RATIO, *_LEGACY):
                if name not in env:
                    os.environ.pop(name, None)
            return mooncake_direct_linker._session_refresh_skip_config()

    def test_ratio_values(self):
        # (ratio variable or None if unset, ratio, warns)
        cases = (
            (None, 0.0, False),
            ("0", 0.0, False),
            ("-0.0", 0.0, False),
            ("", 0.0, False),
            ("0.5", 0.5, False),
            ("0.999", 0.999, False),
            ("1", 0.0, True),
            ("2.5", 0.0, True),
        )
        for raw, expected, warns in cases:
            env = {} if raw is None else {_RATIO: raw}
            with self.subTest(env=env):
                logs = self.assertLogs if warns else self.assertNoLogs
                with logs(level="WARNING"):
                    self.assertEqual(self._config(env), expected)

    def test_legacy_variables_have_no_effect(self):
        ratios = (({}, 0.0), ({_RATIO: "0"}, 0.0), ({_RATIO: "0.5"}, 0.5))
        for ratio_env, expected in ratios:
            for names in (*((name,) for name in _LEGACY), _LEGACY):
                for value in ("5", "abc", "-1", "nan"):
                    env = {**ratio_env, **dict.fromkeys(names, value)}
                    with self.subTest(env=env), self.assertNoLogs(level="WARNING"):
                        self.assertEqual(self._config(env), expected)
        source = inspect.getsource(mooncake_direct_linker)
        for name in _LEGACY:
            self.assertNotIn(name, source)

    def test_invalid_values_are_rejected(self):
        for raw in ("-0.1", "nan", "inf", "-inf", "abc"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self._config({_RATIO: raw})

    def test_group_semantics_disable_skipping(self):
        for grouped in (True, False):
            linker = _linker()
            linker.storage._can_use_group_semantics = lambda g=grouped: g
            with self.subTest(grouped=grouped):
                logs = self.assertLogs if grouped else self.assertNoLogs
                with logs(level="WARNING"):
                    linker._disable_refresh_skip_for_groups()
                self.assertEqual(
                    linker._session_refresh_max_age_s(), 0.0 if grouped else 5.0
                )

    def test_class_defaults_disable_skipping(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        self.assertEqual(linker.session_refresh_max_age_ratio, 0.0)
        self.assertFalse(hasattr(linker, "session_refresh_age_s"))


if __name__ == "__main__":
    unittest.main()
