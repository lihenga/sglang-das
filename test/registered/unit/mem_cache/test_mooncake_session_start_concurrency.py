"""Session-start RPCs overlap only when their keys and RIDs are disjoint."""
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import MooncakeDirectLinker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def linker(start):
    obj = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    obj.host_prefetch_enabled = True
    obj.session_refresh_age_s = 0
    obj.session_lock = threading.Lock()
    obj.host_prefetch_lock = threading.Lock()
    obj.host_prefetch_entries = {}
    obj.prepared_load_sessions = {}
    obj.session_refcounts = {}
    obj.session_sources = {}
    obj.session_lease_base = {}
    obj._log_host_prefetch_metric = MagicMock()
    obj.storage = types.SimpleNamespace(
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
        _tag_keys=lambda keys: keys,
        store=types.SimpleNamespace(
            batch_get_session_start_with_sources=MagicMock(side_effect=start),
            batch_get_session_refresh=MagicMock(side_effect=lambda keys: [0] * len(keys)),
            batch_get_session_end=MagicMock(return_value=0),
        ),
    )
    return obj


def prepare(obj, rid, *keys):
    return obj._prepare_expanded_load(rid, [types.SimpleNamespace(keys=list(keys))])


class TestConcurrentSessionStart(CustomTestCase):
    def test_disjoint_starts_overlap_inside_native_rpc(self):
        both = threading.Barrier(2)
        def start(keys):
            both.wait(timeout=5)
            return [0] * len(keys), ["memory"] * len(keys)
        obj = linker(start)
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(prepare, obj, "a", "k0")
            b = pool.submit(prepare, obj, "b", "k1")
            self.assertTrue(a.result(timeout=8))
            self.assertTrue(b.result(timeout=8))
        self.assertEqual(obj.session_refcounts, {"k0": 1, "k1": 1})
        self.assertEqual(obj.session_starting, {})

    def blocked(self, first_result=None):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def start(keys):
            calls.append(list(keys))
            if len(calls) == 1:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("start was not released")
                if first_result is not None:
                    return first_result, ["memory"] * len(keys)
            return [0] * len(keys), ["memory"] * len(keys)
        return linker(start), entered, release, calls

    def test_shared_key_waits_and_is_not_started_twice(self):
        obj, entered, release, calls = self.blocked()
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(prepare, obj, "a", "shared")
            self.assertTrue(entered.wait(5))
            b = pool.submit(prepare, obj, "b", "shared", "extra")
            release.set()
            self.assertTrue(a.result(timeout=8))
            self.assertTrue(b.result(timeout=8))
        self.assertEqual(calls, [["shared"], ["extra"]])
        self.assertEqual(obj.session_refcounts, {"shared": 2, "extra": 1})
        obj._abort_prepared_load_now("a")
        self.assertEqual(obj.session_refcounts["shared"], 1)
        obj._abort_prepared_load_now("b")
        self.assertEqual(obj.session_refcounts, {})

    def test_same_rid_cannot_overwrite_an_inflight_start(self):
        obj, entered, release, calls = self.blocked()
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(prepare, obj, "same", "k0")
            self.assertTrue(entered.wait(5))
            b = pool.submit(prepare, obj, "same", "k1")
            release.set()
            self.assertTrue(a.result(timeout=8))
            self.assertFalse(b.result(timeout=8))
        self.assertEqual(calls, [["k0"]])
        self.assertEqual(obj.prepared_load_sessions, {"same": ["k0"]})
        self.assertEqual(obj.session_starting, {})

    def test_failed_start_releases_the_waiter_to_retry(self):
        obj, entered, release, calls = self.blocked(first_result=[-1])
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(prepare, obj, "a", "k0")
            self.assertTrue(entered.wait(5))
            b = pool.submit(prepare, obj, "b", "k0")
            release.set()
            self.assertFalse(a.result(timeout=8))
            self.assertTrue(b.result(timeout=8))
        self.assertEqual(calls, [["k0"], ["k0"]])
        self.assertEqual(obj.prepared_load_sessions, {"b": ["k0"]})
        self.assertEqual(obj.session_starting, {})

    def test_partial_start_rolls_back_successful_keys(self):
        obj = linker(lambda keys: ([0, -1], ["memory", "memory"]))
        self.assertFalse(prepare(obj, "a", "k0", "k1"))
        obj.storage.store.batch_get_session_end.assert_called_once_with(["k0"])
        self.assertEqual(obj.session_refcounts, {})
        self.assertEqual(obj.session_starting, {})

    def test_cancel_during_rpc_retires_the_committed_session(self):
        obj, entered, release, _ = self.blocked()
        entry = {"state": "preparing", "session_rid": "session", "queued_at": 0}
        obj.host_prefetch_entries["a"] = entry
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(prepare, obj, "session", "k0")
            self.assertTrue(entered.wait(5))
            obj.cancel_host_prefetch("a")
            self.assertTrue(entry["cancelled"])
            release.set()
            self.assertTrue(future.result(timeout=8))
        obj._finish_host_prefetch("a", entry, "session", "no_prefetch_needed")
        obj.storage.store.batch_get_session_end.assert_called_once_with(["k0"])
        self.assertEqual(obj.prepared_load_sessions, {})
        self.assertEqual(obj.session_starting, {})
        self.assertEqual(obj.host_prefetch_entries, {})
