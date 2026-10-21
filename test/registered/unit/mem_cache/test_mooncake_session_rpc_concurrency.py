"""T01: a slow session RPC must not hold the scheduler's shared lock."""

import threading
import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from test_load_back_construction_consensus import (
    FakeComponent,
    ReachedPrepare,
    _wrapper,
)
from test_mooncake_session_refresh_age import (
    _Clock,
    _linker,
    _prefetched,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSessionRpcConcurrency(CustomTestCase):
    def _prepare(self, linker, rid, keys):
        return linker._prepare_expanded_load(rid, [types.SimpleNamespace(keys=keys)])

    def test_disjoint_starts_overlap_and_admission_stays_responsive(self):
        linker = _linker()
        _prefetched(linker, _Clock())
        release, both, done = threading.Event(), threading.Event(), threading.Event()
        entered = []
        guard = threading.Lock()

        def start(keys):
            with guard:
                entered.append(keys)
                if len(entered) == 2:
                    both.set()
            if not release.wait(5):
                raise TimeoutError("blocked session start was not released")
            return [0] * len(keys), ["dfs"] * len(keys)

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        results = {}
        workers = [
            threading.Thread(
                target=lambda k=k: results.update({k: self._prepare(linker, k, [k])})
            )
            for k in ("a", "b")
        ]

        def admit():
            results["valid"] = linker.revalidate_host_prefetch("rid")
            results["claimed"] = linker.claim_ready_host_prefetch("rid")
            done.set()

        admission = threading.Thread(target=admit)
        try:
            for worker in workers:
                worker.start()
            overlapped = both.wait(1)
            admission.start()
            responsive = done.wait(1)
        finally:
            release.set()
            for thread in workers + [admission]:
                thread.join(5)
        self.assertTrue(overlapped, "unrelated session RPCs were serialized")
        self.assertTrue(responsive, "admission waited for an unrelated session RPC")
        self.assertEqual(
            results, {"a": True, "b": True, "valid": True, "claimed": True}
        )
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_same_key_start_is_shared_and_ended_once(self):
        linker = _linker()
        entered, release = threading.Event(), threading.Event()

        def start(keys):
            entered.set()
            self.assertTrue(release.wait(5))
            return [0], ["dfs"]

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        results = []
        workers = [
            threading.Thread(
                target=lambda rid=rid: results.append(
                    self._prepare(linker, rid, ["shared"])
                )
            )
            for rid in ("a", "b")
        ]
        try:
            workers[0].start()
            self.assertTrue(entered.wait(1))
            workers[1].start()
        finally:
            release.set()
            for worker in workers:
                worker.join(5)
        self.assertEqual(results, [True, True])
        self.assertEqual(linker.session_refcounts, {"shared": 2})
        linker.storage.store.batch_get_session_start_with_sources.assert_called_once()
        linker._abort_prepared_load_now("a")
        linker.storage.store.batch_get_session_end.assert_not_called()
        linker._abort_prepared_load_now("b")
        linker.storage.store.batch_get_session_end.assert_called_once_with(["shared"])

    def test_partial_start_failure_releases_successes_and_wakes_waiter(self):
        linker = _linker()
        entered, release = threading.Event(), threading.Event()

        def start(keys):
            if len(keys) == 2:
                entered.set()
                self.assertTrue(release.wait(5))
                return [0, 7], ["dfs", "dfs"]
            return [0], ["dfs"]

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        results = {}
        a = threading.Thread(
            target=lambda: results.update(
                a=self._prepare(linker, "a", ["shared", "missing"])
            )
        )
        b = threading.Thread(
            target=lambda: results.update(b=self._prepare(linker, "b", ["shared"]))
        )
        try:
            a.start()
            self.assertTrue(entered.wait(1))
            b.start()
        finally:
            release.set()
            a.join(5)
            b.join(5)
        self.assertEqual(results, {"a": False, "b": True})
        self.assertEqual(linker.prepared_load_sessions, {"b": ["shared"]})
        self.assertEqual(linker.session_refcounts, {"shared": 1})
        linker.storage.store.batch_get_session_end.assert_called_once_with(["shared"])
        linker._abort_prepared_load_now("b")
        self.assertEqual(linker.session_refcounts, {})

    def test_end_blocks_only_same_key_restart(self):
        linker = _linker()
        self.assertTrue(self._prepare(linker, "old", ["shared"]))
        entered, release, unrelated = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        restarted = threading.Event()

        def end(keys):
            entered.set()
            self.assertTrue(release.wait(5))

        def start(keys):
            if keys == ["shared"]:
                restarted.set()
            return [0], ["dfs"]

        linker.storage.store.batch_get_session_end.side_effect = end
        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        cleanup = threading.Thread(
            target=linker._abort_prepared_load_now, args=("old",)
        )
        replacement = threading.Thread(
            target=self._prepare, args=(linker, "new", ["shared"])
        )
        other = threading.Thread(
            target=lambda: (self._prepare(linker, "other", ["other"]), unrelated.set())
        )
        try:
            cleanup.start()
            self.assertTrue(entered.wait(1))
            replacement.start()
            other.start()
            responsive = unrelated.wait(1)
            premature_restart = restarted.is_set()
        finally:
            release.set()
            for thread in (cleanup, replacement, other):
                thread.join(5)
        self.assertTrue(responsive, "session end held the global session lock")
        self.assertFalse(
            premature_restart, "old cleanup could end the replacement session"
        )
        self.assertTrue(restarted.is_set())
        self.assertEqual(linker.session_refcounts, {"shared": 1, "other": 1})

    def test_start_completion_cannot_release_a_concurrent_end_reservation(self):
        linker = _linker()
        published, cleanup_queued, finish_end = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        replacement_started = threading.Event()
        original_end = linker._end_session_keys
        errors = []
        starts = []

        def start(keys):
            starts.append(keys)
            if len(starts) == 2:
                replacement_started.set()
            return [0], ["dfs"]

        def delayed_end(keys):
            if keys:
                cleanup_queued.set()
                self.assertTrue(finish_end.wait(5))
            else:
                published.set()
                self.assertTrue(cleanup_queued.wait(5))
            original_end(keys)

        def run(fn, *args):
            try:
                fn(*args)
            except BaseException as error:
                errors.append(error)

        linker.storage.store.batch_get_session_start_with_sources.side_effect = start
        linker._end_session_keys = delayed_end
        first = threading.Thread(
            target=run, args=(self._prepare, linker, "old", ["shared"])
        )
        cleanup = threading.Thread(
            target=run, args=(linker._abort_prepared_load_now, "old")
        )
        replacement = threading.Thread(
            target=run, args=(self._prepare, linker, "new", ["shared"])
        )
        try:
            first.start()
            self.assertTrue(published.wait(1))
            cleanup.start()
            self.assertTrue(cleanup_queued.wait(1))
            first.join(1)  # Complete the original start while end is still pending.
            self.assertFalse(first.is_alive())
            replacement.start()
            premature = replacement_started.wait(0.2)
        finally:
            finish_end.set()
            for thread in (first, cleanup, replacement):
                if thread.ident is not None:
                    thread.join(5)
        self.assertFalse(
            premature, "start completion unblocked a key still being ended"
        )
        self.assertEqual(errors, [])
        self.assertTrue(replacement_started.is_set())
        self.assertEqual(linker.session_refcounts, {"shared": 1})

    def test_scheduler_defers_ordinary_and_memory_only_session_start(self):
        for prefetched in (False, True):
            with self.subTest(prefetched=prefetched):
                component = FakeComponent(PoolName.KV)
                cache = types.SimpleNamespace(
                    page_size=1,
                    tree_core=types.SimpleNamespace(
                        empty_match_result=types.SimpleNamespace(
                            device_indices=torch.empty(0, dtype=torch.int64)
                        )
                    ),
                    _components_tuple=(component,),
                    _all_reduce_attn_groups=lambda *args: None,
                )
                backend = MagicMock()
                backend.get_host_prefetch_status.return_value = "no_prefetch_needed"
                backend.revalidate_no_prefetch_needed.return_value = True
                backend.revalidate_load.return_value = True
                backend.prepare_load.side_effect = AssertionError(
                    "scheduler started a session RPC"
                )
                wrapper = _wrapper(cache, backend, prefetched)
                with self.assertRaises(ReachedPrepare):
                    wrapper.load_back(types.SimpleNamespace(rid="r", last_node="node"))
                backend.prepare_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
