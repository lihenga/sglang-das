"""Queued prefetches share a native submission, retaining request lifetimes."""

import threading
import types
import unittest
from contextlib import ExitStack
from queue import Queue

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (  # noqa: E402
    MooncakeDirectLinker,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class NativeStore:
    def __init__(self):
        self.submissions = []
        self.ended = []
        self.reading = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failed_key = None
        self.raise_read = False

    def batch_get_session_start_with_sources(self, keys):
        return [0] * len(keys), ["dfs"] * len(keys)

    def batch_get_session_prefetch(self, keys):
        self.submissions.append([list(keys)])
        return [0] * len(keys)

    def batch_get_session_prefetch_batch(self, requests):
        self.submissions.append([list(keys) for keys in requests])
        self.reading.set()
        assert self.release.wait(timeout=5)
        if self.raise_read:
            raise RuntimeError("injected read failure")
        return [
            [-1 if key == self.failed_key else 0 for key in keys] for keys in requests
        ]

    def batch_get_session_end(self, keys):
        self.ended.extend(keys)

    def batch_get_session_refresh(self, keys):
        return [0] * len(keys)


def make_linker(native):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_enabled = True
    linker.host_prefetch_limit = 32
    linker.host_prefetch_lock = threading.Lock()
    linker.host_prefetch_entries = {}
    linker.host_prefetch_queue = Queue()
    linker.host_prefetch_generation = 0
    linker.session_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lease_base = {}
    linker.session_refresh_age_s = 5
    linker._dfs_read_policy_allows_read = lambda: True
    linker.pool_group = types.SimpleNamespace(
        resolve_transfers=lambda transfers, **_: list(transfers)
    )
    linker.storage = types.SimpleNamespace(
        store=native,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: list(keys),
    )
    return linker


class TestMooncakeQueuedPrefetchBatch(CustomTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            envs.SGLANG_MOONCAKE_ENABLE_PREFETCH_BATCHING.override(True)
        )
        self.native = NativeStore()
        self.linker = make_linker(self.native)
        self.worker = None

    def tearDown(self):
        self.native.release.set()
        if self.worker is not None:
            if self.worker.is_alive():
                self.linker.host_prefetch_queue.put(None)
            self.worker.join(timeout=10)
            self.assertFalse(self.worker.is_alive())
        for rid in list(self.linker.host_prefetch_entries):
            self.linker.cancel_host_prefetch(rid)
        self.assertFalse(self.linker.prepared_load_sessions)
        self.assertFalse(self.linker.session_refcounts)

    def submit(self, rid, keys):
        self.assertTrue(
            self.linker.submit_host_prefetch(
                rid, [PoolTransfer(name=PoolName.KV, keys=list(keys))]
            )
        )

    def start_and_finish(self):
        self.linker.host_prefetch_queue.put(None)
        self.worker = threading.Thread(
            target=self.linker.host_prefetch_thread_func, daemon=True
        )
        self.worker.start()
        self.worker.join(timeout=10)
        self.assertFalse(self.worker.is_alive())
        self.assertEqual(self.linker.host_prefetch_queue.unfinished_tasks, 0)

    def test_one_worker_reads_all_queued_requests_and_releases_separately(self):
        for index in range(5):
            self.submit(str(index), [f"key-{index}"])
        self.start_and_finish()
        self.assertEqual(
            self.native.submissions, [[[f"key-{index}"] for index in range(5)]]
        )
        self.assertTrue(
            all(
                entry["state"] == "dfs_prefetched"
                for entry in self.linker.host_prefetch_entries.values()
            )
        )
        self.linker.cancel_host_prefetch("1")
        self.assertEqual(self.native.ended, ["key-1"])
        self.assertEqual(len(self.linker.prepared_load_sessions), 4)
        self.assertEqual(self.linker.get_host_prefetch_status("2"), "dfs_prefetched")

    def test_failed_request_does_not_fail_its_peer(self):
        self.native.failed_key = "bad"
        self.submit("a", ["good"])
        self.submit("b", ["bad"])
        self.start_and_finish()
        self.assertEqual(self.linker.get_host_prefetch_status("a"), "dfs_prefetched")
        self.assertEqual(self.linker.get_host_prefetch_status("b"), "failed")
        self.assertEqual(self.native.ended, ["bad"])
        self.assertEqual(self.linker.session_refcounts, {"good": 1})

    def test_cancel_during_read_keeps_session_until_batch_returns(self):
        self.native.release.clear()
        self.submit("a", ["a"])
        self.submit("b", ["b"])
        self.worker = threading.Thread(
            target=self.linker.host_prefetch_thread_func, daemon=True
        )
        self.worker.start()
        self.assertTrue(self.native.reading.wait(timeout=5))
        self.linker.cancel_host_prefetch("a")
        self.assertEqual(self.native.ended, [])
        self.assertEqual(self.linker.session_refcounts, {"a": 1, "b": 1})
        self.linker.host_prefetch_queue.put(None)
        self.native.release.set()
        self.worker.join(timeout=10)
        self.assertFalse(self.worker.is_alive())
        self.assertIsNone(self.linker.get_host_prefetch_status("a"))
        self.assertEqual(self.native.ended, ["a"])
        self.assertEqual(self.linker.get_host_prefetch_status("b"), "dfs_prefetched")

    def test_cancel_while_later_request_prepares_is_not_submitted(self):
        preparing = threading.Event()
        release_prepare = threading.Event()
        original_start = self.native.batch_get_session_start_with_sources

        def start(keys):
            if "b" in keys:
                preparing.set()
                assert release_prepare.wait(timeout=5)
            return original_start(keys)

        self.native.batch_get_session_start_with_sources = start
        self.submit("a", ["shared", "a"])
        self.submit("b", ["shared", "b"])
        self.linker.host_prefetch_queue.put(None)
        self.worker = threading.Thread(
            target=self.linker.host_prefetch_thread_func, daemon=True
        )
        self.worker.start()
        try:
            self.assertTrue(preparing.wait(timeout=5))
            self.assertEqual(self.native.submissions, [])
            self.linker.cancel_host_prefetch("a")
        finally:
            release_prepare.set()
        self.worker.join(timeout=10)
        self.assertFalse(self.worker.is_alive())
        self.assertEqual(self.native.submissions, [[["shared", "b"]]])
        self.assertEqual(self.native.ended, ["a"])
        self.assertEqual(self.linker.session_refcounts, {"shared": 1, "b": 1})
        self.assertIsNone(self.linker.get_host_prefetch_status("a"))
        self.assertEqual(self.linker.get_host_prefetch_status("b"), "dfs_prefetched")
        self.assertEqual(self.linker.host_prefetch_queue.unfinished_tasks, 0)

    def test_native_exception_cleans_all_prepared_sessions(self):
        self.native.raise_read = True
        self.submit("a", ["a"])
        self.submit("b", ["b"])
        self.start_and_finish()
        self.assertEqual(set(self.native.ended), {"a", "b"})
        self.assertFalse(self.linker.prepared_load_sessions)
        self.assertTrue(
            all(
                entry["state"] == "failed"
                for entry in self.linker.host_prefetch_entries.values()
            )
        )

    def test_old_wheel_keeps_single_request_path(self):
        self.native.batch_get_session_prefetch_batch = None
        self.submit("a", ["a"])
        self.submit("b", ["b"])
        self.start_and_finish()
        self.assertEqual(self.native.submissions, [[["a"]], [["b"]]])
        self.assertEqual(self.linker.session_refcounts, {"a": 1, "b": 1})

    def test_explicit_disable_keeps_single_request_path(self):
        with envs.SGLANG_MOONCAKE_ENABLE_PREFETCH_BATCHING.override(False):
            self.submit("a", ["a"])
            self.submit("b", ["b"])
            self.start_and_finish()
        self.assertEqual(self.native.submissions, [[["a"]], [["b"]]])
        self.assertEqual(self.linker.session_refcounts, {"a": 1, "b": 1})

    def test_unset_defaults_to_single_request_path_with_new_wheel(self):
        class SingleRequestQueue(Queue):
            def qsize(self):
                raise AssertionError("disabled worker must not inspect batch size")

            def get_nowait(self):
                raise AssertionError("disabled worker must not drain queued requests")

        self.linker.host_prefetch_queue = SingleRequestQueue()
        with envs.SGLANG_MOONCAKE_ENABLE_PREFETCH_BATCHING.override(True):
            envs.SGLANG_MOONCAKE_ENABLE_PREFETCH_BATCHING.clear()
            self.assertFalse(envs.SGLANG_MOONCAKE_ENABLE_PREFETCH_BATCHING.is_set())
            self.assertFalse(envs.SGLANG_MOONCAKE_ENABLE_PREFETCH_BATCHING.get())
            self.submit("a", ["a"])
            self.submit("b", ["b"])
            self.start_and_finish()
        self.assertEqual(self.native.submissions, [[["a"]], [["b"]]])
        self.assertEqual(self.linker.session_refcounts, {"a": 1, "b": 1})

    def test_shared_key_remains_until_both_requests_release(self):
        self.submit("a", ["shared", "a"])
        self.submit("b", ["shared", "b"])
        self.start_and_finish()
        self.assertEqual(self.linker.session_refcounts["shared"], 2)
        self.linker.cancel_host_prefetch("a")
        self.assertNotIn("shared", self.native.ended)
        self.assertEqual(self.linker.session_refcounts, {"shared": 1, "b": 1})
        self.linker.cancel_host_prefetch("b")
        self.assertEqual(self.native.ended.count("shared"), 1)

    def test_cancelled_queued_request_is_not_submitted(self):
        self.submit("a", ["a"])
        self.submit("b", ["b"])
        self.linker.cancel_host_prefetch("a")
        self.start_and_finish()
        self.assertEqual(self.native.submissions, [[["b"]]])
        self.assertEqual(self.linker.session_refcounts, {"b": 1})


if __name__ == "__main__":
    unittest.main()
