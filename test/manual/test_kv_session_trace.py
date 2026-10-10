"""CPU-only tests; runnable directly without importing torch or the server."""

import importlib.util
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

MODULE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/mem_cache/unified_cache/kv_session_trace.py"
)
spec = importlib.util.spec_from_file_location("kv_session_trace", MODULE)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class TestKVSessionTrace(unittest.TestCase):
    def setUp(self):
        self.enabled = patch.object(trace, "ENABLED", True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)
        self.native = threading.local()

        def setter(context):
            previous = getattr(self.native, "context", "original")
            self.native.context = context
            return previous

        self.setter = Mock(side_effect=setter)
        self.owner = SimpleNamespace(
            tp_rank=3,
            storage=SimpleNamespace(
                store=SimpleNamespace(set_session_trace_context=self.setter)
            ),
        )
        self.logs = patch.object(trace._logger, "info")
        self.logger = self.logs.start()
        self.addCleanup(self.logs.stop)

    def records(self):
        return [json.loads(call.args[1]) for call in self.logger.call_args_list]

    def test_nested_native_context_restored_after_failure(self):
        with trace.scope(self.owner, "outer", "rid-a"):
            outer = self.native.context
            with self.assertRaisesRegex(RuntimeError, "original error"):
                with trace.scope(self.owner, "inner", "rid-a"):
                    self.assertNotEqual(outer, self.native.context)
                    raise RuntimeError("original error")
            self.assertEqual(self.native.context, outer)
        self.assertEqual(self.native.context, "original")
        records = self.records()
        self.assertEqual(records[1]["parent_trace"], records[0]["trace_id"])
        self.assertEqual(records[2]["exception_type"], "RuntimeError")
        self.assertEqual(records[0]["rid"], "rid-a")

    def test_disabled_never_calls_native_bridge_or_formats_keys(self):
        with (
            patch.object(trace, "ENABLED", False),
            patch.object(
                trace, "key_hash", side_effect=AssertionError("unexpected hashing")
            ),
        ):
            with trace.scope(self.owner, "disabled"):
                trace.keys(self.owner, "keys", "rid", ["secret"])
            wrapped = trace.traced("disabled", rid_arg=0)(lambda _, rid: rid)
            self.assertEqual(wrapped(self.owner, "rid"), "rid")
        self.setter.assert_not_called()
        self.logger.assert_not_called()

    def test_thread_context_does_not_cross_workers(self):
        barrier = threading.Barrier(2)
        observed = []

        def worker(rid):
            with trace.scope(self.owner, "worker", rid):
                context = self.native.context
                barrier.wait(timeout=5)
                observed.append((context, self.native.context))
            observed.append(("original", self.native.context))

        threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(observed), 4)
        self.assertTrue(all(before == after for before, after in observed))
        contexts = [
            record["trace_id"]
            for record in self.records()
            if record["event"] == "worker.begin"
        ]
        self.assertEqual(len(set(contexts)), 2)

    def test_native_missing_is_explicit_and_does_not_block_call(self):
        owner = SimpleNamespace(storage=SimpleNamespace(store=object()))
        with trace.scope(owner, "old_wheel", "rid"):
            pass
        self.assertFalse(self.records()[0]["native_bridge"])

    def test_wrapper_preserves_return_object_and_batch_membership(self):
        expected = object()
        operation = Mock(return_value=expected)
        wrapped = trace.traced("load", batch_arg=1)(operation)
        batch = [("rid-a", []), ("rid-b", [])]
        self.assertIs(wrapped(self.owner, 17, batch), expected)
        operation.assert_called_once_with(self.owner, 17, batch)
        self.assertEqual(self.records()[0]["rids"], ["rid-a", "rid-b"])
        self.assertEqual(self.records()[0]["load_batch"], 17)

    def test_full_hash_membership_is_chunked_and_has_no_raw_keys(self):
        keys = [f"private-key-{i}" for i in range(257)]
        self.owner.session_refcounts = dict.fromkeys(keys, 2)
        trace.keys(self.owner, "acquire", "rid", keys)
        records = self.records()
        self.assertEqual([item["offset"] for item in records], [0, 128, 256])
        self.assertEqual(sum(len(item["hash_refs"]) for item in records), 257)
        self.assertNotIn("private-key", str(records))
        self.assertEqual(records[0]["hash_refs"][0], [trace.key_hash(keys[0]), 2])

    def test_fnv_matches_published_64_bit_vectors(self):
        self.assertEqual(trace.key_hash(""), 14695981039346656037)
        self.assertEqual(trace.key_hash("hello"), 0xA430D84680AABD0B)


if __name__ == "__main__":
    unittest.main()
