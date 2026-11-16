import ast
import concurrent.futures
import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.disaggregation.mooncake.source_drain import SourceDrainTracker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeSourceDrain(CustomTestCase):
    def test_registers_ownership_before_a_queue_can_consume_the_chunk(self):
        tracker = SourceDrainTracker()
        chunk = SimpleNamespace(room=9)
        observations = []

        class ImmediateQueue:
            def put(self, value):
                observations.append(tracker.drained(value.room))

        tracker.publish(chunk, ImmediateQueue(), lambda: True)
        self.assertEqual(observations, [False])
        self.assertFalse(tracker.drained(9))
        tracker.complete(chunk)
        self.assertTrue(tracker.drained(9))

    def test_queued_and_dequeued_chunks_both_block_reclamation(self):
        tracker = SourceDrainTracker()
        work = queue.Queue()
        first, second = SimpleNamespace(room=9), SimpleNamespace(room=9)
        tracker.publish(first, work, lambda: True)
        tracker.publish(second, work, lambda: True)
        self.assertIs(work.get_nowait(), first)
        tracker.complete(first)
        self.assertFalse(tracker.drained(9))
        self.assertIs(work.get_nowait(), second)
        self.assertFalse(tracker.drained(9))
        tracker.complete(second)
        self.assertTrue(tracker.drained(9))

    def test_unknown_publication_outcome_never_proves_drain(self):
        tracker = SourceDrainTracker()
        chunk = SimpleNamespace(room=9)

        class AmbiguousQueue:
            def put(self, value):
                raise RuntimeError("queue may already have published the pointer")

        with self.assertRaises(RuntimeError):
            tracker.publish(chunk, AmbiguousQueue(), lambda: True)
        tracker.complete(chunk)
        self.assertFalse(tracker.drained(9))

    def test_closed_publication_does_not_create_a_pending_owner(self):
        tracker = SourceDrainTracker()
        work = queue.Queue()
        tracker.publish(SimpleNamespace(room=9), work, lambda: False)
        self.assertTrue(work.empty())
        self.assertTrue(tracker.drained(9))

    def test_failed_write_waits_for_running_peers_without_deferred_decode_release(self):
        path = (
            Path(__file__).parents[4]
            / "python/sglang/srt/disaggregation/mooncake/conn.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_await_transfer_futures"
        )
        namespace = {"concurrent": concurrent}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        started, release, returned = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def running_write():
            started.set()
            release.wait(timeout=10)
            return 0

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(running_write)
            self.assertTrue(started.wait(timeout=5))
            failed = executor.submit(lambda: -1)
            owner = SimpleNamespace(enable_deferred_decode_kv_release=False)

            def await_chunk():
                results.append(namespace[method.name](owner, [failed, running]))
                returned.set()

            waiter = threading.Thread(target=await_chunk, daemon=True)
            waiter.start()
            try:
                self.assertFalse(returned.wait(timeout=0.1))
            finally:
                release.set()
            waiter.join(timeout=5)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(results, [-1])


if __name__ == "__main__":
    unittest.main()
