import ast
import concurrent.futures
import logging
import queue
import threading
import time
import unittest
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import AbortAck, AbortNotification
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.mooncake.notifications import DrainedAbortNotifier
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

    def test_abort_after_sender_retirement_retries_when_notifier_capacity_returns(self):
        """A single late ABORT gets its ACK without a sender or chunk callback."""
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._stopped = False
        manager.deferred_bootstrap = None
        manager.enable_deferred_decode_kv_release = True
        manager.bootstrap_timeout = manager.waiting_timeout = 10
        manager.request_status = {}
        manager._staging_outstanding = {}
        manager._source_drain = SourceDrainTracker()
        manager._deferred_ack_targets = {}
        manager._deferred_ack_deadlines = OrderedDict()
        manager._drained_abort_capacity = 4096
        manager._control_notification_lock = threading.RLock()
        manager._pending_control_notifications = OrderedDict()
        manager._control_rooms = OrderedDict()
        manager._deferred_ack_poisoned_rooms = set()
        manager._prefill_unique_rank = lambda: 0
        entered, release, delivered = (threading.Event() for _ in range(3))
        expected = tuple(AbortAck(9, 0, 17).to_zmq())
        wire = []

        class Socket:
            def setsockopt(self, *args):
                pass

            def connect(self, endpoint):
                pass

            def close(self, **kwargs):
                pass

            def send_multipart(self, parts, flags):
                message = tuple(parts)
                wire.append(message)
                if message == (b"blocker",):
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release the full queue")
                if message == expected:
                    delivered.set()

        with patch(
            "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
            return_value=SimpleNamespace(socket=lambda *_: Socket()),
        ):
            notifier = DrainedAbortNotifier(
                capacity=1, retry_pending=manager._retry_pending_control_notifications
            )
            manager._drained_abort_notifier = notifier
            try:
                self.assertTrue(
                    notifier.submit(
                        1, "tcp://busy:1", False, [b"blocker"], float("inf")
                    )
                )
                self.assertTrue(entered.wait(5))
                self.assertTrue(
                    manager._handle_abort_notification(
                        AbortNotification(9, "peer", 8000, 17).to_zmq()
                    )
                )
                self.assertNotIn(9, manager._deferred_ack_targets)
                self.assertEqual(len(manager._pending_control_notifications), 1)
                self.assertEqual(len(notifier._pending), 1)
                release.set()
                self.assertTrue(delivered.wait(5))
                self.assertNotIn(9, manager._deferred_ack_targets)
            finally:
                release.set()
                notifier.close()
        self.assertEqual(wire, [(b"blocker",), expected])
        self.assertEqual(manager.request_status, {})

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
        namespace = {"concurrent": concurrent, "logger": logging.getLogger(__name__)}
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


class TestMooncakeLateBootstrap(CustomTestCase):
    """A drained abort stays terminal when a delayed destination arrives."""

    def _make_room(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._drained_abort_lock = threading.Lock()
        manager._drained_abort_rooms = OrderedDict()
        manager._drained_abort_capacity = 4096
        manager.bootstrap_timeout, manager.waiting_timeout = 30, 600
        manager.request_status = {9: KVPoll.Failed}
        manager.transfer_infos = {}
        manager.req_to_decode_prefix_len = {}
        manager.req_to_pd_hidden_meta = {}
        manager.failure_lock = threading.Lock()
        manager.failure_records = {9: "original failure"}
        manager.deferred_bootstrap = None
        manager._deferred_ack_targets = {}
        manager._deferred_ack_deadlines = OrderedDict()
        manager._control_notification_lock = threading.RLock()
        manager._pending_control_notifications = OrderedDict()
        manager._control_rooms = OrderedDict()
        manager._deferred_ack_poisoned_rooms = set()
        manager._staging_outstanding = {}
        tracker = SourceDrainTracker()
        manager._source_drain = tracker
        manager.source_transfers_drained = tracker.drained
        manager.resolve_kv_replica_factor = lambda infos: None
        notifications = []

        def send(**message):
            # A slow PUSH must not hold the bootstrap/cleanup lock.
            self.assertTrue(manager._drained_abort_lock.acquire(blocking=False))
            manager._drained_abort_lock.release()
            notifications.append(message)

        manager.send_kv_status_message = send

        def queue_notification(room, endpoint, port):
            # This boundary records accepted ownership; delivery runs elsewhere.
            send(
                targets=[(endpoint, port)],
                bootstrap_room=room,
                status=KVPoll.Failed,
                failure_reason="Prefill request aborted after source drain",
            )
            return True

        manager.queue_drained_abort_notification = queue_notification
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room = manager, 9
        sender.conclude_state = KVPoll.Failed
        sender._drained_abort_targets = set()
        sender._drained_abort_notified_targets = set()
        manager._reserve_control_room(9, active=True)
        return manager, sender, tracker, notifications

    @staticmethod
    def _info(endpoint="decode"):
        return SimpleNamespace(
            endpoint=endpoint,
            dst_port=8000,
            is_dummy=False,
            decode_prefix_len=4,
            spec_metadata=None,
        )

    def test_late_destination_does_not_reopen_cleared_abort(self):
        manager, sender, _, notifications = self._make_room()
        manager.transfer_infos[9] = {"first": self._info()}
        sender.notify_drained_abort()
        self.assertNotIn(9, manager.request_status)
        self.assertNotIn(9, manager.transfer_infos)
        self.assertNotIn(9, manager.failure_records)
        manager._register_bootstrap_transfer(9, "late", self._info("late"), 1)
        self.assertNotIn(9, manager.request_status)
        self.assertNotIn(9, manager.transfer_infos)
        self.assertEqual(
            [item["targets"] for item in notifications],
            [[("decode", 8000)], [("late", 8000)]],
        )
        self.assertTrue(all(item["status"] == KVPoll.Failed for item in notifications))

    def test_reading_failure_cause_keeps_targets_for_the_drained_notification(self):
        manager, sender, _, notifications = self._make_room()
        manager.transfer_infos[9] = {"session": self._info()}
        with self.assertRaisesRegex(Exception, "original failure"):
            sender.failure_exception()
        self.assertNotIn(9, manager.transfer_infos)
        sender.notify_drained_abort()
        self.assertEqual(notifications[0]["targets"], [("decode", 8000)])

    def test_abort_waits_for_source_then_includes_late_target(self):
        manager, sender, tracker, notifications = self._make_room()
        chunk = SimpleNamespace(room=9)
        tracker.publish(chunk, queue.Queue(), lambda: True)
        manager._register_bootstrap_transfer(9, "late", self._info("late"), 1)
        sender.notify_drained_abort()
        self.assertEqual(notifications, [])
        self.assertEqual(manager.request_status[9], KVPoll.Failed)
        tracker.complete(chunk)
        sender.notify_drained_abort()
        self.assertNotIn(9, manager.request_status)
        self.assertEqual(notifications[0]["targets"], [("late", 8000)])
        self.assertEqual(notifications[0]["status"], KVPoll.Failed)

    def test_terminal_capacity_backpressures_without_expiring_live_records(self):
        manager, _, _, _ = self._make_room()
        now = time.monotonic()
        manager._drained_abort_rooms.update((room, now + 600) for room in range(4096))
        manager._prune_drained_abort_rooms()
        self.assertEqual(len(manager._drained_abort_rooms), 4096)
        self.assertIsNone(manager.finish_drained_abort(4096))
        self.assertEqual(len(manager._drained_abort_rooms), 4096)
        self.assertIn(0, manager._drained_abort_rooms)
        self.assertNotIn(4096, manager._drained_abort_rooms)
        self.assertEqual(manager._drained_abort_rooms[0], now + 600)
        self.assertEqual(manager.finish_drained_abort(0), set())
        self.assertEqual(manager._drained_abort_rooms[0], now + 600)
        manager._drained_abort_rooms = OrderedDict([(9, now - 1)])
        manager._register_bootstrap_transfer(9, "new", self._info(), 1)
        self.assertEqual(manager.request_status[9], KVPoll.Failed)
        # Reopening is allowed after expiry only once the old status is cleared.
        manager.request_status.pop(9)
        manager._register_bootstrap_transfer(9, "new", self._info(), 1)
        self.assertEqual(manager.request_status[9], KVPoll.WaitingForInput)

    def test_hidden_transfer_failure_uses_target_status_transport(self):
        manager, _, _, notifications = self._make_room()
        manager.session_lock = threading.Lock()
        manager.session_failures = defaultdict(int)
        manager.failed_sessions = set()
        manager.record_failure = lambda room, reason: manager.failure_records.update(
            {room: reason}
        )
        path = (
            Path(__file__).parents[4]
            / "python/sglang/srt/disaggregation/hidden_events.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "mark_session_failed_and_sync"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {"KVPoll": KVPoll, "logger": logging.getLogger(__name__)}
        exec(compile(module, str(path), "exec"), namespace)
        events = SimpleNamespace(owner=manager, wake_ack_waiters=lambda room: None)
        req = self._info()
        req.room, req.mooncake_session_id = 9, "session"
        namespace[method.name](
            events,
            kv_chunk=SimpleNamespace(room=9),
            req=req,
            prefill_unique_rank=0,
            failure_reason="hidden write failed",
        )
        self.assertEqual(manager.request_status[9], KVPoll.Failed)
        self.assertEqual(manager.failed_sessions, {"session"})
        self.assertEqual(manager.failure_records[9], "hidden write failed")
        self.assertEqual(notifications[0]["failure_reason"], "hidden write failed")


if __name__ == "__main__":
    unittest.main()
