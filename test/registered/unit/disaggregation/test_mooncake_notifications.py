"""Drained notification saturation cannot block scheduling or revive rooms."""

import queue
import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.srt.disaggregation.common.conn import AbortAck, AckTarget, KVTransferError
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.mooncake.notifications import DrainedAbortNotifier
from sglang.srt.disaggregation.mooncake.source_drain import SourceDrainTracker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDrainedNotifications(CustomTestCase):
    def _make_sender(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._drained_abort_lock = threading.Lock()
        manager._drained_abort_rooms = OrderedDict()
        manager._drained_abort_capacity = 4096
        manager._source_drain = SourceDrainTracker()
        manager._stopped = False
        manager._staging_outstanding = {}
        manager._deferred_ack_targets = {}
        manager._deferred_ack_poisoned_rooms = set()
        manager.bootstrap_timeout = manager.waiting_timeout = 10
        manager._prefill_unique_rank = lambda: 0
        manager._wake_pd_hidden_ack_waiters = lambda _: None
        manager.request_status = {7: KVPoll.Failed}
        manager.transfer_infos = {
            7: {"session": SimpleNamespace(endpoint="peer", dst_port=1, is_dummy=False)}
        }
        manager.req_to_decode_prefix_len = {7: 10}
        manager.req_to_pd_hidden_meta = {7: {"pd_hidden": True}}
        manager.deferred_bootstrap = None
        manager.failure_lock = threading.Lock()
        manager.failure_records = {7: "failure"}
        manager._drained_abort_notifier = SimpleNamespace(submit=lambda *args: True)
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room = manager, 7
        sender.conclude_state = None
        sender._owns_bootstrap_room = True
        sender._drained_abort_targets = set()
        sender._drained_abort_notified_targets = set()
        return manager, sender

    def test_blocked_worker_keeps_capacity_and_retries_without_blocking_submitter(self):
        entered, release, delivered = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        attempts = []

        class Socket:
            def setsockopt(self, *args):
                pass

            def connect(self, endpoint):
                pass

            def close(self, **kwargs):
                pass

            def send_multipart(self, parts, flags):
                attempts.append(tuple(parts))
                if len(attempts) == 1:
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release worker")
                    raise zmq.Again()
                delivered.set()

        with patch(
            "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
            return_value=SimpleNamespace(socket=lambda *_: Socket()),
        ):
            notifier = DrainedAbortNotifier(capacity=1)
            try:
                self.assertTrue(
                    notifier.submit(1, "tcp://peer:1", False, [b"failed"], float("inf"))
                )
                self.assertTrue(entered.wait(5))
                self.assertFalse(
                    notifier.submit(2, "tcp://peer:2", False, [b"failed"], float("inf"))
                )
                self.assertTrue(
                    notifier.submit(1, "tcp://peer:1", False, [b"failed"], float("inf"))
                )
                release.set()
                self.assertTrue(delivered.wait(5))
            finally:
                release.set()
                notifier.close()
        self.assertEqual(attempts, [(b"failed",), (b"failed",)])
        self.assertFalse(notifier.submit(3, "tcp://peer:3", False, [], float("inf")))

    def test_terminal_capacity_never_evicts_unexpired_room(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._drained_abort_lock = threading.Lock()
        manager._drained_abort_capacity = 1
        manager._drained_abort_rooms = OrderedDict()
        manager.transfer_infos = {}
        manager.bootstrap_timeout = manager.waiting_timeout = 10
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=100
        ):
            self.assertEqual(manager.finish_drained_abort(1), set())
            self.assertIsNone(manager.finish_drained_abort(2))
            self.assertEqual(list(manager._drained_abort_rooms), [1])
            self.assertEqual(manager.finish_drained_abort(1), set())
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=401
        ):
            self.assertEqual(manager.finish_drained_abort(2), set())
            self.assertEqual(list(manager._drained_abort_rooms), [2])

    def test_prefill_readiness_retries_after_full_queue_without_registration(self):
        """A full queue cannot consume readiness before the worker owns delivery."""
        manager, sender = self._make_sender()
        bootstrap = manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
        bootstrap.open(7, sender)
        bootstrap.register(7, ("peer", 1))
        manager.request_status[7] = KVPoll.Bootstrapping
        sender._prefill_complete_time = None
        entered, release, delivered = (threading.Event() for _ in range(3))
        wire = []
        expected = tuple(
            manager._encode_kv_status_message(
                bootstrap_room=7, status=KVPoll.WaitingForInput, failure_reason=None
            )
        )

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

        with (
            patch(
                "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
                return_value=SimpleNamespace(socket=lambda *_: Socket()),
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.notifications.logger.warning"
            ) as warning,
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
                sender.mark_prefill_complete()
                self.assertTrue(bootstrap.rooms[7].ready)
                self.assertFalse(bootstrap.rooms[7].ready_notified)
                self.assertEqual(len(notifier._pending), 1)
                queue_bootstrap = manager._queue_bootstrap_notification
                encoding_failures = []

                def retry_bootstrap(room, notification):
                    if not encoding_failures:
                        encoding_failures.append(True)
                        raise ValueError("injected producer encoding failure")
                    return queue_bootstrap(room, notification)

                manager._queue_bootstrap_notification = retry_bootstrap
                release.set()
                self.assertTrue(delivered.wait(5))
                self.assertTrue(bootstrap.rooms[7].ready_notified)
            finally:
                release.set()
                notifier.close()
            self.assertEqual(warning.call_args.args[1:], (1, 0))
        self.assertEqual(wire, [(b"blocker",), expected])
        self.assertIsNone(notifier._retry_pending)
        self.assertEqual(notifier._pending, {})

    def test_worker_preserves_status_and_distinct_ack_generations(self):
        """Sharing an endpoint cannot merge a Failed frame with either ACK."""
        entered, release, delivered = (threading.Event() for _ in range(3))
        wire = []

        class Socket:
            def setsockopt(self, *args):
                pass

            def connect(self, endpoint):
                pass

            def close(self, **kwargs):
                pass

            def send_multipart(self, parts, flags):
                wire.append(tuple(parts))
                if len(wire) == 1:
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release the worker")
                if len(wire) == 3:
                    delivered.set()

        messages = [
            [b"7", b"0", b"0"],
            AbortAck(7, 0, 9).to_zmq(),
            AbortAck(7, 0, 10).to_zmq(),
        ]
        with patch(
            "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
            return_value=SimpleNamespace(socket=lambda *_: Socket()),
        ):
            notifier = DrainedAbortNotifier(capacity=3)
            try:
                self.assertTrue(
                    notifier.submit(7, "tcp://peer:1", False, messages[0], float("inf"))
                )
                self.assertTrue(entered.wait(5))
                for parts in messages[1:]:
                    self.assertTrue(
                        notifier.submit(7, "tcp://peer:1", False, parts, float("inf"))
                    )
                self.assertFalse(
                    notifier.submit(8, "tcp://peer:1", False, [], float("inf"))
                )
                release.set()
                self.assertTrue(delivered.wait(5))
            finally:
                release.set()
                notifier.close()
        self.assertEqual(wire, [tuple(parts) for parts in messages])

    def test_endpoint_backoff_survives_new_work_and_aggregates_expiry_logs(self):
        """New submissions cannot hammer a dead endpoint or delay a healthy one."""
        notifier = DrainedAbortNotifier.__new__(DrainedAbortNotifier)
        notifier.capacity = 32
        notifier._pending = OrderedDict()
        notifier._closed = False
        notifier._retry_pending = None
        now, attempts, delivered = [0.0], [], []

        class Condition:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def notify(self):
                pass

            def wait_for(self, predicate):
                assert predicate()

            def wait(self, timeout):
                if not notifier._pending:
                    notifier._closed = True
                elif now[0] == 0:
                    now[0] = 0.01
                    notifier.submit(10, "dead", False, [b"new"], 0.4)
                    notifier.submit(20, "healthy", False, [b"healthy"], 0.4)
                elif now[0] == 0.01:
                    now[0] = 0.02
                    notifier.submit(11, "dead", False, [b"newer"], 0.4)
                else:
                    now[0] += timeout

        class Socket:
            def setsockopt(self, *args):
                pass

            def connect(self, endpoint):
                self.endpoint = endpoint

            def close(self, **kwargs):
                pass

            def send_multipart(self, parts, flags):
                if self.endpoint == "dead":
                    attempts.append(now[0])
                    raise RuntimeError("peer unavailable")
                delivered.append((now[0], tuple(parts)))

        notifier._condition = Condition()
        for room in range(10):
            notifier.submit(room, "dead", False, [str(room).encode()], 0.4)
        with (
            patch(
                "sglang.srt.disaggregation.mooncake.notifications.time.monotonic",
                side_effect=lambda: now[0],
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
                return_value=SimpleNamespace(socket=lambda *_: Socket()),
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.notifications.logger.warning"
            ) as warning,
        ):
            notifier._run()
        self.assertEqual(len(attempts), 4)
        for actual, expected in zip(attempts, [0.0, 0.05, 0.15, 0.35]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(delivered, [(0.01, (b"healthy",))])
        self.assertEqual(notifier._pending, {})
        self.assertEqual(len(warning.call_args_list), 1)
        self.assertEqual(warning.call_args.args[1:], (4, 12))

    def test_queue_refusal_retains_sender_until_notification_is_owned(self):
        """Queue refusal clears transport maps but keeps a terminal retry owner."""
        manager, sender = self._make_sender()
        accepted = []
        manager._drained_abort_notifier.submit = lambda *args: bool(accepted)
        self.assertFalse(sender.notify_drained_abort())
        self.assertEqual(manager.failure_records, {7: "failure"})
        self.assertEqual(sender._drained_abort_notified_targets, set())
        self.assertEqual(sender._drained_abort_targets, {("peer", 1)})
        self.assertNotIn(7, manager.transfer_infos)
        self.assertNotIn(7, manager.req_to_decode_prefix_len)
        self.assertNotIn(7, manager.req_to_pd_hidden_meta)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        manager.request_status.pop(7)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        manager.request_status[7] = KVPoll.Failed
        manager._register_bootstrap_transfer(
            7, "late", SimpleNamespace(endpoint="late", dst_port=2, is_dummy=False), 1
        )
        accepted.append(True)
        self.assertTrue(sender.notify_drained_abort())
        self.assertEqual(manager.failure_records, {})
        self.assertEqual(
            sender._drained_abort_notified_targets, {("peer", 1), ("late", 2)}
        )
        self.assertNotIn(7, manager.request_status)

    def test_terminal_refusal_clears_maps_and_later_completes(self):
        """A full terminal table must not strand the sender's next cleanup poll."""
        manager, sender = self._make_sender()
        manager._drained_abort_capacity = 1
        manager._drained_abort_rooms[1] = float("inf")
        self.assertFalse(sender.notify_drained_abort())
        self.assertNotIn(7, manager.transfer_infos)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        info = SimpleNamespace(endpoint="late", dst_port=2, is_dummy=False)
        manager._register_bootstrap_transfer(7, "late", info, 1)
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        manager._drained_abort_rooms.clear()
        self.assertTrue(sender.notify_drained_abort())
        self.assertEqual(
            sender._drained_abort_notified_targets, {("peer", 1), ("late", 2)}
        )
        self.assertNotIn(7, manager.request_status)

    def test_expired_marker_during_backpressure_preserves_late_targets(self):
        """TTL expiry while delivery is refused cannot revive or lose a peer."""
        manager, sender = self._make_sender()
        manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
        manager.deferred_bootstrap.open(7, sender)
        accepted = []
        manager._drained_abort_notifier.submit = lambda *args: bool(accepted)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=100
        ):
            self.assertFalse(sender.notify_drained_abort())
        self.assertFalse(sender._owns_bootstrap_room)
        self.assertNotIn(7, manager.transfer_infos)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=401
        ):
            info = SimpleNamespace(endpoint="late", dst_port=2, is_dummy=False)
            manager._register_bootstrap_transfer(7, "late", info, 1)
            self.assertEqual(manager.request_status[7], KVPoll.Failed)
            accepted.append(True)
            self.assertTrue(sender.notify_drained_abort())
            self.assertEqual(manager._drained_abort_rooms[7], 701)
        self.assertEqual(
            sender._drained_abort_notified_targets, {("peer", 1), ("late", 2)}
        )

    def test_bootstrap_endpoint_arriving_after_close_survives_queue_refusal(self):
        """Deferred close must retain a late bootstrap endpoint for its retry owner."""
        manager, sender = self._make_sender()
        bootstrap = manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
        bootstrap.open(7, sender)
        accepted = []
        manager._drained_abort_notifier.submit = lambda *args: bool(accepted)
        self.assertFalse(sender.notify_drained_abort())
        self.assertIsNone(bootstrap.rooms[7].owner)
        manager._notify_bootstrap(7, bootstrap.register(7, ("bootstrap-late", 3)))
        accepted.append(True)
        self.assertTrue(sender.notify_drained_abort())
        self.assertEqual(
            sender._drained_abort_notified_targets, {("peer", 1), ("bootstrap-late", 3)}
        )

    def test_bootstrap_registration_during_handoff_keeps_cleanup_owner(self):
        """A late endpoint refused during handoff must participate in its result."""
        manager, sender = self._make_sender()
        bootstrap = manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
        bootstrap.open(7, sender)
        injected, accept_late = [], []

        def submit(room, endpoint, ipv6, parts, deadline):
            if endpoint == "tcp://bootstrap-late:3":
                return bool(accept_late)
            if not injected:
                injected.append(True)
                manager._notify_bootstrap(
                    7, bootstrap.register(7, ("bootstrap-late", 3))
                )
            return True

        manager._drained_abort_notifier.submit = submit
        self.assertFalse(sender.notify_drained_abort())
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender._drained_abort_notified_targets, {("peer", 1)})
        accept_late.append(True)
        self.assertTrue(sender.notify_drained_abort())
        self.assertEqual(
            sender._drained_abort_notified_targets, {("peer", 1), ("bootstrap-late", 3)}
        )

    def test_failure_clear_does_not_hold_source_lock_during_control_sends(self):
        """Block ACK delivery; failure inspection and another room still drain."""
        manager, sender = self._make_sender()
        manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
        manager.deferred_bootstrap.open(7, sender)
        manager.deferred_bootstrap.register(7, ("peer", 1))
        manager.register_deferred_ack_target(7, AckTarget("peer", 1, 9))
        chunk = SimpleNamespace(room=8)
        manager._source_drain.publish(chunk, queue.Queue(), lambda: True)
        entered, release, returned, drained, delivered = (
            threading.Event() for _ in range(5)
        )
        wire, failures = [], []

        class Socket:
            def setsockopt(self, *args):
                pass

            def connect(self, endpoint):
                pass

            def close(self, **kwargs):
                pass

            def send_multipart(self, parts, flags):
                wire.append((threading.current_thread().name, tuple(parts)))
                if len(wire) == 1:
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release the control socket")
                if len(wire) >= 2:
                    delivered.set()

        sock = Socket()
        manager._send_multipart_locked = (
            lambda endpoint, parts, **kwargs: sock.send_multipart(parts, zmq.NOBLOCK)
        )

        def inspect_failure():
            try:
                sender.failure_exception()
            except Exception as exc:
                failures.append(exc)
            finally:
                returned.set()

        def drain_other_room():
            manager._source_drain.complete(chunk)
            drained.set()

        with patch(
            "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
            return_value=SimpleNamespace(socket=lambda *_: sock),
        ):
            notifier = DrainedAbortNotifier()
            manager._drained_abort_notifier = notifier
            caller = threading.Thread(target=inspect_failure, daemon=True)
            drainer = threading.Thread(target=drain_other_room, daemon=True)
            try:
                caller.start()
                self.assertTrue(entered.wait(5))
                drainer.start()
                self.assertTrue(returned.wait(5))
                self.assertTrue(drained.wait(5))
                self.assertEqual(sender.poll(), KVPoll.Failed)
            finally:
                release.set()
                caller.join(5)
                if drainer.ident is not None:
                    drainer.join(5)
                try:
                    self.assertTrue(delivered.wait(5))
                finally:
                    notifier.close()
        self.assertIsInstance(failures[0], KVTransferError)
        self.assertIn("failure", str(failures[0]))
        self.assertTrue(all(name == "mooncake-drained-notify" for name, _ in wire))
        self.assertIn(tuple(AbortAck(7, 0, 9).to_zmq()), [parts for _, parts in wire])
        self.assertIn(
            tuple(
                manager._encode_kv_status_message(
                    bootstrap_room=7,
                    status=KVPoll.Failed,
                    failure_reason="Prefill bootstrap failed",
                )
            ),
            [parts for _, parts in wire],
        )
        self.assertNotIn(7, manager.transfer_infos)

    def test_ack_waits_for_queued_source_and_keeps_rejected_generation(self):
        """A parked/queued write and a full worker cannot authorize reuse."""
        manager, _ = self._make_sender()
        chunk = SimpleNamespace(room=7)
        manager._source_drain.publish(chunk, queue.Queue(), lambda: True)
        manager.register_deferred_ack_target(7, AckTarget("peer", 1, 9))
        wire, accepted = [], []

        def submit(room, endpoint, ipv6, parts, deadline):
            if not accepted:
                return False
            wire.append(tuple(parts))
            return True

        manager._drained_abort_notifier.submit = submit
        self.assertFalse(manager._maybe_ack_drained_abort(7))
        self.assertEqual(wire, [])
        manager._source_drain.complete(chunk)
        self.assertFalse(manager._maybe_ack_drained_abort(7))
        self.assertIn(7, manager._deferred_ack_targets)
        manager.register_deferred_ack_target(7, AckTarget("peer", 1, 10))
        accepted.append(True)
        self.assertTrue(manager._maybe_ack_drained_abort(7))
        self.assertEqual(wire, [tuple(AbortAck(7, 0, 10).to_zmq())])
        self.assertNotIn(7, manager._deferred_ack_targets)

    def test_failed_room_cannot_become_ready_when_terminal_table_is_full(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._drained_abort_lock = threading.Lock()
        manager._drained_abort_rooms = OrderedDict()
        manager.transfer_infos = {}
        manager.request_status = {7: KVPoll.Failed}
        manager.update_status = Mock()
        manager.source_transfers_drained = lambda _: False
        info = SimpleNamespace(endpoint="peer", dst_port=1, is_dummy=False)
        manager._register_bootstrap_transfer(7, "s", info, 1)
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        self.assertEqual(manager.transfer_infos[7], {"s": info})
        manager.update_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
