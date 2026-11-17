"""Drained notification saturation cannot block scheduling or revive rooms."""

import queue
import socket
import threading
import time
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
        manager._deferred_ack_deadlines = OrderedDict()
        manager._control_notification_lock = threading.RLock()
        manager._pending_control_notifications = OrderedDict()
        manager._control_rooms = OrderedDict()
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
        manager._reserve_control_room(7, active=True)
        return manager, sender

    def _make_constructor_manager(self):
        manager, _ = self._make_sender()
        manager.request_status.clear()
        manager.failure_records.clear()
        manager.transfer_infos.clear()
        manager.req_to_decode_prefix_len.clear()
        manager.req_to_pd_hidden_meta.clear()
        manager._control_rooms.clear()
        manager.deferred_bootstrap = DeferredBootstrap(10, capacity=1)
        manager.is_dummy_cp_rank = False
        manager.enable_trace = False
        return manager

    def test_aged_active_room_keeps_fresh_success_while_outbox_is_full(self):
        """A long-lived sender cannot expire a new Success before handoff."""
        manager = self._make_constructor_manager()
        with (
            patch(
                "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
                return_value=100,
            ),
            patch(
                "sglang.srt.disaggregation.common.conn.get_parallel",
                return_value=SimpleNamespace(num_dp_ranks=1),
            ),
        ):
            sender = MooncakeKVSender(manager, "prefill:8998", 7)
            manager.deferred_bootstrap.register(7, ("peer", 1))
        self.assertEqual(manager._control_rooms[7].deadline, 400)
        manager._drained_abort_capacity = 2
        for room in (97, 98):
            manager._pending_control_notifications[
                (room, "tcp://busy:99", (str(room).encode(),))
            ] = (False, 701)
        accepted, wire = [], []

        def submit(room, endpoint, ipv6, parts, deadline):
            if not accepted:
                return False
            wire.append(tuple(parts))
            return True

        manager._drained_abort_notifier.submit = submit
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=401
        ):
            manager.update_status(7, KVPoll.Success)
            sender.clear()
            self.assertNotIn(7, manager.deferred_bootstrap.rooms)
            self.assertNotIn(7, manager.request_status)
            manager._retry_pending_control_notifications()
            expected = tuple(
                manager._encode_kv_status_message(
                    bootstrap_room=7,
                    status=KVPoll.Success,
                    failure_reason=None,
                )
            )
            pending = manager._control_rooms[7].targets["tcp://peer:1"][1]
            self.assertIn((expected, 701), pending.values())
            self.assertEqual(wire, [])
            manager._pending_control_notifications.clear()
            accepted.append(True)
            manager._retry_pending_control_notifications()
        self.assertEqual(wire, [expected])
        self.assertNotIn(7, manager._control_rooms)
        self.assertEqual(manager._control_credits_used(), 0)

    def test_constructor_bootstrap_rejection_preserves_existing_reservations(self):
        """A refused bootstrap open cannot acquire credits or retire another owner."""
        for existing in ("none", "reservation", "sender"):
            with self.subTest(existing=existing):
                manager = self._make_constructor_manager()
                first = None
                with patch(
                    "sglang.srt.disaggregation.common.conn.get_parallel",
                    return_value=SimpleNamespace(num_dp_ranks=1),
                ):
                    if existing == "sender":
                        first = MooncakeKVSender(manager, "prefill:8998", 7)
                    else:
                        manager.deferred_bootstrap.open(99, object())
                        if existing == "reservation":
                            self.assertTrue(manager._reserve_control_room(7))
                    owner = manager._control_rooms.get(7)
                    credits = manager._control_credits_used()
                    active = owner.active if owner is not None else None
                    rejected = MooncakeKVSender(manager, "prefill:8998", 7)
                self.assertFalse(rejected._owns_bootstrap_room)
                self.assertEqual(rejected.poll(), KVPoll.Failed)
                self.assertEqual(manager._control_credits_used(), credits)
                self.assertIs(manager._control_rooms.get(7), owner)
                rejected.clear()
                self.assertEqual(manager._control_credits_used(), credits)
                self.assertIs(manager._control_rooms.get(7), owner)
                if owner is not None:
                    self.assertEqual(owner.active, active)
                if first is not None:
                    self.assertIs(manager.deferred_bootstrap.rooms[7].owner, first)
                    self.assertEqual(manager.request_status[7], KVPoll.Failed)
                    self.assertIn(7, manager.failure_records)
                else:
                    self.assertNotIn(7, manager.request_status)
                    self.assertNotIn(7, manager.failure_records)

    def test_success_waits_for_close_before_immediate_room_reuse(self):
        """A retry between enqueue and close cannot retire the next room owner."""
        manager = self._make_constructor_manager()
        closing, release, attempted, closed, published = (
            threading.Event() for _ in range(5)
        )
        failures, acquired, replacements, wire = [], [], [], []
        with patch(
            "sglang.srt.disaggregation.common.conn.get_parallel",
            return_value=SimpleNamespace(num_dp_ranks=1),
        ):
            sender = MooncakeKVSender(manager, "prefill:8998", 7)
            bootstrap = manager.deferred_bootstrap
            bootstrap.register(7, ("peer", 1))
            manager.update_status(7, KVPoll.Success)
            previous = manager._control_rooms[7]
            lock = manager._control_notification_lock
            close = manager._close_control_room

            class ObservedLock:
                def __enter__(self):
                    if threading.current_thread() is retry_thread:
                        obtained = lock.acquire(blocking=False)
                        acquired.append(obtained)
                        attempted.set()
                        if not obtained:
                            lock.acquire()
                    else:
                        lock.acquire()
                    return self

                def __exit__(self, *args):
                    lock.release()

            def close_after_enqueue(*args, **kwargs):
                self.assertTrue(previous.targets["tcp://peer:1"][1])
                closing.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release terminal close")
                close(*args, **kwargs)
                closed.set()

            def submit(room, endpoint, ipv6, parts, deadline):
                self.assertTrue(closed.is_set())
                self.assertNotIn(7, manager._control_rooms)
                wire.append(tuple(parts))
                replacements.append(MooncakeKVSender(manager, "prefill:8998", 7))
                published.set()
                return True

            def run(action):
                try:
                    action()
                except BaseException as exc:
                    failures.append(exc)

            clear_thread = threading.Thread(
                target=lambda: run(sender.clear), daemon=True
            )
            retry_thread = threading.Thread(
                target=lambda: run(manager._retry_pending_control_notifications),
                daemon=True,
            )
            manager._control_notification_lock = ObservedLock()
            manager._close_control_room = close_after_enqueue
            manager._drained_abort_notifier.submit = submit
            try:
                clear_thread.start()
                self.assertTrue(closing.wait(5))
                retry_thread.start()
                self.assertTrue(attempted.wait(5))
                self.assertEqual(acquired, [False])
                self.assertFalse(published.is_set())
            finally:
                release.set()
                clear_thread.join(5)
                if retry_thread.ident is not None:
                    retry_thread.join(5)
            self.assertFalse(clear_thread.is_alive())
            self.assertFalse(retry_thread.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(published.is_set())
            self.assertEqual(
                wire,
                [
                    tuple(
                        manager._encode_kv_status_message(
                            bootstrap_room=7,
                            status=KVPoll.Success,
                            failure_reason=None,
                        )
                    )
                ],
            )
            self.assertEqual(len(replacements), 1)
            self.assertIs(bootstrap.rooms[7].owner, replacements[0])
            current = manager._control_rooms[7]
            self.assertIsNot(current, previous)
            self.assertTrue(current.active)
            self.assertFalse(current.closed)
            self.assertEqual(manager.request_status[7], KVPoll.Bootstrapping)

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
                self.assertTrue(bootstrap.rooms[7].ready_notified)
                self.assertEqual(len(notifier._pending), 1)
                self.assertEqual(len(manager._control_rooms[7].targets), 1)
                submit = notifier.submit
                encoding_failures = []

                def retry_bootstrap(*args):
                    if not encoding_failures:
                        encoding_failures.append(True)
                        raise ValueError("injected producer encoding failure")
                    return submit(*args)

                notifier.submit = retry_bootstrap
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

    def test_queue_refusal_keeps_terminal_and_late_status_owners_after_clear(self):
        """A retired sender and a tombstone's late peer survive submit=False."""
        manager, sender = self._make_sender()
        accepted, wire = [], []

        def submit(room, endpoint, ipv6, parts, deadline):
            if not accepted:
                return False
            wire.append((endpoint, tuple(parts)))
            return True

        manager._drained_abort_notifier.submit = submit
        self.assertTrue(sender.notify_drained_abort())
        self.assertNotIn(7, manager.request_status)
        self.assertNotIn(7, manager.transfer_infos)
        self.assertNotIn(7, manager.failure_records)
        manager._register_bootstrap_transfer(
            7, "late", SimpleNamespace(endpoint="late", dst_port=2, is_dummy=False), 1
        )
        manager._retry_pending_control_notifications()
        self.assertEqual(len(manager._pending_control_notifications), 2)
        self.assertEqual(wire, [])
        accepted.append(True)
        manager._retry_pending_control_notifications()
        expected = tuple(
            manager._encode_kv_status_message(
                bootstrap_room=7,
                status=KVPoll.Failed,
                failure_reason="Prefill request aborted after source drain",
            )
        )
        self.assertEqual(wire, [("tcp://peer:1", expected), ("tcp://late:2", expected)])
        self.assertEqual(manager._pending_control_notifications, {})
        self.assertNotIn(7, manager.request_status)

    def test_terminal_refusal_clears_maps_and_later_completes(self):
        """A full terminal table must not strand the sender's next cleanup poll."""
        manager, sender = self._make_sender()
        manager._drained_abort_capacity = 2
        manager._drained_abort_rooms.update({1: float("inf"), 2: float("inf")})
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

    def test_success_after_clear_retries_without_a_room_or_sender_callback(self):
        """Deferred Success must outlive close deleting the bootstrap table entry."""
        manager, sender = self._make_sender()
        bootstrap = manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
        bootstrap.open(7, sender)
        bootstrap.register(7, ("peer", 1))
        manager.request_status[7] = KVPoll.Success
        accepted, wire = [], []

        def submit(room, endpoint, ipv6, parts, deadline):
            if not accepted:
                return False
            wire.append(tuple(parts))
            return True

        manager._drained_abort_notifier.submit = submit
        sender.clear()
        self.assertFalse(sender._owns_bootstrap_room)
        self.assertNotIn(7, bootstrap.rooms)
        self.assertNotIn(7, manager.request_status)
        manager._retry_pending_control_notifications()
        self.assertEqual(len(manager._pending_control_notifications), 1)
        self.assertEqual(wire, [])
        accepted.append(True)
        manager._retry_pending_control_notifications()
        self.assertEqual(
            wire,
            [
                tuple(
                    manager._encode_kv_status_message(
                        bootstrap_room=7,
                        status=KVPoll.Success,
                        failure_reason=None,
                    )
                )
            ],
        )
        self.assertEqual(manager._pending_control_notifications, {})

    def test_full_outbox_and_notifier_keep_reserved_success_and_late_failure(self):
        """Double saturation cannot discard clear's last owner or a late tombstone."""
        for success in (True, False):
            with self.subTest(success=success):
                manager, sender = self._make_sender()
                manager._drained_abort_capacity = 2
                bootstrap = manager.deferred_bootstrap = DeferredBootstrap(
                    10, capacity=1
                )
                bootstrap.open(7, sender)
                if success:
                    bootstrap.register(7, ("peer", 1))
                    manager.request_status[7] = KVPoll.Success
                notifier = DrainedAbortNotifier.__new__(DrainedAbortNotifier)
                notifier.capacity = 1
                notifier._pending = OrderedDict()
                notifier._condition = threading.Condition()
                notifier._closed = False
                self.assertTrue(
                    notifier.submit(
                        99, "tcp://busy:99", False, [b"occupied"], float("inf")
                    )
                )
                manager._drained_abort_notifier = notifier
                for room in (97, 98):
                    manager._pending_control_notifications[
                        (room, "tcp://busy:99", (str(room).encode(),))
                    ] = (False, float("inf"))
                self.assertEqual(
                    manager._control_credits_used(), manager._drained_abort_capacity
                )
                self.assertFalse(manager._reserve_control_room(8, active=True))
                sender.clear()
                if success:
                    self.assertNotIn(7, bootstrap.rooms)
                    endpoint, status, reason = "tcp://peer:1", KVPoll.Success, None
                else:
                    manager._notify_bootstrap(7, bootstrap.register(7, ("late", 3)))
                    endpoint, status, reason = (
                        "tcp://late:3",
                        KVPoll.Failed,
                        "Prefill bootstrap failed",
                    )
                expected = tuple(
                    manager._encode_kv_status_message(
                        bootstrap_room=7,
                        status=status,
                        failure_reason=reason,
                    )
                )
                self.assertIn(
                    expected,
                    [
                        parts
                        for parts, _ in manager._control_rooms[7]
                        .targets[endpoint][1]
                        .values()
                    ],
                )
                manager._retry_pending_control_notifications()
                self.assertEqual(len(manager._pending_control_notifications), 2)
                self.assertEqual(len(notifier._pending), 1)
                self.assertIn(
                    expected,
                    [
                        parts
                        for parts, _ in manager._control_rooms[7]
                        .targets[endpoint][1]
                        .values()
                    ],
                )
                observed = []
                for _ in range(5):
                    notifier._pending.clear()
                    manager._retry_pending_control_notifications()
                    observed.extend(key[2] for key in notifier._pending)
                self.assertIn(expected, observed)
                self.assertEqual(manager._pending_control_notifications, {})
                if success:
                    self.assertNotIn(7, manager._control_rooms)

    def test_bootstrap_endpoint_arriving_after_close_survives_queue_refusal(self):
        """Failed bootstrap tombstones retain delivery after the sender retires."""
        manager, sender = self._make_sender()
        bootstrap = manager.deferred_bootstrap = DeferredBootstrap(timeout=10)
        bootstrap.open(7, sender)
        accepted, wire = [], []

        def submit(room, endpoint, ipv6, parts, deadline):
            if not accepted:
                return False
            wire.append((endpoint, tuple(parts)))
            return True

        manager._drained_abort_notifier.submit = submit
        self.assertTrue(sender.notify_drained_abort())
        self.assertIsNone(bootstrap.rooms[7].owner)
        manager._notify_bootstrap(7, bootstrap.register(7, ("bootstrap-late", 3)))
        manager._retry_pending_control_notifications()
        self.assertEqual(len(manager._pending_control_notifications), 2)
        accepted.append(True)
        manager._retry_pending_control_notifications()
        self.assertIn(
            (
                "tcp://bootstrap-late:3",
                tuple(
                    manager._encode_kv_status_message(
                        bootstrap_room=7,
                        status=KVPoll.Failed,
                        failure_reason="Prefill bootstrap failed",
                    )
                ),
            ),
            wire,
        )
        self.assertEqual(manager._pending_control_notifications, {})

    def test_normal_conclude_uses_the_same_retry_owner_for_success_and_failure(self):
        """Normal completion and explicit transfer errors cannot bypass the outbox."""
        for status in (KVPoll.Success, KVPoll.Failed):
            with self.subTest(status=status):
                manager, _ = self._make_sender()
                manager.request_status[7] = KVPoll.Transferring
                manager.failure_records.clear()
                accepted, wire = [], []

                def submit(room, endpoint, ipv6, parts, deadline):
                    if not accepted:
                        return False
                    wire.append(tuple(parts))
                    return True

                manager._drained_abort_notifier.submit = submit
                manager.conclude_transfer(bootstrap_room=7, status=status)
                manager._retry_pending_control_notifications()
                self.assertEqual(len(manager._pending_control_notifications), 1)
                accepted.append(True)
                manager._retry_pending_control_notifications()
                self.assertEqual(
                    wire,
                    [
                        tuple(
                            manager._encode_kv_status_message(
                                bootstrap_room=7,
                                status=status,
                                failure_reason=(
                                    "KV transfer failed"
                                    if status == KVPoll.Failed
                                    else None
                                ),
                            )
                        )
                    ],
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
                self.assertTrue(returned.wait(5))
                # Drive the outbox after installing the real notifier. Starting
                # its callback in __init__ could consume ACKs via the old stub.
                manager._retry_pending_control_notifications()
                self.assertTrue(entered.wait(5))
                drainer.start()
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
        manager.register_deferred_ack_target(7, AckTarget("peer", 1, 10))
        manager._source_drain.complete(chunk)
        self.assertTrue(manager._maybe_ack_drained_abort(7))
        manager._retry_pending_control_notifications()
        self.assertEqual(len(manager._pending_control_notifications), 1)
        self.assertNotIn(7, manager._deferred_ack_targets)
        accepted.append(True)
        manager._retry_pending_control_notifications()
        self.assertEqual(wire, [tuple(AbortAck(7, 0, 10).to_zmq())])
        self.assertNotIn(7, manager._deferred_ack_targets)
        self.assertEqual(manager._pending_control_notifications, {})

    def test_ack_owner_budget_and_deadline_span_source_wait_and_queue_refusal(self):
        """Refused ACKs cannot accumulate or renew their lifetime on retry."""
        manager, _ = self._make_sender()
        manager._drained_abort_capacity = 1
        manager._drained_abort_notifier.submit = lambda *args: False
        manager._staging_outstanding[7] = 1
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=100
        ):
            manager.register_deferred_ack_target(7, AckTarget("peer", 1, 9))
            manager.register_deferred_ack_target(8, AckTarget("other", 2, 1))
            self.assertNotIn(8, manager._deferred_ack_targets)
            self.assertEqual(list(manager._deferred_ack_deadlines.values()), [400])
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=200
        ):
            manager.register_deferred_ack_target(7, AckTarget("peer", 1, 10))
            self.assertEqual(list(manager._deferred_ack_deadlines.values()), [400])
            manager._staging_outstanding[7] = 0
            self.assertTrue(manager._maybe_ack_drained_abort(7))
            manager._retry_pending_control_notifications()
            self.assertEqual(
                list(manager._pending_control_notifications.values()), [(False, 400)]
            )
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=401
        ):
            manager._retry_pending_control_notifications()
            self.assertEqual(manager._pending_control_notifications, {})
            self.assertEqual(manager._deferred_ack_deadlines, {})
            self.assertEqual(manager._deferred_ack_targets, {})
            manager.register_deferred_ack_target(8, AckTarget("other", 2, 1))
            self.assertIn(8, manager._deferred_ack_targets)

    def test_poisoned_write_never_emits_an_ack_and_owner_expires(self):
        """Neither chunk completion nor retry expiry proves a poisoned write drained."""
        manager, _ = self._make_sender()
        wire = []
        manager._drained_abort_notifier.submit = lambda *args: wire.append(args) or True
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=100
        ):
            manager.register_deferred_ack_target(7, AckTarget("peer", 1, 9))
            manager._source_drain.poison(7)
            self.assertFalse(manager._maybe_ack_drained_abort(7))
            manager._retry_pending_control_notifications()
            self.assertEqual(wire, [])
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=401
        ):
            manager._retry_pending_control_notifications()
            self.assertEqual(wire, [])
            self.assertEqual(manager._deferred_ack_targets, {})
            self.assertEqual(manager._deferred_ack_deadlines, {})
            self.assertFalse(manager._source_drain.drained(7))

    def test_again_retries_without_exponential_delay_and_drains_a_healthy_burst(self):
        """Ordinary PUSH backpressure must not become endpoint failure backoff."""
        notifier = DrainedAbortNotifier.__new__(DrainedAbortNotifier)
        notifier.capacity = 64
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
                else:
                    now[0] += timeout

        class Socket:
            def setsockopt(self, *args):
                pass

            def connect(self, endpoint):
                pass

            def close(self, **kwargs):
                pass

            def send_multipart(self, parts, flags):
                attempts.append(now[0])
                if len(attempts) <= 5:
                    raise zmq.Again()
                delivered.append((now[0], tuple(parts)))

        notifier._condition = Condition()
        for room in range(64):
            notifier.submit(room, "healthy", False, [str(room).encode()], 1)
        with (
            patch(
                "sglang.srt.disaggregation.mooncake.notifications.time.monotonic",
                side_effect=lambda: now[0],
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
                return_value=SimpleNamespace(socket=lambda *_: Socket()),
            ),
        ):
            notifier._run()
        self.assertEqual(
            [parts for _, parts in delivered],
            [(str(room).encode(),) for room in range(64)],
        )
        for index, attempted_at in enumerate(attempts[:6]):
            self.assertAlmostEqual(attempted_at, index * 0.001)
        self.assertEqual({sent_at for sent_at, _ in delivered}, {attempts[5]})

    def test_native_zmq_immediate_connection_and_healthy_status_ack_burst(self):
        """A real IMMEDIATE handshake may Again; all subsequent frames survive."""
        context = zmq.Context.instance()
        receiver = context.socket(zmq.PULL)
        reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        endpoint = f"tcp://127.0.0.1:{port}"
        receiver.setsockopt(zmq.RCVTIMEO, 5000)
        again = threading.Event()
        options = {}

        class Socket:
            def __init__(self):
                self.socket = context.socket(zmq.PUSH)

            def setsockopt(self, option, value):
                options[option] = value
                self.socket.setsockopt(option, value)

            def connect(self, address):
                self.socket.connect(address)

            def close(self, **kwargs):
                self.socket.close(**kwargs)

            def send_multipart(self, parts, flags):
                try:
                    self.socket.send_multipart(parts, flags=flags)
                except zmq.Again:
                    again.set()
                    raise

        messages = [
            tuple(AbortAck(7, 0, generation).to_zmq()) for generation in range(32)
        ]
        manager, _ = self._make_sender()
        messages += [
            tuple(
                manager._encode_kv_status_message(
                    bootstrap_room=room,
                    status=KVPoll.Success,
                    failure_reason=None,
                )
            )
            for room in range(32)
        ]
        with patch(
            "sglang.srt.disaggregation.mooncake.notifications.zmq.Context.instance",
            return_value=SimpleNamespace(socket=lambda *_: Socket()),
        ):
            notifier = DrainedAbortNotifier(capacity=len(messages))
            try:
                for index, parts in enumerate(messages):
                    self.assertTrue(
                        notifier.submit(
                            index, endpoint, False, parts, time.monotonic() + 30
                        )
                    )
                self.assertTrue(again.wait(5))
                reserved.close()
                receiver.bind(endpoint)
                self.assertEqual(
                    [tuple(receiver.recv_multipart()) for _ in messages], messages
                )
                self.assertEqual(options[zmq.IMMEDIATE], 1)
                self.assertEqual(options[zmq.SNDHWM], 1000)
            finally:
                notifier.close()
                receiver.close(linger=0)
                reserved.close()

    def test_failed_room_cannot_become_ready_when_terminal_table_is_full(self):
        manager, _ = self._make_sender()
        manager.transfer_infos = {}
        manager.update_status = Mock()
        manager.source_transfers_drained = lambda _: False
        info = SimpleNamespace(endpoint="peer", dst_port=1, is_dummy=False)
        manager._register_bootstrap_transfer(7, "s", info, 1)
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        self.assertEqual(manager.transfer_infos[7], {"s": info})
        manager.update_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
