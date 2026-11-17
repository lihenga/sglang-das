"""Drained notification saturation cannot block scheduling or revive rooms."""

import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.mooncake.notifications import DrainedAbortNotifier
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDrainedNotifications(CustomTestCase):
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

    def test_queue_refusal_retains_sender_until_notification_is_owned(self):
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.bootstrap_room = 7
        sender.is_transfer_drained = lambda: True
        sender.clear = Mock()
        accepted = []
        manager = SimpleNamespace(
            finish_drained_abort=lambda _: {("peer", 1)},
            queue_drained_abort_notification=lambda *args: bool(accepted),
            failure_lock=threading.Lock(),
            failure_records={7: "failure"},
        )
        sender.kv_mgr = manager
        self.assertFalse(sender.notify_drained_abort())
        self.assertEqual(manager.failure_records, {7: "failure"})
        self.assertEqual(sender._drained_abort_notified_targets, set())
        sender.clear.assert_not_called()
        accepted.append(True)
        self.assertTrue(sender.notify_drained_abort())
        self.assertEqual(manager.failure_records, {})
        self.assertEqual(sender._drained_abort_notified_targets, {("peer", 1)})

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
