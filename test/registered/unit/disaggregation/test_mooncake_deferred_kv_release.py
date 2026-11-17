"""Deferred decode-side KV release on the Mooncake backend.

The bootstrap thread records the ACK target while a transfer is outstanding.
Only the transfer worker can send that ACK after its write has drained.
"""

import threading
import unittest
from collections import OrderedDict, defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
from test_deferred_decode_kv_release import (
    ABORT_GENERATION,
    ABORT_SCENARIOS,
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
)

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    ABORT_ACK_TAG,
    ABORT_TAG,
    AbortAck,
    AbortNotification,
    AckTarget,
    CommonKVManager,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.hidden_events import PDHiddenEventManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.mooncake.source_drain import SourceDrainTracker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 106
DECODE_IP = "10.0.0.2"
DECODE_PORT = 6000


def _manager(
    *, enabled: bool = True, status: KVPoll | None = KVPoll.WaitingForInput
) -> MooncakeKVManager:
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager._source_drain = SourceDrainTracker()
    manager._transfer_completion_condition = threading.Condition()
    manager.pd_hidden_events = PDHiddenEventManager(manager)
    manager.enable_deferred_decode_kv_release = enabled
    manager.request_status = {} if status is None else {ROOM: status}
    manager._deferred_ack_targets = {}
    manager._deferred_ack_deadlines = OrderedDict()
    manager._drained_abort_capacity = 4096
    manager._control_notification_lock = threading.RLock()
    manager._control_rooms = OrderedDict()
    manager._pending_control_notifications = OrderedDict()
    manager.bootstrap_timeout = manager.waiting_timeout = 10
    manager.deferred_bootstrap = None
    manager._stopped = False
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = {}
    manager.check_status = lambda room: manager.request_status[room]
    manager.update_status = lambda room, value: manager.request_status.__setitem__(
        room, value
    )
    manager._sent = []
    manager._send_abort_ack = lambda *args: manager._sent.append(args)
    manager._prefill_unique_rank = lambda: 0

    def enqueue_ack(room, endpoint, port, parts, *, deadline=None):
        ack = AbortAck.from_zmq(parts)
        manager._sent.append((room, AckTarget(endpoint, port, ack.generation)))
        return True

    manager._queue_control_notification = enqueue_ack
    manager._send_multipart_locked = MagicMock()
    return manager


def _chunk() -> TransferKVChunk:
    return TransferKVChunk(
        room=ROOM,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=False,
        chunk_id=0,
        prefill_aux_index=None,
        state_indices=None,
    )


class TestMooncakeAbortNotification(
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
    CustomTestCase,
):
    room = ROOM
    decode_ip = DECODE_IP
    decode_port = DECODE_PORT
    # Mooncake's target implementation marks even retired rooms Failed while
    # a counted write remains, preventing new work before the deferred ACK.
    abort_scenarios = tuple(
        case._replace(expected_status=KVPoll.Failed) if case.outstanding else case
        for case in ABORT_SCENARIOS
    )

    def _make_abort_manager(self, status: KVPoll | None):
        return _manager(status=status)

    def _provoke_worker_failure(self, manager) -> None:
        manager.enable_trace = False
        manager.bootstrap_port = 8998
        manager._staging_outstanding = defaultdict(int)
        manager.check_status = MagicMock(side_effect=RuntimeError("worker error"))
        queue = MagicMock()
        queue.get.return_value = _chunk()

        with self.assertRaisesRegex(RuntimeError, "Transfer thread failed"):
            manager.transfer_worker(queue, MagicMock())
        manager.check_status = lambda room: manager.request_status[room]

    def test_worker_failure_retains_source_poison_after_logical_room_reuse(self):
        """A fatal worker error is not proof that a remote write has stopped."""
        manager = _manager()
        target = AckTarget(DECODE_IP, DECODE_PORT, ABORT_GENERATION)
        manager._deferred_ack_targets[ROOM] = {(target.ip, target.port): target}
        self._provoke_worker_failure(manager)
        self.assertEqual(manager._staging_outstanding[ROOM], 1)
        self.assertNotIn(ROOM, manager._deferred_ack_targets)
        self._dispatch_abort(manager)
        self.assertNotIn(ROOM, manager._deferred_ack_targets)
        self.assertEqual(manager._sent, [])

        manager.request_status.pop(ROOM, None)
        CommonKVManager.update_status(manager, ROOM, KVPoll.Bootstrapping)
        self._dispatch_abort(manager)
        self.assertFalse(manager.source_transfers_drained(ROOM))
        self.assertEqual(
            manager._deferred_ack_targets[ROOM], {(target.ip, target.port): target}
        )
        self.assertEqual(manager._sent, [])

    def test_generationless_abort_marks_failed_without_ack_or_warning(self):
        # A decode that did not arm (prealloc abort) sends no generation; that
        # is normal and must not warn about mixed versions.
        manager = _manager()
        message = AbortNotification(ROOM, DECODE_IP, DECODE_PORT).to_zmq()

        with patch(
            "sglang.srt.disaggregation.common.conn.logger.warning_once"
        ) as warning:
            claimed = manager._handle_abort_notification(message)

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._deferred_ack_targets, {})
        self.assertEqual(manager._sent, [])
        warning.assert_not_called()

    def test_abort_without_valid_return_address_still_marks_room_failed(self):
        messages = (
            ("missing", [ABORT_TAG, str(ROOM).encode("ascii")]),
            (
                "malformed port",
                [
                    ABORT_TAG,
                    str(ROOM).encode("ascii"),
                    DECODE_IP.encode("ascii"),
                    b"bad",
                ],
            ),
        )

        for name, message in messages:
            with self.subTest(name=name):
                manager = _manager()

                claimed = manager._handle_abort_notification(message)

                self.assertTrue(claimed)
                self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
                self.assertEqual(manager._deferred_ack_targets, {})
                self.assertEqual(manager._sent, [])

    def test_feature_off_sends_legacy_ack(self):
        manager = _manager(enabled=False)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        payload = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(payload, [ABORT_ACK_TAG, str(ROOM).encode("ascii")])


if __name__ == "__main__":
    unittest.main()
