"""Bounded, best-effort status delivery with sockets owned by one worker."""

import logging
import threading
import time
from collections import OrderedDict

import msgspec
import zmq
from prometheus_client import Counter

logger = logging.getLogger(__name__)

DRAINED_ABORT_REJECTIONS = Counter(
    "sglang:mooncake_drained_abort_rejections_total",
    "Rejected Mooncake terminal markers or asynchronous control notifications.",
    ["reason"],
)


class ControlRoomOwner(msgspec.Struct):
    deadline: float
    active: bool = False
    closed: bool = False
    failed: bool = False
    # Reserve bootstrap and first destination credits before their addresses
    # are known. Each endpoint has one payload per status, not per error string.
    targets: dict = msgspec.field(default_factory=dict)


class DrainedAbortNotifier:
    def __init__(self, capacity=4096, retry_pending=None):
        self.capacity = capacity
        self._pending = OrderedDict()
        self._condition = threading.Condition()
        self._closed = False
        self._retry_pending = retry_pending
        self._thread = threading.Thread(
            target=self._run, name="mooncake-drained-notify", daemon=True
        )
        self._thread.start()

    def submit(self, room, endpoint, ipv6, parts, deadline) -> bool:
        # Status and generation-tagged ACKs for one room must not coalesce.
        parts = tuple(parts)
        key = (room, endpoint, parts)
        with self._condition:
            if self._closed:
                DRAINED_ABORT_REJECTIONS.labels("notifier_closed").inc()
                return False
            if key in self._pending:
                return True
            if len(self._pending) >= self.capacity:
                DRAINED_ABORT_REJECTIONS.labels("notification_capacity").inc()
                return False
            self._pending[key] = (ipv6, parts, deadline)
            self._condition.notify()
            return True

    def close(self):
        with self._condition:
            self._closed = True
            self._retry_pending = None
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join()

    def _run(self):
        # No manager socket/endpoint lock is ever acquired by this worker.
        sockets = OrderedDict()
        retry_at = {}
        retry_delay = {}
        failures = expired_count = 0
        last_summary = time.monotonic()
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._closed
                        or self._pending
                        or self._retry_pending is not None
                    )
                    if self._closed:
                        return
                    retry_pending = self._retry_pending
                # Producers retain rejected work in their existing room tables.
                # Never call back while holding the queue lock (source -> queue).
                if retry_pending is not None:
                    try:
                        retry_pending()
                    except Exception:
                        failures += 1
                with self._condition:
                    if self._closed:
                        return
                    tasks = list(self._pending.items())
                for key, task in tasks:
                    _, endpoint, _ = key
                    ipv6, parts, deadline = task
                    now = time.monotonic()
                    expired = now >= deadline
                    if not expired and now < retry_at.get(endpoint, 0):
                        continue
                    sent = False
                    if not expired:
                        try:
                            sock = sockets.get(endpoint)
                            if sock is None:
                                if len(sockets) >= self.capacity:
                                    _, old = sockets.popitem(last=False)
                                    old.close(linger=0)
                                sock = zmq.Context.instance().socket(zmq.PUSH)
                                sockets[endpoint] = sock
                                sock.setsockopt(zmq.LINGER, 0)
                                sock.setsockopt(zmq.IMMEDIATE, 1)
                                sock.setsockopt(zmq.SNDHWM, 1000)
                                if ipv6:
                                    sock.setsockopt(zmq.IPV6, 1)
                                sock.connect(endpoint)
                            sockets.move_to_end(endpoint)
                            sock.send_multipart(parts, flags=zmq.NOBLOCK)
                            sent = True
                            retry_at.pop(endpoint, None)
                            retry_delay.pop(endpoint, None)
                        except zmq.Again:
                            # IMMEDIATE also reports Again during connection
                            # setup and ordinary HWM pressure, not peer failure.
                            retry_at[endpoint] = time.monotonic() + 0.001
                        except Exception:
                            failures += 1
                            sock = sockets.pop(endpoint, None)
                            if sock is not None:
                                sock.close(linger=0)
                            delay = min(5.0, retry_delay.get(endpoint, 0.025) * 2)
                            retry_delay[endpoint] = delay
                            retry_at[endpoint] = time.monotonic() + delay
                    if sent or expired:
                        with self._condition:
                            # A retry may have been submitted after a prior task
                            # completed; never remove a different generation.
                            if self._pending.get(key) is task:
                                del self._pending[key]
                        if expired:
                            expired_count += 1
                with self._condition:
                    endpoints = {key[1] for key in self._pending}
                    retry_at = {k: v for k, v in retry_at.items() if k in endpoints}
                    retry_delay = {
                        k: v for k, v in retry_delay.items() if k in endpoints
                    }
                    now = time.monotonic()
                    if (failures or expired_count) and now - last_summary >= 30:
                        logger.warning(
                            "Mooncake control notifications: %d errors, "
                            "%d expired, %d pending",
                            failures,
                            expired_count,
                            len(self._pending),
                        )
                        failures = expired_count = 0
                        last_summary = now
                    if not self._closed:
                        wake_at = min(
                            [task[2] for task in self._pending.values()]
                            + [retry_at.get(endpoint, now) for endpoint in endpoints],
                            default=now + 5,
                        )
                        if self._retry_pending is not None:
                            wake_at = min(wake_at, now + 0.05)
                        self._condition.wait(timeout=max(0.001, wake_at - now))
        finally:
            if failures or expired_count:
                logger.warning(
                    "Mooncake control notifications: %d errors, %d expired",
                    failures,
                    expired_count,
                )
            for sock in sockets.values():
                sock.close(linger=0)
