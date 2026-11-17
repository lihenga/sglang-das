"""Bounded, best-effort status delivery with sockets owned by one worker."""

import logging
import threading
import time
from collections import OrderedDict

import zmq

logger = logging.getLogger(__name__)


class DrainedAbortNotifier:
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self._pending = OrderedDict()
        self._condition = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="mooncake-drained-notify", daemon=True
        )
        self._thread.start()

    def submit(self, room, endpoint, ipv6, parts, deadline) -> bool:
        key = (room, endpoint)
        with self._condition:
            if self._closed:
                return False
            if key in self._pending:
                return True
            if len(self._pending) >= self.capacity:
                return False
            self._pending[key] = (ipv6, tuple(parts), deadline)
            self._condition.notify()
            return True

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self):
        # No manager socket/endpoint lock is ever acquired by this worker.
        sockets = OrderedDict()
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending)
                    if self._closed:
                        return
                    tasks = list(self._pending.items())
                for key, task in tasks:
                    room, endpoint = key
                    ipv6, parts, deadline = task
                    expired = time.monotonic() >= deadline
                    sent = False
                    if not expired:
                        try:
                            sock = sockets.get(endpoint)
                            if sock is None:
                                if len(sockets) >= self.capacity:
                                    _, old = sockets.popitem(last=False)
                                    old.close(linger=0)
                                sock = zmq.Context.instance().socket(zmq.PUSH)
                                sock.setsockopt(zmq.LINGER, 0)
                                sock.setsockopt(zmq.IMMEDIATE, 1)
                                sock.setsockopt(zmq.SNDHWM, 1)
                                if ipv6:
                                    sock.setsockopt(zmq.IPV6, 1)
                                sockets[endpoint] = sock
                                sock.connect(endpoint)
                            sockets.move_to_end(endpoint)
                            sock.send_multipart(parts, flags=zmq.NOBLOCK)
                            sent = True
                        except zmq.Again:
                            pass
                        except Exception:
                            logger.warning(
                                "Drained-abort notification failed for room %s",
                                room,
                                exc_info=True,
                            )
                            sock = sockets.pop(endpoint, None)
                            if sock is not None:
                                sock.close(linger=0)
                    if sent or expired:
                        with self._condition:
                            # A retry may have been submitted after a prior task
                            # completed; never remove a different generation.
                            if self._pending.get(key) is task:
                                del self._pending[key]
                        if expired:
                            logger.warning(
                                "Drained-abort notification expired for room %s", room
                            )
                with self._condition:
                    if not self._closed:
                        self._condition.wait(timeout=0.05)
        finally:
            for sock in sockets.values():
                sock.close(linger=0)
