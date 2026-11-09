"""CommonKVManager._connect: monitored dead sockets are rebuilt with bounded backoff.

Covers the fix for a cached PUSH socket whose *first* connection attempt fails
(dead port): the old code only watched EVENT_DISCONNECTED, which never fires for
a connection that never succeeded, so the dead socket was cached and returned to
every later sender forever (RECONNECT_IVL=-1 makes libzmq never retry on its
own). The fix also watches EVENT_CLOSED and rebuilds after a bounded backoff.
"""

import socket
import threading
import time
import unittest
from unittest.mock import patch

import zmq

from sglang.srt.disaggregation.common import conn as conn_module
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_manager():
    mgr = object.__new__(CommonKVManager)
    mgr._zmq_ctx = zmq.Context()
    mgr._socket_cache = {}
    mgr._monitor_cache = {}
    mgr._socket_unreachable_since = {}
    mgr._socket_send_locks = {}
    mgr._socket_lock = threading.Lock()
    return mgr


class TestCommonKVManagerConnect(unittest.TestCase):
    def test_initial_connect_failure_backs_off_then_rebuilds(self):
        # Reserve then release a port so the very first connect fails (nothing
        # listens there); libzmq surfaces this asynchronously as EVENT_CLOSED,
        # not EVENT_DISCONNECTED (that one needs a connection that first
        # succeeded).
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"
        probe.close()

        mgr = _make_manager()
        try:
            old = mgr._connect(endpoint)
            # Wait for libzmq to report the failed connection on the monitor.
            monitor = mgr._monitor_cache[endpoint]
            self.assertTrue(monitor.poll(3000), "no monitor event for dead endpoint")

            with patch.object(
                envs.SGLANG_DISAGGREGATION_ZMQ_SOCKET_RECONNECT_BACKOFF_MS,
                "get",
                return_value=50,
            ):
                with self.assertRaises(zmq.Again):
                    mgr._connect(endpoint)
                time.sleep(0.06)
                new = mgr._connect(endpoint)
            self.assertIsNot(new, old)
        finally:
            mgr._zmq_ctx.destroy(linger=0)

    def test_healthy_cached_socket_is_reused(self):
        peer = zmq.Context().socket(zmq.PULL)
        port = peer.bind_to_random_port("tcp://127.0.0.1")
        endpoint = f"tcp://127.0.0.1:{port}"

        mgr = _make_manager()
        try:
            first = mgr._connect(endpoint)
            first.send_multipart([b"hello"])
            self.assertTrue(peer.poll(3000), "healthy endpoint did not receive data")
            self.assertEqual(peer.recv_multipart(), [b"hello"])
            second = mgr._connect(endpoint)
            self.assertIs(first, second)
        finally:
            mgr._zmq_ctx.destroy(linger=0)
            peer.close(linger=0)

    def test_disconnect_manager_endpoint_clears_unreachable_state(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"
        probe.close()

        mgr = _make_manager()
        try:
            mgr._connect(endpoint)
            monitor = mgr._monitor_cache[endpoint]
            self.assertTrue(monitor.poll(3000), "no monitor event for dead endpoint")
            with self.assertRaises(zmq.Again):
                mgr._connect(endpoint)  # Observe EVENT_CLOSED and begin backoff.
            self.assertIn(endpoint, mgr._socket_unreachable_since)

            mgr._disconnect_manager_endpoint(endpoint)

            self.assertNotIn(endpoint, mgr._socket_cache)
            self.assertNotIn(endpoint, mgr._monitor_cache)
            self.assertNotIn(endpoint, mgr._socket_unreachable_since)
        finally:
            mgr._zmq_ctx.destroy(linger=0)


if __name__ == "__main__":
    unittest.main()
