import importlib.util
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/srt/mem_cache/storage/mooncake_store/dfs_health.py"
)
_SPEC = importlib.util.spec_from_file_location("mooncake_dfs_health", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
AVAILABLE = _MODULE.AVAILABLE
UNAVAILABLE = _MODULE.UNAVAILABLE
MooncakeDfsHealthManager = _MODULE.MooncakeDfsHealthManager
resolve_dfs_health_config = _MODULE.resolve_dfs_health_config


class _FakeStore:
    def __init__(self, read_probes, write_probes):
        self.read_probes = iter(read_probes)
        self.write_probes = iter(write_probes)
        self.policies = []

    def probe_dfs_read(self):
        return next(self.read_probes)

    def probe_dfs_write(self):
        return next(self.write_probes)

    def set_dfs_access_policy(self, allow_read, allow_write):
        self.policies.append((allow_read, allow_write))
        return 0


register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class MooncakeDfsHealthManagerTest(CustomTestCase):
    def setUp(self):
        super().setUp()
        overrides = ExitStack()
        self.addCleanup(overrides.close)
        for field in (
            envs.SGLANG_MOONCAKE_DFS_HEALTH_PROBE_INTERVAL_S,
            envs.SGLANG_MOONCAKE_DFS_HEALTH_FAILURE_THRESHOLD,
            envs.SGLANG_MOONCAKE_DFS_HEALTH_SUCCESS_THRESHOLD,
            envs.SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S,
        ):
            overrides.enter_context(field.override(None))
            field.clear()

    def test_resolver_uses_cli_then_environment_then_defaults(self):
        env = {
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_INTERVAL_S": "4",
            "SGLANG_MOONCAKE_DFS_HEALTH_FAILURE_THRESHOLD": "5",
            "SGLANG_MOONCAKE_DFS_HEALTH_SUCCESS_THRESHOLD": "6",
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S": "7",
        }
        server_args = type(
            "ServerArgs",
            (),
            {
                "mooncake_dfs_health_probe_interval_s": 8,
                "mooncake_dfs_health_failure_threshold": None,
                "mooncake_dfs_health_success_threshold": None,
                "mooncake_dfs_health_probe_timeout_s": None,
            },
        )()
        for name, value in env.items():
            getattr(envs, name).set(value)
        self.assertEqual(
            resolve_dfs_health_config(server_args=server_args),
            {
                "probe_interval_s": 8,
                "failure_threshold": 5,
                "success_threshold": 6,
                "probe_timeout_s": 7,
            },
        )

    def test_resolver_uses_defaults_when_unset(self):
        self.assertEqual(
            resolve_dfs_health_config(),
            {
                "probe_interval_s": 10,
                "failure_threshold": 3,
                "success_threshold": 3,
                "probe_timeout_s": 2,
            },
        )

    def test_resolver_rejects_non_positive_value(self):
        envs.SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S.set(0)
        with self.assertRaisesRegex(ValueError, "probe_timeout_s must be positive"):
            resolve_dfs_health_config()

    def test_unknown_initial_native_policy_prevents_client_startup(self):
        """Clients must not warm up unless native fail-closed setup is confirmed."""

        class Store(_FakeStore):
            def set_dfs_access_policy(self, read, write):
                return -1

        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                with self.assertRaisesRegex(
                    RuntimeError, "set_dfs_access_policy failed"
                ):
                    MooncakeDfsHealthManager(
                        Store([], []), defer_policy_application=deferred
                    )

    def _manager(self, read_probes, write_probes):
        env = {
            "SGLANG_MOONCAKE_DFS_HEALTH_FAILURE_THRESHOLD": "2",
            "SGLANG_MOONCAKE_DFS_HEALTH_SUCCESS_THRESHOLD": "2",
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_INTERVAL_S": "10",
        }
        for name, value in env.items():
            getattr(envs, name).set(value)
        store = _FakeStore(read_probes, write_probes)
        return store, MooncakeDfsHealthManager(store)

    def _wait_for_policy(self, manager, policy):
        with manager._policy_condition:
            self.assertTrue(
                manager._policy_condition.wait_for(
                    lambda: manager._effective_policy == policy
                    and not manager._policy_dirty,
                    timeout=3,
                ),
                f"native policy did not converge to {policy}",
            )

    def test_write_degrades_without_disabling_dfs_read(self):
        store, manager = self._manager(
            [0, 0, 0, 0],
            [0, 0, -1, -1],
        )
        manager.probe_once()
        manager.probe_once()
        self.assertEqual(store.policies[-1], (True, True))

        manager.probe_once()
        snapshot = manager.probe_once()
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, UNAVAILABLE)
        self.assertEqual(store.policies[-1], (True, False))

    def test_read_and_write_recover_independently(self):
        store, manager = self._manager(
            [-1, -1, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, 0],
        )
        manager.probe_once()
        manager.probe_once()
        self.assertEqual(store.policies[-1], (False, False))

        manager.probe_once()
        snapshot = manager.probe_once()
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, UNAVAILABLE)
        self.assertEqual(store.policies[-1], (True, False))

        manager.probe_once()
        snapshot = manager.probe_once()
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(store.policies[-1], (True, True))

    def test_deferred_policy_waits_for_coordinated_result(self):
        store = _FakeStore([0], [0])
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 1,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
            defer_policy_application=True,
        )
        self.addCleanup(manager.close)

        snapshot = manager.probe_once()
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(manager.local_policy(), (True, True))
        self.assertEqual(store.policies, [(False, False)])

        manager.apply_effective_policy(True, False)
        self._wait_for_policy(manager, (True, False))
        manager.apply_effective_policy(True, False)
        self.assertEqual(store.policies, [(False, False), (True, False)])

    def test_deferred_stuck_policy_coalesces_updates_without_blocking_scheduler(self):
        """A pending native call must not block polls or overwrite a newer target."""
        entered = threading.Event()
        release = threading.Event()
        submitted = threading.Event()

        class Store(_FakeStore):
            def set_dfs_access_policy(self, read, write):
                self.policies.append((read, write))
                if (read, write) == (True, True):
                    entered.set()
                    release.wait()
                return 0

        store = Store([], [])
        manager = MooncakeDfsHealthManager(store, defer_policy_application=True)
        self.addCleanup(manager.close)
        self.addCleanup(release.set)

        def publish():
            manager.apply_effective_policy(True, True)
            submitted.set()

        publisher = threading.Thread(target=publish, daemon=True)
        publisher.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(submitted.wait(1), "scheduler waited for a native RPC")
            worker = manager._policy_thread
            for _ in range(10):
                manager.apply_effective_policy(False, True)
                manager.apply_effective_policy(True, False)
            manager.apply_effective_policy(False, False)
            self.assertIs(manager._policy_thread, worker)
            self.assertEqual(store.policies, [(False, False), (True, True)])
            # Rank-local probes remain independent of a pending policy RPC.
            self.assertEqual(manager.local_policy(), (False, False))
            release.set()
            self._wait_for_policy(manager, (False, False))
            self.assertEqual(
                store.policies, [(False, False), (True, True), (False, False)]
            )
        finally:
            release.set()
            publisher.join(timeout=1)

    def test_deferred_policy_retries_without_scheduler_or_probe_calls(self):
        """A failed native update must recover even when no scheduler polls follow."""
        failed = threading.Event()

        class Store(_FakeStore):
            def set_dfs_access_policy(self, read, write):
                self.policies.append((read, write))
                if len(self.policies) == 2:
                    failed.set()
                    raise RuntimeError("lost native reply")
                return 0

        store = Store([], [])
        manager = MooncakeDfsHealthManager(
            store, {"probe_interval_s": 1}, defer_policy_application=True
        )
        self.addCleanup(manager.close)
        self.assertFalse(manager.apply_effective_policy(True, False))
        self.assertTrue(failed.wait(1))
        self._wait_for_policy(manager, (True, False))
        self.assertEqual(store.policies, [(False, False), (True, False), (True, False)])

    def test_deferred_close_is_bounded_with_a_stuck_native_policy(self):
        """Closing must not join an uninterruptible native RPC indefinitely."""
        entered = threading.Event()
        release = threading.Event()
        closed = threading.Event()

        class Store(_FakeStore):
            def set_dfs_access_policy(self, read, write):
                self.policies.append((read, write))
                if read:
                    entered.set()
                    release.wait()
                return 0

        store = Store([], [])
        manager = MooncakeDfsHealthManager(store, defer_policy_application=True)
        self.addCleanup(manager.close)
        self.addCleanup(release.set)
        publisher = threading.Thread(
            target=lambda: manager.apply_effective_policy(True, True), daemon=True
        )

        def close():
            manager.close()
            closed.set()

        closer = threading.Thread(target=close, daemon=True)
        publisher.start()
        try:
            self.assertTrue(entered.wait(1))
            closer.start()
            self.assertTrue(closed.wait(2), "close waited for a stuck native RPC")
            self.assertFalse(release.is_set())
            self.assertTrue(manager._policy_thread.is_alive())
            self.assertFalse(manager.apply_effective_policy(False, False))
        finally:
            release.set()
            publisher.join(timeout=1)
            if closer.ident is not None:
                closer.join(timeout=1)
        manager._policy_thread.join(timeout=1)
        self.assertFalse(manager._policy_thread.is_alive())
        self.assertEqual(store.policies, [(False, False), (True, True)])

    def test_stuck_read_does_not_block_or_fail_write_probe(self):
        release_read = threading.Event()

        class Store(_FakeStore):
            def __init__(self):
                super().__init__([], [0, 0])
                self.read_calls = 0

            def probe_dfs_read(self):
                self.read_calls += 1
                release_read.wait()
                return 0

        store = Store()
        self.addCleanup(release_read.set)
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 1,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
        )

        started_at = time.monotonic()
        snapshot = manager.probe_once()

        self.assertLess(time.monotonic() - started_at, 1.5)
        self.assertEqual(snapshot.read_state, UNAVAILABLE)
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(store.policies[-1], (False, True))
        snapshot = manager.probe_once()
        self.assertEqual(store.read_calls, 1)
        self.assertEqual(snapshot.write_state, AVAILABLE)

    def test_stuck_write_does_not_block_or_fail_read_probe(self):
        release_write = threading.Event()

        class Store(_FakeStore):
            def __init__(self):
                super().__init__([0, 0], [])
                self.write_calls = 0

            def probe_dfs_write(self):
                self.write_calls += 1
                release_write.wait()
                return 0

        store = Store()
        self.addCleanup(release_write.set)
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 1,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
        )

        started_at = time.monotonic()
        snapshot = manager.probe_once()

        self.assertLess(time.monotonic() - started_at, 1.5)
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, UNAVAILABLE)
        self.assertEqual(store.policies[-1], (True, False))
        snapshot = manager.probe_once()
        self.assertEqual(store.write_calls, 1)
        self.assertEqual(snapshot.read_state, AVAILABLE)

    def test_late_probe_success_does_not_reset_timeout_failures(self):
        release_probes = [threading.Event(), threading.Event()]
        first_read_finished = threading.Event()
        first_write_finished = threading.Event()

        class Store(_FakeStore):
            def __init__(self):
                super().__init__([], [])
                self.read_calls = 0
                self.write_calls = 0

            def _probe(self, capability):
                if capability == "read":
                    call_index = self.read_calls
                    self.read_calls += 1
                else:
                    call_index = self.write_calls
                    self.write_calls += 1
                release_probes[call_index].wait()
                if call_index == 0:
                    if capability == "read":
                        first_read_finished.set()
                    else:
                        first_write_finished.set()
                return 0

            def probe_dfs_read(self):
                return self._probe("read")

            def probe_dfs_write(self):
                return self._probe("write")

        store = Store()
        self.addCleanup(release_probes[0].set)
        self.addCleanup(release_probes[1].set)
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 2,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
        )

        first = manager.probe_once()
        self.assertEqual(first.read_failures, 1)
        self.assertEqual(first.write_failures, 1)

        release_probes[0].set()
        self.assertTrue(first_read_finished.wait(1))
        self.assertTrue(first_write_finished.wait(1))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with manager._probe_lock:
                if all(call is None for call in manager._probe_calls.values()):
                    break
            time.sleep(0.01)
        else:
            self.fail("late probe calls did not release their in-flight slots")

        second = manager.probe_once()
        self.assertEqual(store.read_calls, 2)
        self.assertEqual(store.write_calls, 2)
        self.assertEqual(second.read_state, UNAVAILABLE)
        self.assertEqual(second.write_state, UNAVAILABLE)
        self.assertEqual(second.read_failures, 2)
        self.assertEqual(second.write_failures, 2)

    def test_failed_policy_application_is_retried_without_a_health_transition(self):
        class Store(_FakeStore):
            failures = 1

            def set_dfs_access_policy(self, read, write):
                self.policies.append((read, write))
                if read and write and self.failures:
                    self.failures -= 1
                    return -1
                return 0

        store = Store([0, 0], [0, 0])
        manager = MooncakeDfsHealthManager(store, {"success_threshold": 1})
        with self.assertLogs(level="WARNING"):
            manager.probe_once()
        self.assertTrue(manager._policy_dirty)
        manager.probe_once()
        self.assertFalse(manager._policy_dirty)
        self.assertEqual(store.policies[-2:], [(True, True), (True, True)])

    def test_coordinated_failed_reply_reapplies_even_when_target_returns_to_old_policy(
        self,
    ):
        """A lost disable reply must not make re-enable skip its native update."""
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                disabled = threading.Event()

                class Store(_FakeStore):
                    def set_dfs_access_policy(self, read, write):
                        self.policies.append((read, write))
                        if not read and not write and len(self.policies) > 1:
                            # The native side applied the disable, but its reply
                            # was lost. Returning to the old target needs an RPC.
                            disabled.set()
                            return -1
                        return 0

                store = Store([], [])
                manager = MooncakeDfsHealthManager(
                    store,
                    {"probe_interval_s": 1},
                    defer_policy_application=deferred,
                )
                self.addCleanup(manager.close)
                manager.apply_effective_policy(True, True)
                self._wait_for_policy(manager, (True, True))
                self.assertFalse(manager.apply_effective_policy(False, False))
                self.assertTrue(disabled.wait(1))
                self.assertEqual(manager._effective_policy, (True, True))
                self.assertTrue(manager._policy_dirty)
                manager.apply_effective_policy(True, True)
                self._wait_for_policy(manager, (True, True))
                self.assertEqual(store.policies[-2:], [(False, False), (True, True)])


if __name__ == "__main__":
    unittest.main()
