import importlib.util
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
UNKNOWN = _MODULE.UNKNOWN
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


register_cpu_ci(est_time=13, suite="base-a-test-cpu")


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
                "probe_interval_s": 5,
                "failure_threshold": 2,
                "success_threshold": 2,
                "probe_timeout_s": 5,
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

    def _storage_controller(self, storage=None):
        from sglang.srt.managers.cache_controller import HiCacheController

        controller = object.__new__(HiCacheController)
        controller.enable_storage = storage is not None
        controller.storage_backend = storage
        controller.storage_backend_type = "mooncake" if storage is not None else None
        controller.storage_stop_event = threading.Event()
        controller.prefetch_hits_sync_groups = []
        controller.prefetch_completion_sync_groups = []
        controller.tp_group = controller.attn_cp_group = controller.attn_tp_group = None
        controller.pp_group = None
        controller.page_size = 16
        controller.host_memory_mode = "cache"
        controller.mem_pool_host = SimpleNamespace(size=1024)
        controller.storage_host_pool = controller.mem_pool_host
        controller.page_get_func = controller._generic_page_get
        controller.page_set_func = controller._generic_page_set
        # Exercise real thread start/stop; native I/O bodies wait for shutdown.
        controller.prefetch_thread_func = controller.storage_stop_event.wait
        controller.prefetch_io_aux_func = controller.storage_stop_event.wait
        controller.prefetch_sync_thread_func = controller.storage_stop_event.wait
        controller.backup_thread_func = controller.storage_stop_event.wait
        return controller

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
        self.assertEqual(store.policies[-1], (False, False))

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
        self.assertEqual(store.policies[-1], (False, False))

        manager.probe_once()
        snapshot = manager.probe_once()
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(store.policies[-1], (True, True))

    def test_write_failure_cannot_be_overridden_by_a_read_only_policy(self):
        """A successful read probe cannot reopen DFS while writes are unhealthy."""
        store, manager = self._manager([0, 0], [-1, -1])
        manager.probe_once()
        manager.probe_once()
        self.assertEqual(manager.snapshot().read_state, AVAILABLE)
        self.assertEqual(manager.local_policy(), (False, False))
        manager.apply_effective_policy(True, False)
        self.assertEqual(store.policies[-1], (False, False))

    def test_data_timeout_trips_policy_and_recovers_only_after_fresh_probes(self):
        """Business I/O failures bypass the failure threshold, not recovery hysteresis."""
        for capability, error, disabled_policy in (
            ("read", -1607, (False, True)),
            ("write", -1606, (False, False)),
        ):
            with self.subTest(capability=capability):
                store, manager = self._manager([0] * 4, [0] * 4)
                self.addCleanup(manager.close)
                manager.probe_once()
                manager.probe_once()
                self.assertEqual(manager.local_policy(), (True, True))

                manager.report_operation_timeout(capability, error)
                snapshot = manager.snapshot()
                self.assertEqual(getattr(snapshot, f"{capability}_state"), UNAVAILABLE)
                self.assertEqual(getattr(snapshot, f"{capability}_error"), error)
                self.assertEqual(manager.local_policy(), disabled_policy)
                # A delayed collective/probe result must not reopen the failed capability.
                manager.apply_effective_policy(True, True)
                self._wait_for_policy(manager, disabled_policy)

                manager.probe_once()
                self.assertEqual(manager.local_policy(), disabled_policy)
                manager.probe_once()
                self._wait_for_policy(manager, (True, True))
                self.assertEqual(store.policies[-1], (True, True))

    def test_data_timeout_never_reopens_consensus_disabled_capability(self):
        """A rank-local timeout can restrict, but cannot undo a peer's disable."""
        store, manager = self._manager([0, 0], [0, 0])
        self.addCleanup(manager.close)
        manager.probe_once()
        manager.probe_once()
        manager.apply_effective_policy(False, False)
        manager.report_operation_timeout("read", -1607)
        self._wait_for_policy(manager, (False, False))
        self.assertEqual(manager.local_policy(), (False, True))
        self.assertEqual(store.policies[-1], (False, False))

    def test_data_timeout_coalesces_a_pending_enable_without_blocking(self):
        """Timeout cleanup remains responsive while a native policy RPC is stuck."""
        entered = threading.Event()
        release = threading.Event()

        class Store(_FakeStore):
            def set_dfs_access_policy(self, read, write):
                self.policies.append((read, write))
                if read and write:
                    entered.set()
                    release.wait()
                return 0

        store = Store([], [])
        manager = MooncakeDfsHealthManager(store, defer_policy_application=True)
        self.addCleanup(manager.close)
        self.addCleanup(release.set)
        manager.apply_effective_policy(True, True)
        self.assertTrue(entered.wait(1))
        manager.report_operation_timeout("write", -1606)
        self.assertEqual(manager.local_policy(), (False, False))
        self.assertEqual(manager._desired_policy, (False, False))
        self.assertEqual(store.policies, [(False, False), (True, True)])
        release.set()
        self._wait_for_policy(manager, (False, False))
        self.assertEqual(store.policies[-1], (False, False))

    def test_probe_started_before_data_timeout_cannot_count_towards_recovery(self):
        """A late successful probe must not erase a newer business-I/O failure."""
        entered = threading.Event()
        release = threading.Event()

        class Store(_FakeStore):
            read_calls = 0

            def probe_dfs_read(self):
                self.read_calls += 1
                if self.read_calls == 3:
                    entered.set()
                    release.wait()
                return 0

        store = Store([], [0] * 5)
        manager = MooncakeDfsHealthManager(store, {"success_threshold": 2})
        self.addCleanup(manager.close)
        self.addCleanup(release.set)
        manager.probe_once()
        manager.probe_once()
        probe = threading.Thread(target=manager.probe_once, daemon=True)
        probe.start()
        self.assertTrue(entered.wait(1))
        manager.report_operation_timeout("write", -1606)
        release.set()
        probe.join(timeout=1)
        self.assertFalse(probe.is_alive())
        self.assertEqual(manager.snapshot().write_successes, 0)
        self.assertEqual(manager.local_policy(), (False, False))
        manager.probe_once()
        self.assertEqual(manager.local_policy(), (False, False))
        manager.probe_once()
        self.assertEqual(manager.local_policy(), (True, True))

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

        manager.apply_effective_policy(False, True)
        self._wait_for_policy(manager, (False, True))
        manager.apply_effective_policy(False, True)
        self.assertEqual(store.policies, [(False, False), (False, True)])

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
        self.assertFalse(manager.apply_effective_policy(False, True))
        self.assertTrue(failed.wait(1))
        self._wait_for_policy(manager, (False, True))
        self.assertEqual(store.policies, [(False, False), (False, True), (False, True)])

    def test_close_refuses_a_stuck_native_policy_in_both_application_modes(self):
        """An unfinished policy RPC must retain its client, including sync callers."""
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                entered = threading.Event()
                release = threading.Event()
                closed = threading.Event()
                errors = []

                class Store(_FakeStore):
                    def set_dfs_access_policy(self, read, write):
                        self.policies.append((read, write))
                        if read:
                            entered.set()
                            release.wait()
                        return 0

                store = Store([], [])
                manager = MooncakeDfsHealthManager(
                    store, defer_policy_application=deferred
                )
                self.addCleanup(manager.close)
                self.addCleanup(release.set)
                publisher = threading.Thread(
                    target=lambda: manager.apply_effective_policy(True, True),
                    daemon=True,
                )

                def close():
                    try:
                        manager.close()
                    except RuntimeError as exc:
                        errors.append(str(exc))
                    finally:
                        closed.set()

                closer = threading.Thread(target=close, daemon=True)
                publisher.start()
                try:
                    self.assertTrue(entered.wait(1))
                    closer.start()
                    self.assertTrue(closed.wait(2), "close waited for a stuck RPC")
                    self.assertEqual(len(errors), 1)
                    self.assertIn("close incomplete", errors[0])
                    self.assertIs(manager._store, store)
                    self.assertFalse(release.is_set())
                    self.assertFalse(manager.apply_effective_policy(False, False))
                    self.assertEqual(manager.local_policy(), (False, False))
                finally:
                    release.set()
                    publisher.join(timeout=1)
                    if closer.ident is not None:
                        closer.join(timeout=1)
                manager.close()
                self.assertEqual(store.policies, [(False, False), (True, True)])

    def test_probe_close_refusal_retains_storage_owner_until_drain(self):
        """A hung probe cannot permit storage close or spawn replacement workers."""
        from sglang.srt.mem_cache.storage import StorageBackendFactory
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )
        from sglang.srt.mem_cache.unified_cache.storage_attachment import (
            StorageAttachment,
        )

        entered = {capability: threading.Event() for capability in ("read", "write")}
        release = threading.Event()

        class Store(_FakeStore):
            def __init__(self):
                super().__init__([], [])
                self.calls = {"read": 0, "write": 0}

            def _probe(self, capability):
                self.calls[capability] += 1
                if self.calls[capability] > 1:
                    entered[capability].set()
                    release.wait()
                return 0

            def probe_dfs_read(self):
                return self._probe("read")

            def probe_dfs_write(self):
                return self._probe("write")

        store = Store()
        manager = MooncakeDfsHealthManager(
            store,
            {"probe_timeout_s": 1, "success_threshold": 1},
        )
        # Use the real storage close boundary; only the native client is fake.
        storage = object.__new__(MooncakeStore)
        storage.store = store
        storage._client_metrics_reporter = None
        storage._dfs_health_manager = manager
        self.addCleanup(storage.close)
        self.addCleanup(release.set)
        controller = self._storage_controller(storage)
        controller._start_storage_threads()
        self.addCleanup(controller.detach_storage_backend)
        self.addCleanup(release.set)
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=True,
            enable_storage_metrics=False,
            drain_storage_control_queues_local=lambda: None,
            ongoing_prefetch={},
            ongoing_backup={},
            _storage_prefetch_hit_remaining_by_reqid={},
            prefetch_loaded_tokens_by_reqid={},
            prefetch_loaded_storage_start_by_reqid={},
            storage_prefetch_retries={},
            _all_reduce_attn_groups=lambda tensor, op: None,
        )
        attachment = StorageAttachment(cache)
        manager.probe_once()
        self.assertEqual(manager.local_policy(), (True, True))
        manager.start()
        self.assertTrue(entered["read"].wait(1))
        self.assertTrue(entered["write"].wait(1))
        with manager._probe_lock:
            workers = [call.thread for call in manager._probe_calls.values()]
        health_worker = manager._thread

        replacement_store = _FakeStore([0], [0])
        replacement = MooncakeDfsHealthManager(replacement_store)
        replacement_storage = object.__new__(MooncakeStore)
        replacement_storage.store = replacement_store
        replacement_storage._client_metrics_reporter = None
        replacement_storage._dfs_health_manager = replacement
        self.addCleanup(replacement_storage.close)
        groups = [object(), object()]
        controller.prefetch_hits_sync_groups = groups[:1]
        controller.prefetch_completion_sync_groups = groups[1:]
        destroyed = []
        with (
            patch.object(
                StorageBackendFactory,
                "create_backend",
                return_value=replacement_storage,
            ) as factory,
            patch.object(
                controller,
                "_generate_storage_config",
                return_value=SimpleNamespace(is_mla_model=False, tp_rank=0),
            ),
            patch(
                "torch.distributed.destroy_process_group", side_effect=destroyed.append
            ),
        ):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "detach incomplete"):
                    attachment.detach()
                self.assertFalse(cache.enable_storage)
                self.assertFalse(controller.enable_storage)
                self.assertIs(controller.storage_backend, storage)
                self.assertIs(storage._dfs_health_manager, manager)
                self.assertIs(storage.store, store)
                self.assertIs(manager._thread, health_worker)
                self.assertTrue(all(worker.is_alive() for worker in workers))
                self.assertEqual(store.calls, {"read": 2, "write": 2})
                self.assertEqual(destroyed, [])
                self.assertTrue(controller.storage_stop_event.is_set())
                self.assertEqual(manager.local_policy(), (False, False))
                self.assertEqual(manager.snapshot().read_state, UNKNOWN)
                self.assertEqual(manager.snapshot().write_state, UNKNOWN)
                for operation in (manager.start, manager.probe_once):
                    with self.assertRaisesRegex(RuntimeError, "closing or closed"):
                        operation()
                self.assertFalse(manager.apply_effective_policy(True, True))
                with self.assertRaisesRegex(RuntimeError, "cleanup is incomplete"):
                    controller.attach_storage_backend("mooncake")
                self.assertFalse(attachment.attach("mooncake")[0])
                self.assertEqual(factory.call_count, 0)

            # Shutdown must retry a retained owner even with storage disabled.
            attachment.shutdown()
            self.assertIs(controller.storage_backend, storage)
            self.assertEqual(factory.call_count, 0)
            release.set()
            self.assertTrue(attachment.detach()[0])
            self.assertIsNone(controller.storage_backend)
            self.assertIsNone(storage._dfs_health_manager)
            self.assertEqual(destroyed, groups)
            self.assertFalse(controller.storage_stop_event.is_set())
            for worker in workers:
                worker.join(timeout=1)
                self.assertFalse(worker.is_alive())
            self.assertFalse(health_worker.is_alive())
            self.assertEqual(manager.snapshot().read_state, UNKNOWN)
            self.assertEqual(manager.snapshot().write_state, UNKNOWN)
            # A local UNKNOWN does not confirm a native disable.
            self.assertEqual(store.policies, [(False, False), (True, True)])

            controller.attach_storage_backend("mooncake")
            self.assertEqual(factory.call_count, 1)
            self.assertIs(controller.storage_backend, replacement_storage)
            self.assertTrue(controller.enable_storage)
            self.assertEqual(replacement.local_policy(), (False, False))
            self.assertEqual(replacement.snapshot().read_state, UNKNOWN)
            self.assertEqual(replacement_store.policies, [(False, False)])
            controller.detach_storage_backend()

    def test_attach_rollback_retains_backend_when_native_close_refuses(self):
        """Failed initialization must not drop a client that still owns native I/O."""
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        drained = threading.Event()

        class Backend:
            def register_mem_pool_host(self, pool):
                raise ValueError("host registration failed")

            def close(self):
                if not drained.is_set():
                    raise RuntimeError("native operation still pending")

        backend = Backend()
        controller = self._storage_controller()
        self.addCleanup(controller.detach_storage_backend)
        self.addCleanup(drained.set)
        with (
            patch.object(
                StorageBackendFactory, "create_backend", return_value=backend
            ) as factory,
            patch.object(
                controller,
                "_generate_storage_config",
                return_value=SimpleNamespace(is_mla_model=False, tp_rank=0),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "attach failed and cleanup"):
                controller.attach_storage_backend("mooncake")
            self.assertIs(controller.storage_backend, backend)
            self.assertEqual(controller.storage_backend_type, "mooncake")
            self.assertFalse(controller.enable_storage)
            self.assertTrue(controller.storage_stop_event.is_set())
            with self.assertRaisesRegex(RuntimeError, "cleanup is incomplete"):
                controller.attach_storage_backend("mooncake")
            self.assertEqual(factory.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, "close incomplete"):
                controller.detach_storage_backend()
            self.assertIs(controller.storage_backend, backend)
            drained.set()
            controller.detach_storage_backend()
            self.assertIsNone(controller.storage_backend)
            self.assertIsNone(controller.storage_backend_type)
            self.assertFalse(controller.storage_stop_event.is_set())

    def test_partial_detach_keeps_pp_stages_and_world_reattach_in_step(self):
        """PP=2/TP=1 must refuse world group creation when one stage retains a client."""
        import torch

        from sglang.srt.mem_cache.storage import StorageBackendFactory
        from sglang.srt.mem_cache.unified_cache.storage_attachment import (
            StorageAttachment,
        )
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        drained = threading.Event()
        self.addCleanup(drained.set)
        policies = []

        class Backend:
            def __init__(self, rank):
                self.rank = rank

            def close(self):
                if self.rank == 0 and not drained.is_set():
                    raise RuntimeError("native probe still pending")

            def get_local_dfs_health_policy(self):
                return False, False

            def apply_dfs_health_policy(self, read, write):
                policies.append((self.rank, read, write))

        # Model the external PP completion exchange and reject different operation
        # order, shapes or dtypes instead of silently pairing them.
        barrier = threading.Barrier(2, timeout=3)
        buffers = [None, None]
        traces = [[], []]

        def reduce(rank, tensor, op):
            signature = (tuple(tensor.shape), tensor.dtype, op)
            traces[rank].append(signature)
            buffers[rank] = (tensor, signature)
            barrier.wait()
            if buffers[0][1] != buffers[1][1]:
                raise AssertionError(f"collective mismatch: {buffers}")
            values = torch.stack([entry[0].clone() for entry in buffers])
            if op == torch.distributed.ReduceOp.MIN:
                tensor.copy_(values.min(dim=0).values)
            else:
                self.assertEqual(op, torch.distributed.ReduceOp.MAX)
                tensor.copy_(values.max(dim=0).values)
            barrier.wait()

        caches = []
        attachments = []
        for rank in range(2):
            controller = self._storage_controller(Backend(rank))
            controller.write_queue = []
            controller.ack_write_queue = []
            controller.ack_load_queue = []
            controller.prefetch_hit_queue = None
            self.addCleanup(controller.detach_storage_backend)
            cache = object.__new__(UnifiedRadixCache)
            cache.tree_core = SimpleNamespace(
                enable_storage=True, write_back_duplicate_reclaim_digest=0
            )
            cache.cache_controller = controller
            cache.linker = None
            cache._dfs_policy_sync_enabled = True
            cache.host_memory_mode = "cache"
            cache.pp_rank = rank
            cache.pp_size = 2
            cache.tp_world_size = 1
            cache.attn_cp_group = cache.attn_tp_group = None
            cache.work_list = []
            cache.buffer_pipeline = None
            cache.enable_storage_metrics = False
            cache.ongoing_prefetch = {}
            cache.ongoing_backup = {}
            cache._storage_prefetch_hit_remaining_by_reqid = {}
            cache.prefetch_loaded_tokens_by_reqid = {}
            cache.prefetch_loaded_storage_start_by_reqid = {}
            cache.storage_prefetch_retries = {}
            cache._all_reduce = lambda tensor, op, rank=rank: reduce(rank, tensor, op)
            caches.append(cache)
            attachments.append(StorageAttachment(cache))
        self.addCleanup(drained.set)

        with self.assertRaisesRegex(RuntimeError, "detach incomplete"):
            attachments[0].detach()
        self.assertTrue(attachments[1].detach()[0])
        self.assertIsNotNone(caches[0].cache_controller.storage_backend)
        self.assertIsNone(caches[1].cache_controller.storage_backend)
        self.assertTrue(all(not cache.enable_storage for cache in caches))

        def run_ranks(action):
            results = [None, None]
            errors = []

            def run(rank):
                try:
                    results[rank] = action(rank)
                except BaseException as exc:
                    errors.append(exc)
                    barrier.abort()

            workers = [
                threading.Thread(target=run, args=(rank,), daemon=True)
                for rank in range(2)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertFalse(errors, repr(errors))
            return results

        for _ in range(2):
            run_ranks(lambda rank: caches[rank].check_hicache_events())
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(len(traces[0]), 2)
        self.assertTrue(all(signature[0] == (4,) for signature in traces[0]))
        self.assertEqual(policies, [])
        world_cpu_group = object()
        world_inputs = []
        world_lock = threading.Lock()

        def world_reduce(tensor, op, group):
            self.assertIs(group, world_cpu_group)
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            self.assertEqual(tensor.device.type, "cpu")
            with world_lock:
                world_inputs.append(tensor)
            barrier.wait()
            tensor.copy_(torch.stack(world_inputs).max(dim=0).values)
            barrier.wait()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "sglang.srt.mem_cache.unified_cache.storage_attachment.get_world_group",
                return_value=SimpleNamespace(cpu_group=world_cpu_group),
            ),
            patch(
                "torch.distributed.all_reduce", side_effect=world_reduce
            ) as world_sync,
            patch.object(StorageBackendFactory, "create_backend") as factory,
            patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group"
            ) as create_group,
        ):
            attempts = run_ranks(lambda rank: attachments[rank].attach("mooncake"))
            self.assertTrue(all(not success for success, _ in attempts))
            self.assertEqual(world_sync.call_count, 2)
            self.assertEqual(len(world_inputs), 2)
            self.assertEqual(factory.call_count, 0)
            self.assertEqual(create_group.call_count, 0)
        drained.set()
        self.assertTrue(all(attachment.detach()[0] for attachment in attachments))
        run_ranks(lambda rank: caches[rank].check_hicache_events())
        self.assertTrue(
            all(cache.cache_controller.storage_backend is None for cache in caches)
        )
        self.assertEqual(traces[0], traces[1])

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
        self.assertEqual(store.policies[-1], (False, False))
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
