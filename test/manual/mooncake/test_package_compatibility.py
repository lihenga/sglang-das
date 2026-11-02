"""CPU-only API negotiation checks; no Torch, native wheel, or master needed.

Run: python test/manual/mooncake/test_package_compatibility.py
The fake APIs follow the specified community, HYGON L3, and L4 signatures.
"""

import __future__

import ast
import importlib.util
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Load the dependency-free negotiator without importing SGLang's GPU runtime.
_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/storage/mooncake_store/compat.py"
)
_SPEC = importlib.util.spec_from_file_location("mooncake_compat", _PATH)
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)


class CommunityStore:
    def setup(
        self,
        local_hostname,
        metadata_server,
        global_segment_size,
        local_buffer_size,
        protocol,
        rdma_devices,
        master_server_addr,
        engine=None,
        enable_ssd_offload=False,
        ssd_offload_path="",
        tenant_id="default",
        enable_client_http_server=False,
        client_http_port=9300,
    ):
        self.options = {
            "tenant_id": tenant_id,
            "enable_ssd_offload": enable_ssd_offload,
            "ssd_offload_path": ssd_offload_path,
            "enable_client_http_server": enable_client_http_server,
            "client_http_port": client_http_port,
        }
        return 0


class L3Store(CommunityStore):
    def batch_get_into_layouts(self, layouts):
        pass


class L4Store(L3Store):
    def __init__(self):
        self.releases = 0

    def setup(self, *args, enable_dfs_prefetch=True, **kwargs):
        self.prefetch = enable_dfs_prefetch
        return super().setup(*args, **kwargs)

    def get_capabilities(self):
        return {
            "setup_parameters": list(
                compat.inspect.signature(CommunityStore.setup).parameters
            )
            + ["enable_dfs_prefetch"],
            "features": {"dfs_prefetch": True, "dfs_health": True},
        }

    def disable_dfs_prefetch(self):
        self.releases += 1
        return True

    probe_dfs_read = probe_dfs_write = set_dfs_access_policy = lambda *args: 0
    batch_get_session_start_with_sources = lambda *args: None
    batch_get_session_prefetch = batch_get_session_refresh = lambda *args: None
    dfs_prefetch_arena_available = lambda *args: True
    create_read_plan = lambda *args: None


_ARGS = ("127.0.0.1:0", "P2PHANDSHAKE", 1024, 0, "tcp", "", "127.0.0.1:50051", None)


class TestPackageCompatibility(unittest.TestCase):
    def test_real_store_constructor_accepts_legacy_distributed_packages(self):
        # Exercise the actual constructor, isolating Torch/native allocation.
        # Only imports and external services are replaced; setup negotiation,
        # startup policy selection, and the order before warmup are real code.
        source = _PATH.with_name("mooncake_store.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        definition = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MooncakeStore"
        )
        for package in (CommunityStore, L3Store):
            for prefetch in (False, True):
                with self.subTest(package=package.__name__, prefetch=prefetch):
                    native = package()

                    class Base:
                        def _import_mooncake_store(self):
                            return lambda: native

                        def _import_mooncake_group_semantics(self):
                            return None, False

                        def _load_config(self, config):
                            return SimpleNamespace(
                                global_segment_size=1024,
                                check_server=False,
                                standalone_storage=False,
                                device_name="",
                                local_hostname="127.0.0.1:0",
                                metadata_server="P2PHANDSHAKE",
                                protocol="tcp",
                                master_server_address="127.0.0.1:50051",
                                enable_ssd_offload=False,
                                ssd_offload_path=None,
                                tenant_id="default",
                            )

                    namespace = {
                        "HiCacheStorage": type("HiCacheStorage", (), {}),
                        "MooncakeBaseStore": Base,
                        "DEFAULT_TENANT_ID": "default",
                        "DEFAULT_LOCAL_BUFFER_SIZE": 0,
                        "logger": logging.getLogger("compat_startup_test"),
                        "setup_store": compat.setup_store,
                        "supports_feature": compat.supports_feature,
                        "_get_mooncake_client_http_setup_kwargs": lambda *a: {},
                        "get_server_args": Mock(side_effect=ValueError("no server")),
                        "_get_service_endpoint": lambda args: None,
                        "os": os,
                    }
                    module = ast.Module(body=[definition], type_ignores=[])
                    exec(
                        compile(
                            module,
                            str(source),
                            "exec",
                            flags=__future__.annotations.compiler_flag,
                        ),
                        namespace,
                    )
                    store_cls = namespace["MooncakeStore"]
                    store_cls.warmup = Mock()
                    config = SimpleNamespace(
                        extra_config={},
                        tp_rank=0,
                        tp_size=1,
                        pp_rank=0,
                        pp_size=1,
                        attn_cp_rank=0,
                        attn_cp_size=1,
                        is_mla_model=True,
                        enable_storage_metrics=False,
                        model_name="test-model",
                        should_split_heads=False,
                    )
                    with patch.dict(
                        os.environ,
                        {
                            "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR": "distributed_storage_backend"
                        },
                    ):
                        store = store_cls(config, enable_dfs_prefetch=prefetch)
                    store_cls.warmup.assert_called_once_with()
                    self.assertIsNone(store.get_local_dfs_health_policy())
                    self.assertEqual(native.options["tenant_id"], "default")

    def test_setup_matrix_preserves_storage_options(self):
        for cls in (CommunityStore, L3Store, L4Store):
            for metrics in (False, True):
                for prefetch in (False, True):
                    with self.subTest(
                        package=cls.__name__, metrics=metrics, prefetch=prefetch
                    ):
                        store = cls()
                        options = {
                            "tenant_id": "tenant-a",
                            "enable_ssd_offload": True,
                            "ssd_offload_path": "/dfs",
                            "enable_client_http_server": metrics,
                            "client_http_port": 0,
                        }
                        if not prefetch:
                            options["enable_dfs_prefetch"] = False
                        self.assertEqual(compat.setup_store(store, _ARGS, options), 0)
                        self.assertEqual(store.options["tenant_id"], "tenant-a")
                        self.assertTrue(store.options["enable_ssd_offload"])
                        self.assertEqual(store.options["ssd_offload_path"], "/dfs")
                        self.assertEqual(
                            store.options["enable_client_http_server"], metrics
                        )
                        if cls is L4Store:
                            self.assertEqual(store.prefetch, prefetch)

    def test_feature_matrix_without_version_strings(self):
        for cls, layout, l4 in (
            (CommunityStore, False, False),
            (L3Store, True, False),
            (L4Store, True, True),
        ):
            with self.subTest(package=cls.__name__):
                store = cls()
                self.assertEqual(compat.supports_feature(store, "read_layout"), layout)
                for feature in ("dfs_health", "dfs_prefetch", "read_plan"):
                    self.assertEqual(compat.supports_feature(store, feature), l4)
                self.assertEqual(
                    compat.resolve_read_plan_options(store, True, True), (l4, l4)
                )

    def test_standalone_declaration_overrides_callable_but_unsupported_api(self):
        store = L4Store()
        declaration = store.get_capabilities()
        declaration["features"]["dfs_prefetch"] = False
        declaration["features"]["read_plan"] = False
        store.get_capabilities = lambda: declaration
        self.assertFalse(compat.supports_feature(store, "dfs_prefetch"))
        self.assertEqual(
            compat.resolve_read_plan_options(store, True, True), (False, False)
        )

    def test_legacy_disable_is_noop_and_l4_releases_arena(self):
        compat.disable_dfs_prefetch(CommunityStore())
        compat.disable_dfs_prefetch(L3Store())
        store = L4Store()
        compat.disable_dfs_prefetch(store)
        self.assertEqual(store.releases, 1)

    def test_standalone_without_local_arena_does_not_call_unsupported_release(self):
        store = L4Store()
        declaration = store.get_capabilities()
        declaration["client_mode"] = "standalone"
        declaration["features"]["dfs_prefetch"] = False
        store.get_capabilities = lambda: declaration
        store.disable_dfs_prefetch = Mock(return_value=False)
        compat.disable_dfs_prefetch(store)
        store.disable_dfs_prefetch.assert_not_called()

    def test_busy_arena_and_partial_prefetch_do_not_silently_succeed(self):
        store = L4Store()
        store.disable_dfs_prefetch = lambda: False
        with self.assertRaisesRegex(RuntimeError, "requests have started"):
            compat.disable_dfs_prefetch(store)
        store.disable_dfs_prefetch = None
        with self.assertRaisesRegex(RuntimeError, "release its arena"):
            compat.disable_dfs_prefetch(store)

    def test_old_prefetch_package_uses_startup_release_when_gate_is_missing(self):
        class OldPrefetchStore(L4Store):
            setup = CommunityStore.setup
            get_capabilities = None

        store = OldPrefetchStore()
        self.assertEqual(
            compat.setup_store(store, _ARGS, {"enable_dfs_prefetch": False}), 0
        )
        self.assertEqual(store.releases, 1)

    def test_setup_failure_never_calls_release(self):
        store = L4Store()
        store.get_capabilities = None
        store.setup = lambda *args: -200
        self.assertEqual(
            compat.setup_store(store, _ARGS, {"enable_dfs_prefetch": False}), -200
        )
        self.assertEqual(store.releases, 0)

    def test_declared_setup_gate_is_honored_without_prefetch_methods(self):
        class GateOnlyStore(CommunityStore):
            def setup(self, *args, enable_dfs_prefetch=True):
                self.prefetch = enable_dfs_prefetch
                return 0

        store = GateOnlyStore()
        compat.setup_store(store, _ARGS, {"enable_dfs_prefetch": False})
        self.assertFalse(store.prefetch)

    def test_unrelated_typeerror_does_not_drop_options_or_retry(self):
        store = CommunityStore()
        store.setup = Mock(side_effect=TypeError("internal failure with tenant_id"))
        with self.assertRaisesRegex(TypeError, "internal failure"):
            compat.setup_store(store, _ARGS, {"tenant_id": "tenant-a"})
        self.assertEqual(store.setup.call_count, 1)

    def test_nested_keyword_typeerror_does_not_retry_setup(self):
        store = SimpleNamespace(
            setup=Mock(
                side_effect=TypeError(
                    "helper() got an unexpected keyword argument 'client_http_port'"
                )
            )
        )
        with self.assertRaisesRegex(TypeError, "helper"):
            compat.setup_store(store, _ARGS, {"client_http_port": 0})
        self.assertEqual(store.setup.call_count, 1)

    def test_unparseable_binding_error_is_not_treated_as_unsupported_keywords(self):
        store = SimpleNamespace(
            setup=Mock(
                side_effect=TypeError(
                    "setup(): incompatible function arguments. Supported types: unknown\nInvoked with: Store; kwargs: client_http_port=0"
                )
            )
        )
        with self.assertRaises(TypeError):
            compat.setup_store(store, _ARGS, {"client_http_port": 0})
        self.assertEqual(store.setup.call_count, 1)

    def test_unsupported_storage_options_fail_before_setup(self):
        for name, value in (
            ("tenant_id", "tenant-a"),
            ("enable_ssd_offload", True),
            ("ssd_offload_path", "/dfs"),
        ):
            with self.subTest(option=name):
                store = SimpleNamespace(setup=Mock())
                store.get_capabilities = lambda: {"setup_parameters": []}
                with self.assertRaisesRegex(RuntimeError, name):
                    compat.setup_store(store, _ARGS, {name: value})
                store.setup.assert_not_called()

    def test_pybind_error_does_not_confuse_invoked_keywords_with_supported_ones(self):
        calls = []

        def setup(*args, **kwargs):
            calls.append(kwargs)
            if "enable_client_http_server" in kwargs:
                raise TypeError(
                    "setup(): incompatible function arguments. The following argument types are supported:\n"
                    "1. (self: Store, local_hostname: str, tenant_id: str = 'default') -> int\n"
                    "Invoked with: Store; kwargs: tenant_id='tenant-a', enable_client_http_server=True"
                )
            return 0

        self.assertEqual(
            compat.setup_store(
                SimpleNamespace(setup=setup),
                (),
                {"tenant_id": "tenant-a", "enable_client_http_server": True},
            ),
            0,
        )
        self.assertEqual(
            calls,
            [
                {"tenant_id": "tenant-a", "enable_client_http_server": True},
                {"tenant_id": "tenant-a"},
            ],
        )

    def test_python_unexpected_keyword_retries_only_diagnostics(self):
        calls = []

        def setup(*args, **kwargs):
            calls.append(kwargs)
            if "client_http_port" in kwargs:
                raise TypeError(
                    "setup() got an unexpected keyword argument 'client_http_port'"
                )
            return 0

        self.assertEqual(
            compat.setup_store(
                SimpleNamespace(setup=setup), (), {"client_http_port": 0}
            ),
            0,
        )
        self.assertEqual(calls, [{"client_http_port": 0}, {}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
