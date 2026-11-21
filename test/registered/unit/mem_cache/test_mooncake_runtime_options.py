import argparse
import inspect
import unittest

from sglang.srt.arg_groups.arg_utils import namespace_of
from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store.dfs_health import (
    resolve_dfs_health_config,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    _local_buffer_size,
)
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMooncakeRuntimeOptions(CustomTestCase):
    def test_local_buffer_size_rejects_negative_values(self):
        """A negative Mooncake buffer budget must fail before native setup."""
        for value in (1 << 30, 0):
            with self.subTest(value=value):
                with envs.SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE.override(value):
                    self.assertEqual(_local_buffer_size(), value)

        with envs.SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE.override(-1):
            with self.assertRaisesRegex(ValueError, "must be nonnegative"):
                _local_buffer_size()
        with envs.SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE.override("garbage"):
            with self.assertRaises(ValueError):
                _local_buffer_size()

    def test_new_options_do_not_shift_existing_positional_arguments(self):
        """Mooncake options must not intercept positional backend arguments."""
        parameters = list(inspect.signature(ServerArgs).parameters)
        linker = parameters.index("enable_unified_cache_external_linker")
        self.assertEqual(
            parameters[linker + 1], "unified_cache_external_linker_backend"
        )
        self.assertGreater(
            parameters.index("mooncake_enable_page_wise_load"),
            parameters.index("grpc_response_timeout_secs"),
        )
        self.assertGreater(
            parameters.index("unified_cache_external_linker_config"),
            parameters.index("mooncake_dfs_health_probe_timeout_s"),
        )

    def test_launch_flags_reach_the_memory_namespace(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-config",
                '{"memory_replica_num":0,"dfs_replica_num":1}',
                "--mooncake-enable-page-wise-load",
                "--mooncake-page-wise-load-threshold",
                "4",
                "--mooncake-enable-waiting-queue-dfs-prefetch",
                "--mooncake-waiting-queue-dfs-prefetch-workers",
                "1",
                "--mooncake-waiting-queue-dfs-prefetch-max-requests",
                "12",
                "--mooncake-dfs-health-probe-timeout-s",
                "7",
            ]
        )
        expected = {
            "unified_cache_external_linker_config": (
                '{"memory_replica_num":0,"dfs_replica_num":1}'
            ),
            "mooncake_enable_page_wise_load": True,
            "mooncake_page_wise_load_threshold": 4,
            "mooncake_enable_waiting_queue_dfs_prefetch": True,
            "mooncake_waiting_queue_dfs_prefetch_workers": 1,
            "mooncake_waiting_queue_dfs_prefetch_max_requests": 12,
            "mooncake_dfs_health_probe_timeout_s": 7,
        }
        for name, value in expected.items():
            self.assertEqual(getattr(parsed, name), value)
            self.assertEqual(namespace_of(ServerArgs)[name], "memory")
        with get_context().override_server_args(**expected):
            for name, value in expected.items():
                self.assertEqual(getattr(get_memory(), name), value)
            with envs.SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S.override(99):
                self.assertEqual(
                    resolve_dfs_health_config(server_args=get_memory())[
                        "probe_timeout_s"
                    ],
                    7,
                )


if __name__ == "__main__":
    unittest.main()
