import pkgutil
import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2


class TestLegacyPluginHookTargets(unittest.TestCase):
    def test_legacy_targets_resolve_to_live_entry_points(self):
        self.assertIs(
            pkgutil.resolve_name(
                "sglang.srt.speculative.eagle_worker.EAGLEWorker"
            ),
            EAGLEWorkerV2,
        )
        self.assertTrue(
            callable(
                pkgutil.resolve_name(
                    "sglang.srt.server_args._resolve_speculative_algorithm_alias"
                )
            )
        )
        self.assertTrue(
            callable(
                pkgutil.resolve_name(
                    "sglang.srt.server_args.ServerArgs._handle_speculative_decoding"
                )
            )
        )

    def test_handler_uses_the_plugin_compatible_resolver(self):
        server_args = object.__new__(ServerArgs)
        with patch.object(
            server_args_module, "_handle_speculative_decoding_impl"
        ) as implementation:
            server_args._handle_speculative_decoding()

        implementation.assert_called_once_with(
            server_args,
            resolve_speculative_algorithm_alias=(
                server_args_module._resolve_speculative_algorithm_alias
            ),
        )


if __name__ == "__main__":
    unittest.main()
