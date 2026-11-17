"""Reject an external cache configuration that disables all prefix matches."""

import unittest

from sglang.srt.arg_groups.hicache_hook import handle_hicache
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestExternalLinkerRadixGate(CustomTestCase):
    def test_linker_requires_radix_cache(self):
        args = ServerArgs(
            model_path="dummy",
            enable_unified_cache_external_linker=True,
            disable_radix_cache=True,
        )
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            handle_hicache(args)

    def test_linker_with_radix_cache_is_valid(self):
        handle_hicache(
            ServerArgs(model_path="dummy", enable_unified_cache_external_linker=True)
        )

    def test_disabling_radix_without_linker_is_valid(self):
        handle_hicache(ServerArgs(model_path="dummy", disable_radix_cache=True))


if __name__ == "__main__":
    unittest.main()
