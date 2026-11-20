"""DFS policy integration across decode offload and unified-cache rank groups."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeDfsPolicyConsensus(CustomTestCase):
    def _owner(self):
        applied = []
        owner = SimpleNamespace(
            get_local_dfs_health_policy=lambda: (True, True),
            apply_dfs_health_policy=lambda read, write: applied.append((read, write)),
        )
        return owner, applied

    def _decode_manager(self, owner):
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        manager.cache_controller = SimpleNamespace(
            storage_backend=owner, ack_write_queue=[], ack_backup_queue=Queue()
        )
        manager.tp_group = "tp"
        manager.attn_cp_group = "cp"
        manager.pp_group = "pp"
        manager.tp_world_size = 2
        manager._dfs_policy_sync_enabled = None
        manager._dfs_policy_failed_target = None
        manager._dfs_policy_consecutive_failures = 0
        manager._dfs_policy_next_retry_at = 0.0
        manager._dfs_policy_last_warning_at = float("-inf")
        manager._dfs_policy_suppressed_warnings = 0
        return manager

    def test_decode_offload_observes_every_group_and_recovers(self):
        """One unhealthy TP/CP/PP peer must disable DFS on an otherwise healthy rank."""
        for failed_group in ("tp", "cp", "pp"):
            with self.subTest(failed_group=failed_group):
                owner, applied = self._owner()
                manager = self._decode_manager(owner)
                unhealthy = True

                def reduce(values, *, op, group):
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    if (
                        unhealthy
                        and group == failed_group
                        and values.tolist() == [1, 1]
                    ):
                        values[1] = 0

                with (
                    patch.object(torch.distributed, "get_world_size", return_value=2),
                    patch.object(torch.distributed, "all_reduce", side_effect=reduce),
                ):
                    manager.check_offload_progress()
                    self.assertEqual(applied[-1], (False, False))
                    unhealthy = False
                    manager.check_offload_progress()
                    self.assertEqual(applied[-1], (True, True))

    def test_decode_startup_rejects_inconsistent_dfs_capability(self):
        """Mismatched health configuration must fail before ranks choose different collectives."""
        owner, applied = self._owner()
        manager = self._decode_manager(owner)

        def reduce(values, *, op, group):
            if group == "pp" and values.tolist() == [1, -1]:
                values[0] = 0

        with (
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "all_reduce", side_effect=reduce),
            self.assertRaisesRegex(ValueError, "differs across TP/CP/PP ranks"),
        ):
            manager.check_offload_progress()
        self.assertEqual(applied, [])

    def test_unified_cache_pp_policy_and_capability_are_collective(self):
        """Attention-local health must not re-enable DFS over a failed PP peer."""
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch):
                owner, applied = self._owner()
                cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
                cache.linker = None
                cache.cache_controller = SimpleNamespace(storage_backend=owner)
                cache.attn_tp_group = "tp"
                cache.attn_cp_group = "cp"
                cache.pp_group = "pp"
                cache.pp_size = 2
                cache._dfs_policy_sync_enabled = None

                def reduce(values, *, op, group):
                    if group == "pp":
                        if mismatch and values.tolist() == [1, -1]:
                            values[0] = 0
                        elif values.tolist() == [1, 1]:
                            values[:] = 0

                with (
                    patch.object(torch.distributed, "get_world_size", return_value=2),
                    patch.object(torch.distributed, "all_reduce", side_effect=reduce),
                ):
                    if mismatch:
                        with self.assertRaisesRegex(
                            ValueError, "differs across attention/PP ranks"
                        ):
                            cache._sync_dfs_health_policy()
                        self.assertEqual(applied, [])
                    else:
                        cache._sync_dfs_health_policy()
                        self.assertEqual(applied[-1], (False, False))


if __name__ == "__main__":
    unittest.main()
