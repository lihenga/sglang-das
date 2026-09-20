import unittest

from sglang.srt.managers.schedule_batch import split_cached_prefix_by_tier
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestScheduleBatchCacheBreakdown(unittest.TestCase):
    def test_rematched_dfs_prefix_keeps_its_external_source(self):
        source_counts = {
            "l3_mooncake_memory": 0,
            "l4_mooncake_dfs": 65280,
            "l4_mooncake_local_disk": 0,
        }

        device, host, storage = split_cached_prefix_by_tier(
            prefix_len=65280,
            host_hit_len=0,
            storage_hit_len=65280,
            source_counts=source_counts,
        )
        counted_sources = {
            "l1_device": device,
            **source_counts,
        }

        self.assertEqual((device, host, storage), (0, 0, 65280))
        self.assertEqual(sum(counted_sources.values()), 65280)

    def test_immediate_load_back_keeps_device_and_dfs_counts_separate(self):
        source_counts = {
            "l3_mooncake_memory": 0,
            "l4_mooncake_dfs": 65280,
            "l4_mooncake_local_disk": 0,
        }

        device, host, storage = split_cached_prefix_by_tier(
            prefix_len=65536,
            host_hit_len=65280,
            storage_hit_len=65280,
            source_counts=source_counts,
        )

        self.assertEqual((device, host, storage), (256, 0, 65280))
        self.assertEqual(device + sum(source_counts.values()), 65536)

    def test_regular_device_hit_keeps_existing_breakdown(self):
        device, host, storage = split_cached_prefix_by_tier(
            prefix_len=65280,
            host_hit_len=0,
            storage_hit_len=0,
            source_counts={
                "l3_mooncake_memory": 0,
                "l4_mooncake_dfs": 0,
                "l4_mooncake_local_disk": 0,
            },
        )

        self.assertEqual((device, host, storage), (65280, 0, 0))

    def test_external_source_overrun_is_not_clamped(self):
        with self.assertRaises(ValueError):
            split_cached_prefix_by_tier(
                prefix_len=65280,
                host_hit_len=0,
                storage_hit_len=65280,
                source_counts={
                    "l3_mooncake_memory": 0,
                    "l4_mooncake_dfs": 65281,
                    "l4_mooncake_local_disk": 0,
                },
            )


if __name__ == "__main__":
    unittest.main()
