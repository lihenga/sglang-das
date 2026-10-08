import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.scheduler_components.kv_capacity_observer import (
    ScheduleKVCapacityObserver,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    KVCapacitySnapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestScheduleKVCapacityObserver(unittest.TestCase):
    def setUp(self):
        self.l3_snapshot = KVCapacitySnapshot(
            available_slots=25,
            capacity_slots=100,
            available_bytes=1000,
            capacity_bytes=4000,
        )
        self.linker = mock.Mock()
        self.linker.get_local_kv_bytes_per_slot.return_value = 40
        self.linker.get_kv_capacity_snapshot.return_value = self.l3_snapshot
        self.allocator = SimpleNamespace(size=128, available_size=lambda: 96)
        self.observer = ScheduleKVCapacityObserver.create(
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=SimpleNamespace(linker=self.linker),
        )

    def test_snapshot_reports_l1_and_l3_slots_and_bytes(self):
        with mock.patch(
            "sglang.srt.managers.scheduler_components.kv_capacity_observer."
            "get_global_tracing_enabled",
            return_value=True,
        ):
            snapshots = self.observer.snapshot()

        self.assertEqual(
            snapshots["l1"],
            KVCapacitySnapshot(
                available_slots=96,
                capacity_slots=128,
                available_bytes=3840,
                capacity_bytes=5120,
            ),
        )
        self.assertEqual(snapshots["l3"], self.l3_snapshot)

    def test_snapshot_omits_l3_when_query_is_unavailable(self):
        self.linker.get_kv_capacity_snapshot.return_value = None
        with mock.patch(
            "sglang.srt.managers.scheduler_components.kv_capacity_observer."
            "get_global_tracing_enabled",
            return_value=True,
        ):
            snapshots = self.observer.snapshot()

        self.assertEqual(set(snapshots), {"l1"})

    def test_snapshot_fast_returns_when_tracing_is_disabled(self):
        with mock.patch(
            "sglang.srt.managers.scheduler_components.kv_capacity_observer."
            "get_global_tracing_enabled",
            return_value=False,
        ):
            self.assertIsNone(self.observer.snapshot())

        self.linker.get_kv_capacity_snapshot.assert_not_called()

    def test_create_rejects_tree_without_supported_linker(self):
        self.assertIsNone(
            ScheduleKVCapacityObserver.create(
                token_to_kv_pool_allocator=self.allocator,
                tree_cache=SimpleNamespace(linker=None),
            )
        )

    def test_timeline_attrs_flattens_before_and_after_snapshots(self):
        before = {
            "l1": KVCapacitySnapshot(96, 128, 3840, 5120),
            "l3": self.l3_snapshot,
        }
        after = {"l1": KVCapacitySnapshot(80, 128, 3200, 5120)}

        attrs = ScheduleKVCapacityObserver.timeline_attrs(before, after)

        self.assertEqual(attrs["l1_kv_available_slots_before"], 96)
        self.assertEqual(attrs["l1_kv_capacity_slots_before"], 128)
        self.assertEqual(attrs["l1_kv_available_bytes_after"], 3200)
        self.assertEqual(attrs["l1_kv_capacity_bytes_after"], 5120)
        self.assertEqual(attrs["l3_kv_available_slots_before"], 25)
        self.assertNotIn("l3_kv_available_slots_after", attrs)


if __name__ == "__main__":
    unittest.main()
