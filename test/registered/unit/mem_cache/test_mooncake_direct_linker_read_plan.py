import types
import unittest
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    ReadPlanLoadCounter,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeMemoryUsage,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    KVCapacitySnapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeDirectLinkerCapacity(CustomTestCase):
    def test_computes_bytes_per_slot_from_all_pool_components(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.page_size = 4
        linker.pools = {
            PoolName.KV: types.SimpleNamespace(
                buffer_meta=[[(0, 0, 16), (0, 0, 32)], [(0, 0, 48)]]
            )
        }

        self.assertEqual(linker._compute_kv_bytes_per_slot(), 24)

    def test_converts_cluster_bytes_to_complete_logical_slots(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.storage = types.SimpleNamespace(
            get_memory_usage=lambda: MooncakeMemoryUsage(
                used_bytes=1000,
                available_bytes=3000,
                capacity_bytes=4000,
            )
        )
        linker._cluster_kv_bytes_per_slot = 80
        linker._capacity_warning_logged = False

        self.assertEqual(
            linker.get_kv_capacity_snapshot(),
            KVCapacitySnapshot(
                available_slots=37,
                capacity_slots=50,
                available_bytes=3000,
                capacity_bytes=4000,
            ),
        )

    def test_capacity_query_failure_is_fail_open_and_warns_once(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.storage = types.SimpleNamespace(
            get_memory_usage=Mock(side_effect=TimeoutError("admin timeout"))
        )
        linker._cluster_kv_bytes_per_slot = 80
        linker._capacity_warning_logged = False

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="WARNING",
        ) as logs:
            self.assertIsNone(linker.get_kv_capacity_snapshot())
            self.assertIsNone(linker.get_kv_capacity_snapshot())

        self.assertEqual(len(logs.output), 1)


class TestMooncakeDirectLinkerReadPlan(CustomTestCase):
    def test_read_plan_failure_is_deferred_until_after_forward(self):
        counter = ReadPlanLoadCounter(num_layers=2)
        index = counter.update_producer()
        counter.set_consumer(index)
        plan = Mock()
        plan.wait.side_effect = RuntimeError("range get failed: rc=-707")
        counter.bind(index, plan)

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="ERROR",
        ) as logs:
            counter.wait_until(0)
            counter.wait_until(1)

        self.assertEqual(len(logs.output), 1)
        self.assertNotIn(index, counter.plans)
        self.assertNotIn(index, counter.reported)

    def test_successful_read_plan_load_reports_success(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.read_plan_enabled = True
        linker.tp_rank = 0
        linker.load_with_read_plan = Mock()

        success = linker.load_layer_wise(7, [])

        self.assertIs(success, True)
        linker.load_with_read_plan.assert_called_once_with(7, [])

    def test_layout_expands_packed_layer_mapping(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.num_layers = 3
        linker.storage = types.SimpleNamespace(
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=lambda keys: [f"tagged:{key}" for key in keys],
        )

        pool = types.SimpleNamespace(
            layer_mapping={0: (0, 2), 1: 1},
            buffer_meta=[
                [(100, 10, 1), (110, 11, 2), (120, 12, 3)],
                [(200, 20, 4), (210, 21, 5), (220, 22, 6)],
            ],
            _component_offsets=[[7, 8, 9], [17, 18, 19]],
            packed=True,
            prepare_locations=lambda indices: [int(value) for value in indices],
        )
        linker.pools = {PoolName.KV: pool}

        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([4, 5]),
            keys=["page-0"],
        )
        layouts = linker._prepare_read_plan_layouts([[transfer]])

        self.assertEqual(len(layouts), 1)
        keys, locations, packed, layers = layouts[0]
        self.assertEqual(keys, ["tagged:page-0"])
        self.assertEqual(locations, [4, 5])
        self.assertTrue(packed)
        self.assertEqual(
            layers,
            [
                [
                    (100, 10, 1, 7),
                    (120, 12, 3, 9),
                    (200, 20, 4, 17),
                    (220, 22, 6, 19),
                ],
                [(110, 11, 2, 8), (210, 21, 5, 18)],
                [],
            ],
        )


if __name__ == "__main__":
    unittest.main()
