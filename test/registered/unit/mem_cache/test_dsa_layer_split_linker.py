"""CPU ownership/mapping and byte-replay tests for DSA LayerSplit direct L3.

Uses tiny CPU tensors, not an accelerator or a running Mooncake service. Actual
CP broadcasts, RDMA and model accuracy still require an HCU serving test.
"""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _DsaStrategy
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    _build_dsa_device_pool_group,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _DsaPool:
    def __init__(
        self,
        rank=0,
        cp_size=8,
        layers=61,
        *,
        start=0,
        rank_offset=0,
        indexer_layers=None,
        capacity=8,
        page_size=2,
        kv_dim=4,
        sharded=True,
    ):
        self.start_layer, self.end_layer = start, start + layers
        self.layer_num, self.page_size = layers, page_size
        self.layer_shard_rank, self.layer_shard_size = rank, cp_size
        self.layer_shard_rank_offset = rank_offset
        self.layer_shard_enabled = sharded
        self.dtype, self.store_dtype = torch.float8_e4m3fn, torch.uint8
        self.index_k_cache_mode = "fp8"
        self.indexer_layer_ids = (
            tuple(range(start, start + layers))
            if indexer_layers is None
            else tuple(indexer_layers)
        )
        rows = capacity + page_size
        self.kv_buffer = [
            torch.full(
                (rows if self._is_layer_owned(layer) else 0, 1, kv_dim),
                (layer + 1) % 251,
                dtype=torch.uint8,
            )
            for layer in range(start, start + layers)
        ]
        self.index_k_with_scale_buffer = [
            torch.full(
                (
                    rows // page_size if self._is_layer_owned(layer) else 0,
                    page_size * 3,
                ),
                (layer + 101) % 251,
                dtype=torch.uint8,
            )
            for layer in self.indexer_layer_ids
        ]
        self.remote_kv_buffer = torch.full((rows, 1, kv_dim), 255, dtype=torch.uint8)

    def _is_layer_owned(self, layer):
        if not self.layer_shard_enabled:
            return True
        rank = (
            self.layer_shard_rank - self.layer_shard_rank_offset
        ) % self.layer_shard_size
        base, remainder = divmod(self.layer_num, self.layer_shard_size)
        start = rank * base + min(rank, remainder)
        end = start + base + (rank < remainder)
        return start <= layer - self.start_layer < end


class TestDsaLayerSplitLinker(unittest.TestCase):
    def test_uneven_layer_counts_cover_the_model_once(self):
        for layers in (8, 9, 60, 61, 78):
            for cp_size in (2, 4, 8):
                with self.subTest(layers=layers, cp_size=cp_size):
                    owned = []
                    for rank in range(cp_size):
                        group = _build_dsa_device_pool_group(
                            _DsaPool(rank, cp_size, layers), 2
                        )
                        owned.extend(group.entry_map[PoolName.KV].layer_mapping)
                    self.assertEqual(sorted(owned), list(range(layers)))

    def test_cp8_persists_each_target_and_rotated_draft_layer_once(self):
        kv_count = indexer_count = 0
        for rank in range(8):
            target = _DsaPool(rank)
            draft = _DsaPool(rank, layers=1, start=61, rank_offset=7)
            group = _build_dsa_device_pool_group(target, 2, (draft,))
            kv = group.entry_map[PoolName.KV]
            indexer = group.entry_map[PoolName.INDEXER]
            self.assertEqual(group.num_layers, 61)
            self.assertTrue(group.rank_replicated)
            self.assertTrue(group.storage_layout_tag.startswith("dsals_v1_cp8_"))
            self.assertTrue(all(buffer.shape[0] for buffer in kv.kv_buffer))
            self.assertTrue(all(buffer.shape[0] for buffer in indexer.kv_buffer))
            self.assertNotIn(
                target.remote_kv_buffer.data_ptr(),
                [buffer.data_ptr() for buffer in kv.kv_buffer],
            )
            kv_count += len(kv.kv_buffer)
            indexer_count += len(indexer.kv_buffer)
            if rank == 7:
                self.assertEqual(kv.layer_mapping[0], (len(kv.kv_buffer) - 1,))
                self.assertEqual(
                    indexer.layer_mapping[0], (len(indexer.kv_buffer) - 1,)
                )
            elif rank == 0:
                self.assertEqual(kv.layer_mapping[0], 0)
            else:
                self.assertNotIn(0, kv.layer_mapping)
        self.assertEqual(kv_count, 62)
        self.assertEqual(indexer_count, 62)

    def test_replicated_draft_is_still_present_on_each_rank(self):
        for rank in range(8):
            target = _DsaPool(rank)
            draft = _DsaPool(layers=1, start=61, sharded=False)
            group = _build_dsa_device_pool_group(target, 2, (draft,))
            entry = group.entry_map[PoolName.KV]
            self.assertEqual(entry.layer_mapping[0][-1], len(entry.kv_buffer) - 1)

    def test_sparse_indexer_mapping_uses_global_ids_and_compact_buffer_indices(self):
        target = _DsaPool(
            rank=1, cp_size=2, layers=5, start=10, indexer_layers=(10, 13)
        )
        group = _build_dsa_device_pool_group(target, 2)
        self.assertEqual(group.entry_map[PoolName.KV].layer_mapping, {3: 0, 4: 1})
        self.assertEqual(group.entry_map[PoolName.INDEXER].layer_mapping, {3: 0})

    def test_empty_owned_indexer_family_is_omitted(self):
        target = _DsaPool(rank=1, cp_size=2, layers=4, indexer_layers=(0, 1))
        group = _build_dsa_device_pool_group(target, 2)
        self.assertEqual(set(group.entry_map), {PoolName.KV})

    def test_unsharded_layout_and_packed_draft_mapping_are_unchanged(self):
        target = _DsaPool(cp_size=1, layers=3, indexer_layers=(0, 2), sharded=False)
        draft = _DsaPool(cp_size=1, layers=1, start=3, sharded=False)
        group = _build_dsa_device_pool_group(target, 2, (draft,))
        self.assertEqual(group.storage_layout_tag, "")
        self.assertEqual(
            group.entry_map[PoolName.KV].layer_mapping, {0: (0, 3), 1: 1, 2: 2}
        )
        self.assertEqual(
            group.entry_map[PoolName.INDEXER].layer_mapping, {0: (0, 2), 2: 1}
        )

    def test_storage_identity_ignores_addresses_and_capacity(self):
        first = _build_dsa_device_pool_group(_DsaPool(capacity=8), 2)
        second = _build_dsa_device_pool_group(_DsaPool(capacity=16), 2)
        self.assertEqual(first.storage_layout_tag, second.storage_layout_tag)

    def test_storage_identity_separates_incompatible_layouts(self):
        target = _DsaPool()
        baseline = _build_dsa_device_pool_group(target, 2).storage_layout_tag
        variants = [
            _DsaPool(cp_size=4),
            _DsaPool(start=1),
            _DsaPool(kv_dim=8),
            _DsaPool(indexer_layers=range(0, 61, 2)),
            _DsaPool(rank_offset=1),
        ]
        changed_dtype = _DsaPool()
        changed_dtype.dtype = torch.bfloat16
        variants.append(changed_dtype)
        for variant in variants:
            with self.subTest(variant=variant.__dict__):
                self.assertNotEqual(
                    baseline,
                    _build_dsa_device_pool_group(variant, 2).storage_layout_tag,
                )
        with_draft = _build_dsa_device_pool_group(
            target, 2, (_DsaPool(layers=1, start=61, rank_offset=7),)
        )
        self.assertNotEqual(baseline, with_draft.storage_layout_tag)

    def test_sharded_byte_replay_from_packed_objects_to_owned_layer_ranges(self):
        indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        for rank in range(8):
            target = _DsaPool(rank)
            draft = _DsaPool(rank, layers=1, start=61, rank_offset=7)
            group = _build_dsa_device_pool_group(target, 2, (draft,))
            for entry in group.entries:
                pointers, sizes = entry.get_page_buffer_meta(indices)
                per_page = len(entry.kv_buffer)
                objects = [
                    b"".join(
                        ctypes.string_at(pointer, size)
                        for pointer, size in zip(
                            pointers[offset : offset + per_page],
                            sizes[offset : offset + per_page],
                        )
                    )
                    for offset in range(0, len(pointers), per_page)
                ]
                for buffer in entry.kv_buffer:
                    buffer.zero_()
                locations = entry.prepare_locations(indices)
                for layer in range(group.num_layers):
                    meta = entry.get_prepared_layer_range_meta(locations, layer)
                    if meta is None:
                        continue
                    for data, pointers, sizes, offsets in zip(objects, *meta):
                        for pointer, size, offset in zip(pointers, sizes, offsets):
                            ctypes.memmove(pointer, data[offset : offset + size], size)
                pointers, sizes = entry.get_page_buffer_meta(indices)
                actual = b"".join(
                    ctypes.string_at(pointer, size)
                    for pointer, size in zip(pointers, sizes)
                )
                self.assertEqual(actual, b"".join(objects))
            self.assertTrue(torch.all(target.remote_kv_buffer == 255))

    def test_bad_target_or_draft_page_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "page sizes"):
            _build_dsa_device_pool_group(_DsaPool(), 4)
        with self.assertRaisesRegex(ValueError, "page sizes"):
            _build_dsa_device_pool_group(_DsaPool(), 2, (_DsaPool(page_size=4),))

    def test_read_plan_metadata_matches_legacy_owned_layer_ranges(self):
        indices = torch.tensor([2, 3, 6, 7], dtype=torch.int64)
        for rank in range(8):
            group = _build_dsa_device_pool_group(
                _DsaPool(rank), 2, (_DsaPool(rank, layers=1, start=61, rank_offset=7),)
            )
            linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
            linker.num_layers, linker.pools = group.num_layers, group.entry_map
            linker.storage = SimpleNamespace(
                _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
                _tag_keys=lambda keys: keys,
            )
            transfers = group.resolve_transfers(
                [
                    PoolTransfer(
                        name=PoolName.KV, keys=["a", "b"], device_indices=indices
                    )
                ]
            )
            layouts = linker._prepare_read_plan_layouts([transfers])
            for entry, (keys, locations, packed, layout) in zip(group.entries, layouts):
                self.assertEqual(keys, ["a", "b"])
                self.assertTrue(packed)
                for layer in range(group.num_layers):
                    expected = entry.get_prepared_layer_range_meta(locations, layer)
                    if expected is None:
                        self.assertEqual(layout[layer], [])
                        continue
                    pointers, sizes, offsets = expected
                    actual = [
                        (
                            [
                                base + row * stride
                                for base, stride, _, _ in layout[layer]
                            ],
                            [size for _, _, size, _ in layout[layer]],
                            [offset for _, _, _, offset in layout[layer]],
                        )
                        for row in locations
                    ]
                    self.assertEqual(actual, list(zip(pointers, sizes, offsets)))

    def test_missing_owned_buffer_is_rejected(self):
        target = _DsaPool()
        target.kv_buffer[0] = target.kv_buffer[0][:0]
        with self.assertRaisesRegex(ValueError, "owned DSA layer"):
            _build_dsa_device_pool_group(target, 2)

    def test_nonempty_nonowned_buffer_is_rejected(self):
        target = _DsaPool()
        target.kv_buffer[-1] = torch.zeros_like(target.kv_buffer[0])
        with self.assertRaisesRegex(ValueError, "Non-owned"):
            _build_dsa_device_pool_group(target, 2)

    def test_bad_indexer_mapping_is_rejected(self):
        for ids in ((0, 0), (0, 61)):
            target = _DsaPool()
            target.indexer_layer_ids = ids
            with self.assertRaisesRegex(ValueError, "distinct local"):
                _build_dsa_device_pool_group(target, 2)
        target = _DsaPool()
        target.index_k_with_scale_buffer.pop()
        with self.assertRaisesRegex(ValueError, "buffer count"):
            _build_dsa_device_pool_group(target, 2)

    def test_missing_scaled_indexer_is_rejected(self):
        target = _DsaPool()
        target.index_k_with_scale_buffer = None
        with self.assertRaisesRegex(ValueError, "scaled Index-K"):
            _build_dsa_device_pool_group(target, 2)

    def test_oversized_draft_depth_is_rejected(self):
        target = _DsaPool(cp_size=2, layers=2)
        draft = _DsaPool(cp_size=2, layers=3, start=2)
        with self.assertRaisesRegex(ValueError, "exceed"):
            _build_dsa_device_pool_group(target, 2, (draft,))

    def test_rank_with_no_physical_pools_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "no more CP ranks"):
            _build_dsa_device_pool_group(_DsaPool(rank=7, layers=1), 2)

    def test_pipeline_parallel_layer_split_is_rejected_before_pool_assembly(self):
        with self.assertRaisesRegex(ValueError, "pp_size=1"):
            _DsaStrategy().build_direct_linker_pool_group(
                kvcache=_DsaPool(),
                params=SimpleNamespace(pp_size=2, mtp_draft_device_pools=()),
                page_size=2,
            )

    def test_unsharded_pipeline_parallel_dispatch_is_unchanged(self):
        group = _DsaStrategy().build_direct_linker_pool_group(
            kvcache=_DsaPool(sharded=False),
            params=SimpleNamespace(pp_size=2, mtp_draft_device_pools=()),
            page_size=2,
        )
        self.assertEqual(group.storage_layout_tag, "")


class TestDsaLayerSplitCollectiveRevalidation(unittest.TestCase):
    def test_remote_rank_miss_rejects_reload_on_all_groups(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker._revalidate_load_local = Mock(return_value=True)
        linker._load_revalidation_groups = ("cp", "tp")

        def peer_missed(verdict, op, group):
            if group == "cp":
                verdict.zero_()

        with patch.object(
            torch.distributed, "all_reduce", side_effect=peer_missed
        ) as reduce:
            self.assertFalse(linker.revalidate_load([]))
        self.assertEqual(reduce.call_count, 2)

    def test_local_exception_still_participates_in_every_collective(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker._revalidate_load_local = Mock(side_effect=RuntimeError("evicted"))
        linker._load_revalidation_groups = ("cp", "tp")
        with (
            patch.object(torch.distributed, "all_reduce") as reduce,
            self.assertLogs(level="ERROR"),
        ):
            self.assertFalse(linker.revalidate_load([]))
        self.assertEqual(reduce.call_count, 2)


if __name__ == "__main__":
    unittest.main()
