"""CPU regression tests for the backported Mamba direct-linker contract.

Run with Python + torch. Load the tested definitions from source so the tests
also run without SGLang's accelerator-specific import dependencies. This checks
real tensor layouts and byte transfers, not GPU kernels or a Mooncake service.
"""

import ast
import ctypes
import sys
import unittest
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def definitions(path, names, namespace, methods=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if methods is not None:
        for node in nodes:
            node.bases = []
            node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


NS = dict(torch=torch, dataclass=dataclass, replace=replace, Enum=Enum)
definitions(SRT / "mem_cache/hicache_storage.py", {"PoolName", "PoolHitPolicy", "PoolTransfer"}, NS)
definitions(SRT / "mem_cache/unified_cache/component_type.py", {"ComponentType"}, NS)
definitions(SRT / "mem_cache/unified_cache/components/tree_component.py", {"LinkerTransferPhase", "ExternalLinkerLoadPhase"}, NS)
definitions(SRT / "mem_cache/allocator/mamba.py", {"MambaSlotAllocator"}, NS)
NS.update(TreeComponent=object, EvictParams=SimpleNamespace)
definitions(SRT / "mem_cache/unified_cache/components/mamba_component.py", {"MambaComponent"}, NS, {"build_external_linker_transfer", "update_external_linker_load"})
NS["MambaComponent"].component_type = NS["ComponentType"].MAMBA
definitions(SRT / "mem_cache/unified_cache/unified_cache_linker.py", {"UnifiedCacheLinkerWrapper"}, NS, {"_update_load", "_select_adopted_pages"})
definitions(SRT / "mem_cache/hybrid_cache/hybrid_pool_assembler.py", {"StackStrategy", "_MambaStrategy"}, NS)
definitions(SRT / "mem_cache/hybrid_cache/linker_pool_assembler.py", {"DevicePoolEntry", "DevicePoolGroup", "_mha_device_pool_entry", "_build_mamba_device_pool_group", "resolve_hybrid_device_pool_group"}, NS)
PoolName, PoolTransfer = NS["PoolName"], NS["PoolTransfer"]
Phase, LoadPhase = NS["LinkerTransferPhase"], NS["ExternalLinkerLoadPhase"]


class HybridLinearKVPool:
    pass


class MLATokenToKVPool:
    pass


class _Slots(list):
    """Index-only stand-in for the dependency-free ownership checks."""

    def __getitem__(self, index):
        value = super().__getitem__(index)
        return _Slots(value) if isinstance(index, slice) else value

    def numel(self):
        return len(self)


class TestMambaDirectContract(unittest.TestCase):
    def setUp(self):
        self.allocator = Mock()
        self.allocator.alloc.return_value = _Slots([1, 2])
        self.component = NS["MambaComponent"].__new__(NS["MambaComponent"])
        self.component.cache = SimpleNamespace(
            pp_size=1, evict=Mock(),
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=self.allocator, mamba_pool=SimpleNamespace()
            ),
        )

    def test_strategy_delegates_to_mamba_builder(self):
        module = ModuleType("sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler")
        module._build_mamba_device_pool_group = Mock(return_value="mamba-group")
        with patch.dict(sys.modules, {module.__name__: module}):
            result = NS["_MambaStrategy"]().build_direct_linker_pool_group(
                kvcache="kv", params="params", page_size=128
            )
        self.assertEqual(result, "mamba-group")
        module._build_mamba_device_pool_group.assert_called_once_with("kv", "params", 128)

    def test_builder_maps_stage_local_layers_and_rejects_missing_state(self):
        memory_module = ModuleType("sglang.srt.mem_cache.memory_pool")
        memory_module.HybridLinearKVPool = HybridLinearKVPool
        memory_module.MLATokenToKVPool = MLATokenToKVPool
        kv = HybridLinearKVPool()
        kv.full_kv_pool = MLATokenToKVPool()
        kv.full_kv_pool.kv_buffer = [object(), object()]
        kv.start_layer = 4
        kv.full_attention_layer_id_mapping = {5: 0, 7: 1}
        params = SimpleNamespace(
            mtp_draft_device_pools=(), sidecar_draft_device_pools=(),
            req_to_token_pool=SimpleNamespace(
                mamba_ckpt_pool=None,
                short_conv_pool=SimpleNamespace(enabled=False),
                ngram_pool=SimpleNamespace(enabled=False),
                mamba_map={4: 0, 6: 1},
                mamba_pool=SimpleNamespace(
                    num_mamba_layers=2,
                    mamba_cache=SimpleNamespace(
                        temporal=_Slots([object(), object()]),
                        conv=[_Slots([object(), object()])],
                    ),
                ),
            ),
        )
        with patch.dict(sys.modules, {memory_module.__name__: memory_module}), patch.dict(NS, {
            "get_memory": lambda: SimpleNamespace(enable_unified_memory=False),
            "get_parallel": lambda: SimpleNamespace(attn_dcp_size=1),
            "DevicePoolEntry": lambda **kwargs: SimpleNamespace(**kwargs),
        }):
            group = NS["_build_mamba_device_pool_group"](kv, params, 128)
            self.assertEqual(group.num_layers, 4)
            self.assertFalse(group.rank_replicated)
            self.assertEqual(group.entry_map[PoolName.KV].layer_mapping, {1: 0, 3: 1})
            state = group.entry_map[PoolName.MAMBA]
            self.assertEqual(state.layer_mapping, {0: 0, 2: 1})
            self.assertEqual(state.page_size, 1)
            self.assertEqual(len(state.components), 2)
            params.req_to_token_pool.short_conv_pool.enabled = True
            with self.assertRaisesRegex(ValueError, "side states"):
                NS["_build_mamba_device_pool_group"](kv, params, 128)

    def test_failed_allocation_evicts_then_degrades_to_miss(self):
        self.allocator.alloc.side_effect = [None, None]
        self.assertIsNone(self.component.build_external_linker_transfer(Phase.LOAD, None, ["end"]))
        self.component.cache.evict.assert_called_once()
        self.assertEqual(self.allocator.alloc.call_count, 2)

    def test_load_and_abort_own_both_slots(self):
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["start", "end"])
        self.allocator.alloc.assert_called_once_with(2)
        self.assertEqual(transfer.keys, ["end", "end"])
        self.component.update_external_linker_load(LoadPhase.ABORT, None, None, transfer, 256)
        self.allocator.free.assert_called_once_with(_Slots([1, 2]))

    def test_pipeline_load_allocates_one_request_state(self):
        self.component.cache.pp_size = 2
        self.allocator.alloc.return_value = _Slots([3])
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["end"])
        self.allocator.alloc.assert_called_once_with(1)
        self.assertEqual(transfer.keys, ["end"])

    def test_offload_uses_checkpoint_hash_and_ignores_tombstones(self):
        data = SimpleNamespace(value=_Slots([3]))
        node = SimpleNamespace(hash_value=["start", "end"], component_data={NS["ComponentType"].MAMBA: data})
        transfer = self.component.build_external_linker_transfer(Phase.OFFLOAD, node, None)
        self.assertEqual(transfer.keys, ["end"])
        self.assertEqual(transfer.device_indices, [3])
        data.value = None
        self.assertIsNone(self.component.build_external_linker_transfer(Phase.OFFLOAD, node, None))

    def test_prepare_supersedes_old_request_state_and_deferred_clear(self):
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["end"])
        old = Mock()
        old.unsqueeze.return_value = _Slots([7])
        req = SimpleNamespace(mamba_pool_idx=old, mamba_cow_src_index=6, mamba_needs_clear=True)
        self.component.update_external_linker_load(LoadPhase.PREPARE, req, None, transfer, 256)
        self.allocator.free.assert_called_once_with(_Slots([7]))
        self.assertEqual(req.mamba_pool_idx, 2)
        self.assertIsNone(req.mamba_cow_src_index)
        self.assertFalse(req.mamba_needs_clear)

    def test_commit_keeps_private_state_even_without_adopted_kv_pages(self):
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["end"])
        wrapper = NS["UnifiedCacheLinkerWrapper"]()
        wrapper.cache = SimpleNamespace(page_size=128)
        result = wrapper._update_load(
            LoadPhase.COMMIT, None, [(self.component, transfer)], 256,
            insert_result=SimpleNamespace(mamba_exist=True, adopted_ranges={}),
        )
        self.assertEqual(result[0].device_indices, [2])
        self.assertEqual(result[0].keys, ["end"])


@unittest.skipIf(torch is None, "PyTorch is required for tensor layout and byte replay")
class TestMambaDirectLinker(unittest.TestCase):
    def setUp(self):
        self.memory = SimpleNamespace(enable_unified_memory=False)
        self.parallel = SimpleNamespace(attn_dcp_size=1)
        NS.update(get_memory=lambda: self.memory, get_parallel=lambda: self.parallel)
        self.kv = HybridLinearKVPool()
        self.kv.start_layer = 4
        self.kv.full_attention_layer_id_mapping = {5: 0, 7: 1}
        self.kv.full_kv_pool = MLATokenToKVPool()
        self.kv.full_kv_pool.kv_buffer = [torch.zeros((8, 1, 5), dtype=torch.uint8) for _ in range(2)]
        self.state = SimpleNamespace(
            temporal=torch.zeros((2, 8, 3), dtype=torch.uint8),
            conv=[torch.zeros((2, 8, 2), dtype=torch.uint8)],
        )
        self.req_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=self.state, num_mamba_layers=2),
            mamba_map={4: 0, 6: 1},
            mamba_ckpt_pool=None,
            short_conv_pool=SimpleNamespace(enabled=False),
            ngram_pool=SimpleNamespace(enabled=False),
            mamba_allocator=NS["MambaSlotAllocator"](7, "cpu"),
        )
        self.params = SimpleNamespace(req_to_token_pool=self.req_pool, mtp_draft_device_pools=(), sidecar_draft_device_pools=())
        self.cache = SimpleNamespace(req_to_token_pool=self.req_pool, pp_size=1, evict=Mock())
        self.component = NS["MambaComponent"].__new__(NS["MambaComponent"])
        self.component.cache = self.cache
        memory_module = ModuleType("sglang.srt.mem_cache.memory_pool")
        memory_module.HybridLinearKVPool = HybridLinearKVPool
        memory_module.MLATokenToKVPool = MLATokenToKVPool
        linker_module = ModuleType("sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler")
        linker_module._build_mamba_device_pool_group = NS["_build_mamba_device_pool_group"]
        strategy_module = ModuleType("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler")
        strategy_module._select_strategy = lambda kv, ct: NS["_MambaStrategy"]()
        patcher = patch.dict(sys.modules, {
            memory_module.__name__: memory_module,
            linker_module.__name__: linker_module,
            strategy_module.__name__: strategy_module,
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def group(self):
        return NS["resolve_hybrid_device_pool_group"](
            kvcache=self.kv, params=self.params, page_size=2,
            components={NS["ComponentType"].FULL, NS["ComponentType"].MAMBA},
        )

    def test_strategy_maps_stage_local_layers_and_distinct_index_spaces(self):
        self.assertTrue(NS["_MambaStrategy"]().matches(self.kv, {NS["ComponentType"].FULL, NS["ComponentType"].MAMBA}))
        group = self.group()
        self.assertEqual(group.num_layers, 4)
        self.assertFalse(group.rank_replicated)
        kv, state = group.entry_map[PoolName.KV], group.entry_map[PoolName.MAMBA]
        self.assertEqual(kv.layer_mapping, {1: 0, 3: 1})
        self.assertEqual(state.layer_mapping, {0: 0, 2: 1})
        self.assertEqual(kv.prepare_locations(torch.tensor([2, 3])), [2])
        self.assertEqual(state.prepare_locations(torch.tensor([3])), [3])

    def test_rejects_unsupported_storage_modes(self):
        for obj, attr, value in (
            (self.parallel, "attn_dcp_size", 2),
            (self.memory, "enable_unified_memory", True),
            (self.params, "mtp_draft_device_pools", (object(),)),
            (self.req_pool, "mamba_ckpt_pool", object()),
            (self.req_pool.short_conv_pool, "enabled", True),
            (self.req_pool.ngram_pool, "enabled", True),
            (self.kv, "full_kv_pool", object()),
        ):
            with self.subTest(attr=attr), patch.object(obj, attr, value):
                with self.assertRaises(ValueError):
                    self.group()

    def test_conv_only_and_draft_kv_layout(self):
        self.state.temporal = torch.empty(0)
        self.params.sidecar_draft_device_pools = (SimpleNamespace(
            page_size=2, layer_num=1, k_scale_buffer=None, native_k_scale_buffer=None,
            k_buffer=[torch.zeros((8, 3), dtype=torch.uint8)],
            v_buffer=[torch.zeros((8, 4), dtype=torch.uint8)],
        ),)
        group = self.group()
        self.assertEqual(len(group.entry_map[PoolName.MAMBA].components), 1)
        self.assertEqual(group.sources[PoolName.DRAFT], PoolName.KV)
        draft = group.entry_map[PoolName.DRAFT]
        self.assertEqual(draft.layer_mapping, {0: 0})
        self.assertFalse(draft.packed)
        pointers, sizes, offsets = draft.get_prepared_layer_range_meta([2], 0)
        self.assertEqual(len(pointers), 2)
        self.assertEqual(sizes, [[6], [8]])
        self.assertEqual(offsets, [[0], [0]])

    def test_load_allocation_failure_and_abort_return_slots(self):
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["a", "b"])
        self.assertEqual(transfer.keys, ["b", "b"])
        self.assertEqual(self.req_pool.mamba_allocator.available_size(), 5)
        self.component.update_external_linker_load(LoadPhase.ABORT, None, None, transfer, 4)
        self.assertEqual(self.req_pool.mamba_allocator.available_size(), 7)
        self.req_pool.mamba_allocator.alloc(7)
        self.assertIsNone(self.component.build_external_linker_transfer(Phase.LOAD, None, ["b"]))
        self.cache.evict.assert_called_once()

    def test_offload_uses_only_checkpoint_endpoint(self):
        node = SimpleNamespace(hash_value=["a", "b"], component_data={
            NS["ComponentType"].MAMBA: SimpleNamespace(value=torch.tensor([3]))
        })
        transfer = self.component.build_external_linker_transfer(Phase.OFFLOAD, node, None)
        self.assertEqual(transfer.keys, ["b"])
        self.assertEqual(transfer.device_indices.tolist(), [3])

    def test_layerwise_byte_restore_keeps_request_and_checkpoint_independent(self):
        group = self.group()
        entry = group.entry_map[PoolName.MAMBA]
        for i, buffer in enumerate(entry.kv_buffer):
            buffer[6].fill_(i + 11)
        ptrs, sizes = entry.get_page_buffer_meta(torch.tensor([6]))
        blob = b"".join(ctypes.string_at(p, s) for p, s in zip(ptrs, sizes))
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["b"])
        req = SimpleNamespace(mamba_pool_idx=None, mamba_cow_src_index=torch.tensor([6]), mamba_needs_clear=True)
        self.component.update_external_linker_load(LoadPhase.PREPARE, req, None, transfer, 4)
        self.assertIsNone(req.mamba_cow_src_index)
        self.assertFalse(req.mamba_needs_clear)
        for layer in range(group.num_layers):
            meta = entry.get_prepared_layer_range_meta(transfer.device_indices.tolist(), layer)
            if meta is None:
                continue
            for dests, counts, offsets in zip(*meta):
                for dest, count, offset in zip(dests, counts, offsets):
                    ctypes.memmove(dest, blob[offset:offset + count], count)
        node_slot, request_slot = transfer.device_indices.tolist()
        self.assertEqual(req.mamba_pool_idx.item(), request_slot)
        for buffer in entry.kv_buffer:
            torch.testing.assert_close(buffer[node_slot], buffer[6])
            torch.testing.assert_close(buffer[request_slot], buffer[6])
            buffer[request_slot].zero_()
            self.assertTrue(buffer[node_slot].count_nonzero())

    def test_commit_preserves_request_load_when_tree_already_has_state(self):
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["b"])
        # load_back discards the duplicate checkpoint slot before COMMIT.
        self.req_pool.mamba_allocator.free(transfer.device_indices[:1])
        wrapper = NS["UnifiedCacheLinkerWrapper"]()
        wrapper.cache = SimpleNamespace(page_size=2)
        result = wrapper._update_load(
            LoadPhase.COMMIT, SimpleNamespace(), [(self.component, transfer)], 4,
            insert_result=SimpleNamespace(mamba_exist=True, adopted_ranges={}),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].device_indices.tolist(), [2])
        self.assertEqual(result[0].keys, ["b"])

    def test_pipeline_load_uses_one_request_owned_state(self):
        self.cache.pp_size = 2
        transfer = self.component.build_external_linker_transfer(Phase.LOAD, None, ["b"])
        self.assertEqual(transfer.keys, ["b"])
        self.assertEqual(transfer.device_indices.numel(), 1)


if __name__ == "__main__":
    unittest.main()
