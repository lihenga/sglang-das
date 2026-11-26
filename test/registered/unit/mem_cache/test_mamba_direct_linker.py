"""CPU regression tests for the backported Mamba direct-linker contract.

Run with Python + torch. Load the tested definitions from source so the tests
also run without SGLang's accelerator-specific import dependencies. This checks
real tensor layouts and byte transfers, not GPU kernels or a Mooncake service.
"""

import ast
import ctypes
import hashlib
import json
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


NS = dict(torch=torch, dataclass=dataclass, replace=replace, Enum=Enum, hashlib=hashlib, json=json)
definitions(SRT / "mem_cache/hicache_storage.py", {"PoolName", "PoolHitPolicy", "PoolTransfer"}, NS)
definitions(SRT / "mem_cache/unified_cache/component_type.py", {"ComponentType"}, NS)
definitions(SRT / "mem_cache/unified_cache/components/tree_component.py", {"LinkerTransferPhase", "ExternalLinkerLoadPhase"}, NS)
definitions(SRT / "mem_cache/allocator/mamba.py", {"MambaSlotAllocator"}, NS)
NS.update(TreeComponent=object, EvictParams=SimpleNamespace)
definitions(SRT / "mem_cache/unified_cache/components/mamba_component.py", {"MambaComponent"}, NS, {"build_external_linker_transfer", "update_external_linker_load"})
NS["MambaComponent"].component_type = NS["ComponentType"].MAMBA
definitions(SRT / "mem_cache/unified_cache/unified_cache_linker.py", {"UnifiedCacheLinkerWrapper"}, NS, {"_update_load", "_select_adopted_pages"})
definitions(SRT / "mem_cache/hybrid_cache/hybrid_pool_assembler.py", {"StackStrategy", "_MambaStrategy"}, NS)
definitions(SRT / "mem_cache/hybrid_cache/linker_pool_assembler.py", {"DevicePoolEntry", "DevicePoolGroup", "_mha_device_pool_entry", "_build_mamba_device_pool_group", "_gqa_page_buffers", "_qsa_index_pages", "_mamba_entry_with_ple", "_build_gqa_mamba_device_pool_group", "resolve_hybrid_device_pool_group"}, NS)
definitions(SRT / "mem_cache/storage/mooncake_store/mooncake_direct_linker.py", {"_DraftLayerTransferCounter"}, NS)
definitions(SRT / "mem_cache/storage/mooncake_store/mooncake_direct_linker.py", {"MooncakeDirectLinker"}, NS, {"_prepare_read_plan_layouts"})
definitions(SRT / "mem_cache/storage/mooncake_store/mooncake_store.py", {"MooncakeStore"}, NS, {"_get_hybrid_page_component_keys", "_pack_multi_buffer_meta"})
PoolName, PoolTransfer = NS["PoolName"], NS["PoolTransfer"]
Phase, LoadPhase = NS["LinkerTransferPhase"], NS["ExternalLinkerLoadPhase"]


class HybridLinearKVPool:
    pass


class MLATokenToKVPool:
    pass


class MHATokenToKVPool:
    pass


class QSATokenToKVPool(HybridLinearKVPool):
    pass


class QwenDSATokenToKVPool(HybridLinearKVPool):
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
        self.parallel = SimpleNamespace(attn_dcp_size=1, tp_size=2)
        NS.update(get_memory=lambda: self.memory, get_parallel=lambda: self.parallel)
        self.kv = HybridLinearKVPool()
        self.kv.use_mla = True
        self.kv.start_layer = 4
        self.kv.full_attention_layer_id_mapping = {5: 0, 7: 1}
        self.kv.full_kv_pool = MLATokenToKVPool()
        self.kv.full_kv_pool.kv_buffer = [torch.zeros((8, 1, 5), dtype=torch.uint8) for _ in range(2)]
        self.state = SimpleNamespace(
            temporal=torch.zeros((2, 8, 3), dtype=torch.uint8),
            conv=[torch.zeros((2, 8, 2), dtype=torch.uint8)],
        )
        self.req_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=self.state,
                num_mamba_layers=2,
                enable_linear_replayssm=False,
                enable_linear_replayssm_spec=False,
                replayssm_write_pos=None,
            ),
            mamba_map={4: 0, 6: 1},
            mamba_ckpt_pool=None,
            short_conv_pool=SimpleNamespace(enabled=False),
            ngram_pool=SimpleNamespace(enabled=False),
            mamba_allocator=NS["MambaSlotAllocator"](7, "cpu"),
        )
        self.params = SimpleNamespace(req_to_token_pool=self.req_pool, mtp_draft_device_pools=(), sidecar_draft_device_pools=(), pp_size=1, attn_cp_size=1)
        self.cache = SimpleNamespace(req_to_token_pool=self.req_pool, pp_size=1, evict=Mock())
        self.component = NS["MambaComponent"].__new__(NS["MambaComponent"])
        self.component.cache = self.cache
        memory_module = ModuleType("sglang.srt.mem_cache.memory_pool")
        memory_module.HybridLinearKVPool = HybridLinearKVPool
        memory_module.MLATokenToKVPool = MLATokenToKVPool
        memory_module.MHATokenToKVPool = MHATokenToKVPool
        qsa_module = ModuleType("sglang.srt.mem_cache.qsa_kv_pool")
        qsa_module.QSATokenToKVPool = QSATokenToKVPool
        qsa_module.QwenDSATokenToKVPool = QwenDSATokenToKVPool
        linker_module = ModuleType("sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler")
        linker_module._build_mamba_device_pool_group = NS["_build_mamba_device_pool_group"]
        strategy_module = ModuleType("sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler")
        strategy_module._select_strategy = lambda kv, ct: NS["_MambaStrategy"]()
        patcher = patch.dict(sys.modules, {
            memory_module.__name__: memory_module,
            qsa_module.__name__: qsa_module,
            linker_module.__name__: linker_module,
            strategy_module.__name__: strategy_module,
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def group(self, *, mooncake=False):
        return NS["resolve_hybrid_device_pool_group"](
            kvcache=self.kv, params=self.params, page_size=2,
            components={NS["ComponentType"].FULL, NS["ComponentType"].MAMBA},
            allow_gqa_mamba=mooncake,
        )

    def _setup_qwen(self, pages=4):
        def pooled_mha(layer_count):
            pool = MHATokenToKVPool()
            pool.page_size, pool.head_num, pool.head_dim, pool.v_head_dim = 2, 2, 3, 4
            pool.dtype = torch.bfloat16
            pool.use_hnd = pool.is_quantized_kv_cache = False
            pool.k_buffer = [
                torch.zeros(pages, 2, 2, 3, dtype=pool.dtype) for _ in range(layer_count)
            ]
            pool.v_buffer = [
                torch.zeros(pages, 2, 4, 2, dtype=pool.dtype) for _ in range(layer_count)
            ]
            return pool

        self.kv.use_mla = False
        self.kv.full_kv_pool = pooled_mha(2)
        draft = HybridLinearKVPool()
        draft.full_kv_pool = pooled_mha(1)
        draft.mamba_pool = self.req_pool.mamba_pool
        self.params.mtp_draft_device_pools = (draft,)
        return draft

    def test_qwen_packed_mtp_writes_and_restores_hcu_kv_and_gdn_bytes(self):
        self._setup_qwen()
        group = self.group(mooncake=True)
        self.assertEqual(group.draft_transfer_layer, 3)
        self.assertEqual(
            group.entry_map[PoolName.KV].layer_mapping, {1: (0, 2), 3: (1, 3, 4, 5)}
        )
        store = NS["MooncakeStore"]()
        store.registered_pools = group.entry_map
        store.mla_suffix = store.mha_suffix = group.storage_layout_tag
        store._tag_keys = list
        linker = NS["MooncakeDirectLinker"]()
        linker.storage = store
        linker.pools = group.entry_map
        linker.num_layers = group.num_layers
        for entry in group.entries:
            with self.subTest(pool=entry.name):
                for i, tensor in enumerate(entry.kv_buffer, 1):
                    tensor[1].fill_(i)
                page_size = entry.page_size
                write_indices = torch.arange(page_size, 2 * page_size)
                read = PoolTransfer(
                    name=entry.name,
                    keys=["prefix"],
                    host_indices=torch.arange(2 * page_size, 3 * page_size),
                )
                component_keys, multiplier = store._get_hybrid_page_component_keys(
                    read.keys, read
                )
                self.assertEqual(len(component_keys), 1)
                self.assertEqual(multiplier, 1)
                ptrs, sizes = entry.get_page_buffer_meta(write_indices)
                blob = b"".join(ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes))
                [(keys, [row], packed, layers)] = linker._prepare_read_plan_layouts([[read]])
                self.assertEqual(keys, component_keys)
                self.assertTrue(packed)
                for layer in layers:
                    for ptr, stride, size, offset in layer:
                        ctypes.memmove(ptr + row * stride, blob[offset : offset + size], size)
                for tensor in entry.kv_buffer:
                    expected, restored = tensor[1], tensor[2]
                    self.assertEqual(
                        ctypes.string_at(restored.data_ptr(), restored.nbytes),
                        ctypes.string_at(expected.data_ptr(), expected.nbytes),
                    )

        counter = Mock()
        NS["_DraftLayerTransferCounter"](counter, group.draft_transfer_layer).wait_until(0)
        counter.wait_until.assert_called_once_with(3)

    def test_qwen_layout_namespace_and_unsupported_modes(self):
        draft = self._setup_qwen()
        full_tag = self.group(mooncake=True).storage_layout_tag
        for obj, field, value, error in (
            (draft, "mamba_pool", object(), "shared target and draft GDN"),
            (self.params, "pp_size", 2, "PP and CP size 1"),
            (draft.full_kv_pool, "is_quantized_kv_cache", True, "scaled KV"),
            (self.kv.full_kv_pool, "v_buffer", [torch.zeros(4, 2, 2, 4)] * 2, "page-major"),
        ):
            with self.subTest(field=field), patch.object(obj, field, value):
                with self.assertRaisesRegex(ValueError, error):
                    self.group(mooncake=True)
        self._setup_qwen(pages=6)
        self.assertEqual(self.group(mooncake=True).storage_layout_tag, full_tag)
        self.params.mtp_draft_device_pools = ()
        self.assertNotEqual(self.group(mooncake=True).storage_layout_tag, full_tag)
        with self.assertRaisesRegex(ValueError, "plain MLA"):
            self.group()

    def _setup_flash_next(self, pool_class):
        draft = self._setup_qwen()
        qsa = pool_class()
        qsa.__dict__.update(vars(self.kv))
        if pool_class is QSATokenToKVPool:
            qsa.qsa_compress_ratio = 2
            qsa.qsa_compressed_k_buffer_pool = [torch.zeros(4, 1, 5) for _ in range(2)]
        else:
            qsa.dsa_index_k_buffer_pool = [torch.zeros(8, 1, 5) for _ in range(2)]
        self.kv = qsa
        self.params.mtp_draft_device_pools = ()
        self.req_pool.short_conv_pool = SimpleNamespace(
            enabled=True, layer_map={5: 0}, conv_state=torch.zeros(1, 8, 6)
        )
        self.req_pool.ngram_pool = SimpleNamespace(
            enabled=True, context=torch.zeros(8, 2, dtype=torch.int64)
        )
        return draft

    def test_flash_next_restores_qsa_and_ple_from_the_same_prefix(self):
        for pool_class in (QSATokenToKVPool, QwenDSATokenToKVPool):
            with self.subTest(pool=pool_class.__name__):
                self._setup_flash_next(pool_class)
                group = self.group(mooncake=True)
                self.assertTrue(group.storage_layout_tag.startswith("qsa_gdn_v1_"))
                self.assertEqual(
                    group.entry_map[PoolName.KV].layer_mapping,
                    {1: (0, 2, 4), 3: (1, 3, 5)},
                )
                self.assertEqual(
                    group.entry_map[PoolName.MAMBA].layer_mapping,
                    {0: [0, 2, 5], 1: [4], 2: [1, 3]},
                )
                store = NS["MooncakeStore"]()
                store.registered_pools = group.entry_map
                store.mla_suffix = store.mha_suffix = group.storage_layout_tag
                store._tag_keys = list
                linker = NS["MooncakeDirectLinker"]()
                linker.storage = store
                linker.pools = group.entry_map
                linker.num_layers = group.num_layers

                def transfers(page, slots):
                    return group.resolve_transfers([
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=["prefix"],
                            device_indices=torch.arange(page * 2, page * 2 + 2),
                        ),
                        PoolTransfer(
                            name=PoolName.MAMBA,
                            keys=["prefix"] * len(slots),
                            device_indices=torch.tensor(slots),
                        ),
                    ])

                source = transfers(1, [6])
                destination = transfers(2, [2, 3])
                blobs = {}
                for write in source:
                    entry = group.entry_map[write.name]
                    row = 1 if write.name == PoolName.KV else 6
                    for i, tensor in enumerate(entry.kv_buffer, 1):
                        tensor[row].fill_(i)
                    keys, _ = store._get_hybrid_page_component_keys(write.keys, write)
                    (key,) = keys
                    ptrs, sizes = entry.get_page_buffer_meta(write.host_indices)
                    (ptrs,), (sizes,) = store._pack_multi_buffer_meta(keys, ptrs, sizes)
                    blobs[key] = b"".join(
                        ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)
                    )

                for (keys, rows, packed, layers), read in zip(
                    linker._prepare_read_plan_layouts([destination]), destination
                ):
                    self.assertTrue(packed)
                    entry = group.entry_map[read.name]
                    for layer, ranges in enumerate(layers):
                        legacy = entry.get_prepared_layer_range_meta(rows, layer)
                        if not ranges:
                            self.assertIsNone(legacy)
                            continue
                        ptrs, sizes, offsets = legacy
                        for i, (key, row) in enumerate(zip(keys, rows)):
                            self.assertEqual(
                                list(zip(ptrs[i], sizes[i], offsets[i])),
                                [(p + row * stride, n, off) for p, stride, n, off in ranges],
                            )
                            for ptr, size, offset in zip(ptrs[i], sizes[i], offsets[i]):
                                ctypes.memmove(
                                    ptr, blobs[key][offset : offset + size], size
                                )
                    src = 1 if read.name == PoolName.KV else 6
                    for tensor in entry.kv_buffer:
                        for row in rows:
                            self.assertTrue(torch.equal(tensor[row], tensor[src]))

    def test_flash_next_rejects_missing_index_pages_and_mtp(self):
        draft = self._setup_flash_next(QSATokenToKVPool)
        with patch.object(self.kv, "qsa_compress_ratio", 3):
            with self.assertRaisesRegex(ValueError, "align with compression groups"):
                self.group(mooncake=True)
        with patch.object(self.kv, "qsa_compressed_k_buffer_pool", []):
            with self.assertRaisesRegex(ValueError, "index buffer for each KV layer"):
                self.group(mooncake=True)
        self.params.mtp_draft_device_pools = (draft,)
        with self.assertRaisesRegex(ValueError, "QSA/GDN.*does not support MTP"):
            self.group(mooncake=True)

    def test_flash_next_waits_before_consuming_index_or_ngram_state(self):
        methods = {}
        definitions(
            SRT / "mem_cache/qsa_kv_pool.py",
            {"QSATokenToKVPool", "QwenDSATokenToKVPool"},
            methods,
            {"get_qsa_compressed_k_buffer", "get_dsa_index_k_buffer"},
        )
        definitions(
            SRT / "mem_cache/memory_pool.py",
            {"HybridReqToTokenPool"},
            methods,
            {"get_ngram_context"},
        )
        for pool_name, buffer_name, accessor in (
            (
                "QSATokenToKVPool",
                "qsa_compressed_k_buffer_pool",
                "get_qsa_compressed_k_buffer",
            ),
            ("QwenDSATokenToKVPool", "dsa_index_k_buffer_pool", "get_dsa_index_k_buffer"),
        ):
            pool = methods[pool_name]()
            buffer = torch.zeros(1)
            setattr(pool, buffer_name, [buffer])
            pool._transfer_full_attention_id = lambda layer: 0
            pool._wait_for_layer = Mock(side_effect=lambda layer: buffer.fill_(layer))
            self.assertEqual(getattr(pool, accessor)(3).item(), 3)

        pool = methods["HybridReqToTokenPool"]()
        calls = Mock()
        pool.layer_transfer_counter = calls.counter
        pool.ngram_pool = calls.ngram
        pool.get_ngram_context("checkpoint")
        self.assertEqual(
            [entry[0] for entry in calls.mock_calls],
            ["counter.wait_until", "ngram.get_context"],
        )
        calls.counter.wait_until.assert_called_once_with(0)

    def test_qwen_rejects_both_replayssm_modes(self):
        self._setup_qwen()
        state_pool = self.req_pool.mamba_pool
        for field, cursor in (
            ("enable_linear_replayssm", torch.zeros(8, dtype=torch.int32)),
            ("enable_linear_replayssm_spec", None),
        ):
            with (
                self.subTest(mode=field),
                patch.object(state_pool, field, True),
                patch.object(state_pool, "replayssm_write_pos", cursor),
                self.assertRaisesRegex(ValueError, "does not support ReplaySSM"),
            ):
                self.group(mooncake=True)

    def test_qwen_nhd_layout_error_identifies_draft_and_supported_settings(self):
        draft = self._setup_qwen().full_kv_pool
        draft.k_buffer = [torch.zeros(8, 2, 3)]
        draft.v_buffer = [torch.zeros(8, 2, 4)]
        with self.assertRaisesRegex(
            ValueError,
            "MTP draft.*SGLANG_KV_LAYOUT_HCU_FA=1.*SGLANG_USE_HND_KVCACHE=1",
        ):
            self.group(mooncake=True)

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
            self.assertTrue(torch.equal(buffer[node_slot], buffer[6]))
            self.assertTrue(torch.equal(buffer[request_slot], buffer[6]))
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
