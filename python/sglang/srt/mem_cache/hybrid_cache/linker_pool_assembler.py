"""Materialize hybrid device pools for external cache linkers."""

from __future__ import annotations

import hashlib
import json

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class DevicePoolEntry:
    """Zero-copy linker view over one physical device pool."""

    def __init__(
        self,
        *,
        name: PoolName,
        indices_from_pool: PoolName,
        device_pool: Any,
        components: Sequence[Sequence[torch.Tensor]],
        layer_mapping: dict[int, int | Sequence[int]],
        page_size: int,
        rows_are_pages: bool,
        packed: bool = True,
        index_mapper: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.packed = packed
        self._index_mapper = index_mapper
        self._page_offsets = torch.arange(page_size)
        self._row_span = 1 if rows_are_pages else page_size

        if not self.components or any(not component for component in self.components):
            raise ValueError(f"Device pool {name} has no storage buffers.")
        self.kv_buffer = [buffer for group in self.components for buffer in group]
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)

        self.buffer_meta = [
            [
                (
                    buffer.data_ptr(),
                    buffer.stride(0) * buffer.element_size(),
                    buffer.nbytes // buffer.shape[0] * self._row_span,
                )
                for buffer in component
            ]
            for component in self.components
        ]

        self._component_offsets = []
        offset = 0
        for component in self.buffer_meta:
            if not packed:
                offset = 0
            offsets = []
            for _, _, size in component:
                offsets.append(offset)
                offset += size
            self._component_offsets.append(offsets)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._index_mapper(indices) if self._index_mapper else indices

    def _rows(self, indices: torch.Tensor) -> list[int]:
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Pool {self.name} got {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(f"Pool {self.name} requires aligned contiguous pages.")
        rows = (
            starts.div(self.page_size, rounding_mode="floor")
            if self._row_span == 1
            else starts
        )
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            raise ValueError(
                f"Pool {self.name} row range [{first_row}, {last_row}) exceeds "
                f"buffer shapes {[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )
        return rows.tolist()

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices)
        ptrs = [
            base_ptr + row * row_stride
            for row in rows
            for component in self.buffer_meta
            for base_ptr, row_stride, _ in component
        ]
        sizes = [
            size
            for _ in rows
            for component in self.buffer_meta
            for _, _, size in component
        ]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices)

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        mapped = self.layer_mapping.get(layer)
        if mapped is None:
            return None
        buffer_indices = [mapped] if isinstance(mapped, int) else list(mapped)

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            for buffer_index in buffer_indices:
                base_ptr, row_stride, size = component[buffer_index]
                items.append((base_ptr, row_stride, size, offsets[buffer_index]))

        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [
                base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items
            ]
            row_sizes = [size for _, _, size, _ in items]
            row_offsets = [offset for _, _, _, offset in items]
            if self.packed:
                ptrs.append(row_ptrs)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
            else:
                ptrs.extend([[value] for value in row_ptrs])
                sizes.extend([[value] for value in row_sizes])
                offsets.extend([[value] for value in row_offsets])
        return ptrs, sizes, offsets


class DevicePoolGroup:
    """Physical device pools sharing one logical linker layer range."""

    def __init__(
        self,
        entries: Sequence[DevicePoolEntry],
        num_layers: int,
        page_size: int,
        *,
        rank_replicated: bool = False,
        storage_layout_tag: str = "",
    ):
        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        if len(self.entries) != len(self.entry_map):
            raise ValueError("DevicePoolGroup contains duplicate pool names.")
        self.sources = {entry.name: entry.indices_from_pool for entry in self.entries}
        self.num_layers = num_layers
        self.page_size = page_size
        self.rank_replicated = rank_replicated
        self.storage_layout_tag = storage_layout_tag
        self.kv_buffer = None

    def resolve_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        allow_partial: bool = False,
        allow_missing_kv: bool = False,
    ) -> list[PoolTransfer]:
        """Expand logical component transfers into physical device pools."""
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if not any(transfer.keys for transfer in transfers):
            return []
        if not allow_missing_kv and (kv is None or not kv.keys):
            return []
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        resolved = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None or not source.keys:
                continue
            indices = source.device_indices
            resolved.append(
                replace(
                    source,
                    name=name,
                    host_indices=(
                        self.entry_map[name].translate_indices(indices)
                        if indices is not None
                        else None
                    ),
                    keys=list(source.keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                )
            )
        return resolved


def _deepseek_v4_state_views(state_pools: list[Any], global_layers: list[int]):
    views = []
    for layer in global_layers:
        pool = state_pools[layer]
        state = pool.kv_score_buffer.kv_score
        ring = int(pool.ring_size)
        usable = state.shape[0] // ring * ring
        views.append(
            state.view(torch.uint8)
            .reshape(state.shape[0], -1)[:usable]
            .reshape(usable // ring, -1)
        )
    return views


def _with_packed_draft_mapping(
    layer_mapping: dict[int, int],
    *,
    target_device_layer_num: int,
    draft_layer_num: int,
    transfer_layer_num: int | None = None,
) -> dict[int, int | tuple[int, ...]]:
    """Attach draft depth N to the same transfer layer as target layer N."""
    layer_count = (
        len(layer_mapping) if transfer_layer_num is None else transfer_layer_num
    )
    if draft_layer_num > layer_count:
        raise ValueError(
            "Packed draft layers exceed the target transfer layer count: "
            f"{draft_layer_num} > {layer_count}."
        )
    result: dict[int, int | tuple[int, ...]] = dict(layer_mapping)
    for depth in range(draft_layer_num):
        # Draft KV is replicated even when this rank does not own target depth N.
        target = (layer_mapping[depth],) if depth in layer_mapping else ()
        result[depth] = (*target, target_device_layer_num + depth)
    return result


def _build_deepseek_v4_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _dsv4_indexer_regions,
        _resolve_deepseek_v4_layer_mappings,
    )

    mappings = _resolve_deepseek_v4_layer_mappings(kvcache)
    if getattr(kvcache, "_unified_kv", False) or isinstance(
        kvcache.c4_kv_pool, HiSparseC4DevicePool
    ):
        raise ValueError(
            "The direct external linker does not support unified-KV or HiSparse."
        )
    if kvcache.swa_page_size != page_size:
        raise ValueError(
            "DeepSeek V4 SWA page size must match the tree page size: "
            f"{kvcache.swa_page_size} != {page_size}."
        )

    draft_swa_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.swa_kv_pool.kv_buffer
    ]
    swa_mapping = _with_packed_draft_mapping(
        mappings.swa,
        target_device_layer_num=len(kvcache.swa_kv_pool.kv_buffer),
        draft_layer_num=len(draft_swa_buffers),
    )
    entries = [
        DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=kvcache.swa_kv_pool,
            components=[[*kvcache.swa_kv_pool.kv_buffer, *draft_swa_buffers]],
            layer_mapping=swa_mapping,
            page_size=page_size,
            rows_are_pages=True,
        )
    ]

    def add(name, source, pool, buffers, layer_mapping):
        if layer_mapping:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=source,
                    device_pool=pool,
                    components=[buffers],
                    layer_mapping=layer_mapping,
                    page_size=page_size,
                    rows_are_pages=True,
                )
            )

    add(
        PoolName.DEEPSEEK_V4_C4,
        PoolName.KV,
        kvcache.c4_kv_pool,
        kvcache.c4_kv_pool.kv_buffer,
        mappings.c4,
    )
    for region in _dsv4_indexer_regions(kvcache, page_size):
        add(
            region.name,
            PoolName.KV,
            kvcache.c4_indexer_kv_pool,
            region.device_buffers,
            mappings.c4,
        )
    add(
        PoolName.DEEPSEEK_V4_C128,
        PoolName.KV,
        kvcache.c128_kv_pool,
        kvcache.c128_kv_pool.kv_buffer,
        mappings.c128,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.SWA,
        kvcache.compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.compress_state_pools,
            mappings.c4_state_global_layers,
        ),
        mappings.c4_state,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        PoolName.SWA,
        kvcache.indexer_compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.indexer_compress_state_pools,
            mappings.c4_state_global_layers,
        ),
        mappings.c4_state,
    )
    return DevicePoolGroup(
        entries,
        mappings.transfer_layer_num,
        page_size,
        rank_replicated=True,
    )


def _dsa_layer_split_buffers(pool: Any, page_size: int):
    """Select persistent layers, never the empty placeholders or read scratch."""
    if pool.page_size != page_size:
        raise ValueError("DSA target/draft page sizes must match the tree page size.")
    start = pool.start_layer
    kv_layers = list(range(start, start + pool.layer_num))
    indexer_layers = list(getattr(pool, "indexer_layer_ids", kv_layers))
    indexer_buffers = pool.index_k_with_scale_buffer
    if indexer_buffers is None:
        raise ValueError("DSA direct LayerSplit requires scaled Index-K buffers.")
    if len(indexer_layers) != len(set(indexer_layers)) or any(
        layer not in kv_layers for layer in indexer_layers
    ):
        raise ValueError("DSA Index-K layer ids must be distinct local model layers.")

    def select(buffers, layers):
        if len(buffers) != len(layers):
            raise ValueError("DSA buffer count does not match its layer mapping.")
        selected, mapping = [], {}
        sharded = getattr(pool, "layer_shard_enabled", False)
        for layer, buffer in zip(layers, buffers):
            owned = not sharded or pool._is_layer_owned(layer)
            if not owned:
                if buffer.shape[0] != 0:
                    raise ValueError("Non-owned DSA layers must have empty buffers.")
                continue
            if buffer.shape[0] == 0:
                raise ValueError("An owned DSA layer has no persistent buffer.")
            mapping[layer - start] = len(selected)
            selected.append(buffer)
        return selected, mapping

    kv_buffers, kv_mapping = select(pool.kv_buffer, kv_layers)
    indexer_buffers, indexer_mapping = select(indexer_buffers, indexer_layers)
    return kv_buffers, kv_mapping, indexer_buffers, indexer_mapping


def _dsa_layer_split_storage_tag(kvcache, draft_pools, entries, page_size):
    """Separate owned-layer bytes from replicated and incompatible DSA layouts."""
    layout = {
        "page_size": page_size,
        "pools": [
            {
                "stage": [pool.start_layer, pool.start_layer + pool.layer_num],
                "shard_size": getattr(pool, "layer_shard_size", 1),
                "rank_offset": getattr(pool, "layer_shard_rank_offset", 0),
                "dtype": str(getattr(pool, "dtype", None)),
                "store_dtype": str(getattr(pool, "store_dtype", None)),
                "index_mode": str(getattr(pool, "index_k_cache_mode", None)),
                "indexer_layers": list(
                    getattr(
                        pool,
                        "indexer_layer_ids",
                        range(pool.start_layer, pool.start_layer + pool.layer_num),
                    )
                ),
            }
            for pool in (kvcache, *draft_pools)
        ],
        "entries": [
            {
                "name": str(entry.name),
                "layers": entry.layer_mapping,
                "buffers": [
                    [
                        [
                            str(buffer.dtype),
                            list(buffer.shape[1:]),
                            list(buffer.stride()),
                        ]
                        for buffer in component
                    ]
                    for component in entry.components
                ],
            }
            for entry in entries
        ],
    }
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    return f"dsals_v1_cp{kvcache.layer_shard_size}_{digest}"


def _build_dsa_layer_split_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...],
) -> DevicePoolGroup:
    kv_buffers, kv_mapping, indexer_buffers, indexer_mapping = (
        _dsa_layer_split_buffers(kvcache, page_size)
    )
    num_layers = kvcache.layer_num
    draft_depth = 0
    for draft in mtp_draft_device_pools:
        if draft_depth + draft.layer_num > num_layers:
            raise ValueError(
                "Packed draft layers exceed the target transfer layer count."
            )
        draft_kv, draft_kv_mapping, draft_indexer, draft_indexer_mapping = (
            _dsa_layer_split_buffers(draft, page_size)
        )
        # A GLM draft can itself be LayerSplit (including a rotated owner),
        # unlike V4's replicated draft. Its local depth is the counter index;
        # its global layer id is only used to decide physical ownership.
        for buffers, mapping, extra_buffers, extra_mapping in (
            (kv_buffers, kv_mapping, draft_kv, draft_kv_mapping),
            (indexer_buffers, indexer_mapping, draft_indexer, draft_indexer_mapping),
        ):
            offset = len(buffers)
            for depth, buffer_index in extra_mapping.items():
                layer = draft_depth + depth
                target = (mapping[layer],) if layer in mapping else ()
                mapping[layer] = (*target, offset + buffer_index)
            buffers.extend(extra_buffers)
        draft_depth += draft.layer_num

    entries = []
    for name, buffers, mapping, rows_are_pages in (
        (PoolName.KV, kv_buffers, kv_mapping, False),
        (PoolName.INDEXER, indexer_buffers, indexer_mapping, True),
    ):
        if buffers:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=PoolName.KV,
                    device_pool=kvcache,
                    components=[buffers],
                    layer_mapping=mapping,
                    page_size=page_size,
                    rows_are_pages=rows_are_pages,
                )
            )
    if not entries:
        raise ValueError(
            "DSA direct LayerSplit requires at least one persistent KV or draft "
            "buffer on every CP rank; use no more CP ranks than target layers."
        )
    return DevicePoolGroup(
        entries,
        num_layers,
        page_size,
        rank_replicated=True,
        storage_layout_tag=_dsa_layer_split_storage_tag(
            kvcache, mtp_draft_device_pools, entries, page_size
        ),
    )


def _build_dsa_device_pool_group(kvcache: Any, page_size: int, mtp_draft_device_pools: tuple[Any, ...] = ()) -> DevicePoolGroup:
    if getattr(kvcache, "layer_shard_enabled", False):
        return _build_dsa_layer_split_device_pool_group(
            kvcache, page_size, mtp_draft_device_pools
        )
    if kvcache.page_size != page_size:
        raise ValueError(
            "DSA KV page size must match the tree page size: "
            f"{kvcache.page_size} != {page_size}."
        )
    num_layers = kvcache.layer_num
    indexer_buffers = kvcache.index_k_with_scale_buffer

    # Some DSA layers reuse another layer's top-k result and therefore do not
    # own an Index-K buffer. Map the active model layers to the compact buffer
    # list instead of assuming one Index-K buffer per model layer.
    indexer_layer_ids = getattr(kvcache, "indexer_layer_ids", None)
    if indexer_layer_ids is None:
        indexer_mapping = {layer: layer for layer in range(num_layers)}
        expected_indexer_buffers = num_layers
    else:
        start_layer = getattr(kvcache, "start_layer", 0)
        indexer_mapping = {
            layer_id - start_layer: buffer_index
            for buffer_index, layer_id in enumerate(indexer_layer_ids)
        }
        invalid_layers = [
            layer for layer in indexer_mapping if not 0 <= layer < num_layers
        ]
        if invalid_layers:
            raise ValueError(
                "DSA Index-K layers are outside the local layer range: "
                f"{invalid_layers}, num_layers={num_layers}, start_layer={start_layer}."
            )
        expected_indexer_buffers = len(indexer_layer_ids)
    if len(indexer_buffers) != expected_indexer_buffers:
        raise ValueError(
            "DSA Index-K buffer count does not match its layer mapping: "
            f"buffers={len(indexer_buffers)}, "
            f"mapped_layers={expected_indexer_buffers}."
        )
    if any(pool.page_size != page_size for pool in mtp_draft_device_pools):
        raise ValueError("DSA MTP page size must match the tree page size.")
    draft_kv_buffers = [
        buffer for pool in mtp_draft_device_pools for buffer in pool.kv_buffer
    ]
    draft_indexer_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.index_k_with_scale_buffer
    ]
    if len(draft_kv_buffers) != len(draft_indexer_buffers):
        raise ValueError("DSA MTP KV and indexer draft layer counts must match.")
    kv_layer_mapping = _with_packed_draft_mapping(
        {layer: layer for layer in range(num_layers)},
        target_device_layer_num=num_layers,
        draft_layer_num=len(draft_kv_buffers),
    )
    indexer_mapping = _with_packed_draft_mapping(
        indexer_mapping,
        target_device_layer_num=len(indexer_buffers),
        draft_layer_num=len(draft_indexer_buffers),
        transfer_layer_num=num_layers,
    )
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.kv_buffer, *draft_kv_buffers]],
            layer_mapping=kv_layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
        ),
        DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*indexer_buffers, *draft_indexer_buffers]],
            layer_mapping=indexer_mapping,
            page_size=page_size,
            rows_are_pages=True,
        ),
    ]
    return DevicePoolGroup(entries, num_layers, page_size, rank_replicated=True)


def _build_glm5_next_device_pool_group(
    kvcache: Any, params: Any, page_size: int
) -> DevicePoolGroup:
    """GLM DSA pages and KDA checkpoints, including locally owned MTP pages."""
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.mem_cache.pool_host.glm5_next import is_hcu_glm_pool
    from sglang.srt.runtime_context import get_memory, get_parallel

    memory = get_memory()
    if (
        memory.enable_unified_memory
        or memory.enable_hisparse
        or get_parallel().attn_dcp_size != 1
        or params.pp_size != 1
        or memory.unified_cache_external_linker_backend != "mooncake"
        or envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get() != "python"
    ):
        raise ValueError(
            "GLM direct linker requires Mooncake, Python TreeCore, PP=1 and DCP=1 "
            "without unified memory or HiSparse (prefill CP is supported)."
        )
    req_pool = params.req_to_token_pool
    if getattr(req_pool, "mamba_ckpt_pool", None) is not None:
        raise ValueError("GLM direct linker does not support int8 Mamba checkpoints.")
    for name in ("short_conv_pool", "ngram_pool"):
        if getattr(getattr(req_pool, name, None), "enabled", False):
            raise ValueError(
                "GLM direct linker does not support auxiliary slot states."
            )
    if kvcache.mamba_pool is not req_pool.mamba_pool:
        raise ValueError("GLM direct linker requires the target request's Mamba pool.")

    target = kvcache.full_kv_pool
    full_mapping = {
        layer - kvcache.start_layer: index
        for layer, index in kvcache.full_attention_layer_id_mapping.items()
    }
    mamba_mapping = {
        layer - kvcache.start_layer: index
        for layer, index in req_pool.mamba_map.items()
    }
    if (
        not full_mapping
        or not mamba_mapping
        or min((*full_mapping, *mamba_mapping)) < 0
    ):
        raise ValueError("GLM direct linker requires valid FULL and MAMBA layer maps.")
    num_layers = max(*full_mapping, *mamba_mapping) + 1
    pools_and_mappings = [(target, full_mapping)]
    draft_depth = 0
    for draft in params.mtp_draft_device_pools:
        if isinstance(draft, HybridLinearKVPool):
            if draft.mamba_pool is not req_pool.mamba_pool:
                raise ValueError(
                    "GLM direct linker cannot omit separate draft Mamba state."
                )
            draft = draft.full_kv_pool
        # Draft loads ride the first transfer layers, before target forward ends.
        mapping = {draft_depth + i: i for i in range(draft.layer_num)}
        draft_depth += draft.layer_num
        if draft_depth > num_layers:
            raise ValueError("GLM draft layers exceed the target transfer layer count.")
        pools_and_mappings.append((draft, mapping))

    entries = []
    for name, rows_are_pages in ((PoolName.KV, False), (PoolName.INDEXER, True)):
        buffers, layer_mapping = [], {}
        for pool, mapping in pools_and_mappings:
            if not is_hcu_glm_pool(pool) or not pool.use_dsa:
                raise ValueError(
                    "GLM direct linker requires GLM DSA target/draft pools."
                )
            if pool.page_size != page_size or page_size % pool.index_kpool:
                raise ValueError(
                    "GLM KV/tree pages must match and contain whole index groups."
                )
            source = (
                pool.kv_buffer
                if name == PoolName.KV
                else (
                    pool.index_k_with_scale_buffer
                    if pool.use_fp8_index_k_cache
                    else pool.index_k_buffer
                )
            )
            if len(source) != pool.layer_num:
                raise ValueError(
                    "GLM linker buffer count does not match the layer map."
                )
            for layer, index in mapping.items():
                buffer = source[index]
                owned = pool._is_layer_owned(pool.start_layer + index)
                if not owned:
                    if buffer.numel():
                        raise ValueError(
                            "GLM linker cannot persist remote scratch buffers."
                        )
                    continue
                if not buffer.numel():
                    raise ValueError("GLM owned KV/indexer layer has an empty buffer.")
                layer_mapping.setdefault(layer, []).append(len(buffers))
                buffers.append(buffer)
        # A CP rank can own no DSA layers. Its checkpoint still participates in
        # the rank-wide restorable-prefix intersection; no dummy KV is stored.
        if buffers:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=PoolName.KV,
                    device_pool=target,
                    components=[buffers],
                    layer_mapping=layer_mapping,
                    page_size=page_size,
                    rows_are_pages=rows_are_pages,
                )
            )

    state = req_pool.mamba_pool.mamba_cache
    components = [
        [value[index] for index in range(req_pool.mamba_pool.num_mamba_layers)]
        for value in (state.temporal, *state.conv)
        if value.numel()
    ]
    entries.append(
        DevicePoolEntry(
            name=PoolName.MAMBA,
            indices_from_pool=PoolName.MAMBA,
            device_pool=req_pool.mamba_pool,
            components=components,
            layer_mapping=mamba_mapping,
            page_size=1,
            rows_are_pages=True,
        )
    )
    for entry in entries:
        for buffer in entry.kv_buffer:
            # A row may be transposed internally, but its physical bytes must
            # form one dense region for the zero-copy range transfer.
            row_elements = buffer.numel() // buffer.shape[0]
            row_span = 1 + sum(
                (size - 1) * stride
                for size, stride in zip(buffer.shape[1:], buffer.stride()[1:])
            )
            if row_span != row_elements or buffer.stride(0) < row_elements:
                raise ValueError(
                    "GLM direct linker requires dense, non-overlapping state rows."
                )
    group = DevicePoolGroup(entries, num_layers, page_size, rank_replicated=False)
    # Different ownership, draft, dtype or index compression must never read
    # each other's packed objects, even for the same model and token hashes.
    layout = [
        (
            str(entry.name),
            entry.page_size,
            entry.layer_mapping,
            [
                (list(buffer.shape[1:]), list(buffer.stride()), str(buffer.dtype))
                for buffer in entry.kv_buffer
            ],
        )
        for entry in entries
    ]
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    group.storage_layout_tag = f"glm53_direct_v1_{digest}"
    return group


def resolve_hybrid_device_pool_group(
    *,
    kvcache: Any,
    page_size: int,
    params: Any,
    components: set[ComponentType],
) -> DevicePoolGroup:
    """Materialize a direct-linker pool group through the hybrid registry."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _select_strategy,
    )

    return _select_strategy(kvcache, components).build_direct_linker_pool_group(
        kvcache=kvcache,
        params=params,
        page_size=page_size,
    )
