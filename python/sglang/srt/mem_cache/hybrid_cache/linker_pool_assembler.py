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
from sglang.srt.runtime_context import get_memory, get_parallel


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

    def get_layer_range_components(self, layer: int):
        mapped = self.layer_mapping.get(layer)
        if mapped is None:
            return None
        buffer_indices = [mapped] if isinstance(mapped, int) else list(mapped)

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            for buffer_index in buffer_indices:
                base_ptr, row_stride, size = component[buffer_index]
                items.append((base_ptr, row_stride, size, offsets[buffer_index]))
        return items

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        items = self.get_layer_range_components(layer)
        if items is None:
            return None
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
        draft_transfer_layer: int | None = None,
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
        self.draft_transfer_layer = draft_transfer_layer
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


def _deepseek_v4_layer_split_storage_tag(kvcache: Any, entries, page_size: int) -> str:
    """Identify persisted bytes independently of addresses and cache capacity."""
    layout = {
        "cp_size": kvcache.cp_size,
        "stage": [kvcache.start_layer, kvcache.end_layer],
        "compression_ratios": list(kvcache.compression_ratios),
        "page_size": page_size,
        "pools": [
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
    return f"dsv4ls_v1_cp{kvcache.cp_size}_{digest}"


def _build_deepseek_v4_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
        CpCacheLayerSplitDeepSeekV4TokenToKVPool,
    )
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _resolve_deepseek_v4_layer_mappings,
    )

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

    layer_split = isinstance(kvcache, CpCacheLayerSplitDeepSeekV4TokenToKVPool)
    transfer_layer_num = kvcache.end_layer - kvcache.start_layer
    if layer_split:
        owned = kvcache.get_hicache_host_layer_mapping()
        swa = owned["swa"]
        c4, c128 = owned["c4_kv"], owned["c128_kv"]
        indexer = owned["c4_indexer"]
        c4_state, indexer_state = owned["c4_state"], owned["c4_indexer_state"]

        def state_layers(mapping):
            return [
                kvcache.start_layer + layer
                for layer, _ in sorted(mapping.items(), key=lambda item: item[1])
            ]

        c4_state_layers = state_layers(c4_state)
        indexer_state_layers = state_layers(indexer_state)
    else:
        mappings = _resolve_deepseek_v4_layer_mappings(kvcache)
        swa, c4, c128 = mappings.swa, mappings.c4, mappings.c128
        indexer = mappings.c4
        c4_state = indexer_state = mappings.c4_state
        c4_state_layers = indexer_state_layers = mappings.c4_state_global_layers

    draft_swa_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.swa_kv_pool.kv_buffer
    ]
    swa_mapping = _with_packed_draft_mapping(
        swa,
        target_device_layer_num=len(kvcache.swa_kv_pool.kv_buffer),
        draft_layer_num=len(draft_swa_buffers),
        transfer_layer_num=transfer_layer_num if layer_split else None,
    )
    entries = []

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
        PoolName.SWA,
        PoolName.SWA,
        kvcache.swa_kv_pool,
        [*kvcache.swa_kv_pool.kv_buffer, *draft_swa_buffers],
        swa_mapping,
    )
    add(
        PoolName.DEEPSEEK_V4_C4,
        PoolName.KV,
        kvcache.c4_kv_pool,
        kvcache.c4_kv_pool.kv_buffer,
        c4,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.KV,
        kvcache.c4_indexer_kv_pool,
        kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
        indexer,
    )
    add(
        PoolName.DEEPSEEK_V4_C128,
        PoolName.KV,
        kvcache.c128_kv_pool,
        kvcache.c128_kv_pool.kv_buffer,
        c128,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.SWA,
        kvcache.compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.compress_state_pools,
            c4_state_layers,
        ),
        c4_state,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        PoolName.SWA,
        kvcache.indexer_compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.indexer_compress_state_pools,
            indexer_state_layers,
        ),
        indexer_state,
    )
    return DevicePoolGroup(
        entries,
        transfer_layer_num,
        page_size,
        rank_replicated=True,
        storage_layout_tag=(
            _deepseek_v4_layer_split_storage_tag(kvcache, entries, page_size)
            if layer_split
            else ""
        ),
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


def _build_dsa_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
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


def _mha_device_pool_entry(
    name: PoolName,
    pool: Any,
    layer_mapping: dict[int, int],
    page_size: int,
    indices_from_pool: PoolName,
) -> DevicePoolEntry:
    if pool.k_scale_buffer is not None or pool.native_k_scale_buffer is not None:
        raise ValueError("The direct external linker does not support scaled draft KV.")
    return DevicePoolEntry(
        name=name,
        indices_from_pool=indices_from_pool,
        device_pool=pool,
        # This branch's MooncakeStore uses separate DRAFT_k / DRAFT_v keys.
        # Keep offsets relative to each object for range reads and ReadPlan.
        components=[pool.k_buffer, pool.v_buffer],
        layer_mapping=layer_mapping,
        page_size=page_size,
        rows_are_pages=False,
        packed=False,
    )


def _build_mamba_device_pool_group(
    kvcache: Any, params: Any, page_size: int, *, allow_gqa: bool = False
) -> DevicePoolGroup:
    """Assemble attention pages and TP-sharded Mamba state slots for direct I/O.

    Based on shequ-code/main's implementation (introduced in eb91a41f38),
    adapted to this branch's HybridReqToTokenPool state ownership.
    """
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MLATokenToKVPool,
    )

    full = kvcache.full_kv_pool
    if (
        type(kvcache) is not HybridLinearKVPool
        or (not allow_gqa and type(full) is not MLATokenToKVPool)
        or get_parallel().attn_dcp_size > 1
        or (not allow_gqa and params.mtp_draft_device_pools)
    ):
        raise ValueError(
            "The direct external linker supports FULL + MAMBA models only with a "
            "plain MLA pool, without DCP or packed draft KV."
        )
    if get_memory().enable_unified_memory:
        raise ValueError(
            "The Mamba direct external linker does not support unified memory."
        )
    req_pool = params.req_to_token_pool
    if (
        req_pool.mamba_ckpt_pool is not None
        or req_pool.short_conv_pool.enabled
        or req_pool.ngram_pool.enabled
    ):
        raise ValueError(
            "The direct external linker does not support int8 Mamba checkpoints "
            "or Mamba side states yet."
        )
    state_pool = req_pool.mamba_pool
    # Direct GQA/GDN transfers carry temporal and conv checkpoints. Detect
    # ReplaySSM by mode because fold-every-commit does not allocate a cursor.
    if allow_gqa and (
        state_pool.enable_linear_replayssm or state_pool.enable_linear_replayssm_spec
    ):
        raise ValueError(
            "Mooncake GQA/GDN direct linking does not support ReplaySSM. "
            "Disable --enable-linear-replayssm and --enable-linear-replayssm-spec."
        )
    state = state_pool.mamba_cache
    start = kvcache.start_layer
    mamba_entry = DevicePoolEntry(
        name=PoolName.MAMBA,
        indices_from_pool=PoolName.MAMBA,
        device_pool=state_pool,
        components=[
            [tensor[layer] for layer in range(state_pool.num_mamba_layers)]
            for tensor in (state.temporal, *state.conv)
            if tensor.numel() > 0
        ],
        layer_mapping={gid - start: lid for gid, lid in req_pool.mamba_map.items()},
        page_size=1,
        rows_are_pages=True,
    )
    if allow_gqa:
        return _build_gqa_mamba_device_pool_group(kvcache, params, page_size, mamba_entry)

    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=full,
            components=[full.kv_buffer],
            layer_mapping={
                gid - start: lid
                for gid, lid in kvcache.full_attention_layer_id_mapping.items()
            },
            page_size=page_size,
            rows_are_pages=False,
        ),
        mamba_entry,
    ]
    for draft in getattr(params, "sidecar_draft_device_pools", ()):
        if draft.page_size != page_size:
            raise ValueError("Draft KV page size must match the tree page size.")
        entries.append(
            _mha_device_pool_entry(
                PoolName.DRAFT,
                draft,
                {layer: layer for layer in range(draft.layer_num)},
                page_size,
                PoolName.KV,
            )
        )
    # Mamba state is TP-sharded even when the MLA KV is replicated.
    return DevicePoolGroup(
        entries,
        max(layer for entry in entries for layer in entry.layer_mapping) + 1,
        page_size,
    )


def _gqa_page_buffers(pool, page_size: int, *, role: str) -> list[torch.Tensor]:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    if type(pool) is not MHATokenToKVPool:
        raise ValueError(
            f"Mooncake GQA/GDN requires a standard MHATokenToKVPool for {role}, "
            f"got {type(pool).__name__}. The envelope layout selected by "
            "--enable-page-major-kv-layout is unsupported by this linker."
        )
    if pool.page_size != page_size:
        raise ValueError("Mooncake GQA/GDN requires MHA pools with matching page sizes.")
    if pool.is_quantized_kv_cache or any(
        getattr(pool, name, None) is not None
        for name in (
            "k_scale_buffer",
            "v_scale_buffer",
            "native_k_scale_buffer",
            "native_v_scale_buffer",
        )
    ):
        raise ValueError("Mooncake GQA/GDN does not support scaled KV buffers.")
    keys, values = pool.k_buffer, pool.v_buffer
    if not keys or len(keys) != len(values):
        raise ValueError("Mooncake GQA/GDN requires paired K/V buffers.")
    # HCU FA uses BHSD keys and BHDS values. HND pools use BHSD for both.
    value_shape = (
        (page_size, pool.v_head_dim)
        if pool.use_hnd
        else (pool.v_head_dim, page_size)
    )
    for layer, (key, value) in enumerate(zip(keys, values)):
        if (
            key.ndim != 4
            or key.shape[1:] != (pool.head_num, page_size, pool.head_dim)
            or value.shape != (key.shape[0], pool.head_num, *value_shape)
            or not key.is_contiguous()
            or not value.is_contiguous()
        ):
            raise ValueError(
                "Mooncake GQA/GDN requires contiguous page-major K/V buffers "
                f"for {role} (pool layer {layer}, page_size={page_size}, "
                f"K={tuple(key.shape)}, V={tuple(value.shape)}). "
                "Use SGLANG_KV_LAYOUT_HCU_FA=1 on HCU, or "
                "SGLANG_USE_HND_KVCACHE=1 with an attention backend that supports HND. "
                "Token-major NHD buffers are unsupported."
            )
    return [*keys, *values]


def _build_gqa_mamba_device_pool_group(kvcache, params, page_size, mamba_entry):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if params.pp_size != 1 or params.attn_cp_size != 1:
        raise ValueError("Mooncake GQA/GDN supports TP with PP and CP size 1.")
    if params.sidecar_draft_device_pools:
        raise ValueError("Mooncake GQA/GDN requires packed MTP draft KV.")
    if any(not tensor.is_contiguous() for tensor in mamba_entry.kv_buffer):
        raise ValueError("Mooncake GQA/GDN requires contiguous GDN state slots.")

    full = kvcache.full_kv_pool
    buffers = _gqa_page_buffers(full, page_size, role="target")
    mha_pools = [full]
    n = len(full.k_buffer)
    full_mapping = {
        gid - kvcache.start_layer: (lid, n + lid)
        for gid, lid in kvcache.full_attention_layer_id_mapping.items()
    }
    mamba_mapping = mamba_entry.layer_mapping
    layers = full_mapping.keys() | mamba_mapping.keys()
    if (
        sorted(lid for lid, _ in full_mapping.values()) != list(range(n))
        or sorted(mamba_mapping.values())
        != list(range(mamba_entry.device_pool.num_mamba_layers))
        or full_mapping.keys() & mamba_mapping.keys()
        or layers != set(range(len(layers)))
    ):
        raise ValueError("Mooncake GQA/GDN requires a complete, disjoint layer map.")

    final_layer = len(layers) - 1
    for draft in params.mtp_draft_device_pools:
        draft_pool = draft
        if isinstance(draft, HybridLinearKVPool):
            if draft.mamba_pool is not mamba_entry.device_pool:
                raise ValueError("Mooncake MTP requires shared target and draft GDN state.")
            draft_pool = draft.full_kv_pool
        draft_buffers = _gqa_page_buffers(draft_pool, page_size, role="MTP draft")
        mha_pools.append(draft_pool)
        start = len(buffers)
        full_mapping[final_layer] = (
            *full_mapping.get(final_layer, ()),
            *range(start, start + len(draft_buffers)),
        )
        buffers.extend(draft_buffers)

    kv_entry = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=full,
        components=[buffers],
        layer_mapping=full_mapping,
        page_size=page_size,
        rows_are_pages=True,
    )
    entries = [kv_entry, mamba_entry]
    # Encode the object layout without including allocation capacity. Raw FP8
    # buffers can have identical uint8 layouts while using different encodings.
    schema = [
        [(str(pool.dtype), pool.use_hnd) for pool in mha_pools],
        [
            (
                str(entry.name),
                entry.page_size,
                sorted(entry.layer_mapping.items()),
                [
                    [(list(t.shape[1:]), str(t.dtype)) for t in component]
                    for component in entry.components
                ],
            )
            for entry in entries
        ],
    ]
    digest = hashlib.sha256(json.dumps(schema).encode()).hexdigest()[:24]
    return DevicePoolGroup(
        entries,
        len(layers),
        page_size,
        storage_layout_tag=f"gqa_gdn_v1_tp{get_parallel().tp_size}_{digest}",
        draft_transfer_layer=final_layer if params.mtp_draft_device_pools else None,
    )


def resolve_hybrid_device_pool_group(
    *,
    kvcache: Any,
    page_size: int,
    params: Any,
    components: set[ComponentType],
    allow_gqa_mamba: bool = False,
) -> DevicePoolGroup:
    """Materialize a direct-linker pool group through the hybrid registry."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _select_strategy,
    )

    if allow_gqa_mamba and components == {ComponentType.FULL, ComponentType.MAMBA}:
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(kvcache, HybridLinearKVPool) and not kvcache.use_mla:
            return _build_mamba_device_pool_group(
                kvcache, params, page_size, allow_gqa=True
            )

    group = _select_strategy(kvcache, components).build_direct_linker_pool_group(
        kvcache=kvcache,
        params=params,
        page_size=page_size,
    )
    if (
        getattr(params, "sidecar_draft_device_pools", ())
        and PoolName.DRAFT not in group.entry_map
    ):
        raise ValueError(
            "The direct external linker does not support this model's draft KV."
        )
    return group
