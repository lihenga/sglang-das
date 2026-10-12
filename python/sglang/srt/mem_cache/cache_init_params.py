from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        TreeComponent,
    )


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    pp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False
    enable_session_radix_cache: bool = False

    enable_mamba_extra_buffer: bool = False
    enable_mamba_extra_buffer_lazy: bool = False

    pp_rank: int = 0
    pp_size: int = 1

    attn_cp_rank: int = 0
    attn_cp_size: int = 1

    chunked_prefill_size: Optional[int] = None

    sliding_window_size: Optional[int] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    cache_ttl_seconds: Optional[float] = None

    tree_components: Optional[tuple[ComponentType, ...]] = None
    component_registry_override: Optional[dict[ComponentType, type[TreeComponent]]] = (
        None
    )

    mtp_draft_device_pools: tuple[object, ...] = ()

    # 普通 DP 会为每个 DP rank 启动一套彼此独立的 TP/PP default world，
    # 因而仅使用 distributed global rank 不能区分这些物理 scheduler 进程。
    # 单独传递 dp_rank，供 Mooncake 等存储客户端计算全局唯一的指标端口；
    # DP-attention 模式的 DP 维已经编码在 global rank 中，此字段应为 None。
    dp_rank: Optional[int] = None
