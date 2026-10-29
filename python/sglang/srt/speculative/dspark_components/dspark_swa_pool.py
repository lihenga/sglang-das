"""DSPARK draft SWA ring KV pool: geometry resolution and mapping upkeep.

Opt-in via SGLANG_DSPARK_SWA_KV_POOL=1. The draft has 4 sliding layers
(window 512, never read beyond the window) and 1 full layer. With the flag,
the draft pool is built as a SWAKVPool whose sliding sub-pool is a per-req
ring addressed by

    swa_loc = req_slot * ring_tokens + pos % ring_tokens

so the sliding layers store ~ring tokens per request instead of full-length
storage. ``full_to_swa_index_mapping`` (a SWAKVPool native) is the single
piece of state: it maps a full-pool loc to the ring slot. Attention backends
translate reads through it (update_sliding_window_buffer, triton
swa_out_cache_loc) and the target-hidden injection translates writes through
it, so keeping it filled with the idempotent formula before every
draft-side windowed access is the only contract.

Windowed-out tokens map to -1: the scatter skips them and a stale entry from
a recycled loc cannot alias into the live ring.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule

logger = logging.getLogger(__name__)


def resolve_dspark_swa_layer_split(draft_hf_config) -> Tuple[List[int], List[int]]:
    """Draft-local sliding/full layer ids from target_layer_ids + layer_types."""
    target_layer_ids = draft_hf_config.target_layer_ids
    layer_types = draft_hf_config.layer_types
    swa_ids: List[int] = []
    full_ids: List[int] = []
    for draft_idx, target_idx in enumerate(target_layer_ids):
        if target_idx < len(layer_types) and layer_types[target_idx] == "sliding_attention":
            swa_ids.append(draft_idx)
        else:
            full_ids.append(draft_idx)
    if not swa_ids:
        raise ValueError(
            "SGLANG_DSPARK_SWA_KV_POOL requires sliding draft layers; "
            f"target_layer_ids={target_layer_ids} has none."
        )
    if not full_ids:
        raise ValueError(
            "SGLANG_DSPARK_SWA_KV_POOL requires at least one full draft layer "
            f"to anchor the full pool; target_layer_ids={target_layer_ids}."
        )
    return swa_ids, full_ids


def resolve_dspark_swa_ring_tokens(window: int, page_size: int) -> int:
    """Ring slots per request. Default 2x window page-aligned: reads touch
    [pos-window+1, pos] while a step writes up to verify window more tokens,
    so ring must exceed window + writes to keep read/write slots disjoint."""
    ring = envs.SGLANG_DSPARK_SWA_RING_TOKENS.get()
    if ring > 0:
        return ring
    ring = 2 * window
    return (ring + page_size - 1) // page_size * page_size


def build_draft_swa_memory_pool_config(
    target_config,
    *,
    swa_tokens: int,
):
    """Draft-side MemoryPoolConfig: full pool aligned with the target, sliding
    sub-pool sized to the ring. The draft worker consumes this config verbatim
    (KVCacheConfigurator.configure passes it through for draft workers)."""
    import dataclasses

    return dataclasses.replace(
        target_config,
        max_total_num_tokens=target_config.max_total_num_tokens,
        full_max_total_num_tokens=target_config.max_total_num_tokens,
        swa_max_total_num_tokens=swa_tokens,
    )


def create_dspark_swa_ring_mapping(size_full: int, page_size: int, device: str):
    """All -1: an unfilled loc translates to skip, never to pool slot 0."""
    n = size_full + page_size
    mapping = torch.full((n + 1,), -1, dtype=torch.int64, device=device)
    mapping[-1] = -1
    return mapping


def is_dspark_swa_ring_pool(pool) -> bool:
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    return type(pool) is SWAKVPool


def dspark_pd_skipped_draft_layers(draft_model_path: str) -> Optional[set]:
    """Draft-local layer ids the P->D full-length KV transfer must skip under
    SGLANG_DSPARK_PD_SKIP_SWA_DRAFT_KV: the sliding layers, which ship only
    their last window as the DSPARK_DRAFT_SWA state component into the
    decode peer's ring. None when the gate is off or the draft config has no
    sliding layers."""
    if not envs.SGLANG_DSPARK_PD_SKIP_SWA_DRAFT_KV.get():
        return None
    import json
    import os

    try:
        with open(os.path.join(draft_model_path, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(
            "SGLANG_DSPARK_PD_SKIP_SWA_DRAFT_KV=1 but the draft config at %s "
            "is unreadable (%s); transferring all draft layers",
            draft_model_path,
            e,
        )
        return None
    layer_types = cfg.get("layer_types", [])
    skipped = {
        idx
        for idx, target_idx in enumerate(cfg.get("target_layer_ids", []))
        if target_idx < len(layer_types)
        and layer_types[target_idx] == "sliding_attention"
    }
    return skipped or None


def apply_dspark_swa_pool_geometry(
    draft_model_runner,
    draft_model,
    req_to_token_pool,
    target_memory_pool_config,
):
    """Flip the draft model_config / runner view to hybrid SWA and return
    (draft MemoryPoolConfig, window, ring). Call before the draft worker
    allocates its pools; the configurator then builds the SWAKVPool
    (MLA full sub-pool + MLA sliding sub-pool) on its own."""
    window = draft_model.get_attention_sliding_window_size()
    if not window or int(window) <= 0:
        raise ValueError(
            "SGLANG_DSPARK_SWA_KV_POOL=1 but the draft model reports no SWA "
            "window (get_attention_sliding_window_size -> "
            f"{window}); the draft is not a sliding-window draft."
        )
    window = int(window)

    model_config = draft_model_runner.model_config
    swa_ids, full_ids = resolve_dspark_swa_layer_split(model_config.hf_config)
    model_config.is_hybrid_swa = True
    model_config.swa_attention_layer_ids = swa_ids
    model_config.full_attention_layer_ids = full_ids
    # Sliding draft layers share the target's latent geometry (kv_lora_rank +
    # qk_rope), so the swa sub-pool per-token bytes equal the full sub-pool's.
    model_config.swa_kv_lora_rank = model_config.kv_lora_rank
    model_config.swa_qk_rope_head_dim = model_config.qk_rope_head_dim

    # Cached at ModelRunner.__init__; the pool configurator reads these.
    draft_model_runner.is_hybrid_swa = True

    ring = resolve_dspark_swa_ring_tokens(window, get_schedule().page_size)
    num_req_slots = req_to_token_pool.req_to_token.shape[0]
    swa_tokens = num_req_slots * ring
    # Read by the injection/proposal sites to keep the ring mapping filled.
    draft_model_runner.dspark_swa_window = window
    draft_model_runner.dspark_swa_ring = ring
    draft_config = build_draft_swa_memory_pool_config(
        target_memory_pool_config, swa_tokens=swa_tokens
    )
    logger.info(
        "DSPARK draft SWA pool: window=%d, ring=%d, req_slots=%d, "
        "swa_layers=%s, full_layers=%s, swa_pool_tokens=%d, "
        "full_pool_tokens=%d (est. saving %.2f GB vs full-length draft pool)",
        window,
        ring,
        num_req_slots,
        swa_ids,
        full_ids,
        swa_tokens,
        draft_config.max_total_num_tokens,
        (len(swa_ids) * draft_config.max_total_num_tokens - swa_tokens)
        * (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
        / 2**30,
    )
    return draft_config, window, ring


@torch.inference_mode()
def fill_dspark_swa_ring_for_draft_forward(
    draft_model_runner,
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    draft_positions: Optional[torch.Tensor],
    draft_cache_locs: Optional[torch.Tensor],
    block_size: Optional[int] = None,
) -> None:
    """Refresh the ring mapping before a draft forward: (a) the draft block's
    write slots and (b) each req's readable window [prefix-window, prefix).
    Idempotent; re-running (b) also repairs stale entries after loc recycling
    (retract / rebootstrap). No-op unless the draft pool is the ring SWAKVPool.

    draft_positions/draft_cache_locs None: the block inputs are generated at
    graph replay (spark_dspark_graph_inputs), so the forward batch tensors
    still hold the previous step's rows. Derive the slots the generator will
    produce instead: pos = prefix + col, loc = req_to_token[slot, pos]."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    pool = draft_model_runner.token_to_kv_pool
    if type(pool) is not SWAKVPool:
        return
    if req_pool_indices.shape[0] == 0:
        return
    mapping = pool.full_to_swa_index_mapping
    window = draft_model_runner.dspark_swa_window
    ring = draft_model_runner.dspark_swa_ring
    req_to_token = draft_model_runner.req_to_token_pool.req_to_token
    bs = req_pool_indices.shape[0]
    if draft_positions is None:
        cols = torch.arange(block_size, device=req_to_token.device, dtype=torch.int64)
        pos_2d = prefix_lens[:bs].to(torch.int64).view(-1, 1) + cols
        draft_positions = pos_2d.reshape(-1)
        draft_cache_locs = req_to_token[
            req_pool_indices.to(torch.int64).view(-1, 1), pos_2d
        ].reshape(-1)
    # (a) write slots: every draft token lands inside the window (block pos =
    # prefix tail), so no final_pos filter is needed.
    block_stride = draft_positions.shape[0] // bs
    req_slots = (
        req_pool_indices.to(torch.int64).repeat_interleave(block_stride).to(draft_positions.device)
    )
    scatter_dspark_swa_ring_mapping(
        mapping,
        req_slots=req_slots,
        positions=draft_positions,
        full_locs=draft_cache_locs,
        window=window,
        ring=ring,
    )
    # (b) readable prefix window, one short range per req. No row carry-over
    # from another req slot: a finished req's ring rows are overwritten once
    # its slot is reused, so they are never a valid source. D-side radix reuse
    # is capped at the window start instead (dspark_swa_decode_prefix_cap),
    # which keeps the whole window inside the freshly transferred ring rows.
    for req_slot, prefix in zip(
        req_pool_indices.tolist(), prefix_lens.tolist()
    ):
        start = max(0, int(prefix) - window)
        if start >= prefix:
            continue
        locs = req_to_token[req_slot, start:prefix]
        pos = torch.arange(start, int(prefix), device=locs.device)
        scatter_dspark_swa_ring_mapping(
            mapping,
            req_slots=torch.full_like(pos, req_slot),
            positions=pos,
            full_locs=locs,
            window=window,
            ring=ring,
        )


def dspark_swa_decode_prefix_window(server_args) -> int:
    """Draft SWA window the D-side radix reuse must leave to fresh transfer
    under SGLANG_DSPARK_SWA_KV_POOL, 0 when the gate is off. A radix-cached
    prefix keeps its full-pool KV but not its sliding-layer ring rows, and its
    shared locs would alias two reqs' ring slots in full_to_swa_index_mapping,
    so the last `window` prompt tokens must come fresh from prefill."""
    if not envs.SGLANG_DSPARK_SWA_KV_POOL.get():
        return 0
    algo = getattr(server_args, "speculative_algorithm", None)
    if algo is None or not str(algo).upper().endswith("DSPARK"):
        return 0
    import json
    import os

    path = os.path.join(server_args.speculative_draft_model_path, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    window = int(cfg.get("dspark_sliding_window") or 0)
    if window <= 0:
        raise ValueError(
            f"SGLANG_DSPARK_SWA_KV_POOL=1 but {path} has no dspark_sliding_window"
        )
    return window


def dspark_swa_decode_prefix_cap(input_len: int, window: int, page_size: int) -> int:
    """Largest page-aligned reusable prefix whose end is at least `window`
    tokens before the prompt end."""
    cap = max(0, input_len - window)
    return cap // page_size * page_size


def dspark_pd_skip_swa_window(draft_model_path: str) -> int:
    """Draft SWA window prefill ships the sliding layers for under
    SGLANG_DSPARK_PD_SKIP_SWA_DRAFT_KV, 0 when the gate is off."""
    if not dspark_pd_skipped_draft_layers(draft_model_path):
        return 0
    import json
    import os

    path = os.path.join(draft_model_path, "config.json")
    with open(path) as f:
        window = int(json.load(f).get("dspark_sliding_window") or 0)
    if window <= 0:
        raise ValueError(
            f"SGLANG_DSPARK_PD_SKIP_SWA_DRAFT_KV=1 but {path} has no dspark_sliding_window"
        )
    return window


def dspark_draft_swa_state_buf_infos(
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    draft_layer_ids: List[int],
    target_layer_num: int,
):
    """DSPARK_DRAFT_SWA state component: the sliding draft layers' per-page
    buffers, tagged with global layer ids (target_layer_num + draft id) so
    prefill (full-length draft pool, picked layers) and decode (ring sub-pool,
    swa_attention_layer_ids order) pair entries by layer id."""
    return (
        list(data_ptrs),
        list(data_lens),
        list(item_lens),
        [target_layer_num + i for i in draft_layer_ids],
    )


def dspark_draft_swa_window_start(seq_len: int, window: int, page_size: int) -> int:
    """First position of the shipped sliding window, page aligned so both
    peers address whole pages; both sides derive it from the prompt length."""
    return max(0, seq_len - window) // page_size * page_size


def dspark_draft_swa_ring_pages(
    req_slot: int, seq_len: int, window: int, ring: int, page_size: int
) -> np.ndarray:
    """Decode-side ring pages for the window positions, in position order:
    ring row req_slot*ring + pos%ring, so position page k lands on ring page
    req_slot*ring/page_size + k % (ring/page_size) (ring is page aligned)."""
    assert ring % page_size == 0, (ring, page_size)
    ring_pages = ring // page_size
    start = dspark_draft_swa_window_start(seq_len, window, page_size) // page_size
    end = (seq_len + page_size - 1) // page_size
    pages = np.arange(start, end, dtype=np.int64)
    return (req_slot * ring_pages + pages % ring_pages).astype(np.int32)


@torch.inference_mode()
def scatter_dspark_swa_ring_mapping(
    mapping: torch.Tensor,
    *,
    req_slots: torch.Tensor,
    positions: torch.Tensor,
    full_locs: torch.Tensor,
    window: int,
    ring: int,
    final_pos: Optional[torch.Tensor] = None,
) -> None:
    """mapping[full_locs] = req_slot*ring + pos%ring for windowed-in tokens,
    -1 otherwise. Idempotent: safe to re-run over overlapping ranges, which
    also refreshes stale entries after loc recycling (retract / rebootstrap).

    mapping: int64 [size_full + pad]; req_slots/positions/full_locs: int64
    [n] per-token; final_pos: int64 [n] per-token (same expansion as
    req_slots) when given -- tokens older than final_pos-window map to -1."""
    if full_locs.numel() == 0:
        return
    req_slots = req_slots.to(torch.int64)
    positions = positions.to(torch.int64)
    full_locs = full_locs.to(torch.int64)
    swa_locs = req_slots * ring + positions % ring
    if final_pos is not None:
        keep = positions > final_pos.to(torch.int64) - window
        swa_locs = torch.where(keep, swa_locs, torch.full_like(swa_locs, -1))
    mapping[full_locs] = swa_locs
