"""Opt-in fixed-shape context injection for Spark DSpark's raw FP8 draft KV."""

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)


def supports_masked_context_commit(
    pool, target_hidden, positions, loc_2d, commit_lens
):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.runtime_context import get_parallel

    if type(pool) is MLATokenToKVPool:
        dtype_pool = pool
    elif type(pool) is SWAKVPool:
        # Ring draft pool (SGLANG_DSPARK_SWA_KV_POOL=1): both sub-pools are
        # MLA fp8; sliding layers write through full_to_swa_index_mapping.
        if pool.full_to_swa_index_mapping is None:
            return False
        dtype_pool = pool.full_kv_pool
    else:
        return False
    if any(value is None for value in (target_hidden, positions, loc_2d, commit_lens)):
        return False
    if not (
        dtype_pool.dtype == torch.float8_e4m3fn
        and dtype_pool.store_dtype == torch.uint8
        and not dtype_pool.use_dsa
        and not dtype_pool.dsa_kv_cache_store_fp8
        and get_parallel().attn_dcp_size == 1
    ):
        return False
    return (
        target_hidden.is_cuda
        and target_hidden.dtype == torch.bfloat16
        and target_hidden.ndim == 2
        and positions.device == target_hidden.device
        and positions.dtype == torch.int64
        and positions.ndim == 1
        and loc_2d.device == target_hidden.device
        and loc_2d.dtype == torch.int64
        and loc_2d.ndim == 2
        and loc_2d.is_contiguous()
        and 0 <= loc_2d.shape[0] <= 4
        and loc_2d.shape[1] > 0
        and commit_lens.device == target_hidden.device
        and commit_lens.dtype == torch.int32
        and commit_lens.shape == (loc_2d.shape[0],)
        and commit_lens.is_contiguous()
        and target_hidden.shape[0] == positions.numel() == loc_2d.numel()
    )


def write_masked_context_commit(
    model, *, pool, target_hidden, positions, loc_2d, commit_lens
):
    from sglang.srt.layers.attention.spark_dspark_committed_kv_ops import (
        store_committed_mla_fp8,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    if target_hidden.shape[0] == 0:
        return
    # These projections and normalizations are independent for every token.
    # Use the native implementation and its existing graph cache on all rows;
    # only the final cache writes depend on the accepted-prefix lengths.
    projected = model._project_context_kv(target_hidden, positions)
    # SWAKVPool (ring draft pool): sliding layers write the ring slot through
    # full_to_swa_index_mapping; -1 entries (windowed-out) are skipped by the
    # store kernel. get_key_buffer already routes to the right sub-pool.
    is_ring_pool = type(pool) is SWAKVPool
    swa_loc_2d = pool.full_to_swa_index_mapping[loc_2d] if is_ring_pool else None
    for layer, (nope, rope) in zip(model.model.layers, projected):
        cache = pool.get_key_buffer(layer.self_attn.attn_mqa.layer_id)
        locs = loc_2d
        if is_ring_pool and pool.layers_mapping[layer.self_attn.attn_mqa.layer_id][1]:
            locs = swa_loc_2d
        store_committed_mla_fp8(cache, locs, commit_lens, nope, rope)


def _is_ring_swa_pool(pool) -> bool:
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    return type(pool) is SWAKVPool and pool.full_to_swa_index_mapping is not None


def write_ring_swa_commit(
    model, *, pool, main_hidden, positions, swa_loc
):
    """Sliding-layer half of the commit injection for the ring draft pool.

    swa_loc is the flat [bs*stride] int32 ring-slot tensor from
    BuildCommitInjectLayout: -1 already encodes both the commit gate and the
    window filter, so the store runs with a single all-committed group
    (commit_lens = [rows]) and relies on the kernel's loc >= 0 skip. Full
    layers are skipped -- their KV for committed tokens is what the draft
    forward itself wrote into the full sub-pool."""
    from sglang.srt.layers.attention.spark_dspark_committed_kv_ops import (
        store_committed_mla_fp8,
    )

    if main_hidden is None or main_hidden.shape[0] == 0:
        return
    rows = main_hidden.shape[0]
    projected = model._project_context_kv(main_hidden, positions)
    loc_2d = swa_loc.reshape(1, rows).to(torch.int64)
    commit_lens = torch.full((1,), rows, dtype=torch.int32, device=swa_loc.device)
    for layer, (nope, rope) in zip(model.model.layers, projected):
        layer_id = layer.self_attn.attn_mqa.layer_id
        sub_layer_id, is_swa = pool.layers_mapping[layer_id]
        if not is_swa:
            continue
        cache = pool.swa_kv_pool.get_key_buffer(sub_layer_id)
        store_committed_mla_fp8(cache, loc_2d, commit_lens, nope, rope)


def install_spark_dspark_committed_kv_adapter():
    if os.environ.get("SGLANG_SPARK_DSPARK_MASKED_KV_COMMIT", "0") != "1":
        return False
    from tokenflow.models.sparkv1.dspark_draft import SparkV1DSparkDraftModel

    original = SparkV1DSparkDraftModel.write_target_hidden_kv
    if getattr(original, "_sglang_masked_fp8_commit", False):
        return True

    @functools.wraps(original)
    def write_target_hidden_kv(
        self,
        *,
        target_hidden=None,
        pool,
        positions,
        cache_loc=None,
        cache_loc_2d=None,
        commit_lens=None,
        main_hidden=None,
        swa_loc=None,
    ):
        if main_hidden is None and swa_loc is None and supports_masked_context_commit(
            pool, target_hidden, positions, cache_loc_2d, commit_lens
        ):
            return write_masked_context_commit(
                self,
                pool=pool,
                target_hidden=target_hidden,
                positions=positions,
                loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
        if swa_loc is not None and _is_ring_swa_pool(pool):
            # The native writer type-checks MLATokenToKVPool and cannot
            # address the ring sub-pool; write the sliding layers here. Full
            # layers keep the draft forward's own KV for committed tokens.
            return write_ring_swa_commit(
                self,
                pool=pool,
                main_hidden=main_hidden if main_hidden is not None else target_hidden,
                positions=positions,
                swa_loc=swa_loc,
            )
        return original(
            self,
            target_hidden=target_hidden,
            pool=pool,
            positions=positions,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
            main_hidden=main_hidden,
            swa_loc=swa_loc,
        )

    write_target_hidden_kv._sglang_masked_fp8_commit = True
    SparkV1DSparkDraftModel.write_target_hidden_kv = write_target_hidden_kv
    logger.info(
        "Spark DSpark masked FP8 KV commit enabled: native fixed-shape context "
        "projection; rejected rows are not written; no host-visible compaction"
    )
    return True
