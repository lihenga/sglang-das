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
    from sglang.srt.runtime_context import get_parallel

    if type(pool) is not MLATokenToKVPool:
        return False
    if any(value is None for value in (target_hidden, positions, loc_2d, commit_lens)):
        return False
    if not (
        pool.dtype == torch.float8_e4m3fn
        and pool.store_dtype == torch.uint8
        and not pool.use_dsa
        and not pool.dsa_kv_cache_store_fp8
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

    if target_hidden.shape[0] == 0:
        return
    # These projections and normalizations are independent for every token.
    # Use the native implementation and its existing graph cache on all rows;
    # only the final cache writes depend on the accepted-prefix lengths.
    projected = model._project_context_kv(target_hidden, positions)
    for layer, (nope, rope) in zip(model.model.layers, projected):
        cache = pool.get_key_buffer(layer.self_attn.attn_mqa.layer_id)
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
