"""Spark-only deferred RoPE/concat/cache-store for the audited HCU MLA layout.

The compiled plugin's post-prepare clamp/smooth remains in place. A plan is
carried by ForwardBatch rather than changing TokenFlow's nine-field tuple ABI.
"""

from dataclasses import dataclass
import logging
import os

import torch

logger = logging.getLogger(__name__)
_logged = False


@dataclass(frozen=True)
class SparkMLAPreprocessPlan:
    layer_id: int
    query_rope: torch.Tensor
    key_rope: torch.Tensor
    positions: torch.Tensor
    cosine_sine: torch.Tensor
    cache: torch.Tensor
    neox: bool


def maybe_defer_spark_mla(
    attn, forward_batch, positions, query_nope, query_rope, key_nope, key_rope,
    *, fusion_plan, llama_4_scaling,
):
    if os.getenv("SGLANG_SPARK_LIGHTOP_PREPROCESS", "0") != "1":
        return False
    if (type(attn).__name__ != "SparkV1AttentionMLA" or fusion_plan is not None
            or llama_4_scaling is not None
            or not (forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_decode())
            or getattr(attn, "_q8kv8_qprep_overlap_pending", False)):
        return False
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend, get_token_to_kv_pool,
    )
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    parallel = get_parallel()
    backend, pool = get_attn_backend(), get_token_to_kv_pool()
    if (parallel.attn_tp_size != 1 or parallel.dcp_enabled
            or type(backend).__name__ != "DeepseekSparseAttnBackend"
            or backend.dsa_decode_impl != "flashmla_kv"
            or type(pool).__name__ != "DSATokenToKVPool"
            or not pool.dsa_kv_cache_store_fp8
            or getattr(pool.set_mla_kv_buffer, "__func__", None)
            is not MLATokenToKVPool.set_mla_kv_buffer
            or getattr(pool._write_mla_kv_buffer, "__func__", None)
            is not MLATokenToKVPool._write_mla_kv_buffer):
        return False
    rope = attn.rotary_emb
    cache = rope.cos_sin_cache
    tokens = positions.numel()
    if (rope.head_size != 64 or rope.rotary_dim != 64
            or cache.dtype != torch.bfloat16 or cache.ndim != 2
            or cache.shape[1] != 64 or cache.stride(1) != 1
            or positions.ndim != 1 or positions.dtype != torch.int64
            or not positions.is_contiguous() or not 0 < tokens <= 36
            or query_nope is None or query_nope.ndim != 3
            or query_nope.shape != (tokens, 128, 512)
            or query_rope.shape != (tokens, 128, 64)
            or key_nope.shape != (tokens, 1, 512)
            or key_rope.shape != (tokens, 1, 64)):
        return False
    tensors = (query_nope, query_rope, key_nope, key_rope, cache)
    if not all(t.is_cuda and t.dtype == torch.bfloat16 and t.stride(-1) == 1
               and t.device == positions.device for t in tensors):
        return False
    locations = forward_batch.out_cache_loc
    if (locations.dtype != torch.int64 or not locations.is_contiguous()
            or locations.numel() != tokens or locations.device != positions.device):
        return False
    destination = pool.kv_buffer[attn.layer_id - pool.start_layer]
    if (destination.dtype != torch.uint8 or not destination.is_contiguous()
            or destination.ndim != 3 or destination.shape[1:] != (1, 656)):
        return False
    from lightop import op

    if not hasattr(op, "spark_mla_rope_cat_store"):
        raise RuntimeError("Spark MLA preprocessing requires the matching LightOp wheel")
    if forward_batch.spark_mla_preprocess is not None:
        raise RuntimeError("Unconsumed Spark MLA preprocessing plan")
    forward_batch.spark_mla_preprocess = SparkMLAPreprocessPlan(
        layer_id=attn.layer_id, query_rope=query_rope, key_rope=key_rope,
        positions=positions, cosine_sine=cache, cache=destination,
        neox=bool(rope.is_neox_style),
    )
    return True


def consume_spark_mla(attn, forward_batch, query_nope, query_rope, key_nope,
                      key_rope, positions):
    plan = forward_batch.spark_mla_preprocess
    if plan is None:
        return None
    forward_batch.spark_mla_preprocess = None
    # The plugin may replace the normalized/smoothed latent tensors. Its RoPE
    # inputs must stay untouched until this operator applies the native rotation.
    if (plan.layer_id != attn.layer_id or plan.query_rope is not query_rope
            or plan.key_rope is not key_rope or plan.positions is not positions):
        raise RuntimeError("Spark MLA post-prepare contract changed")
    from lightop import op

    query = torch.empty((*query_nope.shape[:-1], 576), dtype=query_nope.dtype,
                        device=query_nope.device)
    key = torch.empty_like(key_rope, memory_format=torch.contiguous_format)
    op.spark_mla_rope_cat_store(
        positions, query_nope, query_rope, key_nope, key_rope,
        plan.cosine_sine, forward_batch.out_cache_loc, query, key, plan.cache,
        plan.neox, 1e-6, 8,
    )
    global _logged
    if not _logged:
        logger.info("Spark LightOp block-specialized RoPE/Q-concat/FP8-KV-store active")
        _logged = True
    return query, key
