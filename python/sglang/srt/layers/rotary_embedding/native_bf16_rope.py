# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Optional LightOp RoPE adapter; unsupported cases use the caller's native path."""

from contextvars import ContextVar
from functools import lru_cache

import torch

_ROPE_OUTPUT = ContextVar("spark_native_rope_output", default=None)


@lru_cache(maxsize=1)
def _lightop_rope():
    try:
        from lightop.attention.rope import (
            ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI,
            rotary_embedding_deepseek_fuse,
        )
    except ImportError:
        return None
    if ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI not in (1, 2):
        return None
    return rotary_embedding_deepseek_fuse, ROTARY_EMBEDDING_BF16_ROUNDED_OUT_ABI


def try_native_bf16_rope(rope, positions, query, key, offsets=None):
    cache = rope.cos_sin_cache
    if (
        offsets is not None
        or rope.rotary_dim != 64
        or rope.head_size != 64
        or query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or cache.dtype != torch.bfloat16
        or not query.is_cuda
        or positions.ndim != 1
        or positions.dtype != torch.int64
        or cache.ndim != 2
        or cache.shape[1] != 64
        or cache.shape[0] == 0
        or cache.stride(1) != 1
        or query.ndim not in (3, 4)
        or key.ndim != query.ndim
        or (query.ndim == 4 and (query.shape[0] != 1 or key.shape[0] != 1))
        or query.shape[-1] != 64
        or key.shape[-1] != 64
        or query.shape[-2] == 0
        or key.shape[-2] == 0
        or query.shape[-3] != positions.numel()
        or key.shape[-3] != positions.numel()
        or not (query.device == key.device == cache.device == positions.device)
    ):
        return None
    loaded = _lightop_rope()
    if loaded is None:
        return None
    operation, abi = loaded
    q = query.squeeze(0) if query.ndim == 4 else query
    k = key.squeeze(0) if key.ndim == 4 else key
    query_out = _ROPE_OUTPUT.get()
    if query_out is None:
        query_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    elif not (abi >= 2 and query_out.shape == q.shape and query_out.dtype == q.dtype
              and query_out.device == q.device and query_out.stride(2) == 1
              and query_out.stride(1) >= 64
              and query_out.stride(0) >= q.shape[1] * query_out.stride(1)):
        return None
    key_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    operation(
        positions, q, k, 64, cache, rope.is_neox_style,
        round_bf16_products=True, query_out=query_out, key_out=key_out,
    )
    # Match the existing YaRN forward_native result layout exactly.
    if query.ndim == 4 or rope.is_neox_style:
        query_out = query_out.unsqueeze(0)
        key_out = key_out.unsqueeze(0)
    return query_out, key_out


def run_native_bf16_rope_to(rope, positions, query, key, output):
    token = _ROPE_OUTPUT.set(output)
    try:
        return rope(positions, query, key)
    finally:
        _ROPE_OUTPUT.reset(token)
