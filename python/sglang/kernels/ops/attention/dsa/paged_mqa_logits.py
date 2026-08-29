# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache

import torch
import triton
import triton.language as tl


def _restore_row_stride(logits: torch.Tensor) -> torch.Tensor:
    """Undo PyTorch's DLPack stride normalization on single-row DeepGEMM logits.

    DeepGEMM returns paged-MQA logits as a row-padded view:
    ``torch.empty(num_rows, aligned_len)[:, :max_len]`` with ``aligned_len``
    256-element (1024-byte) aligned. tvm-ffi builds of DeepGEMM (sgl-deep-gemm
    >= 0.1.x) round-trip that view through DLPack on return, and PyTorch's
    DLPack *exporter* rewrites the stride of every size<2 dim to 1 whenever it
    differs from the packed expectation (pytorch/pytorch#83158). A one-row
    result whose row is actually padded (``max_len % 256 != 0``, e.g. any
    ``model context_len + 4`` page-table width at bs=1 decode capture) therefore
    arrives with ``stride() == (1, 1)`` instead of ``(aligned_len, 1)``, which
    violates the fused top-k v2 kernel ABI (``score_stride % 4 == 0``, enforced
    both in ``dsa_topk_backend._topk_transform_v2_paged`` and by the kernel's
    own RuntimeCheck).

    For ``num_rows <= 1`` the row stride is semantically arbitrary (row 0 is
    the only row ever addressed and ``(num_rows - 1) * stride(0)`` contributes
    nothing to the storage extent), so restoring a 16-byte-aligned value is a
    pure metadata rewrite: same storage, same data pointer, no copy and no
    kernel launch -- trivially CUDA-graph-capture-safe. Multi-row results keep
    their true strides through DLPack (no size<2 dim) and pass through
    untouched.
    """
    if logits.shape[0] <= 1 and logits.stride(0) % 4 != 0:
        width = logits.shape[1]
        logits = logits.as_strided((logits.shape[0], width), ((width + 3) // 4 * 4, 1))
    return logits


def deepgemm_paged_mqa_logits_native(
    fp8_paged_mqa_logits_fn: Callable[..., torch.Tensor],
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    *,
    q_offset: int,
    B: int,
    next_n: int,
) -> torch.Tensor:
    # block_tables[::next_n] de-expands the caller's repeat_interleave without a
    # copy (DeepGEMM only checks `stride(1) == 1`).
    logits = fp8_paged_mqa_logits_fn(
        q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2]),
        kv_cache_fp8,
        weights[:q_offset],
        ctx_lens_2d,
        block_tables[::next_n],
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )
    return _restore_row_stride(logits)


def deepgemm_paged_mqa_logits_split(
    fp8_paged_mqa_logits_fn: Callable[..., torch.Tensor],
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    *,
    q_offset: int,
) -> torch.Tensor:
    q_fp8 = q_fp8.unsqueeze(1)
    logits = fp8_paged_mqa_logits_fn(
        q_fp8[:q_offset],
        kv_cache_fp8,
        weights[:q_offset],
        ctx_lens_2d,
        block_tables,
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )
    return _restore_row_stride(logits)


@lru_cache(maxsize=1)
def _use_gfx936_accuracy_fallback() -> bool:
    if os.environ.get("SGLANG_TOKENFLOW_LEGACY_COMPAT") != "1":
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx936"


@triton.jit
def _fp8e4m3_to_f32(value, EXPONENT_BIAS: tl.constexpr):
    value = value.to(tl.uint8)
    sign = (value & 0x80) != 0
    magnitude = value & 0x7F
    exponent = (magnitude >> 3).to(tl.int32)
    mantissa = (magnitude & 0x7).to(tl.float32)
    decoded = tl.where(
        exponent == 0,
        mantissa * tl.exp2(-EXPONENT_BIAS - 2.0),
        (1.0 + mantissa * 0.125)
        * tl.exp2(exponent.to(tl.float32) - EXPONENT_BIAS),
    )
    return tl.where(sign, -decoded, decoded)


@triton.jit(
    do_not_specialize=["table_stride", "logits_stride", "NUM_TOKEN_PROGRAMS"]
)
def _fp8_paged_mqa_logits_gfx936_kernel(
    q_u8,
    cache_u8,
    weights,
    seq_lens,
    block_tables,
    logits,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    cache_stride: tl.constexpr,
    weight_stride: tl.constexpr,
    table_stride,
    logits_stride,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EXPONENT_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NUM_TOKEN_PROGRAMS,
):
    row = tl.program_id(0)
    seq_len = tl.load(seq_lens + row)
    tile = tl.program_id(1)
    dim = tl.arange(0, HEAD_DIM)

    while tile * BLOCK_T < seq_len:
        positions = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        valid = positions < seq_len
        pages = tl.load(
            block_tables + row * table_stride + positions,
            mask=valid,
            other=0,
        )
        k = _fp8e4m3_to_f32(
            tl.load(
                cache_u8 + pages[:, None] * cache_stride + dim[None, :],
                mask=valid[:, None],
                other=0,
            ),
            EXPONENT_BIAS,
        )
        score = tl.zeros((BLOCK_T,), tl.float32)

        for head in tl.static_range(NUM_HEADS):
            q = _fp8e4m3_to_f32(
                tl.load(
                    q_u8
                    + row * q_row_stride
                    + head * q_head_stride
                    + dim
                ),
                EXPONENT_BIAS,
            )
            dot = tl.sum(k * q[None, :], axis=1)
            weight = tl.load(weights + row * weight_stride + head)
            score += tl.maximum(dot, 0.0) * weight

        scale_base = cache_u8 + pages * cache_stride + HEAD_DIM
        s0 = tl.load(scale_base, mask=valid, other=0).to(tl.uint32)
        s1 = tl.load(scale_base + 1, mask=valid, other=0).to(tl.uint32)
        s2 = tl.load(scale_base + 2, mask=valid, other=0).to(tl.uint32)
        s3 = tl.load(scale_base + 3, mask=valid, other=0).to(tl.uint32)
        scale_bits = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24)
        scale = scale_bits.to(tl.float32, bitcast=True)
        tl.store(
            logits + row * logits_stride + positions,
            score * scale,
            mask=valid,
        )
        tile += NUM_TOKEN_PROGRAMS


def _triton_fp8_paged_mqa_logits_gfx936(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Graph-safe FP32 fallback for gfx936, which cannot dot FP8 inputs."""
    rows, num_heads, head_dim = q_fp8.shape
    block_size = kv_cache_fp8.shape[1]
    assert q_fp8.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert q_fp8.is_contiguous()
    assert block_size == 1
    assert head_dim == 128
    assert kv_cache_fp8.shape[2:] == (1, head_dim + 4)
    assert weights.shape == (rows, num_heads)
    assert seq_lens.numel() >= rows
    assert seq_lens.is_contiguous()
    assert block_tables.shape[0] >= rows

    logits = torch.full(
        (rows, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    cache_bytes = kv_cache_fp8.view(torch.uint8).reshape(
        kv_cache_fp8.shape[0], head_dim + 4
    )
    weights = weights.float().contiguous()
    block_t = 16
    num_token_programs = min(256, triton.cdiv(max_seq_len, block_t))
    _fp8_paged_mqa_logits_gfx936_kernel[(rows, num_token_programs)](
        q_fp8.view(torch.uint8),
        cache_bytes,
        weights,
        seq_lens.reshape(-1),
        block_tables,
        logits,
        q_fp8.stride(0),
        q_fp8.stride(1),
        cache_bytes.stride(0),
        weights.stride(0),
        block_tables.stride(0),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        EXPONENT_BIAS=8 if q_fp8.dtype == torch.float8_e4m3fnuz else 7,
        BLOCK_T=block_t,
        NUM_TOKEN_PROGRAMS=num_token_programs,
        num_warps=4,
    )
    return logits


def _torch_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Accuracy-first ragged MQA fallback for gfx936.

    gfx936 cannot compile AITER's Triton FP8 dot kernel. Convert the already
    quantized FP8 values to FP32 and process query rows in bounded chunks so
    the temporary ``[rows, heads, kv_tokens]`` scores stay below 128 MiB.
    """
    rows, num_heads, head_dim = q_fp8.shape
    kv_tokens = kv_fp8.shape[0]
    assert kv_fp8.shape == (kv_tokens, head_dim)
    assert kv_scales.numel() == kv_tokens
    assert weights.shape == (rows, num_heads)
    assert cu_starts.numel() >= rows and cu_ends.numel() >= rows

    logits = torch.empty(
        (rows, kv_tokens), dtype=torch.float32, device=q_fp8.device
    )
    if rows == 0 or kv_tokens == 0:
        return logits

    # Bound the largest temporary. Each score is FP32; Q/K conversions and the
    # final logits are much smaller than the per-head score tensor.
    score_budget_bytes = 128 * 1024 * 1024
    bytes_per_query_row = num_heads * kv_tokens * torch.float32.itemsize
    rows_per_chunk = max(1, score_budget_bytes // max(bytes_per_query_row, 1))
    rows_per_chunk = min(rows, rows_per_chunk)

    kv_f32_t = kv_fp8.float().transpose(0, 1)
    scales_f32 = kv_scales.reshape(-1).float().view(1, 1, kv_tokens)
    column_ids = torch.arange(kv_tokens, device=q_fp8.device).view(1, kv_tokens)

    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        chunk_rows = end - start
        scores = torch.matmul(
            q_fp8[start:end].float().reshape(chunk_rows * num_heads, head_dim),
            kv_f32_t,
        ).view(chunk_rows, num_heads, kv_tokens)
        scores.mul_(scales_f32).relu_()
        scores.mul_(weights[start:end].float().unsqueeze(-1))
        reduced = scores.sum(dim=1)

        valid = (column_ids >= cu_starts[start:end].reshape(-1, 1)) & (
            column_ids < cu_ends[start:end].reshape(-1, 1)
        )
        logits[start:end] = reduced.masked_fill(~valid, float("-inf"))

    return logits


def aiter_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
    *,
    clean_logits: bool,
) -> torch.Tensor:
    """Call the installed AITER ragged-MQA API across package layouts."""
    if _use_gfx936_accuracy_fallback():
        return _torch_fp8_mqa_logits(
            q_fp8, kv_fp8, kv_scales, weights, cu_starts, cu_ends
        )

    try:
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
    except ModuleNotFoundError as exc:
        if exc.name != "aiter.ops.triton.fp8_mqa_logits":
            raise
        # Newer AITER packages moved the wrapper and always clean invalid logits.
        from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

        return fp8_mqa_logits(
            q_fp8, kv_fp8, kv_scales, weights, cu_starts, cu_ends
        )

    return fp8_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scales,
        weights,
        cu_starts,
        cu_ends,
        clean_logits=clean_logits,
    )


def aiter_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    *,
    preshuffle: bool,
    kv_block_size: int,
) -> torch.Tensor:
    if _use_gfx936_accuracy_fallback():
        assert not preshuffle
        assert kv_block_size == 1
        return _triton_fp8_paged_mqa_logits_gfx936(
            q_fp8,
            kv_cache_fp8,
            weights,
            seq_lens,
            block_tables,
            max_seq_len,
        )

    try:
        from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
    except ModuleNotFoundError as exc:
        # Newer AITER releases moved this Triton kernel under attention/ while
        # keeping the callable contract unchanged.
        if exc.name != "aiter.ops.triton.pa_mqa_logits":
            raise
        import aiter.ops.triton.attention.pa_mqa_logits as pa_mqa_logits

        deepgemm_fp8_paged_mqa_logits = (
            pa_mqa_logits.deepgemm_fp8_paged_mqa_logits
        )

    q_fp8 = q_fp8.unsqueeze(1)
    batch_size, next_n, _, _ = q_fp8.shape
    logits = torch.empty(
        (batch_size * next_n, max_seq_len),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        logits,
        seq_lens,
        block_tables,
        max_seq_len,
        Preshuffle=preshuffle,
        KVBlockSize=kv_block_size,
    )
    return logits


def cutedsl_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_1d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    max_seq_len: int,
    *,
    q_offset: int,
    B: int,
    next_n: int,
    is_target_verify: bool,
    dsl_expand_factor: int,
    dsl_atom: int,
    blocksize: int,
    sm_count: int,
    get_paged_mqa_logits_metadata_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsa.cutedsl_paged_mqa_logits import (
        CuteDSLPagedMQALogitsRunner,
    )

    dsl_atom_split = dsl_expand_factor > 1 and next_n == dsl_expand_factor * dsl_atom
    if is_target_verify and dsl_atom_split:
        exp_B = B * dsl_expand_factor
        q_dsl = q_fp8[:q_offset].view(exp_B, dsl_atom, q_fp8.shape[1], q_fp8.shape[2])
        ctx_lens_1d = ctx_lens_1d.repeat_interleave(dsl_expand_factor)
        block_tables_dsl = block_tables[::next_n].repeat_interleave(
            dsl_expand_factor, dim=0
        )
        schedule_metadata = get_paged_mqa_logits_metadata_fn(
            ctx_lens_1d.unsqueeze(-1), blocksize, sm_count
        )
    elif is_target_verify and next_n >= 2:
        # Native single-launch: one task per batch entry (the kernel iterates
        # next_n internally), so the schedule must be built from B-length
        # context lens, not the caller's [B, next_n] or per-token layout.
        q_dsl = q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2])
        block_tables_dsl = block_tables[::next_n]
        schedule_metadata = get_paged_mqa_logits_metadata_fn(
            ctx_lens_1d.unsqueeze(-1), blocksize, sm_count
        )
    else:
        q_dsl = q_fp8[:q_offset].unsqueeze(1)
        block_tables_dsl = block_tables[:B]

    return CuteDSLPagedMQALogitsRunner.forward(
        q_dsl,
        kv_cache_fp8.view(torch.uint8),
        weights[:q_offset],
        ctx_lens_1d,
        block_tables_dsl,
        schedule_metadata,
        max_seq_len,
    )
