"""RoPE-only fusion preserving native BF16 intermediate rounding.

No normalization, clamp, quantization, cache generation or cache-store changes.
Unsupported contracts return None for the original native implementation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _native_bf16_rope_kernel(Q, K, P, C, OQ, OK,
    T: tl.constexpr, QH: tl.constexpr, KH: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    PS: tl.constexpr, CS: tl.constexpr, NEOX: tl.constexpr,
    QB: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid < QB:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        valid = idx < T * QH * 64
        d = idx % 64
        h = idx // 64 % QH
        row = idx // (64 * QH)
        ptr, out = Q, OQ
        base = row * QS0 + h * QS1
        stride = QS2
    else:
        idx = (pid - QB) * BLOCK + tl.arange(0, BLOCK)
        valid = idx < T * KH * 64
        d = idx % 64
        h = idx // 64 % KH
        row = idx // (64 * KH)
        ptr, out = K, OK
        base = row * KS0 + h * KS1
        stride = KS2
    if NEOX:
        partner = (d + 32) % 64
        freq = d % 32
        negative = d < 32
    else:
        partner = d ^ 1
        freq = d // 2
        negative = d % 2 == 0
    pos = tl.load(P + row * PS, valid, other=0)
    x = tl.load(ptr + base + d * stride, valid, other=0).to(tl.float32)
    y = tl.load(ptr + base + partner * stride, valid, other=0).to(tl.float32)
    c = tl.load(C + pos * CS + freq, valid, other=0).to(tl.float32)
    s = tl.load(C + pos * CS + freq + 32, valid, other=0).to(tl.float32)
    y = tl.where(negative, -y, y)
    # Native eager materializes BOTH BF16 products before its BF16 addition.
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    b = (y * s).to(tl.bfloat16).to(tl.float32)
    tl.store(out + idx, a + b, valid)


def try_native_bf16_rope(rope, positions, query, key, offsets=None):
    cache = rope.cos_sin_cache
    if (offsets is not None or rope.rotary_dim != 64 or rope.head_size != 64
        or query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16
        or cache.dtype != torch.bfloat16 or not query.is_cuda
        or positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64)
        or cache.ndim != 2 or cache.shape[1] != 64 or cache.stride(1) != 1
        or query.ndim not in (3, 4) or key.ndim != query.ndim
        or (query.ndim == 4 and (query.shape[0] != 1 or key.shape[0] != 1))
        or query.shape[-1] != 64 or key.shape[-1] != 64
        or query.shape[-3] != positions.numel() or key.shape[-3] != positions.numel()
        or query.shape[-2] == 0 or key.shape[-2] == 0
        or not (query.device == key.device == cache.device == positions.device)):
        return None
    qshape, kshape = query.shape, key.shape
    # Native NeoX repeat(1,1,2) adds a leading singleton to 3D inputs.
    if rope.is_neox_style and query.ndim == 3:
        qshape, kshape = (1, *qshape), (1, *kshape)
    oq = torch.empty(qshape, dtype=query.dtype, device=query.device)
    ok = torch.empty(kshape, dtype=key.dtype, device=key.device)
    if not positions.numel():
        return oq, ok
    block = 256
    qb, kb = triton.cdiv(query.numel(), block), triton.cdiv(key.numel(), block)
    _native_bf16_rope_kernel[(qb + kb,)](
        query, key, positions, cache, oq, ok, positions.numel(),
        query.shape[-2], key.shape[-2], *query.stride()[-3:], *key.stride()[-3:],
        positions.stride(0), cache.stride(0), rope.is_neox_style, qb, block,
        num_warps=4, enable_fp_fusion=False)
    return oq, ok
