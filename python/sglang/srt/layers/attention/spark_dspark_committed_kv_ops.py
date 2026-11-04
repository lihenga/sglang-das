"""Fixed-shape prefix commits into Spark draft's native raw FP8 MLA cache."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["VERIFY_STRIDE"])
def _store_committed_mla_bytes(
    cache,
    locs,
    commit_lens,
    nope,
    rope,
    CACHE_ROW: tl.constexpr,
    NOPE_ROW: tl.constexpr,
    ROPE_ROW: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    VERIFY_STRIDE,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(1)
    offset = tl.program_id(0)
    row = group * VERIFY_STRIDE + offset
    committed = offset < tl.load(commit_lens + group)
    # Rejected rows may contain invalid locations. Do not load their location,
    # their projected values, or form a cache write for them. loc -1 is the
    # windowed-out sentinel (DSPARK ring draft pool): skip it too.
    if committed:
        loc = tl.load(locs + row)
        if loc >= 0:
            cols = tl.arange(0, BLOCK)
            k = tl.load(nope + row * NOPE_ROW + cols, cols < NOPE_DIM, other=0)
            r = tl.load(rope + row * ROPE_ROW + cols, cols < ROPE_DIM, other=0)
            tl.store(cache + loc * CACHE_ROW + cols, k, cols < NOPE_DIM)
            tl.store(cache + loc * CACHE_ROW + NOPE_DIM + cols, r, cols < ROPE_DIM)


def store_committed_mla_fp8(cache, loc_2d, commit_lens, nope, rope):
    """Keep the native Torch FP8 cast, then copy only committed rows as bytes."""
    rows = loc_2d.numel()
    if rows == 0:
        return
    assert cache.is_cuda and cache.is_contiguous()
    assert cache.dtype in (torch.uint8, torch.float8_e4m3fn)
    assert loc_2d.ndim == 2 and loc_2d.is_contiguous()
    assert loc_2d.dtype == torch.int64 and loc_2d.device == cache.device
    assert commit_lens.shape == (loc_2d.shape[0],)
    assert commit_lens.dtype == torch.int32 and commit_lens.is_contiguous()
    assert commit_lens.device == cache.device
    assert nope.shape[0] == rope.shape[0] == rows
    assert nope.device == rope.device == cache.device
    assert nope.dtype == rope.dtype == torch.bfloat16
    nope_dim, rope_dim = nope.shape[-1], rope.shape[-1]
    assert nope.numel() == rows * nope_dim
    assert rope.numel() == rows * rope_dim
    assert cache.numel() == cache.shape[0] * (nope_dim + rope_dim)

    # MLATokenToKVPool._write_mla_kv_buffer casts each BF16 component with
    # Tensor.to(dtype), then reinterprets its bytes. Retain that exact rounding
    # rather than substituting a different Triton FP8 conversion on HIP.
    nope_bytes = nope.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    rope_bytes = rope.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    cache_bytes = cache.view(torch.uint8)
    _store_committed_mla_bytes[(loc_2d.shape[1], loc_2d.shape[0])](
        cache_bytes,
        loc_2d,
        commit_lens,
        nope_bytes,
        rope_bytes,
        cache_bytes.stride(0),
        nope_bytes.stride(0),
        rope_bytes.stride(0),
        nope_dim,
        rope_dim,
        loc_2d.shape[1],
        triton.next_power_of_2(max(nope_dim, rope_dim)),
    )
