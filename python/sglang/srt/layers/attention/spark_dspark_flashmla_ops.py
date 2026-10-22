"""Spark draft's FP8-prefix/BF16-extend window for BF16 sparse FlashMLA.

Eight query tokens share one gathered prefix. Per-query sparse indices retain the exact left-window mask.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

WINDOW = 512
BLOCK = 8
KEY_DIM = 576
VALUE_DIM = 512
GATHER_WIDTH = WINDOW + BLOCK
INDEX_WIDTH = 640


@triton.jit
def _gather_window(
    cache, prefix_indptr, prefix_indices, extend, qo_indptr, gathered,
    cache_row_stride: tl.constexpr, cache_dim_stride: tl.constexpr,
    extend_row_stride: tl.constexpr, extend_dim_stride: tl.constexpr,
    prefix_scale,
    COLS: tl.constexpr, DIMS: tl.constexpr, WINDOW_SIZE: tl.constexpr,
    KEY_WIDTH: tl.constexpr, GATHER_SLOTS: tl.constexpr,
):
    request = tl.program_id(0)
    columns = tl.program_id(1) * COLS + tl.arange(0, COLS)
    dims = tl.program_id(2) * DIMS + tl.arange(0, DIMS)
    prefix_begin = tl.load(prefix_indptr + request)
    raw_prefix_length = tl.load(prefix_indptr + request + 1) - prefix_begin
    prefix_length = tl.minimum(raw_prefix_length, WINDOW_SIZE)
    prefix_begin += raw_prefix_length - prefix_length
    extend_begin = tl.load(qo_indptr + request)
    extend_length = tl.load(qo_indptr + request + 1) - extend_begin

    valid_prefix = (columns < prefix_length) & (columns < WINDOW_SIZE)
    locations = tl.load(prefix_indices + prefix_begin + columns,
                        mask=valid_prefix, other=0).to(tl.int64)
    prefix = tl.load(
        cache + locations[:, None] * cache_row_stride + dims[None, :] * cache_dim_stride,
        mask=valid_prefix[:, None] & (dims[None, :] < KEY_WIDTH), other=0.0,
    ).to(tl.float32) * prefix_scale

    extend_column = columns - WINDOW_SIZE
    valid_extend = (extend_column >= 0) & (extend_column < extend_length)
    current = tl.load(
        extend + (extend_begin + extend_column[:, None]) * extend_row_stride
        + dims[None, :] * extend_dim_stride,
        mask=valid_extend[:, None] & (dims[None, :] < KEY_WIDTH), other=0.0,
    ).to(tl.float32)
    value = tl.where(valid_prefix[:, None], prefix, current)
    tl.store(
        gathered + (request * GATHER_SLOTS + columns[:, None]) * KEY_WIDTH + dims[None, :],
        value,
        mask=(columns[:, None] < GATHER_SLOTS) & (dims[None, :] < KEY_WIDTH),
    )


@triton.jit
def _window_indices(prefix_indptr, qo_indptr, indices, lengths, WIDTH: tl.constexpr,
                    QUERY_BLOCK: tl.constexpr, WINDOW_SIZE: tl.constexpr,
                    GATHER_SLOTS: tl.constexpr, INDEX_COLS: tl.constexpr,
                    MASK_WINDOW: tl.constexpr):
    row = tl.program_id(0)
    request, query = row // QUERY_BLOCK, row % QUERY_BLOCK
    prefix_length = tl.minimum(
        tl.load(prefix_indptr + request + 1) - tl.load(prefix_indptr + request),
        WINDOW_SIZE,
    )
    extend_length = tl.load(qo_indptr + request + 1) - tl.load(qo_indptr + request)
    # Existing Triton reference uses q_pos <= k_pos + window, including equality.
    start = tl.maximum(0, prefix_length + query - MASK_WINDOW)
    prefix_visible = prefix_length - start
    total = tl.where(query < extend_length, prefix_visible + extend_length, 0)
    columns = tl.arange(0, WIDTH)
    local = tl.where(columns < prefix_visible, start + columns,
                     WINDOW_SIZE + columns - prefix_visible)
    selected = request * GATHER_SLOTS + local
    tl.store(indices + row * INDEX_COLS + columns,
             tl.where(columns < total, selected, -1), mask=columns < INDEX_COLS)
    tl.store(lengths + row, total)


@triton.jit
def _zero_empty_rows(output, lengths, ROW_WIDTH: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(lengths + row) == 0:
        lane = tl.arange(0, TILE)
        for start in range(0, tl.cdiv(ROW_WIDTH, TILE)):
            columns = start * TILE + lane
            tl.store(output + row * ROW_WIDTH + columns, 0.0, mask=columns < ROW_WIDTH)


@dataclass
class DraftWindowWorkspace:
    kv: torch.Tensor
    indices: torch.Tensor
    lengths: torch.Tensor
    window_size: int
    mask_window_size: int


def allocate_workspace(batch_size, device, window_size=WINDOW, mask_window_size=None):
    # Spark uses512 for the backend's prefix metadata and520 for the layer's
    # mask (512 + block8). These are separate contracts, not one window value.
    assert 0 < window_size and window_size + BLOCK <= INDEX_WIDTH
    if mask_window_size is None:
        mask_window_size = window_size
    assert mask_window_size > 0
    return DraftWindowWorkspace(
        kv=torch.empty((batch_size, window_size + BLOCK, KEY_DIM), dtype=torch.bfloat16, device=device),
        indices=torch.empty((batch_size * BLOCK, 1, INDEX_WIDTH), dtype=torch.int32, device=device),
        lengths=torch.empty((batch_size * BLOCK,), dtype=torch.int32, device=device),
        window_size=window_size,
        mask_window_size=mask_window_size,
    )


def gather_draft_window(cache, prefix_indptr, prefix_indices, extend, qo_indptr,
                        workspace, prefix_scale=1.0):
    batch_size = prefix_indptr.numel() - 1
    window_size = workspace.window_size
    gather_width = window_size + BLOCK
    assert cache.ndim == extend.ndim == 3
    assert cache.shape[1:] == extend.shape[1:] == (1, KEY_DIM)
    assert workspace.kv.shape == (batch_size, gather_width, KEY_DIM)
    assert qo_indptr.numel() == batch_size + 1
    if batch_size == 0:
        return
    _gather_window[(batch_size, triton.cdiv(gather_width, 8), triton.cdiv(KEY_DIM, 128))](
        cache, prefix_indptr, prefix_indices, extend, qo_indptr, workspace.kv,
        cache.stride(0), cache.stride(2), extend.stride(0), extend.stride(2),
        prefix_scale, COLS=8, DIMS=128, WINDOW_SIZE=window_size, KEY_WIDTH=KEY_DIM,
        GATHER_SLOTS=gather_width, num_warps=4,
    )
    _window_indices[(batch_size * BLOCK,)](
        prefix_indptr, qo_indptr, workspace.indices, workspace.lengths,
        WIDTH=triton.next_power_of_2(INDEX_WIDTH), QUERY_BLOCK=BLOCK,
        WINDOW_SIZE=window_size, GATHER_SLOTS=gather_width, INDEX_COLS=INDEX_WIDTH,
        MASK_WINDOW=workspace.mask_window_size,
        num_warps=4,
    )


def attention_from_window(q, workspace, softmax_scale):
    from flash_mla import flash_mla_sparse_fwd

    assert q.ndim == 3 and q.shape[0] == workspace.lengths.numel()
    assert q.dtype == torch.bfloat16 and q.shape[2] == KEY_DIM
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], VALUE_DIM))
    output = flash_mla_sparse_fwd(
        q=q, kv=workspace.kv.view(-1, 1, KEY_DIM), indices=workspace.indices,
        sm_scale=softmax_scale, d_v=VALUE_DIM, topk_length=workspace.lengths,
    )[0]
    _zero_empty_rows[(q.shape[0],)](
        output, workspace.lengths, ROW_WIDTH=q.shape[1] * VALUE_DIM, TILE=1024,
        num_warps=4,
    )
    return output
