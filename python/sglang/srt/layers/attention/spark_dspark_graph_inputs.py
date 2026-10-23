"""Capture the small Spark draft input copies with an indirect source table.

Only source addresses and live sizes are uploaded per replay. The copy/padding
kernel is part of the existing draft graph, before attention metadata refresh.
Other input layouts retain the registry's normal fill path.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _refresh_spark_draft_inputs(
    sources, dst_ids, dst_positions, dst_locs, dst_reqs, dst_lens,
    dst_mrope, dst_count, dst_global, dst_logprob, full_positions, full_locs,
    LOC_INT32: tl.constexpr, REQ_INT32: tl.constexpr, LENS_INT32: tl.constexpr,
    SEQ_FILL: tl.constexpr, MROPE_STRIDE: tl.constexpr,
    HAS_COUNT: tl.constexpr, FILL_GLOBAL: tl.constexpr,
    GLOBAL_SIZE: tl.constexpr, LOGPROB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    sampling_temperatures, sampling_greedy,
    HAS_SAMPLER: tl.constexpr,
):
    enabled = tl.load(sources + 9)
    if enabled:
        ids = tl.load(sources).to(tl.pointer_type(tl.int64))
        positions = tl.load(sources + 1).to(tl.pointer_type(tl.int64))
        locs = tl.load(sources + 2).to(tl.pointer_type(tl.int32 if LOC_INT32 else tl.int64))
        reqs = tl.load(sources + 3).to(tl.pointer_type(tl.int32 if REQ_INT32 else tl.int64))
        lens = tl.load(sources + 4).to(tl.pointer_type(tl.int32 if LENS_INT32 else tl.int64))
        raw_bs = tl.load(sources + 5)
        padded_bs = tl.load(sources + 6)
        raw_tokens = tl.load(sources + 7)
        padded_tokens = tl.load(sources + 8)
        i = tl.arange(0, BLOCK)
        valid_token = i < raw_tokens
        generated = tl.load(sources + 10)
        if generated:
            rows = i // 8
            columns = i % 8
            prefix = tl.load(lens + rows, valid_token, other=0)
            slot = tl.load(reqs + rows, valid_token, other=0)
            bonus_address = tl.load(sources + 11)
            if tl.load(sources + 15):
                bonus = tl.load(bonus_address.to(tl.pointer_type(tl.int32)) + rows, valid_token, other=0).to(tl.int64)
            else:
                bonus = tl.load(bonus_address.to(tl.pointer_type(tl.int64)) + rows, valid_token, other=0)
            mask_id = tl.load(sources + 12)
            table = tl.load(sources + 13).to(tl.pointer_type(tl.int32))
            stride = tl.load(sources + 14)
            token_positions = prefix + columns
            locations = tl.load(table + slot * stride + token_positions, valid_token, other=0)
            tl.store(dst_ids + i, tl.where(columns == 0, bonus, mask_id), valid_token)
            tl.store(dst_positions + i, tl.where(valid_token, token_positions, 0), i < padded_tokens)
            tl.store(dst_locs + i, locations, i < padded_tokens)
            # The target verifies the bonus plus eight proposal tokens.
            full_rows = i // 9
            full_columns = i % 9
            valid_full = i < raw_bs * 9
            full_prefix = tl.load(lens + full_rows, valid_full, other=0)
            full_slot = tl.load(reqs + full_rows, valid_full, other=0)
            full_pos = full_prefix + full_columns
            full_loc = tl.load(table + full_slot * stride + full_pos, valid_full, other=0)
            tl.store(full_positions + i, full_pos, valid_full)
            tl.store(full_locs + i, full_loc, valid_full)
        else:
            tl.store(dst_ids + i, tl.load(ids + i, valid_token, other=0), valid_token)
            tl.store(dst_positions + i, tl.load(positions + i, valid_token, other=0), i < padded_tokens)
            tl.store(dst_locs + i, tl.load(locs + i, valid_token, other=0), i < padded_tokens)
        valid_req = i < raw_bs
        tl.store(dst_reqs + i, tl.load(reqs + i, valid_req, other=0), i < padded_bs)
        tl.store(dst_lens + i, tl.load(lens + i, valid_req, other=SEQ_FILL), i < padded_bs)
        for axis in tl.static_range(3):
            tl.store(dst_mrope + axis * MROPE_STRIDE + i, 0,
                     (i >= raw_tokens) & (i < padded_tokens))
        if HAS_COUNT:
            tl.store(dst_count, raw_tokens)
        if FILL_GLOBAL:
            tl.store(dst_global + i, padded_tokens, i < GLOBAL_SIZE)
            tl.store(dst_logprob + i, padded_tokens, i < LOGPROB_SIZE)

    if HAS_SAMPLER:
        mode = tl.load(sources + 18)
        if mode:
            i = tl.arange(0, BLOCK)
            raw_bs = tl.load(sources + 5)
            if mode == 1:
                temp_ptr = tl.load(sources + 16).to(tl.pointer_type(tl.float32))
                topk_ptr = tl.load(sources + 17).to(tl.pointer_type(tl.int32))
                temperature = tl.load(temp_ptr + i, i < raw_bs, other=1.0)
                topk = tl.load(topk_ptr + i, i < raw_bs, other=1)
            else:
                temperature = tl.full((BLOCK,), 1.0, tl.float32)
                topk = tl.full((BLOCK,), 1, tl.int32)
            temperature = tl.maximum(temperature, 1e-5,
                                     propagate_nan=tl.PropagateNan.ALL)
            tl.store(sampling_temperatures + i, temperature, i < raw_bs)
            tl.store(sampling_greedy + i, (topk <= 1).to(tl.int8), i < raw_bs)


_ALLOWED_SLOTS = frozenset({
    "input_ids", "positions", "out_cache_loc", "req_pool_indices", "seq_lens",
    "seq_lens_cpu", "mrope_positions", "num_token_non_padded",
    "global_num_tokens_gpu", "global_num_tokens_for_logprob_gpu",
})


@dataclass(frozen=True)
class SparkGraphSamplingSources:
    temperatures: Optional[torch.Tensor]
    top_ks: Optional[torch.Tensor]


@dataclass(frozen=True)
class SparkDSparkGraphInputs:
    """Per-forward sources for integer-only draft/verify input generation."""

    bonus_tokens: torch.Tensor
    mask_token_id: int
    req_to_token: torch.Tensor
    sampling: Optional[SparkGraphSamplingSources] = None


class SparkDraftGraphInputStager:
    """A bounded pinned upload ring and a graph-resident input refresh kernel."""

    def __init__(self, buffers, registry, *, seq_len_fill_value, require_gathered_buffer,
                 attn_tp_size, enable_prefill_cp, max_bs, captured_req_width,
                 enable_pdmux=False, enable_two_batch_overlap=False,
                 is_encoder_decoder=False, has_lora=False, pp_size=1, sampler=None):
        if (attn_tp_size != 1 or enable_prefill_cp or max_bs > 4
                or enable_pdmux or enable_two_batch_overlap or is_encoder_decoder
                or has_lora or pp_size != 1
                or captured_req_width != 8
                or buffers.global_num_tokens_gpu.numel() > 64
                or buffers.global_num_tokens_for_logprob_gpu.numel() > 64
                or not set(registry.slot_names()).issubset(_ALLOWED_SLOTS)):
            raise ValueError("Captured Spark draft input staging requires the audited BS1..4 layout")
        self.buffers = buffers
        self.seq_len_fill_value = seq_len_fill_value
        self.require_gathered_buffer = require_gathered_buffer
        self.full_positions = torch.zeros((max_bs, 9), dtype=torch.int64, device=buffers.input_ids.device)
        self.full_locs = torch.zeros_like(self.full_positions)
        self.sampler = (sampler if sampler is not None
                        and sampler.folded_sampling and sampler._graph_params_enabled else None)
        if self.sampler is not None:
            self.sampler._graph_input_stager = self
        self.sources = torch.zeros(19, dtype=torch.int64, device=buffers.input_ids.device)
        # Storage and supported shapes are immutable for this graph runner.
        # Cache aliases only; contents are refreshed for every replay.
        self.generated_views = []
        for bs in range(max_bs + 1):
            ids = buffers.input_ids[:bs * 8]
            full_locs = self.full_locs[:bs]
            self.generated_views.append(SimpleNamespace(
                input_ids=ids, block_ids=ids.view(bs, 8),
                positions=buffers.positions[:bs * 8],
                locations=buffers.out_cache_loc[:bs * 8],
                full_positions=self.full_positions[:bs],
                full_locations=full_locs, flat_locations=full_locs.reshape(-1),
            ))
        self._cpu_mirrors = {
            (raw, padded): (buffers.seq_lens_cpu[:raw],
                            buffers.seq_lens_cpu[raw:padded])
            for padded in range(max_bs + 1) for raw in range(padded + 1)
        }
        self._host_ring = []
        for _ in range(16):
            host = torch.zeros(19, dtype=torch.int64, pin_memory=True)
            self._host_ring.append((host, host.numpy(), torch.cuda.Event(), False))
        self._next_host = 0
        self._served_batch = False
        self._source_dtypes = tuple(getattr(buffers, name).dtype for name in (
            "input_ids", "positions", "out_cache_loc", "req_pool_indices", "seq_lens"))
        if self._source_dtypes[:2] != (torch.int64, torch.int64):
            raise ValueError("Spark draft token IDs and positions must use int64")

    def supports_generation(self, req_indices, seq_lens, bonus_tokens, req_to_token):
        return (
            req_indices.is_cuda and req_indices.is_contiguous()
            and req_indices.dtype == self._source_dtypes[3]
            and seq_lens.is_cuda and seq_lens.is_contiguous()
            and seq_lens.dtype == self._source_dtypes[4]
            and bonus_tokens.is_cuda and bonus_tokens.is_contiguous()
            and bonus_tokens.dtype in (torch.int32, torch.int64)
            and bonus_tokens.numel() == seq_lens.numel()
            and req_indices.numel() == seq_lens.numel()
            and seq_lens.numel() <= self.buffers.req_pool_indices.numel()
            and req_to_token.is_cuda and req_to_token.is_contiguous()
            and req_to_token.dtype == torch.int32 and req_to_token.ndim == 2
            and all(t.device == self.buffers.input_ids.device for t in
                    [req_indices, seq_lens, bonus_tokens, req_to_token])
        )

    def prepare_capture(self):
        if self._served_batch:
            raise RuntimeError("Reinitialize the Spark input stager before recapturing a live runner")
        self.sources.zero_()

    def stage(self, forward_batch, *, raw_bs, padded_bs, raw_num_tokens,
              padded_num_tokens, pp_proxy_tensors=None, **unused):
        self._served_batch = True
        buffers = self.buffers
        sources = [forward_batch.input_ids, forward_batch.positions,
                   forward_batch.out_cache_loc, forward_batch.req_pool_indices,
                   forward_batch.seq_lens]
        supported = (
            pp_proxy_tensors is None
            and getattr(forward_batch, "mrope_positions", None) is None
            and getattr(forward_batch, "encoder_lens", None) is None
            and forward_batch.num_token_non_padded_cpu == raw_num_tokens
            and 0 <= raw_bs <= padded_bs <= 4
            and raw_num_tokens == raw_bs * 8
            and padded_num_tokens == padded_bs * 8
            and all(x is not None and x.is_cuda and x.ndim == 1 and x.is_contiguous()
                    and x.dtype == dtype and x.device == buffers.input_ids.device
                    for x, dtype in zip(sources, self._source_dtypes))
        )
        generated = getattr(forward_batch, "spark_dspark_graph_inputs", None)
        if generated is not None:
            if not supported or not self.supports_generation(
                forward_batch.req_pool_indices, forward_batch.seq_lens,
                generated.bonus_tokens, generated.req_to_token,
            ):
                raise ValueError("Generated Spark draft inputs require the validated graph layout")
        # The scheduling pipeline has at most two forwards in flight. Keeping
        # 16 upload slots also permits CPU/GPU enqueue stress tests; never reuse
        # pinned host bytes before their asynchronous H2D copy completes.
        index = self._next_host
        host, view, ready, used = self._host_ring[index]
        if used and not ready.query():
            raise RuntimeError("Spark input upload ring exceeded 16 outstanding copies")
        self._next_host = (index + 1) % len(self._host_ring)
        stream = torch.cuda.current_stream()
        if supported:
            for tensor in sources:
                tensor.record_stream(stream)
            view[:] = [*(x.data_ptr() for x in sources), raw_bs, padded_bs,
                       raw_num_tokens, padded_num_tokens, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if generated is not None:
                generated.bonus_tokens.record_stream(stream)
                generated.req_to_token.record_stream(stream)
                view[10:16] = [1, generated.bonus_tokens.data_ptr(), generated.mask_token_id,
                             generated.req_to_token.data_ptr(), generated.req_to_token.shape[1],
                             int(generated.bonus_tokens.dtype == torch.int32)]
        else:
            view[:] = 0
            view[5] = raw_bs
        if self.sampler is not None:
            sampling = generated.sampling if generated is not None else None
            if sampling is not None and sampling.temperatures is None:
                view[16:19] = [0, 0, 2]
            else:
                temperatures = (sampling.temperatures if sampling is not None
                                else self.sampler._raw_temperatures)
                top_ks = (sampling.top_ks if sampling is not None
                          else self.sampler._raw_top_ks)
                temperatures.record_stream(stream)
                top_ks.record_stream(stream)
                view[16:19] = [temperatures.data_ptr(), top_ks.data_ptr(), 1]
        self.sources.copy_(host, non_blocking=True)
        ready.record(stream)
        self._host_ring[index] = (host, view, ready, True)
        if not supported:
            return False
        host_lens = forward_batch.seq_lens_cpu
        head, tail = self._cpu_mirrors[(raw_bs, padded_bs)]
        if host_lens is not None:
            head.copy_(host_lens)
        if raw_bs < padded_bs:
            tail.fill_(self.seq_len_fill_value)
        return True

    def capture_refresh(self):
        b = self.buffers
        _refresh_spark_draft_inputs[(1,)](
            self.sources, b.input_ids, b.positions, b.out_cache_loc,
            b.req_pool_indices, b.seq_lens, b.mrope_positions,
            b.num_token_non_padded if b.num_token_non_padded is not None else b.req_pool_indices,
            b.global_num_tokens_gpu, b.global_num_tokens_for_logprob_gpu,
            self.full_positions, self.full_locs,
            LOC_INT32=self._source_dtypes[2] == torch.int32,
            REQ_INT32=self._source_dtypes[3] == torch.int32,
            LENS_INT32=self._source_dtypes[4] == torch.int32,
            SEQ_FILL=self.seq_len_fill_value, MROPE_STRIDE=b.mrope_positions.stride(0),
            HAS_COUNT=b.num_token_non_padded is not None,
            FILL_GLOBAL=self.require_gathered_buffer,
            GLOBAL_SIZE=b.global_num_tokens_gpu.numel(),
            LOGPROB_SIZE=b.global_num_tokens_for_logprob_gpu.numel(), BLOCK=64,
            sampling_temperatures=(self.sampler.temperatures if self.sampler is not None else b.input_ids),
            sampling_greedy=(self.sampler.greedy_mask.view(torch.int8) if self.sampler is not None else b.input_ids),
            HAS_SAMPLER=self.sampler is not None,
        )
