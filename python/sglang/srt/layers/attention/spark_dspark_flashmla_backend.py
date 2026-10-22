"""Sparse BF16 FlashMLA for Spark's eight-token DSpark proposal.

The inherited Triton backend owns cache writes, physical slot translation and
eager/Graph metadata. Only its final extend-attention kernel is replaced. The
historical FP8 prefix is gathered into a small BF16 window; the eight fresh KV
rows retain their BF16 values and bidirectional visibility.
"""

from __future__ import annotations

import logging
import math

import torch

from sglang.srt.layers.attention.spark_dspark_flashmla_ops import (
    BLOCK,
    KEY_DIM,
    VALUE_DIM,
    WINDOW,
    allocate_workspace,
    attention_from_window,
    gather_draft_window,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

logger = logging.getLogger(__name__)

MASK_WINDOW = WINDOW + BLOCK
NUM_HEADS = 128
SOFTMAX_SCALE = 192**-0.5


class SparkDSparkFlashMLAAttnBackend(TritonAttnBackend):
    """Opt-in draft backend; other attention shapes retain the native path."""

    def __init__(self, model_runner):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        if not model_runner.is_draft_worker:
            raise ValueError("spark_draft_flashmla is a Spark DSpark draft backend")
        architecture = model_runner.model_config.hf_config.architectures
        if architecture != ["SparkV1DSparkDraftModel"]:
            raise ValueError(f"Unsupported draft architecture: {architecture}")
        if not isinstance(model_runner.token_to_kv_pool, MLATokenToKVPool):
            raise ValueError("Spark draft FlashMLA requires the native MLA KV pool")
        if model_runner.sliding_window_size != WINDOW:
            raise ValueError("Spark draft's historical-prefix window must be 512")

        super().__init__(model_runner)
        if not self.use_mla or self.dcp_size != 1:
            raise ValueError("Spark draft FlashMLA requires unsharded MLA attention")
        if self.enable_deterministic or self.use_aiter_triton_extend_fwd:
            raise ValueError("This backend requires the native extend-attention path")
        if self.use_verify_splitkv or self.use_verify_shared_kv:
            raise ValueError("Another verify-kernel override is already enabled")
        if self.token_to_kv_pool.dtype != torch.float8_e4m3fn:
            raise ValueError("Spark draft FlashMLA requires FP8 E4M3 prefix storage")

        self._native_extend_attention_fwd = self.extend_attention_fwd
        self._draft_windows = {}
        self._allocate_draft_windows(self.req_to_token_pool.size)
        self.extend_attention_fwd = torch.compiler.disable(self._extend_flashmla)
        logger.info(
            "Spark DSpark FlashMLA: prefix=%d, mask=%d, block=%d, "
            "FP8 prefix -> BF16 window; native KV writes/metadata retained",
            WINDOW,
            MASK_WINDOW,
            BLOCK,
        )

    def _allocate_draft_windows(self, max_bs):
        for batch_size in range(1, max_bs + 1):
            if batch_size not in self._draft_windows:
                self._draft_windows[batch_size] = allocate_workspace(
                    batch_size,
                    self.device,
                    window_size=WINDOW,
                    mask_window_size=MASK_WINDOW,
                )

    def init_cuda_graph_state(
        self,
        max_bs,
        max_num_tokens,
        kv_indices_buf=None,
        cuda_graph_num_kv_splits_buf=None,
    ):
        # All scratch addresses exist before graph capture. Replays only refill
        # these buffers and the parent's existing physical-index metadata.
        # DSpark proposes a TARGET_VERIFY block, so it needs no Triton
        # single-token decode logits/LSE scratch. A later decode capture can
        # still allocate that scratch through the parent's capture path.
        self._allocate_draft_windows(max_bs)
        return super().init_cuda_graph_state(
            max_bs,
            max_num_tokens,
            kv_indices_buf,
            cuda_graph_num_kv_splits_buf,
            allocate_decode_scratch=False,
        )

    @staticmethod
    def _supports_flashmla(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        custom_mask,
        is_causal,
        max_len_extend,
        k_scale,
        v_scale,
        sm_scale,
        logit_cap,
        sliding_window_size,
        sinks,
        xai_temperature_len,
        lse_extend,
        skip_prefix,
        skip_extend,
        score_mod,
        aux_tensors,
    ):
        batch_size = qo_indptr.numel() - 1
        return (
            batch_size > 0
            and max_len_extend == BLOCK
            and q_extend.shape == (batch_size * BLOCK, NUM_HEADS, KEY_DIM)
            and k_extend.shape == (batch_size * BLOCK, 1, KEY_DIM)
            and v_extend.shape == (batch_size * BLOCK, 1, VALUE_DIM)
            and o_extend.shape == (batch_size * BLOCK, NUM_HEADS, VALUE_DIM)
            and q_extend.dtype == k_extend.dtype == v_extend.dtype == torch.bfloat16
            and o_extend.dtype == torch.bfloat16
            and k_buffer.ndim == v_buffer.ndim == 3
            and k_buffer.shape[1:] == (1, KEY_DIM)
            and v_buffer.shape[1:] == (1, VALUE_DIM)
            and k_buffer.dtype == v_buffer.dtype == torch.float8_e4m3fn
            and k_buffer.data_ptr() == v_buffer.data_ptr()
            and k_buffer.stride(0) == v_buffer.stride(0)
            and q_extend.is_contiguous()
            and o_extend.is_contiguous()
            and not is_causal
            and custom_mask is None
            and isinstance(k_scale, (int, float))
            and k_scale == v_scale
            and math.isfinite(k_scale)
            and k_scale > 0
            and isinstance(sm_scale, (int, float))
            and math.isclose(sm_scale, SOFTMAX_SCALE, rel_tol=1e-7)
            and logit_cap == 0
            and sliding_window_size == MASK_WINDOW
            and sinks is None
            and xai_temperature_len <= 0
            and lse_extend is None
            and not skip_prefix
            and not skip_extend
            and score_mod is None
            and aux_tensors is None
        )

    def _extend_flashmla(
        self,
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        is_causal,
        mask_indptr,
        max_len_extend,
        k_scale,
        v_scale,
        sm_scale=None,
        logit_cap=0.0,
        skip_prefix_custom_mask=True,
        sliding_window_size=-1,
        sinks=None,
        window_kv_offsets=None,
        xai_temperature_len=-1,
        lse_extend=None,
        skip_prefix=False,
        skip_extend=False,
        page_size=1,
        score_mod=None,
        aux_tensors=None,
        extend_seq_lens_cpu=None,
    ):
        if not self._supports_flashmla(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            custom_mask,
            is_causal,
            max_len_extend,
            k_scale,
            v_scale,
            sm_scale,
            logit_cap,
            sliding_window_size,
            sinks,
            xai_temperature_len,
            lse_extend,
            skip_prefix,
            skip_extend,
            score_mod,
            aux_tensors,
        ):
            # Warmup, prompt KV injection and unsupported semantics retain the
            # original implementation. The optimized route must be verified
            # explicitly in the full-model profile before claiming a gain.
            return self._native_extend_attention_fwd(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                custom_mask,
                is_causal,
                mask_indptr,
                max_len_extend,
                k_scale,
                v_scale,
                sm_scale,
                logit_cap=logit_cap,
                skip_prefix_custom_mask=skip_prefix_custom_mask,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=xai_temperature_len,
                lse_extend=lse_extend,
                skip_prefix=skip_prefix,
                skip_extend=skip_extend,
                page_size=page_size,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
            )

        batch_size = qo_indptr.numel() - 1
        workspace = self._draft_windows[batch_size]
        gather_draft_window(
            k_buffer,
            kv_indptr,
            kv_indices,
            k_extend,
            qo_indptr,
            workspace,
            prefix_scale=k_scale,
        )
        output = attention_from_window(q_extend, workspace, sm_scale)
        o_extend.copy_(output)
