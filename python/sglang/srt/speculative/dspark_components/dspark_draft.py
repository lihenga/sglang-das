from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.environ import envs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_planner import (
    VerifyWindow,
    alloc_verify_window,
)
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    spec_scale_global_num_tokens,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import is_hcu
from sglang.srt.utils.invariants import Bucket, Invariant, NotNaN, expect

logger = logging.getLogger(__name__)


def _one_hot_token0(probs: torch.Tensor) -> torch.Tensor:
    degenerate = torch.isnan(probs[:, :1])
    one_hot = torch.zeros_like(probs)
    one_hot[:, 0] = 1.0
    return torch.where(degenerate, one_hot, probs)


# Draft step logits: NaN is a bug, but -inf is legitimate (masking). The data
# layer lives downstream (probs one-hot below, or the fast kernel's clamp).
_DRAFT_STEP_LOGITS = Invariant("dspark.draft.step_logits", Bucket.GUARD, NotNaN())
# Draft sampling probs: SOFTEN (tolerate + count), matching the original
# unconditional clamp; an all-NaN row would otherwise make multinomial raise.
_DRAFT_PROBS = Invariant(
    "dspark.draft.probs", Bucket.SOFTEN, NotNaN(), recover=_one_hot_token0
)


def _make_num_token_non_padded(
    num_tokens: int, device: str | torch.device
) -> Optional[torch.Tensor]:
    if not enable_num_token_non_padded():
        return None
    return torch.tensor(num_tokens, dtype=torch.int32).to(device, non_blocking=True)


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool
    verify_window: Optional[VerifyWindow] = None


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    confidence_tap: Optional[torch.Tensor] = None
    folded: bool = False
    verify_window: Optional[VerifyWindow] = None


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
    prefill_tail_hidden_states: torch.Tensor | None = None,
    prefill_tail_valid_mask: torch.Tensor | None = None,
    prefill_tail_start_positions: torch.Tensor | None = None,
    prefill_tail_hidden_projected: bool = True,
) -> DFlashDraftInputV2:
    return make_draft_input_v2(
        bonus_tokens=bonus_tokens,
        new_seq_lens=new_seq_lens,
        prefill_tail_hidden_states=prefill_tail_hidden_states,
        prefill_tail_valid_mask=prefill_tail_valid_mask,
        prefill_tail_start_positions=prefill_tail_start_positions,
        prefill_tail_hidden_projected=prefill_tail_hidden_projected,
    )


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    greedy_mask = resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            expect(_DRAFT_STEP_LOGITS, step_logits, msg=f"step {step_idx}")
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            expect(_DRAFT_STEP_LOGITS, step_logits, msg=f"step {step_idx}")
            if fast_sampling:
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                probs = expect(_DRAFT_PROBS, probs)
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    draft_tokens, corrected_logits = markov_head.sample_block(
        base_logits,
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=sampler,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
    )


class DraftBlockProposer:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        gamma: int,
        mask_token_id: int,
        draft_block_spec_info,
        dp_moe_sync: bool = False,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.gamma = gamma
        self._mask_token_id = mask_token_id
        self._draft_block_spec_info = draft_block_spec_info
        self._draft_sampler = None
        self._dp_moe_sync = dp_moe_sync
        self._prep_cache_enabled = (
            is_hcu()
            and type(draft_model).__name__ == "SparkV1DSparkDraftModel"
            and os.getenv("SGLANG_HCU_DSPARK_FAST_PREP", "0") == "1"
        )
        self._prep_tensor_cache = OrderedDict()
        if self._prep_cache_enabled:
            logger.info(
                "Spark DSpark immutable count metadata cache enabled; "
                "bounded to 16 entries"
            )

    def _cached_count_tensor(self, values, dtype):
        # Constants are never mutated; new request mixes get separate storage.
        # The LRU bounds retained memory when serving batch compositions change.
        key = (dtype, values)
        tensor = self._prep_tensor_cache.get(key)
        if tensor is None:
            tensor = torch.tensor(
                values, dtype=dtype, device=self.draft_model_runner.device
            )
            self._prep_tensor_cache[key] = tensor
            if len(self._prep_tensor_cache) > 16:
                self._prep_tensor_cache.popitem(last=False)
        else:
            self._prep_tensor_cache.move_to_end(key)
        tensor.record_stream(torch.get_device_module(tensor.device).current_stream())
        return tensor


    def attach_draft_sampler(self, draft_sampler) -> None:
        self._draft_sampler = draft_sampler
        backend = getattr(self.draft_model_runner, "attn_backend", None)
        bind = getattr(backend, "bind_draft_sampler", None)
        if bind is not None:
            bind(draft_sampler)

    def _base_logits_context(self):
        if self._dp_moe_sync:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def _generated_input_stager(self):
        runner = self.draft_model_runner.decode_cuda_graph_runner
        backend = getattr(runner, "attn_backend", None)
        stager = getattr(backend, "_input_stager", None)
        if (
            self.gamma != 8 or stager is None
            or not envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get()
            or not hasattr(self.draft_model, "forward_embed")
        ):
            return None
        return stager

    def can_generate_window(self, batch, draft_input, req_to_token):
        stager = self._generated_input_stager()
        return (
            stager is not None and batch.can_run_dp_cuda_graph
            and stager.supports_generation(
                batch.req_pool_indices, batch.seq_lens,
                draft_input.bonus_tokens, req_to_token,
            )
        )

    def _native_draft_inputs(self, verify_window, draft_input, bs, device):
        ids = torch.full(
            (bs, self.gamma), int(self._mask_token_id), dtype=torch.long, device=device
        )
        ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        positions = verify_window.positions_2d[:, :self.gamma].reshape(-1)
        locations = verify_window.verify_cache_loc_2d[:, :self.gamma].reshape(-1)
        return ids, positions, locations

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: Optional[VerifyWindow],
        bs: int,
        device: str,
        target_model,
        sampling_info,
        block_pos_offsets=None,
        target_model_runner=None,
    ) -> DraftProposal:
        embed_module = unwrap_lora_layer(target_model.get_input_embeddings())
        draft_sampler = self._draft_sampler
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        fwd = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embed_module,
            draft_sampler=draft_sampler,
            sampling_info=sampling_info,
            block_pos_offsets=block_pos_offsets,
            target_model_runner=target_model_runner,
        )
        draft_block_ids = fwd.draft_block_ids

        folded_confidence = None
        confidence_tap = None
        folded = False
        if (
            envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get()
            and draft_sampler is not None
            and fwd.can_run_graph
            and (all_greedy or draft_sampler.folded_sampling)
        ):
            folded = True
            if draft_sampler.folded_sampling:
                greedy_mask = draft_sampler.greedy_mask[:bs]
                temperatures = draft_sampler.temperatures[:bs]
                # The sampling accept path needs the markov-corrected block
                # logits; greedy accept only compares tokens.
                corrected_logits = (
                    None
                    if all_greedy
                    else draft_sampler.corrected_out[: bs * self.gamma].view(
                        bs, self.gamma, -1
                    )
                )
            else:
                # Greedy-only folding: the hook argmaxed every row and kept no
                # sampling buffers, so derive the params on the fly.
                greedy_mask = resolve_greedy_mask(
                    bs=bs, sampling_info=sampling_info, device=device
                )
                if sampling_info is None:
                    temperatures = torch.ones(bs, dtype=torch.float32, device=device)
                else:
                    temperatures = (
                        sampling_info.temperatures.view(-1)
                        .to(torch.float32)
                        .clamp_min(1e-5)
                    )
                corrected_logits = None
            draft_block = DraftBlockResult(
                draft_tokens=draft_sampler.out[: bs * self.gamma].view(bs, self.gamma),
                corrected_logits=corrected_logits,
                greedy_mask=greedy_mask,
                temperatures=temperatures,
            )
            if draft_sampler.confidence_out is not None:
                folded_confidence = draft_sampler.confidence_out[:bs]
        else:
            with self._base_logits_context():
                base_logits, confidence_tap = self.draft_model.compute_base_logits(
                    fwd.raw_hidden
                )
                base_logits = base_logits.view(bs, self.gamma, -1)
            draft_block = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=draft_block_ids[:, 0],
                draft_hidden=fwd.draft_hidden_3d,
                sampling_info=sampling_info,
                markov_head=self.draft_model.markov_head,
                device=device,
            )
        return DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=fwd.draft_hidden_3d,
            confidence=folded_confidence,
            confidence_tap=confidence_tap,
            folded=folded,
            verify_window=fwd.verify_window,
        )

    def run_idle_participation(self, batch: ScheduleBatch) -> None:
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        device = self.draft_model_runner.device
        empty_long = torch.empty((0,), dtype=torch.int64, device=device)
        idle_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=empty_long,
            req_pool_indices=empty_long,
            seq_lens=empty_long,
            out_cache_loc=empty_long,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            positions=empty_long,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_num_token_non_padded(idle_batch)
        self._fill_dp_moe_sync_metadata(idle_batch, batch)
        graph_runner = self.draft_model_runner.decode_cuda_graph_runner
        if graph_runner is None or not graph_runner.can_run_graph(idle_batch):
            self._detach_eager_count_metadata(idle_batch)
        with torch.inference_mode():
            result = self.draft_model_runner.forward(idle_batch)
            head = self.draft_model.lm_head
            if (
                getattr(head, "_node_local_lm_head_group", None) is not None
                and not result.can_run_graph
            ):
                # Captured draft graphs already include the projection tail.
                # Eager idle ranks must join the active peers' head collective.
                empty_hidden = head.weight.new_empty((0, head.weight.shape[1]))
                with self._base_logits_context():
                    self.draft_model.compute_base_logits(empty_hidden)

    def _run_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: Optional[VerifyWindow],
        bs: int,
        device: str,
        embed_module,
        draft_sampler=None,
        sampling_info=None,
        block_pos_offsets=None,
        target_model_runner=None,
    ) -> DraftForwardResult:
        gamma = self.gamma
        prefix_lens = batch.seq_lens
        generated_inputs = None
        generated_views = None
        if verify_window is None:
            from sglang.srt.layers.attention.spark_dspark_graph_inputs import (
                SparkDSparkGraphInputs,
            )

            stager = self._generated_input_stager()
            assert stager is not None and target_model_runner is not None
            generated_views = stager.generated_views[bs]
            draft_block_ids = generated_views.block_ids
            draft_positions = generated_views.positions
            draft_cache_loc = generated_views.locations
            generated_inputs = SparkDSparkGraphInputs(
                bonus_tokens=draft_input.bonus_tokens,
                mask_token_id=int(self._mask_token_id),
                req_to_token=target_model_runner.req_to_token_pool.req_to_token,
                sampling=(draft_sampler.graph_sampling_sources(sampling_info, bs)
                          if draft_sampler is not None else None),
            )
            verify_window = VerifyWindow(
                positions_2d=generated_views.full_positions,
                verify_cache_loc=generated_views.flat_locations,
                verify_cache_loc_2d=generated_views.full_locations,
            )
        else:
            draft_block_ids, draft_positions, draft_cache_loc = self._native_draft_inputs(
                verify_window, draft_input, bs, device
            )

        draft_owns_embed = envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get() and hasattr(
            self.draft_model, "forward_embed"
        )
        draft_input_embeds: Optional[torch.Tensor] = None
        if not draft_owns_embed:
            noise_embedding = embed_module(draft_block_ids)
            draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu = batch.seq_lens_cpu + gamma
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            draft_seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            draft_seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        else:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")

        draft_num_tokens = bs * gamma
        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=(generated_views.input_ids if generated_views is not None
                       else draft_block_ids.flatten()),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=draft_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=draft_seq_lens_cpu,
            positions=draft_positions,
            input_embeds=draft_input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            spark_dspark_graph_inputs=generated_inputs,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=(
                None if self._prep_cache_enabled
                else _make_num_token_non_padded(draft_num_tokens, device)
            ),
            num_token_non_padded_cpu=draft_num_tokens,
        )
        self._fill_num_token_non_padded(draft_forward_batch)
        self._fill_dp_moe_sync_metadata(draft_forward_batch, batch)
        graph_runner = self.draft_model_runner.decode_cuda_graph_runner
        can_run_graph = graph_runner is not None and graph_runner.can_run_graph(
            draft_forward_batch
        )
        if draft_sampler is not None and can_run_graph:
            draft_sampler.stage_sampling_params(
                bs=bs, sampling_info=sampling_info,
                deferred=(generated_inputs is not None and generated_inputs.sampling is not None),
            )
        if not can_run_graph:
            if generated_inputs is not None:
                # Eligibility can change with the shared DP graph tier. Materialize
                # the original private tensors before taking the eager path.
                verify_window = alloc_verify_window(
                    batch=batch, bs=bs, device=device,
                    verify_num_draft_tokens=gamma + 1,
                    block_pos_offsets=block_pos_offsets,
                    model_runner=target_model_runner,
                )
                draft_block_ids, draft_positions, draft_cache_loc = self._native_draft_inputs(
                    verify_window, draft_input, bs, device
                )
                draft_forward_batch.input_ids = draft_block_ids.flatten()
                draft_forward_batch.positions = draft_positions
                draft_forward_batch.out_cache_loc = draft_cache_loc
                draft_forward_batch.spark_dspark_graph_inputs = None
            self._detach_eager_count_metadata(draft_forward_batch)
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        logits_output = draft_out.logits_output
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        draft_hidden_3d = raw_hidden.view(bs, gamma, -1)
        return DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=raw_hidden,
            draft_hidden_3d=draft_hidden_3d,
            can_run_graph=draft_out.can_run_graph,
            verify_window=verify_window,
        )

    def _detach_eager_count_metadata(self, forward_batch: ForwardBatch) -> None:
        # The graph registry copies count constants into its own storage. Eager
        # MLP-sync padding instead writes the live batch tensors in place, so it
        # needs private storage rather than an alias of an immutable cache entry.
        if not self._prep_cache_enabled:
            return
        for name in (
            "num_token_non_padded",
            "global_num_tokens_gpu",
            "global_num_tokens_for_logprob_gpu",
        ):
            tensor = getattr(forward_batch, name, None)
            if tensor is not None:
                setattr(forward_batch, name, tensor.clone())

    def _fill_num_token_non_padded(self, forward_batch: ForwardBatch) -> None:
        """Populate num_token_non_padded, mirroring ForwardBatch.init_new.

        The draft and idle batches here are built by calling ForwardBatch
        directly, which skips the init_new block that sets these fields. They
        are only used when moe_ep_size > 1: with EP the decode CUDA graph
        registers a num_token_non_padded slot whose post_fill converts this
        global count into a per-attn-TP-rank local one, so leaving it None
        makes DP+EP+DSpark decode fail with a TypeError on the first idle
        participation. Kept independent of _dp_moe_sync, which gates a
        different concern and is off for the draft batch in some configs.
        """
        num_tokens = (
            len(forward_batch.input_ids) if forward_batch.input_ids is not None else 0
        )
        if enable_num_token_non_padded():
            if self._prep_cache_enabled:
                forward_batch.num_token_non_padded = self._cached_count_tensor(
                    num_tokens, torch.int32
                )
            else:
                forward_batch.num_token_non_padded = (
                    torch.tensor(num_tokens, dtype=torch.int32)
                    .pin_memory()
                    .to(self.draft_model_runner.device, non_blocking=True)
                )
        forward_batch.num_token_non_padded_cpu = num_tokens

    def _fill_dp_moe_sync_metadata(
        self, forward_batch: ForwardBatch, batch: ScheduleBatch
    ) -> None:
        # The dense DSpark draft still reuses the target batch's graph tier.
        # Set graph eligibility before the DP-MoE-only metadata early return.
        forward_batch.can_run_dp_cuda_graph = batch.can_run_dp_cuda_graph
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        # Graph bucket selection uses the raw per-rank request counts.  Keep
        # them separate from global_num_tokens_cpu below, which is scaled into
        # draft-token units for DP/MoE synchronization.
        forward_batch.original_global_num_tokens_cpu = batch.global_num_tokens
        from sglang.srt.layers.node_local_lm_head import populate_lm_head_dp_metadata

        populate_lm_head_dp_metadata(forward_batch, batch)
        gnt, gnt_logprob = spec_scale_global_num_tokens(
            self._draft_block_spec_info,
            batch.global_num_tokens,
            batch.global_num_tokens_for_logprob,
        )
        device = self.draft_model_runner.device
        forward_batch.original_global_num_tokens_cpu = batch.global_num_tokens
        num_tokens = forward_batch.input_ids.numel()
        if enable_num_token_non_padded():
            if self._prep_cache_enabled:
                # _fill_num_token_non_padded has already installed this value.
                if forward_batch.num_token_non_padded is None:
                    forward_batch.num_token_non_padded = self._cached_count_tensor(
                        num_tokens, torch.int32
                    )
            else:
                forward_batch.num_token_non_padded = torch.tensor(
                    num_tokens, dtype=torch.int32, device=device
                )
        forward_batch.num_token_non_padded_cpu = num_tokens
        forward_batch.global_num_tokens_cpu = gnt
        forward_batch.global_num_tokens_for_logprob_cpu = gnt_logprob
        if self._prep_cache_enabled:
            forward_batch.global_num_tokens_gpu = self._cached_count_tensor(
                tuple(gnt), torch.int64
            )
            forward_batch.global_num_tokens_for_logprob_gpu = self._cached_count_tensor(
                tuple(gnt_logprob), torch.int64
            )
        else:
            forward_batch.global_num_tokens_gpu = torch.tensor(gnt, dtype=torch.int64).to(
                device, non_blocking=True
            )
            forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
                gnt_logprob, dtype=torch.int64
            ).to(device, non_blocking=True)
