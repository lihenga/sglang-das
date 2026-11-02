"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_buffer import (
    compute_grid_segments,
    staging_grid_tokens,
)
from sglang.srt.disaggregation.utils import (
    EXTERNAL_KV_LOAD_ERR_TYPE,
    is_external_kv_load_failure,
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    all_reduce_min_attn_cp_tp_group,
    build_kv_layer_ids,
    build_staging_slot_metadata,
    get_dsa_tail_state_indices,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_aborted,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    poll_and_all_reduce_pp,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.cleanup import CleanupOutcomeUnknown, run_cleanup_step
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_QUEUE,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

DAS_PREFETCH_TRACE_ENABLED = os.environ.get(
    "SGLANG_DAS_PREFETCH_TRACE", ""
).strip().lower() in {"1", "true", "yes", "on"}


def trace_disagg_prefill(
    event: str, rid, bootstrap_room, rank, **state
) -> None:
    """Emit an opt-in, content-free timestamp for a disagg prefill request."""
    if not DAS_PREFETCH_TRACE_ENABLED:
        return
    details = " ".join(f"{key}={value}" for key, value in state.items())
    logger.info(
        "DAS_PREFETCH_TRACE phase=%s rid=%s bootstrap_room=%s rank=%s "
        "wall_ns=%d monotonic_ns=%d%s",
        event,
        rid if rid is not None else "-",
        bootstrap_room if bootstrap_room is not None else "-",
        rank if rank is not None else "-",
        time.time_ns(),
        time.monotonic_ns(),
        f" {details}" if details else "",
    )

_is_npu = is_npu()


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    # Force only before/during the first attempt (count is 1 while it runs).
    if retry_prob <= 0 or req.prefill_attempt_count > 1 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def _transfer_start_layer(*, pool, hf_text_config) -> int:
    # Hybrid pools count all layers in start_layer, but peer KV lists contain only
    # full-attention layers, so translate to a full-attention-relative offset.
    if not isinstance(pool, HybridLinearKVPool):
        return pool.start_layer
    return sum(
        1 for lid in hf_text_config.full_attention_layer_ids if lid < pool.start_layer
    )


def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        scheduler_stage_metrics: SchedulerStageMetricsRecorder,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.scheduler = scheduler
        self.scheduler_stage_metrics = scheduler_stage_metrics
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.effective_max_total_num_tokens
        )
        self.transfer_backend = transfer_backend
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            if self.is_mla_backend:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                    "(e.g. GQA, MHA). MLA models should not set this flag."
                )
            page_size = self.scheduler.token_to_kv_pool_allocator.page_size
            # Same source as send_kv_chunk's staging grid below, so validation
            # and the grid cannot disagree after a post-publish override.
            chunked_prefill_size = get_schedule().chunked_prefill_size
            cps = chunked_prefill_size or 8192
            # Staging slices each send into a fixed page-aligned grid, so an
            # unbounded (-1) or non-page-aligned chunk size has no valid grid.
            if cps <= 0 or cps % page_size != 0:
                raise RuntimeError(
                    f"SGLANG_DISAGG_STAGING_BUFFER requires a positive "
                    f"chunked_prefill_size that is a multiple of page_size "
                    f"({page_size}); got {chunked_prefill_size}."
                )
            if self.pp_size > 1 and self.transfer_backend != TransferBackend.MOONCAKE:
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER with pp_size > 1 is only "
                    "supported by Mooncake."
                )
            if get_parallel().enable_prefill_context_parallel:
                # CP rewrites index_slice per rank, breaking the chunk grid.
                raise RuntimeError(
                    "SGLANG_DISAGG_STAGING_BUFFER does not support "
                    "prefill context parallelism."
                )
        self.kv_manager = self._init_kv_manager()

        use_dsv4_full_token_pool = (
            self.scheduler.tp_worker.is_hybrid_swa
            and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
            and envs.SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL.get()
        )
        if use_dsv4_full_token_pool:
            self.max_total_num_tokens = (
                self.scheduler.tp_worker.model_runner.max_token_pool_size
            )
            logger.info(
                "DeepSeek-V4 PD prefill admission uses full token pool capacity: %d",
                self.max_total_num_tokens,
            )
        elif self.scheduler.tp_worker.is_hybrid_swa:
            # Hybrid-SWA prefill otherwise consumes the full prompt in the SWA pool.
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        kv_args.kv_cache_dtype_str = (
            self.scheduler.tp_worker.model_runner.kv_cache_dtype_str
        )
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        layer_shard_rank = getattr(self.token_to_kv_pool, "layer_shard_rank", None)
        layer_shard_size = getattr(self.token_to_kv_pool, "layer_shard_size", 1)
        transfer_draft_cache = (
            not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1
        )
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else _transfer_start_layer(
                pool=self.token_to_kv_pool,
                hf_text_config=self.scheduler.model_config.hf_text_config,
            )
        )
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )

        draft_kv_pool = self.draft_token_to_kv_pool if transfer_draft_cache else None
        num_draft_entries = 0
        if draft_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                draft_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            num_draft_entries = len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.token_to_kv_pool,
            draft_token_to_kv_pool=draft_kv_pool,
            num_draft_entries=num_draft_entries,
            num_hidden_layers=self.scheduler.model_config.num_hidden_layers,
        )
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = get_disagg().disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # V4's KVCache is organized by compression-ratio
            # buckets rather than by layer.
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
            )

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            staging_slots = build_staging_slot_metadata(
                kv_layer_ids=kv_args.kv_layer_ids,
                num_draft_entries=num_draft_entries,
                kv_pool=kv_pool,
                draft_kv_pool=draft_kv_pool,
            )
            if staging_slots is not None:
                k_buffers, v_buffers, slot_layer_ids = staging_slots
                kv_manager.set_kv_buffer_tensors(
                    k_buffers,
                    v_buffers,
                    kv_pool.page_size,
                    slot_layer_ids=slot_layer_ids,
                )
        return kv_manager

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity."""
        if self._check_if_req_exceed_kv_capacity(req):
            return False

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
            req_has_disagg_prefill_dp_rank=req.disagg_prefill_dp_rank is not None,
        )
        self._process_req(req)
        req.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if req.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        assert req.metadata_buffer_index is not None
        return True

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        assert req.pending_bootstrap, "finalize_bootstrap is not idempotent"
        if not self.ensure_metadata_buffer(req):
            return False

        req.time_stats.set_bootstrap_done_time()
        decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
        num_kv_indices = len(req.origin_input_ids)
        req.start_send_idx = decode_prefix_len
        # Base of the staging chunk grid (suffix-relative send coordinates).
        req.disagg_decode_prefix_len = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        num_pages = kv_to_page_num(
            num_kv_indices_to_send,
            self.scheduler.token_to_kv_pool_allocator.page_size,
        )
        req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
        req.pending_bootstrap = False
        return True

    def add(self, req: Req, num_kv_heads: int) -> bool:
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)
        return True

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> List[Req] | tuple[List[Req], List[Req]]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        pp_good_rids: RIDs that PP consensus determined as WaitingForInput.
        pp_bad_rids: RIDs that PP consensus determined as Failed.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()

        if len(self.queue) == 0:
            if return_failed_reqs is False:
                return []
            else:
                return [], []

        if self.pp_size > 1:
            polls = poll_and_all_reduce_pp(
                (req.rid for req in self.queue),
                KVPoll.WaitingForInput,
                pp_good_rids,
                pp_bad_rids,
            )
            uncovered = [i for i, poll in enumerate(polls) if poll is None]
            if uncovered:
                local_polls = poll_and_all_reduce_attn_cp_tp_group(
                    [self.queue[i].disagg_kv_sender for i in uncovered],
                    self.scheduler.attn_cp_cpu_group,
                    self.scheduler.attn_tp_cpu_group,
                )
                for i, local_poll in zip(uncovered, local_polls):
                    if local_poll == KVPoll.Failed:
                        polls[i] = KVPoll.Failed
        else:
            polls = poll_and_all_reduce_attn_cp_tp_group(
                [req.disagg_kv_sender for req in self.queue],
                self.scheduler.attn_cp_cpu_group,
                self.scheduler.attn_tp_cpu_group,
            )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll is None:
                continue

            if poll == KVPoll.Failed:
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif poll == KVPoll.Bootstrapping:
                if (
                    req.prefill_attempt_count < get_disagg().optimistic_prefill_attempts
                    and not req.is_retracted  # engine paused
                ):
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                    bootstrapped_reqs.append(req)
                    indices_to_remove.add(i)
                    req.time_stats.set_wait_queue_entry_time()
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    if not self.ensure_metadata_buffer(req):
                        continue  # no more metadata buffer
                    req.prefill_attempt_count += 1
                elif not self.finalize_bootstrap(req):
                    continue
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in pop_bootstrapped"
                )

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def release_memory_occupation(self):
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def _observe_grammar_first_mask_fill(self: Scheduler, req: Req) -> None:
        # GLM NOTE: the prefill engine samples exactly one token per request,
        # so this is the per-request cost of one (possibly JIT) mask fill.
        stats = req.grammar.grammar_stats
        if (
            self.metrics_reporter.enable_metrics
            and stats is not None
            and stats.first_mask_fill_time is not None
        ):
            self.metrics_reporter.metrics_collector.observe_grammar_first_mask_fill(
                stats.first_mask_fill_time
            )

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def _admit_prefill_bootstrapped_reqs(self: Scheduler) -> None:
        """Move requests whose bootstrap is ready into the waiting queue."""
        bootstrap_queue = self.disagg_prefill_bootstrap_queue
        bootstrapped_reqs = bootstrap_queue.pop_bootstrapped()
        if DAS_PREFETCH_TRACE_ENABLED:
            for req in bootstrapped_reqs:
                trace_disagg_prefill(
                    "bootstrap_ready",
                    req.rid,
                    req.bootstrap_room,
                    self.ps.tp_rank,
                    bootstrap_pending=getattr(req, "pending_bootstrap", False),
                    prefill_attempt=req.prefill_attempt_count,
                    bootstrap_queue_size=len(bootstrap_queue.queue),
                )
        self.waiting_queue.extend(bootstrapped_reqs)
        if DAS_PREFETCH_TRACE_ENABLED:
            for req in bootstrapped_reqs:
                trace_disagg_prefill(
                    "waiting_queue_admit",
                    req.rid,
                    req.bootstrap_room,
                    self.ps.tp_rank,
                    waiting_queue_size=len(self.waiting_queue),
                    bootstrap_queue_size=len(bootstrap_queue.queue),
                    bootstrap_pending=getattr(req, "pending_bootstrap", False),
                )

    def maybe_supplemental_poll_final_chunks(self: Scheduler) -> List[Req]:
        """Give newly submitted final chunks one bounded chance to finish."""
        timeout_ms = self._mooncake_final_poll_timeout_ms
        if timeout_ms <= 0:
            final_rooms = getattr(self, "_disagg_final_chunk_rooms", None)
            if final_rooms:
                final_rooms.clear()
            return []
        final_rooms = getattr(self, "_disagg_final_chunk_rooms", None)
        if not final_rooms:
            return []
        self._disagg_final_chunk_rooms = set()

        remaining_rooms = {
            req.bootstrap_room
            for req in self.disagg_prefill_inflight_queue
            if req.bootstrap_room in final_rooms
        }
        if not remaining_rooms:
            return []

        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        wait_for_rooms = getattr(kv_mgr, "wait_for_transfer_rooms", None)
        if wait_for_rooms is None:
            return []

        start = time.perf_counter()
        all_done = wait_for_rooms(remaining_rooms, timeout_ms / 1000.0)
        waited = time.perf_counter()
        done_reqs = self.process_disagg_prefill_inflight_queue()
        self._record_final_poll_wait(
            len(remaining_rooms),
            len(done_reqs),
            waited - start,
            time.perf_counter() - waited,
            all_done,
        )
        return done_reqs

    def _record_final_poll_wait(
        self: Scheduler,
        num_rooms: int,
        num_done: int,
        wait_s: float,
        poll_s: float,
        all_done: bool,
    ) -> None:
        """Accumulate final-chunk wait stats and log a summary every minute.

        wait_ms is this rank's local wait; poll_ms is the extra all-reduced poll,
        which includes waiting for slower ranks. Their sum is the added
        scheduler stall per round.
        """
        now = time.monotonic()
        stats = getattr(self, "_final_poll_wait_stats", None)
        if stats is None:
            stats = self._final_poll_wait_stats = {"start": now}
        stats["rounds"] = stats.get("rounds", 0) + 1
        stats["rooms"] = stats.get("rooms", 0) + num_rooms
        stats["done"] = stats.get("done", 0) + num_done
        stats["wait_s"] = stats.get("wait_s", 0.0) + wait_s
        stats["poll_s"] = stats.get("poll_s", 0.0) + poll_s
        stats["max_s"] = max(stats.get("max_s", 0.0), wait_s + poll_s)
        stats["timeouts"] = stats.get("timeouts", 0) + (not all_done)
        if now - stats["start"] < 60.0:
            return
        rounds = stats["rounds"]
        logger.info(
            "Final KV chunk poll (last %.0f s): rounds=%d rooms=%d done=%d "
            "local_wait_ms avg=%.1f extra_poll_ms avg=%.1f stall_ms total=%.0f "
            "max=%.1f timeouts=%d",
            now - stats["start"],
            rounds,
            stats["rooms"],
            stats["done"],
            stats["wait_s"] * 1000 / rounds,
            stats["poll_s"] * 1000 / rounds,
            (stats["wait_s"] + stats["poll_s"]) * 1000,
            stats["max_s"] * 1000,
            stats["timeouts"],
        )
        self._final_poll_wait_stats = {"start": now}

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.
        """
        candidates = [req for req in self.waiting_queue if not is_aborted(req)]
        if not candidates:
            return
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in candidates],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        failed = set()
        for req, poll in zip(candidates, polls):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        if failed:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in failed
            ]

    def has_bootstrapped_waiting_req(self: Scheduler) -> bool:
        return any(
            not req.pending_bootstrap and not is_aborted(req)
            for req in self.waiting_queue
        )

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
        running_batch: ScheduleBatch,
        last_batch: Optional[ScheduleBatch],
    ) -> NextBatchPlan:
        if self.enable_waiting_queue_dfs_prefetch:
            self._begin_scheduler_iteration()
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        running_batch.batch_is_full = False

        self.resolve_waiting_queue_bootstrap()

        self.process_prefill_chunk(last_batch=last_batch, running_batch=running_batch)

        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        if DAS_PREFETCH_TRACE_ENABLED and batch is not None:
            for req in batch.reqs:
                trace_disagg_prefill(
                    "prefill_batch_scheduled",
                    req.rid,
                    req.bootstrap_room,
                    self.ps.tp_rank,
                    batch_size=len(batch.reqs),
                    waiting_queue_size=len(self.waiting_queue),
                    prefill_attempt=req.prefill_attempt_count,
                )
        running_batch = prefill_plan.running_batch
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        while True:
            if self._forward_ingress_enabled:
                self._process_deferred_reqs()

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self._admit_prefill_bootstrapped_reqs()

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()
            if self._mooncake_final_poll_timeout_ms > 0:
                self.maybe_supplemental_poll_final_chunks()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue
            self._admit_prefill_bootstrapped_reqs()

            # Get the next batch to run
            plan = self.get_next_disagg_prefill_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()
            if self._mooncake_final_poll_timeout_ms > 0:
                self.maybe_supplemental_poll_final_chunks()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()
        auxiliary_output_starts = (
            self.batch_result_processor.snapshot_auxiliary_output_starts(batch, result)
        )
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logprob_pt = 0
        aborted_reqs: List[Req] = []
        assert batch.spec_info is result.next_draft_input
        draft_input = result.next_draft_input
        draft_hidden_states_cpu = None
        draft_dsa_topk_indices_cpu = None
        if self.spec_algorithm.is_eagle() and draft_input is not None:
            draft_hidden_states_cpu = draft_input.hidden_states.to(
                "cpu", non_blocking=False
            )
            if batch.spec_info.dsa_topk_indices is not None:
                draft_dsa_topk_indices_cpu = batch.spec_info.dsa_topk_indices.to(
                    "cpu", non_blocking=False
                )
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                # Never send KV from a failed external load to decode.
                if req.discard_output_reason is not None:
                    req.to_finish = req.discard_output_reason
                if is_aborted(req):
                    if self._retire_aborted_prefill_result(req):
                        req.time_stats.set_completion_time()
                        aborted_reqs.append(req)
                    advance_logprob_pt(i, req)
                    continue

                # Test hook: exercise the release/requeue retry path.
                if req.pending_bootstrap and should_force_retry(req):
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    continue

                req.output_ids.append(next_token_id)
                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
                    self._observe_grammar_first_mask_fill(req)
                    if is_aborted(req):
                        if self._retire_aborted_prefill_result(req):
                            req.time_stats.set_completion_time()
                            aborted_reqs.append(req)
                        advance_logprob_pt(i, req)
                        continue

                maybe_cache_unfinished_req(req, self.tree_cache)
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and draft_input is not None:
                    req.output_topk_p = draft_input.topk_p[i]
                    req.output_topk_index = draft_input.topk_index[i]
                    req.hidden_states_tensor = draft_hidden_states_cpu[i].clone()
                    if draft_dsa_topk_indices_cpu is not None:
                        req.output_dsa_topk_indices = draft_dsa_topk_indices_cpu[
                            i
                        ].clone()
                    else:
                        req.output_dsa_topk_indices = None
                else:
                    req.hidden_states_tensor = None
                    req.output_dsa_topk_indices = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                if req.return_sampling_mask:
                    self.batch_result_processor.add_sampling_mask_return_values(
                        i, req, logits_output
                    )
                if not req.pending_bootstrap:
                    self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()

            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1

                # Still chunking iff its next chunk was launched: either it is
                # still self.chunked_req, or its final chunk (extend_range
                # reaching the end of the input) is in flight. A yielded req
                # is neither, so do its deferred release here.
                still_chunking = self.chunked_req is req or (
                    req.extend_range is not None
                    and req.extend_range.end >= len(req.origin_input_ids)
                )
                # Abort is terminal. Do not requeue an aborted optimistic
                # request merely because bootstrap is still pending.
                if is_aborted(req):
                    if not still_chunking and self._retire_aborted_prefill_result(req):
                        req.time_stats.set_completion_time()
                        aborted_reqs.append(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.pending_bootstrap and not still_chunking:
                    self.optimistic_release_and_requeue(req)
                    advance_logprob_pt(i, req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                # In non-overlap-mode, KV is sent in process_prefill_chunk
                # Only send when req's sender is initialized
                if self.enable_overlap and not req.pending_bootstrap:
                    assert req.metadata_buffer_index >= 0, (
                        f"Req {req.rid} does not have metadata buffer allocated"
                    )
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        if auxiliary_output is not None:
            self.batch_result_processor.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

        if aborted_reqs:
            self.output_streamer.stream_output(
                aborted_reqs,
                any(req.return_logprob for req in aborted_reqs),
            )

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )
        self.maybe_send_health_check_signal()

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_QUEUE)
    def abort_disagg_prefill_request(self: Scheduler, req: Req) -> None:
        """Close publication now; the request's queue owns drained cleanup."""
        if req.to_finish is None and req.finished_reason is None:
            req.to_finish = FINISH_ABORT("Aborted by AbortReq.")
        self.abort_external_kv_request(
            req,
            forward_drained=False,
            sender_terminal=False,
            emit_response=False,
        )

    def prefill_abort_sender_terminal(self: Scheduler, req: Req) -> bool:
        """A failure on a peer rank does not prove the local sender stopped."""
        if req.disagg_kv_sender is None:
            return True
        try:
            sender = req.disagg_kv_sender
            if sender.poll() not in (KVPoll.Success, KVPoll.Failed):
                return False
            # Mooncake logical Failed is immediate; source ownership is separate.
            drained = getattr(sender, "is_transfer_drained", None)
            return bool(drained()) if drained is not None else True
        except Exception:
            logger.exception("Failed to poll aborting sender for request %s", req.rid)
            return False

    def prefill_abort_forward_drained(self: Scheduler, req: Req) -> bool:
        """Do not reclaim a chunk still referenced by an unresolved result."""
        if req.inflight_middle_chunks > 0:
            return False
        return not any(
            any(candidate is req for candidate in batch.reqs)
            for batch, _ in getattr(self, "result_queue", ())
        )

    def abort_external_kv_request(
        self: Scheduler,
        req: Req,
        *,
        forward_drained: bool,
        sender_terminal: bool,
        abort_message: str = "External KV load failed",
        is_insert: bool = False,
        emit_response: bool = False,
    ) -> bool:
        """Retry incomplete cleanup stages; callers must keep the Req reachable.

        With emit_response=False, completion covers resources only. The caller
        owns response delivery and must track external_kv_response_sent.
        """
        try:
            if getattr(self, "_pending_chunked_abort_req", None) is req:
                req.external_kv_abort_response_via_chunked = True
            if getattr(req, "external_kv_abort_response_via_chunked", False):
                emit_response = False
            # A resource-only caller may return later to deliver the response.
            if req.external_kv_cleanup_done:
                if emit_response and not req.external_kv_response_sent:
                    self.output_streamer.stream_output([req], req.return_logprob)
                    req.external_kv_response_sent = True
                return True

            if not req.external_kv_abort_requested:
                req.skip_radix_cache_insert = True
                if req.to_finish is None and req.finished_reason is None:
                    req.to_finish = FINISH_ABORT(
                        message=abort_message,
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        err_type=EXTERNAL_KV_LOAD_ERR_TYPE,
                    )
                req.external_kv_abort_requested = True
                self._external_kv_abort_rids.add(req.rid)

            if not req.external_kv_pending_chunk_cleared:
                if self.disaggregation_mode == DisaggregationMode.PREFILL:
                    self.clear_pending_chunk_send(req)
                req.external_kv_pending_chunk_cleared = True

            if not req.external_kv_linker_released:
                self._release_aborted_request(req.rid)
                req.external_kv_linker_released = True

            sender = req.disagg_kv_sender
            if sender is None:
                req.external_kv_sender_abort_requested = True
            elif not req.external_kv_sender_abort_requested:
                sender.abort()
                req.external_kv_sender_abort_requested = True

            if (
                not forward_drained
                or not sender_terminal
                # Recheck after abort closes publication; the caller's earlier
                # observation may precede the final enqueue.
                or not self.prefill_abort_sender_terminal(req)
            ):
                return False

            # A decode request can only be failed safely after every local
            # source transfer has drained. The Mooncake sender fans this
            # terminal marker out to its saved/current decode targets.
            notify_drained_abort = getattr(sender, "notify_drained_abort", None)
            if notify_drained_abort is not None:
                notify_drained_abort()

            if not req.external_kv_finish_state_applied:
                if req.to_finish is not None:
                    req.update_finish_state()
                elif req.finished_reason is None:
                    req.finished_reason = FINISH_ABORT(
                        message=abort_message,
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        err_type=EXTERNAL_KV_LOAD_ERR_TYPE,
                    )
                # Keep the existing failure response shape for logprob requests.
                # prepare_abort initializes the empty logprob lists, but its new
                # reason must not replace the original error type or message.
                if req.return_logprob:
                    reason = req.finished_reason
                    prepare_abort(req, reason.message, status_code=reason.status_code)
                    req.finished_reason = reason
                req.external_kv_finish_state_applied = True

            if not req.external_kv_metadata_released:
                if self.disaggregation_mode in (
                    DisaggregationMode.PREFILL,
                    DisaggregationMode.DECODE,
                ):
                    maybe_release_metadata_buffer(
                        req,
                        self.req_to_metadata_buffer_idx_allocator,
                    )
                req.external_kv_metadata_released = True

            if not req.external_kv_cache_released:
                # Bootstrap may fail before any KV or Mamba allocation exists.
                if (
                    req.kv.req_pool_idx is not None
                    or req.kv is not None
                    or req.kv.mamba_pool_idx is not None
                    or any(
                        name.startswith("kv.")
                        for name in getattr(req, "external_kv_cleanup_steps", {})
                    )
                ):
                    release_kv_cache(req, self.tree_cache, is_insert=is_insert)
                req.external_kv_cache_released = True

            if emit_response and not req.external_kv_response_sent:
                req.time_stats.set_completion_time()
                self.output_streamer.stream_output([req], req.return_logprob)
                req.external_kv_response_sent = True

            self._external_kv_abort_rids.discard(req.rid)
            # The chunked protocol owner clears its marker after delivery.
            req.external_kv_cleanup_done = True
            return True
        except CleanupOutcomeUnknown:
            # The original exception was logged on the first attempt. Keep the
            # request reachable without flooding logs on every scheduler tick.
            return False
        except Exception:
            logger.exception(
                "External KV abort cleanup failed for request %s; pending cleanup retained",
                req.rid,
            )
            return False

    def process_disagg_prefill_inflight_queue(
        self: Scheduler,
        transfer_status: Optional[Tuple[List[str], List[str]]] = None,
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        transfer_status: For PP, the globally agreed successful and failed rids.
        """
        if len(self.disagg_prefill_inflight_queue) == 0:
            return []

        done_reqs = []
        success_rids = set(transfer_status[0]) if transfer_status is not None else set()
        failed_rids = set(transfer_status[1]) if transfer_status is not None else set()

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        undone_reqs: List[Req] = []
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if req.external_kv_cleanup_done:
                # Local cleanup is terminal; only the shared removal verdict
                # remains pending. Do not re-enter failure or success cleanup.
                done_reqs.append(req)
                continue
            # Check if this request was marked for external KV abort
            external_abort_pending = req.rid in self._external_kv_abort_rids

            if external_abort_pending:
                reason = req.to_finish or req.finished_reason
                cleanup_done = self.abort_external_kv_request(
                    req,
                    forward_drained=self.prefill_abort_forward_drained(req),
                    sender_terminal=self.prefill_abort_sender_terminal(req),
                    abort_message=getattr(reason, "message", "External KV load failed"),
                    is_insert=False,
                    emit_response=True,
                )
                if cleanup_done:
                    done_reqs.append(req)
                else:
                    undone_reqs.append(req)
                continue

            if transfer_status is not None:
                consensus_failed = req.rid in failed_rids
                failure_pending = isinstance(req.finished_reason, FINISH_ABORT)
                if consensus_failed or failure_pending:
                    if consensus_failed and not failure_pending:
                        prepare_abort(
                            req,
                            (
                                "Prefill transfer failed on another PP rank; "
                                "waiting for the local transfer to stop"
                            ),
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    _, cleanup_done = self.handle_inflight_transfer_failure(
                        req,
                        sender_terminal=self.prefill_abort_sender_terminal(req),
                    )
                    if cleanup_done:
                        done_reqs.append(req)
                    else:
                        undone_reqs.append(req)
                    continue

                if req.rid not in success_rids:
                    undone_reqs.append(req)
                    continue

            if req.pending_bootstrap:
                # Parked: prefill finished before bootstrap completed.
                if self.handle_pending_bootstrap(req, poll):
                    self.send_kv_chunk(req, last_chunk=True)
                    undone_reqs.append(req)
                elif req.external_kv_abort_requested:
                    if req.external_kv_cleanup_done:
                        done_reqs.append(req)
                    else:
                        undone_reqs.append(req)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                # todo: set Transferring correctly in backend
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                if not isinstance(req.finished_reason, FINISH_ABORT):
                    req.finished_reason = FINISH_LENGTH(length=0)
                release_kv_cache(req, self.tree_cache)  # unlock the tree
                # FIXME: clean up req's data in transfer engine
                req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                _, cleanup_done = self.handle_inflight_transfer_failure(
                    req, sender_terminal=self.prefill_abort_sender_terminal(req)
                )
                if cleanup_done:
                    done_reqs.append(req)
                else:
                    undone_reqs.append(req)
            else:
                raise RuntimeError(
                    f"Unexpected poll state {poll} for req {req.rid} in inflight queue"
                )

        # Local cleanup can finish before its TP/CP peers. Keep the request in
        # every inflight queue until all peers agree it can be removed.
        keep_ids = {id(req) for req in undone_reqs}
        remove_flags = [
            int(id(req) not in keep_ids) for req in self.disagg_prefill_inflight_queue
        ]
        global_remove_flags = all_reduce_min_attn_cp_tp_group(
            remove_flags,
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        ready_ids = {
            id(req)
            for req, remove_flag in zip(
                self.disagg_prefill_inflight_queue, global_remove_flags
            )
            if remove_flag
        }
        done_reqs = [req for req in done_reqs if id(req) in ready_ids]
        undone_reqs = [
            req for req in self.disagg_prefill_inflight_queue if id(req) not in ready_ids
        ]

        for req in done_reqs:
            req.time_stats.set_completion_time()

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        output_reqs = [
            req
            for req in done_reqs
            if not req.external_kv_response_sent
            and not getattr(req, "external_kv_abort_response_via_chunked", False)
        ]
        if output_reqs:
            self.output_streamer.stream_output(
                output_reqs,
                any(req.return_logprob for req in output_reqs),
                None,
            )
        for req in done_reqs:
            if req.external_kv_metadata_released:
                continue
            maybe_release_metadata_buffer(
                req,
                self.req_to_metadata_buffer_idx_allocator,
            )

        self.disagg_prefill_inflight_queue = undone_reqs
        if getattr(self, "enable_unified_cache_external_linker", False):
            self.tree_cache.retry_stranded_failed_linker_chains()

        return done_reqs

    def handle_inflight_transfer_failure(
        self: Scheduler,
        req: Req,
        *,
        sender_terminal: bool,
    ) -> Tuple[Optional[Exception], bool]:
        """Record an inflight transfer failure and run retryable cleanup."""
        error_message = (
            f"Prefill transfer failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        exc: Optional[Exception] = None
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            exc = e
            error_message += f" with exception {e}"
        if not req.prefill_transfer_failure_reported:
            req.prefill_transfer_failure_reported = True
            if getattr(exc, "is_from_another_rank", False):
                logger.debug(error_message)
            else:
                logger.warning(error_message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_transfer_failed_reqs()

        if req.to_finish is None and req.finished_reason is None:
            req.to_finish = FINISH_ABORT(
                error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        cleanup_done = self.abort_external_kv_request(
            req,
            forward_drained=self.prefill_abort_forward_drained(req),
            sender_terminal=sender_terminal,
            abort_message=error_message,
            is_insert=False,
            emit_response=True,
        )
        if not cleanup_done:
            self._external_kv_abort_rids.add(req.rid)
        return exc, cleanup_done

    def clear_pending_chunk_send(self: Scheduler, req: Req) -> None:
        """Drop `req` from the sent-but-unconcluded chunk set.

        Every path that retires a request without a `last_chunk=True` send must
        call this: a stale entry holds the unified-memory compaction gate closed
        for the process lifetime.
        """
        self.disagg_prefill_pending_chunk_rids.discard(req.rid)

    def _retire_aborted_prefill_result(self: Scheduler, req: Req) -> bool:
        """Release an aborted request when its last prefill result is safe."""
        if req.external_kv_abort_requested:
            req.pending_bootstrap = False
            self.abort_external_kv_request(
                req, forward_drained=True,
                sender_terminal=self.prefill_abort_sender_terminal(req),
                emit_response=True,
            )
            if req not in self.disagg_prefill_inflight_queue:
                self.disagg_prefill_inflight_queue.append(req)
            return False
        self.clear_pending_chunk_send(req)
        owns_resources = (
            req.kv.holds_kv or req.kv.holds_mamba or req.metadata_buffer_index >= 0
        )
        if not owns_resources:
            # A bootstrap failure or earlier abort already retired it.
            return False

        sender = req.disagg_kv_sender
        if sender is not None:
            try:
                sender.abort()
            except Exception:
                # Transport notification is best effort; local ownership must
                # still be released or the next idle invariant check will fail.
                logger.exception("Failed to notify KV sender of abort for %s", req.rid)

        if req.to_finish is not None and not req.finished():
            req.update_finish_state()
        maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        req.pending_bootstrap = False
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache, is_insert=False)
        return True

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        req.pending_bootstrap = False

        if req.to_finish is None and req.finished_reason is None:
            req.to_finish = FINISH_ABORT(
                error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
            )
        self.abort_external_kv_request(
            req,
            forward_drained=self.prefill_abort_forward_drained(req),
            sender_terminal=self.prefill_abort_sender_terminal(req),
            abort_message=error_message,
            is_insert=False,
            emit_response=True,
        )
        if req not in self.disagg_prefill_inflight_queue:
            # Local cleanup can finish before its TP/CP peers. Enqueue on every
            # rank and let the existing inflight removal consensus retire it.
            self.disagg_prefill_inflight_queue.append(req)

        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        # Also covers the external KV linker: a waiting-queue prefetch taken
        # at arrival would otherwise keep its budget slot and pinned session.
        self._release_aborted_request(req.rid)

    def get_transferred_rids(
        self: Scheduler,
    ) -> Tuple[List[str], List[str]]:
        """
        Used by PP to inspect local terminal transfers without popping requests.
        """
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )

        success_rids: List[str] = []
        failed_rids: List[str] = []

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success:
                success_rids.append(req.rid)
            elif poll == KVPoll.Failed:
                failed_rids.append(req.rid)

        return success_rids, failed_rids

    def handle_pending_bootstrap(self: Scheduler, req: Req, poll: KVPoll) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            return False
        elif poll == KVPoll.WaitingForInput:
            if should_force_retry(req):  # test hook
                return False
            # Metadata buffer was allocated in pop_bootstrapped before
            # the request entered the waiting queue, so finalize should not fail.
            assert self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
            return True
        else:
            raise RuntimeError(
                f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            )

    def check_bootstrap(self: Scheduler, req: Req) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.pending_bootstrap:
            return True
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender],
            self.attn_cp_cpu_group,
            self.attn_tp_cpu_group,
        )
        return self.handle_pending_bootstrap(req, polls[0])

    def process_prefill_chunk(
        self: Scheduler,
        last_batch: Optional[ScheduleBatch],
        running_batch: ScheduleBatch,
    ) -> None:
        chunked_req_to_exclude = set()
        if (req := self.chunked_req) is not None:
            chunked_req_to_exclude.add(req)
            maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

            if not self.check_bootstrap(req):
                if is_aborted(req):
                    # bootstrap failed
                    self.chunked_req = None
                elif self.has_bootstrapped_waiting_req():
                    # optimistic request yields to waiting requests
                    self.chunked_req = None
                    if not self.enable_overlap:
                        self.optimistic_release_and_requeue(req)
                # else: still bootstrapping, keep computing without sending
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                req.tmp_end_idx = min(
                    req.extend_range.end,
                    len(req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(req)

            if self.chunked_req is not None:
                running_batch.batch_is_full = False

        if last_batch and last_batch.forward_mode.is_extend():
            if last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(last_batch.chunked_req)

            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

    def maybe_send_cached_prefix_chunk(
        self: Scheduler, req: Req, *, cache_load_pending: bool = False
    ) -> None:
        if not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get():
            return

        # Staging sends into positional grid slots, so the early-send boundary
        # must stay stable across the request's batches: snapshot the at-rest
        # prefix on the first batch. Non-staging reads the live prefix.
        if self.enable_staging and req.early_send_prefix_end is None:
            req.early_send_prefix_end = max(
                0, len(req.prefix_indices) - req.host_hit_length
            )

        # Keep the staging boundary above stable even when this batch's send
        # is deferred: the next chunk must not snapshot a different prefix.
        if req.pending_bootstrap or cache_load_pending:
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = (
            req.early_send_prefix_end
            if self.enable_staging
            else len(req.prefix_indices) - req.host_hit_length
        )
        if cached_end <= req.start_send_idx:
            return
        if cached_end % self.token_to_kv_pool_allocator.page_size != 0:
            # DCP radix hits can end on a logical cache-page boundary that is
            # not a complete physical DCP page. The regular final send covers
            # the full range; only skip this optional early-send optimization.
            return
        # Early-send issues the KV read before this step's forward is enqueued,
        # but under overlap scheduling the PRIOR step's prefill forward may still
        # be writing these prefix pages on forward_stream. Record a completion
        # event now so the transfer worker can wait on those writes before the
        # RDMA read, instead of racing them.
        if self.enable_overlap:
            ev = torch.cuda.Event()
            ev.record(self.forward_stream)
            req.disagg_kv_sender._early_send_wait_event = ev
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> None:
        """
        Send a prefilled chunk to the decode server
        """
        # Only the non-PP event loops drain these rooms, so do not collect them
        # otherwise (or when the supplemental poll is disabled).
        if (
            last_chunk
            and self.ps.pp_size == 1
            and self._mooncake_final_poll_timeout_ms > 0
        ):
            final_rooms = getattr(self, "_disagg_final_chunk_rooms", None)
            if final_rooms is None:
                final_rooms = self._disagg_final_chunk_rooms = set()
            final_rooms.add(req.bootstrap_room)

        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, transfer_input_len)
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size
            if self.enable_staging:
                # Staging identifies chunks positionally against a uniform
                # prefetched grid, so non-last sends must end on a grid
                # boundary; the remainder rides with the next send.
                grid_tokens = staging_grid_tokens(
                    get_schedule().chunked_prefill_size, page_size
                )
                base = req.disagg_decode_prefix_len
                end_idx = base + ((end_idx - base) // grid_tokens) * grid_tokens

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return

        state_indices: Optional[List] = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)

            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # that decode used to register the destination row.
            seq_len = min(req.extend_range.end, transfer_input_len)
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.translate_mamba_indices(
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            req.kv.req_pool_idx
                        ]
                    )
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(req.disagg_decode_prefix_len, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _full_kv_pages_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _dsa_tail_payload():
                return get_dsa_tail_state_indices(
                    self.token_to_kv_pool_allocator.get_kvcache(),
                    req.kv.req_pool_idx,
                    seq_len,
                )

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.kv.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                online = is_dsv4_c128_online_enabled()
                ring_size = (
                    1
                    if online
                    else self.token_to_kv_pool_allocator.get_kvcache().get_ring_size(
                        128
                    )
                )
                return get_dsv4_c128_state_indices(
                    int(req.kv.req_pool_idx),
                    c128_seq_len,
                    online=online,
                    ring_size=ring_size,
                )

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.SWA: _swa_payload,
                StateType.DSA: _full_kv_pages_payload,
                StateType.DSA_TAIL: _dsa_tail_payload,
                StateType.MINIMAX_INDEX_K: _full_kv_pages_payload,
                StateType.SWA_RING: _swa_ring_payload,
                StateType.C128_STATE: _c128_state_payload,
                StateType.BLOCK_SCALE: _full_kv_pages_payload,
                StateType.BLOCK_SCALE_SWA: _swa_payload,
            }
            if _is_npu and isinstance(
                self.token_to_kv_pool_allocator.get_kvcache(),
                DeepSeekV4TokenToKVPool,
            ):
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )

                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        req.kv.req_pool_idx,
                        seq_len,
                        page_size,
                        prefix_len=req.disagg_decode_prefix_len,
                    )
                )
            state_indices = [
                payloads[st]() if st in payloads else None for st in state_types
            ]

        if self.enable_staging:
            # One sender.send per grid slot; the sender's cumulative page
            # counter marks only the final sub-send of the final chunk as
            # is_last, routing aux/state correctly.
            segments = compute_grid_segments(
                start_idx,
                end_idx,
                req.disagg_decode_prefix_len,
                staging_grid_tokens(get_schedule().chunked_prefill_size, page_size),
            )
        else:
            segments = [(start_idx, end_idx)]

        for seg_start, seg_end in segments:
            is_final_segment = seg_end == end_idx
            kv_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, seg_start:seg_end
            ]
            # Unified memory: req_to_token holds VIRTUAL ids; the transfer needs
            # physical ones. Per segment, since each is its own gather.
            kv_indices = (
                self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                    kv_indices
                )
            )
            page_indices = kv_to_page_indices(kv_indices, page_size)
            segment_is_last = last_chunk and is_final_segment
            if not req.disagg_kv_sender.should_send_kv_chunk(
                len(page_indices), segment_is_last
            ):
                continue
            req.disagg_kv_sender.send(
                page_indices,
                state_indices if segment_is_last else None,
                num_kv_tokens=seg_end - seg_start,
            )
        req.start_send_idx = end_idx
        # A last chunk needs no entry: every `last_chunk=True` call site has
        # already put the request on `disagg_prefill_inflight_queue`.
        if last_chunk:
            self.disagg_prefill_pending_chunk_rids.discard(req.rid)
        else:
            self.disagg_prefill_pending_chunk_rids.add(req.rid)

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        max_attempts = get_disagg().optimistic_prefill_attempts
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        self.clear_pending_chunk_send(req)  # re-sends from scratch
        req.tmp_end_idx = -1
        req.disagg_decode_prefix_len = 0
        req.early_send_prefix_end = None
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.prefill_attempt_count >= max_attempts:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill attempts "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            req.prefill_attempt_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill yielded "
                f"({req.prefill_attempt_count}/{max_attempts} attempts used)"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
