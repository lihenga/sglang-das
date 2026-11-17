"""Mooncake waiting-queue work; the scheduler thread owns request/tree updates."""

from __future__ import annotations

import contextvars
import dataclasses
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state import create_custom_parallel_group
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.utils import is_health_check_generate_req
from sglang.srt.runtime_context import (
    get_memory,
    get_observability,
    get_parallel,
)
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)
DAS_PREFETCH_TRACE_ENABLED = envs.SGLANG_DAS_PREFETCH_TRACE.get()


def trace_disagg_prefill(event: str, rid, bootstrap_room, rank, **state) -> None:
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


class MooncakePrefetchSchedulerMixin:
    BG_PREFETCH_JOIN_GRACE_EPOCHS = 4
    EXTERNAL_ADMISSION_PENDING_LIMIT = 8
    FORWARD_DEFERRED_HIGH_WATER = 1024
    FORWARD_RECV_LIMIT = 128
    enable_waiting_queue_dfs_prefetch = False
    _forward_ingress_enabled = False
    _mooncake_final_poll_timeout_ms = 0.0

    def _trace_prefill_reqs(self, event: str, payloads, **state) -> None:
        if not DAS_PREFETCH_TRACE_ENABLED:
            return
        for payload in payloads:
            batch = getattr(payload, "batch", None)
            requests = batch if isinstance(batch, (list, tuple)) else (payload,)
            for req in requests:
                rid = getattr(req, "rid", None)
                if rid is None:
                    continue
                trace_disagg_prefill(
                    event,
                    rid,
                    getattr(req, "bootstrap_room", None),
                    get_parallel().tp_rank,
                    **state,
                )

    def _bg_worker_loop(self):
        """Execute one deterministic external-prefetch round per scheduler step."""
        while True:
            with self._bg_condition:
                self._bg_condition.wait_for(
                    lambda: self._bg_stop_flag
                    or self._bg_completed_epoch < self._bg_requested_epoch
                )
                if self._bg_stop_flag:
                    return
                epoch = self._bg_completed_epoch + 1

            try:
                if self.enable_waiting_queue_dfs_prefetch:
                    self._run_bg_prefetch_round()
            except BaseException as exc:
                self._bg_error = exc
                logger.exception("Background DFS prefetch round failed")
                with self._bg_condition:
                    self._bg_stop_flag = True
                    self._bg_completed_epoch = epoch
                    self._bg_condition.notify_all()
                return

            with self._bg_condition:
                self._bg_completed_epoch = epoch
                self._bg_condition.notify_all()

    def _run_bg_prefetch_round(self) -> None:
        """Run one CP/TP-ordered round over retained immutable job snapshots."""
        epoch = self._bg_completed_epoch + 1
        while True:
            try:
                job = self._bg_prefetch_jobs.get_nowait()
            except queue.Empty:
                break
            self._bg_pending_prefetch_jobs.setdefault(job.rid, job)
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_job_dequeued",
                    job.rid,
                    self._bg_prefetch_rooms.get(job.rid),
                    get_parallel().tp_rank,
                    epoch=epoch,
                    locally_eligible=job.locally_eligible,
                    transfer_count=len(job.transfers),
                )

        local_status = {
            rid: (job.locally_eligible, job.cancelled.is_set())
            for rid, job in self._bg_pending_prefetch_jobs.items()
        }
        if DAS_PREFETCH_TRACE_ENABLED:
            for rid, (locally_eligible, cancelled) in local_status.items():
                trace_disagg_prefill(
                    "bg_prefetch_round_start",
                    rid,
                    self._bg_prefetch_rooms.get(rid),
                    get_parallel().tp_rank,
                    epoch=epoch,
                    locally_eligible=locally_eligible,
                    cancelled=cancelled,
                )
        rank_statuses = {torch.distributed.get_rank(): local_status}
        # First propagate a fixed-width presence flag. Empty rounds can skip
        # object serialization, while CP then TP reductions make the decision
        # identical for every participant in the 2-D attention topology.
        has_jobs = torch.tensor([int(bool(local_status))], dtype=torch.int)
        self._all_reduce_bg_prefetch_groups(has_jobs, op=torch.distributed.ReduceOp.MAX)
        if int(has_jobs.item()) == 0:
            self._bg_prefetch_join_age.clear()
            return

        # Gathering per-rank snapshots lets every participant distinguish an
        # ineligible request from one whose snapshot has not arrived yet.
        for group in (self._bg_attn_cp_cpu_group, self._bg_attn_tp_cpu_group):
            if group is None:
                continue
            gathered = [None] * torch.distributed.get_world_size(group=group)
            torch.distributed.all_gather_object(gathered, rank_statuses, group=group)
            merged_statuses = {}
            for subgroup_statuses in gathered:
                merged_statuses.update(subgroup_statuses)
            rank_statuses = {
                rank: merged_statuses[rank] for rank in sorted(merged_statuses)
            }

        all_rids = sorted(
            {rid for statuses in rank_statuses.values() for rid in statuses}
        )
        for rid in all_rids:
            self._bg_prefetch_join_age[rid] = self._bg_prefetch_join_age.get(rid, 0) + 1

        for rid in all_rids:
            rid_statuses = [statuses.get(rid) for statuses in rank_statuses.values()]
            job = self._bg_pending_prefetch_jobs.get(rid)
            room = (
                self._bg_prefetch_rooms.get(rid) if DAS_PREFETCH_TRACE_ENABLED else None
            )
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_round_joined",
                    rid,
                    room,
                    get_parallel().tp_rank,
                    epoch=epoch,
                    rank_count=len(rank_statuses),
                    has_local_job=job is not None,
                    join_age=self._bg_prefetch_join_age[rid],
                )

            # Any participant cancelling the request retires all retained
            # snapshots immediately, including ranks that never saw a job.
            if any(status is not None and status[1] for status in rid_statuses):
                if DAS_PREFETCH_TRACE_ENABLED:
                    trace_disagg_prefill(
                        "bg_prefetch_submit_skipped",
                        rid,
                        room,
                        get_parallel().tp_rank,
                        epoch=epoch,
                        reason="cancelled",
                    )
                self._finish_bg_prefetch_job(rid, False)
                continue

            all_ranks_have_job = all(status is not None for status in rid_statuses)
            if not all_ranks_have_job:
                if (
                    self._bg_prefetch_join_age[rid]
                    >= self.BG_PREFETCH_JOIN_GRACE_EPOCHS
                ):
                    if DAS_PREFETCH_TRACE_ENABLED:
                        trace_disagg_prefill(
                            "bg_prefetch_submit_skipped",
                            rid,
                            room,
                            get_parallel().tp_rank,
                            epoch=epoch,
                            reason="missing_rank_job",
                        )
                    self._finish_bg_prefetch_job(rid, False)
                continue

            if not all(status[0] for status in rid_statuses):
                if DAS_PREFETCH_TRACE_ENABLED:
                    trace_disagg_prefill(
                        "bg_prefetch_submit_skipped",
                        rid,
                        room,
                        get_parallel().tp_rank,
                        epoch=epoch,
                        reason="ineligible_rank",
                    )
                self._finish_bg_prefetch_job(rid, False)
                continue

            # The gathered snapshot guarantees every participant has an
            # eligible job. Submit is local and may fail independently; every
            # rank still enters the fixed-width result reduction.
            assert job is not None
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_submit_start",
                    rid,
                    room,
                    get_parallel().tp_rank,
                    epoch=epoch,
                    transfer_count=len(job.transfers),
                )
            try:
                submitted = job.cache_linker.submit_host_prefetch(
                    rid, list(job.transfers)
                )
            except BaseException:
                submitted = False
            submitted = submitted and not job.cancelled.is_set()
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_submit_local_result",
                    rid,
                    room,
                    get_parallel().tp_rank,
                    epoch=epoch,
                    submitted=submitted,
                )
            globally_submitted = torch.tensor([int(submitted)], dtype=torch.int)
            self._all_reduce_bg_prefetch_groups(globally_submitted)
            submitted = int(globally_submitted.item()) == 1
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_submit_done",
                    rid,
                    room,
                    get_parallel().tp_rank,
                    epoch=epoch,
                    submitted=submitted,
                )
            if not submitted:
                job.cache_linker.cancel_host_prefetch(rid)
            self._finish_bg_prefetch_job(rid, submitted)

        for rid in tuple(self._bg_prefetch_join_age):
            if rid not in all_rids:
                self._bg_prefetch_join_age.pop(rid, None)

    def _finish_bg_prefetch_job(self, rid: str, submitted: bool) -> None:
        job = self._bg_pending_prefetch_jobs.pop(rid, None)
        self._bg_prefetch_join_age.pop(rid, None)
        if job is not None:
            self._log_waiting_queue_prefetch_event(
                "submitted" if submitted else "skipped"
            )
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_ack_enqueued",
                    rid,
                    self._bg_prefetch_rooms.get(rid),
                    get_parallel().tp_rank,
                    submitted=submitted,
                )
            self._bg_prefetch_acks.put((job, submitted))

    def _log_waiting_queue_prefetch_event(self, event: str, count: int = 1) -> None:
        reporter = getattr(self, "metrics_reporter", None)
        collector = getattr(reporter, "metrics_collector", None)
        if collector is not None:
            try:
                collector.waiting_queue_prefetch_events_total.labels(
                    **collector.labels, event=event
                ).inc(count)
            except Exception:
                logger.warning(
                    "Failed to record waiting-queue prefetch event.", exc_info=True
                )

    def _all_reduce_bg_prefetch_groups(
        self,
        tensor: torch.Tensor,
        op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.MIN,
    ) -> None:
        for group in (self._bg_attn_cp_cpu_group, self._bg_attn_tp_cpu_group):
            if group is not None:
                torch.distributed.all_reduce(tensor, op=op, group=group)

    def _request_bg_prefetch_epoch(self) -> None:
        if not self.enable_waiting_queue_dfs_prefetch:
            return
        # A failed round also sets the stop flag: report the failure first.
        if self._bg_error is not None:
            raise RuntimeError(
                "Background DFS prefetch is unusable"
            ) from self._bg_error
        with self._bg_condition:
            if self._bg_stop_flag:
                return
            self._bg_requested_epoch += 1
            self._bg_condition.notify()

    def _stop_bg_prefetch_worker(self, timeout_s: float = 30.0) -> bool:
        """Best-effort stop of the background prefetch worker.

        Returns True once the worker has exited. Shutting ranks do not agree
        on a last round, so a worker blocked in a round whose peers already
        stopped may not exit within timeout_s; the caller must then leave
        every resource that worker may still use untouched.
        """
        thread = getattr(self, "_bg_thread", None)
        if thread is None:
            return True
        with self._bg_condition:
            self._bg_stop_flag = True
            self._bg_condition.notify_all()
        thread.join(timeout_s)
        return not thread.is_alive()

    def _drain_bg_prefetch_acks(self) -> None:
        while True:
            try:
                job, submitted = self._bg_prefetch_acks.get_nowait()
            except queue.Empty:
                return
            room = (
                self._bg_prefetch_rooms.pop(job.rid, None)
                if DAS_PREFETCH_TRACE_ENABLED
                else None
            )
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_ack_received",
                    job.rid,
                    room,
                    get_parallel().tp_rank,
                    submitted=submitted,
                )
            accepted = self.tree_cache.complete_external_linker_prefetch(job, submitted)
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "bg_prefetch_ack_applied",
                    job.rid,
                    room,
                    get_parallel().tp_rank,
                    submitted=submitted,
                    accepted=accepted,
                )

    def _begin_scheduler_iteration(self) -> None:
        if not self.enable_waiting_queue_dfs_prefetch:
            return
        self._drain_bg_prefetch_acks()

    def _waiting_queue_prefetch_configured(self) -> bool:
        """This rank's own conditions for waiting-queue prefetch, before the
        linker's capability."""
        return bool(
            get_memory().mooncake_enable_waiting_queue_dfs_prefetch
            and get_memory().enable_unified_cache_external_linker
            and get_memory().unified_cache_external_linker_backend == "mooncake"
            and self.disaggregation_mode
            in (DisaggregationMode.NULL, DisaggregationMode.PREFILL)
            and self.schedule_policy == "fcfs"
            and get_parallel().pp_size == 1
        )

    def _resolve_prefetch_startup_switches(
        self, group: Optional[torch.distributed.ProcessGroup]
    ) -> Tuple[bool, bool]:
        """Agree on (prefetch, forward ingress) before any prefetch group exists.

        create_custom_parallel_group gathers on the whole world, so every
        scheduler rank joins this one reduction unconditionally: a rank whose
        own flags, Mooncake capability or KV canary mode disagree turns the
        feature off everywhere instead of skipping the group creation alone.
        The same reduction requires SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS to be
        on or off everywhere (positive values may differ); otherwise ranks
        would disagree on the supplemental poll collective. That poll only runs
        on a PD prefill with pp_size == 1 (final chunks are recorded only
        there), so other roles do not read the variable and report it off.
        """

        linker_prefetch = (
            get_memory().mooncake_enable_waiting_queue_dfs_prefetch
            and get_memory().enable_unified_cache_external_linker
            and get_memory().unified_cache_external_linker_backend == "mooncake"
            and self.tree_cache.waiting_queue_prefetch_enabled()
        )
        prefetch_local = self._waiting_queue_prefetch_configured() and linker_prefetch
        # Forward ingress matches (and may split) radix nodes while the
        # forward runs; KV canary's sweep walks them from the forward thread.
        ingress_local = (
            str(get_observability().kv_canary).strip().lower() == "none"
            and self.input_blocker is None
        )
        final_poll_on, final_poll_valid = False, True
        final_poll_timeout_ms = 0.0
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and get_parallel().pp_size == 1
        ):
            try:
                final_poll_timeout_ms = envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.get()
                final_poll_on = final_poll_timeout_ms > 0
            except Exception:
                final_poll_on, final_poll_valid = False, False

        flags = torch.tensor(
            [
                int(prefetch_local),
                int(ingress_local),
                int(final_poll_on),
                -int(final_poll_on),
                int(final_poll_valid),
            ],
            dtype=torch.int64,
        )
        if group is not None:
            torch.distributed.all_reduce(
                flags, op=torch.distributed.ReduceOp.MIN, group=group
            )
        prefetch, ingress, final_poll_min, neg_final_poll_max, final_poll_valid = (
            int(value) for value in flags
        )
        if not final_poll_valid or final_poll_min != -neg_final_poll_max:
            raise ValueError(
                "SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS must be valid and either "
                "positive on every rank or <= 0 on every rank; this rank has "
                f"on={final_poll_on}, valid={final_poll_valid}"
            )
        self._mooncake_final_poll_timeout_ms = (
            final_poll_timeout_ms if final_poll_on else 0.0
        )

        enabled = bool(prefetch)
        if linker_prefetch and not enabled:
            # Before any job exists: keep the linker consistent with the world.
            self.tree_cache.disable_waiting_queue_prefetch()
            logger.warning(
                "Waiting-queue DFS prefetch disabled: this or another rank is not "
                "configured for it or lacks the Mooncake capability."
            )
        forward_ingress = enabled and bool(ingress)
        if enabled and not forward_ingress:
            logger.info(
                "Waiting-queue DFS prefetch runs without receiving requests "
                "during forward because KV canary or the input blocker is enabled."
            )
        return enabled, forward_ingress

    def _init_waiting_queue_prefetch_runtime(self) -> None:
        world = get_parallel().world_group
        self.enable_waiting_queue_dfs_prefetch, self._forward_ingress_enabled = (
            self._resolve_prefetch_startup_switches(
                world.cpu_group if world.world_size > 1 else None
            )
        )
        self._forward_launch_executor = None
        self._forward_deferred_reqs = []
        self._bg_thread = None
        self._bg_attn_cp_cpu_group = None
        self._bg_attn_tp_cpu_group = None
        if not self.enable_waiting_queue_dfs_prefetch:
            return
        for attr, group in (
            ("_bg_attn_cp_cpu_group", self.attn_cp_cpu_group),
            ("_bg_attn_tp_cpu_group", self.attn_tp_cpu_group),
        ):
            if group is not None and torch.distributed.get_world_size(group) > 1:
                ranks = torch.distributed.get_process_group_ranks(group)
                setattr(self, attr, create_custom_parallel_group(ranks, backend="gloo"))
        if self._forward_ingress_enabled:
            groups = {}
            for name, group in (
                ("ingress_attn_cp_cpu_group", self.attn_cp_cpu_group),
                ("ingress_attn_tp_cpu_group", self.attn_tp_cpu_group),
                ("ingress_tp_cpu_group", self.tp_cpu_group),
            ):
                groups[name] = group
                if group is not None and torch.distributed.get_world_size(group) > 1:
                    groups[name] = create_custom_parallel_group(
                        torch.distributed.get_process_group_ranks(group), backend="gloo"
                    )
            self.request_receiver = dataclasses.replace(self.request_receiver, **groups)
            self.tree_cache.attn_cp_group = groups["ingress_attn_cp_cpu_group"]
            self.tree_cache.attn_tp_group = groups["ingress_attn_tp_cpu_group"]
            self.tree_cache.tp_group = groups["ingress_tp_cpu_group"]
            self._forward_launch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mooncake-forward"
            )
        self._bg_prefetch_jobs = queue.Queue()
        self._bg_prefetch_acks = queue.Queue()
        self._bg_pending_prefetch_jobs = {}
        self._bg_prefetch_join_age = {}
        self._bg_prefetch_rooms = {} if DAS_PREFETCH_TRACE_ENABLED else None
        self._bg_condition = threading.Condition()
        self._bg_requested_epoch = self._bg_completed_epoch = 0
        self._bg_error = None
        self._bg_stop_flag = False
        self._bg_thread = threading.Thread(
            target=self._bg_worker_loop, daemon=True, name="mooncake-prefetch-round"
        )
        self._bg_thread.start()

    def _prefetch_external_kvcache(self, req: Req, *, is_retracted: bool = False):
        if (
            self.enable_waiting_queue_dfs_prefetch
            and not is_retracted
            and req.prefill_attempt_count == 0
        ):
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "radix_lookup_start",
                    req.rid,
                    req.bootstrap_room,
                    get_parallel().tp_rank,
                    attempt=req.prefill_attempt_count,
                )
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "radix_lookup_done",
                    req.rid,
                    req.bootstrap_room,
                    get_parallel().tp_rank,
                    device_hit_tokens=req.prefix_len,
                    external_host_hit_tokens=req.host_hit_length,
                )
                trace_disagg_prefill(
                    "radix_prefetch_prepare_start",
                    req.rid,
                    req.bootstrap_room,
                    get_parallel().tp_rank,
                )
            job = self.tree_cache.prepare_external_linker_prefetch(req)
            if DAS_PREFETCH_TRACE_ENABLED:
                trace_disagg_prefill(
                    "radix_prefetch_prepare_done",
                    req.rid,
                    req.bootstrap_room,
                    get_parallel().tp_rank,
                    job_created=job is not None,
                    locally_eligible=(
                        job.locally_eligible if job is not None else False
                    ),
                    transfer_count=(len(job.transfers) if job is not None else 0),
                )
            if job is not None:
                if self._bg_prefetch_rooms is not None:
                    self._bg_prefetch_rooms[job.rid] = req.bootstrap_room
                if DAS_PREFETCH_TRACE_ENABLED:
                    trace_disagg_prefill(
                        "bg_prefetch_job_enqueue_start",
                        job.rid,
                        req.bootstrap_room,
                        get_parallel().tp_rank,
                        requested_epoch=self._bg_requested_epoch,
                        completed_epoch=self._bg_completed_epoch,
                    )
                self._bg_prefetch_jobs.put(job)
                self._log_waiting_queue_prefetch_event("enqueued")
                if DAS_PREFETCH_TRACE_ENABLED:
                    trace_disagg_prefill(
                        "bg_prefetch_job_enqueued",
                        job.rid,
                        req.bootstrap_room,
                        get_parallel().tp_rank,
                        requested_epoch=self._bg_requested_epoch,
                        completed_epoch=self._bg_completed_epoch,
                        queue_size=self._bg_prefetch_jobs.qsize(),
                    )

    def _process_mooncake_cleanup_requests(self) -> None:
        pending = self._mooncake_cleanup_requests
        if not self._coordinated_kv_cleanup and not pending:
            return
        present = torch.tensor([int(bool(pending))], dtype=torch.int)
        self._all_reduce_mooncake_cleanup(present, torch.distributed.ReduceOp.MAX)
        if not present.item():
            return
        local = {
            (
                req.rid,
                req.cache_request_handle.lifecycle_id,
                req.cache_request_handle.attempt_id,
            ): req
            for req in pending.values()
        }
        keys = set(local)
        for group in (
            (self.attn_cp_cpu_group, self.attn_tp_cpu_group)
            if self._coordinated_kv_cleanup
            else ()
        ):
            if group is not None and torch.distributed.get_world_size(group) > 1:
                gathered = [None] * torch.distributed.get_world_size(group)
                torch.distributed.all_gather_object(gathered, sorted(keys), group=group)
                keys.update(key for member in gathered for key in member)
        keys = sorted(keys)
        ready = torch.ones((len(keys), 2), dtype=torch.int)
        for index, key in enumerate(keys):
            req = local.get(key)
            if req is not None:
                ready[index, 0] = int(self.prefill_abort_forward_drained(req))
                ready[index, 1] = int(self.prefill_abort_sender_terminal(req))
        self._all_reduce_mooncake_cleanup(ready, torch.distributed.ReduceOp.MIN)
        completed = torch.ones(len(keys), dtype=torch.int)
        for index, key in enumerate(keys):
            req = local.get(key)
            if req is None:
                continue
            done = self.abort_external_kv_request(
                req,
                forward_drained=bool(ready[index, 0]),
                sender_terminal=bool(ready[index, 1]),
                abort_message=getattr(
                    req.to_finish or req.finished_reason,
                    "message",
                    "External KV load failed",
                ),
                emit_response=False,
            )
            completed[index] = -1 if req.external_kv_cleanup_quarantined else int(done)
        self._all_reduce_mooncake_cleanup(completed, torch.distributed.ReduceOp.MIN)
        inflight = getattr(self, "disagg_prefill_inflight_queue", None) or ()
        for index, key in enumerate(keys):
            req = local.get(key)
            if req is not None and completed[index] < 0:
                self._quarantine_external_kv_request(req)
                continue
            if (
                req is None
                or not completed[index]
                or any(candidate is req for candidate in inflight)
            ):
                continue
            if req.external_kv_abort_response_via_chunked:
                self._process_external_kv_chunked_abort(req)
            else:
                self.abort_external_kv_request(
                    req, forward_drained=True, sender_terminal=True, emit_response=True
                )
            if req.external_kv_response_sent:
                pending.pop(id(req), None)

    def _apply_external_linker_admission(self, req) -> bool:
        # The adder has reached and matched this request after its capacity gates.
        state = self.tree_cache.prepare_external_linker_loads([req])[0]
        if state == "pending":
            return False
        if self.tree_cache.supports_mamba():
            # Probe rounds never retain COW metadata or reserve a Mamba slot.
            # Materialize from a fresh local match only once admission can run.
            req.init_next_round_input(self.tree_cache)
        if state == "failed":
            self.tree_cache.cancel_waiting_queue_prefetch(req.rid)
            req.host_hit_length = req.swa_host_hit_length = (
                req.mamba_host_hit_length
            ) = 0
        return True

    def _all_reduce_mooncake_cleanup(self, tensor, op) -> None:
        if (
            not self._coordinated_kv_cleanup
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        for group in (self.attn_cp_cpu_group, self.attn_tp_cpu_group):
            if group is not None and torch.distributed.get_world_size(group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)

    def _forward_with_waiting_queue_ingress(self, batch, **fwd_kwargs):
        if not self._forward_ingress_enabled or not (
            batch.forward_mode.is_extend() or batch.is_extend_in_batch
        ):
            return self.model_worker.forward_batch_generation(batch, **fwd_kwargs)

        # Keep the scheduler as the sole owner of its sockets, requests, and
        # radix tree. Only the model launch moves to another Python thread.
        def launch_forward():
            if self.device != "cpu":
                self.device_module.set_device(self.tp_worker.gpu_id)
            with (
                DynamicGradMode(),
                self.device_module.StreamContext(self.forward_stream),
            ):
                self.forward_stream.wait_stream(self.schedule_stream)
                return self.model_worker.forward_batch_generation(batch, **fwd_kwargs)

        forward_context = contextvars.copy_context()
        future = self._forward_launch_executor.submit(
            forward_context.run, launch_forward
        )
        deferred_reqs = []

        sync_groups = self.request_receiver.ingress_sync_groups()
        while True:
            # A failed launch or background round is fatal: raise it here, at
            # the next checkpoint, instead of after every rank has finished.
            # The other ranks then depend on this node's fatal path (SIGQUIT to
            # the local process tree), remote timeouts or external supervision.
            if future.done() and future.exception() is not None:
                future.result()
            if self._bg_error is not None:
                raise RuntimeError(
                    "Background DFS prefetch is unusable"
                ) from self._bg_error

            # Every rank that shares a receive broadcast must enter the same
            # number of them, even when its model launch finishes earlier.
            active = torch.tensor(
                [
                    int(not future.done()),
                    int(
                        len(self._forward_deferred_reqs) + len(deferred_reqs)
                        >= self.FORWARD_DEFERRED_HIGH_WATER
                    ),
                ],
                dtype=torch.int,
            )
            for group in sync_groups:
                torch.distributed.all_reduce(
                    active, op=torch.distributed.ReduceOp.MAX, group=group
                )
            if not int(active[0].item()):
                break
            # All receive peers pause together, while still joining forward polls.
            if int(active[1].item()):
                wait([future], timeout=0.001)
                continue

            with self.device_module.StreamContext(self.schedule_stream):
                receiver = self.request_receiver
                limit = receiver.max_recv_per_poll
                recv_reqs = receiver.recv_requests(
                    defer_mm=True,
                    max_recv_per_poll=(
                        min(limit, self.FORWARD_RECV_LIMIT)
                        if limit >= 0
                        else self.FORWARD_RECV_LIMIT
                    ),
                )
                self._trace_prefill_reqs(
                    "scheduler_receive", recv_reqs, ingress="forward"
                )
                received_generate = False
                for recv_req in recv_reqs:
                    # Once a control request is deferred, preserve input order.
                    if deferred_reqs:
                        deferred_reqs.append(recv_req)
                        self._trace_prefill_reqs(
                            "forward_defer",
                            [recv_req],
                            reason="ordered_after_deferred",
                            deferred_count=len(deferred_reqs),
                            forward_iter=self.forward_ct,
                        )
                    elif (
                        is_health_check_generate_req(recv_req)
                        and not get_parallel().attn_dp_enabled
                    ):
                        # A forward is running, so the server is busy: answer
                        # as process_input_requests does when not fully idle.
                        self.return_health_check_ipcs.append(
                            getattr(recv_req, "http_worker_ipc", None)
                        )
                    elif isinstance(
                        recv_req, TokenizedGenerateReqInput
                    ) and self._can_prepare_forward_ingress(recv_req):
                        self.handle_generate_request(recv_req)
                        self._log_waiting_queue_prefetch_event("forward_received")
                        received_generate = True
                    elif isinstance(recv_req, BatchTokenizedGenerateReqInput) and all(
                        self._can_prepare_forward_ingress(req) for req in recv_req
                    ):
                        self.handle_batch_generate_request(recv_req)
                        self._log_waiting_queue_prefetch_event(
                            "forward_received", len(recv_req)
                        )
                        received_generate = True
                    else:
                        deferred_reqs.append(recv_req)
                        self._trace_prefill_reqs(
                            "forward_defer",
                            [recv_req],
                            reason="unsupported_during_forward",
                            deferred_count=len(deferred_reqs),
                            forward_iter=self.forward_ct,
                        )
                if received_generate:
                    self._request_bg_prefetch_epoch()
            if not recv_reqs:
                wait([future], timeout=0.001)

        result = future.result()
        # The old non-overlap path put forward and cache updates on one stream.
        # Keep that ordering when the forward runs on its own.
        self.schedule_stream.wait_stream(self.forward_stream)
        self._trace_prefill_reqs(
            "forward_deferred_for_replay",
            deferred_reqs,
            deferred_count=len(deferred_reqs),
            forward_iter=self.forward_ct,
        )
        self._forward_deferred_reqs.extend(deferred_reqs)
        return result

    @staticmethod
    def _can_prepare_forward_ingress(req):
        return (
            not getattr(req, "need_wait_for_mm_inputs", False)
            and getattr(req, "mm_inputs", None) is None
            and getattr(req, "input_embeds", None) is None
            and getattr(req, "session_id", None) is None
            and getattr(req, "session_params", None) is None
            and getattr(req, "lora_id", None) is None
            and getattr(req, "positional_embed_overrides", None) is None
            and all(
                getattr(req.sampling_params, name, None) is None
                for name in ("json_schema", "regex", "ebnf", "structural_tag")
            )
        )
