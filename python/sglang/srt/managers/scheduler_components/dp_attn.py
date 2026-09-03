# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import world_dp_gather_enabled
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.recv_skipper import (
    SchedulerRecvSkipper,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()
logger = logging.getLogger(__name__)

# Shared with the scheduler's PD decode StepInfo protocol.
DP_DECODE_STEP_PROTOCOL_VERSION = 2
DP_DECODE_STEP_BUILD_ID = 2026071602
_MLP_SYNC_INFO_WIDTH = 7
_DP_DECODE_STEP_INFO_WIDTH = 10


def _resolve_elastic_world_dp_size(
    dp_size: int,
    *,
    group: torch.distributed.ProcessGroup,
    local_num_tokens: int,
    local_forward_mode: int,
) -> int:
    if not world_dp_gather_enabled():
        return dp_size

    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.layers.dp_attention import get_attention_dp_size

    live_dp_size = get_attention_dp_size()
    effective_ep_size = ElasticEPStateManager.get_effective_ep_size()
    world_size = torch.distributed.get_world_size(group)

    if live_dp_size != effective_ep_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync dp_size is out of sync: "
            f"rank={torch.distributed.get_rank(group)} "
            f"live_dp_size={live_dp_size} effective_ep_size={effective_ep_size} "
            f"world_size={world_size} server_args_dp_size={dp_size} "
            f"local_num_tokens={local_num_tokens} "
            f"local_forward_mode={local_forward_mode}"
        )
    if live_dp_size > world_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync dp_size exceeds WORLD size: "
            f"rank={torch.distributed.get_rank(group)} "
            f"live_dp_size={live_dp_size} world_size={world_size} "
            f"effective_ep_size={effective_ep_size}"
        )

    return live_dp_size


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_run_decode_cuda_graph: bool
    can_run_prefill_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # Appended after the seven model-facing MLP sync fields:
    # [protocol, build_id, epoch, transfer, prealloc, retracted, running, paused,
    #  pd_elapsed_ms, pd_over_budget]
    scheduler_step_info: Optional[list[int]] = None
    gathered_scheduler_step_info: Optional[torch.Tensor] = None

    # some gathered elements
    tp0_info_cpu: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        values = [
            self.num_tokens,
            self.num_tokens_for_logprob,
            int(self.can_run_decode_cuda_graph),
            int(self.is_extend_in_batch),
            int(self.local_can_run_tbo),
            self.local_forward_mode,
            int(self.can_run_prefill_cuda_graph),
        ]
        if self.scheduler_step_info is not None:
            if len(self.scheduler_step_info) != _DP_DECODE_STEP_INFO_WIDTH:
                raise RuntimeError(
                    "PD Decode StepInfo must contain exactly "
                    f"{_DP_DECODE_STEP_INFO_WIDTH} scheduler fields"
                )
            values.extend(self.scheduler_step_info)
        return torch.tensor(values, device=device, dtype=dtype)

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        values = [
            0,  # num_tokens
            0,  # num_tokens_for_logprob
            1,  # can_run_decode_cuda_graph
            0,  # is_extend_in_batch
            1,  # local_can_run_tbo
            ForwardMode.IDLE.value,  # local_forward_mode
            0,  # can_run_prefill_cuda_graph
        ]
        if self.scheduler_step_info is not None:
            values.extend(self.scheduler_step_info)
        return torch.tensor(values, device=device, dtype=dtype)

    def all_gather(
        self,
        device,
        group: torch.distributed.ProcessGroup,
        use_all_reduce: bool = False,
    ):
        local_info_tensor = self._get_local_tensor(device=device)
        fallback_tensor = self._get_fallback_tensor(device=device)
        info_width = local_info_tensor.numel()
        # Inactive max_world_size slots must decode as IDLE. repeat() (not
        # expand().contiguous()) so the buffer never aliases fallback_tensor:
        # at world size 1 the expanded view is already contiguous, contiguous()
        # is a no-op, and the masked fallback writes below would then read and
        # write the same storage.
        global_info_tensor = fallback_tensor.repeat(
            self.dp_size, self.tp_size * self.cp_size, 1
        )

        expected_world = self.dp_size * self.tp_size * self.cp_size
        actual_world = torch.distributed.get_world_size(group)
        if self.scheduler_step_info is not None and actual_world != expected_world:
            raise RuntimeError(
                "PD Decode scheduler group size mismatch: "
                f"actual={actual_world} expected={expected_world}"
            )

        if use_all_reduce:
            if self.scheduler_step_info is not None:
                raise RuntimeError(
                    "PD Decode StepInfo cannot use the elastic WORLD all-reduce"
                )
            # Admission can expose different WORLD sizes; use fixed global slots.
            global_info_tensor.zero_()
            flat_info = global_info_tensor.view(-1, info_width)
            rank = torch.distributed.get_rank(group)
            if 0 <= rank < flat_info.shape[0]:
                flat_info[rank] = local_info_tensor
            torch.distributed.all_reduce(
                global_info_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
            missing = flat_info.abs().sum(dim=1) == 0
            flat_info[missing] = fallback_tensor
        elif self.scheduler_step_info is not None and device == "cpu":
            # Customized HCU Gloo builds may not implement _allgather_base,
            # which backs all_gather_into_tensor. Keep the validated list form
            # on the PD Decode single-clock path.
            gathered = [
                torch.empty_like(local_info_tensor) for _ in range(actual_world)
            ]
            torch.distributed.all_gather(gathered, local_info_tensor, group=group)
            global_info_tensor = torch.stack(gathered, dim=0).view(
                self.dp_size, self.tp_size * self.cp_size, info_width
            )
        else:
            torch.distributed.all_gather_into_tensor(
                global_info_tensor.flatten(),
                local_info_tensor,
                group=group,
            )

        if self.scheduler_step_info is not None:
            step_all = global_info_tensor.view(-1, info_width)[
                :, _MLP_SYNC_INFO_WIDTH:
            ].cpu()
            versions = step_all[:, 0]
            builds = step_all[:, 1]
            epochs = step_all[:, 2]
            if (
                int(versions.min().item()) != DP_DECODE_STEP_PROTOCOL_VERSION
                or int(versions.max().item()) != DP_DECODE_STEP_PROTOCOL_VERSION
            ):
                raise RuntimeError(
                    "PD Decode StepInfo protocol mismatch across ranks: "
                    f"{versions.tolist()}"
                )
            if (
                int(builds.min().item()) != DP_DECODE_STEP_BUILD_ID
                or int(builds.max().item()) != DP_DECODE_STEP_BUILD_ID
            ):
                raise RuntimeError(
                    "PD Decode StepInfo build mismatch across ranks: "
                    f"{builds.tolist()}"
                )
            if int(epochs.min().item()) != int(epochs.max().item()):
                raise RuntimeError(
                    "PD Decode scheduler epoch mismatch across ranks: "
                    f"{epochs.tolist()}"
                )
            paused = step_all[:, 7]
            if int(paused.min().item()) != int(paused.max().item()):
                raise RuntimeError(
                    "PD Decode paused-state mismatch across ranks: "
                    f"{paused.tolist()}"
                )

        tp_info = global_info_tensor.view(
            self.dp_size * self.tp_size * self.cp_size, info_width
        )
        num_ranks_in_tp_info = tp_info.shape[0]
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks
        if tp_active_ranks.shape[0] < num_ranks_in_tp_info:
            tp_active_ranks = torch.ones(
                num_ranks_in_tp_info,
                dtype=tp_active_ranks.dtype,
                device=tp_active_ranks.device,
            )
        inactive = tp_active_ranks[:num_ranks_in_tp_info] == 0
        if self.scheduler_step_info is None:
            tp_info[inactive] = fallback_tensor
        else:
            # Preserve each rank's epoch/diagnostics while masking only its
            # model-facing MLP sync fields.
            tp_info[inactive, :_MLP_SYNC_INFO_WIDTH] = fallback_tensor[
                :_MLP_SYNC_INFO_WIDTH
            ]

        # One D2H for every field: each `.item()` / `.tolist()` on a device
        # tensor is its own stream sync. Copy the whole tensor, not the
        # `[:, 0, :]` slice -- that slice is non-contiguous once
        # attn_tp * attn_cp > 1, adding a gather kernel inside the wait.
        tp0_info_cpu = global_info_tensor.cpu()[:, 0, :]
        self.tp0_info_cpu = tp0_info_cpu
        self.global_num_tokens = tp0_info_cpu[:, 0].tolist()
        self.global_num_tokens_for_logprob = tp0_info_cpu[:, 1].tolist()
        self.can_run_decode_cuda_graph = bool(tp0_info_cpu[:, 2].min())
        self.is_extend_in_batch = bool(tp0_info_cpu[:, 3].max())
        self.can_run_prefill_cuda_graph = bool(tp0_info_cpu[:, 6].min())
        if self.scheduler_step_info is not None:
            step_tp0 = tp0_info_cpu[:, _MLP_SYNC_INFO_WIDTH:]
            self.gathered_scheduler_step_info = step_tp0
            epoch = int(step_tp0[0, 2].item())
            over_budget = int(step_tp0[:, 9].max().item())
            if torch.distributed.get_rank(group) == 0 and (
                over_budget or epoch % 1024 == 0
            ):

                def _minmax(col: int) -> tuple[int, int]:
                    return (
                        int(step_tp0[:, col].min().item()),
                        int(step_tp0[:, col].max().item()),
                    )

                logger.info(
                    "DP Decode StepInfo epoch=%s transfer=%s prealloc=%s "
                    "retracted=%s running=%s paused=%s pd_ms=%s over_budget=%s",
                    epoch,
                    _minmax(3),
                    _minmax(4),
                    _minmax(5),
                    _minmax(6),
                    _minmax(7),
                    _minmax(8),
                    _minmax(9),
                )
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                tp0_info_cpu[:, 5].tolist()
            )


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if skip_all_gather:
        batch.global_num_tokens_all_dp = None
        batch.global_num_tokens_for_logprob_all_dp = None
    else:
        batch.global_num_tokens_all_dp = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob_all_dp = (
            mlp_sync_info.global_num_tokens_for_logprob
        )

    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_run_decode_cuda_graph
    batch.can_run_dp_breakable_cuda_graph = mlp_sync_info.can_run_prefill_cuda_graph


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    model_runner: ModelRunner,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
    dwdp: bool = False,
    scheduler_step_info: Optional[list[int]] = None,
    sync_group_override: Optional[torch.distributed.ProcessGroup] = None,
):
    # Check if other DP workers have running batches
    if (
        local_batch is None
        or local_batch.forward_mode.is_prebuilt()
        or local_batch.forward_mode.is_idle()
    ):
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    # With a single DP replica and no residual attention-TP/MLP-TP gather,
    # every CP rank receives the same work through the CP control broadcast.
    # The scheduler metadata all-gather is therefore redundant and can hang
    # on DCU CPU process groups during the idle loop.  Reuse the established
    # skip path, which records each rank's local token count as required by CP.
    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get() or (
        dp_size == 1 and attn_tp_size == 1 and not require_mlp_tp_gather
    )
    can_run_decode_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph
    breakable_prefill = check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
    prefill_graph_runner = (
        model_runner.prefill_cuda_graph_runner if breakable_prefill else None
    )
    can_run_prefill_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_idle()
        # Breakable Cuda Graph Backend Check.
        or (
            local_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
            and (
                prefill_graph_runner is None
                or prefill_graph_runner.can_replay_locally(
                    batch_size=local_batch.batch_size(),
                    num_tokens=local_batch.extend_num_tokens,
                    input_embeds=local_batch.input_embeds,
                    replace_embeds=None,
                    prefix_lens=local_batch.prefix_lens,
                    is_target_verify=local_batch.forward_mode.is_target_verify(),
                    capture_hidden_mode=None,
                    return_logprob=local_batch.return_logprob,
                    lora_ineligible=prefill_graph_runner.enable_lora,
                )
            )
            and breakable_prefill
        )
    )

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    use_world_group = sync_group_override is None and world_dp_gather_enabled()
    if sync_group_override is not None:
        group = sync_group_override
        device = "cpu"
    elif use_world_group:
        from sglang.srt.distributed.parallel_state import get_world_group

        world = get_world_group()
        group = torch.distributed.group.WORLD
        device = world.device
    elif len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)
    if use_world_group:
        dp_size = _resolve_elastic_world_dp_size(
            dp_size,
            group=group,
            local_num_tokens=num_tokens,
            local_forward_mode=local_forward_mode,
        )

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_run_decode_cuda_graph=can_run_decode_cuda_graph,
        can_run_prefill_cuda_graph=can_run_prefill_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
        scheduler_step_info=scheduler_step_info,
    )

    if scheduler_step_info is not None and skip_all_gather:
        raise RuntimeError(
            "PD Decode DP sync is incompatible with SGLANG_SCHEDULER_SKIP_ALL_GATHER"
        )

    if not skip_all_gather:
        mlp_sync_info.all_gather(
            device=device,
            group=group,
            use_all_reduce=use_world_group,
        )

        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info_cpu[:, 4:6],
            )
        )

    # Decide whether to emit idle batch
    if skip_all_gather:
        # Skip idle batch when attn-dp=1 (and always under DWDP: ranks run independently)
        need_idle_batch = not dwdp and dp_size > 1
    else:
        need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0

    batch_to_gather = local_batch
    if need_idle_batch:
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()

    if batch_to_gather is not None:
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    # Set on `local_batch`, not `batch_to_gather`: for PREBUILT batches the
    # scheduler's `last_batch` is the prebuilt batch, not its inner idle batch.
    if local_batch is not None and not skip_all_gather:
        local_batch.recv_skipper_forward_mode = (
            SchedulerRecvSkipper.derive_forward_mode(
                mlp_sync_info.tp0_info_cpu[:, 5].tolist()
            )
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerDPAttnAdapter:
    model_runner: ModelRunner
    tp_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    offload_tags: set[str]
    ps: ParallelState
    server_args: ServerArgs
    model_config: ModelConfig
    enable_overlap: bool
    spec_algorithm: SpeculativeAlgorithm
    get_require_mlp_sync: Callable[[], bool]
    # The scheduler supplies the dedicated group and epoch-tagged StepInfo;
    # the adapter folds them into the same all-gather as the MLP metadata.
    get_pd_decode_step_context: Callable[
        [], Optional[tuple[torch.distributed.ProcessGroup, int, list[int]]]
    ]
    set_dp_scheduler_epoch: Callable[[int], None]

    def prepare_mlp_sync_batch(self, local_batch: ScheduleBatch):
        step_context = self.get_pd_decode_step_context()
        if step_context is None:
            sync_group, epoch, scheduler_step_info = None, None, None
        else:
            sync_group, epoch, scheduler_step_info = step_context

        result = prepare_mlp_sync_batch_raw(
            local_batch,
            model_runner=self.model_runner,
            dp_size=get_parallel().dp_size,
            attn_tp_size=self.ps.attn_tp_size,
            attn_cp_size=self.ps.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=cuda_graph_fully_disabled(),
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=get_schedule().disable_overlap_schedule,
            offload_tags=self.offload_tags,
            dwdp=get_parallel().dwdp_size > 1,
            scheduler_step_info=scheduler_step_info,
            sync_group_override=sync_group,
        )
        if epoch is not None:
            # Advance only after every rank completed and validated this epoch.
            self.set_dp_scheduler_epoch(epoch + 1)
        return result

    def maybe_prepare_mlp_sync_batch(
        self,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.get_require_mlp_sync() for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.get_require_mlp_sync():
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def get_idle_batch(self) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
