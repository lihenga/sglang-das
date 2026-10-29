"""Restore global token order for Spark auxiliary captures under prefill CP."""

import functools
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

@dataclass
class _LocalAuxProjection:
    model: object
    projected: bool = False


_LOCAL_AUX_PROJECTION = ContextVar("spark_cp_local_aux_projection", default=None)
_PROJECTED_TARGET_INPUT = ContextVar("spark_cp_projected_target_input", default=None)


@contextmanager
def cp_prefill_aux_projection(model, *, enabled=True):
    """Lease a request-local projector only for the actual Spark draft class."""
    state = _LocalAuxProjection(model)
    supported = (
        enabled
        and type(model).__name__ == "SparkV1DSparkDraftModel"
        and type(model).__module__ == "tokenflow.models.sparkv1.dspark_draft"
    )
    token = _LOCAL_AUX_PROJECTION.set(state if supported else None)
    try:
        yield state
    finally:
        _LOCAL_AUX_PROJECTION.reset(token)


@contextmanager
def projected_target_hidden_input(model, *, enabled=True):
    """Keep the native KV writer interface; the target projection is done."""
    token = _PROJECTED_TARGET_INPUT.set(model if enabled else None)
    try:
        yield
    finally:
        _PROJECTED_TARGET_INPUT.reset(token)



def _gather_cp_aux_hidden_states(
    hidden_states, aux_hidden_states, forward_batch, cp_size
):
    """Use the same token permutation as the native final-hidden gather.

    Spark's native backbone gathers its final output but returns rank-local
    intermediate captures. Logits pruning and DSpark KV injection both consume
    the global token order. Pack features for one collective, then preserve the
    caller's tensor/list representation. Already-global captures need no work.
    """
    captures = (
        [aux_hidden_states]
        if isinstance(aux_hidden_states, torch.Tensor)
        else aux_hidden_states
    )
    if not captures:
        return aux_hidden_states
    global_rows = hidden_states.shape[0]
    if all(capture.shape[0] == global_rows for capture in captures):
        return aux_hidden_states
    if not all(
        capture.ndim == 2
        and capture.shape[0] == captures[0].shape[0]
        and capture.device == hidden_states.device
        and capture.dtype == hidden_states.dtype
        for capture in captures
    ):
        raise RuntimeError(
            "Spark CP auxiliary captures have inconsistent shapes or dtype"
        )

    packed = pack_aux_hidden_states(aux_hidden_states).contiguous()
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    projection = _LOCAL_AUX_PROJECTION.get()
    actual_rows = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if type(actual_rows) is not int:
        lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
        actual_rows = sum(lengths) if isinstance(lengths, (list, tuple)) else None
    if (
        projection is not None
        and not projection.projected
        and cp_size > 1
        and type(actual_rows) is int
        and actual_rows >= 1024
        and (actual_rows + cp_size - 1) // cp_size == packed.shape[0]
        and packed.is_cuda
        and packed.dtype == torch.bfloat16
        and not torch.cuda.is_current_stream_capturing()
        and is_dsa_prefill_cp_round_robin_split()
        and packed.shape[0] * cp_size == global_rows
        and packed.shape[1]
        == projection.model.config.hidden_size * projection.model.num_context_features
    ):
        method = type(projection.model).project_target_hidden
        if not getattr(method, "_sglang_cp_target_projection", False):
            raise RuntimeError("Spark CP target projector adapter is not installed")
        local = method.__wrapped__(projection.model, packed).contiguous()
        if (
            local.ndim != 2
            or local.shape != (packed.shape[0], projection.model.config.hidden_size)
            or local.dtype != packed.dtype
            or local.device != packed.device
        ):
            raise RuntimeError("Spark local target projection changed its contract")
        gathered = cp_all_gather_rerange_output(
            local,
            cp_size,
            forward_batch,
            torch.cuda.current_stream(local.device),
            reorder_feature_block_size=min(local.shape[1], 2048),
        )
        if gathered.shape[0] != global_rows:
            raise RuntimeError("Spark projected CP gather has inconsistent token rows")
        projection.projected = True
        return gathered

    gathered = cp_all_gather_rerange_output(
        packed,
        cp_size,
        forward_batch,
        torch.cuda.current_stream(packed.device),
        reorder_feature_block_size=min(hidden_states.shape[-1], 2048),
    )
    if gathered.shape[0] != global_rows:
        raise RuntimeError(
            "Spark CP auxiliary gather returned "
            f"{gathered.shape[0]} rows; final hidden has {global_rows}"
        )
    if isinstance(aux_hidden_states, torch.Tensor):
        return gathered
    widths = [capture.shape[1] for capture in captures]
    return list(gathered.split(widths, dim=-1))


def _gather_tp_aux_hidden_states(hidden_states, aux_hidden_states, tp_group, tp_size):
    """Align local and already-gathered Spark captures before feature packing."""
    captures = (
        [aux_hidden_states]
        if isinstance(aux_hidden_states, torch.Tensor)
        else aux_hidden_states
    )
    if not captures:
        return aux_hidden_states
    rows = hidden_states.shape[0]
    gathered = []
    for capture in captures:
        if capture.ndim != 2 or capture.device != hidden_states.device:
            raise RuntimeError("Spark TP auxiliary capture has invalid rank/device")
        if capture.shape[0] == rows:
            gathered.append(capture)
        elif capture.shape[0] * tp_size == rows:
            gathered.append(tp_group.all_gather(capture, dim=0))
        else:
            raise RuntimeError(
                f"Spark TP auxiliary rows {capture.shape[0]} cannot match "
                f"final rows {rows} with attention TP{tp_size}"
            )
    return gathered[0] if isinstance(aux_hidden_states, torch.Tensor) else gathered


def _install_cp_target_projection():
    """Partition token-independent context projection across prefill CP ranks."""
    from tokenflow.models.sparkv1.dspark_draft import SparkV1DSparkDraftModel

    original = SparkV1DSparkDraftModel.project_target_hidden
    if getattr(original, "_sglang_cp_target_projection", False):
        return

    @functools.wraps(original)
    def project(self, target_hidden):
        if _PROJECTED_TARGET_INPUT.get() is self:
            if target_hidden.ndim != 2 or target_hidden.shape[1] != self.config.hidden_size:
                raise RuntimeError("Spark preprojected target hidden has invalid shape")
            return target_hidden
        parallel = get_parallel()
        size = parallel.attn_cp_size
        rows = target_hidden.shape[0]
        if size <= 1 or rows < 1024 or torch.cuda.is_current_stream_capturing():
            return original(self, target_hidden)
        # The existing auxiliary gather already restored global token order.
        # Partition that order, then gather the projected rows in the same order.
        local_rows = (rows + size - 1) // size
        start = parallel.attn_cp_rank * local_rows
        local = target_hidden[start:min(start + local_rows, rows)]
        if local.shape[0] != local_rows:
            local = torch.nn.functional.pad(local, (0, 0, 0, local_rows - local.shape[0]))
        projected = original(self, local).contiguous()
        output = projected.new_empty((size * local_rows, projected.shape[1]))
        parallel.attn_cp_group.all_gather_into_tensor(output, projected)
        return output[:rows]

    project._sglang_cp_target_projection = True
    SparkV1DSparkDraftModel.project_target_hidden = project


def install_spark_v1_dspark_cp_adapter() -> bool:
    """Add the missing output gather without replacing native Spark math."""
    from tokenflow.models.sparkv1.model import SparkV1Model

    _install_cp_target_projection()

    original = SparkV1Model.forward
    if getattr(original, "_sglang_spark_cp_aux_gather", False):
        return True

    @functools.wraps(original)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        result = original(self, input_ids, positions, forward_batch, *args, **kwargs)
        parallel = get_parallel()
        if parallel.attn_cp_size <= 1 and parallel.attn_tp_size <= 1:
            return result
        if (
            not self.pp_group.is_last_rank
            or not isinstance(result, tuple)
            or len(result) != 2
        ):
            return result
        hidden_states, aux_hidden_states = result
        if (
            parallel.attn_tp_size > 1
            and (
                os.getenv("SGLANG_SPARK_DSPARK_ATTENTION_TP8", "0") == "1"
                or (
                    parallel.attn_tp_size == 2
                    and os.getenv("SGLANG_SPARK_DSPARK_ATTENTION_TP2", "0") == "1"
                )
            )
        ):
            aux_hidden_states = _gather_tp_aux_hidden_states(
                hidden_states,
                aux_hidden_states,
                parallel.attn_tp_group,
                parallel.attn_tp_size,
            )
        if (
            parallel.attn_cp_size <= 1
            or getattr(forward_batch, "attn_cp_metadata", None) is None
            or not forward_batch.forward_mode.is_context_parallel_extend()
        ):
            return hidden_states, aux_hidden_states
        aux_hidden_states = _gather_cp_aux_hidden_states(
            hidden_states, aux_hidden_states, forward_batch, get_parallel().attn_cp_size
        )
        return hidden_states, aux_hidden_states

    forward._sglang_spark_cp_aux_gather = True
    SparkV1Model.forward = forward
    logger.info("Installed Spark CP/TP auxiliary-hidden gather in global token order")
    return True
