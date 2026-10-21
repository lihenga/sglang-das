"""Restore global token order for Spark auxiliary captures under prefill CP."""

import functools
import logging

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


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
    gathered = cp_all_gather_rerange_output(
        packed, cp_size, forward_batch, torch.cuda.current_stream(packed.device)
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


def install_spark_v1_dspark_cp_adapter() -> bool:
    """Add the missing output gather without replacing native Spark math."""
    from tokenflow.models.sparkv1.model import SparkV1Model

    original = SparkV1Model.forward
    if getattr(original, "_sglang_spark_cp_aux_gather", False):
        return True

    @functools.wraps(original)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        result = original(self, input_ids, positions, forward_batch, *args, **kwargs)
        if (
            get_parallel().attn_cp_size <= 1
            or getattr(forward_batch, "attn_cp_metadata", None) is None
            or not forward_batch.forward_mode.is_context_parallel_extend()
            or not self.pp_group.is_last_rank
            or not isinstance(result, tuple)
            or len(result) != 2
        ):
            return result
        hidden_states, aux_hidden_states = result
        aux_hidden_states = _gather_cp_aux_hidden_states(
            hidden_states, aux_hidden_states, forward_batch, get_parallel().attn_cp_size
        )
        return hidden_states, aux_hidden_states

    forward._sglang_spark_cp_aux_gather = True
    SparkV1Model.forward = forward
    logger.info("Installed Spark prefill CP auxiliary-hidden gather in global token order")
    return True
