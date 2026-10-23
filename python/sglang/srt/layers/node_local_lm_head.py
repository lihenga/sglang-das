"""Opt-in node-local LM-head weight sharding for pure attention DP decode."""

from __future__ import annotations

import logging
import os
from functools import wraps
from functools import lru_cache

import torch


def requested_lm_head_tp_size() -> int:
    size = int(os.getenv("SGLANG_PERF_LM_HEAD_TP_SIZE", "0"))
    if size not in (0, 8):
        raise ValueError("SGLANG_PERF_LM_HEAD_TP_SIZE supports 0 (off) or 8")
    return size


def validate_lm_head_tp_layout(parallel, size: int) -> int:
    if (
        size != 8
        or parallel.tp_size % size
        or parallel.attn_dp_size != parallel.tp_size
        or parallel.attn_tp_size != 1
    ):
        raise ValueError("Node-local TP8 LM head requires pure attention DP, TP divisible by 8, attention TP1")
    return parallel.tp_rank % size


def node_local_groups(tp_ranks, size: int, local_world_size: int):
    ranks = list(tp_ranks)
    if size != 8 or local_world_size != 8 or len(ranks) % size:
        raise ValueError("Node-local TP8 LM head requires 8 visible GPUs per node")
    groups = [ranks[i : i + size] for i in range(0, len(ranks), size)]
    for group in groups:
        if group != list(range(group[0], group[0] + size)) or len({r // local_world_size for r in group}) != 1:
            raise ValueError(f"LM-head group is not contiguous and node-local: {group}")
    return groups


def node_local_row_layout(global_counts_cpu, has_gpu_counts: bool, local_rows: int, tp_ranks, group_ranks, rank: int):
    """Use shared CPU counts in eager mode; graphs have equal padded rows.

    No device scalar is read on the CPU. All eight participants derive the
    same communication shape, including a node with no active requests.
    """
    if global_counts_cpu is None:
        if not has_gpu_counts:
            raise ValueError("TP8 LM head requires DP row metadata")
        return local_rows
    if len(global_counts_cpu) != len(tp_ranks):
        raise ValueError("LM-head DP row count vector has the wrong length")
    positions = {r: i for i, r in enumerate(tp_ranks)}
    counts = [global_counts_cpu[positions[r]] for r in group_ranks]
    if any(type(c) is not int or c < 0 for c in counts):
        raise ValueError("LM-head row counts must be nonnegative integers")
    if counts[group_ranks.index(rank)] != local_rows:
        raise ValueError(f"LM-head pruned rows disagree with shared DP metadata: local={local_rows}, counts={counts}")
    return max(counts)


def reassemble_node_logits(received, size: int, local_rows: int):
    # Each source contributes its vocabulary shard for this destination's rows.
    rows = received.shape[0] // size
    return received.view(size, rows, -1).permute(1, 0, 2).reshape(rows, -1)[:local_rows]


def populate_lm_head_dp_metadata(forward_batch, schedule_batch):
    """Keep LM-head counts separate from DeepEP's local-only MLP metadata."""
    if not requested_lm_head_tp_size():
        return
    counts = schedule_batch.global_num_tokens_all_dp
    logprob_counts = schedule_batch.global_num_tokens_for_logprob_all_dp
    if counts is None or logprob_counts is None:
        raise ValueError("TP8 LM head requires the scheduler's full DP count exchange")
    forward_batch.lm_head_global_num_tokens_cpu = list(counts)
    if forward_batch.spec_info is not None:
        from sglang.srt.speculative.spec_info import spec_scale_global_num_tokens

        _, logprob_counts = spec_scale_global_num_tokens(
            forward_batch.spec_info, counts, logprob_counts
        )
    forward_batch.lm_head_global_num_tokens_for_logprob_cpu = list(logprob_counts)


class NodeLocalLMHeadGroup:
    def __init__(self, size: int):
        from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group, get_tp_group

        self.tp_ranks = tuple(get_tp_group().ranks)
        self.rank = torch.distributed.get_rank()
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_SIZE") or torch.cuda.device_count())
        groups = node_local_groups(self.tp_ranks, size, local_size)
        self.ranks = next(g for g in groups if self.rank in g)
        self.size = size
        self.local_rank = self.ranks.index(self.rank)
        # This helper contains a WORLD collective. Every TP rank enters once
        # during target LogitsProcessor construction, before graph capture.
        self.process_group = create_custom_parallel_group(self.ranks, backend="gloo")
        self.comm = PyNcclCommunicator(self.process_group, device=torch.cuda.current_device())
        if not self.comm.available:
            raise RuntimeError("Node-local LM head requires PyNccl")
        warmup = torch.zeros(1, device="cuda", dtype=torch.float32)
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(warmup)
        torch.cuda.synchronize()

    def project(self, hidden_states, lm_head, metadata, compute, embedding_bias=None):
        if lm_head.tp_size != self.size or lm_head.num_embeddings != lm_head.org_vocab_size or lm_head.num_embeddings_padded != lm_head.num_embeddings:
            raise ValueError("TP8 LM head requires an unpadded vocabulary without added tokens")
        if lm_head.shard_indices.org_vocab_start_index != self.local_rank * lm_head.num_embeddings_per_partition:
            raise ValueError("LM-head loaded vocabulary shard disagrees with node communicator rank")
        local_rows = hidden_states.shape[0]
        counts = getattr(metadata, "lm_head_global_num_tokens_for_logprob_cpu", None)
        if counts is None:
            # Captured batches use equal padded rows and GPU-only metadata.
            counts = metadata.global_num_tokens_for_logprob_cpu
        padded_rows = node_local_row_layout(
            counts,
            metadata.global_num_tokens_for_logprob_gpu is not None,
            local_rows, self.tp_ranks, self.ranks, self.rank,
        )
        if padded_rows == 0:
            return hidden_states.new_empty((0, lm_head.num_embeddings))
        local = hidden_states.new_zeros((padded_rows, hidden_states.shape[1]))
        local[:local_rows].copy_(hidden_states)
        gathered = hidden_states.new_empty((self.size * padded_rows, hidden_states.shape[1]))
        with self.comm.change_state(enable=True):
            self.comm.all_gather(gathered, local)
        logits = compute(gathered, lm_head, embedding_bias).contiguous()
        received = torch.empty_like(logits)
        with self.comm.change_state(enable=True):
            self.comm.all_to_all_single(received.view(-1), logits.view(-1))
        return reassemble_node_logits(received, self.size, local_rows)


@lru_cache(maxsize=None)
def get_node_local_lm_head_group(size: int):
    # Target and draft must share the same communicator and initialization epoch.
    return NodeLocalLMHeadGroup(size)


logger = logging.getLogger(__name__)


def configure_dspark_lm_head(model):
    """Give a Spark draft its own node-local vocabulary shard.

    Native TokenFlow keeps ownership of forward, muP scaling, confidence and
    sampling. A tied draft embedding remains replicated for token lookup; only
    the projection reads a contiguous vocabulary shard of that same weight.
    """
    size = requested_lm_head_tp_size()
    if not size:
        return
    from sglang.srt.runtime_context import get_parallel

    validate_lm_head_tp_layout(get_parallel(), size)
    if type(model).__name__ != "SparkV1DSparkDraftModel":
        raise ValueError("Node-local DSpark head is validated only for SparkV1")
    head = model.lm_head
    if head is None or head.weight.dtype != torch.bfloat16:
        raise ValueError("Node-local DSpark head requires a loaded BF16 head")
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    if should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None)):
        raise ValueError("Quantized DSpark vocabulary heads are unsupported")
    group = get_node_local_lm_head_group(size)
    vocab = int(head.org_vocab_size)
    if vocab % size:
        raise ValueError("Node-local DSpark vocabulary must be divisible by 8")
    if head.weight.shape[0] == vocab:
        # A distinct module prevents changing a checkpoint-tied token embedding.
        from types import SimpleNamespace

        shard = torch.nn.Module()
        begin = group.local_rank * (vocab // size)
        shard.weight = torch.nn.Parameter(
            head.weight[begin : begin + vocab // size].contiguous().clone(),
            requires_grad=False,
        )
        shard.tp_size = size
        shard.org_vocab_size = vocab
        shard.num_embeddings = vocab
        shard.num_embeddings_padded = vocab
        shard.num_embeddings_per_partition = vocab // size
        shard.shard_indices = SimpleNamespace(org_vocab_start_index=begin)
        shard.quant_method = None
        model.lm_head = shard
        head = shard
    elif head.weight.shape[0] != vocab // size or head.tp_size != size:
        raise ValueError("DSpark LM-head shard does not match the node group")

    head._node_local_lm_head_group = group
    original_forward = model.forward

    @wraps(original_forward)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        # Graph capture supplies padded metadata. Eager decode supplies shared
        # CPU row counts, including empty ranks. No device scalar is read here.
        head._node_local_lm_head_metadata = forward_batch
        return original_forward(input_ids, positions, forward_batch, *args, **kwargs)

    model.forward = forward
    logger.info(
        "Node-local TP%d DSpark LM head: vocabulary shard %s; native sampler retained",
        size, tuple(head.weight.shape),
    )
