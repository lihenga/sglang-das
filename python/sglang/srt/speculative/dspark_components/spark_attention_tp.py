"""Opt-in Spark MoE draft support for attention TP8 / DP4 / EP32.

TokenFlow routes Spark's native MoE draft through the DSV4 worker contract,
while its decoder uses SparkV1MoE rather than DeepseekV4MoE. Keep the generic
DSV4 restriction intact and admit only this explicit Spark deployment.
"""

import os


def supports_spark_target_graph(server_args):
    """Allow target-only graphs up to BS16; draft collectives stay eager."""
    max_bs = getattr(server_args, "cuda_graph_max_bs_decode", None)
    buckets = getattr(server_args, "cuda_graph_bs_decode", None)
    return (
        os.getenv("SGLANG_SPARK_DSPARK_TP8_TARGET_GRAPH", "0") == "1"
        and os.getenv("SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH", "0") == "1"
        and os.getenv("SGLANG_DSPARK_FOLDED_PROPOSAL", "0") == "0"
        and getattr(server_args, "cuda_graph_backend_decode", None) == "breakable"
        and max_bs is not None
        and 0 < max_bs <= 16
        and bool(buckets)
        and all(0 < bs <= max_bs for bs in buckets)
    )


def supports_spark_attention_tp(
    *, parallel, target_model, server_args, graph_disabled, overlap_disabled
):
    model_type = type(target_model)
    return (
        os.getenv("SGLANG_SPARK_DSPARK_ATTENTION_TP8", "0") == "1"
        and model_type.__name__ == "SparkV1ForCausalLM"
        and model_type.__module__ == "tokenflow.models.sparkv1.model"
        and parallel.attn_tp_size == 8
        and parallel.attn_cp_size == 1
        and parallel.pp_size == 1
        and parallel.tp_size == 32
        and parallel.moe_ep_size == 32
        and server_args.dp_size == 4
        and server_args.nnodes == 4
        and server_args.moe_a2a_backend == "deepep"
        and server_args.moe_runner_backend == "deep_gemm"
        and server_args.speculative_moe_a2a_backend in (None, "deepep")
        and server_args.speculative_moe_runner_backend in (None, "deep_gemm")
        and os.getenv("SGLANG_RAGGED_VERIFY_MODE", "static") == "static"
        and (graph_disabled or supports_spark_target_graph(server_args))
        and overlap_disabled
    )


def broadcast_attention_tp(tensor, group):
    """Keep stochastic Markov decisions and KV commit lengths replica-wide.

    Broadcast each sampled token before the next Markov step consumes it.
    Identical initial seeds alone do not guarantee identical RNG positions
    after different rank-local warmup or idle participation histories.
    """
    if group is not None:
        group.broadcast(tensor, src=0)
    return tensor
