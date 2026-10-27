"""Spark TP support stays explicit; stochastic choices stay DP-replica-local."""

from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.speculative.dspark_components.spark_attention_tp import (
    broadcast_attention_tp,
    supports_spark_attention_tp,
    supports_spark_target_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _layout():
    model = type("SparkV1ForCausalLM", (), {})
    model.__module__ = "tokenflow.models.sparkv1.model"
    return dict(
        parallel=NS(
            attn_tp_size=8, attn_cp_size=1, pp_size=1, tp_size=32, moe_ep_size=32
        ),
        target_model=model(),
        server_args=NS(
            dp_size=4,
            nnodes=4,
            moe_a2a_backend="deepep",
            moe_runner_backend="deep_gemm",
            speculative_moe_a2a_backend="deepep",
            speculative_moe_runner_backend="deep_gemm",
        ),
        graph_disabled=True,
        overlap_disabled=True,
    )


def test_support_requires_explicit_opt_in_and_native_spark():
    with patch.dict("os.environ", {"SGLANG_SPARK_DSPARK_ATTENTION_TP8": "0"}):
        assert not supports_spark_attention_tp(**_layout())
    with patch.dict(
        "os.environ",
        {
            "SGLANG_SPARK_DSPARK_ATTENTION_TP8": "1",
            "SGLANG_RAGGED_VERIFY_MODE": "static",
        },
    ):
        assert supports_spark_attention_tp(**_layout())
        args = _layout()
        args["target_model"] = type("DeepseekV4ForCausalLM", (), {})()
        assert not supports_spark_attention_tp(**args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_disabled", False),
        ("overlap_disabled", False),
        ("parallel.attn_cp_size", 8),
        ("parallel.moe_ep_size", 16),
        ("parallel.attn_tp_size", 4),
        ("parallel.pp_size", 2),
        ("server_args.dp_size", 16),
        ("server_args.moe_a2a_backend", "none"),
        ("server_args.speculative_moe_a2a_backend", "megamoe"),
    ],
)
def test_unsupported_layouts_keep_the_moe_guard(field, value):
    args = _layout()
    if "." in field:
        obj, attr = field.split(".")
        setattr(args[obj], attr, value)
    else:
        args[field] = value
    with patch.dict(
        "os.environ",
        {
            "SGLANG_SPARK_DSPARK_ATTENTION_TP8": "1",
            "SGLANG_RAGGED_VERIFY_MODE": "static",
        },
    ):
        assert not supports_spark_attention_tp(**args)


def _worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=60),
    )
    try:
        groups = [dist.new_group(list(range(i, i + 4)), backend="gloo") for i in (0, 4)]
        leader = rank // 4 * 4

        class ReplicaGroup:
            def broadcast(self, tensor, src=0):
                assert src == 0
                dist.broadcast(tensor, src=leader, group=groups[rank // 4])

        group = ReplicaGroup()
        # Deliberately different rank RNG positions must yield one Markov chain.
        torch.manual_seed(100 + rank)
        for _ in range(8):
            token = torch.randint(0, 100000, (3,))
            broadcast_attention_tp(token, group)
            gathered = [torch.empty_like(token) for _ in range(8)]
            dist.all_gather(gathered, token)
            assert all(
                torch.equal(gathered[leader], t) for t in gathered[leader : leader + 4]
            )
            assert not torch.equal(gathered[0], gathered[4])
        # Acceptance length and bonus token drive allocator/next-step state.
        for dtype in (torch.int32, torch.int64):
            decision = torch.tensor([rank, rank + 17], dtype=dtype)
            broadcast_attention_tp(decision, group)
            assert decision.tolist() == [leader, leader + 17]
    finally:
        dist.destroy_process_group()


def test_real_gloo_replica_decisions_are_isolated(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=8, join=True)


def test_tp1_does_not_broadcast():
    value = torch.tensor([7])
    assert broadcast_attention_tp(value, None) is value


def _graph_env():
    return {
        "SGLANG_SPARK_DSPARK_ATTENTION_TP8": "1",
        "SGLANG_SPARK_DSPARK_TP8_TARGET_GRAPH": "1",
        "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH": "1",
        "SGLANG_DSPARK_FOLDED_PROPOSAL": "0",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
    }


def _graph_layout():
    args = _layout()
    args["graph_disabled"] = False
    args["server_args"].cuda_graph_max_bs_decode = 16
    args["server_args"].cuda_graph_bs_decode = [1, 2, 4, 8, 12, 16]
    args["server_args"].cuda_graph_backend_decode = "breakable"
    return args


def test_spark_target_graph_requires_separate_opt_in():
    args = _graph_layout()
    with patch.dict("os.environ", _graph_env()):
        assert supports_spark_attention_tp(**args)
        assert supports_spark_target_graph(args["server_args"])
        with patch.dict("os.environ", {"SGLANG_SPARK_DSPARK_TP8_TARGET_GRAPH": "0"}):
            assert not supports_spark_attention_tp(**args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cuda_graph_max_bs_decode", 32),
        ("cuda_graph_max_bs_decode", 0),
        ("cuda_graph_bs_decode", [1, 32]),
        ("cuda_graph_bs_decode", []),
        ("cuda_graph_backend_decode", "full"),
    ],
)
def test_spark_target_graph_rejects_unvalidated_capture_sizes(field, value):
    args = _graph_layout()
    setattr(args["server_args"], field, value)
    with patch.dict("os.environ", _graph_env()):
        assert not supports_spark_attention_tp(**args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH", "0"),
        ("SGLANG_DSPARK_FOLDED_PROPOSAL", "1"),
        ("SGLANG_RAGGED_VERIFY_MODE", "compact"),
    ],
)
def test_spark_target_graph_preserves_draft_and_static_guards(field, value):
    with patch.dict("os.environ", dict(_graph_env(), **{field: value})):
        assert not supports_spark_attention_tp(**_graph_layout())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
