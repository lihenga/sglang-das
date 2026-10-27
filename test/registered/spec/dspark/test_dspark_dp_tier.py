import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_planner import (
    dp_global_verify_tier_num_tokens,
    local_verify_tier_num_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLocalVerifyTierNumTokens(CustomTestCase):
    def test_no_budget_returns_sentinel(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=None,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            -1,
        )

    def test_budget_adds_to_anchor_floor(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=10,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            18,
        )

    # Clamp/floor variants (verify-all clamp, min_verify_len floor, min=0) are
    # covered by the TestBusyIdleGraphKeyIdentity sweep bounds.


class TestDpGlobalVerifyTierNumTokens(CustomTestCase):
    def test_any_sentinel_pins_everyone(self):
        # The sweep never emits a -1 contribution, so this is the only guard
        # on "any rank without a budget pins everyone"; losing it forks graph
        # keys across DP ranks.
        self.assertIsNone(
            dp_global_verify_tier_num_tokens(global_tier_num_tokens=[100, -1, 50, 0])
        )


class TestDraftDpSyncMetadata(CustomTestCase):
    def test_preserves_unscaled_request_counts_for_cuda_graph_admission(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer._prep_cache_enabled = False
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=6,
            num_tokens_for_logprob_per_req=1,
        )
        proposer.draft_model_runner = SimpleNamespace(device="cpu")

        forward_batch = SimpleNamespace(input_ids=torch.arange(6))
        batch = SimpleNamespace(
            global_num_tokens=[1, 3, 0, 2],
            global_num_tokens_for_logprob=[1, 3, 0, 2],
            can_run_dp_cuda_graph=True,
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=True,
        ):
            proposer._fill_dp_moe_sync_metadata(forward_batch, batch)

        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu,
            [1, 3, 0, 2],
        )
        self.assertEqual(forward_batch.global_num_tokens_cpu, [6, 18, 0, 12])
        self.assertEqual(forward_batch.num_token_non_padded.item(), 6)
        self.assertEqual(forward_batch.num_token_non_padded.dtype, torch.int32)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 6)
        self.assertTrue(forward_batch.can_run_dp_cuda_graph)


class TestBusyIdleGraphKeyIdentity(CustomTestCase):
    def test_busy_and_idle_floors_agree_on_random_topologies(self):
        rng = random.Random(20260703)
        for _ in range(2000):
            verify_num_draft_tokens = rng.randint(2, 8)
            min_verify_len = rng.randint(0, verify_num_draft_tokens - 1)
            effective_min = max(min_verify_len, 1)
            num_ranks = rng.randint(1, 8)
            contributions = []
            num_reqs_per_rank = []
            for _ in range(num_ranks):
                if rng.random() < 0.3:
                    num_reqs_per_rank.append(0)
                    contributions.append(0)
                    continue
                bs = rng.randint(1, 512)
                budget = rng.randint(0, bs * verify_num_draft_tokens)
                num_reqs_per_rank.append(bs)
                contributions.append(
                    local_verify_tier_num_tokens(
                        bs=bs,
                        verify_token_budget=budget,
                        verify_num_draft_tokens=verify_num_draft_tokens,
                        min_verify_len=min_verify_len,
                    )
                )
            tier_num_tokens = dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=contributions
            )
            global_num_reqs = max(num_reqs_per_rank)
            if tier_num_tokens is None:
                self.assertEqual(global_num_reqs, 0)
                continue

            self.assertGreaterEqual(tier_num_tokens, global_num_reqs * effective_min)
            self.assertLessEqual(
                tier_num_tokens, global_num_reqs * verify_num_draft_tokens
            )

            busy_floor = min(tier_num_tokens, global_num_reqs * verify_num_draft_tokens)
            self.assertEqual(busy_floor, tier_num_tokens)

            idle_lens_total = global_num_reqs
            idle_bucket_input = max(idle_lens_total, tier_num_tokens)
            self.assertEqual(idle_bucket_input, tier_num_tokens)


def test_non_spark_idle_without_global_token_metadata_keeps_existing_skip():
    from unittest.mock import Mock

    proposer = DraftBlockProposer.__new__(DraftBlockProposer)
    proposer._dp_moe_sync = True
    proposer._attention_tp_group = None
    proposer.draft_model_runner = SimpleNamespace(forward=Mock())
    proposer.run_idle_participation(SimpleNamespace(global_num_tokens=None))
    proposer.draft_model_runner.forward.assert_not_called()


@pytest.mark.parametrize("global_num_tokens", [None, [0]])
def test_spark_tp_idle_uses_padding_block_without_mutating_scheduler_state(
    global_num_tokens,
):
    from unittest.mock import Mock

    proposer = DraftBlockProposer.__new__(DraftBlockProposer)
    proposer._dp_moe_sync = True
    proposer._attention_tp_group = object()
    proposer._prep_cache_enabled = False
    proposer._draft_block_spec_info = SimpleNamespace()
    proposer.gamma, proposer._mask_token_id = 8, 42
    proposer._fill_num_token_non_padded = Mock()
    proposer._fill_dp_moe_sync_metadata = Mock()
    proposer.draft_model_runner = SimpleNamespace(
        device="cpu",
        forward=Mock(),
    )
    proposer.draft_model = SimpleNamespace(
        get_input_embeddings=lambda: lambda ids: torch.zeros((len(ids), 4)),
    )
    batch = SimpleNamespace(global_num_tokens=global_num_tokens)
    with patch(
        "sglang.srt.speculative.dspark_components.dspark_draft.ForwardBatch",
        side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
    ):
        proposer.run_idle_participation(batch)
    proposer.draft_model_runner.forward.assert_called_once()
    block = proposer.draft_model_runner.forward.call_args.args[0]
    assert block.forward_mode.is_target_verify()
    assert block.input_ids.tolist() == [42] * 8
    assert block.out_cache_loc.tolist() == [0] * 8
    assert block.seq_lens.tolist() == [0]
    assert block.seq_lens_cpu.tolist() == [8]
    assert batch.global_num_tokens == global_num_tokens
    metadata = proposer._fill_dp_moe_sync_metadata.call_args.args[1]
    assert metadata.global_num_tokens == [1]


def test_spark_tp_context_preserves_ep_construction_and_scopes_forward():
    from contextlib import contextmanager, nullcontext

    from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2

    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker._draft_dp_context_enabled = False
    worker._spark_attention_tp = True
    group, calls = object(), []

    @contextmanager
    def attention_context(value):
        assert value is group
        calls.append("enter")
        yield
        calls.append("exit")

    module = "sglang.srt.speculative.dspark_components.dspark_worker_v2."
    with (
        patch(
            module + "get_parallel", return_value=SimpleNamespace(attn_tp_group=group)
        ),
        patch(module + "draft_tp_context", side_effect=attention_context),
        patch(module + "speculative_moe_backend_context", side_effect=nullcontext),
        patch(module + "speculative_moe_a2a_backend_context", side_effect=nullcontext),
    ):
        # EP32 constructors see the original full world.
        with worker._draft_context():
            assert calls == []
        worker._draft_worker = object()
        # Execution tensor collectives see only the attention replica.
        with worker._draft_context():
            assert calls == ["enter"]
        assert calls == ["enter", "exit"]


def test_spark_tp_capture_gathers_only_sharded_auxiliary_rows():
    from unittest.mock import Mock

    from sglang.srt.models.sparkv1_dspark_cp import _gather_tp_aux_hidden_states

    full = torch.arange(48, dtype=torch.float32).view(16, 3)
    local = full[:2].clone()
    group = SimpleNamespace(all_gather=Mock(return_value=full.clone()))
    already_full = full + 100
    aligned = _gather_tp_aux_hidden_states(full, [local, already_full], group, 8)
    assert torch.equal(aligned[0], full)
    assert aligned[1] is already_full
    group.all_gather.assert_called_once_with(local, dim=0)
    assert torch.cat(aligned, dim=-1).shape == (16, 6)
    group.all_gather.reset_mock()
    empty = torch.empty((0, 3))
    assert _gather_tp_aux_hidden_states(empty, [empty], group, 8)[0] is empty
    group.all_gather.assert_not_called()


def test_spark_tp_capture_rejects_ambiguous_token_layout():
    import pytest

    from sglang.srt.models.sparkv1_dspark_cp import _gather_tp_aux_hidden_states

    with pytest.raises(RuntimeError, match="cannot match"):
        _gather_tp_aux_hidden_states(torch.empty(16, 3), [torch.empty(3, 3)], None, 8)


def test_spark_tp_logits_reuses_outer_attention_context():
    proposer = DraftBlockProposer.__new__(DraftBlockProposer)
    proposer._dp_moe_sync = True
    proposer._attention_tp_group = object()
    with patch(
        "sglang.srt.speculative.dspark_components.dspark_draft.draft_tp_context",
        side_effect=AssertionError("must not nest TP patches"),
    ):
        with proposer._base_logits_context():
            pass


@pytest.mark.parametrize("sampling,fast", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("has_tp_group", [False, True])
def test_draft_sampling_wraps_only_attention_tp_replicas(sampling, fast, has_tp_group):
    from unittest.mock import Mock

    from sglang.srt.speculative.dspark_components import dspark_draft as draft
    from sglang.srt.speculative.dspark_components import spark_attention_tp

    logits = torch.tensor(
        [[[1.0, 3.0, 2.0], [4.0, 1.0, 2.0]], [[3.0, 2.0, 1.0], [1.0, 2.0, 4.0]]]
    )

    def sample_block(base, *, first_prev_tokens, hidden_states, sampler):
        return torch.stack([sampler(base[:, i], i) for i in range(2)], dim=1), base

    def broadcast(tokens, src):
        assert src == 0
        tokens.fill_(2)

    group = (
        SimpleNamespace(broadcast=Mock(side_effect=broadcast)) if has_tp_group else None
    )
    info = (
        SimpleNamespace(
            is_all_greedy=False, top_ks=torch.full((2,), 3), temperatures=torch.ones(2)
        )
        if sampling
        else None
    )
    kwargs = (
        {"wraps": spark_attention_tp.broadcast_attention_tp}
        if has_tp_group
        else {"side_effect": AssertionError("TP1 must not call the broadcast helper")}
    )
    with (
        patch.object(draft.envs.SGLANG_DSPARK_FAST_SAMPLING, "get", return_value=fast),
        patch.object(spark_attention_tp, "broadcast_attention_tp", **kwargs) as sync,
        patch.object(
            draft.SampleStepTokens,
            "execute",
            side_effect=lambda **kw: torch.tensor([1, 0]),
        ) as kernel,
    ):
        result = draft.sample_draft_block(
            base_logits=logits,
            anchor_tokens=torch.zeros(2, dtype=torch.long),
            draft_hidden=torch.zeros(2, 2, 4),
            sampling_info=info,
            markov_head=SimpleNamespace(sample_block=sample_block),
            device=torch.device("cpu"),
            attention_tp_group=group,
        )
    assert sync.call_count == (2 if has_tp_group else 0)
    if has_tp_group:
        assert torch.equal(result.draft_tokens, torch.full((2, 2), 2))
    elif not sampling:
        assert torch.equal(result.draft_tokens, logits.argmax(dim=-1))
    elif fast:
        assert torch.equal(result.draft_tokens, torch.tensor([[1, 1], [0, 0]]))
    assert kernel.call_count == (2 if sampling and fast else 0)


@pytest.mark.parametrize("has_tp_group", [False, True])
def test_verify_sync_runs_only_with_attention_tp_group(has_tp_group):
    from unittest.mock import Mock

    from sglang.srt.speculative.dspark_components import dspark_verify as verify
    from sglang.srt.speculative.dspark_components import spark_attention_tp

    correct, bonus, trim = torch.tensor([2]), torch.tensor([7]), torch.tensor([2])
    final = SimpleNamespace(
        cap_trim_lens=trim, commit_lens=correct, new_seq_lens=correct
    )
    executor = verify.TargetVerifyExecutor.__new__(verify.TargetVerifyExecutor)
    executor._simulate_acc_len = 0
    executor._attention_tp_group = (
        SimpleNamespace(broadcast=Mock()) if has_tp_group else None
    )
    executor.gamma, executor.verify_num_draft_tokens = 8, 9
    kwargs = (
        {"wraps": spark_attention_tp.broadcast_attention_tp}
        if has_tp_group
        else {"side_effect": AssertionError("TP1 must not call the broadcast helper")}
    )
    with (
        patch.object(
            verify, "accept_draft_tokens", return_value=(correct, bonus, trim)
        ),
        patch.object(
            verify.FinalizeAcceptLens, "execute", return_value=final
        ) as finalize,
        patch.object(verify.BuildOutTokens, "execute", return_value=bonus),
        patch.object(spark_attention_tp, "broadcast_attention_tp", **kwargs) as sync,
    ):
        result = executor.accept_and_finalize(
            folded_accept=False,
            bs=1,
            verify_ids_2d=torch.empty(1, 9, dtype=torch.long),
            target_logits=None,
            draft_block=None,
            sampling_info=None,
            draft_input=None,
            layout=None,
            prefix_lens=correct,
            draft_tokens=bonus,
        )
    assert sync.call_count == (3 if has_tp_group else 0)
    assert finalize.call_args.kwargs["correct_len"] is correct
    assert result.correct_len is correct


def test_spark_tp1_cp1_aux_adapter_preserves_original_fast_return():
    import sys
    from types import ModuleType

    from sglang.srt.models import sparkv1_dspark_cp as adapter

    original_result = object()

    class Model:
        @property
        def pp_group(self):
            raise AssertionError("TP1/CP1 must return before inspecting the output")

        def forward(self, *args, **kwargs):
            return original_result

    module = ModuleType("tokenflow.models.sparkv1.model")
    module.SparkV1Model = Model
    with (
        patch.dict(sys.modules, {module.__name__: module}),
        patch.object(
            adapter,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1, attn_cp_size=1),
        ),
        patch.object(
            adapter.os,
            "getenv",
            side_effect=AssertionError("TP1 must not inspect TP opt-in"),
        ),
    ):
        adapter.install_spark_v1_dspark_cp_adapter()
        assert Model().forward(None, None, object()) is original_result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
