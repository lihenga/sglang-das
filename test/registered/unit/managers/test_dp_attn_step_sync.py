"""CPU-only coverage for DP-attention scheduler synchronization.

These tests intentionally mock process-group operations.  They protect the
fixed-width PD Decode StepInfo wire contract, epoch commit semantics, and the
DP-wide CUDA Graph vote used by MTP/DSA IndexShare without requiring GPUs.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpecInputType  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _step_info(*, protocol=None, epoch=7):
    return [
        dp_attn.DP_DECODE_STEP_PROTOCOL_VERSION if protocol is None else protocol,
        dp_attn.DP_DECODE_STEP_BUILD_ID,
        epoch,
        1,  # transfer
        2,  # prealloc
        3,  # retracted
        4,  # running
        0,  # paused
        5,  # pd_elapsed_ms
        0,  # pd_over_budget
    ]


def _mlp_sync_info(step_info):
    return dp_attn.MLPSyncBatchInfo(
        dp_size=1,
        tp_size=1,
        cp_size=1,
        num_tokens=3,
        num_tokens_for_logprob=3,
        can_run_decode_cuda_graph=True,
        can_run_prefill_cuda_graph=False,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=ForwardMode.DECODE.value,
        scheduler_step_info=step_info,
    )


class TestPDDecodeStepInfo(CustomTestCase):

    def test_dp16_tp1_retains_view_without_replica_reductions(self):
        tensor_min, tensor_max = torch.Tensor.min, torch.Tensor.max

        def no_replica_min(tensor, *args, **kwargs):
            self.assertNotEqual(kwargs.get("dim", args[0] if args else None), 1)
            return tensor_min(tensor, *args, **kwargs)

        def no_replica_max(tensor, *args, **kwargs):
            self.assertNotEqual(kwargs.get("dim", args[0] if args else None), 1)
            return tensor_max(tensor, *args, **kwargs)

        for has_step_info in (False, True):
            with self.subTest(has_step_info=has_step_info):
                info = _mlp_sync_info(_step_info() if has_step_info else None)
                info.dp_size = 16
                shared = info._get_local_tensor("cpu").repeat(16, 1)
                shared[:, 0] = torch.arange(1, 17)
                storage = {}

                def gather_into(output, local, group=None):
                    output.copy_(shared.flatten())
                    storage["ptr"] = output.untyped_storage().data_ptr()

                if has_step_info:
                    storage["ptr"] = shared.untyped_storage().data_ptr()
                with (
                    patch.object(torch.distributed, "get_world_size", return_value=16),
                    patch.object(torch.distributed, "get_rank", return_value=1),
                    patch.object(
                        torch.distributed,
                        "all_gather_into_tensor",
                        side_effect=gather_into,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.pd_decode_sync.gather_pd_decode_step_info",
                        return_value=shared if has_step_info else None,
                    ),
                    patch.object(
                        dp_attn,
                        "get_tp_group",
                        return_value=SimpleNamespace(
                            active_ranks_cpu=torch.ones(16, dtype=torch.int64)
                        ),
                    ),
                    patch.object(
                        torch.Tensor,
                        "clone",
                        side_effect=AssertionError(
                            "TP1 must not clone replica metadata"
                        ),
                    ),
                    patch.object(torch.Tensor, "min", new=no_replica_min),
                    patch.object(torch.Tensor, "max", new=no_replica_max),
                ):
                    info.all_gather(device="cpu", group=object())
                self.assertEqual(info.global_num_tokens, list(range(1, 17)))
                self.assertEqual(
                    info.tp0_info_cpu.untyped_storage().data_ptr(), storage["ptr"]
                )

    def _gather_tp8dp4(self, mutate=None):
        info = _mlp_sync_info(_step_info())
        info.dp_size, info.tp_size = 4, 8

        def gather(outputs, local, group=None):
            self.assertEqual(len(outputs), 32)
            for rank, output in enumerate(outputs):
                output.copy_(local)
                output[0:2] = rank // 8 + 1
                output[15] = rank  # Rank-local PD latency is allowed to differ.
                if mutate is not None:
                    mutate(rank, output)

        with (
            patch.object(torch.distributed, "get_world_size", return_value=32),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(torch.distributed, "all_gather", side_effect=gather),
            patch.object(
                dp_attn,
                "get_tp_group",
                return_value=SimpleNamespace(
                    active_ranks_cpu=torch.ones(32, dtype=torch.int64)
                ),
            ),
        ):
            info.all_gather(device="cpu", group=object())
        return info

    def test_tp8dp4_counts_each_replica_once_and_votes_all_tp_ranks(self):
        def mutate(rank, output):
            if rank == 7:
                output[2] = 0

        info = self._gather_tp8dp4(mutate)
        self.assertEqual(info.global_num_tokens, [1, 2, 3, 4])
        self.assertFalse(info.can_run_decode_cuda_graph)
        self.assertEqual(info.gathered_scheduler_step_info.shape, (4, 10))
        self.assertEqual(
            info.gathered_scheduler_step_info[:, 8].tolist(), [7, 15, 23, 31]
        )

    def test_tp8dp4_rejects_nonleader_epoch_divergence(self):
        def mutate(rank, output):
            if rank == 23:
                output[9] += 1

        with self.assertRaisesRegex(RuntimeError, "epoch mismatch"):
            self._gather_tp8dp4(mutate)

    def test_tp8dp4_rejects_replica_queue_or_forward_divergence(self):
        for col in (0, 5, 10, 11, 12, 13):
            with self.subTest(col=col):

                def mutate(rank, output):
                    if rank == 31:
                        output[col] += 1

                with self.assertRaisesRegex(RuntimeError, "replica state mismatch"):
                    self._gather_tp8dp4(mutate)

    def _mock_cpu_all_gather(self, info):
        def copy_local_to_only_rank(outputs, local, group=None):
            self.assertEqual(len(outputs), 1)
            outputs[0].copy_(local)

        active_ranks = SimpleNamespace(
            active_ranks_cpu=torch.ones(1, dtype=torch.int64)
        )
        with (
            patch.object(torch.distributed, "get_world_size", return_value=1),
            patch.object(torch.distributed, "get_rank", return_value=0),
            patch.object(
                torch.distributed,
                "all_gather",
                side_effect=copy_local_to_only_rank,
            ),
            patch.object(dp_attn, "get_tp_group", return_value=active_ranks),
        ):
            info.all_gather(device="cpu", group=object())

    def test_step_info_has_fixed_width_and_survives_cpu_gather(self):
        step_info = _step_info()
        info = _mlp_sync_info(step_info)

        local = info._get_local_tensor(device="cpu")
        self.assertEqual(local.dtype, torch.int64)
        self.assertEqual(local.numel(), 17)
        self.assertEqual(local[-10:].tolist(), step_info)

        self._mock_cpu_all_gather(info)
        self.assertEqual(info.global_num_tokens, [3])
        self.assertEqual(info.gathered_scheduler_step_info.shape, (1, 10))
        self.assertEqual(info.gathered_scheduler_step_info[0].tolist(), step_info)

    def test_rejects_invalid_width_and_protocol(self):
        with self.assertRaisesRegex(RuntimeError, "exactly 10 scheduler fields"):
            _mlp_sync_info(_step_info()[:-1])._get_local_tensor(device="cpu")

        info = _mlp_sync_info(_step_info(protocol=999))
        with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
            self._mock_cpu_all_gather(info)


class TestSchedulerDPAttnAdapter(CustomTestCase):
    def _adapter(self, set_epoch):
        return dp_attn.SchedulerDPAttnAdapter(
            model_runner=object(),
            tp_group=object(),
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            offload_tags=set(),
            ps=SimpleNamespace(attn_tp_size=1, attn_cp_size=1),
            server_args=object(),
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=object(),
            get_require_mlp_sync=lambda: True,
            get_pd_decode_step_context=lambda: ("scheduler-group", 41, _step_info()),
            set_dp_scheduler_epoch=set_epoch,
        )

    def _adapter_patches(self, raw_result=None, raw_side_effect=None):
        return (
            patch.object(
                dp_attn,
                "prepare_mlp_sync_batch_raw",
                return_value=raw_result,
                side_effect=raw_side_effect,
            ),
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(dp_size=16, dwdp_size=1),
            ),
            patch.object(
                dp_attn,
                "get_schedule",
                return_value=SimpleNamespace(disable_overlap_schedule=False),
            ),
            patch.object(dp_attn, "cuda_graph_fully_disabled", return_value=False),
            patch.object(dp_attn, "require_mlp_tp_gather", return_value=True),
        )

    def test_epoch_advances_only_after_successful_sync(self):
        set_epoch = MagicMock()
        adapter = self._adapter(set_epoch)
        expected = object()
        patches = self._adapter_patches(raw_result=expected)
        with patches[0] as raw, patches[1], patches[2], patches[3], patches[4]:
            self.assertIs(adapter.prepare_mlp_sync_batch(object()), expected)

        self.assertEqual(raw.call_args.kwargs["sync_group_override"], "scheduler-group")
        self.assertEqual(raw.call_args.kwargs["scheduler_step_info"], _step_info())
        set_epoch.assert_called_once_with(42)

        set_epoch.reset_mock()
        patches = self._adapter_patches(raw_side_effect=RuntimeError("sync failed"))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                adapter.prepare_mlp_sync_batch(object())
        set_epoch.assert_not_called()


class TestMTPDSAGraphVote(CustomTestCase):
    def test_missing_dsa_seed_disables_local_decode_graph_vote(self):
        fake_tbo = MagicMock()
        fake_tbo.prepare_all_gather.return_value = (
            True,
            ForwardMode.DECODE.value,
        )
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    index_share_for_mtp_iteration=True,
                    index_topk=2048,
                )
            )
        )
        spec_info = SimpleNamespace(
            spec_input_type=SpecInputType.EAGLE_DRAFT,
            future_indices=torch.tensor([0]),
            future_dsa_topk_indices_available=False,
            dsa_topk_indices=None,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=spec_info,
            batch_size=lambda: 1,
        )
        tp_group = SimpleNamespace(cpu_group=object())

        with (
            patch.object(
                dp_attn.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER,
                "get",
                return_value=False,
            ),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(
                dp_attn,
                "get_spec",
                return_value=SimpleNamespace(speculative_eagle_topk=1),
            ),
            patch.object(
                dp_attn,
                "TboDPAttentionPreparer",
                return_value=fake_tbo,
            ),
        ):
            result = dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=model_runner,
                dp_size=1,
                attn_tp_size=1,
                attn_cp_size=1,
                tp_group=tp_group,
                get_idle_batch=MagicMock(),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=False,
                offload_tags={"cpu"},
            )

        self.assertIs(result, batch)
        self.assertFalse(batch.can_run_dp_cuda_graph)


if __name__ == "__main__":
    unittest.main()
