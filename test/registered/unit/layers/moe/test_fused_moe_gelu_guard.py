"""Regression coverage for the HIP legacy gated-GELU path."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_hcu_ci


register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")
register_hcu_ci(est_time=10, suite="stage-b-test-1-hcu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="The guarded GELU path is HIP-only",
)


def test_legacy_gelu_uses_exact_math_only_with_activation_smooth_scale(
    monkeypatch,
):
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe

    dtype = torch.bfloat16
    device = "cuda"
    hidden_states = torch.zeros((2, 4), device=device, dtype=dtype)
    w1 = torch.zeros((1, 8, 4), device=device, dtype=dtype)
    w2 = torch.zeros((1, 4, 4), device=device, dtype=dtype)
    topk_weights = torch.ones((2, 1), device=device, dtype=torch.float32)
    topk_ids = torch.zeros((2, 1), device=device, dtype=torch.int32)
    kernel_metadata = torch.zeros((1,), device=device, dtype=torch.int32)
    preactivation = torch.tensor(
        [
            [10, -10, 1, -1, 2, 3, 4, 5],
            [0.25, -0.5, 2, -3, -2, 1.5, -4, 0.5],
        ],
        device=device,
        dtype=dtype,
    )
    native_gelu = fused_moe.F.gelu

    monkeypatch.setattr(fused_moe, "_is_hip", True)
    monkeypatch.setattr(fused_moe, "_is_cuda", False)
    monkeypatch.setattr(fused_moe, "_is_xpu", False)
    monkeypatch.setattr(fused_moe, "_use_tokenflow_legacy_compat", True)
    monkeypatch.setattr(
        fused_moe,
        "get_exec",
        lambda: SimpleNamespace(
            moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
        ),
    )

    def run_case(activation_smooth_scale):
        invoke_count = 0
        gelu_kernel_calls = []
        exact_gelu_calls = []
        down_inputs = []

        def fake_invoke_fused_moe_kernel(*args, **kwargs):
            nonlocal invoke_count
            invoke_count += 1
            if invoke_count == 1:
                args[3].copy_(preactivation)
            elif invoke_count == 2:
                down_inputs.append(args[0].clone())
                args[3].zero_()
            else:
                raise AssertionError(f"unexpected MoE GEMM call {invoke_count}")

        def fake_gelu_and_mul(x, out, *args, **kwargs):
            gelu_kernel_calls.append((x.clone(), kwargs))
            half = x.shape[-1] // 2
            out.copy_(
                native_gelu(x[..., :half], approximate="tanh") * x[..., half:]
            )

        def spy_exact_gelu(x, approximate="none"):
            exact_gelu_calls.append(approximate)
            return native_gelu(x, approximate=approximate)

        with monkeypatch.context() as case_patch:
            case_patch.setattr(
                fused_moe,
                "invoke_fused_moe_kernel",
                fake_invoke_fused_moe_kernel,
            )
            case_patch.setattr(fused_moe, "gelu_and_mul", fake_gelu_and_mul)
            case_patch.setattr(
                fused_moe,
                "F",
                SimpleNamespace(gelu=spy_exact_gelu),
            )

            fused_moe._fused_moe_kernel_sequence(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                kernel_metadata,
                kernel_metadata,
                kernel_metadata,
                {"BLOCK_SIZE_M": 1},
                None,
                False,
                b1=None,
                b2=None,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                use_int4_w4a8=False,
                use_mxfp4_w4a16=False,
                use_mxfp4_w4a8=False,
                per_channel_quant=False,
                w1_scale=None,
                w2_scale=None,
                w1_zp=None,
                w2_zp=None,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                activation="gelu",
                is_gated=True,
                no_combine=True,
                inplace=False,
                apply_router_weight_on_input=False,
                routed_scaling_factor=1.0,
                gemm1_alpha=None,
                gemm1_limit=None,
                filter_expert=False,
                activation_smooth_scale=activation_smooth_scale,
            )

        assert invoke_count == 2
        assert len(down_inputs) == 1
        return gelu_kernel_calls, exact_gelu_calls, down_inputs[0]

    gelu_kernel_calls, exact_gelu_calls, _ = run_case(None)
    assert len(gelu_kernel_calls) == 1
    assert exact_gelu_calls == []

    smooth_scale = torch.tensor(
        [0.5, 1.0, 2.0, 3.0], device=device, dtype=dtype
    )
    gelu_kernel_calls, exact_gelu_calls, down_input = run_case(smooth_scale)
    assert gelu_kernel_calls == []
    assert exact_gelu_calls == ["none"]

    half = preactivation.shape[-1] // 2
    expected = (
        native_gelu(preactivation[..., :half], approximate="none")
        * preactivation[..., half:]
    )
    expected = expected.clamp(-8.0, 8.0) * smooth_scale.view(1, -1)
    torch.testing.assert_close(down_input, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
