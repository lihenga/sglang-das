from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_mxfp8: bool = False
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    use_int4_w4a8: bool = False
    use_mxfp4_w4a16: bool = False
    use_mxfp4_w4a8: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    # w13 rows were permuted to interleave gate/up at load, so the activation
    # must be applied by the fused up-GEMM epilogue (see fused_moe_kernel).
    fuse_swiglu_interleaved: bool = False
    activation_smooth_scale: Optional[torch.Tensor] = None


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:
        if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
            from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
                fused_experts_mxfp8,
            )

            out = fused_experts_mxfp8(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                runner_input.topk_weights,
                runner_input.topk_ids,
                quant_info.w13_scale,
                quant_info.w2_scale,
                b1=quant_info.b13,
                b2=quant_info.b2,
                activation=self.config.activation,
                is_gated=self.config.is_gated,
                no_combine=self.config.no_combine,
                inplace=self.config.inplace,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                routed_scaling_factor=self.config.routed_scaling_factor,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_limit=self.config.gemm1_clamp_limit,
                swiglu_limit=self.config.swiglu_limit,
                gate_up_interleaved=self.config.gate_up_interleaved,
            )
            return TritonRunnerOutput(hidden_states=out)

        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
            running_state.get("up_moe_use_tma", False),
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            use_int4_w4a8=quant_info.use_int4_w4a8,
            use_mxfp4_w4a16=quant_info.use_mxfp4_w4a16,
            use_mxfp4_w4a8=quant_info.use_mxfp4_w4a8,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine,
            inplace=self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            routed_scaling_factor=self.config.routed_scaling_factor,
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            swiglu_limit=self.config.swiglu_limit,
            fuse_swiglu_interleaved=quant_info.fuse_swiglu_interleaved,
            activation_smooth_scale=quant_info.activation_smooth_scale,
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
        from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
            fused_experts_mxfp8,
        )

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = fused_experts_mxfp8(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            b1=quant_info.b13,
            b2=quant_info.b2,
            activation=runner_config.activation,
            is_gated=runner_config.is_gated,
            no_combine=runner_config.no_combine,
            inplace=runner_config.inplace,
            apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            gemm1_alpha=runner_config.gemm1_alpha,
            gemm1_limit=runner_config.gemm1_clamp_limit,
            swiglu_limit=runner_config.swiglu_limit,
            gate_up_interleaved=runner_config.gate_up_interleaved,
        )
    else:
        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        # SGLANG_OPT_MOE_QUANT_ONCE: use the caller's pre-quantized activation
        # (per-token-group-128 fp8 q + scales) instead of re-quantizing inside
        # invoke_fused_moe_kernel.
        pre_quant = dispatch_output.hidden_states_pre_quant
        if pre_quant is not None:
            a1_q, a1_scale = pre_quant
        else:
            a1_q, a1_scale = None, quant_info.a13_scale

        output = fused_experts(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_output=dispatch_output.topk_output,
            moe_runner_config=runner_config,
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            use_int4_w4a8=quant_info.use_int4_w4a8,
            use_mxfp4_w4a16=quant_info.use_mxfp4_w4a16,
            use_mxfp4_w4a8=quant_info.use_mxfp4_w4a8,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            a1_q=a1_q,
            fuse_swiglu_interleaved=quant_info.fuse_swiglu_interleaved,
            activation_smooth_scale=quant_info.activation_smooth_scale,
        )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_fused_func("deepep", "triton")
def fused_experts_deepep_to_triton(
    dispatch_output,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    """Run locally received DeepEP routes with the Triton W8A8 MoE."""
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    is_normal = dispatch_output.format.is_deepep_normal()
    if not (is_normal or dispatch_output.format.is_deepep_ll()):
        raise NotImplementedError(
            f"Unsupported DeepEP format: {dispatch_output.format}"
        )
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError("DeepEP + Triton requires BF16 dispatch")
    if dispatch_output.hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "DeepEP + Triton expects BF16 hidden states, got "
            f"{dispatch_output.hidden_states.dtype}"
        )
    if not (quant_info.use_int8_w8a8 and quant_info.per_channel_quant):
        raise NotImplementedError("DeepEP + Triton currently supports W8A8 INT8")

    smooth_scale = getattr(
        runner_config.layer, "routed_down_input_smooth_scale", None
    )
    if runner_config.activation == "gelu" and smooth_scale is not None:
        quant_info = replace(
            quant_info,
            activation_smooth_scale=smooth_scale,
        )

    hidden_states = dispatch_output.hidden_states
    if is_normal:
        local_topk_ids = dispatch_output.topk_ids.to(torch.int32)
        local_topk_weights = dispatch_output.topk_weights
    else:
        num_experts, capacity, hidden_size = hidden_states.shape
        token_ids = torch.arange(
            capacity,
            device=hidden_states.device,
            dtype=dispatch_output.masked_m.dtype,
        )
        expert_ids = torch.arange(
            num_experts, device=hidden_states.device, dtype=torch.int32
        ).unsqueeze(1)
        local_topk_ids = torch.where(
            token_ids.unsqueeze(0) < dispatch_output.masked_m.unsqueeze(1),
            expert_ids,
            -1,
        ).reshape(-1, 1)
        local_topk_weights = torch.ones(
            (num_experts * capacity, 1),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        hidden_states = hidden_states.reshape(-1, hidden_size)

    output = hidden_states
    if output.shape[0] > 0:
        output = fused_experts_none_to_triton(
            StandardDispatchOutput(
                hidden_states=output,
                hidden_states_scale=None,
                topk_output=StandardTopKOutput(
                    local_topk_weights, local_topk_ids, None
                ),
                hidden_states_pre_quant=None,
            ),
            quant_info,
            # Spark's router already folded routed_scaling_factor into weights.
            replace(runner_config, routed_scaling_factor=1.0),
        ).hidden_states

    combine_cls = DeepEPNormalCombineInput if is_normal else DeepEPLLCombineInput
    return combine_cls(
        output.reshape(dispatch_output.hidden_states.shape),
        dispatch_output.topk_ids,
        dispatch_output.topk_weights,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        up_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        use_int4_w4a8=quant_info.use_int4_w4a8,
        use_mxfp4_w4a16=quant_info.use_mxfp4_w4a16,
        use_mxfp4_w4a8=quant_info.use_mxfp4_w4a8,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma
    running_state["up_moe_use_tma"] = up_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )
