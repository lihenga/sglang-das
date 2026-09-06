from __future__ import annotations

import inspect
import logging
from typing import Callable

import torch

from sglang.srt.utils import get_bool_env_var, is_hcu

logger = logging.getLogger(__name__)


def install_spark_v1_int8_deepgemm_adapter() -> bool:
    """Install Spark's exact GELU/clamp/smooth quant between INT8 GEMMs."""
    if (
        not is_hcu()
        or not get_bool_env_var("SGLANG_USE_LIGHTOP")
        or not get_bool_env_var(
            "SGLANG_USE_LIGHTOP_FUSE_GELU_CLAMP_SMOOTH_QUANT", "false"
        )
    ):
        return False

    try:
        from lightop import fuse_silu_mul_clamp_quant, fuse_silu_mul_clamp_quant_ep
        from tokenflow.models.sparkv1.ops.activation import (
            current_spark_v1_gelu_clamp,
        )

        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    except ImportError as exc:
        raise RuntimeError(
            "Spark INT8 DeepGEMM was explicitly enabled, but its LightOp/"
            "TokenFlow adapter cannot be imported"
        ) from exc

    if "smooth_scale" not in inspect.signature(
        fuse_silu_mul_clamp_quant_ep
    ).parameters:
        raise RuntimeError(
            "Spark INT8 DeepGEMM requires a LightOp "
            "fuse_silu_mul_clamp_quant_ep with smooth_scale support"
        )

    original: Callable = DeepEPMoE.forward_groupgemm_w8a8_marlin_masked
    # The normal-mode runner retains native scatter, contiguous GEMMs and
    # gather. Replace only its activation/quantization boundary for Spark.
    # No process-global activation interception or communication changes.
    if "smooth_scale" not in inspect.signature(fuse_silu_mul_clamp_quant).parameters:
        raise RuntimeError("Spark normal-mode GEMM requires smooth-aware GELU quant")
    if not hasattr(DeepEPMoE, "_spark_v1_contiguous_gateup_quant"):
        normal_logged = False

        def contiguous_gateup_quant(self, gateup_output):
            nonlocal normal_logged
            limit = current_spark_v1_gelu_clamp()
            smooth = getattr(self, "routed_down_input_smooth_scale", None)
            if not (
                isinstance(limit, (float, int)) and limit > 0
                and isinstance(smooth, torch.Tensor)
                and smooth.ndim == 1 and smooth.dtype == torch.float32
                and smooth.is_cuda and smooth.is_contiguous()
                and gateup_output.ndim == 2
                and gateup_output.dtype == torch.bfloat16
                and gateup_output.device == smooth.device
                and gateup_output.shape[1] == 2 * smooth.numel()
                and gateup_output.is_contiguous()
            ):
                raise RuntimeError("Invalid Spark normal-mode GELU/clamp/smooth boundary")
            q, scale = fuse_silu_mul_clamp_quant(
                gateup_output, float(limit), use_gelu=True, smooth_scale=smooth
            )
            if not normal_logged:
                logger.info(
                    "Spark normal-mode MoE active: INT8 DeepEP -> contiguous "
                    "DeepGEMM -> LightOp GELU/mul/clamp/smooth/quant -> "
                    "DeepGEMM; native scatter/gather/combine"
                )
                normal_logged = True
            return q, scale.reshape(-1, 1)

        DeepEPMoE._spark_v1_contiguous_gateup_quant = contiguous_gateup_quant
    normal_original = DeepEPMoE.forward_groupgemm_w8a8_marlin_contiguous
    if not getattr(normal_original, "_sglang_spark_contiguous_adapter", False):
        # TokenFlow's AROUND hook replaces the full normal-mode implementation
        # and omits shuffle_unique when calling the installed DeepGEMM wheel.
        # Its default mode=1 expects W6, unlike the mode=0 PD-packed weights.
        # Keep the 0518 native implementation, which explicitly uses the same
        # shuffle selection as its weight loader, plus the local quant hook.
        normal_base = inspect.unwrap(normal_original)
        code = getattr(normal_base, "__code__", None)
        if code is None or "_spark_v1_contiguous_gateup_quant" not in code.co_consts:
            raise RuntimeError("Spark P adapter requires the native contiguous quant hook")

        def forward_contiguous(self, dispatch_output):
            return normal_base(self, dispatch_output)

        forward_contiguous._sglang_spark_contiguous_adapter = True
        forward_contiguous._sglang_original = normal_original
        DeepEPMoE.forward_groupgemm_w8a8_marlin_contiguous = forward_contiguous
        logger.info("Spark normal-mode runner uses native PD weight-layout selection and local GELU fusion")
    if getattr(original, "_sglang_spark_int8_deepgemm_adapter", False):
        return True

    state = {"logged": False}

    def spark_forward_groupgemm_w8a8_marlin_masked(self, dispatch_output):
        limit = current_spark_v1_gelu_clamp()
        smooth_scale = getattr(self, "routed_down_input_smooth_scale", None)
        (
            hidden_states,
            hidden_states_scale,
            _topk_ids,
            _topk_weights,
            masked_m,
            expected_m,
        ) = dispatch_output

        invalid = []
        if limit is None or float(limit) <= 0.0:
            invalid.append("missing GELU clamp")
        if not isinstance(smooth_scale, torch.Tensor):
            invalid.append("missing routed-down smooth scale")
        elif (
            smooth_scale.dim() != 1
            or smooth_scale.dtype != torch.float32
            or not smooth_scale.is_cuda
            or not smooth_scale.is_contiguous()
        ):
            invalid.append("invalid routed-down smooth scale")
        if self.quant_method is None:
            invalid.append("missing slimquant_marlin quant method")
        if not isinstance(hidden_states, torch.Tensor):
            invalid.append("missing DeepEP hidden states")
        elif hidden_states.dim() != 3 or hidden_states.dtype != torch.int8:
            invalid.append("DeepEP payload is not rank-3 INT8")
        if not isinstance(hidden_states_scale, torch.Tensor):
            invalid.append("missing DeepEP token scale")
        elif (
            hidden_states_scale.dtype != torch.float32
            or hidden_states_scale.shape[:2] != hidden_states.shape[:2]
            or hidden_states_scale.shape[-1] != 1
        ):
            invalid.append("DeepEP token scale is not FP32 [..., 1]")
        if not isinstance(masked_m, torch.Tensor):
            invalid.append("missing grouped-GEMM masked_m")
        elif masked_m.dim() != 1 or masked_m.dtype != torch.int32:
            invalid.append("grouped-GEMM masked_m is not rank-1 INT32")
        if invalid:
            raise RuntimeError(
                "Refusing to fall back from the validated Spark INT8 DeepEP -> "
                "DeepGEMM path: " + "; ".join(invalid)
            )

        num_groups, max_tokens, _ = hidden_states.shape
        expected_m = min(max_tokens, expected_m)
        # The HCU PD masked-Marlin implementation uses expected_m to bound
        # its launched M tiles, not only to choose a tuning configuration.
        # The dispatch average can underestimate a hot expert and leave
        # valid rows unwritten. Use the static receive-buffer capacity as
        # the launch bound; masked_m still selects the real rows on device.
        # No host sync, DP metadata changes or graph-address changes.
        gemm_m_bound = max_tokens
        w13_scales = self.w13_weight_scale
        w2_scales = self.w2_weight_scale
        gateup_hidden = w13_scales.size(1)
        if (
            masked_m.numel() != num_groups
            or gateup_hidden != 2 * smooth_scale.numel()
        ):
            raise RuntimeError(
                "Refusing Spark INT8 DeepGEMM with inconsistent expert/token "
                "or routed-down smooth-scale shapes"
            )

        gateup_output = torch.empty(
            (num_groups, max_tokens, gateup_hidden),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
        torch.ops.sglang.m_grouped_i8_gemm_nt_masked(
            hidden_states,
            hidden_states_scale,
            self.w13_weight,
            w13_scales,
            gateup_output,
            masked_m,
            gemm_m_bound,
        )

        down_input, down_input_scale = fuse_silu_mul_clamp_quant_ep(
            gateup_output,
            float(limit),
            masked_m,
            expect_m=expected_m,
            use_gelu=True,
            smooth_scale=smooth_scale,
        )
        del gateup_output

        down_output = torch.empty(
            (num_groups, max_tokens, w2_scales.size(1)),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
        torch.ops.sglang.m_grouped_i8_gemm_nt_masked(
            down_input,
            down_input_scale,
            self.w2_weight,
            w2_scales,
            down_output,
            masked_m,
            gemm_m_bound,
        )

        if not state["logged"]:
            logger.info(
                "Spark low-latency MoE uses INT8 DeepEP -> DeepGEMM "
                "with LightOp GELU/clamp/smooth quant; "
                "masked GEMM launch bound=%d, dispatch average=%d",
                gemm_m_bound, expected_m,
            )
            state["logged"] = True
        return down_output

    setattr(
        spark_forward_groupgemm_w8a8_marlin_masked,
        "_sglang_spark_int8_deepgemm_adapter",
        True,
    )
    setattr(spark_forward_groupgemm_w8a8_marlin_masked, "_sglang_original", original)
    DeepEPMoE.forward_groupgemm_w8a8_marlin_masked = (
        spark_forward_groupgemm_w8a8_marlin_masked
    )
    logger.info("Installed Spark INT8 DeepEP/DeepGEMM adapter")
    return True
