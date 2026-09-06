"""Process-local 0.5.12 Spark adapters for 0.5.18; never patch TokenFlow files.

Normalization/RoPE is deliberately not adapted here. Shared-expert physical
weight remapping is also independent from ordinary gate fusion.
"""
import logging
import inspect
from typing import Callable

import torch

from sglang.srt.utils import get_bool_env_var, is_hcu

logger = logging.getLogger(__name__)


def install_grouped_topk(config) -> bool:
    if not (is_hcu() and get_bool_env_var("SGLANG_USE_LIGHTOP") and
            get_bool_env_var("SGLANG_USE_LIGHTOP_MOE_GATE", "false")):
        return False
    if getattr(config, "calc_denominator_cross_groups", False):
        raise ValueError("LightOp Spark gate cannot normalize across groups")
    from lightop import moe_fuse_gate_per_gourp_topk
    import tokenflow.models.sparkv1.layers.moe as spark_moe

    original = spark_moe.try_spark_v1_npu_grouped_topk
    if getattr(original, "_sglang_0512_gate", False):
        return True

    def gate(logits, correction_bias, top_k, num_pad_experts_per_group,
             routed_scaling_factor):
        supported = (
            logits.ndim == 3 and logits.shape[1:] == (2, 128)
            and top_k == 7 and num_pad_experts_per_group == 1
            and logits.is_cuda and correction_bias.is_cuda
            and logits.device == correction_bias.device
            and correction_bias.numel() == 256
            and (logits.dtype, correction_bias.dtype) in {
                (torch.bfloat16, torch.bfloat16),
                (torch.bfloat16, torch.float32),
                (torch.float32, torch.bfloat16),
                (torch.float32, torch.float32),
            }
        )
        if not supported:
            return original(logits, correction_bias, top_k,
                            num_pad_experts_per_group, routed_scaling_factor)
        # No shared-weight rewriting: preserve independent shared experts.
        return moe_fuse_gate_per_gourp_topk(
            logits.contiguous(), correction_bias.reshape(-1).contiguous(),
            2, top_k, 2, 0, routed_scaling_factor, True,
        )

    gate._sglang_0512_gate = True
    gate._sglang_original = original
    spark_moe.try_spark_v1_npu_grouped_topk = gate
    logger.info("Installed 0512 Spark LightOp per-group gate (no shared remapping)")
    return True


def install_deepep_prepare() -> bool:
    if not (is_hcu() and get_bool_env_var(
            "SGLANG_USE_LIGHTOP_FUSE_DEEPEP_PREP", "false")):
        return False
    from lightop import op
    import tokenflow.models.sparkv1.layers.moe as spark_moe
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        _DeepEPDispatcherImplLowLatency,
        _DeepEPDispatcherImplNormal,
    )
    if not hasattr(op, "fused_deepep_prepare"):
        raise RuntimeError("Requested fused DeepEP preparation is unavailable")
    original_dispatch = _DeepEPDispatcherImplLowLatency.dispatch_a
    original_normal_dispatch = _DeepEPDispatcherImplNormal.dispatch_a
    if getattr(original_dispatch, "_sglang_0512_prepare", False):
        return True
    cls = spark_moe.SparkV1MoE
    original_forward: Callable = cls.forward_deepep
    original_smooth: Callable = spark_moe.apply_input_smooth
    real_torch = spark_moe.torch
    state = {"active": False, "clamp": {}, "prepare": {}}

    def supported(x):
        return (isinstance(x, torch.Tensor) and x.is_cuda
                and x.dtype == torch.bfloat16 and x.ndim == 2
                and x.shape[0] > 0 and x.shape[1] == 6144
                and x.is_contiguous())

    class TorchProxy:
        def __getattr__(self, name):
            return getattr(real_torch, name)

        def clamp(self, x, min=None, max=None, *, out=None):
            if (state["active"] and supported(x) and out is None
                    and isinstance(min, (int, float))
                    and isinstance(max, (int, float)) and min == -max
                    and max >= 0):
                state["clamp"][id(x)] = (float(min), float(max))
                return x
            return real_torch.clamp(x, min=min, max=max, out=out)

    def smooth(module, hidden_states, *, parameter_name="input_smooth_scale"):
        key = id(hidden_states)
        limits = state["clamp"].pop(key, None)
        scale = getattr(module, parameter_name, None)
        if (limits is not None and supported(hidden_states)
                and isinstance(scale, torch.Tensor) and scale.is_cuda
                and scale.device == hidden_states.device
                and scale.dtype == torch.float32 and scale.is_contiguous()
                and scale.numel() == hidden_states.shape[1]):
            state["prepare"][key] = (limits, scale)
            return hidden_states
        if limits is not None:
            hidden_states = real_torch.clamp(hidden_states, *limits)
        return original_smooth(module, hidden_states, parameter_name=parameter_name)

    def forward(self, hidden_states, *args, **kwargs):
        if state["active"]:
            raise RuntimeError("Nested Spark DeepEP preparation is unsupported")
        state["active"] = True
        try:
            result = original_forward(self, hidden_states, *args, **kwargs)
            # Do not silently return an unclamped result if a newer plugin
            # bypasses the intercepted smooth or dispatch entry points.
            if state["clamp"] or state["prepare"]:
                raise RuntimeError("Unconsumed Spark fused-preparation state")
            return result
        finally:
            state["active"] = False
            state["clamp"].clear()
            state["prepare"].clear()

    def prepare(hidden_states, topk_output):
        pending = state["prepare"].pop(id(hidden_states), None)
        if pending is not None:
            limits, scale = pending
            ids = getattr(topk_output, "topk_ids", None)
            if (supported(hidden_states) and isinstance(ids, torch.Tensor)
                    and ids.device == hidden_states.device
                    and ids.dtype == torch.int32 and ids.ndim == 2
                    and ids.shape[0] == hidden_states.shape[0]
                    and 0 < ids.shape[1] <= 32 and ids.is_contiguous()
                    and hasattr(topk_output, "_replace")):
                hidden_states, ids = op.fused_deepep_prepare(
                    hidden_states, scale, ids, *limits)
                topk_output = topk_output._replace(topk_ids=ids)
            else:
                hidden_states = real_torch.clamp(hidden_states, *limits)
                hidden_states = hidden_states * scale.to(hidden_states.dtype)
        return hidden_states, topk_output

    def dispatch(self, hidden_states, topk_output):
        hidden_states, topk_output = prepare(hidden_states, topk_output)
        return original_dispatch(self, hidden_states, topk_output)

    def normal_dispatch(self, hidden_states, topk_output):
        hidden_states, topk_output = prepare(hidden_states, topk_output)
        return original_normal_dispatch(self, hidden_states, topk_output)

    dispatch._sglang_0512_prepare = True
    dispatch._sglang_original = original_dispatch
    normal_dispatch._sglang_0512_prepare = True
    normal_dispatch._sglang_original = original_normal_dispatch
    forward._sglang_original = original_forward
    smooth._sglang_original = original_smooth
    spark_moe.torch = TorchProxy()
    spark_moe.apply_input_smooth = smooth
    cls.forward_deepep = forward
    _DeepEPDispatcherImplLowLatency.dispatch_a = dispatch
    _DeepEPDispatcherImplNormal.dispatch_a = normal_dispatch
    logger.info("Installed 0512 Spark clamp/smooth/ID-widening DeepEP preparation")
    return True


def install_spark_v1_0512_adapters(config):
    return {"grouped_topk": install_grouped_topk(config),
            "deepep_prepare": install_deepep_prepare()}
