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


def install_smooth_quant() -> bool:
    if not (is_hcu() and get_bool_env_var("SGLANG_USE_LIGHTOP") and
            get_bool_env_var("SGLANG_USE_LIGHTOP_FUSE_SMOOTH_PER_TOKEN_QUANT", "false")):
        return False
    from lightop.quant import per_token_dynamic_quant_int8
    import tokenflow.models.sparkv1.layers.attention as attention
    import tokenflow.models.sparkv1.layers.input_smooth as input_smooth
    from sglang.srt.layers.linear import ReplicatedLinear, RowParallelLinear
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
    params = inspect.signature(per_token_dynamic_quant_int8).parameters
    if not {'round_after_smooth', 'limit'}.issubset(params):
        raise RuntimeError('LightOp smooth quant lacks BF16 rounding/limit support')

    def quantize(layer, x, limit=-1.):
        scale = getattr(layer, 'input_smooth_scale', None)
        weight = getattr(layer, 'weight', None)
        scheme = getattr(layer, 'scheme', None)
        if not (isinstance(scheme, CompressedTensorsW8A8Int8)
                and isinstance(x, torch.Tensor) and x.ndim == 2
                and x.shape[0] > 0 and x.is_cuda and x.is_contiguous()
                and x.dtype in (torch.bfloat16, torch.float16)
                and isinstance(scale, torch.Tensor) and scale.ndim == 1
                and scale.device == x.device and scale.dtype == torch.float32
                and scale.numel() == x.shape[-1] and scale.is_contiguous()
                and isinstance(weight, torch.Tensor) and weight.dtype == torch.int8):
            return None
        q, s = per_token_dynamic_quant_int8(x, scale, round_after_smooth=True, limit=float(limit))
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        return [q, s]

    row_cls = attention.SparkV1ClampedRowParallelLinear
    rep_cls = input_smooth.InputSmoothReplicatedLinear
    if getattr(row_cls.forward, '_sglang_0512_smooth', False):
        return True
    row_original, rep_original = row_cls.forward, rep_cls.forward

    def row(self, input_, skip_all_reduce=False, forward_batch=None, **kwargs):
        limit = getattr(self, 'clamp_value', getattr(self, 'limit', None))
        if kwargs or not self.input_is_parallel or not isinstance(limit, (int, float)) or limit < 0:
            return row_original(self, input_, skip_all_reduce=skip_all_reduce, forward_batch=forward_batch, **kwargs)
        quant = quantize(self, input_, limit)
        if quant is None:
            return row_original(self, input_, skip_all_reduce=skip_all_reduce, forward_batch=forward_batch)
        return RowParallelLinear.forward(self, input_, skip_all_reduce=skip_all_reduce,
                                         forward_batch=forward_batch, silu_quant_args=quant)

    def replicated(self, x, *args, **kwargs):
        quant = quantize(self, x) if not args and not kwargs else None
        if quant is None:
            return rep_original(self, x, *args, **kwargs)
        return ReplicatedLinear.forward(self, x, quant_args=quant)

    row._sglang_0512_smooth = replicated._sglang_0512_smooth = True
    row._sglang_original, replicated._sglang_original = row_original, rep_original
    row_cls.forward, rep_cls.forward = row, replicated
    logger.info('Installed 0512 local smooth/per-token quant (separate Torch norms)')
    return True


def _local_smooth_scale(layer, width):
    """Match TokenFlow's full-input/local-input selection, without recasting."""
    scale = getattr(layer, "input_smooth_scale", None)
    if not isinstance(scale, torch.Tensor) or scale.ndim != 1:
        return None
    meta = getattr(layer, "_tokenflow_input_smooth_meta", {})
    spec = meta.get("input_smooth_scale") if isinstance(meta, dict) else None
    if spec is None:
        return scale if scale.numel() == width else None
    if not isinstance(spec, (tuple, list)) or len(spec) != 3:
        return None
    full, rank, parts = spec
    if (not all(isinstance(v, int) for v in spec) or parts <= 0
            or full <= 0 or full % parts or not 0 <= rank < parts
            or scale.numel() != full):
        return None
    if width == full:
        return scale
    shard = full // parts
    return scale.narrow(0, rank * shard, shard) if width == shard else None


def install_dense_quant() -> bool:
    """Port both Dense/shared INT8 input chains, leaving collectives to 0518."""
    if not (is_hcu() and get_bool_env_var("SGLANG_USE_LIGHTOP") and
            get_bool_env_var("SGLANG_USE_LIGHTOP_FUSE_DENSE_GELU_CLAMP_SMOOTH_QUANT", "false")):
        return False
    from lightop import fuse_silu_mul_clamp_quant
    from lightop.quant import per_token_dynamic_quant_int8
    import tokenflow.models.sparkv1.layers.moe as spark_moe
    from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
    if "smooth_scale" not in inspect.signature(fuse_silu_mul_clamp_quant).parameters:
        raise RuntimeError("Dense Spark fusion requires smooth-aware GELU quant")
    original = spark_moe.SparkV1DenseMLP.forward
    if getattr(original, "_sglang_0512_dense", False):
        return True
    installed_signature = inspect.signature(original)
    expected = {"hidden_states", "forward_batch", "should_allreduce_fusion",
                "use_reduce_scatter", "gemm_output_zero_allocator"}
    if not expected.issubset(installed_signature.parameters):
        raise RuntimeError(f"Unsupported TokenFlow DenseMLP ABI: {installed_signature}")
    logged = False

    def forward(self, hidden_states, forward_batch=None,
                should_allreduce_fusion=False, use_reduce_scatter=False,
                gemm_output_zero_allocator=None):
        nonlocal logged
        def fallback():
            return original(self, hidden_states, forward_batch=forward_batch,
                            should_allreduce_fusion=should_allreduce_fusion,
                            use_reduce_scatter=use_reduce_scatter,
                            gemm_output_zero_allocator=gemm_output_zero_allocator)

        limit = getattr(self, "clamp_value", None)
        if (gemm_output_zero_allocator is not None
                or not isinstance(hidden_states, torch.Tensor)
                or hidden_states.ndim != 2 or hidden_states.shape[0] == 0
                or not hidden_states.is_cuda or not hidden_states.is_contiguous()
                or hidden_states.dtype not in (torch.float16, torch.bfloat16)
                or not isinstance(limit, (int, float)) or limit <= 0
                or not getattr(self.fc2, "input_is_parallel", False)
                or getattr(self.fc1, "gather_output", False)):
            return fallback()
        width = getattr(self.fc2, "input_size_per_partition", None)
        s1 = _local_smooth_scale(self.fc1, hidden_states.shape[-1])
        s2 = _local_smooth_scale(self.fc2, width)
        for layer, scale in ((self.fc1, s1), (self.fc2, s2)):
            if (not isinstance(getattr(layer, "scheme", None), CompressedTensorsW8A8Int8)
                    or not isinstance(scale, torch.Tensor)
                    or scale.dtype != torch.float32 or scale.device != hidden_states.device
                    or not scale.is_contiguous()
                    or getattr(getattr(layer, "weight", None), "dtype", None) != torch.int8
                    or getattr(layer, "skip_bias_add", False)):
                return fallback()
        if getattr(self.fc1, "output_size_per_partition", None) != 2 * width:
            return fallback()
        q1, a1 = per_token_dynamic_quant_int8(hidden_states, s1,
                                            round_after_smooth=True, limit=float(limit))
        gate_up, _ = ColumnParallelLinear.forward(
            self.fc1, hidden_states, input_quant_args=[q1, a1.reshape(-1, 1)])
        if gate_up.shape != (hidden_states.shape[0], 2 * width):
            raise RuntimeError("Unexpected Dense/shared local gate/up output shape")
        q2, a2 = fuse_silu_mul_clamp_quant(
            gate_up, float(limit), use_gelu=True, smooth_scale=s2)
        # The input view supplies shape/dtype only; the scheme consumes q2/a2.
        # Native RowParallelLinear still owns symmetric memory, bias and
        # should_skip_mlp_all_reduce()/ForwardFlags decisions in 0518.
        output, _ = RowParallelLinear.forward(
            self.fc2, gate_up[:, :width],
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
            forward_batch=forward_batch, silu_quant_args=[q2, a2.reshape(-1, 1)])
        if not logged:
            logger.info("0512 Dense/shared fusion active: clamp/smooth/quant + GELU/mul/clamp/smooth/quant; native 0518 collectives")
            logged = True
        return output

    forward._sglang_0512_dense = True
    forward._sglang_original = original
    spark_moe.SparkV1DenseMLP.forward = forward
    logger.info("Installed 0512 Dense/shared INT8 fusion adapter")
    return True


def install_spark_v1_0512_adapters(config):
    return {"grouped_topk": install_grouped_topk(config),
            "deepep_prepare": install_deepep_prepare(),
            "smooth_quant": install_smooth_quant(),
            "dense_quant": install_dense_quant()}
