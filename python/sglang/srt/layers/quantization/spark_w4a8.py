"""Opt-in Spark CT W4A8 bridge for the installed HCU SGLang runtime.

CT nibble conversion follows HYGON-AI/sglang-das d341af05e2ce.
Dense layers retain their original compressed-tensors W8A8 schemes.
Checkpoint files are never modified; only locally owned experts are converted.
"""
import logging
import os
import re

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import (
    SlimQuantW4A8Int8MarlinConfig,
    SlimQuantW4A8Int8MarlinMoEMethod,
)

logger = logging.getLogger(__name__)
_EXPERT = re.compile(r"^(.*\.mlp\.routed_experts)\.(gate_up_proj|down_proj)\.(weight_packed|weight_scale|weight_shape)$")


def ct_to_signed_high_first(packed):
    """[N,K/8] int32 offset-8 CT -> [N,K/2] signed INT4 byte packing."""
    if packed.dtype != torch.int32 or not packed.is_contiguous():
        raise ValueError("Expected contiguous INT32 CT expert weights")
    u = packed.view(torch.uint8)
    return (((u << 4) | (u >> 4)) ^ 0x88).view(torch.int8)


def convert_spark_experts(weights):
    from sglang.srt.distributed import (
        get_moe_expert_parallel_rank, get_moe_expert_parallel_world_size,
    )
    rank, world = get_moe_expert_parallel_rank(), get_moe_expert_parallel_world_size()
    converted = 0
    for name, tensor in weights:
        match = _EXPERT.match(name)
        if match is None:
            yield name, tensor
            continue
        prefix, projection, kind = match.groups()
        if tensor.ndim not in (2, 3) or tensor.shape[0] != 256 or 256 % world:
            raise ValueError(f"Unexpected fused Spark expert tensor: {name} {tensor.shape}")
        if kind == "weight_shape":
            continue
        local = 256 // world
        for expert in range(rank * local, (rank + 1) * local):
            key = f"{prefix}.{expert}.{projection}"
            if kind == "weight_packed":
                value = ct_to_signed_high_first(tensor[expert].contiguous())
                yield key + ".weight", value
                # Export recipe excludes routed experts from flex_smooth_quant.
                # TokenFlow expects explicit per-expert smooth metadata.
                yield key + ".input_smooth_scale", torch.ones(
                    tensor.shape[-1] * 8, dtype=torch.float32, device=tensor.device
                )
                converted += 1
            else:
                # Legacy SlimQuant stores scale/16; HIPC post-load restores x16.
                yield key + ".weight_scale", tensor[expert].float() / 16.0
    logger.info("Spark W4A8: converted %d local packed expert matrices, EP rank %d/%d", converted, rank, world)


class SparkW4A8Config(CompressedTensorsConfig, SlimQuantW4A8Int8MarlinConfig):
    """CT linear methods plus the existing HCU W4A8 DeepEP expert runner."""
    spark_w4a8 = True

    def get_quant_method(self, layer, prefix):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        if isinstance(layer, FusedMoE):
            if self.quant_format != "mixed-precision":
                raise ValueError("Spark adapter requires the audited mixed-precision checkpoint")
            group = self.target_scheme_map.get("FusedMoE")
            w, a = group["weights"], group["input_activations"]
            if not (w.num_bits == 4 and w.type == "int" and w.strategy == "channel"
                    and w.symmetric and not w.dynamic and a.num_bits == 8
                    and a.type == "int" and a.strategy == "token" and a.dynamic
                    and a.symmetric and group.get("format") == "pack-quantized"):
                raise ValueError("Unsupported Spark expert quantization scheme")
            if layer.moe_tp_size != 1 or layer.num_fused_shared_experts:
                raise ValueError("Spark W4A8 requires TP=EP and independent shared experts")
            install_runners()
            return SlimQuantW4A8Int8MarlinMoEMethod(self)
        return CompressedTensorsConfig.get_quant_method(self, layer, prefix)


def _gelu_quant(layer, x, masked_m=None, expected_m=None):
    from lightop import fuse_silu_mul_clamp_quant, fuse_silu_mul_clamp_quant_ep
    from tokenflow.models.sparkv1.ops.activation import current_spark_v1_gelu_clamp
    limit = current_spark_v1_gelu_clamp()
    if limit is None or float(limit) <= 0:
        raise RuntimeError("Spark GELU clamp context is missing")
    # Routed experts have no SmoothQuant in this export. Ones preserve the
    # existing Spark GELU/mul/clamp arithmetic without applying W8 checkpoint scales.
    smooth = getattr(layer, "_spark_w4a8_identity_smooth", None)
    if smooth is None:
        smooth = torch.ones(x.shape[-1] // 2, device=x.device, dtype=torch.float32)
        layer.register_buffer("_spark_w4a8_identity_smooth", smooth, persistent=False)
    if masked_m is None:
        q, scale = fuse_silu_mul_clamp_quant(x, float(limit), use_gelu=True, smooth_scale=smooth)
        return q, scale.reshape(-1, 1)
    return fuse_silu_mul_clamp_quant_ep(
        x, float(limit), masked_m, expect_m=expected_m,
        use_gelu=True, smooth_scale=smooth,
    )


@triton.jit
def _layout_copy(src, dst, starts, counts, H: tl.constexpr, CAP: tl.constexpr,
                 TO_MASKED: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // H, offsets % H
    start, count = tl.load(starts + e), tl.load(counts + e)
    flat = (start + row) * H + col
    masked = e * CAP * H + offsets
    valid = (row < CAP) & (row < count)
    if TO_MASKED:
        value = tl.load(src + flat, valid, 0)
        tl.store(dst + masked, value, row < CAP)
    else:
        value = tl.load(src + masked, valid, 0)
        tl.store(dst + flat, value, valid)


def to_masked(x, scale, counts):
    capacity = triton.cdiv(max(counts), 256) * 256
    starts = [0]
    for count in counts[:-1]:starts.append(starts[-1] + count)
    starts_gpu = torch.tensor(starts, dtype=torch.int32, device=x.device)
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device=x.device)
    masked = torch.empty((len(counts), capacity, x.shape[1]), dtype=x.dtype, device=x.device)
    masked_scale = torch.empty((len(counts), capacity, 1), dtype=scale.dtype, device=x.device)
    grid = (triton.cdiv(capacity * x.shape[1], 1024), len(counts))
    _layout_copy[grid](x, masked, starts_gpu, counts_gpu, x.shape[1], capacity, True, 1024)
    _layout_copy[(triton.cdiv(capacity, 256), len(counts))](
        scale, masked_scale, starts_gpu, counts_gpu, 1, capacity, True, 256,
    )
    return masked, masked_scale, starts_gpu, counts_gpu


def from_masked(x, starts_gpu, counts_gpu, total):
    groups, capacity, hidden = x.shape
    out = torch.empty((total, hidden), dtype=x.dtype, device=x.device)
    _layout_copy[(triton.cdiv(capacity * hidden, 1024), groups)](
        x, out, starts_gpu, counts_gpu, hidden, capacity, False, 1024,
    )
    return out


def forward_normal(self, dispatch_output):
    import sglang.srt.layers.moe.ep_moe.layer as ep
    x, scale, ids, weights, counts = dispatch_output
    if counts is None or sum(counts) == 0:
        return x.bfloat16()
    if x.dtype != torch.int8 or scale is None:
        raise RuntimeError("Spark W4A8 normal dispatch requires INT8 payload")
    total, hidden = sum(counts), x.shape[1]
    a = torch.zeros((total, hidden), dtype=x.dtype, device=x.device)
    a_scale = torch.ones((total, 1), dtype=torch.float32, device=x.device)
    counts_gpu = torch.tensor(counts, device=x.device, dtype=torch.int32)
    indices, output_indices = ep._ep_scatter_with_optional_lightop(
        x, scale, ids, counts_gpu, a, a_scale, total,
        counts_are_aligned=all(c % 256 == 0 for c in counts),
    )
    masked, masked_scale, starts_gpu, counts_gpu = to_masked(a, a_scale, counts)
    del a, a_scale
    masked_down = forward_masked(self, (
        masked, masked_scale, None, None, counts_gpu, max(counts),
    ))
    del masked, masked_scale
    down = from_masked(masked_down, starts_gpu, counts_gpu, total)
    result = torch.zeros(x.shape, dtype=torch.bfloat16, device=x.device)
    ep._ep_gather_with_optional_lightop(down, ids, weights, output_indices, result)
    return result


def forward_masked(self, dispatch_output):
    x, scale, _ids, _weights, counts, expected_m = dispatch_output
    if x.dtype != torch.int8 or scale is None:
        raise RuntimeError("Spark W4A8 low-latency dispatch requires INT8 payload")
    groups, capacity, hidden = x.shape
    gate_up = torch.empty((groups, capacity, self.w13_weight_scale.shape[1]), dtype=torch.bfloat16, device=x.device)
    op = torch.ops.sglang.m_grouped_w4a8_gemm_nt_masked_hipc
    op(x, scale, self.w13_weight, self.w13_weight_scale, gate_up, counts, capacity)
    q, q_scale = _gelu_quant(self, gate_up, counts, min(capacity, max(expected_m, 128)))
    del gate_up
    result = torch.empty((groups, capacity, hidden), dtype=torch.bfloat16, device=x.device)
    op(q, q_scale, self.w2_weight, self.w2_weight_scale, result, counts, capacity)
    return result


def install_runners():
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    if getattr(DeepEPMoE.forward_deepgemm_w4a8_marlin_contiguous, "_spark_w4a8", False):
        return
    if os.environ.get("SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE", "true").lower() not in ("0", "false"):
        raise RuntimeError("Spark W4A8 HIPC requires SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0")
    forward_normal._spark_w4a8 = True
    DeepEPMoE.forward_deepgemm_w4a8_marlin_contiguous = forward_normal
    DeepEPMoE.forward_groupgemm_w4a8_marlin_masked = forward_masked
    logger.info("Installed Spark W4A8 HIPC runners with GELU/clamp and identity expert smooth scales")
