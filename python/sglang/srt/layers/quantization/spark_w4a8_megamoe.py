"""Packed W4A8 MegaMoE for Spark CP8/EP8 prefill.

Reuse the SlimQuant loader and native HIPC weight layout. Routing, independent
shared experts and GELU clamp context remain owned by TokenFlow. MegaMoE
returns a fully combined local token slice; it needs no second EP reduction.
"""

import inspect
import logging

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import (
    SlimQuantW4A8Int8MarlinMoEMethod,
)

logger = logging.getLogger(__name__)
_BUFFERS = {}


def _install_native_return_adapter():
    import tokenflow.models.sparkv1.layers.moe as spark_moe

    original = spark_moe.SparkV1MoE.forward
    if getattr(original, "_spark_w4a8_megamoe", False):
        return

    def forward(self, hidden_states, forward_batch=None,
                should_allreduce_fusion=False, use_reduce_scatter=False,
                gemm_output_zero_allocator=None):
        # TokenFlow's ordinary EP path expects rank partials. MegaMoE has
        # already returned the fully combined slice, so skip only that
        # caller's final all-reduce. Do not change layer-communicator flags.
        combined = isinstance(getattr(self.experts, "quant_method", None),
                              SparkW4A8MegaMoEMethod)
        return original(self, hidden_states, forward_batch=forward_batch,
                        should_allreduce_fusion=should_allreduce_fusion or combined,
                        use_reduce_scatter=use_reduce_scatter,
                        gemm_output_zero_allocator=gemm_output_zero_allocator)

    forward._spark_w4a8_megamoe = True
    spark_moe.SparkV1MoE.forward = forward


class SparkW4A8MegaMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config):
        self.quant_config = quant_config
        _install_native_return_adapter()

    def create_weights(self, *args, **kwargs):
        return SlimQuantW4A8Int8MarlinMoEMethod.create_weights(self, *args, **kwargs)

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.utils import is_hcu

        args = get_global_server_args()
        if not (is_hcu() and args.disaggregation_mode == "prefill"
                and args.tp_size == args.ep_size == args.attn_cp_size == 8
                and args.enable_prefill_cp and layer.moe_tp_size == 1
                and layer.num_experts == 256 and layer.top_k == 14
                and layer.hidden_size == 6144
                and layer.intermediate_size_per_partition == 1024
                and moe_runner_config.activation == "gelu"
                and not layer.num_fused_shared_experts and not layer.reduce_results
                and not args.enable_eplb):
            raise ValueError("Spark W4A8 MegaMoE requires CP8/EP8 P with independent shared experts")
        self.capacity = (min(args.chunked_prefill_size, args.max_prefill_tokens) + 7) // 8
        self.capacity = (self.capacity + 63) // 64 * 64
        if self.capacity <= 0:
            raise ValueError("Spark MegaMoE requires a bounded positive prefill chunk")

    def process_weights_after_loading(self, layer):
        import megamoe
        from deepgemm import pack_w4a8_moe_hipc_weight
        from megamoe.w4a8 import w4a8_mega_moe
        from sglang.srt.runtime_context import get_parallel

        if "activation" not in inspect.signature(w4a8_mega_moe).parameters:
            raise RuntimeError("Install the MegaMoE wheel with Spark GELU support")
        for name in ("w13_weight", "w2_weight"):
            weight = getattr(layer, name)
            setattr(layer, name, torch.nn.Parameter(
                pack_w4a8_moe_hipc_weight(weight.data), requires_grad=False))
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name)
            # CT conversion stores the legacy high-nibble scale / 16.
            # MegaMoE consumes signed nibbles directly with the true scale.
            setattr(layer, name, torch.nn.Parameter(
                (scale.data * 16.0).contiguous(), requires_grad=False))
        layer.dispatcher.skip_local_expert_mapping = True
        group = get_parallel().moe_ep_group.device_group
        key = (id(group), layer.w13_weight.device, self.capacity)
        buffer = _BUFFERS.get(key)
        if buffer is None:
            buffer = megamoe.get_symm_buffer_for_mega_moe(
                group, 256, self.capacity, 14, 6144, 1024, quant_mode="w4a8")
            # Reserve the shared route/GEMM workspace before KV pool sizing.
            # Empty ranks participate in both barriers but read no input rows.
            empty = torch.empty((0, 6144), dtype=torch.bfloat16,
                                device=layer.w13_weight.device)
            w4a8_mega_moe(empty, (layer.w13_weight, layer.w13_weight_scale),
                          (layer.w2_weight, layer.w2_weight_scale), buffer,
                          activation_clamp=8.0, activation="gelu")
            torch.cuda.current_stream().synchronize()
            _BUFFERS[key] = buffer
            logger.info("Spark W4A8 MegaMoE initialized: EP8 capacity=%s GELU/clamp, shared workspace before KV sizing", self.capacity)
        self.buffer = buffer

    def _run(self, layer, x, topk_output, pre_quant=None):
        from lightop.quant import per_token_quant_int8
        from megamoe.w4a8 import w4a8_mega_moe
        from tokenflow.models.sparkv1.ops.activation import current_spark_v1_gelu_clamp

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Packed W4A8 MegaMoE supports eager P only")
        if x.shape[0] > self.capacity or x.shape[1] != 6144:
            raise ValueError("Spark MegaMoE input exceeds its CP prefill capacity")
        limit = current_spark_v1_gelu_clamp()
        if limit is None or float(limit) <= 0:
            raise RuntimeError("Spark GELU clamp context is missing")
        tokens = x.shape[0]
        if tokens:
            q, scale = pre_quant if pre_quant is not None else per_token_quant_int8(x.contiguous())
            self.buffer.x[:tokens].copy_(q)
            self.buffer.x_sf[:tokens].copy_(scale.reshape(-1))
            self.buffer.topk_idx[:tokens].copy_(topk_output.topk_ids)
            self.buffer.topk_weights[:tokens].copy_(topk_output.topk_weights)
        output = torch.empty_like(x)
        w4a8_mega_moe(output, (layer.w13_weight, layer.w13_weight_scale),
                      (layer.w2_weight, layer.w2_weight_scale), self.buffer,
                      activation_clamp=float(limit), activation="gelu")
        return output

    def apply(self, layer, dispatch_output, bias=None, **kwargs):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        output = self._run(layer, dispatch_output.hidden_states,
                           dispatch_output.topk_output,
                           getattr(dispatch_output, "hidden_states_pre_quant", None))
        if bias is not None:
            output.add_(bias)
        return StandardCombineInput(hidden_states=output)

    def apply_with_shared_output(self, layer, x, activation="gelu", shared_output=None,
                                 topk_output=None, i_q=None, i_s=None, **kwargs):
        if activation != "gelu":
            raise ValueError("Spark MegaMoE expects GELU")
        output = self._run(layer, x, topk_output,
                           (i_q, i_s) if i_q is not None and i_s is not None else None)
        if shared_output is not None:
            output.add_(shared_output)
        return output
