"""Packed W4A8 MegaMoE for Spark CP8/EP8 prefill.

Reuse the SlimQuant loader and native HIPC weight layout. Routing, independent
shared experts and GELU clamp context remain owned by TokenFlow. MegaMoE
returns a fully combined local token slice; it needs no second EP reduction.
"""

import inspect
from contextvars import ContextVar
import logging

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import (
    SlimQuantW4A8Int8MarlinMoEMethod,
)

logger = logging.getLogger(__name__)
_BUFFERS = {}


_ROUTED_INPUT = ContextVar("spark_w4a8_routed_input", default=None)


def _valid_routed_input(x):
    return (isinstance(x, torch.Tensor) and x.is_cuda
            and x.dtype == torch.bfloat16 and x.ndim == 2
            and x.shape[1] == 6144 and x.shape[0] > 0 and x.is_contiguous())


def _prefill_launch_bound(batch, x, method):
    if x.shape[0] >= method.capacity or not method.bounded_launch:
        return None
    from sglang.srt.layers.attention.dsa.utils import (
        cal_padded_tokens, can_dsa_prefill_cp_round_robin_split,
    )
    from sglang.srt.layers.cp.utils import enable_cp_v2
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    if (batch is None or enable_cp_v2()
            or not batch.forward_mode.is_extend_without_speculative()
            or not can_dsa_prefill_cp_round_robin_split(batch)
            or p.attn_dp_size != 1 or p.attn_tp_size != 1
            or p.attn_cp_size != 8 or p.moe_ep_size != 8
            or batch.global_num_tokens_cpu is None
            or len(batch.global_num_tokens_cpu) != 1):
        return None
    count = cal_padded_tokens(batch)
    if type(count) is not int or count != x.shape[0] or count <= 0:
        return None
    return count * p.moe_ep_size, count


def _install_native_return_adapter():
    import tokenflow.models.sparkv1.layers.moe as spark_moe

    original = spark_moe.SparkV1MoE.forward
    if getattr(original, "_spark_w4a8_megamoe", False):
        return
    original_router = spark_moe.SparkV1Router.forward
    original_smooth = spark_moe.apply_input_smooth
    original_torch = spark_moe.torch

    class RoutedTorch:
        def __getattr__(self, name):
            return getattr(original_torch, name)

        def clamp(self, x, min=None, max=None, *, out=None):
            scope = _ROUTED_INPUT.get()
            if (scope is not None and x is scope.get("routed_input")
                    and out is None and min == -8 and max == 8):
                if scope["clamped"]:
                    raise RuntimeError("Repeated MegaMoE routed clamp")
                scope["clamped"] = True
                return x
            return original_torch.clamp(x, min=min, max=max, out=out)

    def router(self, *args, **kwargs):
        result = original_router(self, *args, **kwargs)
        scope = _ROUTED_INPUT.get()
        if scope is not None and self is scope["owner"].router:
            x = args[0] if args else kwargs.get("hidden_states")
            source = scope["input"]
            if (_valid_routed_input(x) and x.shape == source.shape
                    and x.stride() == source.stride()
                    and x.storage_offset() == source.storage_offset()
                    and x.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()):
                scope["routed_input"] = x
        return result

    def smooth(module, x, *, parameter_name="input_smooth_scale"):
        scope = _ROUTED_INPUT.get()
        if (scope is not None and scope["clamped"]
                and module is scope["owner"].experts
                and x is scope.get("routed_input")
                and parameter_name == "routed_gate_up_input_smooth_scale"):
            scale = getattr(module, parameter_name, None)
            if (isinstance(scale, torch.Tensor) and scale.shape == (6144,)
                    and scale.dtype == torch.float32 and scale.device == x.device
                    and scale.is_contiguous()):
                scope["pending"] = (x, scale)
                return x
            scope["clamped"] = False
            x = original_torch.clamp(x, -8, 8)
        return original_smooth(module, x, parameter_name=parameter_name)

    def forward(self, hidden_states, forward_batch=None,
                should_allreduce_fusion=False, use_reduce_scatter=False,
                gemm_output_zero_allocator=None):
        method = getattr(self.experts, "quant_method", None)
        combined = isinstance(method, SparkW4A8MegaMoEMethod)
        eligible = (combined and _valid_routed_input(hidden_states)
                    and getattr(self, "alt_stream", None) is None
                    and not torch.cuda.is_current_stream_capturing())
        scope = None
        token = None
        if eligible:
            if _ROUTED_INPUT.get() is not None:
                raise RuntimeError("Nested MegaMoE preparation scope")
            scope = dict(owner=self, input=hidden_states, clamped=False,
                         pending=None, consumed=False,
                         bound=_prefill_launch_bound(forward_batch, hidden_states, method))
            token = _ROUTED_INPUT.set(scope)
        try:
            result = original(self, hidden_states, forward_batch=forward_batch,
                              should_allreduce_fusion=should_allreduce_fusion or combined,
                              use_reduce_scatter=use_reduce_scatter,
                              gemm_output_zero_allocator=gemm_output_zero_allocator)
            if scope is not None and (scope["pending"] is not None
                    or scope["clamped"] and not scope["consumed"]):
                raise RuntimeError("Unconsumed MegaMoE input preparation")
            return result
        finally:
            if token is not None:
                _ROUTED_INPUT.reset(token)

    forward._spark_w4a8_megamoe = True
    spark_moe.torch = RoutedTorch()
    spark_moe.SparkV1Router.forward = router
    spark_moe.apply_input_smooth = smooth
    spark_moe.SparkV1MoE.forward = forward


class SparkW4A8MegaMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config):
        self.quant_config = quant_config
        _install_native_return_adapter()

    def create_weights(self, *args, **kwargs):
        return SlimQuantW4A8Int8MarlinMoEMethod.create_weights(self, *args, **kwargs)

    def create_moe_runner(self, layer, moe_runner_config):
        import json
        from pathlib import Path
        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.utils import is_hcu

        args = get_global_server_args()
        config = json.loads((Path(args.model_path) / "config.json").read_text())
        if not (is_hcu() and args.disaggregation_mode == "prefill"
                and args.tp_size == args.ep_size == args.attn_cp_size == 8
                and args.enable_prefill_cp and layer.moe_tp_size == 1
                and layer.num_experts == 256 and layer.top_k == 14
                and layer.hidden_size == 6144
                and layer.intermediate_size_per_partition == 1024
                and moe_runner_config.activation == "gelu"
                and config.get("num_groups") == 2
                and config.get("num_experts_per_tok") == 7
                and not layer.num_fused_shared_experts and not layer.reduce_results
                and not args.enable_eplb):
            raise ValueError("Spark W4A8 MegaMoE requires CP8/EP8 P with independent shared experts")
        self.capacity = (min(args.chunked_prefill_size, args.max_prefill_tokens) + 7) // 8
        self.capacity = (self.capacity + 63) // 64 * 64
        if self.capacity <= 0:
            raise ValueError("Spark MegaMoE requires a bounded positive prefill chunk")
        # Each EP rank owns 32 consecutive slots inside one 128-slot group.
        # A token selects seven experts per group, so no rank receives 14.
        self.rank_route_bound = config["num_experts_per_tok"]

    def process_weights_after_loading(self, layer):
        import megamoe
        from deepgemm import pack_w4a8_moe_hipc_weight
        from megamoe.w4a8 import w4a8_mega_moe
        from sglang.srt.runtime_context import get_parallel

        if "max_routes_per_token_per_rank" not in inspect.signature(w4a8_mega_moe).parameters:
            raise RuntimeError("Install the MegaMoE wheel with bounded route workspace support")
        self.bounded_launch = all(name in inspect.signature(w4a8_mega_moe).parameters
                                  for name in ("max_dispatch_tokens", "max_peer_tokens"))
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
                          activation_clamp=8.0, activation="gelu",
                          max_routes_per_token_per_rank=self.rank_route_bound)
            torch.cuda.current_stream().synchronize()
            _BUFFERS[key] = buffer
            logger.info("Spark W4A8 MegaMoE initialized: EP8 capacity=%s rank_route_bound=%s GELU/clamp, shared workspace before KV sizing", self.capacity, self.rank_route_bound)
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
        scope = _ROUTED_INPUT.get()
        scope = scope if scope is not None and layer is scope["owner"].experts else None
        prepared = scope is not None and scope["pending"] is not None
        if prepared:
            source, smooth = scope["pending"]
            scope["pending"] = None
            if source is not x or pre_quant is not None:
                raise RuntimeError("MegaMoE prepared input identity mismatch")
            from lightop import op
            op.per_token_dynamic_quant_int8(
                self.buffer.x[:tokens], x,
                self.buffer.x_sf[:tokens].view(-1, 1), smooth, True, 8.0)
            scope["consumed"] = True
        if tokens:
            if not prepared:
                q, scale = pre_quant if pre_quant is not None else per_token_quant_int8(x.contiguous())
                self.buffer.x[:tokens].copy_(q)
                self.buffer.x_sf[:tokens].copy_(scale.reshape(-1))
            self.buffer.topk_idx[:tokens].copy_(topk_output.topk_ids)
            self.buffer.topk_weights[:tokens].copy_(topk_output.topk_weights)
        launch_options = {}
        if scope is not None and scope["bound"] is not None:
            dispatch, peer = scope["bound"]
            launch_options = dict(max_dispatch_tokens=dispatch, max_peer_tokens=peer)
        output = torch.empty_like(x)
        w4a8_mega_moe(output, (layer.w13_weight, layer.w13_weight_scale),
                      (layer.w2_weight, layer.w2_weight_scale), self.buffer,
                      activation_clamp=float(limit), activation="gelu",
                      max_routes_per_token_per_rank=self.rank_route_bound, **launch_options)
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
