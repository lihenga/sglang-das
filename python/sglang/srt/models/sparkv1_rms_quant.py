"""Opt-in Spark norm/quant fusion. No MTP, cache or collective changes."""
from contextvars import ContextVar
import functools
import logging
import math

import torch
from sglang.srt.utils import get_bool_env_var, is_hcu

logger = logging.getLogger(__name__)
_PENDING = '_spark_native_rms_pending'
_MARKER = '_spark_native_rms_quant_target'
_TARGET = '_spark_native_rms_norm_target'
_CONSUMER_TYPES = set()
_CLAMPED = '_spark_native_rms_clamped_output'
_DUAL_TARGET = '_spark_native_rms_dual_target'
_RMS_SCOPE = ContextVar('spark_native_rms_scope', default=None)


def take_prequantized(linear, x):
    """Consume once and only for the exact Tensor produced in this forward."""
    pending = linear.__dict__.pop(_PENDING, None)
    if pending is None or pending[0] is not x:
        return None
    # Q-up/shared experts may consume on TokenFlow's existing alternate stream.
    # Its waits already order producer/consumer work, but the new quant buffers
    # must also remain allocated until the consuming stream finishes.
    stream = torch.cuda.current_stream(x.device)
    pending[1].record_stream(stream)
    pending[2].record_stream(stream)
    kind = (type(linear).__name__, x.shape[-1])
    if kind not in _CONSUMER_TYPES:
        logger.info('Spark RMS prequant consumed: linear=%s width=%s', *kind)
        _CONSUMER_TYPES.add(kind)
    return [pending[1], pending[2]]


def _bind(linear, x, q, scale, limit):
    setattr(linear, _PENDING, (x, q, scale, limit))
    setattr(x, _MARKER, (linear, limit))


def _transfer(src, dst, low, high):
    marker = getattr(src, _MARKER, None)
    if marker is None:
        return
    linear, limit = marker
    pending = getattr(linear, _PENDING, None)
    if pending is not None and pending[0] is src and low == -limit and high == limit:
        _bind(linear, dst, pending[1], pending[2], limit)


class _ClampProxy:
    """Preserve the original clamp and carry only a same-forward quant hint."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def clamp(self, x, *args, **kwargs):
        if isinstance(x, torch.Tensor) and kwargs.get('out') is None:
            low = kwargs.get('min', args[0] if args else None)
            high = kwargs.get('max', args[1] if len(args) > 1 else None)
            saved = getattr(x, _CLAMPED, None)
            marker = getattr(x, _MARKER, None)
            if saved is not None and marker is not None:
                clipped, limit, scope = saved
                linear, original_limit = marker
                pending = getattr(linear, _PENDING, None)
                if (scope is _RMS_SCOPE.get() and scope is not None
                        and isinstance(low, (int, float)) and isinstance(high, (int, float))
                        and low == -limit and high == limit and original_limit == limit
                        and pending is not None and pending[0] is x):
                    x.__dict__.pop(_CLAMPED, None)
                    clipped.record_stream(torch.cuda.current_stream(clipped.device))
                    _transfer(x, clipped, low, high)
                    return clipped
        result = self.original.clamp(x, *args, **kwargs)
        if kwargs.get('out') is None:
            low = kwargs.get('min', args[0] if args else None)
            high = kwargs.get('max', args[1] if len(args) > 1 else None)
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                _transfer(x, result, low, high)
        return result


def install_rms_quant():
    flags = {name: get_bool_env_var(name, 'false') for name in (
        'SGLANG_USE_LIGHTOP_RMSNORM_CLAMP',
        'SGLANG_USE_LIGHTOP_FUSE_Q_A_RMSNORM_CLAMP_SMOOTH_QUANT',
        'SGLANG_USE_LIGHTOP_FUSE_DECODER_RMSNORM_CLAMP_SMOOTH_QUANT')}
    if not (is_hcu() and get_bool_env_var('SGLANG_USE_LIGHTOP') and any(flags.values())):
        return False
    import lightop.rmsnorm as rms_ops
    import tokenflow.models.sparkv1.layers.attention as attention
    import tokenflow.models.sparkv1.layers.decoder as decoder
    import tokenflow.models.sparkv1.layers.moe as moe
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.linear import ColumnParallelLinear
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
    if getattr(rms_ops, 'SGLANG_NATIVE_RMS_QUANT_VERSION', 0) < 1:
        raise RuntimeError('RMS fusion requires FP32-sum/out-of-place-residual LightOp; old wheel is unsafe')
    if getattr(RMSNorm.forward, '_spark_native_rms_fused', False):
        return True
    dual_output = get_bool_env_var('SGLANG_DSA_FUSED_RMS_CLAMP_OUTPUT', 'false')
    strided_input = get_bool_env_var('SGLANG_DSA_STRIDED_RMS_INPUT', 'false')
    if (dual_output or strided_input) and rms_ops.SGLANG_NATIVE_RMS_QUANT_VERSION < 2:
        raise RuntimeError('RMS dual output/row stride requires LightOp RMS ABI2')
    fused = rms_ops.rms_norm_dynamic_per_token_quant_sglang
    original_norm = RMSNorm.forward
    clamped_cls = attention.SparkV1ClampedRMSNorm
    original_clamped = clamped_cls.forward
    original_decoder = decoder.SparkV1DecoderLayer.forward
    original_q_b = attention.SparkV1AttentionMLA.q_b_proj_forward
    logged = set()
    binding_kinds = set()

    def valid_linear(linear, width):
        scale = getattr(linear, 'input_smooth_scale', None)
        return (isinstance(getattr(linear, 'scheme', None), CompressedTensorsW8A8Int8)
                and getattr(getattr(linear, 'weight', None), 'dtype', None) == torch.int8
                and isinstance(scale, torch.Tensor) and scale.ndim == 1
                and scale.numel() == width and scale.dtype == torch.float32
                and scale.is_contiguous())

    def run(norm, x, residual, target, limit, clamp_output):
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_cuda
                or x.dtype not in (torch.bfloat16, torch.float16) or x.shape[0] == 0
                or x.shape[-1] <= 0 or x.shape[-1] > 8192 or x.shape[-1] % 16
                or norm.cast_x_before_out_mul or norm.fp32_residual
                or getattr(norm, 'x_pad_to_multiple', 0) != 0
                or norm.variance_size_override is not None
                or norm.override_orig_dtype not in (None, x.dtype)
                or norm.weight.dtype != x.dtype or norm.weight.device != x.device
                or not norm.weight.is_contiguous()
                or not isinstance(limit, (int, float)) or not math.isfinite(limit)
                or (limit < 0 and limit != -1)):
            return None
        if residual is not None and (residual.dtype != x.dtype or residual.device != x.device
                                      or residual.shape != x.shape or not residual.is_contiguous()):
            return None
        smooth = getattr(target, 'input_smooth_scale', None)
        if target is not None and (not valid_linear(target, x.shape[-1]) or smooth.device != x.device):
            return None
        use_stride = (strided_input and residual is None and x.shape[-1] == 1536
                      and x.stride(1) == 1 and x.stride(0) >= x.shape[1]
                      and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0)
        x_contiguous = x if use_stride else x.contiguous()
        normalized = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        clipped = (torch.empty_like(normalized) if dual_output and not clamp_output
                   and target is not None and getattr(norm, _DUAL_TARGET, None) is target
                   and _RMS_SCOPE.get() is not None else None)
        extra = {'clamped_output': clipped} if dual_output else {}
        residual_out = torch.empty_like(residual) if residual is not None else None
        q, scale = fused(x_contiguous, norm.weight, norm.variance_epsilon,
                         residual=residual, residual_output=residual_out,
                         update_input=False, limit=float(limit), smooth_scale=smooth,
                         norm_output=normalized, clamp_norm_output=clamp_output, **extra)
        if target is not None:
            _bind(target, normalized, q, scale, float(limit))
            if clipped is not None:
                setattr(normalized, _CLAMPED, (clipped, float(limit), _RMS_SCOPE.get()))
        label = (x.shape[-1], residual is not None, target is not None, clamp_output)
        if label not in logged:
            logger.info('Spark native-rounding RMS fusion active width=%s residual=%s prequant=%s clamp_output=%s', *label)
            logged.add(label)
        return normalized if residual is None else (normalized, residual_out)

    @functools.wraps(original_norm)
    def norm_forward(self, x, *args, **kwargs):
        target = getattr(self, _TARGET, None)
        residual = args[0] if args else None
        post_residual_addition = args[1] if len(args) > 1 else None
        if target is not None and len(args) <= 2 and post_residual_addition is None and not kwargs:
            result = run(self, x, residual, target[0], target[1], False)
            if result is not None:
                return result
        return original_norm(self, x, *args, **kwargs)

    @functools.wraps(original_clamped)
    def clamped_forward(self, x, *args, **kwargs):
        target = getattr(self, _TARGET, None)
        limit = getattr(self, 'clamp_value', None)
        if not args and not kwargs and (target is not None or flags['SGLANG_USE_LIGHTOP_RMSNORM_CLAMP']):
            result = run(self, x, None, target[0] if target else None, limit, True)
            if result is not None:
                return result
            label = ('clamped_fallback', self.hidden_size, tuple(x.shape), x.dtype, self.weight.dtype)
            if label not in logged:
                logger.info('Spark RMS native fallback: shape=%s dtype=%s weight=%s cast_before=%s fp32_residual=%s override=%s variance_override=%s',
                            tuple(x.shape), x.dtype, self.weight.dtype, self.cast_x_before_out_mul,
                            self.fp32_residual, self.override_orig_dtype, self.variance_size_override)
                logged.add(label)
        return original_clamped(self, x, *args, **kwargs)

    @functools.wraps(original_decoder)
    def decoder_forward(self, *args, **kwargs):
        bindings = []
        attn = self.self_attn
        kvnorm = getattr(attn, 'kv_a_layernorm', None)
        if kvnorm is not None:
            kind = ('kv_norm_entry', type(kvnorm).__name__, kvnorm.hidden_size)
            if kind not in binding_kinds:
                logger.info('Spark standalone RMS entry: type=%s width=%s forward=%s clamp=%s weight=%s',
                            type(kvnorm).__name__, kvnorm.hidden_size,
                            getattr(kvnorm.forward, '__name__', str(type(kvnorm.forward))),
                            getattr(kvnorm, 'clamp_value', None), kvnorm.weight.dtype)
                binding_kinds.add(kind)
        if flags['SGLANG_USE_LIGHTOP_FUSE_DECODER_RMSNORM_CLAMP_SMOOTH_QUANT']:
            bindings.append((self.input_layernorm, getattr(attn, 'fused_qkv_a_proj_with_mqa', None),
                             getattr(attn, 'clamp_value', None)))
            dense = getattr(self.mlp, 'shared_experts', None)
            if dense is None:
                dense = self.mlp
            if get_bool_env_var('SGLANG_USE_LIGHTOP_FUSE_DENSE_GELU_CLAMP_SMOOTH_QUANT', 'false'):
                bindings.append((self.post_attention_layernorm, getattr(dense, 'fc1', None),
                                 getattr(self.mlp, 'clamp_value', None)))
        if flags['SGLANG_USE_LIGHTOP_FUSE_Q_A_RMSNORM_CLAMP_SMOOTH_QUANT']:
            qnorm = getattr(attn, 'q_a_layernorm', None)
            bindings.append((qnorm, getattr(attn, 'q_b_proj', None), getattr(qnorm, 'clamp_value', None)))
        active = []
        for norm, linear, limit in bindings:
            if (norm is not None and linear is not None and isinstance(limit, (int, float))
                    and limit >= 0 and valid_linear(linear, norm.hidden_size)):
                binding_kind = (type(norm).__name__, type(linear).__name__, norm.hidden_size,
                                getattr(norm.forward, '__name__', type(norm.forward).__name__))
                if binding_kind not in binding_kinds:
                    logger.info('Spark RMS binding: norm=%s linear=%s width=%s forward=%s', *binding_kind)
                    binding_kinds.add(binding_kind)
                setattr(norm, _TARGET, (linear, limit));active.append((norm, linear))
                if dual_output and norm is self.input_layernorm:
                    setattr(norm, _DUAL_TARGET, linear)
        scope_token = _RMS_SCOPE.set(object()) if dual_output else None
        try:
            return original_decoder(self, *args, **kwargs)
        finally:
            for norm, linear in active:
                norm.__dict__.pop(_TARGET, None)
                norm.__dict__.pop(_DUAL_TARGET, None)
                linear.__dict__.pop(_PENDING, None)
            if scope_token is not None:
                _RMS_SCOPE.reset(scope_token)

    @functools.wraps(original_q_b)
    def q_b_forward(self, q_lora):
        # q_b is a ColumnParallelLinear, unlike fused_qkv_a's replicated
        # projection. Consume its norm output here without a second smooth/quant.
        linear = self.q_b_proj
        if not isinstance(linear, ColumnParallelLinear):
            return original_q_b(self, q_lora)
        pending = take_prequantized(linear, q_lora)
        if pending is None:
            return original_q_b(self, q_lora)
        q, _ = ColumnParallelLinear.forward(
            linear, q_lora, input_quant_args=[pending[0], pending[1].reshape(-1, 1)])
        return q.view(-1, self.num_local_heads, self.qk_head_dim)

    # Python runs these hints during eager/capture only. Graph replay consumes
    # captured tensor buffers; no .item(), host token counts or MTP state added.
    norm_forward._spark_native_rms_fused = True
    RMSNorm.forward = norm_forward
    clamped_cls.forward = clamped_forward
    decoder.SparkV1DecoderLayer.forward = decoder_forward
    attention.SparkV1AttentionMLA.q_b_proj_forward = q_b_forward
    for module in (decoder, attention, moe):
        if not isinstance(module.torch, _ClampProxy):
            module.torch = _ClampProxy(module.torch)
    logger.info('Installed Spark RMS fusions (native FP32 residual sum; out-of-place outputs): %s', flags)
    return True
