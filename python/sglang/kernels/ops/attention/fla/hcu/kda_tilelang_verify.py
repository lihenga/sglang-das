"""TileLang KDA ``target_verify`` kernel and a drop-in dispatch helper.

This is an optional, faster replacement for the Triton
``fused_sigmoid_gating_delta_rule_update`` used by KDA speculative
``target_verify`` (EAGLE topk=1, ReplaySSM input staging). It is gated by
``SGLANG_KDA_USE_TILELANG_VERIFY`` (default off) and is *fail-open*:
:func:`tilelang_target_verify_or_none` returns ``None`` whenever the feature
is disabled, the call shape is unsupported, or the kernel raises — the caller
then falls back to the Triton path unchanged, so enabling this can never
break correctness or crash CUDA-graph capture (host-side errors; a
device-side sticky CUDA error would poison the context regardless).

Operational log reading (with the flag on): each distinct decline reason
logs one ``[KDA-TileLang] fast path declined: ...`` WARNING; a kernel
failure additionally disables the path for the process (``... disabling
the TileLang verify path ...``). The fast path is in use when the
``[KDA-TileLang] compiling target_verify kernel`` INFO lines are followed
by no decline warnings for the serving shapes.

Kernel semantics mirror ``fused_sigmoid_gating_delta_rule_update`` with
``is_kda=True, use_qk_l2norm_in_kernel=True, lower_bound=-5.0,
beta_scale=1.0, disable_state_update=True, cache_replayssm_inputs=True`` and
``retrieve_parent_token=None`` (linear EAGLE chain). Pooled addressing: the
recurrent state is read from ``h0[slot]`` and ReplaySSM rings are written to
``ring[slot]`` where ``slot = initial_state_indices[n]``; the ring window
``W`` may exceed the number of draft tokens ``S`` (writes land in ``[0, S)``).

The kernel supports any ``2 <= S <= 8`` (odd and even). A whole ``(S, *)``
row lives in one 32-lane warp via an explicit ``warp_row`` layout, so layout
inference does not require ``S`` to be even (auto inference would need
``S * head_dim`` to divide the 256-thread block).

Deferred commit (``defer=True``): instead of a separate post-sampling
ReplaySSM commit kernel re-reading the fp32 state, the previous step's
accepted prefix is folded here. ``pend[slot]`` holds how many ring rows of
the *previous* verify were accepted; the kernel replays them onto the loaded
state, writes the committed state back to ``h0[slot]``, and only then
overwrites the ring with this step's inputs. Invariant maintained by the
caller: committed state == ``h0[slot]`` replayed with
``ring[slot][:pend[slot]]``.
"""

import functools
import logging
from typing import Optional

import torch

try:
    import tilelang
    import tilelang.language as T

    # Older/broken installs may lack the API (or fail at import with
    # something other than ImportError); either way the helper must be
    # importable so the Triton path keeps working (fail-open contract).
    _set_log_level = getattr(tilelang, "set_log_level", None)
    if _set_log_level is not None:
        _set_log_level("WARNING")
    _HAS_TILELANG = True
except Exception:  # tilelang is optional; helper then always returns None
    tilelang = None
    T = None
    _HAS_TILELANG = False

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _pass_configs():
    # Enable fast math, tolerant of the flag rename across tilelang versions
    # (see layers/attention/nsa/tilelang_kernel.py for the same handling).
    cfg = {}
    if hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
        cfg[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = True
    elif hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
        cfg[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = False
    return cfg


def make_kernel_pooled(N, S, HV, K, V, P, PR, W,
                       a_stride=None, b_stride=None, q_stride=None,
                       k_stride=None, v_stride=None, h0_stride=None,
                       rv_slot=None, rk_slot=None, rg_slot=None, rb_slot=None,
                       pool_dtype="float32", defer=False):
    """Build the pooled prim_func.

    ``h0`` is ``(P, HV, V, K)``; rings are ``(PR, HV, W, *)``; ``idx`` is
    ``(N,)`` mapping each sequence to its pool slot. ``pend`` is ``(PR,)``
    and only read when ``defer`` is set (see module docstring).

    All activations (a/b/q/k/v), the state pool and the rings take a
    compile-time outer stride (token stride for activations, per-slot pitch
    for pool/rings): the kernel reads the serving views in place with zero
    host-side copies. ``None`` selects the contiguous layout.
    """
    assert 2 <= S <= 8, "S must be in 2..8"
    assert W >= S
    # The fold reuses this step's (S, *) staging buffers for the previous
    # step's ring rows, so the ring must hold exactly one verify window.
    assert not defer or W == S, "deferred commit requires W == S"
    scale = float(K ** -0.5)
    a_stride = HV * K if a_stride is None else a_stride
    b_stride = HV if b_stride is None else b_stride
    q_stride = HV * K if q_stride is None else q_stride
    k_stride = HV * K if k_stride is None else k_stride
    v_stride = HV * V if v_stride is None else v_stride
    h0_stride = HV * V * K if h0_stride is None else h0_stride
    rv_slot = HV * W * V if rv_slot is None else rv_slot
    rk_slot = HV * W * K if rk_slot is None else rk_slot
    rg_slot = HV * W * K if rg_slot is None else rg_slot
    rb_slot = HV * W if rb_slot is None else rb_slot
    # pool_dtype selects fp32 (default) or fp16 state-pool storage: the h0
    # read is the dominant memory traffic of this kernel and fp16 halves it.
    # Computation stays fp32 either way (loaded as fp16, cast in registers).
    assert pool_dtype in ("float32", "float16"), pool_dtype

    # Warp-per-row layout for the (S, *) register fragments: row i lives
    # entirely inside warp i (thread = i*32 + j//4, reg slot = j%4). This
    # keeps the l2norm reduce intra-warp and, unlike auto layout inference
    # (which needs S*head_dim % 256 == 0, i.e. even S), is valid for any
    # S <= 8 -- so odd S compiles.
    def warp_row(shape):
        return T.Fragment(
            shape,
            forward_thread_fn=lambda i, j: i * 32 + j // 4,
            forward_index_fn=lambda i, j: j % 4,
        )

    @T.prim_func
    def main(
        A_log: T.Tensor((HV,), "float32"),
        dt_bias: T.Tensor((HV, K), "float32"),
        a_raw: T.StridedTensor[(N * S, HV * K), (a_stride, 1), "bfloat16"],
        b_raw: T.StridedTensor[(N * S, HV), (b_stride, 1), "bfloat16"],
        q_raw: T.StridedTensor[(N * S, HV * K), (q_stride, 1), "bfloat16"],
        k_raw: T.StridedTensor[(N * S, HV * K), (k_stride, 1), "bfloat16"],
        v_raw: T.StridedTensor[(N * S, HV * V), (v_stride, 1), "bfloat16"],
        o_out: T.Tensor((N * S, HV, V), "bfloat16"),
        h0: T.StridedTensor[(P, HV, V, K), (h0_stride, V * K, K, 1), pool_dtype],
        idx: T.Tensor((N,), "int32"),
        ring_v: T.StridedTensor[(PR, HV, W, V), (rv_slot, W * V, V, 1), "bfloat16"],
        ring_k: T.StridedTensor[(PR, HV, W, K), (rk_slot, W * K, K, 1), "bfloat16"],
        ring_g: T.StridedTensor[(PR, HV, W, K), (rg_slot, W * K, K, 1), "float32"],
        ring_beta: T.StridedTensor[(PR, HV, W), (rb_slot, W, 1), "float32"],
        pend: T.Tensor((PR,), "int32"),
    ):
        with T.Kernel(N * HV, threads=256) as (bnh,):
            i_n = bnh // HV
            i_hv = bnh % HV
            tok0 = i_n * S
            c0 = i_hv * K
            slot = T.cast(idx[i_n], "int32")
            # -1 padding (CUDA-graph replay pads mamba_indices with PAD_SLOT_ID):
            # clamp the load/store slot to the permanent dummy slot 0 (memory_pool
            # reserves it: "dummy reads/writes land here harmlessly") and zero the
            # initial state via the gate multiply, matching Triton's `if idx >= 0`.
            # Upper-bound invariant (not re-checked here -- no device sync on the
            # hot path): the pool allocator only ever hands out slots in
            # [1, PR-1] (free_slots = range(1, size+1)), so idx > PR-1 cannot
            # occur; if it ever did, the compiled pitch addressing would corrupt
            # a neighbouring slot rather than fault.
            safe = T.if_then_else(slot >= 0, slot, 0)
            gate = T.if_then_else(slot >= 0, 1.0, 0.0)

            qh = T.alloc_shared((S, K), "float32")
            kh_ = T.alloc_shared((S, K), "float32")
            eg = T.alloc_shared((S, K), "float32")
            vs = T.alloc_shared((S, V), "float32")
            bt = T.alloc_shared((S,), "float32")
            u_fr = T.alloc_fragment((V,), "float32")

            h_fr = T.alloc_fragment((V, K), "float32")
            tmp_fr = T.alloc_fragment((V, K), "float32")
            a1_fr = T.alloc_fragment((V,), "float32")
            a2_fr = T.alloc_fragment((V,), "float32")
            a_fr = T.alloc_fragment((S, K), "bfloat16")
            k_fr = T.alloc_fragment((S, K), "bfloat16")
            q_fr = T.alloc_fragment((S, K), "bfloat16")
            v_fr = T.alloc_fragment((S, V), "bfloat16")
            gv_fr = T.alloc_fragment((S, K), "float32")
            xf_fr = T.alloc_fragment((S, K), "float32")
            sq_fr = T.alloc_fragment((S, K), "float32")
            row_fr = T.alloc_fragment((S,), "float32")
            den_fr = T.alloc_fragment((S,), "float32")

            T.annotate_layout({
                a_fr: warp_row((S, K)),
                k_fr: warp_row((S, K)),
                q_fr: warp_row((S, K)),
                gv_fr: warp_row((S, K)),
                xf_fr: warp_row((S, K)),
                sq_fr: warp_row((S, K)),
                v_fr: warp_row((S, V)),
                # Pin h_fr to the layout the fp32 pool path derives natively
                # (16 threads per row x 8 register slots): without the pin an
                # fp16 pool drags h_fr to an 8x16 layout, which pushes the
                # recurrence reductions onto deeper local FADD trees on the
                # already-busy FMA pipe (measured +26% kernel time upstream).
                h_fr: T.Fragment(
                    (V, K),
                    forward_thread_fn=lambda i, j: (i % 16) * 16 + j // 8,
                    forward_index_fn=lambda i, j: (i // 16) * 8 + j % 8,
                ),
            })

            # ---- state in: pooled h0[slot] -> registers ----
            # fp16 pool: load as fp16, cast to fp32 in registers (the gate
            # multiply covers the -1 padding zeroing in the same pass).
            if pool_dtype == "float16":
                h16_fr = T.alloc_fragment((V, K), "float16")
                T.copy(h0[safe, i_hv, 0 : V, 0 : K], h16_fr)
                for i, j in T.Parallel(V, K):
                    h_fr[i, j] = T.cast(h16_fr[i, j], "float32") * gate
            else:
                T.copy(h0[safe, i_hv, 0 : V, 0 : K], h_fr)
                for i, j in T.Parallel(V, K):
                    h_fr[i, j] = h_fr[i, j] * gate
            A_v = T.cast(A_log[i_hv], "float32")

            # ---- deferred commit: fold the previous verify's accepted rows ----
            # Matches kda_replayssm_exact_fold_kernel (rsqrt k-norm, stored
            # gate). The (S, *) staging buffers are borrowed for the old ring
            # rows; the barrier after the fold orders every old-ring read and
            # every fold-step read of the shared buffers before they are
            # refilled and the ring is overwritten below.
            if defer:
                n_pend = T.if_then_else(slot >= 0, T.min(pend[safe], S), 0)
                T.copy(ring_k[safe, i_hv, 0 : S, 0 : K], k_fr)
                T.copy(ring_g[safe, i_hv, 0 : S, 0 : K], gv_fr)
                T.copy(ring_v[safe, i_hv, 0 : S, 0 : V], v_fr)
                for i, j in T.Parallel(S, K):
                    xf_fr[i, j] = T.cast(k_fr[i, j], "float32")
                    sq_fr[i, j] = xf_fr[i, j] * xf_fr[i, j]
                T.reduce_sum(sq_fr, row_fr, dim=1, clear=True)
                for i in T.Parallel(S):
                    den_fr[i] = T.rsqrt(row_fr[i] + 1e-6)
                for i, j in T.Parallel(S, K):
                    kh_[i, j] = xf_fr[i, j] * den_fr[i]
                    eg[i, j] = T.exp(gv_fr[i, j])
                for i, j in T.Parallel(S, V):
                    vs[i, j] = T.cast(v_fr[i, j], "float32")
                for i in T.Parallel(S):
                    bt[i] = ring_beta[safe, i_hv, i]
                T.sync_threads()

                # Static trip count + uniform guard: a data-dependent extent
                # makes tilelang wrap every shared access in bounds checks.
                for s in T.Serial(0, S):
                    if s < n_pend:
                        for i, j in T.Parallel(V, K):
                            h_fr[i, j] = h_fr[i, j] * eg[s, j]
                        for i, j in T.Parallel(V, K):
                            tmp_fr[i, j] = h_fr[i, j] * kh_[s, j]
                        T.reduce_sum(tmp_fr, a1_fr, dim=1, clear=True)
                        for i in T.Parallel(V):
                            u_fr[i] = (vs[s, i] - a1_fr[i]) * bt[s]
                        for i, j in T.Parallel(V, K):
                            h_fr[i, j] = h_fr[i, j] + kh_[s, j] * u_fr[i]
                if n_pend > 0:
                    if pool_dtype == "float16":
                        for i, j in T.Parallel(V, K):
                            h0[safe, i_hv, i, j] = T.cast(h_fr[i, j], "float16")
                    else:
                        T.copy(h_fr, h0[safe, i_hv, 0 : V, 0 : K])
                if pool_dtype == "float16":
                    # Keep the in-register state bit-identical to the fp16
                    # checkpoint just stored: the eager path reads h0 back next
                    # step, so continuing the recurrence from the un-rounded
                    # fp32 value would make deferred commit diverge from eager
                    # by one fp16 rounding per step. The store above already
                    # performed this rounding, so this is a re-use, not extra
                    # precision loss.
                    for i, j in T.Parallel(V, K):
                        h_fr[i, j] = T.cast(
                            T.cast(h_fr[i, j], "float16"), "float32"
                        )
                T.sync_threads()
            # ---- gate prep: g = -5 * sigmoid(exp(A_log) * (a + dt_bias)) ----
            T.copy(a_raw[tok0 : tok0 + S, c0 : c0 + K], a_fr)
            T.copy(k_raw[tok0 : tok0 + S, c0 : c0 + K], k_fr)
            for i, j in T.Parallel(S, K):
                x = T.cast(a_fr[i, j], "float32") + \
                    T.cast(dt_bias[i_hv, j], "float32")
                gv_fr[i, j] = -5.0 * (1.0 / (1.0 + T.exp(-(T.exp(A_v) * x))))
            for i, j in T.Parallel(S, K):
                eg[i, j] = T.exp(gv_fr[i, j])
            T.copy(gv_fr, ring_g[safe, i_hv, 0 : S, 0 : K])
            T.copy(k_fr, ring_k[safe, i_hv, 0 : S, 0 : K])

            # ---- q/k l2norm ----
            T.copy(q_raw[tok0 : tok0 + S, c0 : c0 + K], q_fr)
            for i, j in T.Parallel(S, K):
                xf_fr[i, j] = T.cast(q_fr[i, j], "float32")
                sq_fr[i, j] = xf_fr[i, j] * xf_fr[i, j]
            T.reduce_sum(sq_fr, row_fr, dim=1, clear=True)
            for i in T.Parallel(S):
                den_fr[i] = T.sqrt(row_fr[i] + 1e-6)
            for i, j in T.Parallel(S, K):
                qh[i, j] = xf_fr[i, j] / den_fr[i] * scale

            for i, j in T.Parallel(S, K):
                xf_fr[i, j] = T.cast(k_fr[i, j], "float32")
                sq_fr[i, j] = xf_fr[i, j] * xf_fr[i, j]
            T.reduce_sum(sq_fr, row_fr, dim=1, clear=True)
            for i in T.Parallel(S):
                den_fr[i] = T.sqrt(row_fr[i] + 1e-6)
            for i, j in T.Parallel(S, K):
                kh_[i, j] = xf_fr[i, j] / den_fr[i]

            # ---- v staging + beta = sigmoid(b) ----
            T.copy(v_raw[tok0 : tok0 + S, i_hv * V : i_hv * V + V], v_fr)
            for i, j in T.Parallel(S, V):
                vs[i, j] = T.cast(v_fr[i, j], "float32")
            T.copy(v_fr, ring_v[safe, i_hv, 0 : S, 0 : V])
            for i in T.Parallel(S):
                b_b = T.cast(b_raw[tok0 + i, i_hv], "float32")
                bt[i] = 1.0 / (1.0 + T.exp(-b_b))
                ring_beta[safe, i_hv, i] = 1.0 / (1.0 + T.exp(-b_b))

            T.sync_threads()

            # ---- gated delta-rule recurrence (serial over S) ----
            for s in T.Serial(0, S):
                for i, j in T.Parallel(V, K):           # H <- H * diag(exp(g))
                    h_fr[i, j] = h_fr[i, j] * eg[s, j]
                for i, j in T.Parallel(V, K):
                    tmp_fr[i, j] = h_fr[i, j] * kh_[s, j]
                T.reduce_sum(tmp_fr, a1_fr, dim=1, clear=True)  # H @ k_norm
                for i in T.Parallel(V):
                    u_fr[i] = (vs[s, i] - a1_fr[i]) * bt[s]      # beta*(v - Hk)
                for i, j in T.Parallel(V, K):           # H <- H + u (x) k_norm
                    h_fr[i, j] = h_fr[i, j] + kh_[s, j] * u_fr[i]
                for i, j in T.Parallel(V, K):
                    tmp_fr[i, j] = h_fr[i, j] * qh[s, j]
                T.reduce_sum(tmp_fr, a2_fr, dim=1, clear=True)   # o = H @ q_norm
                T.copy(a2_fr, o_out[tok0 + s, i_hv, 0 : V])

    return main


def _is_capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # non-CUDA device (XPU): no capture concept
        return False


_compiled_keys = set()  # keys successfully compiled by _compiled/precompile


def _compiled_capture_safe(N, S, HV, K, V, P, PR, W, strides, pool_dtype,
                           defer=False):
    """Compile (or fetch) the kernel; never JIT inside a graph capture.

    Compiling runs host-side codegen and CUDA module loads, which are
    illegal mid-capture. On a capture-time cache miss (precompile gap or
    an unlisted batch size) decline: this call falls back to Triton and
    the captured graph permanently uses the Triton kernel for this shape
    (same policy as the upstream kernelhub integration). Outside capture
    this is a plain cached compile.
    """
    key = (N, S, HV, K, V, P, PR, W, strides, pool_dtype, defer)
    if key not in _compiled_keys and _is_capturing():
        logger.warning(
            "[KDA-TileLang] cache miss during CUDA-graph capture (N=%d); "
            "this graph will use the Triton path -- add the batch size to "
            "precompile to enable the fast path", N,
        )
        return None
    return _compiled(*key)


@functools.lru_cache(maxsize=256)
def _compiled(N, S, HV, K, V, P, PR, W, strides, pool_dtype="float32",
              defer=False):
    logger.info(
        "[KDA-TileLang] compiling target_verify kernel: N=%d S=%d HV=%d "
        "K=%d V=%d pool=%d/%d window=%d pool=%s defer=%s strides=%s",
        N, S, HV, K, V, P, PR, W, pool_dtype, defer, strides,
    )
    kw = dict(zip(_STRIDE_KEYS, strides))
    kern = tilelang.jit(make_kernel_pooled, pass_configs=_pass_configs())(
        N, S, HV, K, V, P, PR, W, pool_dtype=pool_dtype, defer=defer, **kw
    )
    _compiled_keys.add((N, S, HV, K, V, P, PR, W, strides, pool_dtype, defer))
    return kern


def precompile(n_list, S, HV, K, V, P, PR, W, pool_dtype="float32",
               defer=False, **strides):
    """Warm the kernel cache ahead of CUDA-graph capture.

    Call this from model warmup with every capture batch size in ``n_list``
    (plus the eager serving sizes you care about): a mid-capture cache miss
    would otherwise run host-side tilelang JIT inside the capture region.
    Pool/pitch parameters (P, PR, W) must be the final serving values.
    ``strides`` are the keyword stride/pitch arguments of
    ``make_kernel_pooled`` (a_stride, b_stride, ..., rb_slot) or empty for
    the contiguous layout. ``pool_dtype`` is the state-pool storage dtype
    ("float32" or "float16"). ``defer`` selects the deferred-commit variant.
    Compilation errors propagate to the caller, who should treat the
    TileLang path as unavailable (the per-call helper stays fail-open).
    """
    strides = _normalize_strides(strides)
    for n in n_list:
        _compiled(n, S, HV, K, V, P, PR, W, strides, pool_dtype, defer)


_STRIDE_KEYS = ("a_stride", "b_stride", "q_stride", "k_stride", "v_stride",
                "h0_stride", "rv_slot", "rk_slot", "rg_slot", "rb_slot")


def _normalize_strides(strides):
    return tuple(strides.get(k) for k in _STRIDE_KEYS)



_fallback_reasons = set()


def _fb(reason_key: str, *detail):
    """Log (once per reason key) why the TileLang verify path declined.

    ``reason_key`` must be a constant string (dict/set lookups and the
    once-only bookkeeping are keyed on it); ``detail`` args are only
    formatted into the message on the first occurrence. The first decline
    of each reason logs at WARNING so silent fallbacks are visible in
    production; repeats are suppressed.
    """
    if reason_key not in _fallback_reasons:
        _fallback_reasons.add(reason_key)
        logger.warning("[KDA-TileLang] fast path declined: %s %s",
                       reason_key, " ".join(str(d) for d in detail) if detail else "")

_disabled_fallback = False


def tilelang_target_verify_or_none(
    *,
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    lower_bound,
    beta_scale,
    cache_steps,
    retrieve_parent_token,
    cache_replayssm_inputs,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    replayssm_pending=None,
    **_ignored,
) -> Optional[torch.Tensor]:
    """Run the TileLang target_verify kernel, or return ``None`` to fall back.

    Fail-open: any unsupported shape/dtype or host-side kernel error returns
    ``None`` instead of raising (a raise here would kill CUDA-graph capture).
    After the first failure the fast path is disabled for the rest of the
    process (sticky), so a pathological call shape cannot repeatedly pay JIT
    recompiles on the serving hot path. Note: this only covers host-side
    errors; an asynchronous device-side (sticky CUDA) error would already
    have poisoned the context regardless of this fallback.
    """
    global _disabled_fallback
    if _disabled_fallback:
        return None
    if not (_HAS_TILELANG and envs.SGLANG_KDA_USE_TILELANG_VERIFY.get()):
        return None
    try:
        return _tilelang_target_verify(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens, lower_bound=lower_bound,
            beta_scale=beta_scale, cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            cache_replayssm_inputs=cache_replayssm_inputs,
            replayssm_rawv=replayssm_rawv, replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g, replayssm_beta=replayssm_beta,
            replayssm_pending=replayssm_pending,
        )
    except Exception as e:  # noqa: BLE001 - deliberate fail-open to Triton
        if isinstance(e, torch.cuda.OutOfMemoryError):
            # Transient allocation failure (e.g. the output torch.empty racing
            # a memory spike): fall back to Triton for THIS call only -- the
            # next call may well succeed, so do not permanently disable.
            logger.warning(
                "[KDA-TileLang] fast path hit a transient OOM; falling back "
                "to Triton for this call only",
            )
            return None
        _disabled_fallback = True
        logger.warning(
            "[KDA-TileLang] fast path failed (%s: %s); disabling the TileLang "
            "verify path for the rest of this process and falling back to "
            "Triton", type(e).__name__, e,
        )
        return None


def _tilelang_target_verify(
    *,
    A_log,
    dt_bias,
    q,
    k,
    v,
    a,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    lower_bound,
    beta_scale,
    cache_steps,
    retrieve_parent_token,
    cache_replayssm_inputs,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    replayssm_pending=None,
) -> Optional[torch.Tensor]:
    # --- semantic gate: exactly the kernel variant we implement ---
    if not cache_replayssm_inputs:
        _fb("replayssm staging off")
        return None
    if retrieve_parent_token is not None:  # EAGLE tree (topk>1) unsupported
        _fb("EAGLE tree mask unsupported")
        return None
    if lower_bound is None or float(lower_bound) != -5.0:
        _fb("lower_bound != -5.0")
        return None
    if float(beta_scale) != 1.0:
        _fb("beta_scale != 1.0")
        return None
    if (replayssm_rawv is None or replayssm_rawk is None
            or replayssm_g is None or replayssm_beta is None):
        _fb("replayssm rings missing")
        return None

    S = int(cache_steps) if cache_steps else 0
    if not (2 <= S <= 8):
        _fb("cache_steps outside 2..8:", S)
        return None
    Tn = q.shape[1]
    if Tn % S != 0:
        _fb("tokens not divisible by cache_steps")
        return None
    N = Tn // S
    # equal-length S-chunks expected for the linear EAGLE topk=1 verify
    # INVARIANT: the kernel never reads cu_seqlens contents -- it addresses
    # rows purely as i_n*S. Every construction site of the verify qsl builds
    # arange(0, T+1, step=S) (eager: hybrid_linear_attn_backend; graph
    # capture/replay: cached_cuda_graph_verify_query_start_loc), and padding
    # tails are gated by idx==-1 instead of zero-length qsl segments. The
    # len==N+1 check below is therefore a shape gate only; a hypothetical
    # ragged qsl would silently miscompute rather than decline.
    if cu_seqlens is None or len(cu_seqlens) - 1 != N:
        _fb("cu_seqlens length != N+1")
        return None

    HV = v.shape[2]
    K = k.shape[3]
    V = v.shape[3]
    H = k.shape[2]
    if H != HV or K != V:  # kernel assumes H == HV and square head dims
        _fb("head geometry H/HV/K/V:", H, HV, K, V)
        return None
    # state pool storage: fp32 (default) or fp16 (halves the dominant h0
    # read traffic); computation stays fp32 either way
    pool_dtype = {torch.float32: "float32",
                  torch.float16: "float16"}.get(initial_state_source.dtype)
    if q.dtype != torch.bfloat16 or pool_dtype is None:
        if q.dtype != torch.bfloat16:
            _fb("activation dtype != bf16:", q.dtype)
        elif pool_dtype is None:
            _fb("state pool dtype unsupported:", initial_state_source.dtype)
        return None
    # serving may hand A_log/dt_bias in broadcast shapes ([1,1,HV,1], flat ...)
    if A_log.numel() != HV or dt_bias.numel() != HV * K:
        _fb("A_log/dt_bias numel mismatch")
        return None
    # gate math consumes dt_bias in fp32 (Triton parity); a non-fp32 input
    # falls back to Triton instead of silently changing gate precision.
    if dt_bias.dtype != torch.float32:
        _fb("dt_bias dtype != fp32:", dt_bias.dtype)
        return None

    W = replayssm_rawv.shape[-2]
    P = initial_state_source.shape[0]
    PR = replayssm_rawv.shape[0]
    if (replayssm_g.shape[0] != PR or replayssm_rawk.shape[0] != PR
            or replayssm_beta.shape[0] != PR):
        _fb("ring first dims disagree")
        return None
    if W < S:
        _fb("ring window W < S:", W, S)
        return None
    defer = replayssm_pending is not None
    if defer and (W != S or tuple(replayssm_pending.shape) != (PR,)
                  or replayssm_pending.dtype != torch.int32):
        _fb("deferred commit needs W == S and an int32 (PR,) pending table")
        return None
    # Ring / pool layouts: the kernel bakes per-slot pitches at compile
    # time, so only the inner dims need to be dense (the per-layer views
    # the Triton kernel gets are sliced along dim 0 and thus strided in
    # dim 0 -- that is fine, the slot pitch becomes a compile key).
    ring_strides = {}
    for name, t, tail, key in (
        ("rawv", replayssm_rawv, (HV, W, V), "rv_slot"),
        ("rawk", replayssm_rawk, (HV, W, K), "rk_slot"),
        ("g", replayssm_g, (HV, W, K), "rg_slot"),
        ("beta", replayssm_beta, (HV, W), "rb_slot"),
    ):
        if tuple(t.shape[1:]) != tail or t.stride(-1) != 1:
            _fb("ring layout mismatch:", name)
            return None
        ring_strides[key] = t.stride(0)
    if (initial_state_source.dim() != 4
            or initial_state_source.stride(-1) != 1
            or initial_state_source.stride(-2) != K
            or initial_state_source.stride(-3) != V * K):
        _fb("state pool layout mismatch")
        return None
    h0_stride = initial_state_source.stride(0)
    h0 = initial_state_source  # already (P, HV, V, K)

    # Activations: accept the natural (N*S, HV*dim) 2D views -- reshape
    # keeps strided inputs as zero-copy views (no .contiguous()), and the
    # token stride becomes a compile key (kernelhub-style specialization).
    q2d = q.reshape(Tn, HV * K)
    k2d = k.reshape(Tn, HV * K)
    v2d = v.reshape(Tn, HV * V)
    a2d = a.reshape(Tn, HV * K)
    b2d = b.reshape(Tn, HV)
    for t2d in (q2d, k2d, v2d, a2d, b2d):
        if t2d.stride(-1) != 1:
            _fb("activation inner dim not dense")
            return None
    A32 = A_log.reshape(-1).to(torch.float32).contiguous()
    # dt_bias is consumed in fp32 (same precision as the Triton path); a
    # non-fp32 tensor falls back earlier in the shape gate below.
    dtb = dt_bias.reshape(HV, K).to(torch.float32).contiguous()
    idx = initial_state_indices.to(torch.int32).contiguous()

    kern = _compiled_capture_safe(N, S, HV, K, V, P, PR, W, (
        a2d.stride(0), b2d.stride(0), q2d.stride(0), k2d.stride(0),
        v2d.stride(0), h0_stride, ring_strides["rv_slot"],
        ring_strides["rk_slot"], ring_strides["rg_slot"],
        ring_strides["rb_slot"],
    ), pool_dtype, defer)
    if kern is None:
        return None

    o = torch.empty(Tn, HV, V, dtype=torch.bfloat16, device=q.device)
    pend = replayssm_pending if defer else _dummy_pending(PR, q.device)

    kern(A32, dtb, a2d, b2d, q2d, k2d, v2d, o, h0, idx,
         replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta, pend)
    return o.view(q.shape[0], Tn, HV, V)


@functools.lru_cache(maxsize=8)
def _dummy_pending(PR, device):
    # Never read by the non-deferred kernel; only satisfies the signature.
    return torch.zeros(PR, dtype=torch.int32, device=device)
