"""Correctness of the TileLang KDA target_verify kernel vs the Triton kernel.

Covers even AND odd S (2..8), pooled state/ring addressing with a shuffled
slot map and a ring window W >= S, in ReplaySSM mode. Skipped when tilelang
is unavailable (e.g. non-DCU/non-GPU CI).
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.kernels.ops.attention.fla.hcu.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.utils.common import get_device
from sglang.test.ci.ci_register import register_cuda_ci

try:
    import tilelang  # noqa: F401

    from sglang.kernels.ops.attention.fla.hcu.kda_tilelang_verify import (
        tilelang_target_verify_or_none,
    )

    _HAS_TILELANG = True
except ImportError:
    _HAS_TILELANG = False

register_cuda_ci(est_time=10, stage="stage-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not (torch.cuda.is_available() or torch.xpu.is_available()),
    "Test requires CUDA or XPU",
)
@unittest.skipIf(not _HAS_TILELANG, "Test requires tilelang")
class TestKDATileLangTargetVerify(unittest.TestCase):
    HV = 64
    K = 128
    V = 128
    N = 8  # sequences (each with S draft tokens)
    P = 32  # state pool size (> N, so slots are sparse)
    W = 8  # ReplaySSM ring window length (>= S)

    def _inputs(self, S: int, seed: int):
        """Build one target_verify case; returns (state, common, make_rings)."""
        dev = get_device()
        HV, K, V, N, P, W = self.HV, self.K, self.V, self.N, self.P, self.W
        g = torch.Generator(device=dev).manual_seed(seed)
        Tn = N * S

        # serving-style broadcast shapes for A_log / dt_bias
        A_log = (torch.randn(HV, generator=g, device=dev) - 4.0).view(1, 1, HV, 1)
        # dt_bias must be fp32: the TileLang kernel consumes it at fp32
        # precision (Triton parity) and gates off on any other dtype.
        dt_bias = torch.randn(
            HV * K, dtype=torch.float32, generator=g, device=dev
        )
        a = torch.randn(Tn, HV * K, dtype=torch.bfloat16, generator=g, device=dev)
        b = torch.randn(Tn, HV, dtype=torch.bfloat16, generator=g, device=dev)
        q = torch.randn(1, Tn, HV, K, dtype=torch.bfloat16, generator=g, device=dev)
        k = torch.randn(1, Tn, HV, K, dtype=torch.bfloat16, generator=g, device=dev)
        v = torch.randn(1, Tn, HV, V, dtype=torch.bfloat16, generator=g, device=dev)
        state = torch.randn(
            P, HV, K, V, dtype=torch.float32, generator=g, device=dev
        )
        # Real pool slots start at 1 (slot 0 is the permanent dummy where
        # PAD_SLOT_ID traffic lands, see MambaPool.free_slots = range(1, n));
        # sample indices accordingly.
        idx = torch.randperm(P - 1, generator=g, device=dev)[:N].to(torch.int32) + 1
        cu = torch.arange(0, Tn + 1, S, dtype=torch.int32, device=dev)

        def make_rings():
            return dict(
                replayssm_rawv=torch.zeros(P, HV, W, V, dtype=torch.bfloat16, device=dev),
                replayssm_rawk=torch.zeros(P, HV, W, K, dtype=torch.bfloat16, device=dev),
                replayssm_g=torch.zeros(P, HV, W, K, dtype=torch.float32, device=dev),
                replayssm_beta=torch.zeros(P, HV, W, dtype=torch.float32, device=dev),
            )

        common = dict(
            A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
            initial_state_indices=idx, cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0,
            softplus_threshold=20.0, is_kda=True, lower_bound=-5.0,
            beta_scale=1.0, disable_state_update=True,
            intermediate_states_buffer=None, intermediate_state_indices=None,
            cache_steps=S, retrieve_parent_token=None,
            cache_replayssm_inputs=True,
        )
        return state, idx, common, make_rings

    def _run(self, S: int, seed: int):
        state, idx, common, make_rings = self._inputs(S, seed)

        st_tr = state.clone()
        r_tr = make_rings()
        o_tr = fused_sigmoid_gating_delta_rule_update(
            initial_state_source=st_tr, **r_tr, **common
        )
        torch.cuda.synchronize()

        st_tl = state.clone()
        r_tl = make_rings()
        with envs.SGLANG_KDA_USE_TILELANG_VERIFY.override(True):
            o_tl = tilelang_target_verify_or_none(
                initial_state_source=st_tl, **r_tl, **common
            )
        torch.cuda.synchronize()
        return o_tr, r_tr, st_tr, o_tl, r_tl, st_tl, state, idx

    def _check(self, S, seed=1000):
        o_tr, r_tr, st_tr, o_tl, r_tl, st_tl, state, idx = self._run(S, seed)
        self.assertIsNotNone(
            o_tl, f"tilelang path returned None (gate rejected) for S={S}"
        )
        o_tr_v = o_tr.reshape(o_tl.shape)
        o_scale = o_tr_v.float().abs().max().item()

        # output within bf16 rounding
        o_err = (o_tl.float() - o_tr_v.float()).abs().max().item()
        self.assertLess(o_err, 0.02 + 0.05 * o_scale, f"o err {o_err} S={S}")
        # raw k/v ring pass-throughs must be bit-exact
        self.assertTrue(
            torch.equal(r_tl["replayssm_rawv"], r_tr["replayssm_rawv"]),
            f"ring_v mismatch S={S}",
        )
        self.assertTrue(
            torch.equal(r_tl["replayssm_rawk"], r_tr["replayssm_rawk"]),
            f"ring_k mismatch S={S}",
        )
        # gates / beta within tight tolerance
        self.assertLess(
            (r_tl["replayssm_g"] - r_tr["replayssm_g"]).abs().max().item(),
            2e-6,
            f"ring_g err S={S}",
        )
        self.assertLess(
            (r_tl["replayssm_beta"] - r_tr["replayssm_beta"]).abs().max().item(),
            1e-6,
            f"ring_beta err S={S}",
        )
        # state is read-only in target_verify
        self.assertTrue(torch.equal(st_tl, state), f"state modified S={S}")
        # untouched pool slots and ring-window tail stay zero
        mask = torch.ones(self.P, dtype=torch.bool, device=state.device)
        mask[idx.long()] = False
        self.assertTrue((r_tl["replayssm_rawv"][mask] == 0).all())
        self.assertTrue((r_tl["replayssm_g"][mask] == 0).all())
        self.assertTrue((r_tl["replayssm_rawv"][idx.long(), :, S:, :] == 0).all())
        self.assertTrue((r_tl["replayssm_g"][idx.long(), :, S:, :] == 0).all())

    def test_odd_S(self):
        for S in (3, 5, 7):
            with self.subTest(S=S):
                self._check(S)

    def test_even_S(self):
        for S in (2, 4, 6, 8):
            with self.subTest(S=S):
                self._check(S)

    def test_window_equals_production_shape(self):
        # Production configures W = speculative_num_draft_tokens (e.g. 6)
        # with S = speculative_num_steps = W - 1. Exercise the W!=8 compile
        # key (the ring tail [S, W) must stay untouched) at that shape.
        self.W = 6
        try:
            self._check(S=5)
        finally:
            self.W = 8

    def test_disabled_returns_none(self):
        # With the flag off (default), the helper must decline so Triton runs.
        self.assertFalse(envs.SGLANG_KDA_USE_TILELANG_VERIFY.get())
        state, _, common, make_rings = self._inputs(S=5, seed=7)
        out = tilelang_target_verify_or_none(
            initial_state_source=state, **make_rings(), **common
        )
        self.assertIsNone(out)

    def test_padding_slot_minus_one(self):
        # CUDA-graph replay pads mamba_indices with PAD_SLOT_ID (-1). The
        # kernel must clamp those rows to the dummy slot 0 with a zeroed
        # initial state (matching Triton's `if idx >= 0`), never read/write
        # out of bounds, and never disturb real slots' rings.
        S, N, seed = 5, 8, 42
        state, idx, common, make_rings = self._inputs(S, seed)
        idx_padded = idx.clone()
        pad_rows = [1, 4, 7]
        idx_padded[pad_rows] = -1
        common = dict(common, initial_state_indices=idx_padded)

        # reference: Triton with the same padded indices
        st_tr = state.clone()
        r_tr = make_rings()
        o_tr = fused_sigmoid_gating_delta_rule_update(
            initial_state_source=st_tr, **r_tr, **common
        )
        torch.cuda.synchronize()

        st_tl = state.clone()
        r_tl = make_rings()
        with envs.SGLANG_KDA_USE_TILELANG_VERIFY.override(True):
            o_tl = tilelang_target_verify_or_none(
                initial_state_source=st_tl, **r_tl, **common
            )
        torch.cuda.synchronize()
        self.assertIsNotNone(o_tl, "tilelang path rejected the padded batch")

        # Triton returns (1, Tn, H, V); tilelang returns (1, Tn, HV, V).
        o_tr_v = o_tr.reshape(o_tl.shape)

        # padded rows: Triton treats -1 as "no initial state" (zeros);
        # the tilelang gate multiply must reproduce that exactly.
        for r in pad_rows:
            o_r_tr = o_tr_v[0, r * S : (r + 1) * S].float()
            o_r_tl = o_tl[0, r * S : (r + 1) * S].float()
            self.assertLess(
                (o_r_tl - o_r_tr).abs().max().item(),
                0.02 + 0.05 * o_r_tr.abs().max().item(),
                f"padded row {r} output mismatch",
            )
        # real (non-padded) rows still match
        real = [i for i in range(N) if i not in pad_rows]
        for r in real:
            o_r_tr = o_tr_v[0, r * S : (r + 1) * S].float()
            o_r_tl = o_tl[0, r * S : (r + 1) * S].float()
            self.assertLess(
                (o_r_tl - o_r_tr).abs().max().item(),
                0.02 + 0.05 * o_r_tr.abs().max().item(),
                f"real row {r} output mismatch",
            )
        # ring writes for padded rows land on the dummy slot 0 (harmless),
        # but real slots' rings must be identical to the Triton run's.
        for name in ("replayssm_rawv", "replayssm_rawk", "replayssm_g",
                     "replayssm_beta"):
            tr_real = r_tr[name].clone()
            tl_real = r_tl[name]
            # The tilelang path routes idx=-1 ring writes to the dummy slot
            # 0 (clamped), while Triton skips them entirely -- so slot 0's
            # ring content legitimately differs. Slot 0 is the permanent
            # dummy (never a real slot: inputs sample from 1..P-1); every
            # other slot must be bit-exact against the Triton run.
            keep = torch.ones(self.P, dtype=torch.bool, device=tl_real.device)
            keep[0] = False
            self.assertTrue(
                torch.equal(tr_real[keep], tl_real[keep]), f"{name} mismatch"
            )
        # state pool must remain untouched
        self.assertTrue(torch.equal(st_tl, state))

    def test_unsupported_shapes_fall_back(self):
        # Negative cases: each must return None (fall back to Triton),
        # exercising the semantic gates in the dispatch helper.
        S, seed = 5, 42
        state, idx, common, make_rings = self._inputs(S, seed)

        with envs.SGLANG_KDA_USE_TILELANG_VERIFY.override(True):
            # S outside [2, 8]
            bad_s = dict(common, cache_steps=9)
            self.assertIsNone(
                tilelang_target_verify_or_none(
                    initial_state_source=state, **make_rings(), **bad_s
                )
            )
            # retrieve_parent_token set (EAGLE tree topk>1 unsupported)
            bad_tree = dict(
                common,
                retrieve_parent_token=torch.zeros(
                    1, device=state.device, dtype=torch.int64
                ),
            )
            self.assertIsNone(
                tilelang_target_verify_or_none(
                    initial_state_source=state, **make_rings(), **bad_tree
                )
            )
            # lower_bound != -5.0
            bad_lb = dict(common, lower_bound=-4.0)
            self.assertIsNone(
                tilelang_target_verify_or_none(
                    initial_state_source=state, **make_rings(), **bad_lb
                )
            )
            # beta_scale != 1.0
            bad_bs = dict(common, beta_scale=0.5)
            self.assertIsNone(
                tilelang_target_verify_or_none(
                    initial_state_source=state, **make_rings(), **bad_bs
                )
            )
            # NOTE (post stride-specialization): a strided ring view with a
            # uniform slot pitch is now a *supported* case (the pitch is a
            # compile key), so the old non-contiguous-ring negative case no
            # longer falls back. The real rejection trigger is a non-dense
            # innermost dim: build (P, HV, W, K) with stride(-1) != 1 by
            # permuting a differently-ordered contiguous tensor -- the shape
            # gate passes but the stride gate must reject.
            rings = make_rings()
            rings["replayssm_g"] = (
                torch.zeros(self.K, self.W, self.HV, self.P,
                            dtype=torch.float32, device=state.device)
                .permute(3, 2, 1, 0)  # (P, HV, W, K) with stride(-1) != 1
            )
            self.assertNotEqual(rings["replayssm_g"].stride(-1), 1)
            self.assertIsNone(
                tilelang_target_verify_or_none(
                    initial_state_source=state, **rings, **common
                )
            )
            # non-fp32 dt_bias (gate precision contract)
            from sglang.kernels.ops.attention.fla.hcu.kda_tilelang_verify import (
                _tilelang_target_verify as _inner,
            )
            rings2 = make_rings()
            common_bf16 = dict(common)
            common_bf16["dt_bias"] = common["dt_bias"].to(torch.bfloat16)
            self.assertIsNone(
                _inner(dt_bias=common_bf16["dt_bias"],
                       q=common["q"], k=common["k"], v=common["v"],
                       a=common["a"], b=common["b"],
                       A_log=common["A_log"],
                       initial_state_source=state,
                       initial_state_indices=idx,
                       cu_seqlens=common["cu_seqlens"],
                       lower_bound=-5.0, beta_scale=1.0,
                       cache_steps=S, retrieve_parent_token=None,
                       cache_replayssm_inputs=True,
                       **rings2)
            )

    def test_fp16_pool(self):
        # fp16 state-pool storage: load as fp16, compute in fp32. The Triton
        # reference consumes the same fp16 pool (it casts on load), so parity
        # is checked end-to-end at fp16 pool precision.
        S, seed = 5, 43
        state, idx, common, make_rings = self._inputs(S, seed)
        state16 = state.to(torch.float16)

        st_tr = state16.clone()
        r_tr = make_rings()
        o_tr = fused_sigmoid_gating_delta_rule_update(
            initial_state_source=st_tr, **r_tr, **common
        )
        torch.cuda.synchronize()

        st_tl = state16.clone()
        r_tl = make_rings()
        with envs.SGLANG_KDA_USE_TILELANG_VERIFY.override(True):
            o_tl = tilelang_target_verify_or_none(
                initial_state_source=st_tl, **r_tl, **common
            )
        torch.cuda.synchronize()
        self.assertIsNotNone(o_tl, "tilelang path rejected the fp16 pool")
        o_tr_v = o_tr.reshape(o_tl.shape)
        scale = o_tr_v.float().abs().max().item()
        err = (o_tl.float() - o_tr_v.float()).abs().max().item()
        # fp16 pool storage loses ~3 decimal digits vs fp32; loosen slightly
        self.assertLess(err, 0.02 + 0.1 * scale, f"fp16 pool o err {err}")
        for name in ("replayssm_rawv", "replayssm_rawk"):
            self.assertTrue(
                torch.equal(r_tl[name], r_tr[name]), f"fp16 pool {name} mismatch"
            )

    def test_dispatch_through_triton_kernel_class(self):
        # Integration: go through HcuKDAKernel.target_verify (the real
        # serving dispatch layer), not just the helper. The helper must
        # actually be consulted (probed via a monkeypatched counter) -- a
        # regression that drops the ring kwargs at the call site would
        # silently fall back to Triton and produce the same output shape.
        import sglang.srt.layers.attention.linear.kernels.kda_hcu as kt_mod
        from sglang.srt.layers.attention.linear.kernels.kda_hcu import (
            HcuKDAKernel,
        )

        kern = HcuKDAKernel.__new__(HcuKDAKernel)  # bypass __init__ deps
        S, seed = 5, 42
        state, idx, common, make_rings = self._inputs(S, seed)
        rings = make_rings()
        rings["cache_replayssm_inputs"] = True  # via **kwargs, like kda_backend

        calls = []
        real_helper = kt_mod.tilelang_target_verify_or_none

        def probe(**kw):
            calls.append(kw)
            return real_helper(**kw)

        kt_mod.tilelang_target_verify_or_none = probe
        try:
            with envs.SGLANG_KDA_USE_TILELANG_VERIFY.override(True):
                out = kern.target_verify(
                    A_log=common["A_log"],
                    dt_bias=common["dt_bias"],
                    q=common["q"], k=common["k"], v=common["v"],
                    a=common["a"], b=common["b"],
                    ssm_states=state,
                    cache_indices=idx,
                    query_start_loc=common["cu_seqlens"],
                    intermediate_states_buffer=None,
                    intermediate_state_indices=None,
                    cache_steps=S,
                    retrieve_parent_token=None,
                    lower_bound=-5.0,
                    beta_scale=1.0,
                    **rings,
                )
        finally:
            kt_mod.tilelang_target_verify_or_none = real_helper
        # the dispatch layer must have consulted the fast path ...
        self.assertEqual(len(calls), 1, "dispatch layer never called the helper")
        self.assertTrue(
            calls[0].get("cache_replayssm_inputs", False),
            "ring kwargs were not forwarded to the fast path",
        )
        # ... and the fast path must have accepted (not declined) ...
        self.assertIsNotNone(out)
        self.assertEqual(tuple(out.shape[-2:]), (self.HV, self.V))
        # ... and actually written the rings (proof the TileLang kernel ran:
        # the Triton fallback would also write them, but combined with the
        # counter + non-None result this pins the whole chain). Only [0, S)
        # is written -- the [S, W) ring tail must stay zero by design.
        self.assertTrue(
            (rings["replayssm_beta"][idx.long(), :, :S] != 0).all(),
            "rings untouched -- neither path executed",
        )
        self.assertTrue(
            (rings["replayssm_beta"][idx.long(), :, S:] == 0).all(),
            "ring tail was written past S",
        )

    def test_deferred_commit_matches_eager(self):
        self._check_deferred_commit_matches_eager(torch.float32)

    def test_deferred_commit_fp16_matches_eager(self):
        # fp16 SSM: the deferred kernel must round its in-register state to the
        # fp16 checkpoint before continuing, otherwise the recurrence diverges
        # from the eager path that reads the rounded checkpoint back next step.
        self._check_deferred_commit_matches_eager(torch.float16)

    def _check_deferred_commit_matches_eager(self, state_dtype):
        # Multi-step: verify(defer) + parked accept lengths must produce the
        # same outputs and (after a final flush) the same checkpoint as
        # verify + eager ReplaySSM commit. Covers a changing batch, -1
        # padding rows and a final explicit flush.
        from sglang.kernels.ops.attention.fla.hcu.kda_replayssm_spec_decode import (
            commit_kda_replayssm_spec_all_layers,
        )
        from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
            set_kda_replayssm_pending,
        )

        S, steps, n_live = 6, 6, 12
        HV, K, V, N, P = self.HV, self.K, self.V, self.N, self.P
        dev = get_device()
        state0, _, _, _ = self._inputs(S, 7)
        state0 = state0.to(state_dtype)
        gen = torch.Generator().manual_seed(7)
        live = torch.randperm(P - 1, generator=gen)[:n_live].to(torch.int32) + 1

        def world():
            return dict(
                st=state0.clone(),
                rawv=torch.zeros(P, HV, S, V, dtype=torch.bfloat16, device=dev),
                rawk=torch.zeros(P, HV, S, K, dtype=torch.bfloat16, device=dev),
                g=torch.zeros(P, HV, S, K, dtype=torch.float32, device=dev),
                beta=torch.zeros(P, HV, S, dtype=torch.float32, device=dev),
                pend=torch.zeros(P, dtype=torch.int32, device=dev),
            )

        def rings(w):
            return dict(
                replayssm_rawv=w["rawv"], replayssm_rawk=w["rawk"],
                replayssm_g=w["g"], replayssm_beta=w["beta"],
            )

        def all_layer(w):
            return dict(
                checkpoint_state=w["st"].unsqueeze(0),
                rawv_cache=w["rawv"].unsqueeze(0),
                rawk_cache=w["rawk"].unsqueeze(0),
                g_cache=w["g"].unsqueeze(0),
                beta_cache=w["beta"].unsqueeze(0),
            )

        ref, dfr = world(), world()
        for step in range(steps):
            _, _, common, _ = self._inputs(S, 100 + step)
            pick = torch.randperm(n_live, generator=gen)[: N - 1]
            idx = torch.cat([live[pick], torch.tensor([-1], dtype=torch.int32)])
            idx = idx.to(dev)
            common = dict(common, initial_state_indices=idx)
            acc = torch.randint(1, S + 1, (N,), generator=gen, dtype=torch.int32)
            acc = acc.to(dev)

            with envs.SGLANG_KDA_USE_TILELANG_VERIFY.override(True):
                o_ref = tilelang_target_verify_or_none(
                    initial_state_source=ref["st"], **rings(ref), **common
                )
                o_dfr = tilelang_target_verify_or_none(
                    initial_state_source=dfr["st"], **rings(dfr),
                    replayssm_pending=dfr["pend"], **common,
                )
            self.assertIsNotNone(o_ref)
            self.assertIsNotNone(o_dfr)
            real = (N - 1) * S
            o_ref_v = o_ref.reshape(-1, HV, V)[:real].float()
            o_dfr_v = o_dfr.reshape(-1, HV, V)[:real].float()
            scale = o_ref_v.abs().max().item()
            err = (o_dfr_v - o_ref_v).abs().max().item()
            self.assertLess(err, 0.02 + 0.02 * scale, f"step {step} o err {err}")

            commit_kda_replayssm_spec_all_layers(
                **all_layer(ref), ssm_state_indices=idx, accept_lens=acc
            )
            set_kda_replayssm_pending(dfr["pend"], idx, acc)

        commit_kda_replayssm_spec_all_layers(
            **all_layer(dfr),
            ssm_state_indices=torch.arange(P, device=dev),
            accept_lens=dfr["pend"],
        )
        dfr["pend"].zero_()
        torch.cuda.synchronize()
        self.assertEqual(int(dfr["pend"].abs().sum()), 0)
        if state_dtype == torch.float16:
            # Defer and eager both store fp16 checkpoints; allow one fp16 ulp
            # of fold-arithmetic difference between the two kernels.
            self.assertTrue(
                torch.allclose(dfr["st"], ref["st"], rtol=1e-2, atol=1e-3),
                "fp16 checkpoint mismatch: "
                f"max={(dfr['st'] - ref['st']).abs().max().item()}",
            )
        else:
            scale = ref["st"].abs().max().item()
            err = (dfr["st"] - ref["st"]).abs().max().item()
            self.assertLess(err, 1e-4 * scale, f"checkpoint err {err}")


if __name__ == "__main__":
    unittest.main()
