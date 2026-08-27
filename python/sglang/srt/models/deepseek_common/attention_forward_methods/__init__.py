import os

from .forward_methods import AttnForwardMethod
from .forward_mha import DeepseekMHAForwardMixin
from .forward_mha_rocm import DeepseekMHARocmForwardMixin
from .forward_mla import DeepseekMLAForwardMixin
from .forward_mla_fused_rope_cpu import DeepseekMLACpuForwardMixin
from .forward_mla_fused_rope_rocm import DeepseekMLAFusedRopeRocmForwardMixin
from .forward_mla_rocm import DeepseekMLARocmForwardMixin

if os.getenv("SGLANG_TOKENFLOW_LEGACY_COMPAT") == "1":
    # In SGLang 0.5.12 this public class name referred to the fused-rope ROCm
    # mixin. Main split it into two mixins, while TokenFlow 0.2.2b0 still
    # inherits only the old name. Restore its missing method surface without
    # changing the class identity (current models inherit both new mixins).
    for _name, _value in DeepseekMLAFusedRopeRocmForwardMixin.__dict__.items():
        if not _name.startswith("_") and not hasattr(
            DeepseekMLARocmForwardMixin, _name
        ):
            setattr(DeepseekMLARocmForwardMixin, _name, _value)

    # TokenFlow's SparkV1 attention object still stores the pre-rename
    # ``use_nsa`` flag, while current DSA forward mixins read ``use_dsa``.
    # A property keeps both names on the same underlying boolean.
    if not hasattr(DeepseekMHAForwardMixin, "use_dsa"):

        def _get_use_dsa(self):
            return self.use_nsa

        def _set_use_dsa(self, value):
            self.use_nsa = value

        setattr(
            DeepseekMHAForwardMixin,
            "use_dsa",
            property(_get_use_dsa, _set_use_dsa),
        )

    # Main routes MLA Q projection through a wrapper that can select newer
    # fused GEMM paths. TokenFlow's older attention class has the same q_b_proj
    # module and dimensions but not the wrapper; retain its original projection
    # semantics for the Torch/Triton compatibility baseline.
    if not hasattr(DeepseekMLAForwardMixin, "q_b_proj_forward"):

        def _q_b_proj_forward(self, q_lora):
            q = self.q_b_proj(q_lora)[0]
            return q.view(-1, self.num_local_heads, self.qk_head_dim)

        setattr(
            DeepseekMLAForwardMixin,
            "q_b_proj_forward",
            _q_b_proj_forward,
        )

    # Current MLA preparation probes these DCP-replication weights even though
    # they stay unused for this TP8/DCP1 launch.  Match current mixin defaults
    # so the legacy TokenFlow subclass exposes the complete attribute surface.
    if not hasattr(DeepseekMLAForwardMixin, "w_kc_qrep"):
        setattr(DeepseekMLAForwardMixin, "w_kc_qrep", None)
    if not hasattr(DeepseekMLAForwardMixin, "q_b_proj_qrep_weight"):
        setattr(DeepseekMLAForwardMixin, "q_b_proj_qrep_weight", None)

    # The legacy Spark constructor accepts ``is_nextn`` but does not retain it.
    # This non-speculative baseline needs the same default as current MLA
    # attention when deciding whether shared-layer indexers should run.
    if not hasattr(DeepseekMLAForwardMixin, "is_nextn"):
        setattr(DeepseekMLAForwardMixin, "is_nextn", False)

    # TokenFlow 0.2.2b0's Cython wrapper unpacks the legacy nine-item prepare
    # tuple.  Main added ``fusion_plan`` as a tenth, optional optimization
    # value.  This baseline disables graph/fused paths, so it must remain None;
    # assert that invariant before restoring the old ABI to avoid silent math
    # changes if the launch configuration changes later.
    _current_forward_absorb_prepare = DeepseekMLAForwardMixin.forward_absorb_prepare
    if not getattr(
        _current_forward_absorb_prepare,
        "_sglang_tokenflow_legacy_compat",
        False,
    ):

        def _wrap_forward_absorb_prepare(original):
            def _legacy_forward_absorb_prepare(
                self,
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                llama_4_scaling=None,
                prev_topk_indices=None,
            ):
                prepared = original(
                    self,
                    positions,
                    hidden_states,
                    forward_batch,
                    zero_allocator,
                    llama_4_scaling,
                    prev_topk_indices,
                )
                if (
                    self.__class__.__module__.startswith("tokenflow.models.sparkv1")
                    and isinstance(prepared, tuple)
                    and len(prepared) == 10
                ):
                    if prepared[-1] is not None:
                        raise RuntimeError(
                            "Legacy TokenFlow cannot consume a non-empty MLA "
                            "fusion plan. Disable fused MLA preparation."
                        )
                    return prepared[:-1]
                return prepared

            _legacy_forward_absorb_prepare._sglang_tokenflow_legacy_compat = True
            return _legacy_forward_absorb_prepare

        setattr(
            DeepseekMLAForwardMixin,
            "forward_absorb_prepare",
            _wrap_forward_absorb_prepare(_current_forward_absorb_prepare),
        )

__all__ = [
    "AttnForwardMethod",
    "DeepseekMHAForwardMixin",
    "DeepseekMHARocmForwardMixin",
    "DeepseekMLACpuForwardMixin",
    "DeepseekMLAForwardMixin",
    "DeepseekMLAFusedRopeRocmForwardMixin",
    "DeepseekMLARocmForwardMixin",
]
