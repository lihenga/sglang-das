"""Opt-in D-only shared-expert stream for TokenFlow's existing DeepEP path.

Do not enable SBO/communication scheduling or fuse shared experts into routing.
The plugin owns wait_stream, record_stream and the post-expert wait_event.
"""
import logging
import torch
from sglang.srt.utils import get_bool_env_var, is_hcu

logger = logging.getLogger(__name__)


def install_shared_expert_overlap():
    if not (is_hcu() and get_bool_env_var("SGLANG_PERF_SHARED_EXPERT_OVERLAP", "false")):
        return False
    import tokenflow.models.sparkv1.layers.moe as spark_moe
    from sglang.srt.server_args import get_global_server_args
    cls = spark_moe.SparkV1MoE
    original = cls.__init__
    if getattr(original, "_sglang_perf_shared_overlap", False):
        return True
    stream = None
    original_forward = cls.forward_deepep

    def initialize(self, *args, **kwargs):
        nonlocal stream
        original(self, *args, **kwargs)
        server = get_global_server_args()
        if (server.disaggregation_mode != "decode" or server.dp_size != 16
                or server.tp_size != 16 or server.ep_size != 16
                or server.deepep_mode != "low_latency"):
            raise RuntimeError("Shared overlap experiment requires D DP16/EP16 DeepEP LL")
        if self.num_fused_shared_experts != 0 or self._fuse_shared_experts_inside_sbo:
            raise RuntimeError("Shared stream must not change physical shared routing or enable SBO")
        shared = getattr(self, "shared_experts", None)
        if shared is None:
            raise RuntimeError("Expected independent TokenFlow shared experts")
        if any(getattr(getattr(shared, name, None), "tp_size", None) != 1
               for name in ("fc1", "fc2")):
            raise RuntimeError("Shared stream experiment requires local TP1 shared GEMMs")
        if stream is None:
            stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.alt_stream = stream
        self._sglang_perf_shared_stream = stream
        logger.info("PERF shared-expert overlap enabled: layer=%s stream=%s; native TokenFlow waits, no SBO",
                    getattr(self, "layer_id", "?"), stream.cuda_stream)

    initialize._sglang_perf_shared_overlap = True
    initialize._sglang_original = original
    cls.__init__ = initialize
    def forward(self, *args, **kwargs):
        expected = self._sglang_perf_shared_stream
        actual = self.alt_stream
        if not getattr(self, "_sglang_perf_shared_forward_seen", False):
            logger.info("PERF shared stream first forward: layer=%s constructor_stream=%s actual_stream=%s",
                        getattr(self, "layer_id", "?"), expected.cuda_stream,
                        getattr(actual, "cuda_stream", None))
            self._sglang_perf_shared_forward_seen = True
        # A containing decoder can assign its own alt_stream after constructing
        # its MoE. The opt-in stream must survive that parent initialization.
        if actual is not expected:
            self.alt_stream = expected
        return original_forward(self, *args, **kwargs)
    forward._sglang_original = original_forward
    cls.forward_deepep = forward
    return True
