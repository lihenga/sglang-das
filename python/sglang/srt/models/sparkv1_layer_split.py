"""Connect TokenFlow Spark attention to the DSA LayerSplit prefetch path."""

import functools
import logging

from sglang.srt.utils import is_hcu

logger = logging.getLogger(__name__)


def install_spark_v1_layer_split_prefetch():
    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    if not (
        is_hcu()
        and server_args.enable_dsa_cache_layer_split
        and server_args.disaggregation_mode == "prefill"
    ):
        return False

    from tokenflow.models.sparkv1.layers.attention import SparkV1AttentionMLA

    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.communicator_dsa_cp import (
        maybe_configure_main_kv_page_plan,
        maybe_prefetch_full_attention_kv,
        maybe_prefetch_next_full_attention_kv,
    )
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    original = SparkV1AttentionMLA.forward
    if getattr(original, "_spark_layer_split_prefetch", False):
        return True

    logged = False

    @functools.wraps(original)
    def forward(self, positions, hidden_states, forward_batch, *args, **kwargs):
        nonlocal logged
        # TBO alternates scratch ownership between child batches. Keep its
        # existing path, as well as speculative/decode forwards, unchanged.
        eligible = (
            dsa_use_prefill_cp(forward_batch)
            and forward_batch.forward_mode.is_extend_without_speculative()
            and not forward_batch.can_run_tbo
            and forward_batch.tbo_parent_token_range is None
        )
        pool = get_token_to_kv_pool()
        if not getattr(pool, "layer_shard_enabled", False):
            return original(
                self, positions, hidden_states, forward_batch, *args, **kwargs
            )

        if not eligible:
            # A preceding ordinary extend may have left a compact mapping.
            # Restore the full-pool layout before a TBO/speculative fallback.
            maybe_configure_main_kv_page_plan(forward_batch)
            return original(
                self, positions, hidden_states, forward_batch, *args, **kwargs
            )

        layer_id = self.layer_id
        end_layer = pool.start_layer + pool.layer_num
        if not pool.start_layer <= layer_id < end_layer:
            return original(
                self, positions, hidden_states, forward_batch, *args, **kwargs
            )

        # Spark has its own Decoder/Model rather than DeepseekV2's forward.
        # Install the compact page plan before its indexer can read the cache.
        # Later layers reuse the broadcast queued after the previous attention.
        maybe_prefetch_full_attention_kv(forward_batch, layer_id)
        output = original(
            self, positions, hidden_states, forward_batch, *args, **kwargs
        )
        if layer_id + 1 < end_layer:
            # Attention has finished using shared scratch. The helper orders
            # Index-K then Main-KV on the side stream, overlapping Spark's MLP.
            # Restrict lookahead to this PP stage, including skip-topk layers.
            maybe_prefetch_next_full_attention_kv(forward_batch, layer_id + 1)
        if not logged:
            logger.info(
                "Spark LayerSplit prefetch active: layers=[%s,%s), "
                "compact Main-KV history, next layer before MLP",
                pool.start_layer,
                end_layer,
            )
            logged = True
        return output

    forward._spark_layer_split_prefetch = True
    SparkV1AttentionMLA.forward = forward
    return True
