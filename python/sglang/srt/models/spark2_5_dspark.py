"""Spark2.5 backbone adapter for SGLang's DSpark runtime."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dspark import DSparkDraftModel, VanillaMarkov
from sglang.srt.models.spark2_5 import (
    Spark2_5Attention,
    Spark2_5MLP,
    _get_attention_sliding_window_size,
)
from sglang.srt.platforms import current_platform


class Spark2_5DSparkAttention(Spark2_5Attention):
    """Spark attention with the KV-only operations required by DSpark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn.attn_type = AttentionType.ENCODER_ONLY

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.q_k_v_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_rope(
        self, positions: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        dummy_q = torch.empty_like(k)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class Spark2_5DSparkDecoderLayer(nn.Module):
    """Spark2.5 decoder layer with the constructor expected by DSpark."""

    def __init__(
        self,
        config,
        layer_id: int,
        attention_conv=None,
        mlp_conv=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if attention_conv is not None or mlp_conv is not None:
            raise ValueError("Spark2.5 DSpark does not support draft convolutions")

        layer_type = config.layer_types[layer_id]
        rope_params = config.rope_parameters[layer_type]
        attention_prefix = f"{prefix}.self_attn" if prefix else "self_attn"
        self.self_attn = Spark2_5DSparkAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
            rope_theta=rope_params["rope_theta"],
            partial_rotary_factor=rope_params.get("partial_rotary_factor", 1.0),
            max_position_embeddings=getattr(
                config, "max_position_embeddings", 8192
            ),
            quant_config=quant_config,
            sliding_window_size=(
                _get_attention_sliding_window_size(config)
                if layer_type == "sliding_attention"
                else -1
            ),
            headwise_attn_output_gate=getattr(
                config, "headwise_attn_output_gate", True
            ),
            prefix=attention_prefix,
        )
        mlp_prefix = f"{prefix}.mlp" if prefix else "mlp"
        self.mlp = Spark2_5MLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=mlp_prefix,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SparkX25DesparkDraftModel(DSparkDraftModel):
    """DSpark draft model whose transformer layers follow Spark2.5."""

    decoder_layer_cls = Spark2_5DSparkDecoderLayer
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.num_stages = int(config.num_hidden_layers)
        target_hidden_size = int(
            getattr(config, "target_hidden_size", config.hidden_size)
        )
        if target_hidden_size != int(config.hidden_size):
            raise ValueError(
                "Spark2.5 DSpark currently requires target_hidden_size == "
                f"hidden_size, got {target_hidden_size} and {config.hidden_size}."
            )
        self.markov_head = VanillaMarkov(
            vocab_size=int(getattr(config, "draft_vocab_size", config.vocab_size)),
            markov_rank=int(config.markov_rank),
        )
        self.fc_norm = nn.ModuleList(
            [
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                for _ in range(self.num_context_features)
            ]
        )

    def get_attention_sliding_window_size(self) -> int:
        return _get_attention_sliding_window_size(self.config)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        target_hidden_size = int(
            getattr(self.config, "target_hidden_size", self.config.hidden_size)
        )
        expected = self.num_context_features * target_hidden_size
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(
                "Spark2.5 DSpark target hidden shape mismatch: expected "
                f"[N, {expected}], got {tuple(target_hidden.shape)}."
            )
        chunks = target_hidden.split(target_hidden_size, dim=-1)
        normalized = torch.cat(
            [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)], dim=-1
        )
        return self.hidden_norm(F.linear(normalized, self.fc.weight))

    def load_weights(self, weights):
        """Load only the ``draft_model.*`` partition of a bundled checkpoint."""

        def draft_weights():
            for name, loaded_weight in weights:
                if name.startswith("draft_model."):
                    yield name.removeprefix("draft_model."), loaded_weight

        super().load_weights(draft_weights())

    def _prepare_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if cache_loc_2d is None or commit_lens is None:
            return target_hidden, positions, cache_loc

        expected_rows = cache_loc_2d.numel()
        if (
            target_hidden.shape[0] != expected_rows
            or positions.numel() != expected_rows
        ):
            raise ValueError(
                "Spark2.5 DSpark commit metadata must match target hidden rows: "
                f"hidden={target_hidden.shape[0]}, positions={positions.numel()}, "
                f"cache_loc_2d={expected_rows}."
            )

        row_width = cache_loc_2d.shape[1]
        device = target_hidden.device
        row_offsets = torch.arange(row_width, device=device)
        valid_mask = row_offsets.view(1, -1) < commit_lens.to(
            device=device, dtype=torch.long
        ).view(-1, 1)
        valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return None

        flat_locs = cache_loc_2d.reshape(-1)
        if flat_locs.device != device:
            flat_locs = flat_locs.to(device=device, non_blocking=True)
        return (
            target_hidden.index_select(0, valid_idx),
            positions.index_select(0, valid_idx),
            flat_locs.index_select(0, valid_idx),
        )

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Materialize Spark2.5 draft KV with hybrid-SWA locations."""
        prepared = self._prepare_target_hidden_kv(
            target_hidden=target_hidden,
            positions=positions,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )
        if prepared is None:
            return
        target_hidden, positions, cache_loc = prepared
        ctx_hidden = self.project_target_hidden(target_hidden)

        if hasattr(pool, "translate_loc_from_full_to_swa"):
            write_loc = KVWriteLoc(
                cache_loc,
                swa_loc=pool.translate_loc_from_full_to_swa(cache_loc),
            )
        else:
            write_loc = cache_loc

        for layer in self.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_rope(positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)

            pool.set_kv_buffer(
                attn.attn,
                write_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.out_proj = None
        current_platform.empty_cache()


EntryClass = [SparkX25DesparkDraftModel]
