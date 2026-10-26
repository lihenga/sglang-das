"""Configuration for the experimental homogeneous-TP KVServe PD adapter.

Keep this module free of accelerator/KVServe imports so disabled deployments
retain their existing dependency and startup behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def resolve_kvserve_config(value: str | None) -> dict | None:
    if value is None:
        return None
    if value == "default":
        return {
            "compression": {
                "enable-compression-pipeline": True,
                "logical_blocks_per_chunk": 64,
            }
        }
    if value == "raw":
        return {"raw_aggregate": {"logical_blocks_per_chunk": 64}}
    if value.startswith("@"):
        value = Path(value[1:]).read_text(encoding="utf-8")
    config = json.loads(value)
    if not isinstance(config, dict):
        raise TypeError("--kvserve-compression expects default, raw, or a JSON object")
    if set(config) - {"compression", "raw_aggregate"}:
        raise ValueError("KVServe JSON must contain compression or raw_aggregate")
    if len(config) != 1:
        raise ValueError("Select exactly one KVServe compression or raw_aggregate mode")
    body = next(iter(config.values()))
    if not isinstance(body, dict):
        raise TypeError("KVServe mode configuration must be a JSON object")
    if "compression" in config and body.get("enable-compression-pipeline") is not True:
        raise ValueError("KVServe compression JSON must explicitly enable the pipeline")
    pages = body.get("logical_blocks_per_chunk", 1)
    slots = body.get("slot_count", 2)
    if type(pages) is not int or not 1 <= pages <= 256:
        raise ValueError("KVServe logical_blocks_per_chunk must be in [1, 256]")
    if type(slots) is not int or not 1 <= slots <= 4:
        raise ValueError("KVServe slot_count must be in [1, 4]")
    return config


def validate_kvserve_launch(args) -> None:
    if getattr(args, "kvserve_compression", None) is None:
        return
    resolve_kvserve_config(args.kvserve_compression)
    if args.disaggregation_mode not in ("prefill", "decode"):
        raise ValueError("KVServe requires --disaggregation-mode prefill or decode")
    if args.disaggregation_transfer_backend != "mooncake":
        raise ValueError("KVServe v1 requires the mooncake PD backend")
    for name in ("pp_size", "dp_size", "dcp_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(
                f"KVServe v1 requires {name}=1 (homogeneous TP is supported)"
            )
    for name in (
        "enable_dp_attention",
        "enable_prefill_cp",
        "enable_hisparse",
        "enable_unified_memory",
        "enable_page_major_kv_layout",
        "disaggregation_decode_enable_radix_cache",
        "disaggregation_decode_enable_offload_kvcache",
        "disaggregation_decode_retraction_rebootstrap",
        "speculative_algorithm",
    ):
        if getattr(args, name, None):
            raise ValueError(f"KVServe v1 smoke tests do not support {name}")
    if args.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise ValueError("KVServe requires BF16 KV cache")
    if args.kvserve_reserved_mb < 1:
        raise ValueError("--kvserve-reserved-mb must be positive")
    if not math.isfinite(args.kvserve_rpc_timeout) or args.kvserve_rpc_timeout <= 0:
        raise ValueError("--kvserve-rpc-timeout must be finite and positive")
