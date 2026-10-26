"""Exercise actual KVServe GPU/HIP encode/decode before launching PD servers."""

import argparse
import importlib
import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--mode", default="raw", help="raw, default, or @profile.json")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--reserved-mb", type=int, default=1024)
    parser.add_argument("--max-relative-l2", type=float, default=0.5)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A working DCU/HIP or CUDA PyTorch environment is required")
    from sglang.srt.disaggregation.kvserve_config import resolve_kvserve_config
    from sglang.srt.disaggregation.mooncake.kvserve_adapter import KVServeAdapter

    cfg = json.loads(args.model_config.read_text(encoding="utf-8"))
    heads = max(1, cfg["num_key_value_heads"] // args.tp_size)
    dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    layers = cfg.get("layer_types", ["full_attention"] * cfg["num_hidden_layers"])
    full_ids = [i for i, kind in enumerate(layers) if kind == "full_attention"]
    device = torch.device("cuda", torch.cuda.current_device())

    def pool():
        shape = (8 * args.page_size, heads, dim)
        full = SimpleNamespace(
            k_buffer=[
                torch.randn(shape, device=device, dtype=torch.bfloat16)
                for _ in full_ids
            ],
            v_buffer=[
                torch.randn(shape, device=device, dtype=torch.bfloat16)
                for _ in full_ids
            ],
        )
        return SimpleNamespace(
            page_size=args.page_size,
            full_kv_pool=full,
            layers_mapping={layer: (i, False) for i, layer in enumerate(full_ids)},
        )

    extra = resolve_kvserve_config(args.mode)
    producer = KVServeAdapter(
        pool(),
        extra,
        cfg["num_hidden_layers"],
        args.tp_size,
        args.tp_rank,
        args.reserved_mb * 2**20,
    )
    consumer = KVServeAdapter(
        pool(),
        extra,
        cfg["num_hidden_layers"],
        args.tp_size,
        args.tp_rank,
        args.reserved_mb * 2**20,
    )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(),
                "runtime": importlib.import_module(
                    "kvserve_v1.compression.runtime"
                ).__file__,
                "profile": producer.profile,
                "mode": args.mode,
                "full_layers": full_ids,
                "local_kv_heads": heads,
                "head_dim": dim,
                "source_slots": producer.slot_count,
                "destination_slots": consumer.slot_count,
            }
        ),
        flush=True,
    )
    try:
        assert producer.profile == consumer.profile
        for source_slot, tokens in product(
            range(producer.slot_count),
            (args.page_size - 1, args.page_size, args.page_size + 1),
        ):
            destination_slot = (source_slot + 1) % consumer.slot_count
            count = (tokens + args.page_size - 1) // args.page_size
            src, dst = [4, 1][:count], [2, 5][:count]
            source_indices = torch.tensor(
                [p * args.page_size + t for p in src for t in range(args.page_size)],
                device=device,
            )[:tokens]
            destination_indices = torch.tensor(
                [p * args.page_size + t for p in dst for t in range(args.page_size)],
                device=device,
            )[:tokens]
            for part in (0, 1):
                encoded = producer.encode(src, part, tokens, source_slot)
                sizes = {name: encoded[name][1] for name in ("payload", "aux", "tail")}
                sizes["u8_bytes"] = encoded["u8_bytes"]
                for name, source_buf, destination_buf in (
                    (
                        "payload",
                        producer.runtime.slots[source_slot].arena_a,
                        consumer.runtime.slots[destination_slot].arena_a,
                    ),
                    (
                        "aux",
                        producer.runtime.slots[source_slot].aux,
                        consumer.runtime.slots[destination_slot].aux,
                    ),
                    (
                        "tail",
                        producer.runtime.slots[source_slot].raw_tail,
                        consumer.runtime.slots[destination_slot].raw_tail,
                    ),
                ):
                    destination_buf[: sizes[name]].copy_(source_buf[: sizes[name]])
                torch.cuda.synchronize()
                consumer.restore(dst, part, tokens, sizes, destination_slot)
                sources = producer.keys if part == 0 else producer.values
                targets = consumer.keys if part == 0 else consumer.values
                relative_l2, maximum_error = 0.0, 0.0
                for source, target in zip(sources, targets):
                    expected = source[source_indices]
                    actual = target[destination_indices]
                    if not torch.isfinite(actual).all():
                        raise AssertionError("Restored KV contains NaN/Inf")
                    if producer.config.uses_aggregate:
                        assert torch.equal(expected, actual), (
                            "Raw KV roundtrip was not exact"
                        )
                    error = actual.float() - expected.float()
                    relative_l2 = max(
                        relative_l2,
                        (error.norm() / expected.float().norm().clamp_min(1e-8)).item(),
                    )
                    maximum_error = max(maximum_error, error.abs().max().item())
                if relative_l2 > args.max_relative_l2:
                    raise AssertionError(
                        f"KV relative L2 {relative_l2} exceeds smoke threshold"
                    )
                print(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "part": part,
                            "source_slot": source_slot,
                            "destination_slot": destination_slot,
                            "relative_l2": relative_l2,
                            "max_abs_error": maximum_error,
                            "wire_bytes": sum(
                                sizes[name] for name in ("payload", "aux", "tail")
                            ),
                        }
                    ),
                    flush=True,
                )
    finally:
        producer.runtime.shutdown()
        consumer.runtime.shutdown()
    print("Actual runtime roundtrip passed. Model task quality still needs evaluation.")


if __name__ == "__main__":
    main()
