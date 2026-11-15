"""Opt-in bounded event tracing for the reserved-node experiment only."""

import json
import os
import time


def make_trace(rank, cache=None):
    folder = os.environ.get("SGLANG_L3_EXPERIMENT_TRACE_DIR")
    if not folder:
        return None
    os.makedirs(folder, exist_ok=True)
    fd = os.open(
        os.path.join(folder, f"cp{rank}-pid{os.getpid()}.jsonl"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )

    def write(event, **fields):
        try:
            _write(event, **fields)
        except Exception:
            # Observability must not change the storage/control outcome.
            return

    def _write(event, **fields):
        row = {"time": time.time(), "cp_rank": rank, "event": event, **fields}
        if cache is not None:
            tree = getattr(cache, "tree_core", None)
            for name in ["component_protected_size_", "component_evictable_size_"]:
                value = getattr(tree, name, None)
                if value is not None:
                    row[name] = {str(k): int(v) for k, v in value.items()}
            allocator = getattr(cache, "token_to_kv_pool_allocator", None)
            if allocator is not None:
                row["free_tokens"] = int(allocator.available_size())
        os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())

    return write


def emit(owner, event, **fields):
    trace = getattr(owner, "_experiment_trace", None)
    if trace is not None:
        trace(event, **fields)
