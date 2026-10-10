"""Default-on, RPC-free correlation for one KV-cache diagnostic run.

No launch changes needed. MOONCAKE_KV_SESSION_TRACE=0 optionally disables it.
Only IDs, hashes and lifecycle metadata are logged; never keys or tensors.
"""

import functools
import itertools
import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager

ENABLED = os.getenv("MOONCAKE_KV_SESSION_TRACE", "1") != "0"
_logger = logging.getLogger(__name__)
_local = threading.local()
_sequence = itertools.count(1)
_host = socket.gethostname()


def key_hash(key):
    """FNV-1a/64, identical to Mooncake's KVSessionKeyHash."""
    value = 14695981039346656037
    for byte in key.encode("utf-8"):
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def event(owner, name, rid=None, **fields):
    if not ENABLED:
        return
    record = dict(
        event=name,
        trace_id=getattr(_local, "trace_id", "none"),
        rid=rid,
        host=_host,
        linker=id(owner),
        pid=os.getpid(),
        thread=threading.get_ident(),
        rank=getattr(owner, "tp_rank", None),
        wall_us=time.time_ns() // 1000,
        mono_us=time.monotonic_ns() // 1000,
    )
    record.update(fields)
    _logger.info("KV_TRACE %s", json.dumps(record, separators=(",", ":")))


@contextmanager
def scope(owner, name, rid=None, **fields):
    if not ENABLED:
        yield {}
        return
    previous = getattr(_local, "trace_id", "none")
    trace_id = f"{_host}:{os.getpid()}:{next(_sequence)}"
    _local.trace_id = trace_id
    store = getattr(getattr(owner, "storage", owner), "store", None)
    setter = getattr(store, "set_session_trace_context", None)
    native_previous = setter(trace_id) if setter is not None else None
    started = time.monotonic_ns()
    result = {}
    event(
        owner,
        name + ".begin",
        rid,
        parent_trace=previous,
        native_bridge=setter is not None,
        **fields,
    )
    try:
        yield result
    except BaseException as exc:
        result["exception_type"] = type(exc).__name__
        raise
    finally:
        event(
            owner,
            name + ".end",
            rid,
            elapsed_us=(time.monotonic_ns() - started) // 1000,
            **result,
        )
        if setter is not None:
            setter(native_previous)
        _local.trace_id = previous


def traced(name, rid_arg=None, batch_arg=None):
    """Wrap existing calls without changing their return/exception behavior."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(self, *args, **kwargs):
            if not ENABLED:
                return function(self, *args, **kwargs)
            owner = getattr(self, "cache_linker", self)
            rid = None
            if rid_arg is not None:
                value = (
                    args[rid_arg]
                    if len(args) > rid_arg
                    else kwargs.get("rid", kwargs.get("req"))
                )
                rid = getattr(value, "rid", value)
            fields = {}
            if batch_arg is not None:
                batch = (
                    args[batch_arg]
                    if len(args) > batch_arg
                    else kwargs["request_transfers"]
                )
                fields["rids"] = [item[0] for item in batch]
                fields["load_batch"] = args[0] if args else kwargs.get("counter_index")
            with scope(owner, name, rid, **fields) as info:
                value = function(self, *args, **kwargs)
                if isinstance(value, (bool, int, str)) or value is None:
                    info["result"] = value
                return value

        return wrapped

    return decorate


def keys(owner, name, rid, key_list):
    if not ENABLED:
        return
    # Full membership is needed to find ALL holders of a failed sampled key.
    # Chunk records to avoid log-line truncation; no raw keys are emitted.
    refs = owner.session_refcounts
    for offset in range(0, len(key_list), 128):
        event(
            owner,
            name,
            rid,
            offset=offset,
            total=len(key_list),
            hash_refs=[
                [key_hash(key), refs.get(key, 0)]
                for key in key_list[offset : offset + 128]
            ],
        )
