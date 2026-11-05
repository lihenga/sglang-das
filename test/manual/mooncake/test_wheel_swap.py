"""Real-wheel Store and direct-linker round trips, using a matching test Master.

MOONCAKE_MASTER=127.0.0.1:PORT MATRIX_CALLER=new|provided python test_wheel_swap.py
Run the same file and settings in environments differing only in Mooncake wheel.
Pool assembly and runtime identity are supplied by this small fixture; native
setup, registrations, writes, reads, counters and worker cleanup are real.
"""

import json
import os
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import mooncake.store as native
import sglang
import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker as module
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


class RecordedStore:
    def __init__(self, store):
        self.native = store
        self.calls = Counter()

    def __getattr__(self, name):
        value = getattr(self.native, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            self.calls[name] += 1
            return value(*args, **kwargs)

        return invoke


def run_case(device, mode):
    os.environ["SGLANG_MOONCAKE_READ_PLAN"] = "0" if mode == "default" else "1"
    os.environ["SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES"] = (
        "1" if mode == "plan" else "0"
    )
    protocol = os.environ.get("MATRIX_PROTOCOL", "tcp")
    if protocol == "tcp":
        os.environ["MC_FORCE_TCP"] = "1"
    else:
        os.environ.pop("MC_FORCE_TCP", None)
    caller = os.environ["MATRIX_CALLER"]
    rows = torch.tensor([3, 1, 4])
    keys = ["wheel-swap-" + uuid4().hex for _ in range(3)]
    entries = []
    for name, width, count in ((PoolName.KV, 4, 2), (PoolName.INDEXER, 3, 1)):
        components = [
            [
                (
                    torch.arange(5 * width, dtype=torch.uint8).reshape(5, width)
                    + 40 * layer
                    + 70 * c
                ).to(device)
                for layer in range(2)
            ]
            for c in range(count)
        ]
        entries.append(
            DevicePoolEntry(
                name=name,
                indices_from_pool=name,
                device_pool=None,
                components=components,
                layer_mapping={0: 0, 1: 1},
                page_size=1,
                rows_are_pages=True,
                packed=True,
            )
        )
    group = DevicePoolGroup(entries, num_layers=2, page_size=1, rank_replicated=True)
    original = [buffer.cpu().clone() for pool in entries for buffer in pool.kv_buffer]
    if device == "cuda":
        torch.cuda.synchronize()
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="wheel-swap",
        extra_config={
            "master_server_address": os.environ["MOONCAKE_MASTER"],
            "metadata_server": "P2PHANDSHAKE",
            "local_hostname": "127.0.0.1:0",
            "global_segment_size": 64 * 1024**2,
            "protocol": protocol,
            "device_name": os.environ.get("MATRIX_DEVICE", ""),
            "check_server": False,
        },
    )
    storage = MooncakeStore(config)
    recorded = RecordedStore(storage.store)
    storage.store = recorded
    args = SimpleNamespace(
        tp_size=1,
        schedule_policy="fcfs",
        disaggregation_mode="prefill",
        mooncake_enable_waiting_queue_dfs_prefetch=True,
        mooncake_page_wise_load_threshold=1,
        mooncake_enable_page_wise_load=mode == "pagewise",
    )
    params = SimpleNamespace(
        page_size=1,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: group),
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        tp_cache_group=None,
        pp_size=1,
        pp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        dp_rank=0,
        enable_metrics=False,
    )
    linker = None
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    module, "resolve_hybrid_device_pool_group", return_value=group
                )
            )
            stack.enter_context(
                patch.object(
                    module.HybridCacheController,
                    "parse_storage_backend_extra_config",
                    return_value=({},),
                )
            )
            stack.enter_context(
                patch.object(
                    module,
                    "get_memory",
                    return_value=SimpleNamespace(
                        hicache_storage_backend_extra_config=None
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    module,
                    "get_model",
                    return_value=SimpleNamespace(model_path="wheel-swap"),
                )
            )
            if hasattr(module, "get_parallel"):
                stack.enter_context(
                    patch.object(
                        module, "get_parallel", return_value=SimpleNamespace(tp_size=1)
                    )
                )
            linker = module.MooncakeDirectLinker(
                args, params, components=set(), storage=storage
            )
        transfers = [
            PoolTransfer(name=entry.name, keys=keys, host_indices=rows)
            for entry in entries
        ]
        result = storage.batch_set_v2(transfers)
        assert all(all(values) for values in result.values()), result
        writes = dict(recorded.calls)
        for pool in entries:
            for buffer in pool.kv_buffer:
                buffer.zero_()
        if device == "cuda":
            torch.cuda.synchronize()
        recorded.calls.clear()
        index = linker.layer_done_counter.update_producer()
        batches = [
            [
                PoolTransfer(name=entry.name, keys=keys[:1], host_indices=rows[:1])
                for entry in entries
            ],
            [
                PoolTransfer(name=entry.name, keys=keys[1:], host_indices=rows[1:])
                for entry in entries
            ],
        ]
        returned = linker.load_layer_wise(index, batches)
        assert returned is not False
        counter = linker.layer_done_counter
        if hasattr(counter, "futures"):
            for future in counter.futures[index]:
                assert future.exception(timeout=10) is None
        else:
            plan = counter.plans[index].result(timeout=10)
            for layer in range(2):
                plan.wait(layer)
        linker.layer_done_counter.set_consumer(index)
        for layer in range(2):
            linker.layer_done_counter.wait_until(layer)
        if hasattr(counter, "finish"):
            assert counter.finish(index)
        for before, after in zip(
            original, [b for entry in entries for b in entry.kv_buffer]
        ):
            expected = torch.zeros_like(before)
            expected[rows] = before[rows]
            assert torch.equal(after.cpu(), expected), (
                device,
                mode,
                before,
                after.cpu(),
            )
        calls = dict(recorded.calls)
        if caller == "new":
            if getattr(linker, "read_plan_enabled", False):
                expected_method = "create_read_plan"
                assert mode != "default"
            elif mode == "pagewise":
                expected_method = "batch_get_into_multi_buffer_ranges"
            else:
                expected_method = "batch_get_into_layouts"
            assert calls.get(expected_method, 0) > 0, (expected_method, calls)
            if expected_method != "batch_get_into_multi_buffer_ranges":
                assert calls.get("batch_get_into_multi_buffer_ranges", 0) == 0, calls
        else:
            assert calls.get("batch_get_into_multi_buffer_ranges", 0) == 4, calls
        print(
            "VERIFIED_WHEEL_SWAP",
            json.dumps(
                {
                    "caller": caller,
                    "device": device,
                    "mode": mode,
                    "protocol": protocol,
                    "sglang": sglang.__file__,
                    "mooncake": native.__file__,
                    "write_calls": writes,
                    "read_calls": calls,
                    "read_plan": getattr(linker, "read_plan_enabled", False),
                    "read_layout": getattr(linker, "read_layout_enabled", False),
                }
            ),
            flush=True,
        )
    finally:
        if linker is not None:
            linker.close()
        else:
            storage.close()
        recorded.native.close()


if __name__ == "__main__":
    for device in os.environ.get("MATRIX_DEVICES", "cpu,cuda").split(","):
        for mode in ("default", "plan", "pagewise"):
            run_case(device, mode)
    print("WHEEL_SWAP_ALL_PASSED", os.environ["MATRIX_CALLER"], flush=True)
