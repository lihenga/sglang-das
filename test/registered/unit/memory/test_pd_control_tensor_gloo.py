"""Exercise the exact installed broadcast function and count reductions on eight CPU ranks."""

import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn


ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt/utils"
HELPER = "sglang.srt.utils.poll_control_tensors"


def installed_paths():
    spec = importlib.util.spec_from_file_location(HELPER, ROOT / "poll_control_tensors.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[HELPER] = helper
    spec.loader.exec_module(helper)
    common = ast.parse((ROOT / "common.py").read_text())
    future = next(n for n in ast.parse("from __future__ import annotations").body)
    implementation = next(n for n in common.body
                          if isinstance(n, ast.FunctionDef) and n.name == "broadcast_pyobj")
    globals_ = dict(torch=torch, dist=dist, pickle=pickle, np=numpy, is_musa=lambda: False,
                    _RC2_P_BROADCAST_MEMORYVIEW=True)
    exec(compile(ast.Module(body=[future, implementation], type_ignores=[]),
                 str(ROOT / "common.py"), "exec"), globals_)
    return helper, globals_["broadcast_pyobj"]


def work(rank, rendezvous, reuse):
    os.environ["SGLANG_RC2_P_CONTROL_TENSOR_REUSE"] = str(int(reuse))
    torch.set_num_threads(1)
    torch.cuda.is_available = lambda: False
    helper, broadcast = installed_paths()
    device = torch.device("cpu")
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=8, timeout=timedelta(seconds=75))
    original = torch.tensor
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    torch.tensor = counted
    try:
        for i in range(32):
            broadcast.__globals__["_RC2_P_BROADCAST_MEMORYVIEW"] = bool(i % 2)
            src = i % 8
            expected = [] if i % 4 == 0 else [{"id": i, "body": b"x" * (1048576 if i == 31 else i * 512)}]
            with torch.inference_mode(i % 2 == 0):
                received = broadcast(expected if rank == src else None, rank, src=src)
                assert received == expected
                scalar = helper.linker_offload_count(rank + i)
                dist.all_reduce(scalar, op=dist.ReduceOp.MIN)
                assert scalar.item() == i
                pair = helper.linker_counts(rank * 2 + i, 7 - rank + i)
                dist.all_reduce(pair, op=dist.ReduceOp.MIN)
                assert pair.tolist() == [i, i]
        assert calls == (0 if reuse else 96), calls
    finally:
        torch.tensor = original
        dist.destroy_process_group()
    if rank == 0:
        if reuse:
            own = helper.broadcast_size_header(0, device)
            with ThreadPoolExecutor(max_workers=1) as thread:
                foreign = thread.submit(helper.broadcast_size_header, 1, device).result()
            assert own.data_ptr() != foreign.data_ptr() and own.item() == 0 and foreign.item() == 1
        print(json.dumps(dict(reuse=reuse, ranks=8, rounds=32, tensor_factory_calls_per_rank=calls)), flush=True)


def test_cpu_gloo_roundtrip():
    with tempfile.TemporaryDirectory() as folder:
        for enabled in (False, True):
            spawn(work, args=(str(Path(folder) / str(enabled)), enabled), nprocs=8)


if __name__ == "__main__":
    test_cpu_gloo_roundtrip()
