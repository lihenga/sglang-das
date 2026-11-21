"""Thread-local CPU buffers for blocking scheduler control collectives."""

import os
from threading import local

import torch


_REUSE = os.getenv("SGLANG_RC2_P_CONTROL_TENSOR_REUSE") == "1"
_thread = local()


def _scratch():
    if not hasattr(_thread, "header"):
        # An inference tensor cannot be updated after leaving inference mode.
        with torch.inference_mode(False):
            _thread.header = torch.empty((1,), dtype=torch.long, device="cpu")
            _thread.offload_count = torch.empty((), dtype=torch.int, device="cpu")
            _thread.linker_counts = torch.empty((2,), dtype=torch.int, device="cpu")
            _thread.load_view, _thread.offload_view = _thread.linker_counts.unbind()
    return _thread


def broadcast_size_header(size, device):
    if _REUSE and device.type == "cpu":
        return _scratch().header.fill_(size)
    return torch.tensor([size], dtype=torch.long, device=device)


def linker_offload_count(num_offloads):
    if _REUSE:
        return _scratch().offload_count.fill_(num_offloads)
    return torch.tensor(num_offloads, dtype=torch.int, device="cpu")


def linker_counts(num_loads, num_offloads):
    if _REUSE:
        buffers = _scratch()
        buffers.load_view.fill_(num_loads)
        buffers.offload_view.fill_(num_offloads)
        return buffers.linker_counts
    return torch.tensor([num_loads, num_offloads], dtype=torch.int, device="cpu")
