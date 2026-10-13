"""DSV4 allocations with real tail capacity for Mooncake XDS registration."""

from math import prod
from typing import Literal

import torch

from sglang.srt.environ import envs


def allocate_dsv4_buffer(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: str,
    init: Literal["zeros", "empty"],
) -> torch.Tensor:
    """Keep the logical layout, padding only the backing storage for XDS.

    Allocate inside the caller's memory-saver/custom-pool context. A zero-offset
    view retains ownership of the entire allocation, including its tail; no
    registration length is fabricated and no shifted pointer hides a misaligned
    storage base. CPU allocations exercise layout/ownership, not GPU alignment.
    """
    factory = torch.zeros if init == "zeros" else torch.empty
    if envs.MOONCAKE_DFS_DIRECT_IO_BACKEND.get() not in (
        "xds-required",
        "xds-auto",
    ):
        return factory(shape, dtype=dtype, device=device)

    alignment = 4096
    numel = prod(shape)
    nbytes = numel * dtype.itemsize
    storage_nbytes = (nbytes + alignment - 1) // alignment * alignment
    backing = factory((storage_nbytes // dtype.itemsize,), dtype=dtype, device=device)
    storage = backing.untyped_storage()
    if backing.is_cuda and storage.data_ptr() % alignment:
        raise RuntimeError(
            "DSV4 XDS allocation requires a 4096-byte aligned GPU storage base; "
            f"device={backing.device}, storage_base={storage.data_ptr():#x}, "
            f"storage_nbytes={storage.nbytes()}, logical_nbytes={nbytes}. "
            "Tail padding cannot fix base alignment; check the GPU/custom pool allocator."
        )
    return backing[:numel].view(shape)
