"""Bounded NHD <-> KVServe HND staging for BF16 full-attention caches.

SWA buffers and the attention cache allocation remain caller-owned. Only a
fixed-size chunk is materialized in KVServe's packed K/V representation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import threading

logger = logging.getLogger(__name__)


class UnfencedCacheWrite(RuntimeError):
    cache_write_unfenced = True


class UnfencedSourceRead(RuntimeError):
    source_read_unfenced = True


def full_attention_buffers(pool):
    if hasattr(pool, "full_kv_pool"):
        layers = sorted(
            (local, global_id)
            for global_id, (local, is_swa) in pool.layers_mapping.items()
            if not is_swa
        )
        layer_ids = [global_id for _, global_id in layers]
        pool = pool.full_kv_pool
    else:
        layer_ids = list(range(pool.start_layer, pool.start_layer + pool.layer_num))
    keys, values = pool.k_buffer, pool.v_buffer
    if not keys or len(keys) != len(values) or len(keys) != len(layer_ids):
        raise ValueError(
            "KVServe requires separate nonempty full-attention K/V buffers"
        )
    return pool, layer_ids, keys, values


def page_token_indices(pages, page_size: int, capacity: int):
    """Validate physical page ids before any gather or scatter submission."""
    import torch

    if not pages or len(set(pages)) != len(pages):
        raise ValueError("KVServe requires nonempty, distinct physical page ids")
    if any(type(page) is not int or page < 0 for page in pages):
        raise ValueError("KVServe page ids must be nonnegative integers")
    if (max(pages) + 1) * page_size > capacity:
        raise ValueError("KVServe page ids exceed the local cache capacity")
    return (
        torch.tensor(pages, dtype=torch.long)[:, None] * page_size
        + torch.arange(page_size, dtype=torch.long)[None, :]
    ).flatten()


def gather_ndh(packed, source, indices, page_size, offset):
    pages = indices.numel() // page_size
    heads, dim = source.shape[1:]
    gathered = (
        source.index_select(0, indices)
        .view(pages, page_size, heads, dim)
        .permute(0, 2, 1, 3)
    )
    packed[:pages, :, :, offset : offset + dim].copy_(gathered)


def scatter_ndh(packed, target, indices, page_size, offset):
    heads, dim = target.shape[1:]
    pages = (indices.numel() + page_size - 1) // page_size
    restored = (
        packed[:pages, :, :, offset : offset + dim]
        .permute(0, 2, 1, 3)
        .reshape(-1, heads, dim)
    )
    target.index_copy_(0, indices, restored[: indices.numel()])


class KVServeAdapter:
    def __init__(self, pool, extra, num_layers, tp_size, tp_rank, budget_bytes):
        import torch

        backend = importlib.import_module("kvserve_v1.compression.runtime")
        if getattr(backend, "API_VERSION", None) != 1:
            raise RuntimeError("KVServe PD adapter requires compression runtime API 1")
        self.config = backend.CompressionConfig.from_extra_config(extra)
        self.page_size = pool.page_size
        self.chunk_pages = self.config.logical_blocks_per_chunk
        self.pool, self.layer_ids, self.keys, self.values = full_attention_buffers(pool)
        self.device = self.keys[0].device
        if self.device.type != "cuda":
            raise ValueError("KVServe PD requires a CUDA/HIP PyTorch device")
        self.heads, self.key_dim = self.keys[0].shape[-2:]
        self.value_dim = self.values[0].shape[-1]
        for key, value in zip(self.keys, self.values):
            if (
                key.ndim != 3
                or value.ndim != 3
                or key.dtype != torch.bfloat16
                or value.dtype != torch.bfloat16
                or tuple(key.shape[1:]) != (self.heads, self.key_dim)
                or tuple(value.shape[1:]) != (self.heads, self.value_dim)
                or key.shape[0] != value.shape[0]
                or key.device != self.device
                or value.device != self.device
            ):
                raise ValueError("KVServe v1 supports dense separate NHD BF16 K/V only")
        specs = {
            f"layers.{layer}": backend.CacheSpec(
                layer,
                self.page_size,
                self.heads,
                self.key_dim,
                torch.bfloat16,
                self.value_dim,
            )
            for layer in self.layer_ids
        }
        with torch.cuda.device(self.device):
            plan = backend.build_memory_plan(
                self.config,
                specs,
                tp_size=tp_size,
                tp_rank=tp_rank,
                num_layers=num_layers,
                device=self.device,
            )
            packed_bytes = (
                len(specs)
                * self.chunk_pages
                * self.heads
                * self.page_size
                * (self.key_dim + self.value_dim)
                * 2
            )
            # One index_select temporary is live at a time. Charge it to the
            # reserved budget as well as the packed chunk and runtime pool.
            gather_bytes = (
                self.chunk_pages
                * self.page_size
                * self.heads
                * max(self.key_dim, self.value_dim)
                * 2
            )
            self.reserved_bytes = plan.reserved_bytes + packed_bytes + gather_bytes
            if self.reserved_bytes > budget_bytes:
                raise ValueError(
                    f"KVServe needs {self.reserved_bytes / 2**20:.1f} MiB per rank; "
                    "increase --kvserve-reserved-mb or reduce logical_blocks_per_chunk"
                )
            self.packed = {
                name: torch.empty(
                    (
                        self.chunk_pages,
                        self.heads,
                        self.page_size,
                        self.key_dim + self.value_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                for name in specs
            }
            self.runtime = backend.CompressionRuntime(plan, "HND", 1, self.device)
            self.runtime.bind_kv_caches(self.packed, {name: 0 for name in specs})
        geometry = {
            "api": 1,
            "profile": self.runtime.wire_profile,
            "layers": self.layer_ids,
            "tp_size": tp_size,
            "tp_rank": tp_rank,
            "page_size": self.page_size,
            "heads": self.heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
        }
        self.profile = hashlib.sha256(
            json.dumps(geometry, sort_keys=True).encode()
        ).hexdigest()
        self.slot_count = len(self.runtime.slots)
        # Packed layout-conversion tensors are shared, unlike runtime wire
        # buffers. Fence each operation before another slot can use scratch.
        # Network transfers/ACK waits do not hold this lock.
        self._staging_lock = threading.Lock()
        self.slot = self.runtime.slots[0]
        logger.info(
            "KVServe bound: layers=%s TP=%d rank=%d chunk_pages=%d reserved=%.1f MiB profile=%s",
            self.layer_ids,
            tp_size,
            tp_rank,
            self.chunk_pages,
            self.reserved_bytes / 2**20,
            self.profile,
        )

    def registered_memory(self):
        return self.runtime.registered_memory()

    @property
    def max_u8_bytes(self):
        return max(self.runtime.layout.packed_bytes_by_part)

    def raw_bytes(self, part, tokens):
        dim = self.key_dim if part == 0 else self.value_dim
        return len(self.layer_ids) * self.heads * dim * tokens * 2

    def validate_pages(self, pages):
        page_token_indices(pages, self.page_size, min(x.shape[0] for x in self.keys))

    def validate_sizes(self, part, tokens, sizes):
        valid = {0: tokens}
        expected_tail = (
            0
            if self.config.uses_aggregate
            else self.runtime.layout.raw_tail_bytes(part, valid)
        )
        if (
            sizes["u8_bytes"] != self.runtime.data_bytes(part, valid)
            or sizes["aux"] != self.runtime.aux_bytes(part, valid)
            or sizes["tail"] != expected_tail
        ):
            raise ValueError("KVServe byte counts do not match the runtime layout")
        if not self.config.uses_codec and sizes["payload"] != sizes["u8_bytes"]:
            raise ValueError(
                "KVServe raw payload size does not match the runtime layout"
            )

    def receive_regions(self, slot_id=0):
        slot = self.runtime.slots[slot_id]
        return {
            "payload": [slot.arena_a.data_ptr(), slot.arena_a.numel()],
            "aux": [slot.aux.data_ptr(), slot.aux.numel()],
            "tail": [slot.raw_tail.data_ptr(), slot.raw_tail.numel()],
        }

    def _indices(self, pages):
        if len(pages) > self.chunk_pages:
            raise ValueError("KVServe chunk exceeds its preallocated capacity")
        return page_token_indices(
            pages, self.page_size, min(x.shape[0] for x in self.keys)
        ).to(self.device)

    def encode(self, pages, part, valid_tokens, slot_id=0):
        import torch

        slot = self.runtime.slots[slot_id]
        stream = self.runtime.streams[slot_id]
        with (
            self._staging_lock,
            torch.no_grad(),
            torch.cuda.device(self.device),
            torch.cuda.stream(stream),
        ):
            try:
                indices = self._indices(pages)
                for packed, key, value in zip(
                    self.packed.values(), self.keys, self.values
                ):
                    source = key if part == 0 else value
                    offset = 0 if part == 0 else self.key_dim
                    gather_ndh(packed, source, indices, self.page_size, offset)
                encoded = self.runtime.encode(
                    slot_id,
                    [list(range(len(pages)))],
                    part,
                    valid_tokens_by_group={0: valid_tokens},
                )
                encoded.event.synchronize()
            finally:
                try:
                    stream.synchronize()
                except Exception as error:
                    raise UnfencedSourceRead(
                        "KVServe source reads could not be fenced; restart this worker"
                    ) from error
        return {
            "payload": [slot.arena_a.data_ptr(), encoded.codec_bytes],
            "aux": [
                slot.aux.data_ptr(),
                self.runtime.aux_bytes(part, {0: valid_tokens}),
            ],
            "tail": [encoded.raw_tail_addr, encoded.raw_tail_bytes],
            "u8_bytes": encoded.u8_bytes,
        }

    def restore(self, pages, part, valid_tokens, sizes, slot_id=0):
        import torch

        stream = self.runtime.streams[slot_id]
        with (
            self._staging_lock,
            torch.no_grad(),
            torch.cuda.device(self.device),
            torch.cuda.stream(stream),
        ):
            try:
                indices = self._indices(pages)[:valid_tokens]
                event = self.runtime.decode(
                    slot_id,
                    [list(range(len(pages)))],
                    part,
                    sizes["payload"],
                    sizes["u8_bytes"],
                    valid_tokens_by_group={0: valid_tokens},
                )
                event.synchronize()
                for packed, key, value in zip(
                    self.packed.values(), self.keys, self.values
                ):
                    target = key if part == 0 else value
                    offset = 0 if part == 0 else self.key_dim
                    scatter_ndh(packed, target, indices, self.page_size, offset)
            finally:
                # Scheduler-thread fencing: even an exception after the first
                # scatter must not let writes outlive receiver.poll().
                try:
                    stream.synchronize()
                except Exception as error:
                    raise UnfencedCacheWrite(
                        "KVServe cache writes could not be fenced; restart this worker"
                    ) from error
