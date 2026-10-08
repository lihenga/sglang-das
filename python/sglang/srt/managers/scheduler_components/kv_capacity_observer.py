from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    KVCapacitySnapshot,
)
from sglang.srt.observability.trace import get_global_tracing_enabled


@dataclass(frozen=True, slots=True)
class ScheduleKVCapacityObserver:
    token_to_kv_pool_allocator: Any
    linker: Any
    local_kv_bytes_per_slot: int

    @classmethod
    def create(cls, *, token_to_kv_pool_allocator, tree_cache):
        linker = getattr(tree_cache, "linker", None)
        if linker is None:
            return None
        local_kv_bytes_per_slot = linker.get_local_kv_bytes_per_slot()
        if local_kv_bytes_per_slot is None:
            return None
        return cls(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            linker=linker,
            local_kv_bytes_per_slot=local_kv_bytes_per_slot,
        )

    def snapshot(self) -> dict[str, KVCapacitySnapshot] | None:
        if not get_global_tracing_enabled():
            return None

        available_slots = self.token_to_kv_pool_allocator.available_size()
        capacity_slots = self.token_to_kv_pool_allocator.size
        snapshots = {
            "l1": KVCapacitySnapshot(
                available_slots=available_slots,
                capacity_slots=capacity_slots,
                available_bytes=available_slots * self.local_kv_bytes_per_slot,
                capacity_bytes=capacity_slots * self.local_kv_bytes_per_slot,
            )
        }
        if (l3_snapshot := self.linker.get_kv_capacity_snapshot()) is not None:
            snapshots["l3"] = l3_snapshot
        return snapshots

    @staticmethod
    def timeline_attrs(
        before: dict[str, KVCapacitySnapshot] | None,
        after: dict[str, KVCapacitySnapshot] | None,
    ) -> dict[str, int] | None:
        attrs = {}
        for phase, snapshots in (("before", before), ("after", after)):
            for level, snapshot in (snapshots or {}).items():
                prefix = f"{level}_kv"
                attrs[f"{prefix}_available_slots_{phase}"] = snapshot.available_slots
                attrs[f"{prefix}_capacity_slots_{phase}"] = snapshot.capacity_slots
                attrs[f"{prefix}_available_bytes_{phase}"] = snapshot.available_bytes
                attrs[f"{prefix}_capacity_bytes_{phase}"] = snapshot.capacity_bytes
        return attrs or None
