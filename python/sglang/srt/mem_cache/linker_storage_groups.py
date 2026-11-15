"""Storage-only page grouping. GPU indices and radix hashes stay unchanged."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class StoragePageRef:
    # The chained hash at the group's last page identifies the whole object.
    # ``half`` remains the field name for compatibility with 128-token plans;
    # for a 256-token object it is the zero-based offset in four GPU pages.
    key: str
    half: int
    pages_per_object: int = 2

    def __post_init__(self):
        if (
            not self.key
            or self.pages_per_object not in (2, 4)
            or not 0 <= self.half < self.pages_per_object
        ):
            raise ValueError("Invalid grouped storage reference.")


def make_storage_refs(
    keys: Sequence[str], start_page: int, pages_per_object: int = 2
) -> list[StoragePageRef]:
    """Map a slice to sequence-aligned groups, dropping an incomplete tail.

    A slice may begin inside an object and read only the missing GPU pages.
    Its final partial object has no closing hash and is not persisted/read.
    """
    if start_page < 0:
        raise ValueError("Negative logical page position.")
    if pages_per_object not in (2, 4):
        raise ValueError("Storage objects require two or four GPU pages.")
    result = []
    for index in range(len(keys)):
        half = (start_page + index) % pages_per_object
        anchor = index + (pages_per_object - 1 - half)
        if anchor >= len(keys):
            break
        result.append(StoragePageRef(keys[anchor], half, pages_per_object))
    return result


def group_storage_refs(
    refs: Sequence[StoragePageRef], *, complete: bool = False
) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    widths = {ref.pages_per_object for ref in refs}
    if len(widths) > 1:
        raise ValueError("Mixed storage object sizes in one transfer.")
    for index, ref in enumerate(refs):
        groups.setdefault(ref.key, []).append(index)
    result = []
    for key, positions in groups.items():
        positions.sort(key=lambda index: refs[index].half)
        halves = [refs[index].half for index in positions]
        if len(halves) != len(set(halves)):
            raise ValueError("Duplicate half in a storage object.")
        if complete and halves != list(range(refs[positions[0]].pages_per_object)):
            raise ValueError("Only complete storage objects may be written.")
        result.append((key, positions))
    return result


def storage_plan_digest(kind: str, sequence: int, payload) -> int:
    """Stable signed-int64-safe identity; excludes physical rank-local slots."""
    data = json.dumps([kind, sequence, payload], separators=(",", ":"))
    return int.from_bytes(sha256(data.encode()).digest()[:7], "big")


def storage_namespace(layout, storage_page_size: int = 128) -> str:
    """Separate wire layouts/topologies as well as the legacy 64-token format."""
    data = json.dumps(layout, separators=(",", ":"), sort_keys=True)
    return f"l3p{storage_page_size}v1_" + sha256(data.encode()).hexdigest()[:24]
