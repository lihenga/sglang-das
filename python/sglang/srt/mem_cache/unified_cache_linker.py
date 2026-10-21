"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. Self-contained: this module owns both halves of the contract.

* :class:`DevicePoolEntry` / :class:`DevicePoolGroup` -- the shared zero-copy
  view over device pools used by direct-linker backends.
* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``_inc_hit_count``    -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InsertParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs


class DevicePoolEntry:
    """Zero-copy linker view over one physical device pool."""

    def __init__(
        self,
        *,
        name: PoolName,
        indices_from_pool: PoolName,
        device_pool: Any,
        components: Sequence[Sequence[torch.Tensor]],
        layer_mapping: dict[int, int],
        page_size: int,
        rows_are_pages: bool,
        packed: bool = True,
        index_mapper: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.packed = packed
        self._index_mapper = index_mapper
        self._page_offsets = torch.arange(page_size)
        self._row_span = 1 if rows_are_pages else page_size

        if hasattr(device_pool, "v_buffer"):
            self.v_buffer = device_pool.v_buffer

        if not self.components or any(not component for component in self.components):
            raise ValueError(f"Device pool {name} has no storage buffers.")
        self.kv_buffer = [buffer for group in self.components for buffer in group]
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)

        self.buffer_meta = [
            [
                (
                    buffer.data_ptr(),
                    buffer.stride(0) * buffer.element_size(),
                    buffer.nbytes // buffer.shape[0] * self._row_span,
                )
                for buffer in component
            ]
            for component in self.components
        ]

        self._component_offsets = []
        offset = 0
        for component in self.buffer_meta:
            if not packed:
                offset = 0
            offsets = []
            for _, _, size in component:
                offsets.append(offset)
                offset += size
            self._component_offsets.append(offsets)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._index_mapper(indices) if self._index_mapper else indices

    def _rows(self, indices: torch.Tensor) -> list[int]:
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Pool {self.name} got {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(f"Pool {self.name} requires aligned contiguous pages.")
        rows = (
            starts.div(self.page_size, rounding_mode="floor")
            if self._row_span == 1
            else starts
        )
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            raise ValueError(
                f"Pool {self.name} row range [{first_row}, {last_row}) exceeds "
                f"buffer shapes {[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )
        return rows.tolist()

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices)
        ptrs = [
            base_ptr + row * row_stride
            for row in rows
            for component in self.buffer_meta
            for base_ptr, row_stride, _ in component
        ]
        sizes = [
            size
            for _ in rows
            for component in self.buffer_meta
            for _, _, size in component
        ]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices)

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        buffer_index = self.layer_mapping.get(layer)
        if buffer_index is None:
            return None

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            base_ptr, row_stride, size = component[buffer_index]
            items.append((base_ptr, row_stride, size, offsets[buffer_index]))

        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [
                base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items
            ]
            row_sizes = [size for _, _, size, _ in items]
            row_offsets = [offset for _, _, _, offset in items]
            if self.packed:
                ptrs.append(row_ptrs)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
            else:
                ptrs.extend([[value] for value in row_ptrs])
                sizes.extend([[value] for value in row_sizes])
                offsets.extend([[value] for value in row_offsets])
        return ptrs, sizes, offsets


class DevicePoolGroup:
    """Physical device pools sharing one logical linker layer range."""

    def __init__(
        self,
        entries: Sequence[DevicePoolEntry],
        num_layers: int,
        page_size: int,
        is_mla_backend: bool = True,
    ):
        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        if len(self.entries) != len(self.entry_map):
            raise ValueError("DevicePoolGroup contains duplicate pool names.")
        self.sources = {entry.name: entry.indices_from_pool for entry in self.entries}
        self.num_layers = num_layers
        self.page_size = page_size
        self.is_mla_backend = is_mla_backend
        self.kv_buffer = None

    def resolve_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        allow_partial: bool = False,
        allow_missing_kv: bool = False,
    ) -> list[PoolTransfer]:
        """Expand logical component transfers into physical device pools."""
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if not any(transfer.keys for transfer in transfers):
            return []
        if not allow_missing_kv and (kv is None or not kv.keys):
            return []
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        resolved = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None or not source.keys:
                continue
            indices = source.device_indices
            resolved.append(
                replace(
                    source,
                    name=name,
                    host_indices=(
                        self.entry_map[name].translate_indices(indices)
                        if indices is not None
                        else None
                    ),
                    keys=list(source.keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                )
            )
        return resolved


logger = logging.getLogger(__name__)


class ExternalLinkerLoadError(RuntimeError):
    """A recoverable external-cache load failure for the current batch."""


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    def waiting_queue_prefetch_enabled(self) -> bool:
        """Whether waiting-queue host prefetch is enabled on every rank."""
        return False

    def get_local_dfs_health_policy(self) -> tuple[bool, bool] | None:
        """Return a rank-local DFS policy candidate, if the backend has one."""
        return None

    def apply_dfs_health_policy(self, allow_read: bool, allow_write: bool) -> None:
        """Apply a cross-rank DFS policy, if the backend supports it."""

    def disable_waiting_queue_prefetch(self) -> None:
        """Turn waiting-queue host prefetch off before any job is submitted."""

    def prepare_load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Prepare sessions on a prefetch/load worker before reading."""
        return False

    def submit_host_prefetch(
        self, rid: str, transfers: list[PoolTransfer]
    ) -> bool:
        """Queue a request-scoped host prefetch without allocating device pages."""
        return False

    def get_host_prefetch_status(self, rid: str) -> str | None:
        """Return the backend host-prefetch state, or ``None`` if absent."""
        return None

    def revalidate_host_prefetch(self, rid: str) -> bool:
        """Refresh the prepared session without copying into device memory."""
        return False

    def revalidate_no_prefetch_needed(self, rid: str) -> bool:
        """Check that a prefetch finished without DFS reads is still loadable."""
        return False

    def claim_ready_host_prefetch(self, rid: str) -> bool:
        """Transfer a READY speculative session to the normal load path."""
        return False

    def cancel_host_prefetch(self, rid: str) -> None:
        """Cancel or retire a request-scoped host prefetch."""

    def abort_prepared_load(self, rid: str) -> None:
        """Release a session claimed by the normal load path."""

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def wait_for_pending_loads(self) -> None:
        """Wait for loads already issued to the backend worker to finish.

        Loads still waiting for ``start_layer_wise_loading`` are not included.
        """

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> tuple[list[str], bool]:
        """Consume the oldest completed load batch and return its result."""

    def cancel_queued_load(self, rid: str) -> bool:
        return False

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``prefix_key`` covers the device-cached prefix plus the restorable tail, so
    it is what gets inserted once the tail lands. ``tail_hashes`` are the
    per-page storage hashes of that tail alone, starting at ``device_hit_len``.
    """

    prefix_key: RadixKey
    tail_hashes: list[str]
    device_hit_len: int


class PreparedHostPrefetch(NamedTuple):
    """Immutable request-local work for the scheduler's background prefetch round."""

    rid: str
    locally_eligible: bool
    transfers: tuple[PoolTransfer, ...]
    cache_linker: UnifiedCacheLinker
    cancelled: threading.Event


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        server_args: ServerArgs,
        params: CacheInitParams,
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            MooncakeDirectLinker,
        )

        self.cache = cache
        self.cache_linker: UnifiedCacheLinker = MooncakeDirectLinker(
            server_args,
            params,
            components=set(cache.components),
        )
        self._waiting_queue_prefetch_enabled = (
            self.cache_linker.waiting_queue_prefetch_enabled()
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Waiting-queue prefetches retain the original hit until admission can
        # rematch it. The backend owns the private session and readiness state.
        self.host_prefetch_hits: dict[str, ExternalCacheHitMarker] | None = (
            {} if self._waiting_queue_prefetch_enabled else None
        )
        # Submission work is prepared and committed only by the scheduler
        # thread. The background worker receives a snapshot and never touches
        # this dict or the cache tree.
        self.pending_host_prefetch_submissions: (
            dict[str, tuple[PreparedHostPrefetch, ExternalCacheHitMarker]] | None
        ) = {} if self._waiting_queue_prefetch_enabled else None
        # Loads in flight, each pinning its inserted endpoint until DMA
        # completes. The anchor is the request's node before the load, so a
        # failed load can walk back exactly the chain it published. Under PP
        # nothing is inserted, so endpoint and anchor are both req.last_node
        # and the loaded slots stay request-owned until the normal
        # post-prefill insert.
        self.pending_loads: dict[str, tuple[NodeId, DecLockRefParams, NodeId]] = {}
        self.failed_chains: dict[str, list[NodeId]] = {}
        self.taken_loads: list[tuple[list[str], bool]] = []
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[tuple[NodeId, DecLockRefParams]] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    def waiting_queue_prefetch_enabled(self) -> bool:
        return self._waiting_queue_prefetch_enabled

    def get_local_dfs_health_policy(self) -> tuple[bool, bool] | None:
        linker = getattr(self, "cache_linker", None)
        getter = getattr(linker, "get_local_dfs_health_policy", None)
        return getter() if callable(getter) else None

    def apply_dfs_health_policy(self, allow_read: bool, allow_write: bool) -> None:
        linker = getattr(self, "cache_linker", None)
        apply_policy = getattr(linker, "apply_dfs_health_policy", None)
        if callable(apply_policy):
            apply_policy(allow_read, allow_write)

    def disable_waiting_queue_prefetch(self) -> None:
        self._waiting_queue_prefetch_enabled = False
        self.cache_linker.disable_waiting_queue_prefetch()

    def get_host_prefetch_admission_state(self, rid: str) -> str:
        if not self._waiting_queue_prefetch_enabled:
            return "not_tracked"
        if rid in self.pending_host_prefetch_submissions:
            return "pending"
        tracked = rid in self.host_prefetch_hits
        status = self.cache_linker.get_host_prefetch_status(rid)
        if status in {"queued", "preparing", "reading"}:
            return "pending"
        if tracked and status == "dfs_prefetched":
            return "dfs_prefetched"
        if tracked and status == "no_prefetch_needed":
            return "no_prefetch_needed"
        if not tracked and status is None:
            return "not_tracked"
        return "terminal"

    def cancel_waiting_queue_prefetch(self, rid: str) -> None:
        getattr(self, "load_admission_verdicts", {}).pop(rid, None)
        self.hit_markers.pop(rid, None)
        if not self._waiting_queue_prefetch_enabled:
            return
        self.host_prefetch_hits.pop(rid, None)
        pending = self.pending_host_prefetch_submissions.pop(rid, None)
        if pending is not None:
            pending[0].cancelled.set()
        self.cache_linker.cancel_host_prefetch(rid)

    def prepare_host_prefetch(self, req: Req) -> PreparedHostPrefetch:
        """Build a request-local snapshot without issuing any collectives."""
        hit = self.hit_markers.get(req.rid)
        transfers = []
        locally_eligible = hit is not None
        if hit is not None:
            try:
                for component in self.cache._components_tuple:
                    transfer = component.build_external_linker_transfer(
                        LinkerTransferPhase.LOOKUP, None, hit.tail_hashes
                    )
                    if transfer is None:
                        locally_eligible = False
                        break
                    # LOOKUP transfers are CPU metadata only. Copy their
                    # mutable key lists so the worker owns its snapshot.
                    transfers.append(
                        PoolTransfer(
                            name=transfer.name,
                            host_indices=transfer.host_indices,
                            device_indices=transfer.device_indices,
                            keys=(list(transfer.keys) if transfer.keys is not None else None),
                            hit_policy=transfer.hit_policy,
                            nodes_to_load=transfer.nodes_to_load,
                            indices_from_pool=transfer.indices_from_pool,
                        )
                    )
            except Exception:
                locally_eligible = False

        job = PreparedHostPrefetch(
            rid=req.rid,
            locally_eligible=locally_eligible,
            transfers=tuple(transfers),
            cache_linker=self.cache_linker,
            cancelled=threading.Event(),
        )
        if hit is not None:
            self.pending_host_prefetch_submissions[req.rid] = (job, hit)
        return job

    def complete_host_prefetch_submission(
        self, job: PreparedHostPrefetch, submitted: bool
    ) -> bool:
        """Commit a background result if this is still the live request."""
        pending = self.pending_host_prefetch_submissions.get(job.rid)
        if pending is None or pending[0] is not job:
            if submitted:
                job.cache_linker.cancel_host_prefetch(job.rid)
            return False

        del self.pending_host_prefetch_submissions[job.rid]
        hit = pending[1]
        if submitted and not job.cancelled.is_set():
            self.host_prefetch_hits[job.rid] = hit
            return True
        if submitted:
            job.cache_linker.cancel_host_prefetch(job.rid)
        return False

    def prefetch_to_host(self, req: Req) -> bool:
        if not self._waiting_queue_prefetch_enabled:
            return False
        hit = self.hit_markers.get(req.rid)
        transfers = []
        locally_eligible = hit is not None
        if hit is not None:
            try:
                for component in self.cache._components_tuple:
                    transfer = component.build_external_linker_transfer(
                        LinkerTransferPhase.LOOKUP, None, hit.tail_hashes
                    )
                    if transfer is None:
                        locally_eligible = False
                        break
                    transfers.append(transfer)
            except Exception:
                locally_eligible = False

        eligible = torch.tensor(int(locally_eligible), dtype=torch.int)
        self.cache._all_reduce_attn_groups(eligible, torch.distributed.ReduceOp.MIN)
        if int(eligible.item()) == 0:
            return False

        try:
            submitted = self.cache_linker.submit_host_prefetch(req.rid, transfers)
        except BaseException:
            submitted = False

        globally_submitted = torch.tensor(int(submitted), dtype=torch.int)
        self.cache._all_reduce_attn_groups(
            globally_submitted, torch.distributed.ReduceOp.MIN
        )
        if int(globally_submitted.item()) == 0:
            self.cache_linker.cancel_host_prefetch(req.rid)
            return False

        self.host_prefetch_hits[req.rid] = hit
        return True

    def has_pending_operations(self) -> bool:
        return bool(self.pending_loads or self.pending_offloads)

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        page = cache.page_size
        device_hit_len = int(result.device_indices.numel())
        self.hit_markers.pop(req.rid, None)
        prefetched_hit = (
            self.host_prefetch_hits.get(req.rid)
            if self._waiting_queue_prefetch_enabled
            else None
        )

        known_hit_len = req.external_cache_hit_length if cache.pp_size > 1 else None
        if known_hit_len is not None:
            # The absolute boundary also survives later local L1 rematches.
            key = key[:known_hit_len]
        elif cache.pp_size > 1:
            # Every PP stage must derive the same absolute boundary before it
            # builds its extend batch.  In particular, a downstream stage
            # cannot stop at its local L1 hit while PP0 extends that hit from
            # L3: the hidden states sent by PP0 would then have a different
            # token count from the downstream batch.  Direct-linker lookups
            # are metadata-only, and PP-aware backends (for example Mooncake)
            # validate every PP shard, so probing from each stage's local L1
            # frontier produces the common restorable boundary without an
            # additional scheduler-side PP control message.
            req.external_cache_hit_length = device_hit_len

        if device_hit_len >= len(key):
            if prefetched_hit is not None:
                self.cancel_waiting_queue_prefetch(req.rid)
            return result

        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        if not tail_hashes:
            if prefetched_hit is not None:
                self.cancel_waiting_queue_prefetch(req.rid)
            return result

        lookup_transfers = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, tail_hashes
            )
            if transfer is None:
                if prefetched_hit is not None:
                    self.cancel_waiting_queue_prefetch(req.rid)
                return result
            lookup_transfers.append(transfer)
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        if prefetched_hit is not None:
            expected_len = prefetched_hit.device_hit_len + len(
                prefetched_hit.tail_hashes
            ) * page
            current_prefix = key[:expected_len]
            same_prefix = (
                current_prefix.extra_key == prefetched_hit.prefix_key.extra_key
                and current_prefix.cache_salt == prefetched_hit.prefix_key.cache_salt
                and current_prefix.is_bigram == prefetched_hit.prefix_key.is_bigram
                and current_prefix.raw_token_ids()
                == prefetched_hit.prefix_key.raw_token_ids()
            )
            reusable = (
                device_hit_len == prefetched_hit.device_hit_len
                and list(tail_hashes[: len(prefetched_hit.tail_hashes)])
                == prefetched_hit.tail_hashes
                and same_prefix
            )
            if reusable:
                hit_pages = len(prefetched_hit.tail_hashes)
                hit_tokens = hit_pages * page
                swa_transfer = by_pool.get(PoolName.SWA)
                self.hit_markers[req.rid] = prefetched_hit
                return result._replace(
                    last_host_node=result.best_match_node,
                    host_hit_length=hit_tokens,
                    swa_host_hit_length=max(
                        result.swa_host_hit_length,
                        min(len(swa_transfer.keys), hit_pages) * page
                        if swa_transfer is not None
                        else 0,
                    ),
                    mamba_host_hit_length=max(
                        result.mamba_host_hit_length,
                        1 if PoolName.MAMBA in by_pool else 0,
                    ),
                )

            self.cancel_waiting_queue_prefetch(req.rid)

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        if known_hit_len is None:
            hit_pages = self._sync_restorable_prefix(
                self.cache_linker.lookup(req.rid, lookup_transfers),
                num_pages=len(tail_hashes),
                device_hit_pages=0,
            )
            if cache.pp_size > 1:
                req.external_cache_hit_length = device_hit_len + hit_pages * page
        else:
            hit_pages = len(tail_hashes)
        if hit_pages == 0:
            return result
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            prefix_key=key[: device_hit_len + hit_tokens],
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in restorable:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> device, then insert ----

    def _build_load_transfers(
        self, req: Req, tail_hashes
    ) -> tuple[list[tuple[TreeComponent, PoolTransfer]], bool, BaseException | None]:
        """Build LOAD transfers without session RPCs on the scheduler.

        Building allocates device slots and can fail on one rank alone, so the
        caller must agree on the returned flag with the other ranks before any
        later collective, and on failure ABORT the transfers returned here (a
        component frees slots it could not hand over). A component returning
        None (not enough room) is an expected miss. An unexpected error may
        have left the tree or an allocator partly changed, so it is returned
        rather than raised: this rank still joins that agreement, and the
        caller re-raises it after cleanup instead of serving on.
        """

        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        try:
            for component in self.cache._components_tuple:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, tail_hashes
                )
                if transfer is None:
                    return component_transfers, False, None
                component_transfers.append((component, transfer))
        except BaseException as error:
            return component_transfers, False, error
        return component_transfers, True, None

    def _abort_disagreed_load(
        self,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        cancel_prefetch: bool,
        pending: BaseException | None,
    ) -> None:
        """Clean up after the ranks agreed a load cannot proceed.

        Frees this rank's transfers, then, on the prefetched path, cancels the
        prefetch (rolling a claimed session back) even if freeing raised.
        Raises the pending build or claim error, which stays the primary one;
        a cleanup error is logged then, and raised otherwise. Freeing stops at
        the first component that raises, so a failed cleanup does not promise
        that every slot was released.
        """
        cleanup_error: BaseException | None = None
        try:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
        except BaseException as error:
            cleanup_error = error
        if cancel_prefetch:
            try:
                self.cache_linker.cancel_host_prefetch(req.rid)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    logger.error(
                        "Cancelling the prefetch of rid=%s also failed",
                        req.rid,
                        exc_info=error,
                    )
        if pending is not None:
            if cleanup_error is not None:
                logger.error(
                    "Cleaning up the failed load of rid=%s also failed",
                    req.rid,
                    exc_info=cleanup_error,
                )
            raise pending
        if cleanup_error is not None:
            raise cleanup_error

    def prepare_load_admission(self, req: Req) -> bool:
        """Wait in the queue for session RPCs, then agree before allocating KV."""
        prepare = getattr(self.cache_linker, "prepare_load_admission", None)
        if not self._waiting_queue_prefetch_enabled or not callable(prepare):
            return True
        hit = self.hit_markers.get(req.rid)
        if hit is None:
            self.cache_linker.abort_prepared_load(req.rid)
        transfers = []
        state = "ready"
        try:
            if hit is None:
                components = ()
            else:
                components = self.cache._components_tuple
            for component in components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, hit.tail_hashes
                )
                if transfer is None:
                    raise ValueError("No metadata for load admission")
                transfers.append(transfer)
            if hit is not None:
                state = prepare(req.rid, transfers)
        except BaseException:
            logger.warning(
                "External load admission failed for rid=%s", req.rid, exc_info=True
            )
            state = "failed"
        verdict = torch.tensor(
            [int(state == "pending"), int(state == "ready")], dtype=torch.int
        )
        self.cache._all_reduce_attn_groups(verdict[:1], torch.distributed.ReduceOp.MAX)
        self.cache._all_reduce_attn_groups(verdict[1:], torch.distributed.ReduceOp.MIN)
        if int(verdict[0]):
            return False
        if not hasattr(self, "load_admission_verdicts"):
            self.load_admission_verdicts = {}
        if hit is not None:
            self.load_admission_verdicts[req.rid] = bool(verdict[1])
        return True

    def load_back(self, req: Req) -> tuple[torch.Tensor, NodeId]:
        cache = self.cache
        empty_indices = cache.tree_core.empty_match_result.device_indices
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return empty_indices, req.last_node

        admission = getattr(self, "load_admission_verdicts", {}).pop(req.rid, None)
        if admission is False:
            self.cancel_waiting_queue_prefetch(req.rid)
            self.cache_linker.abort_prepared_load(req.rid)
            return empty_indices, req.last_node
        prepared_for_load = admission is True
        prefetched = (
            self.host_prefetch_hits.pop(req.rid, None) is not None
            if self._waiting_queue_prefetch_enabled
            else False
        )
        if prefetched:
            status = self.cache_linker.get_host_prefetch_status(req.rid)
            locally_complete = status in {"dfs_prefetched", "no_prefetch_needed"}
            ready = torch.tensor(int(locally_complete), dtype=torch.int)
            cache._all_reduce_attn_groups(ready, torch.distributed.ReduceOp.MIN)
            if int(ready.item()) == 0:
                self.cache_linker.cancel_host_prefetch(req.rid)
                if admission is True:
                    return empty_indices, req.last_node
                self.hit_markers[req.rid] = hit
                return self.load_back(req)

            # Every rank must reach the reduction below, so any local failure,
            # including an exception, becomes a local False.
            try:
                if prepared_for_load:
                    locally_valid = True
                elif status == "no_prefetch_needed":
                    # Its session was released while queued; re-check that
                    # the keys survived so eviction falls back to the normal
                    # path instead of failing the later load.
                    locally_valid = self.cache_linker.revalidate_no_prefetch_needed(
                        req.rid
                    )
                else:
                    locally_valid = (
                        status == "dfs_prefetched"
                        and self.cache_linker.revalidate_host_prefetch(req.rid)
                    )
            except BaseException:
                locally_valid = False
            valid = torch.tensor(int(locally_valid), dtype=torch.int)
            cache._all_reduce_attn_groups(valid, torch.distributed.ReduceOp.MIN)
            if int(valid.item()) == 0:
                self.cache_linker.cancel_host_prefetch(req.rid)
                if admission is True:
                    return empty_indices, req.last_node
                self.hit_markers[req.rid] = hit
                return self.load_back(req)

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        if prefetched:
            # Build before claiming so that the claim reduction also carries
            # the construction verdict: one reduction instead of two.
            component_transfers, constructed, build_error = self._build_load_transfers(
                req, tail_hashes
            )
            claim_error: BaseException | None = None
            try:
                if status == "no_prefetch_needed" and admission is not True:
                    self.cache_linker.cancel_host_prefetch(req.rid)
                    claimed = True
                else:
                    claimed = self.cache_linker.claim_ready_host_prefetch(req.rid)
            except BaseException as error:
                # Still join the reduction below, then re-raise after cleanup.
                # Reporting "not constructed" sends every rank to the miss exit
                # rather than to a normal-path retry this rank would not join.
                claimed, constructed, claim_error = False, False, error
            verdict = torch.tensor([int(claimed), int(constructed)], dtype=torch.int)
            cache._all_reduce_attn_groups(verdict, torch.distributed.ReduceOp.MIN)
            claimed_all, constructed_all = (int(value) for value in verdict)
            if not (claimed_all and constructed_all):
                # Free the device slots, then cancel (which also rolls a
                # locally claimed session back); a build or claim error is
                # raised on its own rank only after both were attempted.
                if build_error is not None and claim_error is not None:
                    logger.error(
                        "Claiming the prefetch of rid=%s also failed",
                        req.rid,
                        exc_info=claim_error,
                    )
                self._abort_disagreed_load(
                    req,
                    component_transfers,
                    prefix_len,
                    cancel_prefetch=True,
                    pending=build_error if build_error is not None else claim_error,
                )
                if not constructed_all or admission is True:
                    # A prepared session cannot be retried after its admission
                    # failed on a peer; recompute consistently instead.
                    # Building failed somewhere: a miss, as on the normal path.
                    return empty_indices, req.last_node
                # Every rank built but a claim failed: retry the normal path,
                # which builds again and agrees on its own construction.
                self.hit_markers[req.rid] = hit
                return self.load_back(req)
            prepared_for_load = True
        else:
            # Ordinary LOAD construction must agree across ranks even when
            # waiting-queue prefetch is disabled.
            component_transfers, constructed, build_error = self._build_load_transfers(
                req, tail_hashes
            )
            all_constructed = torch.tensor(int(constructed), dtype=torch.int)
            cache._all_reduce_attn_groups(
                all_constructed, torch.distributed.ReduceOp.MIN
            )
            if int(all_constructed.item()) == 0:
                # Each rank frees what it built; a component frees its own
                # slots when it fails after allocating.
                self._abort_disagreed_load(
                    req,
                    component_transfers,
                    prefix_len,
                    cancel_prefetch=admission is True,
                    pending=build_error,
                )
                return empty_indices, req.last_node
            # Session start/refresh belongs to the load worker. Admission can
            # still reject an already stale hit through revalidate_load below.

        # Keys can be evicted remotely (master memory-watermark eviction)
        # between the match-time lookup and this load-back; a stale hit would
        # then fail the async layer-wise session fatally. Re-check existence
        # here so the request degrades to a plain cache miss instead.
        revalidate = getattr(self.cache_linker, "revalidate_load", None)
        valid = True
        if not prepared_for_load and revalidate is not None:
            try:
                valid = revalidate([transfer for _, transfer in component_transfers])
            except BaseException:
                valid = False
            verdict = torch.tensor(int(valid), dtype=torch.int)
            cache._all_reduce_attn_groups(verdict, torch.distributed.ReduceOp.MIN)
            valid = bool(verdict.item())
        if not valid:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT,
                req,
                component_transfers,
                prefix_len,
            )
            return empty_indices, req.last_node

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        try:
            prepared_transfers = self._update_load(
                ExternalLinkerLoadPhase.PREPARE,
                req,
                component_transfers,
                prefix_len,
            )
        except BaseException:
            self.cache_linker.abort_prepared_load(req.rid)
            raise

        if cache.pp_size > 1:
            # Load into request-owned slots. The existing PP result ring delays
            # normal insert/dedup until every stage has completed this prefill.
            # This branch publishes no chain, so the anchor is the endpoint
            # itself and a failed load's detach walk is a no-op.
            try:
                self._queue_load(
                    req.rid, req.last_node, prepared_transfers, anchor=req.last_node
                )
            except BaseException:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT,
                    req,
                    component_transfers,
                    prefix_len,
                )
                raise
            return full_transfer.device_indices, req.last_node

        # Insert the newly loaded tail into the tree.
        prefix_indices = torch.cat(
            [req.prefix_indices.to(torch.int64), full_transfer.device_indices]
        )
        mamba_transfer = next(
            (
                transfer
                for _, transfer in component_transfers
                if transfer.name == PoolName.MAMBA
            ),
            None,
        )
        insert_result = cache.insert(
            InsertParams(
                key=hit.prefix_key,
                value=prefix_indices,
                mamba_value=(
                    mamba_transfer.device_indices[:1]
                    if mamba_transfer is not None
                    else None
                ),
                prev_prefix_len=device_hit_len,
                swa_evicted_seqlen=(
                    req.kv.swa_evicted_seqlen if req.kv is not None else 0
                ),
                chunked=True,
                priority=getattr(req, "priority", 0) or 0,
                track_adopted_ranges=True,
            )
        )
        if mamba_transfer is not None and insert_result.mamba_exist:
            cache.req_to_token_pool.mamba_allocator.free(
                mamba_transfer.device_indices[:1]
            )

        canonical_tail = cache.tree_core.collect_full_device_indices(
            insert_result.last_device_node, req.last_node
        )
        assert canonical_tail.numel() == len(tail_hashes) * cache.page_size
        load_transfers = self._update_load(
            ExternalLinkerLoadPhase.COMMIT,
            req,
            component_transfers,
            prefix_len,
            insert_result=insert_result,
            canonical_full=canonical_tail,
        )

        if load_transfers:
            anchor = req.last_node
            node_id = insert_result.last_device_node
            lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
            try:
                queued = self.cache_linker.load(req.rid, load_transfers)
            except BaseException:
                self.cache.dec_lock_ref(node_id, lock_params)
                raise
            if not queued:
                self.cache.dec_lock_ref(node_id, lock_params)
                raise RuntimeError(f"Failed to queue the linker load for {req.rid=}.")
            self.pending_loads[req.rid] = (node_id, lock_params, anchor)

        node = cache.resolve_node_handle(insert_result.last_device_node)
        while node.id != req.last_node:
            node.external_cache_stored = True
            node = node.parent
        return canonical_tail, insert_result.last_device_node

    def _queue_load(
        self,
        rid: str,
        node_id: NodeId,
        transfers: list[PoolTransfer],
        anchor: NodeId,
    ) -> None:
        if not transfers:
            self.cache_linker.abort_prepared_load(rid)
            return
        assert rid not in self.pending_loads
        lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.load(rid, transfers)
        except BaseException:
            self.cache_linker.abort_prepared_load(rid)
            self.cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            self.cache_linker.abort_prepared_load(rid)
            self.cache.dec_lock_ref(node_id, lock_params)
            raise RuntimeError(f"Failed to queue the linker load for rid={rid!r}.")
        self.pending_loads[rid] = (node_id, lock_params, anchor)

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        insert_result=None,
        canonical_full: torch.Tensor | None = None,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            component_canonical = canonical_full
            if phase == ExternalLinkerLoadPhase.COMMIT:
                assert insert_result.adopted_ranges is not None
                coverage_start = prefix_len - len(transfer.device_indices)
                ranges = [
                    (max(start, coverage_start), min(end, prefix_len))
                    for start, end in insert_result.adopted_ranges.get(
                        component.component_type, ()
                    )
                    if max(start, coverage_start) < min(end, prefix_len)
                ]
                indices, keys = self._select_adopted_pages(
                    transfer.device_indices,
                    ranges,
                    prefix_len,
                    transfer.keys,
                )
                if not keys:
                    continue
                transfer.device_indices = indices
                transfer.keys = keys
                component_canonical, _ = self._select_adopted_pages(
                    canonical_full, ranges, prefix_len
                )
            transfer = component.update_external_linker_load(
                phase,
                req,
                full,
                transfer,
                prefix_len,
                insert_result=insert_result,
                canonical_full=component_canonical,
            )
            if transfer is not None:
                result.append(transfer)
        return result

    def _select_adopted_pages(
        self,
        indices: torch.Tensor,
        ranges: Sequence[tuple[int, int]],
        prefix_len: int,
        keys: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        page = self.cache.page_size
        coverage_start = prefix_len - len(indices)
        pages = indices.reshape(-1, page)
        if keys is not None:
            assert len(keys) == len(pages)

        chunks = []
        selected_keys = []
        for start, end in ranges:
            start = max(start, coverage_start)
            end = min(end, prefix_len)
            if start >= end:
                continue
            assert (start - coverage_start) % page == 0
            assert (end - coverage_start) % page == 0
            first = (start - coverage_start) // page
            last = (end - coverage_start) // page
            chunks.append(pages[first:last].reshape(-1))
            if keys is not None:
                selected_keys.extend(keys[first:last])

        if not chunks:
            return indices[:0], selected_keys
        selected = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return selected, selected_keys

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            if not self.cache.resolve_node_handle(node_id).external_cache_stored:
                self._offload_node(node_id)

    def _offload_node(self, node_id: NodeId) -> None:
        cache = self.cache
        node = cache.resolve_node_handle(node_id)
        transfers = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.OFFLOAD, node, None
            )
            if transfer is not None:
                transfers.append(transfer)

        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        node.external_cache_stored = True
        self.pending_offloads.append((node_id, lock_params))

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def drain_offloads(self, finish_count: int) -> None:
        assert finish_count <= len(self.pending_offloads)
        if finish_count == 0:
            return
        successes = torch.tensor(
            [self.cache_linker.pop_completed_offload() for _ in range(finish_count)],
            dtype=torch.int,
            device="cpu",
        )
        self.cache._all_reduce_attn_groups(successes, torch.distributed.ReduceOp.MIN)
        for success in successes.tolist():
            node_id, lock_params = self.pending_offloads.pop(0)
            node = self.cache.resolve_node_handle(node_id)
            node.external_cache_stored = bool(success)
            self.cache.dec_lock_ref(node_id, lock_params)

    def num_completed_loads(self) -> int:
        return self.cache_linker.num_completed_loads()

    def take_completed_loads(self, finish_count: int) -> list[bool]:
        taken = [self.cache_linker.pop_completed_load() for _ in range(finish_count)]
        self.taken_loads.extend(taken)
        return [success for _rids, success in taken]

    def commit_completed_loads(self, successes) -> list[str]:
        assert len(successes) <= len(self.taken_loads)
        failed = []
        for success in successes:
            rids, _local_success = self.taken_loads.pop(0)
            for rid in rids:
                entry = self.pending_loads.pop(rid, None)
                if entry is None:
                    continue
                node_id, lock_params, anchor = entry
                if not success:
                    self._detach_failed_chain(rid, node_id, anchor)
                    failed.append(rid)
                self.cache.dec_lock_ref(node_id, lock_params)
        return failed

    def _detach_failed_chain(self, rid: str, node_id: NodeId, anchor: NodeId) -> None:
        self.failed_chains.setdefault(rid, []).extend(
            self.cache.tree_core.detach_external_load_chain(node_id, anchor)
        )

    def take_failed_chain(self, rid: str):
        return self.failed_chains.pop(rid, [])

    def start_layer_wise_loading(self) -> int:
        return self.cache_linker.start_layer_wise_loading()

    def wait_for_pending_loads(self) -> None:
        self.cache_linker.wait_for_pending_loads()

    # ---- lifecycle ----

    def reset(self) -> None:
        if self.pending_host_prefetch_submissions is not None:
            for job, _hit in self.pending_host_prefetch_submissions.values():
                job.cancelled.set()
            self.pending_host_prefetch_submissions.clear()
        self.cache_linker.reset()
        self.hit_markers.clear()
        if self.host_prefetch_hits is not None:
            self.host_prefetch_hits.clear()
        self._release_pending_locks()
        self.pending_offloads.clear()

    def _release_pending_locks(self) -> None:
        for node_id, lock_params, _anchor in self.pending_loads.values():
            self.cache.dec_lock_ref(node_id, lock_params)
        self.pending_loads.clear()
        self.failed_chains.clear()
        self.taken_loads.clear()

    def release_request(self, rid: str) -> None:
        # failed_chains is deliberately untouched: the chain outlives the
        # request's linker state, and cache_finished_req is what frees it.
        if self._waiting_queue_prefetch_enabled:
            self.cancel_waiting_queue_prefetch(rid)
        else:
            self.hit_markers.pop(rid, None)
        # Only a load that has not started can be cancelled here. One already
        # in flight keeps its pending_loads entry and its lock until
        # commit_completed_loads retires the batch, which the linkers guarantee
        # happens even when the transfer fails.
        if self.cache_linker.cancel_queued_load(rid):
            node_id, lock_params, anchor = self.pending_loads.pop(rid)
            self._detach_failed_chain(rid, node_id, anchor)
            self.cache.dec_lock_ref(node_id, lock_params)

    def close(self) -> None:
        self.cache_linker.close()
        self._release_pending_locks()
