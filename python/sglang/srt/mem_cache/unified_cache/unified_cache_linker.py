"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefParams,
    InsertParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


logger = logging.getLogger(__name__)


_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
    }
)


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

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> tuple[list[str], bool]:
        """Consume the oldest completed load batch and return its request IDs."""

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""

    def waiting_queue_prefetch_enabled(self) -> bool:
        """Whether waiting-queue host prefetch is enabled on every rank."""
        return False

    def waiting_queue_prefetch_limit(self) -> int:
        return 8

    def get_local_dfs_health_policy(self) -> tuple[bool, bool] | None:
        """Return a rank-local DFS policy candidate, if the backend has one."""
        return None

    def apply_dfs_health_policy(self, allow_read: bool, allow_write: bool) -> None:
        """Apply a cross-rank DFS policy, if the backend supports it."""

    def disable_waiting_queue_prefetch(self) -> None:
        """Turn waiting-queue host prefetch off before any job is submitted."""

    def prepare_load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Prepare a normal load while it can still degrade to a cache miss."""
        return False

    def submit_host_prefetch(self, rid: str, transfers: list[PoolTransfer]) -> bool:
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

    def prepare_load_async(self, rid: str, transfers: list[PoolTransfer]) -> str:
        return "ready" if self.prepare_load(rid, transfers) else "failed"

    def revalidate_host_prefetch_async(self, rid: str) -> str:
        return "ready" if self.revalidate_host_prefetch(rid) else "failed"

    def revalidate_no_prefetch_needed_async(self, rid: str) -> str:
        return "ready" if self.revalidate_no_prefetch_needed(rid) else "failed"

    def revalidate_load_async(self, rid: str, transfers: list[PoolTransfer]) -> str:
        return "ready"

    def abort_prepared_load_async(self, rid: str) -> bool:
        self.abort_prepared_load(rid)
        return True

    def wait_for_pending_loads(self) -> None:
        """Optional native-worker barrier; legacy backends wait through their layer counter."""

    def wait_for_load_batch(self, index: int) -> None:
        """Wait for one forward's load verdict; legacy backends use the full barrier."""
        if index >= 0:
            self.wait_for_pending_loads()

    def request_id(self, handle: CacheRequestHandle):
        """Backend ownership key; legacy backends retain the public request id."""
        return handle.rid


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
    request_rid: str | None = None


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    _prefetch_request_limit = 8
    _load_admissions = None
    _admitted_loads = None
    _pending_session_cleanup = None

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full and SWA tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        self.request_ids: dict[str, str] = {}
        self.load_request_rids: dict[str, str] = {}
        self.released_load_ids: set[str] = set()
        swa = cache.components.get(ComponentType.SWA)
        self._skip_swa = swa is not None and is_swa_req_ring(
            cache.token_to_kv_pool_allocator
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        self._waiting_queue_prefetch_enabled = (
            cache_linker.waiting_queue_prefetch_enabled()
        )
        if self._waiting_queue_prefetch_enabled:
            self._prefetch_request_limit = cache_linker.waiting_queue_prefetch_limit()
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
        # rid -> the chain a failed load published for it, endpoint-first.
        # The nodes hold no valid KV and are already cut out of the tree; the
        # cache frees them in that request's own cache_finished_req, which is
        # where its lock on them goes.
        self.failed_chains: dict[str, list[NodeId]] = {}
        # Load batches popped by take_completed_loads and awaiting the reduced
        # verdict in commit_completed_loads.
        self.taken_loads: list[tuple[list[str], bool]] = []
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        self._remember_request(req)
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = result.device_prefix_len
        self.hit_markers.pop(req.rid, None)
        prefetched_hit = (
            self.host_prefetch_hits.get(req.rid)
            if self._waiting_queue_prefetch_enabled
            else None
        )

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
        for component in self._components:
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
            expected_len = (
                prefetched_hit.device_hit_len + len(prefetched_hit.tail_hashes) * page
            )
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
        hit_pages = self._sync_restorable_prefix(
            self.cache_linker.lookup(
                self._backend_request_id(req.rid), lookup_transfers
            ),
            num_pages=len(tail_hashes),
            device_hit_pages=0,
        )
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
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> device, then insert ----

    def prepare_load_admission(self, req: Req) -> str:
        """Poll RPC work before allocating any device slots for the load."""
        hit = self.hit_markers.get(req.rid)
        if hit is None:
            if (
                self._load_admissions is not None
                and self._load_admissions.pop(req.rid, None) is not None
            ):
                self._abort_prepared_session(self._backend_request_id(req.rid))
            return "ready"
        if self._load_admissions is None:
            self._load_admissions, self._admitted_loads = {}, {}
        transfers = []
        constructed = True
        try:
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, hit.tail_hashes
                )
                if transfer is None:
                    constructed = False
                    break
                transfers.append(transfer)
        except Exception:
            constructed = False
            logger.warning(
                "External admission metadata construction failed", exc_info=True
            )
        signature = tuple(
            (transfer.name, tuple(transfer.keys)) for transfer in transfers
        )
        previous = self._load_admissions.setdefault(req.rid, signature)
        backend_id = self._backend_request_id(req.rid)
        mode, state = "validated", "failed"
        try:
            if (
                self._pending_session_cleanup
                and backend_id in self._pending_session_cleanup
            ):
                state = "pending"
            elif constructed and signature == previous:
                status = (
                    self.cache_linker.get_host_prefetch_status(backend_id)
                    if self._waiting_queue_prefetch_enabled
                    else None
                )
                if status == "dfs_prefetched":
                    mode = "dfs_prefetched"
                    state = self.cache_linker.revalidate_host_prefetch_async(backend_id)
                elif status == "no_prefetch_needed":
                    mode = "prepared"
                    state = self.cache_linker.revalidate_no_prefetch_needed_async(
                        backend_id
                    )
                    if state == "ready":
                        state = self.cache_linker.prepare_load_async(
                            backend_id, transfers
                        )
                elif self._waiting_queue_prefetch_enabled:
                    mode = "prepared"
                    state = self.cache_linker.prepare_load_async(backend_id, transfers)
                else:
                    state = self.cache_linker.revalidate_load_async(
                        backend_id, transfers
                    )
        except Exception:
            logger.warning(
                "External load admission RPC failed for %s", req.rid, exc_info=True
            )
        verdict = torch.tensor(
            [int(state == "pending"), int(state not in {"ready", "pending"})],
            dtype=torch.int,
        )
        self.cache._all_reduce_attn_groups(verdict, torch.distributed.ReduceOp.MAX)
        if verdict[0]:
            return "pending"
        self._load_admissions.pop(req.rid, None)
        if verdict[1]:
            self._abort_prepared_session(backend_id)
            self.cancel_waiting_queue_prefetch(req.rid)
            self.hit_markers.pop(req.rid, None)
            req.host_hit_length = req.swa_host_hit_length = (
                req.mamba_host_hit_length
            ) = 0
            return "failed"
        self._admitted_loads[req.rid] = mode
        return "ready"

    def discard_load_admission(self, req: Req) -> None:
        # The adder can reject a ready candidate without consuming its load.
        # Re-enter admission next time instead of reusing an old lease proof.
        if (
            self._admitted_loads is not None
            and self._admitted_loads.pop(req.rid, None) is not None
        ):
            self._abort_prepared_session(self._backend_request_id(req.rid))
            self.cancel_waiting_queue_prefetch(req.rid)

    def _abort_prepared_session(self, backend_id: str) -> None:
        if not self.cache_linker.abort_prepared_load_async(backend_id):
            if self._pending_session_cleanup is None:
                self._pending_session_cleanup = set()
            self._pending_session_cleanup.add(backend_id)

    def poll_session_cleanup(self) -> None:
        if self._pending_session_cleanup:
            for backend_id in tuple(self._pending_session_cleanup):
                if self.cache_linker.abort_prepared_load_async(backend_id):
                    self._pending_session_cleanup.discard(backend_id)

    def load_back(self, req: Req) -> tuple[int, NodeId]:
        cache = self.cache
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node

        admitted = (
            None
            if self._admitted_loads is None
            else self._admitted_loads.pop(req.rid, None)
        )

        prepared_for_load = False
        prefetched = (
            self.host_prefetch_hits.pop(req.rid, None) is not None
            if self._waiting_queue_prefetch_enabled
            else False
        )
        if prefetched:
            status = self.cache_linker.get_host_prefetch_status(
                self._backend_request_id(req.rid)
            )
            locally_complete = status in {"dfs_prefetched", "no_prefetch_needed"}
            ready = torch.tensor(int(locally_complete), dtype=torch.int)
            cache._all_reduce_attn_groups(ready, torch.distributed.ReduceOp.MIN)
            if int(ready.item()) == 0:
                self.cache_linker.cancel_host_prefetch(
                    self._backend_request_id(req.rid)
                )
                if admitted is not None:
                    self._abort_prepared_session(self._backend_request_id(req.rid))
                    return 0, req.last_node
                self.hit_markers[req.rid] = hit
                return self.load_back(req)

            # Every rank must reach the reduction below, so any local failure,
            # including an exception, becomes a local False.
            try:
                if status == "no_prefetch_needed":
                    # Its session was released while queued; re-check that
                    # the keys survived so eviction falls back to the normal
                    # path instead of failing the later load.
                    locally_valid = (
                        admitted is not None
                        or self.cache_linker.revalidate_no_prefetch_needed(
                            self._backend_request_id(req.rid)
                        )
                    )
                else:
                    locally_valid = status == "dfs_prefetched" and (
                        admitted is not None
                        or self.cache_linker.revalidate_host_prefetch(
                            self._backend_request_id(req.rid)
                        )
                    )
            except BaseException:
                locally_valid = False
            valid = torch.tensor(int(locally_valid), dtype=torch.int)
            cache._all_reduce_attn_groups(valid, torch.distributed.ReduceOp.MIN)
            if int(valid.item()) == 0:
                self.cache_linker.cancel_host_prefetch(
                    self._backend_request_id(req.rid)
                )
                if admitted is not None:
                    self._abort_prepared_session(self._backend_request_id(req.rid))
                    return 0, req.last_node
                self.hit_markers[req.rid] = hit
                return self.load_back(req)

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        if prefetched:
            # Build before claiming so that the claim reduction also carries
            # the construction verdict: one reduction instead of two.
            component_transfers, constructed, build_error = self._build_load_transfers(
                req,
                tail_hashes,
                prepare_session=status == "no_prefetch_needed"
                and admitted != "prepared",
            )
            prepared_locally = status == "no_prefetch_needed" and constructed
            claim_error: BaseException | None = None
            try:
                if status == "no_prefetch_needed":
                    self.cache_linker.cancel_host_prefetch(
                        self._backend_request_id(req.rid)
                    )
                    claimed = True
                else:
                    claimed = self.cache_linker.claim_ready_host_prefetch(
                        self._backend_request_id(req.rid)
                    )
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
                    release_session=prepared_locally,
                )
                if not constructed_all:
                    # Building failed somewhere: a miss, as on the normal path.
                    return 0, req.last_node
                if admitted is not None:
                    self._abort_prepared_session(self._backend_request_id(req.rid))
                    return 0, req.last_node
                # Every rank built but a claim failed: retry the normal path,
                # which builds again and agrees on its own construction.
                self.hit_markers[req.rid] = hit
                return self.load_back(req)
            prepared_for_load = True
        else:
            # Ordinary LOAD construction must agree across ranks even when
            # waiting-queue prefetch is disabled.
            component_transfers, constructed, build_error = self._build_load_transfers(
                req,
                tail_hashes,
                prepare_session=self._waiting_queue_prefetch_enabled
                and admitted != "prepared",
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
                    cancel_prefetch=False,
                    pending=build_error,
                    release_session=constructed
                    and self._waiting_queue_prefetch_enabled,
                )
                return 0, req.last_node
            # With prefetch enabled, session start/refresh already checked
            # the master on every rank.
            prepared_for_load = self._waiting_queue_prefetch_enabled

        # Keys can be evicted remotely (master memory-watermark eviction)
        # between the match-time lookup and this load-back; a stale hit would
        # then fail the async layer-wise session fatally. Re-check existence
        # here so the request degrades to a plain cache miss instead.
        revalidate = getattr(self.cache_linker, "revalidate_load", None)
        if (
            not prepared_for_load
            and admitted is None
            and revalidate is not None
            and not revalidate([transfer for _, transfer in component_transfers])
        ):
            self._update_load(
                ExternalLinkerLoadPhase.ABORT,
                req,
                component_transfers,
                prefix_len,
            )
            return 0, req.last_node

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
            self._abort_prepared_session(self._backend_request_id(req.rid))
            raise

        if cache.pp_size > 1:
            self._queue_load(
                req.rid, req.last_node, prepared_transfers, anchor=req.last_node
            )
            return len(full_transfer.device_indices), req.last_node

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable SWA range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-SWA mapping during cache.insert().
        if self._skip_swa:
            if req.kv is None:
                from sglang.srt.managers.schedule_batch import ReqKvInfo

                req.kv = ReqKvInfo(kv_allocated_len=prefix_len)
            req.kv.set_evicted_seqlen(
                ComponentType.SWA,
                max(req.kv.get_evicted_seqlen(ComponentType.SWA), prefix_len),
            )

        # Insert the newly loaded tail into the tree.
        prefix_indices = torch.cat(
            [cache.prefix_device_indices(req), full_transfer.device_indices]
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
                component_evicted_seqlens=(
                    req.kv.component_evicted_seqlens.copy()
                    if req.kv is not None
                    else {}
                ),
                inserted_len=len(hit.prefix_key),
                priority=req.priority or 0,
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

        self._queue_load(
            req.rid,
            insert_result.last_device_node,
            load_transfers,
            anchor=req.last_node,
        )

        cache.tree_core.mark_external_cache_stored_path(
            insert_result.last_device_node, req.last_node
        )
        return len(canonical_tail), insert_result.last_device_node

    def _queue_load(
        self,
        rid: str,
        node_id: NodeId,
        transfers: list[PoolTransfer],
        *,
        anchor: NodeId,
    ) -> None:
        if not transfers:
            self._abort_prepared_session(self._backend_request_id(rid))
            return
        backend_id = self._backend_request_id(rid)
        assert backend_id not in self.pending_loads
        lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.load(backend_id, transfers)
        except BaseException:
            self._detach_failed_chain(rid, node_id, anchor)
            self.cache.dec_lock_ref(node_id, lock_params)
            self._abort_prepared_session(backend_id)
            raise
        if not queued:
            self._detach_failed_chain(rid, node_id, anchor)
            self.cache.dec_lock_ref(node_id, lock_params)
            self._abort_prepared_session(backend_id)
            raise RuntimeError(f"Failed to queue the linker load for rid={rid!r}.")
        self.pending_loads[backend_id] = (node_id, lock_params, anchor)
        if not hasattr(self, "load_request_rids"):
            self.load_request_rids = {}
        self.load_request_rids[backend_id] = rid

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
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is not None:
                if self._skip_swa:
                    transfers = [t for t in transfers if t.name != PoolName.SWA]
                self._offload_node(node_id, transfers)

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def num_completed_loads(self) -> int:
        return self.cache_linker.num_completed_loads()

    def drain_loads(self, finish_count: int) -> None:
        self.commit_completed_loads(self.take_completed_loads(finish_count))

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, success
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        return self.cache_linker.start_layer_wise_loading()

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

    def _release_pending_locks(self) -> None:
        if self._load_admissions is not None:
            self._load_admissions.clear()
            self._admitted_loads.clear()
        if self._pending_session_cleanup is not None:
            self._pending_session_cleanup.clear()
        for node_id, lock_params, _anchor in self.pending_loads.values():
            self.cache.dec_lock_ref(node_id, lock_params)
        self.pending_loads.clear()
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()
        self.failed_chains.clear()
        self.taken_loads.clear()
        self.request_ids.clear()
        self.load_request_rids.clear()
        self.released_load_ids.clear()

    def release_request(self, rid: str) -> None:
        if isinstance(rid, CacheRequestHandle):
            handle = rid
            rid = handle.rid
            expected = self.cache_linker.request_id(handle)
            if getattr(self, "request_ids", {}).get(rid) != expected:
                return
        backend_id = self._backend_request_id(rid)
        preparing = (
            self._load_admissions is not None
            and self._load_admissions.pop(rid, None) is not None
        )
        admitted = (
            self._admitted_loads is not None
            and self._admitted_loads.pop(rid, None) is not None
        )
        if preparing or admitted:
            self._abort_prepared_session(backend_id)
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
        if self.cache_linker.cancel_queued_load(backend_id):
            node_id, lock_params, _anchor = self.pending_loads.pop(backend_id)
            self._detach_failed_chain(rid, node_id, _anchor)
            self.cache.dec_lock_ref(node_id, lock_params)
        if backend_id in self.pending_loads:
            self.released_load_ids.add(backend_id)
        if backend_id not in self.pending_loads:
            getattr(self, "load_request_rids", {}).pop(backend_id, None)
            getattr(self, "request_ids", {}).pop(rid, None)

    def close(self) -> None:
        self.cache_linker.close()
        self._release_pending_locks()

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
        status = self.cache_linker.get_host_prefetch_status(
            self._backend_request_id(rid)
        )
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
        self.hit_markers.pop(rid, None)
        if not self._waiting_queue_prefetch_enabled:
            return
        tracked = self.host_prefetch_hits.pop(rid, None) is not None
        pending = self.pending_host_prefetch_submissions.pop(rid, None)
        if pending is not None:
            pending[0].cancelled.set()
        if tracked or pending is not None:
            self.cache_linker.cancel_host_prefetch(self._backend_request_id(rid))

    def tracked_host_prefetch_rids(self) -> set[str]:
        if not self._waiting_queue_prefetch_enabled:
            return set()
        return set(self.pending_host_prefetch_submissions) | set(
            self.host_prefetch_hits
        )

    def prepare_host_prefetch(self, req: Req) -> PreparedHostPrefetch | None:
        """Build a request-local snapshot without issuing any collectives."""
        self._remember_request(req)
        if len(self.tracked_host_prefetch_rids()) >= self._prefetch_request_limit:
            return None
        hit = self.hit_markers.get(req.rid)
        transfers = []
        locally_eligible = hit is not None
        if hit is not None:
            try:
                for component in self._components:
                    transfer = component.build_external_linker_transfer(
                        LinkerTransferPhase.LOOKUP, None, hit.tail_hashes
                    )
                    if transfer is None:
                        locally_eligible = False
                        break
                    # LOOKUP transfers are CPU metadata only. Copy their
                    # mutable key lists so the worker owns its snapshot.
                    transfers.append(
                        replace(
                            transfer,
                            keys=list(transfer.keys)
                            if transfer.keys is not None
                            else None,
                        )
                    )
            except Exception:
                locally_eligible = False

        if not locally_eligible:
            return None
        job = PreparedHostPrefetch(
            rid=self._backend_request_id(req.rid),
            request_rid=req.rid,
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
        request_rid = job.request_rid or job.rid
        pending = self.pending_host_prefetch_submissions.get(request_rid)
        if pending is None or pending[0] is not job:
            if submitted:
                job.cache_linker.cancel_host_prefetch(job.rid)
            return False

        del self.pending_host_prefetch_submissions[request_rid]
        hit = pending[1]
        if submitted and not job.cancelled.is_set():
            self.host_prefetch_hits[request_rid] = hit
            return True
        if submitted:
            job.cache_linker.cancel_host_prefetch(job.rid)
        return False

    def prefetch_to_host(self, req: Req) -> bool:
        if not self._waiting_queue_prefetch_enabled:
            return False
        self._remember_request(req)
        hit = self.hit_markers.get(req.rid)
        transfers = []
        locally_eligible = hit is not None
        if hit is not None:
            try:
                for component in self._components:
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
            submitted = self.cache_linker.submit_host_prefetch(
                self._backend_request_id(req.rid), transfers
            )
        except BaseException:
            submitted = False

        globally_submitted = torch.tensor(int(submitted), dtype=torch.int)
        self.cache._all_reduce_attn_groups(
            globally_submitted, torch.distributed.ReduceOp.MIN
        )
        if int(globally_submitted.item()) == 0:
            self.cache_linker.cancel_host_prefetch(self._backend_request_id(req.rid))
            return False

        self.host_prefetch_hits[req.rid] = hit
        return True

    def has_pending_operations(self) -> bool:
        return bool(
            self.pending_loads
            or self._load_admissions
            or self._pending_session_cleanup
            or self.pending_offloads
            or self.pending_host_prefetch_submissions
            or self.host_prefetch_hits
        )

    def _build_load_transfers(
        self, req: Req, tail_hashes, *, prepare_session: bool = False
    ) -> tuple[list[tuple[TreeComponent, PoolTransfer]], bool, BaseException | None]:
        """Build LOAD transfers and optionally prepare their session locally.

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
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, tail_hashes
                )
                if transfer is None:
                    return component_transfers, False, None
                component_transfers.append((component, transfer))
        except BaseException as error:
            return component_transfers, False, error
        if prepare_session:
            try:
                prepared = self.cache_linker.prepare_load(
                    self._backend_request_id(req.rid),
                    [transfer for _, transfer in component_transfers],
                )
            except BaseException:
                logger.warning(
                    "External cache session preparation failed", exc_info=True
                )
                prepared = False
            if not prepared:
                logger.warning(
                    "External cache session preparation failed for rid=%s, "
                    "degrading request to cache miss",
                    req.rid,
                )
                return component_transfers, False, None
        return component_transfers, True, None

    def _abort_disagreed_load(
        self,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        cancel_prefetch: bool,
        pending: BaseException | None,
        release_session: bool = False,
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
                self.cache_linker.cancel_host_prefetch(
                    self._backend_request_id(req.rid)
                )
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    logger.error(
                        "Cancelling the prefetch of rid=%s also failed",
                        req.rid,
                        exc_info=error,
                    )
        if release_session:
            try:
                self._abort_prepared_session(self._backend_request_id(req.rid))
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    logger.error(
                        "Releasing the failed load session also failed", exc_info=error
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

    def take_completed_loads(self, finish_count: int) -> list[bool]:
        """Pop finished load batches, reporting the local verdict only.

        Split from the commit half so the caller can MIN-reduce the verdict
        across the attention group first. A load failure is rank-local -- one
        rank's batch_get can miss a key the others read fine -- so a rank that
        poisons and aborts on its own verdict diverges from the ranks that do
        not: the request is aborted on some ranks and served on the rest, and
        the rank owning the output stream emits KV that never arrived.

        This mirrors take_completed_offloads / commit_completed_offloads, which
        already reduce their verdict for the same reason.
        """
        taken = [self.cache_linker.pop_completed_load() for _ in range(finish_count)]
        taken = [
            item
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], bool)
            else (item, True)
            for item in taken
        ]
        self.taken_loads.extend(taken)
        return [success for _rids, success in taken]

    def commit_completed_loads(self, successes: Sequence[bool]) -> list[str]:
        """Apply the reduced verdict; return the rids of the failed batches.

        A failed chain is cut out of the tree and filed under its rid, which
        keeps its unfilled pages out of the store and out of every walk. The
        nodes themselves go when that request runs its own cache_finished_req.
        """
        assert len(successes) <= len(self.taken_loads)
        failed: list[str] = []
        for success in successes:
            rids, _local_success = self.taken_loads.pop(0)
            for backend_id in rids:
                entry = self.pending_loads.pop(backend_id, None)
                rid = getattr(self, "load_request_rids", {}).pop(backend_id, backend_id)
                if entry is None:
                    # Released early by release_request while queued.
                    continue
                node_id, lock_params, anchor = entry
                if not success:
                    self._detach_failed_chain(rid, node_id, anchor)
                    if self.request_ids.get(rid, backend_id) == backend_id:
                        failed.append(rid)
                self.cache.dec_lock_ref(node_id, lock_params)
                if backend_id in self.released_load_ids:
                    self.released_load_ids.discard(backend_id)
                    if self.request_ids.get(rid) == backend_id:
                        self.request_ids.pop(rid, None)
        return failed

    def _detach_failed_chain(self, rid: str, node_id: NodeId, anchor: NodeId) -> None:
        """Cut the chain this load published out of the tree, endpoint first.

        ``external_cache_stored`` is left alone: the chain really did come from
        the store, and clearing it is what makes the unfilled pages eligible
        for write-through -- the opposite of what is wanted.

        Detaching, rather than only flagging, is what keeps the chain from
        being served: match_prefix reads no such flag, so a merely flagged
        chain stayed matchable until the purge managed to drop it, and the
        first request that matched it gave it a device child and blocked the
        reclaim for good.

        The whole chain is filed, not just its endpoint. Deleting the endpoint
        does not cascade: ``_iteratively_delete_tombstone_leaf`` stops at the
        first ancestor still holding a device value, and every node this load
        just filled has one. Endpoint-first order matters too -- a parent only
        becomes a device leaf once its child is gone.

        Filed under the rid because that request's ``cache_finished_req`` is
        the one point where the whole chain becomes reclaimable: Full is a
        path-unlock, so its single ``dec_lock_ref`` drops ``lock_ref`` on every
        node of the chain at once.
        """
        self.failed_chains.setdefault(rid, []).extend(
            self.cache.tree_core.detach_external_load_chain(node_id, anchor)
        )

    def take_failed_chain(self, rid: str) -> list[NodeId]:
        """Pop the chain filed for this rid, if a load failed for it."""
        return self.failed_chains.pop(rid, [])

    def wait_for_pending_loads(self) -> None:
        self.cache_linker.wait_for_pending_loads()

    def wait_for_load_batch(self, index: int) -> None:
        self.cache_linker.wait_for_load_batch(index)

    def _remember_request(self, req: Req) -> None:
        handle = getattr(req, "cache_request_handle", None)
        backend_id = req.rid if handle is None else self.cache_linker.request_id(handle)
        if not hasattr(self, "request_ids"):
            self.request_ids = {}
        previous = self.request_ids.get(req.rid)
        if previous is not None and previous != backend_id:
            preparing = (
                self._load_admissions is not None
                and self._load_admissions.pop(req.rid, None) is not None
            )
            admitted = (
                self._admitted_loads is not None
                and self._admitted_loads.pop(req.rid, None) is not None
            )
            if preparing or admitted:
                self._abort_prepared_session(previous)
            self.cancel_waiting_queue_prefetch(req.rid)
        self.request_ids[req.rid] = backend_id

    def _backend_request_id(self, rid: str) -> str:
        return getattr(self, "request_ids", {}).get(rid, rid)
