from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from queue import Empty, Queue

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.mooncake_store.compat import (
    disable_dfs_prefetch,
    resolve_read_layout_api,
    resolve_read_plan_options,
    supports_feature,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    DFS_BATCH_GET_TIMEOUT,
)
from sglang.srt.mem_cache.unified_cache.linker_fault_injection import (
    arm_load_failure_injection,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_model,
    get_observability,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
_READ_PLAN_RC_PATTERN = re.compile(r"(?:^|\s)rc=(-?\d+)(?=\s|$)")
device_module = get_device_module()


def _get_mooncake_storage_metrics_dp_rank(server_args, params) -> int:
    return params.dp_rank


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


def _session_refresh_age_config() -> float:
    """Refresh-skip age in seconds from the environment.

    The master's TTL is not visible to the client, so the deployment declares
    it; an age that would not leave any of that TTL disables skipping.
    """
    age_s = float(envs.SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S.get())
    lease_ttl_s = float(envs.SGLANG_MOONCAKE_LEASE_TTL_S.get())
    if not math.isfinite(age_s) or age_s < 0:
        raise ValueError(
            "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S must be a finite value >= 0"
        )
    if not math.isfinite(lease_ttl_s) or lease_ttl_s <= 0:
        raise ValueError("SGLANG_MOONCAKE_LEASE_TTL_S must be a finite value > 0")
    if age_s >= lease_ttl_s:
        logger.warning(
            "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S=%.3f is not below the lease "
            "TTL %.3f s; disabling refresh skipping",
            age_s,
            lease_ttl_s,
        )
        age_s = 0.0
    return age_s


class _SessionJob(Future):
    def __init__(self):
        super().__init__()
        self.valid_until = float("inf")


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int, on_layer_ready=None):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        # Batch indices already logged: one line per failed load, not per layer.
        self.reported: set[int] = set()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        # A private copy, because wait_until clears the traceback of what it
        # catches and the caller still needs its own exception for the
        # logger.exception() that follows.
        failure = RuntimeError(f"{type(error).__name__}: {error}")
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(failure)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        ready = True
        try:
            futures[threshold].result()
        except BaseException as error:
            ready = False
            # Never raise: this runs inside the model forward, where nothing
            # catches before the scheduler's top-level handler and the whole
            # engine goes down. The failure travels out through
            # pop_completed_load(), which aborts the affected requests.
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake layer-wise KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # Every layer re-raises this same exception object, and each
            # appended traceback entry roots a frame chain that pins that
            # layer's activations: +31.9 GiB on one faulted 61-layer forward at
            # --chunked-prefill-size 2048, until the engine OOMed. Nothing
            # reads the traceback.
            error.__traceback__ = None
        finally:
            if self.on_layer_ready is not None:
                self.on_layer_ready(index, threshold, ready)
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)
                self.reported.discard(index)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()
        self.reported.clear()


class ReadPlanLoadCounter:
    """Publish one Mooncake ReadPlan; wait for each layer without holding the GIL."""

    def __init__(self, num_layers: int, on_layer_ready=None):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = self.consumer_index = -1
        self.plans: dict[int, Future] = {}
        # Batch indices already logged: one line per failed plan, not per layer.
        self.reported: set[int] = set()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.plans[self.producer_index] = Future()
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def bind(self, index: int, plan) -> None:
        self.plans[index].set_result(plan)

    def fail(self, index: int, error: BaseException) -> None:
        future = self.plans.get(index)
        if future is not None and not future.done():
            future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        future = self.plans.get(index)
        if future is None:
            return
        ready = True
        try:
            future.result().wait(threshold)
        except BaseException as error:
            ready = False
            # Do not raise from the model forward. A rank-local failure here can
            # strand peer TP/CP ranks in a later model collective and brings the
            # scheduler down. load_layer_wise reports False through the linker
            # completion queue; the cache MIN-reduces that verdict across the
            # attention group and aborts the affected requests after forward.
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake ReadPlan KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # Repeated waits otherwise grow a traceback chain that keeps each
            # layer's activations alive until the failed plan is retired.
            error.__traceback__ = None
        finally:
            if self.on_layer_ready is not None:
                self.on_layer_ready(index, threshold, ready)
            if threshold == self.num_layers - 1:
                self.plans.pop(index, None)
                self.reported.discard(index)

    def reset(self) -> None:
        self.producer_index = self.consumer_index = -1
        self.plans.clear()
        self.reported.clear()


class _CompletedReadPlan:
    def wait(self, group):
        pass


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self._close_workers_stopped = False
        self._closed = False
        self.host_prefetch_enabled = bool(
            get_memory().mooncake_enable_waiting_queue_dfs_prefetch
        )
        if self.host_prefetch_enabled and (
            get_schedule().schedule_policy != "fcfs"
            or get_disagg().disaggregation_mode not in ("null", "prefill")
            or params.pp_size != 1
        ):
            self.host_prefetch_enabled = False
            logger.warning(
                "Waiting-queue DFS prefetch disabled: requires FCFS scheduling, "
                "a prefill or non-disaggregated role, and pp_size=1."
            )
        self.page_size = params.page_size
        self.page_wise_load_threshold = get_memory().mooncake_page_wise_load_threshold
        self.enable_page_wise_load = get_memory().mooncake_enable_page_wise_load
        if self.page_wise_load_threshold <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-threshold must be positive, got "
                f"{self.page_wise_load_threshold}."
            )
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        self._load_revalidation_groups = ()
        if self.pool_group.storage_layout_tag:
            self._load_revalidation_groups = tuple(
                group
                for group in (params.attn_cp_cache_group, params.attn_tp_cache_group)
                if group is not None
                and torch.distributed.get_world_size(group=group) > 1
            )

        tp_rank = 0
        tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        self.tp_rank = tp_rank
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        extra_config.update(get_memory().unified_cache_external_linker_config or {})
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
            dp_rank=params.dp_rank,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.dfs_health import (
                resolve_dfs_health_config,
            )
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(
                storage_config,
                mem_pool=None,
                enable_client_http_server=params.enable_metrics,
                enable_dfs_prefetch=self.host_prefetch_enabled,
                dfs_health_config=resolve_dfs_health_config(server_args=get_memory()),
            )
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        if self.pool_group.storage_layout_tag:
            storage_suffix = f"{self.pool_group.storage_layout_tag}_{storage_suffix}"
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )
        self.read_plan_enabled = envs.SGLANG_MOONCAKE_READ_PLAN.get()
        self.read_plan_reuse_ranges = envs.SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES.get()
        self.read_plan_enabled, self.read_plan_reuse_ranges = resolve_read_plan_options(
            self.storage.store, self.read_plan_enabled, self.read_plan_reuse_ranges
        )
        self.read_layout_enabled = (
            supports_feature(self.storage.store, "read_layout")
            and resolve_read_layout_api(self.storage.store) is not None
        )
        # Skip the admission-time lease refresh of a prefetched request whose
        # oldest session key was leased less than this many seconds ago. The
        # master never evicts a leased object, so the refresh only buys lease
        # time; 0 keeps the refresh unconditional. Must leave enough of the
        # deployment's lease TTL to cover admission through load completion.
        self.host_prefetch_limit = 64
        self.host_prefetch_worker_count = 1
        if self.host_prefetch_enabled:
            self.session_refresh_age_s = _session_refresh_age_config()
            self._disable_refresh_skip_for_groups()
            self.host_prefetch_limit = int(
                get_memory().mooncake_waiting_queue_dfs_prefetch_max_requests
            )
            self.host_prefetch_worker_count = int(
                get_memory().mooncake_waiting_queue_dfs_prefetch_workers
            )
            if self.host_prefetch_limit <= 0:
                raise ValueError(
                    "--mooncake-waiting-queue-dfs-prefetch-max-requests must be "
                    f"positive, got {self.host_prefetch_limit}."
                )
            if self.host_prefetch_worker_count <= 0:
                raise ValueError(
                    "--mooncake-waiting-queue-dfs-prefetch-workers must be "
                    f"positive, got {self.host_prefetch_worker_count}."
                )
            self.host_prefetch_worker_count = min(
                self.host_prefetch_worker_count, self.host_prefetch_limit
            )
        if self.read_plan_reuse_ranges and not self.read_plan_enabled:
            raise ValueError(
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES requires "
                "SGLANG_MOONCAKE_READ_PLAN=1"
            )
        if self.read_plan_reuse_ranges and self.enable_page_wise_load:
            raise ValueError(
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES is incompatible with "
                "page-wise Mooncake loads"
            )
        if self.read_plan_enabled:
            logger.info(
                "Mooncake ReadPlan enabled; address reuse=%s",
                self.read_plan_reuse_ranges,
            )
        prefetch_enabled = self._resolve_host_prefetch_enabled(
            params, requested=self.host_prefetch_enabled
        )
        if self.host_prefetch_enabled and not prefetch_enabled:
            self.disable_waiting_queue_prefetch()
        self.host_prefetch_enabled = prefetch_enabled

        self.storage_metrics_collector = None
        if params.enable_metrics:
            labels = {
                "storage_backend": "mooncake_direct",
                "tp_rank": tp_rank,
                "dp_rank": _get_mooncake_storage_metrics_dp_rank(server_args, params),
                "pp_rank": params.pp_rank,
                "pp_size": params.pp_size,
                "attn_cp_rank": params.attn_cp_rank,
                "attn_cp_size": params.attn_cp_size,
            }
            if get_observability().extra_metric_labels:
                labels.update(get_observability().extra_metric_labels)
            collector_cls = resolve_collector_class(
                STAT_LOGGER_ROLE_STORAGE, StorageMetricsCollector
            )
            self.storage_metrics_collector = collector_cls(labels=labels)

        self.register_buffers()
        self.mla_broadcaster = None
        self.tp_group = tp_group
        self.broadcast_loads = {}
        self.broadcast_events = []
        self.broadcast_outcomes = {}
        self.follower_cleanup_outcomes = {}
        self.load_batch_cleanup_futures = {}
        self.broadcast_failed = set()
        if get_memory().enable_linker_mla_dedup and rank_replicated and tp_size > 1:
            self.mla_broadcaster = LinkerMLADedupBroadcaster.build(
                self.pool_group, params.tp_cache_group, params.attn_tp_cache_group
            )
        on_layer_ready = self._broadcast_loaded_layers if self.mla_broadcaster else None
        if self.read_plan_enabled:
            self.layer_done_counter = ReadPlanLoadCounter(
                self.num_layers, on_layer_ready
            )
        else:
            self.layer_done_counter = LayerWiseLoadCounter(
                self.num_layers, on_layer_ready
            )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.pending_load_tokens: dict[str, int] = {}
        self.pending_load_started: dict[str, float] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        # (rids, success) per started load batch. Pushed from a finally so a
        # failed batch still releases the tree-side locks it pinned.
        self.completed_loads: Queue[tuple[list[str], bool]] = Queue()
        self.load_batch_futures: dict[int, Future] = {}
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, float, object] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.host_prefetch_threads: list[threading.Thread] = []
        self._init_session_runtime()
        if self.host_prefetch_enabled:
            self.host_prefetch_entries: dict[str, dict[str, object]] = {}
            self.host_prefetch_generation = 0
            self.host_prefetch_queue: Queue[tuple[str, list[PoolTransfer]] | None] = (
                Queue()
            )
            for worker_id in range(self.host_prefetch_worker_count):
                thread = threading.Thread(
                    target=self.host_prefetch_thread_func,
                    daemon=True,
                    name=f"mooncake-host-prefetch-tp{tp_rank}-{worker_id}",
                )
                self.host_prefetch_threads.append(thread)
                thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def _init_session_runtime(self):
        self.session_lease_ttl_s = float(envs.SGLANG_MOONCAKE_LEASE_TTL_S.get())
        if not math.isfinite(self.session_lease_ttl_s) or self.session_lease_ttl_s <= 0:
            raise ValueError("SGLANG_MOONCAKE_LEASE_TTL_S must be finite and positive")
        self.session_admission_recheck_s = min(self.session_lease_ttl_s / 2, 1.0)
        self.prepared_load_sessions: dict[str, list[str]] = {}
        self.session_refcounts: dict[str, int] = {}
        self.session_sources: dict[str, str] = {}
        # Conservative base of the RPC that last granted a read lease.
        self.session_lease_base: dict[str, float] = {}
        self.session_lock = threading.Lock()
        self.session_key_locks = {}
        self.preparing_load_sessions = {}
        self.ending_load_sessions = set()
        self.failed_session_keys = set()
        self.failed_session_cleanups = set()
        self.session_cleanup_retry_at = {}
        self.session_cleanup_retry_delay = {}
        self.session_cleanup_log_at = 0.0
        self.session_cleanup_log_failures = 0
        self.session_cleanup_stats = {"failures": 0, "retries": 0, "recovered": 0}
        self.session_jobs = {}
        self.session_admission_deadlines = {}
        self.session_admission_timeouts = set()
        self.cancelled_session_jobs = set()
        self.cancelled_load_session_cleanup = set()
        self.host_prefetch_session_cleanup = {}
        self.host_prefetch_lock = threading.Lock()
        self.session_executor = ThreadPoolExecutor(
            max_workers=self.host_prefetch_worker_count,
            thread_name_prefix="mooncake-session",
        )

    # Defaults for instances built without __init__ (unit-test fixtures).
    session_refresh_age_s = 0.0

    def _disable_refresh_skip_for_groups(self) -> None:
        group_semantics = getattr(self.storage, "_can_use_group_semantics", None)
        if (
            self.session_refresh_age_s > 0
            and callable(group_semantics)
            and group_semantics()
        ):
            # A grouped object's lease is only extended once half of it is
            # left, yet every reply reports the full TTL: the client-side
            # deadline can run up to TTL/2 past the master's.
            logger.warning(
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S is ignored with Mooncake "
                "group semantics"
            )
            self.session_refresh_age_s = 0.0

    def _lease_bases(self) -> dict[str, float]:
        return self.session_lease_base

    def _local_host_prefetch_capability(self, requested: bool) -> tuple[bool, str]:
        """Return whether this rank can run waiting-queue DFS prefetch."""
        if not requested:
            return False, "disabled by --mooncake-enable-waiting-queue-dfs-prefetch"
        store = self.storage.store
        missing = [
            name
            for name in (
                "batch_get_session_start_with_sources",
                "batch_get_session_prefetch",
                "batch_get_session_refresh",
                "dfs_prefetch_arena_available",
                "disable_dfs_prefetch",
            )
            if not callable(getattr(store, name, None))
        ]
        if missing:
            return False, "Mooncake package lacks " + ", ".join(missing)
        if not supports_feature(store, "dfs_prefetch"):
            return False, "Mooncake client mode does not support waiting-queue prefetch"
        try:
            if store.dfs_prefetch_arena_available():
                return True, "ready"
            status = getattr(store, "dfs_prefetch_arena_status", None)
            reason = status() if callable(status) else "unavailable"
        except Exception as exc:
            reason = f"arena query failed: {exc!r}"
        return False, (
            f"DFS prefetch arena unavailable ({reason}); set "
            "MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES and MC_STORE_DFS_PREFETCH_CAPACITY_RATIO on prefill"
        )

    def _resolve_host_prefetch_enabled(
        self, params: CacheInitParams, *, requested: bool
    ) -> bool:
        """Agree on waiting-queue prefetch once, before any worker starts.

        Every rank that builds this linker takes part unconditionally, so the
        scheduler's per-request prefetch collectives are entered by all ranks
        or by none. The groups and their order match
        ``UnifiedRadixCache._all_reduce_attn_groups``.
        """
        local_ok, reason = self._local_host_prefetch_capability(requested)
        verdict = torch.tensor([int(local_ok)], dtype=torch.int)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            reduced = False
            for group in (params.attn_cp_cache_group, params.attn_tp_cache_group):
                if (
                    group is not None
                    and torch.distributed.get_world_size(group=group) > 1
                ):
                    torch.distributed.all_reduce(
                        verdict, op=torch.distributed.ReduceOp.MIN, group=group
                    )
                    reduced = True
            tp_group = params.tp_cache_group
            if (
                not reduced
                and tp_group is not None
                and torch.distributed.get_world_size(group=tp_group) > 1
            ):
                torch.distributed.all_reduce(
                    verdict, op=torch.distributed.ReduceOp.MIN, group=tp_group
                )
        enabled = bool(verdict.item())
        if requested and not enabled:
            logger.warning(
                "Mooncake waiting-queue DFS prefetch disabled on all ranks; "
                "local status: %s (if ready, another rank is unavailable)",
                reason,
            )
        elif enabled:
            logger.info("Mooncake waiting-queue DFS prefetch enabled")
        return enabled

    def waiting_queue_prefetch_enabled(self) -> bool:
        return self.host_prefetch_enabled

    def get_local_dfs_health_policy(self) -> tuple[bool, bool] | None:
        return self.storage.get_local_dfs_health_policy()

    def apply_dfs_health_policy(self, allow_read: bool, allow_write: bool) -> None:
        self.storage.apply_dfs_health_policy(allow_read, allow_write)

    def _report_dfs_operation_timeout(self, capability: str, error: int) -> None:
        self.storage._report_dfs_operation_timeout(capability, error)

    def _dfs_read_policy_allows_read(self) -> bool:
        getter = getattr(self.storage, "get_local_dfs_health_policy", None)
        if not callable(getter):
            return True
        policy = getter()
        return policy is None or (bool(policy[0]) and bool(policy[1]))

    def _reselect_read_plan_sessions_if_needed(self, layouts) -> None:
        """Move borrowed ReadPlan sessions off DFS after read policy closes."""
        if not self.host_prefetch_enabled or self._dfs_read_policy_allows_read():
            return

        keys = list(dict.fromkeys(key for layout in layouts for key in layout[0]))
        if not keys:
            return

        with self._session_key_guard(keys):
            try:
                results, sources = (
                    self.storage.store.batch_get_session_start_with_sources(keys)
                )
                results = list(results)
                sources = list(sources)
            except Exception as error:
                raise RuntimeError(
                    "Mooncake ReadPlan failed to reselect sessions after DFS "
                    "read policy was disabled"
                ) from error

            all_memory = len(sources) == len(keys) and all(
                str(source).lower() == "memory" for source in sources
            )
            if (
                len(results) != len(keys)
                or any(result != 0 for result in results)
                or not all_memory
            ):
                raise RuntimeError(
                    "Mooncake ReadPlan has no usable memory replica after DFS "
                    "read policy was disabled: "
                    f"results={results}, sources={sources}"
                )

            lease_base = time.monotonic()
            with self.session_lock:
                bases = self._lease_bases()
                for index, key in enumerate(keys):
                    if key not in self.session_refcounts:
                        continue
                    source = sources[index] if index < len(sources) else "memory"
                    self.session_sources[key] = str(source).lower()
                    bases[key] = lease_base

            logger.warning(
                "Mooncake ReadPlan reselected %d borrowed session key(s) to "
                "memory because DFS reads are disabled",
                len(keys),
            )

    def disable_waiting_queue_prefetch(self) -> None:
        # Startup only, before any request or prefetch submission. Keep the
        # native client and its pinned arena consistent with the final switch.
        disable_dfs_prefetch(self.storage.store)
        self.host_prefetch_enabled = False

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def revalidate_load(self, transfers: list[PoolTransfer]) -> bool:
        try:
            valid = self._revalidate_load_local(transfers)
        except BaseException:
            logger.exception("Mooncake direct linker load revalidation error")
            valid = False
        groups = getattr(self, "_load_revalidation_groups", ())
        if groups:
            # LayerSplit shards can be evicted independently after lookup.
            # All CP ranks must either restore or recompute the same prefix.
            verdict = torch.tensor([int(valid)], dtype=torch.int)
            for group in groups:
                torch.distributed.all_reduce(
                    verdict, op=torch.distributed.ReduceOp.MIN, group=group
                )
            valid = bool(verdict.item())
        return valid

    def _revalidate_load_local(self, transfers: list[PoolTransfer]) -> bool:
        """Re-check remote existence before the scheduler commits device
        slots to the async layer-wise load.

        The match-time lookup and ``batch_get_session_start`` can be minutes
        apart; master-side memory-watermark eviction in that window expires
        replicas and would fail the layer-wise load fatally. Re-checking here
        lets the caller degrade the request to a plain cache miss instead.

        Errors fail closed so a failed session check cannot be admitted again.
        """
        resolved = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not resolved:
            return True
        key_strs: list[str] = []
        for transfer in resolved:
            component_keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            key_strs.extend(self.storage._tag_keys(component_keys))
        if not key_strs:
            return True
        exist = self.storage._batch_exist(key_strs)
        if len(exist) != len(key_strs):
            return False
        missing = [key for key, state in zip(key_strs, exist) if state != 1]
        if missing:
            logger.warning(
                "Mooncake direct linker load revalidation failed: "
                "missing=%d/%d keys (master eviction race), "
                "degrading request to cache miss",
                len(missing),
                len(key_strs),
            )
            failed_get_cache = getattr(self.storage, "failed_get_cache", None)
            if failed_get_cache is not None:
                failed_get_cache.update_batch([], missing)
            return False
        return True

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        if not self.host_prefetch_enabled:
            with self.session_lock:
                validation = self.session_jobs.get(("refresh", rid))
                if validation is not None:
                    if not validation.done() or not validation.result():
                        return False
                    # Admission's local existence proof has now been consumed.
                    del self.session_jobs[("refresh", rid)]
                self.session_admission_deadlines.pop(rid, None)
                self.session_admission_timeouts.discard(rid)
        self.pending_loads[rid] = expanded
        if self.host_prefetch_enabled:
            with self.host_prefetch_lock:
                entry = self.host_prefetch_entries.get(rid)
                if entry is not None and entry.get("state") == "claimed":
                    entry["state"] = "loading"
        logical_pages = {
            page_key for transfer in expanded for page_key in transfer.keys
        }
        pending_load_tokens = getattr(self, "pending_load_tokens", None)
        if pending_load_tokens is None:
            pending_load_tokens = self.pending_load_tokens = {}
        pending_load_tokens[rid] = len(logical_pages) * self.page_size
        pending_load_started = getattr(self, "pending_load_started", None)
        if pending_load_started is None:
            pending_load_started = self.pending_load_started = {}
        pending_load_started[rid] = time.perf_counter()
        return True

    def prepare_load(
        self, rid: str, transfers: list[PoolTransfer], *, force_refresh=False
    ) -> bool:
        """Acquire key-indexed Mooncake sessions for a future range read."""
        if not self.host_prefetch_enabled:
            return False
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        return self._prepare_expanded_load(rid, expanded, force_refresh=force_refresh)

    def _session_job(self, kind, rid, operation, *, renew=None):
        """Bound queued, running and unconsumed results by request credits."""
        key = (kind, rid)
        with self.session_lock:
            if kind in {"prepare", "refresh"}:
                now = time.monotonic()
                self.session_admission_deadlines.setdefault(
                    rid, now + self.session_lease_ttl_s
                )
            if kind != "end" and rid in self.cancelled_session_jobs:
                return None
            future = self.session_jobs.get(key)
            if future is not None:
                if (
                    kind not in {"prepare", "refresh"}
                    or not future.done()
                    or not future.result()
                    or time.monotonic() < future.valid_until
                ):
                    return future
                del self.session_jobs[key]
                self.session_admission_deadlines[rid] = (
                    time.monotonic() + self.session_lease_ttl_s
                )
                self.session_admission_timeouts.discard(rid)
                if renew is not None:
                    operation = renew
            if (
                kind in {"prepare", "refresh"}
                and time.monotonic() >= self.session_admission_deadlines[rid]
            ):
                return None
            owners = {
                owner for job_kind, owner in self.session_jobs if job_kind == kind
            }
            if kind in {"prepare", "refresh"}:
                owners.update(self.failed_session_cleanups)
            if len(owners) >= self.host_prefetch_limit:
                return None
            future = _SessionJob()
            self.session_jobs[key] = future

        def run():
            try:
                started = time.monotonic()
                result = operation()
                with self.session_lock:
                    if result and kind in {"prepare", "refresh"}:
                        now = time.monotonic()
                        future.valid_until = min(
                            now + self.session_admission_recheck_s,
                            started + self.session_lease_ttl_s / 2,
                        )
                        owner = rid
                        if kind == "refresh" and self.host_prefetch_enabled:
                            with self.host_prefetch_lock:
                                entry = self.host_prefetch_entries.get(rid)
                                if entry is not None:
                                    owner = str(entry["session_rid"])
                        for session_key in self.prepared_load_sessions.get(owner, ()):
                            base = self.session_lease_base.get(session_key)
                            if base is not None:
                                future.valid_until = min(
                                    future.valid_until,
                                    base + self.session_lease_ttl_s / 2,
                                )
                        # A slow RPC cannot publish a fresh proof of an old lease.
                        result = now < future.valid_until
                    cancelled = kind == "prepare" and rid in self.cancelled_session_jobs
                    if not cancelled and kind != "end":
                        future.set_result(result)
                        return
                if kind == "end":
                    future.set_result(result)
                    return
                # Cancellation owns the prepared lease, even if start just won
                # the race. Complete only after this worker has released it.
                self._abort_prepared_load_now(rid)
                future.set_result(False)
            except BaseException:
                logger.exception("Mooncake asynchronous session %s failed", kind)
                future.set_result(False)

        try:
            self.session_executor.submit(run)
        except RuntimeError:
            future.set_result(False)
        return future

    def _session_job_status(self, future, rid: str) -> str:
        with self.session_lock:
            expired = (future is None or not future.done()) and (
                time.monotonic() >= self.session_admission_deadlines[rid]
            )
            report_timeout = expired and rid not in self.session_admission_timeouts
            if expired:
                self.session_admission_timeouts.add(rid)
        if expired:
            if report_timeout:
                self._log_host_prefetch_metric("event", "session_admission_timeout")
                logger.warning(
                    "Mooncake session admission timed out for rid=%s; retaining its active owner",
                    rid,
                )
            return "failed"
        if future is None:
            with self.session_lock:
                cleanup_pressure = bool(self.failed_session_cleanups)
            if cleanup_pressure:
                self._log_host_prefetch_metric("event", "session_cleanup_backpressure")
                return "failed"
            return "pending"
        if not future.done():
            return "pending"
        if not future.result():
            return "failed"
        return "ready" if time.monotonic() < future.valid_until else "pending"

    def prepare_load_async(self, rid: str, transfers: list[PoolTransfer]) -> str:
        """Scheduler entry: poll preparation without issuing any native RPC."""
        if not self.host_prefetch_enabled:
            return "failed"
        return self._session_job_status(
            self._session_job(
                "prepare",
                rid,
                lambda: self.prepare_load(rid, transfers),
                renew=lambda: self.prepare_load(rid, transfers, force_refresh=True),
            ),
            rid,
        )

    def revalidate_host_prefetch_async(self, rid: str) -> str:
        """Poll one admission refresh; consume it when claiming/cancelling."""
        return self._session_job_status(
            self._session_job(
                "refresh",
                rid,
                lambda: self.revalidate_host_prefetch(rid),
                renew=lambda: self.revalidate_host_prefetch(rid, force_refresh=True),
            ),
            rid,
        )

    def revalidate_no_prefetch_needed_async(self, rid: str) -> str:
        return self._session_job_status(
            self._session_job(
                "refresh", rid, lambda: self.revalidate_no_prefetch_needed(rid)
            ),
            rid,
        )

    def revalidate_load_async(self, rid: str, transfers: list[PoolTransfer]) -> str:
        """Only local I/O runs here; the wrapper must agree on status across ranks."""
        return self._session_job_status(
            self._session_job(
                "refresh", rid, lambda: self._revalidate_load_local(transfers)
            ),
            rid,
        )

    def discard_load_admission(self, rid: str) -> None:
        # Budget rejection retires proofs, retaining prepared sessions and buffers.
        with self.session_lock:
            for kind in ("prepare", "refresh"):
                job = self.session_jobs.get((kind, rid))
                if job is not None and job.done():
                    del self.session_jobs[(kind, rid)]
            self.session_admission_deadlines.pop(rid, None)
            self.session_admission_timeouts.discard(rid)

    def abort_prepared_load_async(self, rid: str) -> bool:
        """Return False until all preparation/refresh/end work has retired."""
        with self.session_lock:
            self.cancelled_session_jobs.add(rid)
            preparing = self.preparing_load_sessions.get(rid)
            if preparing is not None:
                preparing["cancelled"] = True
            if any(
                not future.done()
                for (_, owner), future in self.session_jobs.items()
                if owner == rid
            ):
                return False
            if time.monotonic() < self.session_cleanup_retry_at.get(rid, 0):
                return False
            if preparing is not None:
                return False
            if rid in self.ending_load_sessions:
                return False
            if rid not in self.prepared_load_sessions:
                for kind in ("prepare", "refresh", "end"):
                    self.session_jobs.pop((kind, rid), None)
                self.cancelled_session_jobs.discard(rid)
                self.session_admission_deadlines.pop(rid, None)
                self.session_admission_timeouts.discard(rid)
                if self.host_prefetch_enabled:
                    with self.host_prefetch_lock:
                        entry = self.host_prefetch_entries.get(rid)
                        if (
                            entry is not None
                            and entry.get("state") == "cleanup_pending"
                            and entry.get("cleanup_rid") == rid
                        ):
                            del self.host_prefetch_entries[rid]
                return True
            end = self.session_jobs.get(("end", rid))
            if end is not None and end.done() and not end.result():
                # Guard contention is retryable; unknown native outcomes were
                # already quarantined in failed_session_cleanups above.
                del self.session_jobs[("end", rid)]
        future = self._session_job("end", rid, lambda: self.abort_prepared_load(rid))
        if future is None or not future.done() or not future.result():
            return False
        with self.session_lock:
            for kind in ("prepare", "refresh", "end"):
                self.session_jobs.pop((kind, rid), None)
            self.cancelled_session_jobs.discard(rid)
        return True

    def waiting_queue_prefetch_limit(self) -> int:
        return self.host_prefetch_limit

    @contextmanager
    def _session_key_guard(self, keys):
        # Reserve lock identities before waiting; end/start must never use two
        # different locks for a key, including when the last reference is gone.
        with self.session_lock:
            locks = []
            for key in sorted(set(keys)):
                lock, users = self.session_key_locks.get(key, (threading.Lock(), 0))
                self.session_key_locks[key] = (lock, users + 1)
                locks.append((key, lock))
        acquired = []
        try:
            for _, lock in locks:
                lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
            with self.session_lock:
                for key, lock in locks:
                    _, users = self.session_key_locks[key]
                    if users == 1:
                        del self.session_key_locks[key]
                    else:
                        self.session_key_locks[key] = (lock, users - 1)

    def _prepare_expanded_load(
        self, rid: str, expanded: list[PoolTransfer], *, force_refresh=False
    ) -> bool:
        keys = []
        seen = set()
        for transfer in expanded:
            component_keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            for key in self.storage._tag_keys(component_keys):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        if not keys:
            return False
        token = {"cancelled": False}
        with self.session_lock:
            if (
                rid in self.preparing_load_sessions
                or rid in self.ending_load_sessions
                or rid in self.failed_session_cleanups
            ):
                return False
            self.preparing_load_sessions[rid] = token
        try:
            with self._session_key_guard(keys):
                with self.session_lock:
                    if token["cancelled"] or self.failed_session_keys.intersection(
                        keys
                    ):
                        return False
                    existing = self.prepared_load_sessions.get(rid)
                    if existing is not None and existing != keys:
                        return False
                    acquired = []
                    reused_keys = []
                    new_keys = []
                    if existing is not None:
                        reused_keys = keys
                    else:
                        for key in keys:
                            if key in self.session_refcounts:
                                self.session_refcounts[key] += 1
                                self.session_sources.setdefault(key, "unknown")
                                self._lease_bases().pop(key, None)
                                reused_keys.append(key)
                                acquired.append(key)
                            else:
                                new_keys.append(key)
                if existing is None:
                    lease_base = time.monotonic()
                    try:
                        if new_keys:
                            results, sources = (
                                self.storage.store.batch_get_session_start_with_sources(
                                    new_keys
                                )
                            )
                            results, sources = list(results), list(sources)
                        else:
                            results, sources = [], []
                        ok = len(results) == len(new_keys) and all(
                            code == 0 for code in results
                        )
                    except BaseException:
                        logger.warning(
                            "Mooncake get session start raised for rid=%s",
                            rid,
                            exc_info=True,
                        )
                        # The native call has returned; rollback under the same
                        # key guards retires any partially constructed session.
                        results, sources, ok = [], [], False
                    with self.session_lock:
                        # Even a short/failed result may have created a session.
                        # These keys remain exclusively reserved until rollback ends.
                        for index, key in enumerate(new_keys):
                            self.session_refcounts[key] = 1
                            self.session_sources[key] = (
                                str(sources[index]).lower()
                                if index < len(sources)
                                else "unknown"
                            )
                            self._lease_bases()[key] = lease_base
                            acquired.append(key)
                        self.prepared_load_sessions[rid] = acquired
                    if not ok:
                        return self._discard_preparing_session(rid, acquired)
                if reused_keys and not self._refresh_session_keys_locked(
                    reused_keys, force_refresh=force_refresh
                ):
                    return self._discard_preparing_session(
                        rid, self.prepared_load_sessions[rid]
                    )
                with self.session_lock:
                    cancelled = token["cancelled"]
                    owned_keys = self.prepared_load_sessions[rid]
                    if not cancelled:
                        del self.preparing_load_sessions[rid]
                        return True
                return self._discard_preparing_session(rid, owned_keys)
        finally:
            with self.session_lock:
                if self.preparing_load_sessions.get(rid) is token:
                    del self.preparing_load_sessions[rid]

    def _discard_preparing_session(self, rid, keys) -> bool:
        with self.session_lock:
            self.prepared_load_sessions.pop(rid, None)
        if not self._release_session_refs(keys):
            self._retain_failed_session_cleanup(rid, keys)
        return False

    def _retain_failed_session_cleanup(self, rid, keys) -> None:
        with self.session_lock:
            first_failure = rid not in self.failed_session_cleanups
            self.prepared_load_sessions[rid] = [
                key for key in keys if key in self.failed_session_keys
            ]
            self.failed_session_cleanups.add(rid)
            self.cancelled_load_session_cleanup.add(rid)
            delay = min(self.session_cleanup_retry_delay.get(rid, 0.5) * 2, 30.0)
            self.session_cleanup_retry_delay[rid] = delay
            self.session_cleanup_retry_at[rid] = time.monotonic() + delay
            self.session_cleanup_stats["failures"] += 1
        self._log_host_prefetch_metric("event", "session_cleanup_failed")
        if first_failure:
            self._log_host_prefetch_metric("event", "session_cleanup_quarantined")

    def get_session_cleanup_state(self) -> dict[str, int]:
        """A snapshot for diagnostics; quarantined credits survive cache reset."""
        with self.session_lock:
            return {
                **self.session_cleanup_stats,
                "quarantined_requests": len(self.failed_session_cleanups),
                "quarantined_keys": len(self.failed_session_keys),
                "pending_cleanups": len(self.cancelled_load_session_cleanup),
                "active_jobs": sum(
                    not job.done() for job in self.session_jobs.values()
                ),
            }

    def _refresh_session_keys(self, keys: list[str], *, force_refresh=False) -> bool:
        """Serialize refresh with same-key start/end, never unrelated RPCs."""
        with self._session_key_guard(keys):
            return self._refresh_session_keys_locked(keys, force_refresh=force_refresh)

    def _refresh_session_keys_locked(
        self, keys: list[str], *, force_refresh=False
    ) -> bool:
        with self.session_lock:
            if any(
                key not in self.session_refcounts or key in self.failed_session_keys
                for key in keys
            ):
                return False
            bases = [self._lease_bases().get(key) for key in keys]
            private = all(self.session_refcounts.get(key, 0) == 1 for key in keys)
        if not keys:
            return False
        oldest = None if None in bases else min(bases)
        # Include time spent waiting for this key's preceding operation.
        age = float("inf") if oldest is None else time.monotonic() - oldest
        if (
            not force_refresh
            and self._dfs_read_policy_allows_read()
            and self.session_refresh_age_s > 0
            and private
            and age < self.session_refresh_age_s
        ):
            return True
        lease_base = time.monotonic()
        try:
            results = list(self.storage.store.batch_get_session_refresh(keys))
        except BaseException:
            logger.warning("Mooncake session refresh failed", exc_info=True)
            results = []
        ok = len(results) == len(keys) and all(result == 0 for result in results)
        with self.session_lock:
            bases = self._lease_bases()
            for key in keys:
                if not ok:
                    bases.pop(key, None)
                elif key in bases:
                    # Shared or ended keys must not regain a private lease base.
                    bases[key] = lease_base
        return ok

    def submit_host_prefetch(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        if not self.host_prefetch_enabled:
            return False
        kv_transfer = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV), None
        )
        if kv_transfer is None or not kv_transfer.keys:
            return False
        with self.host_prefetch_lock:
            active = sum(
                entry.get("state")
                in {
                    "queued",
                    "preparing",
                    "reading",
                    "dfs_prefetched",
                    "claimed",
                    "loading",
                    "cleanup_pending",
                }
                for entry in self.host_prefetch_entries.values()
            )
            if rid in self.host_prefetch_entries or active >= self.host_prefetch_limit:
                return False
            self.host_prefetch_generation += 1
            generation = self.host_prefetch_generation
            queued_transfers = list(transfers)
            self.host_prefetch_entries[rid] = {
                "state": "queued",
                "queued_at": time.monotonic(),
                "cancelled": False,
                "transfers": queued_transfers,
                "session_rid": self._host_prefetch_session_rid(rid, generation),
            }

        # Queued cancellations retain their entry/credit until the worker
        # consumes the tombstone, bounding retained transfer metadata.
        # Session creation can contact DFS. The scheduler only publishes the
        # work item; this worker owns both preparation and the DFS read.
        self.host_prefetch_queue.put((rid, queued_transfers))
        self._log_host_prefetch_metric("event", "submitted")
        return True

    def get_host_prefetch_status(self, rid: str) -> str | None:
        if not self.host_prefetch_enabled:
            return None
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None:
                return None
            state = str(entry.get("state"))
            # This is a scheduling budget, never evidence that native writes
            # stopped. Cancellation retains the entry/credit until its worker
            # returns and ends the session; recomputation uses no native slots.
            timed_out = entry.get("admission_timed_out", False)
            if (
                state in {"queued", "preparing", "reading"}
                and entry.get("queued_at") is not None
                and time.monotonic() - entry["queued_at"] >= self.session_lease_ttl_s
            ):
                first_timeout = not timed_out
                entry["admission_timed_out"] = timed_out = True
            else:
                first_timeout = False
        if first_timeout:
            self._log_host_prefetch_metric("event", "prefetch_admission_timeout")
            logger.warning(
                "Mooncake prefetch admission timed out for rid=%s; retaining its active owner",
                rid,
            )
        return "admission_timeout" if timed_out else state

    def revalidate_host_prefetch(self, rid: str, *, force_refresh=False) -> bool:
        if not self.host_prefetch_enabled:
            return False
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if (
                entry is None
                or entry.get("state") != "dfs_prefetched"
                or entry.get("cancelled")
            ):
                return False
            session_rid = str(entry["session_rid"])
        with self.session_lock:
            keys = list(self.prepared_load_sessions.get(session_rid, ()))
        return self._refresh_session_keys(keys, force_refresh=force_refresh)

    def revalidate_no_prefetch_needed(self, rid: str) -> bool:
        """Re-check the keys of a request that needed no DFS prefetch.

        The worker released its session on finding no DFS keys, so nothing
        pins the objects while the request waits for admission. Returning
        False sends every rank back to the normal path, whose revalidation
        degrades an evicted prefix to a cache miss.
        """
        if not self.host_prefetch_enabled:
            return False
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if (
                entry is None
                or entry.get("state") != "no_prefetch_needed"
                or entry.get("cancelled")
            ):
                return False
            keys = list(entry.get("no_prefetch_keys", ()))
        if not keys:
            return False
        try:
            exist = list(self.storage._batch_exist(keys))
        except BaseException:
            logger.warning(
                "Mooncake waiting-queue existence check failed for rid=%s",
                rid,
                exc_info=True,
            )
            return False
        missing = len(keys) - sum(1 for state in exist if state == 1)
        if len(exist) != len(keys) or missing:
            logger.warning(
                "Mooncake waiting-queue keys changed while queued for rid=%s: "
                "missing=%d/%d, falling back to the normal load path",
                rid,
                missing,
                len(keys),
            )
            return False
        return True

    def claim_ready_host_prefetch(self, rid: str) -> bool:
        if not self.host_prefetch_enabled:
            return False
        # The bookkeeping lock is never held across a session RPC.
        # Keep session ownership and the claimed state atomic under both locks.
        with self.session_lock, self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None or entry.get("state") != "dfs_prefetched":
                return False
            session_rid = str(entry["session_rid"])
            if (
                rid in self.prepared_load_sessions
                or session_rid not in self.prepared_load_sessions
            ):
                return False
            self.prepared_load_sessions[rid] = self.prepared_load_sessions.pop(
                session_rid
            )
            self.session_jobs.pop(("refresh", rid), None)
            entry["state"] = "claimed"
            ready_at = entry.get("ready_at")
        if ready_at is not None:
            self._log_host_prefetch_metric(
                "claim", "claimed", time.monotonic() - ready_at
            )
        return True

    def cancel_host_prefetch(self, rid: str) -> None:
        if not self.host_prefetch_enabled:
            return
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None:
                return
            state = str(entry.get("state"))
            session_rid = str(entry["session_rid"])
            if state in {"queued", "preparing", "reading", "loading"}:
                entry["cancelled"] = True
                return
            cleanup_rid = entry.get(
                "cleanup_rid", rid if state == "claimed" else session_rid
            )
            entry["cleanup_rid"] = cleanup_rid
            entry["state"] = "cleanup_pending"
            entry["cancelled"] = True
        if state == "dfs_prefetched":
            self._log_host_prefetch_metric("event", "discarded_ready")
        # Register before submission: a full end queue must not lose ownership.
        with self.host_prefetch_lock:
            self.host_prefetch_session_cleanup[rid] = (entry, cleanup_rid)
        self._retry_host_prefetch_session_cleanup(rid, entry, cleanup_rid)

    def _retry_host_prefetch_session_cleanup(self, rid, entry, cleanup_rid) -> None:
        if not self.abort_prepared_load_async(cleanup_rid):
            return
        with self.host_prefetch_lock:
            pending = self.host_prefetch_session_cleanup.get(rid)
            if pending is not None and pending[0] is entry:
                del self.host_prefetch_session_cleanup[rid]
            if self.host_prefetch_entries.get(rid) is entry:
                del self.host_prefetch_entries[rid]
        with self.session_lock:
            refresh = self.session_jobs.get(("refresh", rid))
            if refresh is not None and refresh.done():
                del self.session_jobs[("refresh", rid)]

    @staticmethod
    def _host_prefetch_session_rid(rid: str, generation: int) -> str:
        return f"__sglang_waiting_queue_prefetch__:{rid}:{generation}"

    def host_prefetch_thread_func(self) -> None:
        while True:
            task = self.host_prefetch_queue.get()
            try:
                if task is None:
                    return
                rid, transfers = task
                with self.host_prefetch_lock:
                    entry = self.host_prefetch_entries.get(rid)
                    if entry is None or entry.get("transfers") is not transfers:
                        continue
                    session_rid = str(entry["session_rid"])
                    if entry.get("cancelled"):
                        self.host_prefetch_entries.pop(rid, None)
                        self._log_host_prefetch_metric(
                            "result", "cancelled", time.monotonic() - entry["queued_at"]
                        )
                        continue
                    entry["state"] = "preparing"

                try:
                    prepared = self.prepare_load(session_rid, transfers)
                except BaseException:
                    logger.warning(
                        "Mooncake waiting-queue session preparation failed for rid=%s",
                        rid,
                        exc_info=True,
                    )
                    prepared = False
                with self.session_lock:
                    keys = list(self.prepared_load_sessions.get(session_rid, ()))
                if not prepared or not keys:
                    self._finish_host_prefetch(rid, entry, session_rid, "failed")
                    continue

                with self.session_lock:
                    sources = {
                        key: self.session_sources.get(key, "unknown") for key in keys
                    }
                dfs_keys = [key for key in keys if sources[key] in {"dfs", "unknown"}]
                if not dfs_keys:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "no_prefetch_needed", keys=keys
                    )
                    continue

                with self.host_prefetch_lock:
                    current = self.host_prefetch_entries.get(rid)
                    if current is not entry or current.get("cancelled"):
                        cancelled = True
                    else:
                        cancelled = False
                        current["state"] = "reading"
                if cancelled:
                    self._finish_host_prefetch(rid, entry, session_rid, "failed")
                    continue
                try:
                    results = list(
                        self.storage.store.batch_get_session_prefetch(dfs_keys)
                    )
                    success = len(results) == len(dfs_keys) and all(
                        result == 0 for result in results
                    )
                except BaseException:
                    logger.warning(
                        "Mooncake waiting-queue DFS prefetch failed for rid=%s",
                        rid,
                        exc_info=True,
                    )
                    success = False
                self._finish_host_prefetch(
                    rid,
                    entry,
                    session_rid,
                    "dfs_prefetched" if success else "failed",
                )
            finally:
                self.host_prefetch_queue.task_done()

    def _finish_host_prefetch(
        self,
        rid: str,
        entry: dict[str, object],
        session_rid: str,
        outcome: str,
        *,
        keys: list[str] | None = None,
    ) -> None:
        keep_session = False
        with self.host_prefetch_lock:
            current = self.host_prefetch_entries.get(rid)
            cancelled = bool(
                current is not entry or current is None or current.get("cancelled")
            )
            if (
                current is entry
                and outcome in {"dfs_prefetched", "no_prefetch_needed"}
                and not cancelled
            ):
                if outcome == "no_prefetch_needed":
                    # The session is released below; keep the keys so
                    # admission can re-check them (see
                    # revalidate_no_prefetch_needed). Published together with
                    # the state under the same lock.
                    current["no_prefetch_keys"] = list(keys or ())
                current["state"] = outcome
                if outcome == "dfs_prefetched":
                    current["ready_at"] = time.monotonic()
                keep_session = outcome == "dfs_prefetched"
            if current is entry and not keep_session:
                # Reserve the host credit while native end is still running.
                current["state"] = "cleanup_pending"
                current["cleanup_rid"] = session_rid
        if current is entry:
            self._log_host_prefetch_metric(
                "result",
                "cancelled" if cancelled else outcome,
                time.monotonic() - entry["queued_at"],
            )
        if not keep_session:
            released = self._abort_prepared_load_now(session_rid)
            with self.host_prefetch_lock:
                if current is entry and self.host_prefetch_entries.get(rid) is entry:
                    if not released:
                        entry["state"] = "cleanup_pending"
                        entry["cleanup_rid"] = session_rid
                        self.host_prefetch_session_cleanup[rid] = (entry, session_rid)
                    elif entry.get("cancelled"):
                        del self.host_prefetch_entries[rid]
                    else:
                        entry["state"] = (
                            "no_prefetch_needed"
                            if outcome == "no_prefetch_needed"
                            else "failed"
                        )
                        entry.pop("cleanup_rid", None)

    def _log_host_prefetch_metric(
        self, kind: str, event: str, duration: float = 0.0
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None:
            return
        try:
            if kind == "result":
                collector.log_waiting_queue_prefetch_result(event, duration)
            elif kind == "claim":
                collector.log_waiting_queue_prefetch_claim(duration)
            else:
                collector.log_waiting_queue_prefetch_event(event)
        except Exception:
            logger.warning(
                "Failed to record waiting-queue prefetch metrics.", exc_info=True
            )

    def _release_session_refs(self, keys: list[str]) -> bool:
        # Caller owns all key guards through the native end, so a new start
        # cannot be erased by an older request's delayed end.
        to_end = []
        with self.session_lock:
            for key in keys:
                count = self.session_refcounts.get(key, 0)
                if count <= 1:
                    to_end.append(key)
                else:
                    self.session_refcounts[key] = count - 1
        if to_end:
            try:
                rc = self.storage.store.batch_get_session_end(to_end)
                if rc != 0:
                    raise RuntimeError(f"Mooncake get session cleanup returned {rc}")
            except BaseException as error:
                with self.session_lock:
                    self.failed_session_keys.update(to_end)
                    self.session_cleanup_log_failures += 1
                    now = time.monotonic()
                    report = now >= self.session_cleanup_log_at
                    if report:
                        failures = self.session_cleanup_log_failures
                        self.session_cleanup_log_failures = 0
                        self.session_cleanup_log_at = now + 30.0
                if report:
                    logger.warning(
                        "Mooncake get session cleanup failed: keys=%d, failures_since_log=%d, error=%s",
                        len(to_end),
                        failures,
                        error,
                    )
                return False
            with self.session_lock:
                for key in to_end:
                    self.session_refcounts.pop(key, None)
                    self.session_sources.pop(key, None)
                    self._lease_bases().pop(key, None)
                    self.failed_session_keys.discard(key)
        return True

    def _abort_prepared_load_now(self, rid: str) -> bool:
        with self.session_lock:
            preparing = self.preparing_load_sessions.get(rid)
            if preparing is not None:
                preparing["cancelled"] = True
                return False
            if rid in self.ending_load_sessions:
                return False
            keys = self.prepared_load_sessions.get(rid)
            retry = rid in self.failed_session_cleanups
            if retry:
                self.session_cleanup_stats["retries"] += 1
        if keys is None:
            return True
        if retry:
            self._log_host_prefetch_metric("event", "session_cleanup_retry")
        with self._session_key_guard(keys):
            with self.session_lock:
                preparing = self.preparing_load_sessions.get(rid)
                if preparing is not None:
                    preparing["cancelled"] = True
                    return False
                if self.prepared_load_sessions.get(rid) is not keys:
                    return rid not in self.failed_session_cleanups
                del self.prepared_load_sessions[rid]
                self.ending_load_sessions.add(rid)
            ok = self._release_session_refs(keys)
            if not ok:
                self._retain_failed_session_cleanup(rid, keys)
            with self.session_lock:
                if ok:
                    self.failed_session_cleanups.discard(rid)
                    self.session_cleanup_retry_at.pop(rid, None)
                    self.session_cleanup_retry_delay.pop(rid, None)
                    if retry:
                        self.session_cleanup_stats["recovered"] += 1
                self.ending_load_sessions.discard(rid)
            if ok and retry:
                self._log_host_prefetch_metric("event", "session_cleanup_recovered")
            return ok

    def abort_prepared_load(self, rid: str) -> bool:
        if not self.host_prefetch_enabled:
            return True
        ok = self._abort_prepared_load_now(rid)
        if not ok:
            return False
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is not None and entry.get("state") in {
                "claimed",
                "loading",
                "cleanup_pending",
            }:
                self.host_prefetch_entries.pop(rid, None)
        with self.session_lock:
            self.session_admission_deadlines.pop(rid, None)
            self.session_admission_timeouts.discard(rid)
            for kind in ("prepare", "refresh"):
                job = self.session_jobs.get((kind, rid))
                if job is not None and job.done():
                    del self.session_jobs[(kind, rid)]
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        cancelled = self.pending_loads.pop(rid, None) is not None
        if not cancelled:
            # A dispatched load still needs its token/time record at completion.
            return False
        getattr(self, "pending_load_tokens", {}).pop(rid, None)
        getattr(self, "pending_load_started", {}).pop(rid, None)
        if self.host_prefetch_enabled:
            # Withdrawn work cannot write device slots. Keep its session/host
            # credits until the background end is confirmed, including on errors.
            with self.host_prefetch_lock:
                entry = self.host_prefetch_entries.get(rid)
                if entry is not None:
                    entry["state"] = "cleanup_pending"
                    entry["cancelled"] = True
                    entry["cleanup_rid"] = rid
            with self.session_lock:
                self.cancelled_load_session_cleanup.add(rid)
            if self.abort_prepared_load_async(rid):
                with self.session_lock:
                    self.cancelled_load_session_cleanup.discard(rid)
        return True

    def _poll_cancelled_load_sessions(self) -> None:
        with self.host_prefetch_lock:
            host_pending = tuple(self.host_prefetch_session_cleanup.items())
        for rid, (entry, cleanup_rid) in host_pending:
            self._retry_host_prefetch_session_cleanup(rid, entry, cleanup_rid)
        with self.session_lock:
            pending = tuple(self.cancelled_load_session_cleanup)
        for rid in pending:
            if self.abort_prepared_load_async(rid):
                with self.session_lock:
                    self.cancelled_load_session_cleanup.discard(rid)

    def num_completed_loads(self) -> int:
        self._poll_cancelled_load_sessions()
        while (
            getattr(self, "broadcast_events", None)
            and self.broadcast_events[0][0].query()
        ):
            rids = self.broadcast_events[0][1]
            cleanup = self.follower_cleanup_outcomes.get(tuple(rids))
            if cleanup is not None and not cleanup.done():
                break
            _, rids, success = self.broadcast_events.pop(0)
            if cleanup is not None:
                success = success and cleanup.result()
                del self.follower_cleanup_outcomes[tuple(rids)]
            self.completed_loads.put((rids, success))
        return self.completed_loads.qsize()

    def wait_for_pending_loads(self) -> None:
        self.load_queue.join()
        for event, _, _ in getattr(self, "broadcast_events", ()):
            event.synchronize()
        self.load_batch_futures.clear()
        self.load_batch_cleanup_futures.clear()

    def wait_for_load_batch(self, index: int) -> None:
        if index < 0:
            return
        completed = self.load_batch_futures.get(index)
        if completed is None:
            return
        event = completed.result()
        if event is not None:
            event.synchronize()
        cleanup = self.load_batch_cleanup_futures.get(index)
        if cleanup is not None:
            # Result processing must see this batch's cleanup verdict before
            # emitting tokens. Forward only publishes the GPU event.
            cleanup.result()
            self.load_batch_cleanup_futures.pop(index, None)
        self.load_batch_futures.pop(index, None)

    def pop_completed_load(self) -> tuple[list[str], bool]:
        return self.completed_loads.get_nowait()

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        broadcaster = getattr(self, "mla_broadcaster", None)
        if broadcaster is not None:
            # Insert can adopt different pages on different ranks. Compare the
            # logical load order (not local slots), including empty batches,
            # before deciding collectively whether this batch can broadcast.
            plan = [
                (rid, [(t.name, list(t.keys)) for t in transfers])
                for rid, transfers in self.pending_loads.items()
            ]
            digest = hashlib.sha256(repr(plan).encode()).digest()
            digests = [None] * torch.distributed.get_world_size(self.tp_group)
            torch.distributed.all_gather_object(digests, digest, group=self.tp_group)
            if any(other != digest for other in digests):
                logger.warning("MLA linker load plans differ; using all-rank reads.")
                broadcaster = None
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        self.load_batch_futures[counter_index] = Future()
        if broadcaster is not None:
            indices = {}
            for transfers in pending.values():
                for transfer in transfers:
                    indices.setdefault(transfer.name, []).append(transfer.host_indices)
            prepared = broadcaster.prepare_broadcast(
                {name: torch.cat(parts) for name, parts in indices.items()},
                device_module.current_stream(),
            )
            if not broadcaster.is_src:
                if isinstance(self.layer_done_counter, ReadPlanLoadCounter):
                    self.layer_done_counter.bind(counter_index, _CompletedReadPlan())
                else:
                    for layer in range(self.num_layers):
                        self.layer_done_counter.complete(counter_index, layer)
        ready_event = device_module.Event()
        ready_event.record()
        if broadcaster is not None:
            if broadcaster.is_src:
                self.broadcast_outcomes[counter_index] = Future()
            # Both source and follower publish read readiness before cleanup.
            cleanup = Future()
            self.follower_cleanup_outcomes[tuple(pending)] = cleanup
            self.load_batch_cleanup_futures[counter_index] = cleanup
            self.broadcast_loads[counter_index] = (
                0,
                prepared,
                list(pending),
                ready_event,
            )
        self.load_queue.put((counter_index, pending, ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                success = False
                cleanup = self.load_batch_cleanup_futures.get(counter_index)
                replicated = cleanup is not None
                try:
                    ready_event.synchronize()
                    if replicated and not self.mla_broadcaster.is_src:
                        success = True
                        if self.host_prefetch_enabled:
                            for rid in pending:
                                if not self.abort_prepared_load(rid):
                                    success = False
                    elif self.host_prefetch_enabled:
                        success = self.load_layer_wise(
                            counter_index,
                            list(pending.values()),
                            request_ids=list(pending),
                        )
                    else:
                        success = self.load_layer_wise(
                            counter_index, list(pending.values())
                        )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    self._finish_prefetch_metrics(list(pending), success is True)
                    outcome = self.broadcast_outcomes.get(counter_index)
                    if outcome is not None and not outcome.done():
                        outcome.set_result(False)
                    if cleanup is not None:
                        cleanup.set_result(success)
                    if not replicated:
                        self.completed_loads.put((list(pending), success))
                        self.load_batch_futures[counter_index].set_result(None)
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[list[PoolTransfer]],
        *,
        request_ids: list[str] | None = None,
    ) -> bool:
        started = []
        success = False
        read_plan_enabled = self.read_plan_enabled
        maybe_fail = arm_load_failure_injection(self.tp_rank)
        try:
            if self.host_prefetch_enabled:
                if request_ids is None:
                    raise ValueError("Prefetched loads require request IDs")
                for rid in request_ids:
                    with self.session_lock:
                        prepared = rid in self.prepared_load_sessions
                    if not prepared:
                        raise RuntimeError(
                            f"Load admitted without prepared session: rid={rid}"
                        )
            if read_plan_enabled:
                # Fail before binding: an unrun bound plan cannot wake waiters.
                maybe_fail("read_plan", "before_run")
                if (
                    not self.host_prefetch_enabled
                    and counter_index in self.broadcast_outcomes
                ):
                    # Borrow Python-owned sessions so native final readiness
                    # excludes end; the result barrier still waits for cleanup.
                    self.load_with_read_plan(
                        counter_index, request_transfers, started=started
                    )
                else:
                    self.load_with_read_plan(counter_index, request_transfers)
                success = True
            else:
                batches: dict[PoolName, tuple[list[str], list[int]]] = {}
                for transfers in request_transfers:
                    for transfer in transfers:
                        keys, locations = batches.setdefault(transfer.name, ([], []))
                        component_keys, _ = (
                            self.storage._get_hybrid_page_component_keys(
                                list(transfer.keys), transfer
                            )
                        )
                        keys.extend(self.storage._tag_keys(component_keys))
                        locations.extend(
                            self.pools[transfer.name].prepare_locations(
                                transfer.host_indices
                            )
                        )
                if not self.host_prefetch_enabled:
                    # Match the original path: one session start per pool batch,
                    # with no request-indexed session maps or duplicate key build.
                    for keys, _ in batches.values():
                        self._start_load_sessions(keys, started)
                if self.enable_page_wise_load and any(
                    len(keys) >= self.page_wise_load_threshold
                    for keys, _ in batches.values()
                ):
                    self._load_page_wise(
                        counter_index, batches, maybe_fail, started=started
                    )
                elif self.read_layout_enabled:
                    self._load_with_read_layout(counter_index, batches, maybe_fail)
                else:
                    for layer in range(self.num_layers):
                        for name, (keys, locations) in batches.items():
                            meta = self.pools[name].get_prepared_layer_range_meta(
                                locations, layer
                            )
                            if meta is None:
                                continue
                            ptrs, sizes, offsets = meta
                            maybe_fail(name, f"layer={layer}")
                            result = (
                                self.storage.store.batch_get_into_multi_buffer_ranges(
                                    keys,
                                    ptrs,
                                    sizes,
                                    offsets,
                                )
                            )
                            expected = [sum(item) for item in sizes]
                            if (
                                result is None
                                or isinstance(result, int)
                                or list(result) != expected
                            ):
                                raise RuntimeError(
                                    f"Mooncake range get failed for pool={name}, "
                                    f"layer={layer}: transferred={result}, "
                                    f"expected={expected}"
                                )
                        self.layer_done_counter.complete(counter_index, layer)
                success = True
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            outcome = self.broadcast_outcomes.get(counter_index)
            if outcome is not None:
                # Forward waits for all reads, never for session/buffer teardown.
                outcome.set_result(success)
            if self.host_prefetch_enabled:
                for rid in request_ids or ():
                    if self.abort_prepared_load(rid) is False:
                        success = False
            else:
                for keys in started:
                    try:
                        rc = self.storage.store.batch_get_session_end(keys)
                        if rc != 0:
                            raise RuntimeError(
                                f"Mooncake get session cleanup returned {rc}"
                            )
                    except BaseException as error:
                        self.layer_done_counter.fail(counter_index, error)
                        logger.exception(
                            "Mooncake layer-wise load session cleanup failed"
                        )
                        success = False
        return success

    def _start_load_sessions(self, keys, started) -> None:
        # Even an exception may follow a remotely applied start.
        started.append(keys)
        result = list(self.storage.store.batch_get_session_start(keys))
        if len(result) != len(keys):
            raise RuntimeError("Mooncake get session start result count mismatch")
        failed = [key for key, code in zip(keys, result) if code != 0]
        if failed:
            logger.warning(
                "Mooncake get session start partial failure (keys=%d, failed=%d), retrying failed keys once",
                len(keys),
                len(failed),
            )
            retry = list(self.storage.store.batch_get_session_start(failed))
            if len(retry) != len(failed):
                raise RuntimeError("Mooncake get session retry result count mismatch")
            failed = [key for key, code in zip(failed, retry) if code != 0]
        if failed:
            failed_get_cache = getattr(self.storage, "failed_get_cache", None)
            if failed_get_cache is not None:
                failed_get_cache.update_batch([], failed)
            raise RuntimeError(
                f"Mooncake get session start failed: keys={len(keys)}, "
                f"failed={len(failed)}"
            )

    def _load_with_read_layout(self, counter_index: int, batches, maybe_fail) -> None:
        ReadComponent, ReadLayout, ReadKeyMapping = resolve_read_layout_api(
            self.storage.store
        )

        for layer in range(self.num_layers):
            layouts = []
            expected = []
            for name, (keys, locations) in batches.items():
                pool = self.pools[name]
                items = pool.get_layer_range_components(layer)
                if not items:
                    continue
                maybe_fail(name, f"layer={layer}")
                layouts.append(
                    ReadLayout(
                        keys=keys,
                        rows=locations,
                        key_mapping=(
                            ReadKeyMapping.PER_ROW
                            if pool.packed
                            else ReadKeyMapping.PER_ROW_COMPONENT
                        ),
                        components=[
                            ReadComponent(
                                destination_base=base,
                                destination_stride=stride,
                                byte_count=size,
                                source_offset=offset,
                            )
                            for base, stride, size, offset in items
                        ],
                    )
                )
                expected.append(
                    [sum(item[2] for item in items)] * len(keys)
                    if pool.packed
                    else [item[2] for _ in locations for item in items]
                )
            if layouts:
                result = self.storage.store.batch_get_into_layouts(layouts)
                if result is None or list(map(list, result)) != expected:
                    raise RuntimeError(
                        f"Mooncake layout get failed for layer={layer}: "
                        f"transferred={result}, expected={expected}"
                    )
            self.layer_done_counter.complete(counter_index, layer)

    def _finish_prefetch_metrics(self, rids: list[str], success: bool) -> None:
        pending_load_tokens = getattr(self, "pending_load_tokens", {})
        pending_load_started = getattr(self, "pending_load_started", {})
        completed_at = time.perf_counter()
        tokens = 0
        for rid in rids:
            request_tokens = pending_load_tokens.pop(rid, 0)
            started = pending_load_started.pop(rid, completed_at)
            if request_tokens <= 0:
                continue
            self._log_l4_prefetch_metric(
                request_tokens, completed_at - started, success
            )
            tokens += request_tokens
        if not success or tokens <= 0:
            return
        recorder = getattr(
            getattr(self.storage, "store", None), "record_prefetched_tokens", None
        )
        if recorder is not None:
            try:
                recorder(tokens)
            except BaseException:
                logger.warning(
                    "Failed to record Mooncake prefetched token metric.",
                    exc_info=True,
                )

    def _log_l4_prefetch_metric(
        self, tokens: int, duration: float, success: bool
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None or tokens <= 0:
            return
        try:
            collector.log_l4_prefetch("mooncake", tokens, duration, success)
        except BaseException:
            logger.warning(
                "Failed to record SGLang direct-storage prefetch metrics.",
                exc_info=True,
            )

    def _load_page_wise(
        self, counter_index: int, batches, maybe_fail, *, started
    ) -> None:
        """Load all layer ranges for each page before exposing the data."""
        all_keys: list[str] = []
        all_ptrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        all_offsets: list[list[int]] = []
        for name, (keys, locations) in batches.items():
            ptrs: list[list[int]] = [[] for _ in keys]
            sizes: list[list[int]] = [[] for _ in keys]
            offsets: list[list[int]] = [[] for _ in keys]

            for layer in range(self.num_layers):
                meta = self.pools[name].get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                layer_ptrs, layer_sizes, layer_offsets = meta
                if not (
                    len(layer_ptrs)
                    == len(layer_sizes)
                    == len(layer_offsets)
                    == len(keys)
                ):
                    raise ValueError(
                        f"Mooncake pool={name} layer={layer} produced "
                        f"{len(layer_ptrs)} range entries for {len(keys)} keys."
                    )
                for index in range(len(keys)):
                    ptrs[index].extend(layer_ptrs[index])
                    sizes[index].extend(layer_sizes[index])
                    offsets[index].extend(layer_offsets[index])

            all_keys.extend(keys)
            all_ptrs.extend(ptrs)
            all_sizes.extend(sizes)
            all_offsets.extend(offsets)

        lengths = {
            "keys": len(all_keys),
            "ptrs": len(all_ptrs),
            "sizes": len(all_sizes),
            "offsets": len(all_offsets),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Mooncake page-wise aggregated metadata mismatch: {lengths}."
            )

        # Mooncake's range API is key-major and does not take a pool argument,
        # so differently suffixed physical-pool objects can share one call.
        maybe_fail("aggregated", "complete_page")
        result = self.storage.store.batch_get_into_multi_buffer_ranges(
            all_keys, all_ptrs, all_sizes, all_offsets
        )
        expected = [sum(item) for item in all_sizes]
        if result is None or isinstance(result, int) or list(result) != expected:
            pool_counts = {str(name): len(keys) for name, (keys, _) in batches.items()}
            raise RuntimeError(
                "Mooncake aggregated range get failed for "
                f"pools={pool_counts}, complete_page: transferred={result}, "
                f"expected={expected}"
            )

        # MLA publishes read readiness before teardown; other loads keep their
        # original all-or-nothing page-wise release ordering.
        if counter_index not in self.broadcast_outcomes:
            for keys in list(started):
                started.remove(keys)
                rc = self.storage.store.batch_get_session_end(keys)
                if rc != 0:
                    raise RuntimeError(f"Mooncake get session cleanup returned {rc}")
        for layer in range(self.num_layers):
            self.layer_done_counter.complete(counter_index, layer)

    def _prepare_read_plan_layouts(self, request_transfers: list[list[PoolTransfer]]):
        # Consolidate index copies once per pool, preserving request/key order.
        # Each component describes (base, row stride, byte count, source offset).
        # Locations may be non-contiguous; Mooncake expands addresses in C++.
        batches = {}
        for transfers in request_transfers:
            for transfer in transfers:
                keys, indices = batches.setdefault(transfer.name, ([], []))
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                keys.extend(self.storage._tag_keys(component_keys))
                indices.append(transfer.host_indices)

        layouts = []
        for name, (keys, indices) in batches.items():
            pool = self.pools[name]
            joined = indices[0] if len(indices) == 1 else torch.cat(indices)
            locations = pool.prepare_locations(joined)
            layout = []
            for layer in range(self.num_layers):
                mapped = pool.layer_mapping.get(layer)
                buffer_indices = (
                    ()
                    if mapped is None
                    else (mapped,)
                    if isinstance(mapped, int)
                    else tuple(mapped)
                )
                layout.append(
                    [
                        (*component[buffer_index], offsets[buffer_index])
                        for component, offsets in zip(
                            pool.buffer_meta, pool._component_offsets
                        )
                        for buffer_index in buffer_indices
                    ]
                )
            layouts.append((keys, locations, pool.packed, layout))

        return layouts

    def load_with_read_plan(
        self,
        counter_index: int,
        request_transfers: list[list[PoolTransfer]],
        *,
        started=None,
    ) -> None:
        layouts = self._prepare_read_plan_layouts(request_transfers)
        self._reselect_read_plan_sessions_if_needed(layouts)
        if started is not None:
            keys = list(dict.fromkeys(key for keys, *_ in layouts for key in keys))
            self._start_load_sessions(keys, started)
        plan = self.storage.store.create_read_plan(
            layouts,
            self.num_layers,
            reuse_ranges=self.read_plan_reuse_ranges,
            page_wise=self.enable_page_wise_load,
            borrowed_sessions=self.host_prefetch_enabled or started is not None,
            buffer_owners=self.pools,
        )
        self.layer_done_counter.bind(counter_index, plan)
        # run() and wait() release the GIL. Each layer becomes visible only after
        # every pool's bytes have been checked; the last wait includes cleanup.
        # In page-wise mode a single batch_get carries all groups per key, so the
        # first wait(0) blocks until every page is complete and later waits are
        # no-ops, matching _load_page_wise's all-or-nothing release.
        try:
            plan.run()
        except RuntimeError as error:
            match = _READ_PLAN_RC_PATTERN.search(str(error))
            # Admission backpressure (-1608) does not establish a DFS failure.
            if match and int(match.group(1)) == DFS_BATCH_GET_TIMEOUT:
                self._report_dfs_operation_timeout("read", DFS_BATCH_GET_TIMEOUT)
            raise

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put((expanded, tokens, time.perf_counter(), ready_event))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            metric_recorded = False
            try:
                if task is None:
                    return
                expanded, tokens, started, ready_event = task
                ready_event.synchronize()
                results = self.storage.batch_set_v2(expanded)
                success = all(all(pool_results) for pool_results in results.values())
                self._log_l4_backup_metric(
                    tokens, time.perf_counter() - started, success
                )
                metric_recorded = True
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                if task is not None and not metric_recorded:
                    _, tokens, started, _ = task
                    self._log_l4_backup_metric(
                        tokens, time.perf_counter() - started, False
                    )
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def _log_l4_backup_metric(
        self, tokens: int, duration: float, success: bool
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None:
            return
        try:
            # This branch intentionally does not guess DFS versus local disk.
            # Its Mooncake configuration owns that routing decision.
            collector.log_l4_backup("mooncake", tokens, duration, success)
        except BaseException:
            logger.warning(
                "Failed to record SGLang L4 backup metrics.",
                exc_info=True,
            )

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        # Join native work before resetting pools, including admission RPCs.
        with self.session_lock:
            jobs = list(self.session_jobs.values())
            for kind, rid in self.session_jobs:
                if kind == "prepare":
                    self.cancelled_session_jobs.add(rid)
        for future in jobs:
            future.result()
        if self.host_prefetch_threads:
            with self.host_prefetch_lock:
                for entry in self.host_prefetch_entries.values():
                    entry["cancelled"] = True
            self.host_prefetch_queue.join()
            for rid in list(self.pending_loads):
                self.abort_prepared_load(rid)
        self.pending_loads.clear()
        getattr(self, "pending_load_tokens", {}).clear()
        getattr(self, "pending_load_started", {}).clear()
        self.load_queue.join()
        self.offload_queue.join()
        for rid in list(self.prepared_load_sessions):
            self.abort_prepared_load(rid)
        if self.host_prefetch_threads:
            with self.host_prefetch_lock:
                retained = {}
                for rid, entry in self.host_prefetch_entries.items():
                    owner = entry.get("cleanup_rid", entry["session_rid"])
                    if rid in self.failed_session_cleanups:
                        owner = rid
                    if owner in self.failed_session_cleanups:
                        entry["state"] = "cleanup_pending"
                        entry["cleanup_rid"] = owner
                        retained[rid] = entry
                        self.host_prefetch_session_cleanup[rid] = (entry, owner)
                self.host_prefetch_entries = retained
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.completed_loads.get_nowait()
            except Empty:
                break
        for event, _, _ in self.broadcast_events:
            event.synchronize()
        self.broadcast_events.clear()
        self.broadcast_loads.clear()
        self.broadcast_outcomes.clear()
        self.follower_cleanup_outcomes.clear()
        self.load_batch_cleanup_futures.clear()
        self.broadcast_failed.clear()
        self.load_batch_futures.clear()
        self.layer_done_counter.reset()
        with self.session_lock:
            self.session_jobs = {
                key: job
                for key, job in self.session_jobs.items()
                if key[1] in self.failed_session_cleanups
            }
            self.cancelled_session_jobs.clear()
            self.session_admission_deadlines.clear()
            self.session_admission_timeouts.clear()
            self.cancelled_load_session_cleanup = set(self.failed_session_cleanups)
        with self.host_prefetch_lock:
            self.host_prefetch_session_cleanup = {
                rid: pending
                for rid, pending in self.host_prefetch_session_cleanup.items()
                if pending[1] in self.failed_session_cleanups
            }

    def close(self) -> None:
        if self._closed:
            return
        if not self._close_workers_stopped:
            self.reset()
            self.session_executor.shutdown(wait=True)
            self.load_queue.put(None)
            self.offload_queue.put(None)
            for _ in self.host_prefetch_threads:
                self.host_prefetch_queue.put(None)
            self.load_thread.join()
            self.offload_thread.join()
            for thread in self.host_prefetch_threads:
                thread.join()
            self._close_workers_stopped = True
            logger.info("Mooncake direct linker stats: %s", self.stats)
        # A failed health close retains storage/client ownership. Retrying must
        # not reset or enqueue sentinels into queues whose workers have stopped.
        self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
        self._closed = True

    def _broadcast_loaded_layers(self, index: int, threshold: int, ready=True) -> None:
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, rids, ready_event = batch
        if threshold < first:
            return
        if first == 0:
            device_module.current_stream().wait_event(ready_event)
        last = threshold == self.num_layers - 1
        if last and self.mla_broadcaster.is_src:
            # All reads must finish before publishing; cleanup has its own barrier.
            outcome = self.broadcast_outcomes[index].result()
            self.broadcast_outcomes.pop(index)
            ready = ready and outcome
        # Every layer agrees before conditionally entering its NCCL broadcasts.
        # Deferring this consensus to the last layer can strand peer ranks.
        verdict = torch.tensor(
            [int(ready and index not in self.broadcast_failed)], dtype=torch.int
        )
        torch.distributed.all_reduce(
            verdict, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
        success = bool(verdict.item())
        if success:
            for layer in range(first, threshold + 1):
                self.mla_broadcaster.broadcast_loaded_layer(layer, prepared)
        else:
            self.broadcast_failed.add(index)
        if last:
            event = device_module.Event()
            event.record()
            self.broadcast_events.append((event, rids, success))
            self.load_batch_futures[index].set_result(event)
            self.broadcast_failed.discard(index)
            del self.broadcast_loads[index]
        else:
            self.broadcast_loads[index] = (
                max(first, threshold + 1),
                prepared,
                rids,
                ready_event,
            )

    def request_id(self, handle):
        rid = str(handle.rid)
        return f"sglang:{len(rid)}:{rid}:lifecycle:{handle.lifecycle_id}:attempt:{handle.attempt_id}"
