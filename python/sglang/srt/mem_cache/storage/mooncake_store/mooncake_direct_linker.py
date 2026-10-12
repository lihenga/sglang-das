from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import Future
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
from sglang.srt.mem_cache.unified_cache.linker_fault_injection import (
    arm_load_failure_injection,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_memory, get_model
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _get_mooncake_storage_metrics_dp_rank(server_args, params) -> int:
    """返回指标 label 应使用的 DP rank。

    DP-attention 的 rank 由 attention 并行上下文维护；普通 DP 则使用构建
    cache 时显式传入的物理 replica rank。未启用 DP 时统一记为 0。
    """
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import get_attention_dp_rank

        return get_attention_dp_rank()
    return getattr(params, "dp_rank", None) or 0


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


def _session_refresh_age_config() -> float:
    """从环境变量读取允许跳过 session refresh 的最大 lease 年龄（秒）。

    client 无法查询 master 的真实 TTL，只能由部署显式声明；若配置年龄不小于
    TTL，跳过 refresh 后将没有剩余保护窗口，因此自动退化为每次都刷新。
    """
    age_s = float(os.environ.get("SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S", "0") or 0)
    lease_ttl_s = float(os.environ.get("SGLANG_MOONCAKE_LEASE_TTL_S", "10") or 10)
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


class LayerWiseLoadCounter:
    """与 KV pool 逐层等待 hook 对接的 CPU 完成计数器。

    后台线程为每个 batch、每个 layer 设置 Future；模型 forward 到达某层前
    等待对应 Future。I/O 失败只记录而不从 forward 抛出，避免单个 TP/CP
    rank 提前退出后让其他 rank 永久阻塞在模型 collective。
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        # 同一失败会被每层 wait 重复观察；用 batch index 去重，避免每层打一行。
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
        # 创建不带原 traceback 的私有异常副本。wait_until 会清除它捕获到的
        # traceback，而调用方仍需保留原异常供紧随其后的 logger.exception 使用。
        failure = RuntimeError(f"{type(error).__name__}: {error}")
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(failure)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            # 这里位于模型 forward 内，不能抛异常：否则 scheduler 顶层之前
            # 没有恢复点，其他 rank 还可能继续进入 collective。失败会通过
            # completed_loads 队列离开 forward，再统一 abort 受影响请求。
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake layer-wise KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # 每层等待同一个异常对象都会追加 traceback frame，并间接持有该层
            # activation；大模型故障时会造成显存持续增长。后续逻辑不读取该
            # traceback，因此立即清除是安全且必要的。
            error.__traceback__ = None
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)
                self.reported.discard(index)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()
        self.reported.clear()


class ReadPlanLoadCounter:
    """发布一个 Mooncake ReadPlan，并在不持有 GIL 时按层等待。

    Future 先连接生产者和 forward 消费者，后台线程创建好 native plan 后再
    bind；plan.wait(layer) 与普通逐层 Future 保持相同的失败隔离语义。
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = self.consumer_index = -1
        self.plans: dict[int, Future] = {}
        # 一个 plan 的错误可能在多层 wait 中重复出现，只记录一次。
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
        try:
            future.result().wait(threshold)
        except BaseException as error:
            # 不从 forward 抛出 rank-local 错误，否则 peer TP/CP rank 可能卡在
            # 后续 collective。load_layer_wise 会把 False 写入完成队列，cache
            # 在 attention group 上汇总 verdict，并在 forward 结束后终止请求。
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake ReadPlan KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # 避免重复 wait 形成 traceback 链并长期持有各层 activation。
            error.__traceback__ = None
        finally:
            if threshold == self.num_layers - 1:
                self.plans.pop(index, None)
                self.reported.discard(index)

    def reset(self) -> None:
        self.producer_index = self.consumer_index = -1
        self.plans.clear()
        self.reported.clear()


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self.page_size = params.page_size
        # page-wise 模式只在 key 数达到阈值时启用；小批次仍保留逐层 I/O，
        # 避免聚合所有 layer metadata 的固定开销。
        self.page_wise_load_threshold = (
            server_args.mooncake_page_wise_load_threshold
        )
        self.enable_page_wise_load = server_args.mooncake_enable_page_wise_load
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
        tp_size = server_args.tp_size
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
            dp_rank=getattr(params, "dp_rank", None),
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(
                storage_config,
                mem_pool=None,
                # 只有 SGLang metrics 开启时才让 Mooncake client 监听 HTTP
                # 端口，避免无监控部署占用端口或在多进程下产生冲突。
                enable_client_http_server=params.enable_metrics,
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
        # ReadPlan 为 opt-in：由 Mooncake native 层一次接收全部 pool/layer
        # layout，减少 Python 逐层构造 range metadata 的开销。
        self.read_plan_enabled = os.environ.get("SGLANG_MOONCAKE_READ_PLAN", "0") == "1"
        # range reuse 允许 native plan 复用稳定地址布局，但必须建立在
        # ReadPlan 之上；普通 range-get 路径没有可复用的 plan 状态。
        self.read_plan_reuse_ranges = (
            os.environ.get("SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES", "0") == "1"
        )
        # 若 prefetch 请求最老 key 的 lease 年龄仍小于该阈值，准入时可跳过
        # refresh。master 不会淘汰处于 lease 内的对象，所以 refresh 只是在
        # 延长保护时间；0 保持无条件刷新。阈值必须为准入到 load 完成预留
        # 足够的部署 TTL，不能把“当前仍有效”误当成“加载期间一直有效”。
        self.session_refresh_age_s = _session_refresh_age_config()
        self._disable_refresh_skip_for_groups()
        self.host_prefetch_enabled = bool(
            getattr(
                server_args,
                "mooncake_enable_waiting_queue_dfs_prefetch",
                False,
            )
        )
        self.host_prefetch_limit = int(
            getattr(
                server_args,
                "mooncake_waiting_queue_dfs_prefetch_max_requests",
                8,
            )
        )
        self.host_prefetch_worker_count = int(
            getattr(
                server_args,
                "mooncake_waiting_queue_dfs_prefetch_workers",
                2,
            )
        )
        self.host_prefetch_max_bytes = int(
            getattr(
                server_args,
                "mooncake_waiting_queue_dfs_prefetch_max_bytes",
                1073741824,
            )
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
        if self.host_prefetch_max_bytes <= 0:
            raise ValueError(
                "--mooncake-waiting-queue-dfs-prefetch-max-bytes must be "
                f"positive, got {self.host_prefetch_max_bytes}."
            )
        if self.read_plan_reuse_ranges and not self.read_plan_enabled:
            raise ValueError(
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES requires "
                "SGLANG_MOONCAKE_READ_PLAN=1"
            )
        if self.read_plan_reuse_ranges and self.enable_page_wise_load:
            # page-wise 会把各层 range 重新组织为 key-major 完整页布局，与
            # layer-wise reuse 所假定的稳定 range 形状不兼容。
            raise ValueError(
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES is incompatible with "
                "page-wise Mooncake loads"
            )
        if self.read_plan_enabled:
            # 提前检查 Python binding 能力，避免服务启动成功后第一次 L3
            # 命中才因旧 Mooncake package 缺少 API 而失败。
            if not callable(getattr(self.storage.store, "create_read_plan", None)):
                raise RuntimeError(
                    "Mooncake ReadPlan requires a package with create_read_plan. "
                    "Install the ReadPlan-enabled Mooncake package."
                )
            logger.info(
                "Mooncake ReadPlan enabled; address reuse=%s",
                self.read_plan_reuse_ranges,
            )
        self.host_prefetch_enabled = self._resolve_host_prefetch_enabled(
            params, requested=self.host_prefetch_enabled
        )
        requested_async_lookup = envs.SGLANG_MOONCAKE_ASYNC_LOOKUP.get()
        if requested_async_lookup and params.pp_size != 1:
            raise ValueError("SGLANG_MOONCAKE_ASYNC_LOOKUP currently requires PP1")
        # Waiting-queue DFS prefetch already starts from an eager lookup and
        # owns a private Mooncake session. Preserve that existing path when
        # both opt-ins are set rather than turning its initial lookup into an
        # unfinished asynchronous probe.
        self.async_lookup_enabled = (
            requested_async_lookup and not self.host_prefetch_enabled
        )
        if requested_async_lookup and self.host_prefetch_enabled:
            logger.warning(
                "Mooncake asynchronous lookup disabled because waiting-queue "
                "DFS prefetch is enabled"
            )
        self.async_lookup_workers = 1
        if self.async_lookup_enabled:
            self.async_lookup_workers = envs.SGLANG_MOONCAKE_ASYNC_LOOKUP_WORKERS.get()
            if self.async_lookup_workers < 1:
                raise ValueError("SGLANG_MOONCAKE_ASYNC_LOOKUP_WORKERS must be >= 1")
            logger.info(
                "Mooncake asynchronous L3 lookup enabled (PP1, workers=%d)",
                self.async_lookup_workers,
            )

        self.storage_metrics_collector = None
        if params.enable_metrics:
            # 将并行拓扑写入固定 labels，确保不同 TP/DP/PP/CP 进程的数据
            # 不会聚合到无法区分的同一条 Prometheus time series。
            labels = {
                "storage_backend": "mooncake_direct",
                "tp_rank": tp_rank,
                "dp_rank": _get_mooncake_storage_metrics_dp_rank(server_args, params),
                "pp_rank": params.pp_rank,
                "pp_size": params.pp_size,
                "attn_cp_rank": params.attn_cp_rank,
                "attn_cp_size": params.attn_cp_size,
            }
            if server_args.extra_metric_labels:
                labels.update(server_args.extra_metric_labels)
            collector_cls = resolve_collector_class(
                server_args, STAT_LOGGER_ROLE_STORAGE, StorageMetricsCollector
            )
            self.storage_metrics_collector = collector_cls(labels=labels)

        self.register_buffers()
        # 两种 counter 向模型暴露相同的 wait_until(layer) 契约；区别在于
        # 普通路径每层一个 Future，ReadPlan 路径由单个 native plan 按层等待。
        if self.read_plan_enabled:
            self.layer_done_counter = ReadPlanLoadCounter(self.num_layers)
        else:
            self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        # rid -> 逻辑 token 数。一个逻辑 page 可能展开成 KV、SWA、Indexer
        # 等多个物理 object，因此必须先按 page key 去重，不能按 transfer 求和。
        self.pending_load_tokens: dict[str, int] = {}
        self.prepared_load_sessions: dict[str, list[str]] = {}
        self.session_refcounts: dict[str, int] = {}
        self.session_sources: dict[str, str] = {}
        # key -> 最近一次授予 master read lease 的 RPC（session start/refresh）
        # 发起前记录的 monotonic 时间。该值不会晚于 Mooncake 自身 lease 起点，
        # 因而据此计算出的 age 只会偏大，是保守估计。
        self.session_lease_base: dict[str, float] = {}
        self.session_lock = threading.Lock()
        self.host_prefetch_lock = threading.Lock()
        self.host_prefetch_entries: dict[str, dict[str, object]] = {}
        self.host_prefetch_generation = 0
        self.host_prefetch_queue: Queue[
            tuple[str, list[PoolTransfer]] | None
        ] = Queue()
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        # (rids, success) per started load batch. Pushed from a finally so a
        # failed batch still releases the tree-side locks it pinned.
        self.completed_loads: Queue[tuple[list[str], bool]] = Queue()
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
        if self.host_prefetch_enabled:
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

    # 兼容绕过 __init__ 构造的轻量实例（主要是测试 fixture）；正常实例会用
    # 已校验的环境配置覆盖该类属性。
    session_refresh_age_s = 0.0

    def _disable_refresh_skip_for_groups(self) -> None:
        group_semantics = getattr(self.storage, "_can_use_group_semantics", None)
        if (
            self.session_refresh_age_s > 0
            and callable(group_semantics)
            and group_semantics()
        ):
            # group semantics 只在剩余 lease 低于一半时真正延期，但每次 reply
            # 都报告完整 TTL；据此更新 client deadline 最多会比 master 晚
            # TTL/2，因此该模式下禁止 refresh-skip，采用保守刷新。
            logger.warning(
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S is ignored with Mooncake "
                "group semantics"
            )
            self.session_refresh_age_s = 0.0

    def _lease_bases(self) -> dict[str, float]:
        bases = getattr(self, "session_lease_base", None)
        if bases is None:
            bases = self.session_lease_base = {}
        return bases

    def _local_host_prefetch_capability(self, requested: bool) -> tuple[bool, str]:
        """检查本 rank 是否具备 waiting-queue DFS prefetch 的完整 native 能力。"""
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
            )
            if not callable(getattr(store, name, None))
        ]
        if missing:
            return False, "Mooncake package lacks " + ", ".join(missing)
        try:
            if store.dfs_prefetch_arena_available():
                return True, "ready"
            status = getattr(store, "dfs_prefetch_arena_status", None)
            reason = status() if callable(status) else "unavailable"
        except Exception as exc:
            reason = f"arena query failed: {exc!r}"
        return False, (
            f"DFS prefetch arena unavailable ({reason}); set "
            "MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES on prefill"
        )

    def _resolve_host_prefetch_enabled(
        self, params: CacheInitParams, *, requested: bool
    ) -> bool:
        """在启动 worker 前一次性协商 waiting-queue prefetch 开关。

        所有构造该 linker 的 rank 都无条件参加，使 scheduler 的逐请求
        prefetch collective 要么全员进入、要么全员跳过。通信组及执行顺序与
        ``UnifiedRadixCache._all_reduce_attn_groups`` 完全一致。
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

    def disable_waiting_queue_prefetch(self) -> None:
        # scheduler 在任何 submission 前发现某 rank 不支持 prefetch；本 rank
        # 的 host-prefetch worker 保持启动但停在空队列，不会接触 backend。
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
        # PP0 queries every PP shard; the wrapper's existing TP/CP reduction
        # then selects a boundary that all ranks can restore.
        result = self.storage.batch_exists_v2(page_keys, expanded, query_all_pp=True)
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

    def lookup_in_worker(
        self, rid: str, transfers: list[PoolTransfer]
    ) -> list[int]:
        """Query Mooncake metadata without scheduler-thread cache mutation."""
        return self.lookup(rid, transfers)

    def revalidate_load(self, transfers: list[PoolTransfer]) -> bool:
        valid = self._revalidate_load_local(transfers)
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

        Failures inside the revalidation itself never propagate: this hook
        guards a hot scheduling path, so an unexpected error falls back to
        "validated" and the async load path retains its own semantics.
        """
        try:
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
        except BaseException:
            logger.exception("Mooncake direct linker load revalidation error")
            return True
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
        # lookup 已确定“所有 component 都可恢复”的公共边界，随后 tree insert
        # 会去掉仍驻留在 L1 的 page。因此这里允许部分 physical pool；某次
        # load 甚至可能只剩 SWA 等 side pool，而没有主 KV transfer。
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
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
        return True

    def prepare_load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """为将来的 range read 按 storage key 获取并引用 Mooncake session。"""
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        return self._prepare_expanded_load(rid, expanded)

    def _prepare_expanded_load(
        self, rid: str, expanded: list[PoolTransfer]
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

        with self.session_lock:
            if rid in self.prepared_load_sessions:
                return True
            acquired = []
            new_keys = []
            for key in keys:
                if key in self.session_refcounts:
                    self.session_refcounts[key] += 1
                    self.session_sources.setdefault(key, "unknown")
                    # 另一 holder 可能因 refresh 失败或 read 过期而让 Mooncake
                    # 销毁 session，而这里只能在下次 refresh 时发现。共享过的
                    # key 清除本地 lease base，在所有引用释放并重新 start 前，
                    # 都只能走保守 refresh，不能依赖年龄优化。
                    self._lease_bases().pop(key, None)
                    acquired.append(key)
                else:
                    new_keys.append(key)

            lease_base = time.monotonic()
            try:
                if new_keys and self.host_prefetch_enabled:
                    results, sources = (
                        self.storage.store.batch_get_session_start_with_sources(
                            new_keys
                        )
                    )
                    results = list(results)
                    sources = list(sources)
                elif new_keys:
                    results = list(
                        self.storage.store.batch_get_session_start(new_keys)
                    )
                    sources = ["unknown"] * len(results)
                else:
                    results = []
                    sources = []
            except BaseException:
                logger.warning(
                    "Mooncake get session start raised for rid=%s", rid, exc_info=True
                )
                self._rollback_session_refs_locked(acquired)
                return False

            failed = len(results) != len(new_keys) or any(
                result != 0 for result in results
            )
            for index, key in enumerate(new_keys):
                if index < len(results) and results[index] == 0:
                    self.session_refcounts[key] = 1
                    source = sources[index] if index < len(sources) else "unknown"
                    self.session_sources[key] = str(source).lower()
                    self._lease_bases()[key] = lease_base
                    acquired.append(key)
            if failed:
                self._rollback_session_refs_locked(acquired)
                return False

            self.prepared_load_sessions[rid] = acquired
        return True

    @staticmethod
    def _mooncake_dfs_alignment() -> int:
        try:
            return max(
                4096,
                int(os.environ.get("MOONCAKE_DFS_ALIGNMENT", "4096") or 4096),
            )
        except ValueError:
            return 4096

    @staticmethod
    def _estimate_pinned_arena_capacity(
        object_sizes: list[int], alignment: int
    ) -> int:
        """保守复现 Mooncake 对齐后的 DFS arena 分配容量，用于准入背压。"""
        arena_size = 0
        for object_size in object_sizes:
            aligned_object_size = (
                (object_size + alignment - 1) // alignment
            ) * alignment
            arena_size = ((arena_size + alignment - 1) // alignment) * alignment
            arena_size += aligned_object_size

        if not arena_size:
            return 0
        padded_size = arena_size + alignment - 1
        if padded_size <= 1 << 20:
            capacity = 4 << 10
            while capacity < padded_size:
                capacity *= 2
            return capacity
        large_alignment = 2 << 20
        return (
            (padded_size + large_alignment - 1) // large_alignment
        ) * large_alignment

    def _get_host_prefetch_object_sizes(
        self, transfers: list[PoolTransfer]
    ) -> dict[str, int] | None:
        """把带 pool tag 的 storage key 映射到单页对象的精确 payload 大小。"""
        try:
            expanded = self.pool_group.resolve_transfers(
                transfers, allow_partial=True, allow_missing_kv=True
            )
            if not expanded:
                return None

            object_sizes: dict[str, int] = {}
            for transfer in expanded:
                page_keys = list(transfer.keys or ())
                if not page_keys:
                    continue
                component_keys, key_multiplier = (
                    self.storage._get_hybrid_page_component_keys(
                        page_keys, transfer
                    )
                )
                pool = self.pools.get(transfer.name)
                if pool is None or key_multiplier <= 0:
                    return None

                # 每行 metadata 描述一个逻辑 page；按 MooncakeStore 将 buffer
                # 打包进上述 component key 的相同规则分组，容量估算才与 native
                # arena 实际收到的对象一一对应。
                page_buffer_sizes = [
                    size
                    for component in pool.buffer_meta
                    for _, _, size in component
                ]
                if (
                    not page_buffer_sizes
                    or len(page_buffer_sizes) % key_multiplier
                    or len(component_keys) != len(page_keys) * key_multiplier
                ):
                    return None
                buffers_per_object = len(page_buffer_sizes) // key_multiplier
                if buffers_per_object <= 0:
                    return None
                object_payload_sizes = [
                    sum(
                        page_buffer_sizes[
                            object_index
                            * buffers_per_object : (object_index + 1)
                            * buffers_per_object
                        ]
                    )
                    for object_index in range(key_multiplier)
                ]
                tagged_keys = self.storage._tag_keys(component_keys)
                for index, key in enumerate(tagged_keys):
                    size = object_payload_sizes[index % key_multiplier]
                    if size <= 0:
                        return None
                    object_sizes[key] = max(object_sizes.get(key, 0), size)
            return object_sizes or None
        except Exception:
            logger.warning(
                "Could not estimate Mooncake waiting-queue prefetch bytes.",
                exc_info=True,
            )
            return None

    def submit_host_prefetch(
        self, rid: str, transfers: list[PoolTransfer]
    ) -> bool:
        if not self.host_prefetch_enabled:
            return False
        kv_transfer = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV), None
        )
        if kv_transfer is None or not kv_transfer.keys:
            return False
        object_sizes = self._get_host_prefetch_object_sizes(transfers)
        if object_sizes is None:
            return False

        alignment = self._mooncake_dfs_alignment()
        estimated_bytes = self._estimate_pinned_arena_capacity(
            list(object_sizes.values()), alignment
        )
        if not estimated_bytes or estimated_bytes > self.host_prefetch_max_bytes:
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
                }
                for entry in self.host_prefetch_entries.values()
            )
            reserved_bytes = sum(
                int(entry.get("reserved_bytes", 0))
                for entry in self.host_prefetch_entries.values()
            )
            if (
                rid in self.host_prefetch_entries
                or active >= self.host_prefetch_limit
                or reserved_bytes + estimated_bytes > self.host_prefetch_max_bytes
            ):
                return False
            self.host_prefetch_generation += 1
            generation = self.host_prefetch_generation
            queued_transfers = list(transfers)
            self.host_prefetch_entries[rid] = {
                "state": "queued",
                "cancelled": False,
                "transfers": queued_transfers,
                "session_rid": self._host_prefetch_session_rid(rid, generation),
                "object_sizes": object_sizes,
                "reserved_bytes": estimated_bytes,
            }

        # session 创建本身可能访问 DFS。scheduler 只发布 work item，后台
        # worker 独占 preparation 与 DFS read，避免 scheduler 热线程被 I/O 阻塞。
        self.host_prefetch_queue.put((rid, queued_transfers))
        return True

    def get_host_prefetch_status(self, rid: str) -> str | None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None:
                return None
            return str(entry.get("state"))

    def revalidate_host_prefetch(self, rid: str) -> bool:
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
            # 没有 lease base 的 key（本代曾被共享，或 refresh 曾失败）按无限
            # 老处理，强制刷新，不能根据不可信的本地时间跳过 RPC。
            bases = [self._lease_bases().get(key) for key in keys]
            private = all(self.session_refcounts.get(key, 0) == 1 for key in keys)
        if not keys:
            return False
        oldest = None if None in bases else min(bases)
        # 必须在获得 session_lock 后再计算 age：锁可能被另一请求跨越较慢的
        # session-start RPC 持有，等待锁的时间同样已经消耗 master lease。
        age = float("inf") if oldest is None else time.monotonic() - oldest
        if (
            self.session_refresh_age_s > 0
            and private
            and age < self.session_refresh_age_s
        ):
            # 此时每个 key 至少还剩 ``TTL - age`` 的 master read lease；该窗口
            # 内 master 既不会淘汰对象，也不会用新对象替换它。
            return True
        lease_base = time.monotonic()
        try:
            results = list(self.storage.store.batch_get_session_refresh(keys))
        except BaseException:
            logger.warning(
                "Mooncake waiting-queue prefetch refresh failed for rid=%s",
                rid,
                exc_info=True,
            )
            results = []
        ok = len(results) == len(keys) and all(result == 0 for result in results)
        with self.session_lock:
            bases = self._lease_bases()
            for key in keys:
                if not ok:
                    # refresh 失败可能已让 Mooncake 回收 session，本地 base
                    # 立即失效，禁止后续请求继续走 refresh-skip。
                    bases.pop(key, None)
                elif key in bases:
                    # 只有整代都由单一 holder 私有的 key 才保留 base；已经 end
                    # 或曾共享的 key 没有可安全续写的本地 deadline。
                    bases[key] = lease_base
        return ok

    def revalidate_no_prefetch_needed(self, rid: str) -> bool:
        """重新检查“无需 DFS prefetch”请求的 key 是否仍然存在。

        worker 发现没有 DFS key 后会释放 session，请求等待准入期间没有 lease
        pin 住对象。返回 False 会让所有 rank 回到普通路径，由其 revalidation
        把已淘汰前缀降级成 cache miss，而不是在异步 load 中失败。
        """
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
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None or entry.get("state") != "dfs_prefetched":
                return False
            session_rid = str(entry["session_rid"])
            with self.session_lock:
                if (
                    rid in self.prepared_load_sessions
                    or session_rid not in self.prepared_load_sessions
                ):
                    return False
                self.prepared_load_sessions[rid] = self.prepared_load_sessions.pop(
                    session_rid
                )
            entry["state"] = "claimed"
        return True

    def cancel_host_prefetch(self, rid: str) -> None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None:
                return
            state = str(entry.get("state"))
            session_rid = str(entry["session_rid"])
            if state in {"preparing", "reading", "loading"}:
                entry["cancelled"] = True
                return
            self.host_prefetch_entries.pop(rid, None)
        self._abort_prepared_load_now(rid if state == "claimed" else session_rid)

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
                        continue
                    entry["state"] = "preparing"

                try:
                    prepared = self.prepare_load(session_rid, transfers)
                except BaseException:
                    logger.warning(
                        "Mooncake waiting-queue session preparation failed "
                        "for rid=%s",
                        rid,
                        exc_info=True,
                    )
                    prepared = False
                with self.session_lock:
                    keys = list(self.prepared_load_sessions.get(session_rid, ()))
                if not prepared or not keys:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
                    continue

                with self.session_lock:
                    sources = {
                        key: self.session_sources.get(key, "unknown")
                        for key in keys
                    }
                object_sizes = entry.get("object_sizes", {})
                if not isinstance(object_sizes, dict) or any(
                    key not in object_sizes for key in keys
                ):
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
                    continue
                dfs_keys = [
                    key
                    for key in keys
                    if sources[key] in {"dfs", "unknown"}
                ]
                if not dfs_keys:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "no_prefetch_needed", keys=keys
                    )
                    continue

                alignment = self._mooncake_dfs_alignment()
                dfs_estimated_bytes = self._estimate_pinned_arena_capacity(
                    [int(object_sizes[key]) for key in dfs_keys], alignment
                )
                with self.host_prefetch_lock:
                    current = self.host_prefetch_entries.get(rid)
                    if current is not entry or current.get("cancelled"):
                        cancelled = True
                        within_budget = False
                    else:
                        cancelled = False
                        other_reserved_bytes = sum(
                            int(other.get("reserved_bytes", 0))
                            for other_rid, other in self.host_prefetch_entries.items()
                            if other_rid != rid
                        )
                        within_budget = (
                            other_reserved_bytes + dfs_estimated_bytes
                            <= self.host_prefetch_max_bytes
                        )
                        if within_budget:
                            current["reserved_bytes"] = dfs_estimated_bytes
                            current["state"] = "reading"
                if cancelled:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
                    continue
                if not within_budget:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
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
                    # session 会在下方释放，但保留 key 供准入时重新检查（见
                    # revalidate_no_prefetch_needed）。key 与 state 在同一把锁下
                    # 同时发布，读者不会看到终态却拿不到对应 key 集合。
                    current["no_prefetch_keys"] = list(keys or ())
                    current["reserved_bytes"] = 0
                current["state"] = outcome
                keep_session = outcome == "dfs_prefetched"
            if current is entry:
                if cancelled:
                    self.host_prefetch_entries.pop(rid, None)
                elif outcome not in {"dfs_prefetched", "no_prefetch_needed"}:
                    current["state"] = "failed"
                    current["reserved_bytes"] = 0
        if not keep_session:
            self._abort_prepared_load_now(session_rid)

    def _rollback_session_refs_locked(self, keys: list[str]) -> None:
        to_end = []
        for key in keys:
            count = self.session_refcounts.get(key, 0)
            if count <= 1:
                self.session_refcounts.pop(key, None)
                self.session_sources.pop(key, None)
                self._lease_bases().pop(key, None)
                to_end.append(key)
            else:
                self.session_refcounts[key] = count - 1
        if to_end:
            try:
                self.storage.store.batch_get_session_end(to_end)
            except BaseException:
                logger.warning(
                    "Mooncake get session cleanup failed for %d keys.",
                    len(to_end),
                    exc_info=True,
                )

    def _abort_prepared_load_now(self, rid: str) -> None:
        with self.session_lock:
            keys = self.prepared_load_sessions.pop(rid, [])
            self._rollback_session_refs_locked(keys)

    def abort_prepared_load(self, rid: str) -> None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is not None and entry.get("state") in {"claimed", "loading"}:
                self.host_prefetch_entries.pop(rid, None)
        self._abort_prepared_load_now(rid)

    def cancel_queued_load(self, rid: str) -> bool:
        getattr(self, "pending_load_tokens", {}).pop(rid, None)
        cancelled = self.pending_loads.pop(rid, None) is not None
        if cancelled:
            self.abort_prepared_load(rid)
        return cancelled

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

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
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        ready_event = device_module.Event()
        ready_event.record()
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
                try:
                    ready_event.synchronize()
                    success = self.load_layer_wise(
                        counter_index, list(pending.items())
                    )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    # 无论正常返回还是异常，都必须消费本批次的 token 记录并
                    # 发布完成事件；否则指标状态和 tree-side node lock 会泄漏。
                    self._finish_prefetch_metrics(list(pending), success is True)
                    self.completed_loads.put((list(pending), success))
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
    ) -> bool:
        success = False
        maybe_fail = arm_load_failure_injection(self.tp_rank)
        try:
            # 所有读取路径（普通逐层、page-wise、ReadPlan）统一借用 prepared
            # session。已有 DFS 预取 session 可直接复用，没有则在此同步建立。
            for rid, transfers in request_transfers:
                with self.session_lock:
                    prepared = rid in self.prepared_load_sessions
                if not prepared and not self._prepare_expanded_load(rid, transfers):
                    raise RuntimeError(
                        f"Mooncake get session preparation failed for rid={rid}."
                    )
            if getattr(self, "read_plan_enabled", False):
                # ReadPlan 自己负责 native range 的展开和逐层 wait；成功必须
                # 显式返回 True，避免后台线程把 None 当作失败发布。
                self.load_with_read_plan(counter_index, request_transfers)
                return True
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for _rid, transfers in request_transfers:
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    keys.extend(self.storage._tag_keys(component_keys))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
            if self.enable_page_wise_load and any(
                len(keys) >= self.page_wise_load_threshold
                for keys, _ in batches.values()
            ):
                self._load_page_wise(counter_index, batches, maybe_fail)
                success = True
                return success

            for layer in range(self.num_layers):
                # 默认路径逐层完成所有 pool 的 range read；只有该 layer 的
                # 每个 physical pool 都校验成功后，才允许模型读取这一层。
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    maybe_fail(name, f"layer={layer}")
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
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
            # borrowed session 的所有权仍属于 linker；无论 I/O 是否成功都在
            # 整批结束后递减引用，最后一个 holder 才真正结束 Mooncake session。
            for rid, _ in request_transfers:
                self.abort_prepared_load(rid)
        return success

    def _finish_prefetch_metrics(self, rids: list[str], success: bool) -> None:
        """结算一批 direct load 的成功 token 指标并清理临时计数。"""
        pending_load_tokens = getattr(self, "pending_load_tokens", {})
        tokens = sum(pending_load_tokens.pop(rid, 0) for rid in rids)
        if not success or tokens <= 0:
            return
        self._log_prefetched_tokens(tokens)
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

    def _log_prefetched_tokens(self, tokens: int) -> None:
        """隔离指标后端异常，禁止监控故障影响 KV 数据路径。"""
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None or tokens <= 0:
            return
        try:
            collector.log_prefetched_tokens(tokens)
        except BaseException:
            logger.warning(
                "Failed to record SGLang direct-storage prefetch metrics.",
                exc_info=True,
            )

    def _load_page_wise(self, counter_index: int, batches, maybe_fail) -> None:
        """一次聚合读取每个 page 的全部 layer range，再整体发布。

        与逐层路径相比，该模式牺牲 layer/compute overlap，换取跨 layer、跨
        physical pool 的单次 Mooncake range 调用。任何 object short read 都
        使整批失败，不能向模型暴露只填充了部分层的 page。
        """
        all_keys: list[str] = []
        all_ptrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        all_offsets: list[list[int]] = []
        for name, (keys, locations) in batches.items():
            # 临时数组保持 key-major 结构：每个 key 对应其所有 layer、所有
            # component 的目标地址、长度和 object 内偏移。
            ptrs: list[list[int]] = [[] for _ in keys]
            sizes: list[list[int]] = [[] for _ in keys]
            offsets: list[list[int]] = [[] for _ in keys]

            for layer in range(self.num_layers):
                meta = self.pools[name].get_prepared_layer_range_meta(
                    locations, layer
                )
                if meta is None:
                    continue
                layer_ptrs, layer_sizes, layer_offsets = meta
                if not (
                    len(layer_ptrs)
                    == len(layer_sizes)
                    == len(layer_offsets)
                    == len(keys)
                ):
                    # metadata 数量不齐时禁止进入 native API，避免 C++ 侧按
                    # 不一致数组长度读取并写入错误地址。
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

        # 不同 pool 的 key 已带独立 suffix，而 range API 本身不接收 pool
        # 参数，所以可安全展平成一个全局批次。调用前再次验证四组元数据对齐。
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

        # 故障注入也以整个聚合调用为单位，模拟真实 native range get 失败。
        maybe_fail("aggregated", "complete_page")
        result = self.storage.store.batch_get_into_multi_buffer_ranges(
            all_keys, all_ptrs, all_sizes, all_offsets
        )
        expected = [sum(item) for item in all_sizes]
        # Mooncake 可能返回 None、scalar error code 或逐 key byte count。
        # 只有结果长度和值都与期望完全相等才算成功，short read 不可接受。
        if result is None or isinstance(result, int) or list(result) != expected:
            pool_counts = {
                str(name): len(keys) for name, (keys, _) in batches.items()
            }
            raise RuntimeError(
                "Mooncake aggregated range get failed for "
                f"pools={pool_counts}, complete_page: transferred={result}, "
                f"expected={expected}"
            )

        # 所有 page 全量成功后一次性释放所有 layer，维持 all-or-nothing 可见性。
        for layer in range(self.num_layers):
            self.layer_done_counter.complete(counter_index, layer)

    def _prepare_read_plan_layouts(
        self, request_transfers: list[tuple[str, list[PoolTransfer]]]
    ):
        # 每个 pool 只合并一次 index，同时保持 request/key 顺序。每个 component
        # 描述 (base address, row stride, byte count, source offset)；location
        # 可以不连续，最终地址展开由 Mooncake C++ ReadPlan 完成。
        batches = {}
        for _rid, transfers in request_transfers:
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
                # None 表示该逻辑层没有该 sidecar；int 表示单 buffer；tuple
                # 表示 target 与 MTP draft 等多个 packed buffer 同属此层。
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
        request_transfers: list[tuple[str, list[PoolTransfer]]],
    ) -> None:
        layouts = self._prepare_read_plan_layouts(request_transfers)
        plan = self.storage.store.create_read_plan(
            layouts,
            self.num_layers,
            reuse_ranges=self.read_plan_reuse_ranges,
            page_wise=self.enable_page_wise_load,
            borrowed_sessions=True,
            # 保留 pool/tensor owner，防止 native plan 尚持有地址时 Python
            # 对象已被 GC；session 则由外层 finally 统一归还，plan 只借用。
            buffer_owners=self.pools,
        )
        self.layer_done_counter.bind(counter_index, plan)
        # run()/wait() 会释放 GIL。普通模式下，每层只有在所有 pool 的字节数
        # 校验完成后才可见，最后一次 wait 负责 native cleanup；page-wise 下
        # wait(0) 等待全部 page，后续 wait 为空操作，与 _load_page_wise 的
        # all-or-nothing 发布语义一致。
        plan.run()

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
        # 入队时记录时间而不是 worker 开始时间，使指标包含排队、device event
        # 等待和实际 Mooncake 写入的完整端到端 wall-clock 延迟。
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
            # direct linker 无法可靠判断 Mooncake 最终路由到 DFS 还是本地介质，
            # 因此统一标记为 mooncake，不根据实现细节猜测 source。
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
        with self.host_prefetch_lock:
            for entry in self.host_prefetch_entries.values():
                entry["cancelled"] = True
        if self.host_prefetch_enabled:
            self.host_prefetch_queue.join()
        for rid in list(self.pending_loads):
            self.abort_prepared_load(rid)
        self.pending_loads.clear()
        getattr(self, "pending_load_tokens", {}).clear()
        self.load_queue.join()
        self.offload_queue.join()
        for rid in list(self.prepared_load_sessions):
            self.abort_prepared_load(rid)
        with self.host_prefetch_lock:
            self.host_prefetch_entries.clear()
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
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        for _ in self.host_prefetch_threads:
            self.host_prefetch_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        for thread in self.host_prefetch_threads:
            thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
