import threading
from array import array
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertResult, MatchResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _build_dsa_device_pool_group,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache_linker import (
    DevicePoolEntry,
    DevicePoolGroup,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, slots=None):
        self.slots = slots
        self.freed = []
        self.mapping = []

    def available_size(self):
        return 100

    def alloc(self, size):
        if self.slots is None:
            return torch.arange(1, size + 1, dtype=torch.int64)
        value = self.slots[:size].clone()
        self.slots = self.slots[size:]
        return value

    def free(self, value):
        self.freed.append(value.clone())

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def test_sparse_multi_component_layer_ranges():
    k0 = torch.zeros((8, 3), dtype=torch.uint8)
    k2 = torch.zeros((8, 5), dtype=torch.uint8)
    v0 = torch.zeros((8, 7), dtype=torch.uint8)
    v2 = torch.zeros((8, 11), dtype=torch.uint8)
    pool = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[[k0, k2], [v0, v2]],
        layer_mapping={0: 0, 2: 1},
        page_size=2,
        rows_are_pages=False,
        packed=False,
    )

    indices = torch.tensor([0, 1, 4, 5])
    locations = pool.prepare_locations(indices)
    assert locations == [0, 4]
    pointers, sizes = pool.get_page_buffer_meta(indices)
    assert pointers == [
        buffer[row].data_ptr() for row in locations for buffer in (k0, k2, v0, v2)
    ]
    assert sizes == [6, 10, 14, 22] * 2
    assert pool.get_prepared_layer_range_meta(locations, 1) is None

    pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
    assert pointers == [
        [k2[0].data_ptr()],
        [v2[0].data_ptr()],
        [k2[4].data_ptr()],
        [v2[4].data_ptr()],
    ]
    assert sizes == [[10], [22], [10], [22]]
    assert offsets == [[6], [14], [6], [14]]


def test_dsa_device_pool_group_preserves_sparse_indexer_layers():
    kv_buffers = [torch.zeros((8, 2), dtype=torch.uint8) for _ in range(3)]
    indexer_buffers = [torch.zeros((4, 2), dtype=torch.uint8) for _ in range(2)]
    kvcache = SimpleNamespace(
        page_size=2,
        layer_num=3,
        start_layer=10,
        kv_buffer=kv_buffers,
        index_k_with_scale_buffer=indexer_buffers,
        indexer_layer_id_to_index={10: 0, 12: 1},
    )

    group = _build_dsa_device_pool_group(kvcache, page_size=2)
    indexer = group.entry_map[PoolName.INDEXER]
    locations = indexer.prepare_locations(torch.tensor([0, 1]))

    assert indexer.layer_mapping == {0: 0, 2: 1}
    assert indexer.get_prepared_layer_range_meta(locations, 1) is None
    pointers, sizes, offsets = indexer.get_prepared_layer_range_meta(locations, 2)
    assert pointers == [[indexer_buffers[1][0].data_ptr()]]
    assert sizes == [[indexer_buffers[1][0].nbytes]]
    assert offsets == [[indexer_buffers[0][0].nbytes]]


def test_dsa_device_pool_group_drops_layer_split_placeholders():
    kv_buffers = [
        torch.zeros((0, 2), dtype=torch.uint8),
        torch.zeros((8, 2), dtype=torch.uint8),
        torch.zeros((0, 2), dtype=torch.uint8),
    ]
    indexer_buffers = [
        torch.zeros((0, 2), dtype=torch.uint8),
        torch.zeros((4, 2), dtype=torch.uint8),
    ]
    kvcache = SimpleNamespace(
        page_size=2,
        layer_num=3,
        start_layer=10,
        kv_buffer=kv_buffers,
        index_k_with_scale_buffer=indexer_buffers,
        indexer_layer_id_to_index={10: 0, 11: 1},
    )

    group = _build_dsa_device_pool_group(kvcache, page_size=2)
    kv = group.entry_map[PoolName.KV]
    indexer = group.entry_map[PoolName.INDEXER]

    assert kv.components == [[kv_buffers[1]]]
    assert kv.layer_mapping == {1: 0}
    assert indexer.components == [[indexer_buffers[1]]]
    assert indexer.layer_mapping == {1: 0}

    locations = kv.prepare_locations(torch.tensor([0, 1]))
    assert kv.get_prepared_layer_range_meta(locations, 0) is None
    assert indexer.get_prepared_layer_range_meta(locations, 0) is None


def test_lookup_returns_sparse_mamba_boundaries():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    pools = [
        SimpleNamespace(
            name=name,
            indices_from_pool=name,
            translate_indices=lambda indices: indices,
        )
        for name in (PoolName.KV, PoolName.MAMBA)
    ]
    linker.pool_group = DevicePoolGroup(pools, num_layers=4, page_size=1)
    linker.pools = linker.pool_group.entry_map
    linker.stats = {"lookup": 0}
    linker.storage = MooncakeStore.__new__(MooncakeStore)
    linker.storage.mem_pool_host = SimpleNamespace(kv_buffer=None)
    linker.storage.pp_rank = 0
    linker.storage.pp_size = 1
    linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
        [f"{key}_{transfer.name}" for key in keys],
        1,
    )
    linker.storage._tag_keys = lambda keys: keys
    linker.storage._batch_exist = lambda keys: [
        int(key.endswith("_kv") or key[0] in ("b", "d")) for key in keys
    ]
    valid = linker.lookup(
        "rid",
        [
            PoolTransfer(name=PoolName.KV, keys=["a", "b", "c", "d"]),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["d"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ],
    )
    assert valid == [2, 4]


def test_lookup_requires_each_pp_shard_for_a_page():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    pools = [
        SimpleNamespace(
            name=name,
            indices_from_pool=name,
            translate_indices=lambda indices: indices,
        )
        for name in (PoolName.KV, PoolName.INDEXER)
    ]
    linker.pool_group = DevicePoolGroup(pools, num_layers=4, page_size=1)
    linker.pools = linker.pool_group.entry_map
    linker.stats = {"lookup": 0}
    linker.storage = MooncakeStore.__new__(MooncakeStore)
    linker.storage.mem_pool_host = SimpleNamespace(kv_buffer=None)
    linker.storage.pp_rank = 0
    linker.storage.pp_size = 2
    linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
        [f"{key}_cp0_pp0_{transfer.name}" for key in keys],
        1,
    )
    linker.storage._tag_keys = lambda keys: keys
    linker.storage._batch_exist = lambda keys: [
        int(key != "b_cp0_pp1_indexer") for key in keys
    ]

    valid = linker.lookup(
        "rid",
        [
            PoolTransfer(name=PoolName.KV, keys=["a", "b"]),
            PoolTransfer(name=PoolName.INDEXER, keys=["a", "b"]),
        ],
    )

    assert valid == [1]


def test_revalidate_load_rejects_evicted_remote_object():
    missing_updates = []
    transfers = [PoolTransfer(name=PoolName.KV, keys=["a", "b"])]
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker._load_revalidation_groups = ()
    linker.pool_group = SimpleNamespace(
        resolve_transfers=lambda items, **kwargs: items
    )
    linker.storage = SimpleNamespace(
        _get_hybrid_page_component_keys=lambda keys, transfer: (
            [f"{key}_component" for key in keys],
            1,
        ),
        _tag_keys=lambda keys: [f"tenant/{key}" for key in keys],
        _batch_exist=lambda keys: [1, 0],
        failed_get_cache=SimpleNamespace(
            update_batch=lambda hits, missing: missing_updates.append((hits, missing))
        ),
    )

    assert linker.revalidate_load(transfers) is False
    assert missing_updates == [([], ["tenant/b_component"])]


def test_tail_hashes_honor_radix_key_limit():
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(page_size=256)
    key = RadixKey(array("q", range(256)), limit=255)
    result = SimpleNamespace(last_device_node=None)

    assert wrapper._tail_hashes(key, result, device_hit_len=0) == []


def test_every_pp_stage_probes_l3_and_agrees_on_extend_boundary():
    """A downstream PP stage must not stop at its shorter local-only hit."""

    canonical_hit_len = 5
    prompt_len = 7
    lookups = []

    class _Component:
        @staticmethod
        def build_external_linker_transfer(_phase, _node, keys):
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

    def run_match(pp_rank, device_hit_len):
        cache = SimpleNamespace(
            page_size=1,
            pp_size=2,
            pp_rank=pp_rank,
            _components_tuple=(_Component(),),
            _all_reduce_attn_groups=lambda _tensor, _op: None,
        )
        backend = SimpleNamespace(
            lookup=lambda rid, transfers: lookups.append(
                (pp_rank, rid, len(transfers[0].keys))
            )
            or [len(transfers[0].keys)]
        )
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = backend
        wrapper.hit_markers = {}
        wrapper.host_prefetch_hits = {}
        wrapper._waiting_queue_prefetch_enabled = False
        wrapper._tail_hashes = lambda _key, _result, local_hit: [
            f"page-{index}" for index in range(local_hit, canonical_hit_len)
        ]

        req = SimpleNamespace(rid=f"req-{pp_rank}", external_cache_hit_length=None)
        root = object()
        result = MatchResult(
            device_indices=torch.arange(device_hit_len, dtype=torch.int64),
            last_device_node=root,
            last_host_node=root,
            best_match_node=root,
        )
        matched = wrapper.match(RadixKey(array("q", range(prompt_len))), req, result)
        return req, matched

    pp0_req, pp0_match = run_match(pp_rank=0, device_hit_len=2)
    pp1_req, pp1_match = run_match(pp_rank=1, device_hit_len=3)

    assert [rank for rank, _rid, _pages in lookups] == [0, 1]
    assert pp0_req.external_cache_hit_length == canonical_hit_len
    assert pp1_req.external_cache_hit_length == canonical_hit_len
    assert (
        len(pp0_match.device_indices) + pp0_match.host_hit_length
        == canonical_hit_len
    )
    assert (
        len(pp1_match.device_indices) + pp1_match.host_hit_length
        == canonical_hit_len
    )
    assert prompt_len - canonical_hit_len == 2  # identical extend length on both stages


def test_load_and_offload_share_gc_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(mooncake_direct_linker, "freeze_gc", calls.append)
    monkeypatch.setattr(
        mooncake_direct_linker.device_module,
        "Event",
        lambda: SimpleNamespace(record=lambda: None),
    )

    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker.page_size = 2
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    linker.gc_frozen = False
    linker.offload_queue = Queue()
    linker.pending_loads = {"first": [object()]}
    linker.layer_done_counter = SimpleNamespace(update_producer=lambda: 3)
    linker.load_queue = Queue()
    linker.stats = {"load": 0}

    assert linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert linker.start_layer_wise_loading() == 3
    linker.pending_loads = {"second": [object()]}
    assert linker.start_layer_wise_loading() == 3

    assert calls == ["Mooncake direct linker"]
    assert linker.stats["load"] == 2


def test_load_waits_for_scheduler_stream(monkeypatch):
    event_calls = []
    loaded = threading.Event()

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_direct_linker.device_module, "Event", _Event)

    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.gc_frozen = True
    linker.pending_loads = {"rid": [object()]}
    linker.layer_done_counter = SimpleNamespace(update_producer=lambda: 7)
    linker.load_queue = Queue()
    linker.completed_loads = Queue()
    linker.host_prefetch_enabled = False
    linker.stats = {"load": 0}

    def load_layer_wise(counter_index, transfers):
        assert event_calls == ["record", "synchronize"]
        assert counter_index == 7
        assert len(transfers) == 1
        loaded.set()

    linker.load_layer_wise = load_layer_wise
    thread = threading.Thread(target=linker.load_thread_func, daemon=True)
    thread.start()

    assert linker.start_layer_wise_loading() == 7
    assert loaded.wait(timeout=5)
    linker.load_queue.join()
    linker.load_queue.put(None)
    thread.join(timeout=5)


def test_offload_runs_on_background_thread(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()
    worker_threads = []
    event_calls = []

    monkeypatch.setattr(mooncake_direct_linker, "freeze_gc", lambda _: None)

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_direct_linker.device_module, "Event", _Event)

    class _Storage:
        def batch_set_v2(self, transfers):
            assert event_calls == ["record", "synchronize"]
            worker_threads.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=5)
            return {
                transfer.name: [True] * len(transfer.keys) for transfer in transfers
            }

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
        get_hybrid_pool_buffer=lambda: [],
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_size = 2
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    linker.pools = linker.pool_group.entry_map
    linker.storage = _Storage()
    linker.gc_frozen = False
    linker.stats = {"lookup": 0, "load": 0, "offload": 0}
    linker.offload_queue = Queue()
    linker.offload_results = Queue()
    linker.offload_thread = threading.Thread(
        target=linker.offload_thread_func, daemon=True
    )
    linker.offload_thread.start()

    assert linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert started.wait(timeout=5)
    assert linker.num_completed_offloads() == 0
    assert worker_threads == [linker.offload_thread.ident]
    assert worker_threads[0] != caller_thread

    release.set()
    linker.offload_queue.join()
    assert linker.num_completed_offloads() == 1
    assert linker.pop_completed_offload()
    linker.offload_queue.put(None)
    linker.offload_thread.join(timeout=5)


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    results = []
    linker = SimpleNamespace(
        offload=lambda transfers: True,
        num_completed_offloads=lambda: len(results),
        pop_completed_offload=lambda: results.pop(0),
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache_linker = linker
    wrapper.pending_offloads = []
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(id=node_id, external_cache_stored=False)
    wrapper.cache = SimpleNamespace(
        _components_tuple=(_Component(),),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
        resolve_node_handle=lambda value: node if value == node_id else None,
    )

    wrapper.offload_nodes([node_id])
    assert locks == [node_id]
    assert node.external_cache_stored
    assert not unlocks

    results.append(False)
    assert wrapper.num_completed_offloads() == 1
    wrapper.drain_offloads(finish_count=1)
    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


def test_check_hicache_events_drains_common_tp_offloads():
    drained = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        num_completed_offloads=lambda: 3,
        drain_offloads=drained.append,
    )

    def reduce_to_common_count(count, op):
        assert op == torch.distributed.ReduceOp.MIN
        count.fill_(1)

    cache._all_reduce_attn_groups = reduce_to_common_count
    cache.check_hicache_events()

    assert drained == [1]


def test_sync_dfs_health_policy_applies_attention_group_minimum():
    applied = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        get_local_dfs_health_policy=lambda: (True, True),
        apply_dfs_health_policy=lambda read, write: applied.append((read, write)),
    )

    def reduce_to_common_policy(policy, op):
        assert op == torch.distributed.ReduceOp.MIN
        policy.copy_(torch.tensor([0, 1], dtype=torch.int))

    cache._all_reduce_attn_groups = reduce_to_common_policy
    cache._sync_dfs_health_policy()

    assert applied == [(False, True)]


def test_deepseek_v4_device_pool_group_maps_sparse_sidecars():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4LayerItem,
        DeepSeekV4TokenToKVPool,
    )

    def state_pool():
        return SimpleNamespace(
            ring_size=2,
            kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
        )

    kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    kvcache._unified_kv = False
    kvcache.start_layer = 0
    kvcache.end_layer = 3
    kvcache.swa_page_size = 2
    kvcache.swa_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)]
    )
    kvcache.c4_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)]
    )
    kvcache.c4_indexer_kv_pool = SimpleNamespace(
        index_k_with_scale_buffer=[
            torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
        ]
    )
    kvcache.c128_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)]
    )
    kvcache.layer_mapping = [
        DeepSeekV4LayerItem(4, 0),
        DeepSeekV4LayerItem(128, 0),
        DeepSeekV4LayerItem(4, 1),
    ]
    kvcache.compress_state_pools = [state_pool(), None, state_pool()]
    kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

    group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=2,
        params=SimpleNamespace(req_to_token_pool=None),
        components={ComponentType.FULL, ComponentType.SWA},
    )
    assert group.num_layers == 3
    assert set(group.entry_map) == {
        PoolName.SWA,
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    }
    assert group.sources[PoolName.DEEPSEEK_V4_C4] == PoolName.KV
    assert group.sources[PoolName.DEEPSEEK_V4_C4_STATE] == PoolName.SWA
    c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
    pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 2
    assert sizes == [5, 5]
    _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
    assert sizes == [[5]]
    assert offsets == [[5]]
    assert c4_pool.get_prepared_layer_range_meta([0], 1) is None


def test_swa_device_pool_group_maps_mha_buffers_and_layouts():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    def mha_pool(k_dims, v_dims, *, rows_are_pages):
        pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        pool.size = 6
        pool.page_size = 2
        rows = 4 if rows_are_pages else 8
        shape = lambda dim: (rows, 2, dim) if rows_are_pages else (rows, dim)
        pool.k_buffer = [torch.zeros(shape(dim), dtype=torch.uint8) for dim in k_dims]
        pool.v_buffer = [torch.zeros(shape(dim), dtype=torch.uint8) for dim in v_dims]
        pool.k_scale_buffer = None
        pool.v_scale_buffer = None
        return pool

    kvcache = SWAKVPool.__new__(SWAKVPool)
    kvcache.page_size = 2
    kvcache.full_kv_pool = mha_pool([3, 5], [7, 11], rows_are_pages=False)
    kvcache.swa_kv_pool = mha_pool([3, 5, 7], [11, 13, 17], rows_are_pages=True)
    kvcache.layers_mapping = {
        0: (0, True),
        1: (0, False),
        2: (1, True),
        3: (1, False),
        4: (2, True),
    }

    group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=2,
        params=SimpleNamespace(
            req_to_token_pool=None,
            mtp_draft_device_pools=(),
        ),
        components={ComponentType.FULL, ComponentType.SWA},
    )

    assert group.num_layers == 5
    assert not group.is_mla_backend
    assert set(group.entry_map) == {PoolName.KV, PoolName.SWA}
    assert group.sources == {
        PoolName.KV: PoolName.KV,
        PoolName.SWA: PoolName.SWA,
    }

    full_pool = group.entry_map[PoolName.KV]
    assert full_pool.packed
    assert full_pool.layer_mapping == {1: 0, 3: 1}
    pointers, sizes = full_pool.get_page_buffer_meta(torch.tensor([0, 1]))
    full_buffers = (
        kvcache.full_kv_pool.k_buffer + kvcache.full_kv_pool.v_buffer
    )
    assert pointers == [buffer[0].data_ptr() for buffer in full_buffers]
    assert sizes == [6, 10, 14, 22]

    swa_pool = group.entry_map[PoolName.SWA]
    assert not swa_pool.packed
    assert hasattr(swa_pool, "v_buffer")
    assert swa_pool.layer_mapping == {0: 0, 2: 1, 4: 2}
    locations = swa_pool.prepare_locations(torch.tensor([2, 3]))
    assert locations == [1]
    pointers, sizes = swa_pool.get_page_buffer_meta(torch.tensor([2, 3]))
    swa_buffers = kvcache.swa_kv_pool.k_buffer + kvcache.swa_kv_pool.v_buffer
    assert pointers == [buffer[1].data_ptr() for buffer in swa_buffers]
    assert sizes == [6, 10, 14, 22, 26, 34]

    pointers, sizes, offsets = swa_pool.get_prepared_layer_range_meta(locations, 2)
    assert pointers == [
        [kvcache.swa_kv_pool.k_buffer[1][1].data_ptr()],
        [kvcache.swa_kv_pool.v_buffer[1][1].data_ptr()],
    ]
    assert sizes == [[10], [26]]
    assert offsets == [[6], [22]]
    assert swa_pool.get_prepared_layer_range_meta(locations, 3) is None

    store = MooncakeStore.__new__(MooncakeStore)
    store.registered_pools = group.entry_map
    store.is_mla_backend = group.is_mla_backend
    store.mha_suffix = "rank"
    store.mla_suffix = "rank"
    keys, multiplier = store._get_hybrid_page_component_keys(
        ["page"], PoolTransfer(name=PoolName.SWA)
    )
    assert keys == [
        f"page_rank_{PoolName.SWA}_k",
        f"page_rank_{PoolName.SWA}_v",
    ]
    assert multiplier == 2


def test_swa_device_pool_group_rejects_mtp_draft_pools():
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    kvcache = SWAKVPool.__new__(SWAKVPool)
    with pytest.raises(ValueError, match="does not yet support MTP"):
        resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=1,
            params=SimpleNamespace(
                req_to_token_pool=None,
                mtp_draft_device_pools=(object(),),
            ),
            components={ComponentType.FULL, ComponentType.SWA},
        )


def test_mamba_strategy_rejects_direct_linker():
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
    kvcache.use_mla = False
    kvcache.full_attention_layer_id_mapping = {0: 0, 2: 1}
    kvcache.full_kv_pool = SimpleNamespace(
        size=6,
        k_scale_buffer=None,
        k_buffer=[torch.zeros((8, 3)), torch.zeros((8, 5))],
        v_buffer=[torch.zeros((8, 7)), torch.zeros((8, 11))],
    )
    req_pool = SimpleNamespace(
        mamba_ckpt_pool=None,
        mamba_map={1: 0, 3: 1},
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.zeros((2, 5, 2, 3)),
                conv=[torch.zeros((2, 5, 4))],
            )
        ),
        translate_mamba_indices=lambda indices: indices,
    )

    with pytest.raises(ValueError, match="does not support the direct external linker"):
        resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(req_to_token_pool=req_pool),
            components={ComponentType.FULL, ComponentType.MAMBA},
        )


def test_dsa_device_pool_group_uses_assembler_strategy():
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    kvcache = DSATokenToKVPool.__new__(DSATokenToKVPool)
    kvcache.page_size = 2
    kvcache.layer_num = 2
    kvcache.kv_buffer = [
        torch.zeros((8, 3), dtype=torch.uint8),
        torch.zeros((8, 5), dtype=torch.uint8),
    ]
    kvcache.index_key_cache = SimpleNamespace(
        buffer=[
            torch.zeros((4, 7), dtype=torch.uint8),
            torch.zeros((4, 11), dtype=torch.uint8),
        ]
    )

    group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=2,
        params=SimpleNamespace(req_to_token_pool=None),
        components={ComponentType.FULL},
    )

    assert group.num_layers == 2
    assert set(group.entry_map) == {PoolName.KV, PoolName.INDEXER}
    assert group.sources == {
        PoolName.KV: PoolName.KV,
        PoolName.INDEXER: PoolName.KV,
    }


def test_device_pool_group_allows_partial_side_pool_load():
    swa_pool = SimpleNamespace(
        name=PoolName.SWA,
        indices_from_pool=PoolName.SWA,
        translate_indices=lambda indices: indices + 100,
    )
    group = DevicePoolGroup([swa_pool], num_layers=1, page_size=2)
    transfer = PoolTransfer(
        name=PoolName.SWA,
        keys=["b", "d"],
        device_indices=torch.tensor([20, 21, 24, 25]),
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )

    assert group.resolve_transfers([transfer]) == []
    [resolved] = group.resolve_transfers(
        [transfer], allow_partial=True, allow_missing_kv=True
    )
    assert resolved.name == PoolName.SWA
    assert resolved.keys == ["b", "d"]
    assert resolved.host_indices.tolist() == [120, 121, 124, 125]


def test_swa_linker_load_prepare_commit_and_abort():
    swa_allocator = _Allocator()
    allocator = SimpleNamespace(
        swa_attn_allocator=swa_allocator,
        set_full_to_swa_mapping=swa_allocator.set_full_to_swa_mapping,
    )
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(
        page_size=1,
        token_to_kv_pool_allocator=allocator,
    )
    component.sliding_window_size = 2
    req = SimpleNamespace(kv=SimpleNamespace(swa_evicted_seqlen=0))
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1, 2, 3, 4]))
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([20, 21]))

    component.update_external_linker_load(
        ExternalLinkerLoadPhase.PREPARE, req, full, swa, prefix_len=4
    )
    mapped_full, mapped_swa = swa_allocator.mapping[0]
    assert mapped_full.tolist() == [3, 4]
    assert mapped_swa.tolist() == [20, 21]
    assert req.kv.swa_evicted_seqlen == 2

    insert_result = InsertResult(
        prefix_len=2,
        adopted_ranges={ComponentType.SWA: [(2, 4)]},
    )
    canonical = torch.tensor([10, 11, 12, 13])
    committed = component.update_external_linker_load(
        ExternalLinkerLoadPhase.COMMIT,
        req,
        full,
        swa,
        prefix_len=4,
        insert_result=insert_result,
        canonical_full=canonical[2:],
    )
    assert committed is swa
    mapped_full, mapped_swa = swa_allocator.mapping[1]
    assert mapped_full.tolist() == [12, 13]
    assert mapped_swa.tolist() == [20, 21]

    aborted = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([30, 31]))
    component.update_external_linker_load(
        ExternalLinkerLoadPhase.ABORT, req, full, aborted, prefix_len=4
    )
    assert swa_allocator.freed[0].tolist() == [30, 31]


def test_mamba_component_rejects_external_linker():
    component = MambaComponent.__new__(MambaComponent)
    with pytest.raises(AssertionError, match="does not support external linker mode"):
        component.build_external_linker_transfer(
            LinkerTransferPhase.LOAD, None, ["a", "b"]
        )


def test_component_commit_filters_overlapping_full_and_swa_load_pages():
    mapping = _Allocator()
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(
            set_full_to_swa_mapping=mapping.set_full_to_swa_mapping
        ),
    )
    full_component = FullComponent.__new__(FullComponent)
    full_component.cache = wrapper.cache
    full_component.component_type = ComponentType.FULL
    swa_component = SWAComponent.__new__(SWAComponent)
    swa_component.cache = wrapper.cache
    swa_component.component_type = ComponentType.SWA
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
    )
    canonical_tail = torch.tensor([10, 11, 102, 103, 14, 15, 106, 107])
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([200, 201, 202, 203, 204, 205, 206, 207]),
    )
    insert_result = InsertResult(
        prefix_len=0,
        adopted_ranges={
            ComponentType.FULL: [(2, 4), (6, 8)],
            ComponentType.SWA: [(2, 4), (6, 8)],
        },
    )
    filtered = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(),
        [(full_component, full), (swa_component, swa)],
        prefix_len=8,
        insert_result=insert_result,
        canonical_full=canonical_tail,
    )
    assert filtered == [full, swa]
    assert full.keys == ["b", "d"]
    assert full.device_indices.tolist() == [102, 103, 106, 107]
    assert swa.keys == ["b", "d"]
    assert swa.device_indices.tolist() == [202, 203, 206, 207]
    mapped_full, mapped_swa = mapping.mapping[0]
    assert mapped_full.tolist() == [102, 103, 106, 107]
    assert mapped_swa.tolist() == [202, 203, 206, 207]
