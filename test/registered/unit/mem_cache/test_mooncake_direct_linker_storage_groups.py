import ctypes
import threading
import time
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.linker_storage_groups import (
    StoragePageRef,
    group_storage_refs,
    make_storage_refs,
)
from sglang.srt.mem_cache.storage.mooncake_store import (
    mooncake_direct_linker as direct_module,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
    ExternalLinkerLoadPhase,
)
from sglang.srt.mem_cache.unified_cache_linker import (
    DevicePoolEntry,
    DevicePoolGroup,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class MemoryStore:
    """Byte-addressed transport: production pointer/offset metadata is exercised."""

    def __init__(self):
        self.objects = {}
        self.sessions = set()
        self.put_calls = []
        self.get_calls = []
        self.fail_put = set()
        self.fail_get = False
        self.short_put = False
        self.delay = None
        self.started = threading.Event()
        self.exists_calls = []

    def batch_is_exist(self, keys):
        self.exists_calls.append(list(keys))
        return [int(key in self.objects) for key in keys]

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, config=None):
        self.started.set()
        if self.delay:
            assert self.delay.wait(10)
        self.put_calls.append(list(keys))
        result = []
        for key, pointers, lengths in zip(keys, ptrs, sizes):
            if any(suffix in key for suffix in self.fail_put):
                result.append(-1)
                continue
            self.objects[key] = b"".join(
                ctypes.string_at(p, n) for p, n in zip(pointers, lengths)
            )
            result.append(0)
        return result[:-1] if self.short_put else result

    def batch_get_session_start(self, keys):
        result = []
        for key in keys:
            exists = key in self.objects
            if exists:
                self.sessions.add(key)
            result.append(0 if exists else -1)
        return result

    def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        self.get_calls.append((list(keys), sizes, offsets))
        if self.fail_get:
            return [-1] * len(keys)
        result = []
        for key, pointers, lengths, starts in zip(keys, ptrs, sizes, offsets):
            assert key in self.sessions
            blob = self.objects[key]
            total = 0
            for pointer, size, offset in zip(pointers, lengths, starts):
                value = blob[offset : offset + size]
                if len(value) != size:
                    total = -1
                    break
                ctypes.memmove(pointer, value, size)
                total += size
            result.append(total)
        return result

    def batch_get_session_end(self, keys):
        for key in keys:
            self.sessions.remove(key)


def make_linker(device="cpu", *, aligned=False, storage_page_size=128):
    # Variable layer widths catch page-major vs layer-major offset mistakes.
    kv = [
        torch.arange(64 * 12 * w, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(64 * 12, w)
        for w in (2, 5, 3)
    ]
    idx = [
        torch.arange(12 * w, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(12, w)
        for w in (23, 47)
    ]
    kv = [buffer.to(device) for buffer in kv]
    idx = [buffer.to(device) for buffer in idx]
    if aligned:

        def align(buffer):
            alignment = 2 * 1024**2
            backing = torch.empty(
                2 * alignment + buffer.nbytes, dtype=torch.uint8, device=device
            )
            offset = (-backing.data_ptr()) % alignment
            view = backing[offset : offset + buffer.nbytes].view_as(buffer)
            view.copy_(buffer)
            return view

        kv = [align(buffer) for buffer in kv]
        idx = [align(buffer) for buffer in idx]
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[kv],
            layer_mapping={0: 0, 1: 1, 3: 2},
            page_size=64,
            rows_are_pages=False,
        ),
        DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[idx],
            layer_mapping={0: 0, 2: 1},
            page_size=64,
            rows_are_pages=True,
        ),
    ]
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.grouped_storage = True
    linker.storage_page_size = storage_page_size
    linker.page_size = 64
    linker.tp_rank = 0
    linker.host_prefetch_enabled = False
    linker.read_plan_enabled = False
    linker.enable_page_wise_load = False
    linker.num_layers = 4
    linker.pool_group = DevicePoolGroup(entries, num_layers=4, page_size=64)
    linker.pools = linker.pool_group.entry_map
    linker.layer_done_counter = LayerWiseLoadCounter(4)
    linker.stats = {"lookup": 0, "offload": 0, "load": 0}
    linker.storage = MooncakeStore.__new__(MooncakeStore)
    linker.storage.store = MemoryStore()
    linker.storage.registered_pools = linker.pools
    linker.storage.is_mla_backend = linker.pool_group.is_mla_backend
    linker.storage.pp_rank = 0
    linker.storage.pp_size = 1
    linker.storage.mla_suffix = "tp0_cp0_pp0"
    linker.storage.mha_suffix = "tp0_cp0_pp0"
    linker.storage.config_prefix = "test_l3p128v1"
    linker.storage._use_group_semantics = False
    linker.storage._replicate_config_cls = SimpleNamespace
    linker.offload_queue = Queue()
    linker.offload_results = Queue()
    return linker


def construct_linker(
    monkeypatch, size=None, *, pp_size=1, page=64, cp_group=None, host_prefetch=False
):
    original = make_linker()
    kvcache = SimpleNamespace(layer_num=4, index_k_with_scale_buffer=[], kv_buffer=[])
    params = SimpleNamespace(
        page_size=page,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: kvcache),
        tp_cache_group=None,
        pp_rank=0,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_cp_cache_group=cp_group,
        attn_tp_cache_group=None,
        mtp_draft_device_pools=(),
        enable_metrics=False,
    )
    extra = {} if size is None else {"direct_storage_page_size": size}
    monkeypatch.setattr(
        direct_module.HybridCacheController,
        "parse_storage_backend_extra_config",
        lambda value: (extra,),
    )
    monkeypatch.setattr(
        direct_module,
        "resolve_hybrid_device_pool_group",
        lambda **kwargs: original.pool_group,
    )
    monkeypatch.setattr(MooncakeDirectLinker, "register_buffers", lambda self: None)
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    args = SimpleNamespace(
        tp_size=1,
        model_path="model",
        hicache_storage_backend_extra_config=None,
        mooncake_page_wise_load_threshold=128,
        mooncake_enable_page_wise_load=False,
        mooncake_enable_waiting_queue_dfs_prefetch=host_prefetch,
        schedule_policy="fcfs",
        disaggregation_mode="prefill",
    )
    original.storage.config_prefix = "tenant_model"
    return MooncakeDirectLinker(
        args, params, components={ComponentType.FULL}, storage=original.storage
    )


def test_opt_in_namespace_and_legacy_default(monkeypatch):
    legacy = construct_linker(monkeypatch)
    assert not legacy.grouped_storage
    assert legacy.storage_page_size == 64
    assert legacy.storage.config_prefix == "tenant_model"
    grouped = construct_linker(monkeypatch, 128)
    assert grouped.grouped_storage and grouped.page_size == 64
    assert grouped.storage.config_prefix.startswith("tenant_model_l3p128v1_")


@pytest.mark.parametrize("size", [True, "128", 0, 65, 512])
def test_invalid_storage_size_rejected(monkeypatch, size):
    with pytest.raises(ValueError, match="direct_storage_page_size"):
        construct_linker(monkeypatch, size)


def test_unsupported_pp_rejected(monkeypatch):
    with pytest.raises(ValueError, match="PP1"):
        construct_linker(monkeypatch, 128, pp_size=2)


@pytest.mark.parametrize(
    "flag", ["SGLANG_MOONCAKE_READ_PLAN", "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES"]
)
def test_grouped_read_plan_rejected_before_io(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match="ReadPlan disabled"):
        construct_linker(monkeypatch, 128)


def test_grouped_waiting_queue_prefetch_rejected_before_io(monkeypatch):
    with pytest.raises(ValueError, match="DFS prefetch"):
        construct_linker(monkeypatch, 128, host_prefetch=True)


def transfer(keys, physical_pages, start=0, pages_per_object=2):
    refs = make_storage_refs(keys, start, pages_per_object)
    indices = torch.cat([torch.arange(p * 64, (p + 1) * 64) for p in physical_pages])
    return PoolTransfer(
        name=PoolName.KV,
        keys=list(keys[: len(refs)]),
        storage_refs=refs,
        device_indices=indices[: len(refs) * 64],
    )


def expanded(linker, t):
    return linker.pool_group.resolve_transfers([t])


@pytest.mark.parametrize(
    "start,count,halves",
    [(0, 1, []), (0, 3, [0, 1]), (1, 1, [1]), (1, 3, [1, 0, 1]), (2, 4, [0, 1, 0, 1])],
)
def test_sequence_aligned_refs(start, count, halves):
    refs = make_storage_refs([str(i) for i in range(count)], start)
    assert [ref.half for ref in refs] == halves
    if start == 0 and count >= 2:
        assert refs[0].key == refs[1].key == "1"


def test_complete_writes_reject_half_and_duplicate():
    with pytest.raises(ValueError, match="complete"):
        group_storage_refs([StoragePageRef("a", 1)], complete=True)
    with pytest.raises(ValueError, match="Duplicate"):
        group_storage_refs([StoragePageRef("a", 0), StoragePageRef("a", 0)])


def test_noncontiguous_gpu_pages_and_exact_half_reload():
    linker = make_linker()
    store = linker.storage.store
    write = transfer(["h0", "h1"], [7, 2])
    assert linker._offload_grouped(expanded(linker, write))
    assert len(store.objects) == 2  # one KV and one indexer, not four page keys
    assert sorted(map(len, store.objects.values())) == [140, 1280]
    expected = []
    for pool in linker.pools.values():
        row = 2 if pool.name == PoolName.INDEXER else 2 * 64
        count = 1 if pool.name == PoolName.INDEXER else 64
        expected.append([b[row : row + count].clone() for b in pool.kv_buffer])
    read = transfer(["h1"], [9], start=1)
    for pool in linker.pools.values():
        row = 9 if pool.name == PoolName.INDEXER else 9 * 64
        count = 1 if pool.name == PoolName.INDEXER else 64
        for b in pool.kv_buffer:
            b[row : row + count].zero_()
    assert linker._lookup_grouped(expanded(linker, read)) == [1]
    counter = linker.layer_done_counter.update_producer()
    assert linker._load_grouped_layer_wise(counter, [expanded(linker, read)])
    assert not store.sessions
    for pool, gold in zip(linker.pools.values(), expected):
        row = 9 if pool.name == PoolName.INDEXER else 9 * 64
        count = 1 if pool.name == PoolName.INDEXER else 64
        assert all(
            torch.equal(b[row : row + count], g) for b, g in zip(pool.kv_buffer, gold)
        )
    # KV second-half layer 1 starts after all three layers of page 0 plus layer 0.
    assert any(offsets == [[640 + 128]] for _, _, offsets in store.get_calls)


def test_shared_prefix_load_deduplicates_sessions_but_writes_both_destinations():
    linker = make_linker()
    assert linker._offload_grouped(expanded(linker, transfer(["a", "b"], [0, 3])))
    a = expanded(linker, transfer(["a", "b"], [5, 6]))
    b = expanded(linker, transfer(["b"], [10], start=1))
    counter = linker.layer_done_counter.update_producer()
    assert linker._load_grouped_layer_wise(counter, [a, b])
    for pool in linker.pools.values():
        span = 1 if pool.name == PoolName.INDEXER else 64
        for buffer in pool.kv_buffer:
            assert torch.equal(
                buffer[6 * span : 7 * span], buffer[10 * span : 11 * span]
            )
    assert not linker.storage.store.sessions


def test_pool_failure_partial_put_and_retry():
    linker = make_linker()
    store = linker.storage.store
    t = expanded(linker, transfer(["a", "b", "c", "d"], [0, 3, 1, 4]))
    store.fail_put.add("indexer")
    assert not linker._offload_grouped(t)
    assert linker._lookup_grouped(t) == []
    assert len(store.objects) == 2
    store.fail_put.clear()
    assert linker._offload_grouped(t)
    assert linker._lookup_grouped(t) == [2, 4]
    assert len(store.objects) == 4
    calls = len(store.put_calls)
    assert linker._offload_grouped(t)
    assert len(store.put_calls) == calls


def test_lookup_never_advertises_a_hole_or_first_half_boundary():
    linker = make_linker()
    t = expanded(linker, transfer(["a", "b", "c", "d"], [0, 1, 2, 3]))
    assert linker._offload_grouped(t)
    del linker.storage.store.objects[
        next(k for k in linker.storage.store.objects if "_b_" in k and "indexer" in k)
    ]
    assert linker._lookup_grouped(t) == []
    odd = expanded(linker, transfer(["d"], [4], start=3))
    assert linker._lookup_grouped(odd) == [1]


@pytest.mark.parametrize("failure", ["range", "evicted", "truncated"])
def test_load_failure_closes_sessions_and_reports_failure(failure):
    linker = make_linker()
    t = expanded(linker, transfer(["a", "b", "c", "d"], [0, 1, 2, 3]))
    assert linker._offload_grouped(t)
    if failure == "range":
        linker.storage.store.fail_get = True
    elif failure == "evicted":
        del linker.storage.store.objects[
            next(k for k in linker.storage.store.objects if "_d_" in k)
        ]
    else:
        key = next(iter(linker.storage.store.objects))
        linker.storage.store.objects[key] = b"x"
    counter = linker.layer_done_counter.update_producer()
    assert not linker._load_grouped_layer_wise(counter, [t])
    assert not linker.storage.store.sessions
    assert all(f.done() for f in linker.layer_done_counter.futures[counter])


def test_short_put_result_is_not_success():
    linker = make_linker()
    linker.storage.store.short_put = True
    with pytest.raises(RuntimeError, match="short"):
        linker._offload_grouped(expanded(linker, transfer(["a", "b"], [0, 1])))


def test_metadata_rpc_failure_is_a_local_miss(monkeypatch):
    linker = make_linker()
    monkeypatch.setattr(
        linker.storage,
        "_batch_exist",
        lambda keys: (_ for _ in ()).throw(RuntimeError("RPC failed")),
    )
    assert linker.lookup("r", [transfer(["a", "b"], [0, 1])]) == []


def test_short_session_start_closes_known_successful_sessions(monkeypatch):
    linker = make_linker()
    t = expanded(linker, transfer(["a", "b", "c", "d"], [0, 1, 2, 3]))
    assert linker._offload_grouped(t)
    store = linker.storage.store
    original = store.batch_get_session_start
    monkeypatch.setattr(
        store, "batch_get_session_start", lambda keys: original(keys[:1])
    )
    counter = linker.layer_done_counter.update_producer()
    assert not linker._load_grouped_layer_wise(counter, [t])
    assert not store.sessions


def test_slow_put_has_no_completion_until_done():
    linker = make_linker()
    store = linker.storage.store
    store.delay = threading.Event()
    t = expanded(linker, transfer(["a", "b"], [0, 1]))
    linker.offload_queue.put(
        (t, 128, time.perf_counter(), SimpleNamespace(synchronize=lambda: None))
    )
    worker = threading.Thread(target=linker.offload_thread_func)
    worker.start()
    try:
        assert store.started.wait(5)
        assert linker.num_completed_offloads() == 0
        store.delay.set()
        linker.offload_queue.join()
        assert linker.num_completed_offloads() == 1
        assert linker.pop_completed_offload()
    finally:
        store.delay.set()
        linker.offload_queue.put(None)
        worker.join(10)
        assert not worker.is_alive()


def make_wrapper(nodes):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache_linker = SimpleNamespace(
        grouped_storage=True, offload=lambda t: queued.append(t) or True
    )
    wrapper.pending_offloads = []
    wrapper.grouped_offloads = {}
    wrapper.storage_plan_sequence = 0
    wrapper._waiting_queue_prefetch_enabled = False
    queued, locks, unlocks = [], [], []
    component = FullComponent.__new__(FullComponent)
    component.component_type = ComponentType.FULL
    wrapper.cache = SimpleNamespace(
        page_size=64,
        pp_size=1,
        _components_tuple=(component,),
        resolve_node_handle=lambda node_id: nodes[node_id],
        inc_lock_ref=lambda node_id: (
            locks.append(node_id) or SimpleNamespace(to_dec_params=lambda: node_id)
        ),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
        _all_reduce_attn_groups=lambda tensor, op: None,
    )
    component.cache = wrapper.cache
    return wrapper, queued, locks, unlocks


def node(node_id, keys, pages, parent=None):
    indices = (
        torch.cat([torch.arange(p * 64, (p + 1) * 64) for p in pages])
        if pages
        else torch.empty(0, dtype=torch.int64)
    )
    return SimpleNamespace(
        id=node_id,
        key=list(range(64 * len(keys))),
        hash_value=keys,
        parent=parent,
        external_cache_stored=False,
        component_data={ComponentType.FULL: SimpleNamespace(value=indices)},
    )


def test_pair_spanning_radix_nodes_and_tail_not_published():
    root = node(0, [], [])
    parent = node(1, ["a"], [7], root)
    child = node(2, ["b", "c"], [2, 5], parent)
    parent.external_cache_stored = True  # Must still gather its first half.
    wrapper, queued, locks, unlocks = make_wrapper({0: root, 1: parent, 2: child})
    wrapper._offload_node(2)
    t = queued[0][0]
    assert t.keys == ["a", "b"]
    assert t.device_indices.tolist() == list(range(7 * 64, 8 * 64)) + list(
        range(2 * 64, 3 * 64)
    )
    assert locks == [2] and not unlocks
    assert not child.external_cache_stored
    wrapper._offload_node(2)
    assert len(queued) == 1
    wrapper.cache_linker.pop_completed_offload = lambda: True
    wrapper.drain_offloads(1)
    assert not child.external_cache_stored  # page c was deliberately omitted
    assert unlocks == [(2, 2)]


def test_single_tail_neither_queues_nor_pins():
    root = node(0, [], [])
    child = node(1, ["a"], [7], root)
    wrapper, queued, locks, unlocks = make_wrapper({0: root, 1: child})
    wrapper._offload_node(1)
    assert queued == locks == unlocks == []


def test_completed_pair_published_only_after_common_success():
    root = node(0, [], [])
    child = node(1, ["a", "b"], [7, 2], root)
    wrapper, queued, locks, unlocks = make_wrapper({0: root, 1: child})
    wrapper._offload_node(1)
    assert not child.external_cache_stored
    wrapper.cache_linker.pop_completed_offload = lambda: True
    # Another rank failed: simulate the common MIN, despite local success.
    wrapper.cache._all_reduce_attn_groups = lambda tensor, op: tensor.zero_()
    wrapper.drain_offloads(1)
    assert not child.external_cache_stored
    assert unlocks == [(1, 1)]


def test_adoption_filter_preserves_second_half_source_offset():
    wrapper, _, _, _ = make_wrapper({})
    component = wrapper.cache._components_tuple[0]
    t = transfer(["a", "b", "c", "d"], [0, 1, 2, 3])
    result = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(),
        [(component, t)],
        256,
        insert_result=SimpleNamespace(
            adopted_ranges={ComponentType.FULL: [(64, 128), (192, 256)]}
        ),
        canonical_full=t.device_indices,
    )
    assert result[0].keys == ["b", "d"]
    assert result[0].storage_refs == [StoragePageRef("b", 1), StoragePageRef("d", 1)]


def test_reset_releases_pending_grouped_ancestry_locks():
    root = node(0, [], [])
    child = node(1, ["a", "b"], [7, 2], root)
    wrapper, _, _, unlocks = make_wrapper({0: root, 1: child})
    wrapper._offload_node(1)
    wrapper._release_grouped_offload_locks()
    assert unlocks == [(1, 1)]
    assert not wrapper.pending_offloads and not wrapper.grouped_offloads


def test_cold_prefix_probes_only_first_object_per_pool():
    linker = make_linker()
    keys = [str(index) for index in range(16384)]
    t = PoolTransfer(
        name=PoolName.KV, keys=keys, storage_refs=make_storage_refs(keys, 0)
    )
    assert linker.lookup("cold-1m", [t]) == []
    assert [len(keys) for keys in linker.storage.store.exists_calls] == [1, 1]


def test_revalidation_checks_group_keys_and_second_half_only():
    linker = make_linker()
    full = transfer(["a", "b"], [7, 2])
    assert linker._offload_grouped(expanded(linker, full))
    assert linker.revalidate_load([full])
    # The first 64-token hash has no stored object of its own. A GPU hit on
    # that page still allows the second half to restore from the paired key.
    half = transfer(["b"], [9], start=1)
    assert linker.revalidate_load([half])
    key = next(key for key in linker.storage.store.objects if "indexer" in key)
    del linker.storage.store.objects[key]
    assert not linker.revalidate_load([half])


def test_real_tree_split_during_put_preserves_and_releases_both_half_locks():
    from array import array

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=64,
    )
    cache = SimpleNamespace(enable_session_radix_cache=False, is_swa_enabled=False)
    component = FullComponent(cache, params)
    tree = UnifiedTreeCore(params, {ComponentType.FULL: component})
    cache.page_size = 64
    cache.tree_core = tree
    cache._components_tuple = (component,)
    cache.resolve_node_handle = tree.node_by_id
    cache.inc_lock_ref = tree.inc_lock_ref
    cache.dec_lock_ref = tree.dec_lock_ref
    cache._all_reduce_attn_groups = lambda tensor, op: None
    child = tree._new_node()
    child.parent = tree.root_node
    child.key = RadixKey(array("q", range(128)))
    child.hash_value = ["a", "b"]
    child.component_data[ComponentType.FULL].value = torch.cat(
        [torch.arange(7 * 64, 8 * 64), torch.arange(2 * 64, 3 * 64)]
    )
    tree.root_node.children[child.key.child_key(64)] = child
    tree.component_evictable_size_[ComponentType.FULL] = 128
    tree._update_evictable_leaf_sets(child)
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = SimpleNamespace(
        grouped_storage=True,
        offload=lambda transfers: True,
        pop_completed_offload=lambda: True,
    )
    wrapper.pending_offloads = []
    wrapper.grouped_offloads = {}
    wrapper.storage_plan_sequence = 0
    wrapper._offload_node(child.id)
    parent, _ = tree._split_node(child.key, child, 64)
    assert parent.component_data[ComponentType.FULL].lock_ref == 1
    assert child.component_data[ComponentType.FULL].lock_ref == 1
    assert tree.component_protected_size_[ComponentType.FULL] == 128
    wrapper.drain_offloads(1)
    assert parent.component_data[ComponentType.FULL].lock_ref == 0
    assert child.component_data[ComponentType.FULL].lock_ref == 0
    assert tree.component_protected_size_[ComponentType.FULL] == 0
    assert tree.component_evictable_size_[ComponentType.FULL] == 128
    assert child.external_cache_stored and not parent.external_cache_stored


@pytest.mark.parametrize("pages_per_object", [2, 4])
def test_match_to_load_uses_only_missing_gpu_half_and_preserves_cached_page(
    pages_per_object,
):
    from array import array

    from sglang.srt.mem_cache.base_prefix_cache import MatchResult
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_hash_str

    size = pages_per_object * 64
    key = RadixKey(array("q", range(size + 1)))
    hashes = get_hash_str(key[:size], page_size=64)
    physical = [7, 2, 5, 3][:pages_per_object]
    linker = make_linker(storage_page_size=size)
    assert linker._offload_grouped(
        expanded(linker, transfer(hashes, physical, pages_per_object=pages_per_object))
    )
    root = node(0, [], [])
    cached = node(1, hashes[:-1], physical[:-1], root)
    loaded = node(2, hashes[-1:], [9], cached)
    wrapper, _, locks, unlocks = make_wrapper({0: root, 1: cached, 2: loaded})
    wrapper.cache_linker = linker
    wrapper.hit_markers = {}
    wrapper.pending_loads = {}
    wrapper.failed_chains = {}
    wrapper.taken_loads = []
    linker.pending_loads = {}
    cache = wrapper.cache
    cache.is_swa_enabled = False
    allocated = []
    dest = torch.arange(9 * 64, 10 * 64)
    cache.token_to_kv_pool_allocator = SimpleNamespace(
        available_size=lambda: 768,
        alloc=lambda size: allocated.append(size) or dest.clone(),
    )
    cache.get_last_hash_value = lambda node_id: hashes[-2]
    result = MatchResult(
        device_indices=cached.component_data[ComponentType.FULL].value,
        last_device_node=1,
        last_host_node=1,
        best_match_node=1,
        cache_actions=[],
    )
    cache.tree_core = SimpleNamespace(
        empty_match_result=result._replace(
            device_indices=torch.empty(0, dtype=torch.int64)
        ),
        collect_full_device_indices=lambda endpoint, anchor: dest,
    )
    cache.insert = lambda params: SimpleNamespace(
        last_device_node=2, adopted_ranges={ComponentType.FULL: [(size - 64, size)]}
    )
    req = SimpleNamespace(
        rid="shared-prefix", last_node=1, prefix_indices=result.device_indices, kv=None
    )
    matched = wrapper.match(key, req, result)
    assert matched.host_hit_length == 64
    tail, endpoint = wrapper.load_back(req)
    assert allocated == [64] and endpoint == 2 and torch.equal(tail, dest)
    transfers = linker.pending_loads.pop(req.rid)
    assert all(
        t.storage_refs
        == [StoragePageRef(hashes[-1], pages_per_object - 1, pages_per_object)]
        for t in transfers
    )
    before = [
        [b[p * 64 : (p + 1) * 64].clone() for p in physical[:-1]]
        for b in linker.pools[PoolName.KV].kv_buffer
    ]
    counter = linker.layer_done_counter.update_producer()
    assert linker._load_grouped_layer_wise(counter, [transfers])
    wrapper.taken_loads = [([req.rid], True)]
    assert wrapper.commit_completed_loads([True]) == []
    assert locks == [2] and unlocks == [(2, 2)]
    for buffer, gold in zip(linker.pools[PoolName.KV].kv_buffer, before):
        assert all(
            torch.equal(buffer[p * 64 : (p + 1) * 64], g)
            for p, g in zip(physical[:-1], gold)
        )
        assert torch.equal(
            buffer[9 * 64 : 10 * 64],
            buffer[physical[-1] * 64 : (physical[-1] + 1) * 64],
        )


@pytest.mark.parametrize("size", [128, 256])
def test_environment_selection_and_json_precedence(monkeypatch, size):
    monkeypatch.setenv("SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE", str(size))
    selected = construct_linker(monkeypatch)
    assert selected.storage_page_size == size and selected.grouped_storage
    assert f"l3p{size}v1_" in selected.storage.config_prefix
    override = construct_linker(monkeypatch, 64)
    assert override.storage_page_size == 64 and not override.grouped_storage


@pytest.mark.parametrize("raw", ["", "bad", "128.0", "true", "0", "512"])
def test_bad_environment_rejected_before_io(monkeypatch, raw):
    monkeypatch.setenv("SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE", raw)
    with pytest.raises(ValueError, match="SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE"):
        construct_linker(monkeypatch)


@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("count", range(1, 10))
def test_quad_refs_only_end_on_absolute_group_boundaries(start, count):
    keys = [str(i) for i in range(count)]
    refs = make_storage_refs(keys, start, 4)
    expected = max(0, (start + count) // 4 * 4 - start)
    assert len(refs) == expected
    assert [r.half for r in refs] == [(start + i) % 4 for i in range(expected)]
    assert all(
        r.key == keys[i + 3 - r.half] and r.pages_per_object == 4
        for i, r in enumerate(refs)
    )


@pytest.mark.parametrize("offset", range(4))
def test_quad_exact_partial_reload_and_duplicate_destinations(offset):
    linker = make_linker(storage_page_size=256)
    source = [7, 2, 5, 3]
    write = expanded(linker, transfer(list("abcd"), source, pages_per_object=4))
    assert linker._offload_grouped(write)
    assert sorted(map(len, linker.storage.store.objects.values())) == [280, 2560]
    read = expanded(
        linker,
        transfer(
            list("abcd")[offset:],
            [8, 9, 10, 11][: 4 - offset],
            start=offset,
            pages_per_object=4,
        ),
    )
    # Include a second request for the last page of the same stored object.
    shared = expanded(linker, transfer(["d"], [6], start=3, pages_per_object=4))
    counter = linker.layer_done_counter.update_producer()
    assert linker._load_grouped_layer_wise(counter, [read, shared])
    assert not linker.storage.store.sessions
    for pool in linker.pools.values():
        span = 1 if pool.name == PoolName.INDEXER else 64
        for buffer in pool.kv_buffer:
            for dest, src in zip([8, 9, 10, 11], source[offset:]):
                assert torch.equal(
                    buffer[dest * span : (dest + 1) * span],
                    buffer[src * span : (src + 1) * span],
                )
            assert torch.equal(buffer[6 * span : 7 * span], buffer[3 * span : 4 * span])


def test_quad_write_gathers_three_split_ancestors_and_omits_tail():
    root = node(0, [], [])
    a = node(1, ["a"], [7], root)
    b = node(2, ["b"], [2], a)
    c = node(3, ["c"], [5], b)
    child = node(4, ["d", "e"], [3, 6], c)
    wrapper, queued, locks, unlocks = make_wrapper(
        {i: n for i, n in enumerate([root, a, b, c, child])}
    )
    wrapper.cache_linker.storage_page_size = 256
    wrapper._offload_node(4)
    t = queued[0][0]
    assert t.keys == list("abcd")
    assert (
        t.device_indices.tolist()
        == torch.cat(
            [torch.arange(p * 64, (p + 1) * 64) for p in [7, 2, 5, 3]]
        ).tolist()
    )
    assert locks == [4] and not unlocks and wrapper.grouped_offloads[4] is False
    wrapper.cache_linker.pop_completed_offload = lambda: True
    wrapper.drain_offloads(1)
    assert not child.external_cache_stored and unlocks == [(4, 4)]


def test_reference_width_mismatch_rejected_before_sdk_io():
    linker = make_linker(storage_page_size=256)
    with pytest.raises(ValueError, match="configured object size"):
        linker._offload_grouped(expanded(linker, transfer(["a", "b"], [0, 1])))
    assert not linker.storage.store.objects
