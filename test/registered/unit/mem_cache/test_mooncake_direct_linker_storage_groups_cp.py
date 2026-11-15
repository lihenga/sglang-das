"""Eight real Gloo ranks, production queues/collectives, byte-addressed fault I/O."""

import datetime
import json
import tempfile
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_mooncake_direct_linker_storage_groups import (
    construct_linker,
    expanded,
    make_linker,
    make_wrapper,
    node,
    transfer,
)

from sglang.srt.mem_cache.storage.mooncake_store import (
    mooncake_direct_linker as direct_module,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _worker(rank, world, rendezvous, out, storage_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        # Constructor must reject differing configuration on EVERY rank.
        patch = pytest.MonkeyPatch()
        rejected = False
        try:
            construct_linker(
                patch, 64 if rank == world - 1 else 128, cp_group=dist.group.WORLD
            )
        except ValueError:
            rejected = True
        finally:
            patch.undo()
        assert rejected

        if storage_size == 256:
            root = node(0, [], [])
            a = node(1, ["a"], [7], root)
            b = node(2, ["b"], [2], a)
            c = node(3, ["c"], [5], b)
            child = node(4, ["d"], [3], c)
            probe, queued, locks, _ = make_wrapper(
                {i: value for i, value in enumerate([root, a, b, c, child])}
            )
            probe.cache_linker.storage_page_size = 256
            probe.cache._all_reduce_attn_groups = lambda tensor, op: dist.all_reduce(
                tensor, op=op
            )
            if rank == 6:
                b.component_data[next(iter(b.component_data))].value = None
            # A physical ancestor miss is a common skip, not a logical fork.
            probe._offload_node(4)
            assert queued == locks == []
            if rank == 7:
                a.hash_value = ["different-ancestor"]
            mismatch = False
            try:
                probe._offload_node(4)
            except RuntimeError:
                mismatch = True
            assert mismatch

        # Bad or differing environment values must also fail on every rank,
        # before clients start I/O or branch into mode-specific collectives.
        for bad in ["256", "invalid"]:
            patch = pytest.MonkeyPatch()
            rejected = False
            try:
                patch.setenv(
                    "SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE",
                    bad if rank == world - 1 else "128",
                )
                construct_linker(patch, cp_group=dist.group.WORLD)
            except ValueError:
                rejected = True
            finally:
                patch.undo()
            assert rejected

        pages = storage_size // 64
        linker = make_linker(storage_page_size=storage_size)
        linker._load_revalidation_groups = (dist.group.WORLD,)
        linker.gc_frozen = True
        linker.completed_loads = Queue()
        direct_module.device_module = SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None)
        )
        root = node(0, [], [])
        hashes = list("abcd")[:pages]
        physical = [7, 2, 4, 8][:pages]
        child = node(1, hashes, physical, root)
        wrapper, _, locks, unlocks = make_wrapper({0: root, 1: child})
        wrapper.cache_linker = linker

        def reduce(tensor, op):
            assert threading.current_thread() is threading.main_thread()
            dist.all_reduce(tensor, op=op)

        wrapper.cache._all_reduce_attn_groups = reduce
        store = linker.storage.store
        store.delay = threading.Event() if rank == 3 else None
        if rank == 5:
            store.fail_put.add("indexer")
        thread = threading.Thread(target=linker.offload_thread_func)
        thread.start()
        try:
            wrapper._offload_node(1)
            assert store.started.wait(10)
            count = torch.tensor(wrapper.num_completed_offloads())
            dist.all_reduce(count, op=dist.ReduceOp.MIN)
            assert count.item() == 0  # rank 3 is still writing
            assert locks == [1] and unlocks == [] and not child.external_cache_stored
            if store.delay:
                store.delay.set()
            linker.offload_queue.join()
            count = torch.tensor(wrapper.num_completed_offloads())
            dist.all_reduce(count, op=dist.ReduceOp.MIN)
            assert count.item() == 1
            wrapper.drain_offloads(1)
            assert not child.external_cache_stored and unlocks == [(1, 1)]

            lookup = transfer(hashes, physical, pages_per_object=pages)
            local = linker.lookup("r", [lookup])
            common = wrapper._sync_restorable_prefix(
                local, num_pages=pages, device_hit_pages=0
            )
            assert common == 0  # rank 5 has no indexer object
            assert not linker.revalidate_load([lookup])

            store.fail_put.clear()
            store.delay = None
            wrapper._offload_node(1)
            linker.offload_queue.join()
            count = torch.tensor(wrapper.num_completed_offloads())
            dist.all_reduce(count, op=dist.ReduceOp.MIN)
            assert count.item() == 1
            wrapper.drain_offloads(1)
            assert child.external_cache_stored and unlocks == [(1, 1), (1, 1)]
            common = wrapper._sync_restorable_prefix(
                linker.lookup("r", [lookup]), num_pages=pages, device_hit_pages=0
            )
            assert common == pages
            assert linker.revalidate_load([lookup])

            # Local GET success cannot publish a successful global restore.
            store.fail_get = rank == 6
            index = linker.layer_done_counter.update_producer()
            success = linker._load_grouped_layer_wise(index, [expanded(linker, lookup)])
            linker.completed_loads.put((["r"], success))
            wrapper.pending_loads = {"r": (1, 1, 0)}
            wrapper.taken_loads = []
            detached = []
            wrapper._detach_failed_chain = lambda *args: detached.append(args)
            wrapper.cache.linker = wrapper
            wrapper.cache._failed_linker_rids = []
            UnifiedRadixCache._collect_failed_linker_loads(wrapper.cache, 1)
            assert wrapper.cache._failed_linker_rids == ["r"]
            assert detached == [("r", 1, 0)] and not store.sessions
            assert not wrapper.pending_loads

            mismatch = False
            try:
                wrapper._agree_storage_plan("deliberate-mismatch", [rank == 7])
            except RuntimeError:
                mismatch = True
            assert mismatch
            dist.barrier()
            Path(out, f"rank-{rank}.json").write_text(
                json.dumps(
                    dict(
                        rank=rank,
                        configuration_rejected=True,
                        slow_put_pinned=True,
                        single_rank_put_failure_common=True,
                        retry_converged=True,
                        single_rank_get_failure_common=True,
                        task_mismatch_rejected=True,
                        collectives_in_io_workers=False,
                    )
                )
            )
        finally:
            if store.delay:
                store.delay.set()
            linker.offload_queue.put(None)
            thread.join(15)
            assert not thread.is_alive()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("storage_size", [128, 256])
def test_storage_groups_cp8_faults(monkeypatch, storage_size):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    with tempfile.TemporaryDirectory(prefix="l3-grouped-cp8-") as folder:
        mp.start_processes(
            _worker,
            args=(8, folder + "/rendezvous", folder, storage_size),
            nprocs=8,
            join=True,
            start_method="fork",
        )
        rows = [
            json.loads(path.read_text())
            for path in sorted(Path(folder).glob("rank-*.json"))
        ]
        assert len(rows) == 8
        assert all(
            all(
                value is True
                for key, value in row.items()
                if key not in ("rank", "collectives_in_io_workers")
            )
            for row in rows
        )
        assert all(row["collectives_in_io_workers"] is False for row in rows)
