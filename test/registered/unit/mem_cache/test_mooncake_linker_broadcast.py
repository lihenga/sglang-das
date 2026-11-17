import ctypes
import threading
import unittest
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
    ReadPlanLoadCounter,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeLinkerBroadcast(CustomTestCase):
    def test_source_forward_finishes_reads_before_end_and_retains_cleanup_failure(self):
        for mode in ("layer", "page", "read_plan"):
            for prefetch in (False, True):
                for read_failed, end_rc in ((False, 0), (False, -1), (True, 0)):
                    with self.subTest(
                        mode=mode,
                        prefetch=prefetch,
                        read_failed=read_failed,
                        end_rc=end_rc,
                    ):
                        self._source_read_cleanup_flow(
                            mode, prefetch, read_failed, end_rc
                        )

    def _source_read_cleanup_flow(self, mode, prefetch, read_failed, end_rc):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_enabled = prefetch
        linker.host_prefetch_limit = 8
        linker.host_prefetch_lock = threading.Lock()
        linker.host_prefetch_entries = {}
        linker.host_prefetch_worker_count = 1
        linker._init_session_runtime()
        self.addCleanup(linker.session_executor.shutdown, wait=True)
        entered, release, forwarded, retired = [threading.Event() for _ in range(4)]
        self.addCleanup(release.set)
        buffers = [torch.zeros((1, 2), dtype=torch.uint8) for _ in range(2)]
        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[buffers],
            layer_mapping={0: 0, 1: 1},
            page_size=1,
            rows_are_pages=True,
        )
        sessions, broadcasts = set(), []

        def start(keys):
            sessions.update(keys)
            return [0] * len(keys)

        def end(keys):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("session end was not released")
            if end_rc == 0:
                sessions.difference_update(keys)
            return end_rc

        def read(keys, pointers, sizes, offsets):
            if read_failed:
                return [-1] * len(keys)
            for row_ptrs, row_sizes in zip(pointers, sizes):
                for ptr, size in zip(row_ptrs, row_sizes):
                    ctypes.memset(ptr, 7, size)
            return [sum(row) for row in sizes]

        class NativePlan:
            def __init__(self, layouts, layers, **options):
                self.layers = [Future() for _ in range(layers)]
                self.borrowed = options["borrowed_sessions"]

            def run(self):
                if not self.borrowed:
                    start(["key"])
                ok = True
                for layer in range(2):
                    ptrs, sizes, offsets = pool.get_prepared_layer_range_meta(
                        [0], layer
                    )
                    ok = ok and read(["key"], ptrs, sizes, offsets) == [2]
                    if layer == 0:
                        self.layers[layer].set_result(ok)
                if not self.borrowed:
                    ok = end(["key"]) == 0 and ok
                self.layers[-1].set_result(ok)
                if not ok:
                    raise RuntimeError("native read plan failed")

            def wait(self, layer):
                if not self.layers[layer].result(timeout=5):
                    raise RuntimeError("native read plan failed")

        store = SimpleNamespace(
            batch_get_session_start=start,
            batch_get_session_start_with_sources=lambda keys: (
                start(keys),
                ["dfs"] * len(keys),
            ),
            batch_get_session_end=end,
            batch_get_into_multi_buffer_ranges=read,
            create_read_plan=NativePlan,
        )
        linker.storage = SimpleNamespace(
            store=store,
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=lambda keys: list(keys),
        )
        linker.pools = {PoolName.KV: pool}
        linker.num_layers, linker.tp_rank, linker.tp_group = 2, 0, object()
        linker.read_plan_enabled = mode == "read_plan"
        linker.enable_page_wise_load = mode == "page"
        linker.page_wise_load_threshold = 1
        linker.read_layout_enabled = linker.read_plan_reuse_ranges = False
        linker.load_queue, linker.completed_loads = Queue(), Queue()
        linker.load_batch_futures, linker.load_batch_cleanup_futures = {}, {}
        linker.follower_cleanup_outcomes = {}
        linker.broadcast_loads, linker.broadcast_outcomes = {}, {}
        linker.broadcast_failed, linker.broadcast_events = set(), []
        linker.pending_load_tokens, linker.pending_load_started = {}, {}
        linker.mla_broadcaster = SimpleNamespace(
            is_src=True,
            prepare_broadcast=lambda *args: {},
            broadcast_loaded_layer=lambda layer, _: broadcasts.append(layer),
        )
        linker.freeze_gc_once = lambda: None
        linker.stats = {"load": 0}
        counter_cls = (
            ReadPlanLoadCounter if mode == "read_plan" else LayerWiseLoadCounter
        )
        linker.layer_done_counter = counter_cls(2, linker._broadcast_loaded_layers)
        transfer = PoolTransfer(
            name=PoolName.KV, keys=["key"], host_indices=torch.tensor([0])
        )
        if prefetch:
            self.assertTrue(linker._prepare_expanded_load("rid", [transfer]))
        linker.pending_loads = {"rid": [transfer]}
        event = SimpleNamespace(
            record=lambda: None, synchronize=lambda: None, query=lambda: True
        )
        device = SimpleNamespace(
            Event=lambda: event,
            current_stream=lambda: SimpleNamespace(wait_event=lambda _: None),
        )
        errors = []

        def forward():
            try:
                linker.layer_done_counter.set_consumer(0)
                for layer in range(2):
                    linker.layer_done_counter.wait_until(layer)
                forwarded.set()
            except BaseException as error:
                errors.append(error)

        def retire():
            linker.wait_for_load_batch(0)
            retired.set()

        with (
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.device_module",
                device,
            ),
            patch("torch.distributed.get_world_size", return_value=1),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda out, value, **_: out.__setitem__(0, value),
            ),
            patch("torch.distributed.all_reduce"),
        ):
            self.assertEqual(linker.start_layer_wise_loading(), 0)
            following = Future()
            linker.load_batch_futures[1] = following
            linker.load_queue.put(None)
            worker = threading.Thread(target=linker.load_thread_func, daemon=True)
            source = threading.Thread(target=forward, daemon=True)
            waiter = threading.Thread(target=retire, daemon=True)
            worker.start()
            source.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(
                    forwarded.wait(1), "source forward waited for session end"
                )
                self.assertFalse(errors)
                self.assertFalse(linker.load_batch_cleanup_futures[0].done())
                self.assertEqual(linker.num_completed_loads(), 0)
                self.assertEqual(broadcasts, [] if read_failed else [0, 1])
                if not read_failed:
                    self.assertTrue(all(torch.all(buffer == 7) for buffer in buffers))
                waiter.start()
                self.assertFalse(retired.wait(0.05))
                release.set()
                self.assertTrue(retired.wait(5))
                self.assertFalse(following.done())
                self.assertEqual(linker.num_completed_loads(), 1)
                self.assertEqual(
                    linker.pop_completed_load(),
                    (["rid"], not read_failed and end_rc == 0),
                )
                self.assertEqual(sessions, {"key"} if end_rc else set())
            finally:
                release.set()
                worker.join(5)
                source.join(5)
                if waiter.ident is not None:
                    waiter.join(5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(source.is_alive())

    def test_source_final_verdict_does_not_join_the_next_load(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        current, following = Future(), Future()
        current.set_result(True)
        linker.broadcast_outcomes = {0: current, 1: following}
        linker.broadcast_loads = {0: (0, {}, ["rid"], Mock())}
        linker.broadcast_failed, linker.broadcast_events = set(), []
        linker.load_batch_futures = {0: Future()}
        linker.load_batch_cleanup_futures = {}
        linker.follower_cleanup_outcomes = {}
        linker.load_queue = Mock()
        linker.num_layers, linker.tp_group = 2, object()
        linker.mla_broadcaster = SimpleNamespace(
            is_src=True, broadcast_loaded_layer=Mock()
        )
        event = Mock()
        with (
            patch("torch.distributed.all_reduce"),
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.device_module",
                SimpleNamespace(Event=lambda: event, current_stream=Mock()),
            ),
        ):
            linker._broadcast_loaded_layers(0, 1)
        linker.load_queue.join.assert_not_called()
        self.assertFalse(following.done())
        self.assertEqual(linker.broadcast_events, [(event, ["rid"], True)])

    def test_each_cp_shard_has_one_reader_before_warmup(self):
        pool = SimpleNamespace(
            components=[[torch.zeros((4, 2), dtype=torch.uint8)]],
            buffer_meta=[[(0, 0, 2)]],
            page_size=1,
        )
        pools = SimpleNamespace(
            rank_replicated=True, entry_map={"full": pool}, num_layers=1
        )
        for ranks in ([0, 1], [2, 3]):
            readers = []
            for rank in ranks:
                warmed = []
                with (
                    self.subTest(rank=rank, ranks=ranks),
                    patch("torch.distributed.get_rank", return_value=rank),
                    patch(
                        "torch.distributed.get_process_group_ranks", return_value=ranks
                    ),
                    patch(
                        "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                        return_value="attention-broadcast",
                    ),
                    patch(
                        "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
                        return_value=(rank, 4),
                    ),
                    patch.object(
                        LinkerMLADedupBroadcaster,
                        "_warmup_group",
                        lambda broadcaster: warmed.append(broadcaster.is_src),
                    ),
                ):
                    broadcaster = LinkerMLADedupBroadcaster.build(
                        pools, "full-tp", "attention-tp"
                    )
                self.assertEqual(broadcaster.src_global_rank, ranks[0])
                self.assertEqual(warmed, [rank == ranks[0]])
                readers.append(broadcaster.is_src)
            self.assertEqual(readers, [True, False])

    def test_repeated_layer_access_never_reenters_consensus(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.broadcast_loads = {0: (2, object(), ["rid"], Mock())}
        with patch("torch.distributed.all_reduce") as reduce:
            linker._broadcast_loaded_layers(0, 1)
        reduce.assert_not_called()

    def test_follower_forward_does_not_end_sessions_or_retire_worker_accounting(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.broadcast_loads = {0: (1, {}, ["rid"], Mock())}
        linker.broadcast_failed = set()
        linker.broadcast_events = []
        linker.load_batch_futures = {0: Future()}
        linker.load_batch_cleanup_futures = {}
        linker.follower_cleanup_outcomes = {}
        linker.num_layers = 2
        linker.tp_group = object()
        linker.mla_broadcaster = SimpleNamespace(
            is_src=False, broadcast_loaded_layer=Mock()
        )
        linker.host_prefetch_enabled = True
        linker.abort_prepared_load = Mock()
        linker.pending_load_tokens = {"rid": 128}
        linker.pending_load_started = {"rid": 1.0}
        event = Mock()
        with (
            patch("torch.distributed.all_reduce"),
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.device_module",
                SimpleNamespace(Event=lambda: event),
            ),
        ):
            linker._broadcast_loaded_layers(0, 1)
        linker.abort_prepared_load.assert_not_called()
        self.assertEqual(linker.pending_load_tokens, {"rid": 128})
        self.assertEqual(linker.pending_load_started, {"rid": 1.0})
        self.assertEqual(linker.broadcast_events, [(event, ["rid"], True)])
        linker.wait_for_load_batch(0)
        event.synchronize.assert_called_once()
        self.assertFalse(linker.load_batch_futures)


if __name__ == "__main__":
    unittest.main()
