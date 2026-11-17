import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeLinkerBroadcast(CustomTestCase):
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
