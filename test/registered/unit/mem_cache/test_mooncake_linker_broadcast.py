import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    def test_cp_shard_uses_its_own_attention_tp_source(self):
        captured = {}

        class Broadcaster(LinkerMLADedupBroadcaster):
            def __init__(self, pool, group, src_global_rank):
                captured.update(group=group, source=src_global_rank)

            def _warmup_group(self):
                captured["warmup"] = True

        with (
            patch(
                "torch.distributed.get_process_group_ranks", return_value=[4, 5]
            ) as ranks,
            patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                return_value="attention-broadcast",
            ) as create,
        ):
            Broadcaster.build(object(), "full-tp", "attention-tp")
        ranks.assert_called_once_with("attention-tp")
        create.assert_called_once_with([4, 5], backend="nccl")
        self.assertEqual(
            captured, {"group": "attention-broadcast", "source": 4, "warmup": True}
        )

    def test_repeated_layer_access_never_reenters_consensus(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.broadcast_loads = {0: (2, object(), ["rid"], Mock())}
        with patch("torch.distributed.all_reduce") as reduce:
            linker._broadcast_loaded_layers(0, 1)
        reduce.assert_not_called()

    def test_follower_completion_releases_session_and_accounting(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.broadcast_loads = {0: (1, {}, ["rid"], Mock())}
        linker.broadcast_failed = set()
        linker.broadcast_events = []
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
        linker.abort_prepared_load.assert_called_once_with("rid")
        self.assertFalse(linker.pending_load_tokens)
        self.assertFalse(linker.pending_load_started)
        self.assertEqual(linker.broadcast_events, [(event, ["rid"], True)])


if __name__ == "__main__":
    unittest.main()
