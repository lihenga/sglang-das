"""Retire the current load before streaming without joining the next load."""

import __future__

import ast
import logging
import queue
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _scheduler_method(name, namespace):
    path = Path(__file__).parents[4] / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    method.decorator_list = []
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


class TestMooncakeLinkerResults(CustomTestCase):
    def test_current_verdict_does_not_wait_for_the_next_native_load(self):
        self._assert_current_verdict_scope(broadcast=False)

    def test_mla_source_verdict_does_not_wait_for_the_next_native_load(self):
        self._assert_current_verdict_scope(broadcast=True)

    def _assert_current_verdict_scope(self, *, broadcast):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.load_queue = queue.Queue()
        linker.completed_loads = queue.Queue()
        linker.load_batch_futures = {}
        linker.load_batch_cleanup_futures = {}
        linker.follower_cleanup_outcomes = {}
        linker.broadcast_loads = {}
        linker.broadcast_outcomes, linker.broadcast_failed, linker.broadcast_events = (
            {},
            set(),
            [],
        )
        linker.num_layers, linker.tp_group = 1, object()
        linker.mla_broadcaster = (
            SimpleNamespace(
                is_src=True, prepare_broadcast=Mock(), broadcast_loaded_layer=Mock()
            )
            if broadcast
            else None
        )
        linker._poll_cancelled_load_sessions = Mock()
        linker.host_prefetch_enabled = False
        linker._finish_prefetch_metrics = Mock()
        linker.layer_done_counter = LayerWiseLoadCounter(1)
        linker.freeze_gc_once = Mock()
        linker.stats = {"load": 0}
        current_started, release_current, next_started, release_next, retired = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def load(index, transfers):
            if index == 1:
                next_started.set()
                release_next.wait(timeout=5)
                return True
            current_started.set()
            release_current.wait(timeout=5)
            return False

        linker.load_layer_wise = load
        with (
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.device_module",
                SimpleNamespace(Event=Mock, current_stream=Mock()),
            ),
            patch("torch.distributed.get_world_size", return_value=1),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda out, value, **kw: out.__setitem__(0, value),
            ),
        ):
            for index in (0, 1):
                linker.pending_loads = {str(index): []}
                self.assertEqual(linker.start_layer_wise_loading(), index)
        linker.load_queue.put(None)
        worker = threading.Thread(target=linker.load_thread_func, daemon=True)
        worker.start()

        def retire():
            if broadcast:
                with (
                    patch("torch.distributed.all_reduce"),
                    patch(
                        "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.device_module",
                        SimpleNamespace(Event=Mock, current_stream=Mock()),
                    ),
                ):
                    linker._broadcast_loaded_layers(0, 0)
            linker.wait_for_load_batch(0)
            retired.set()

        waiter = threading.Thread(target=retire, daemon=True)
        try:
            self.assertTrue(current_started.wait(timeout=5))
            waiter.start()
            self.assertFalse(retired.wait(timeout=0.1), "retired before the verdict")
            release_current.set()
            self.assertTrue(next_started.wait(timeout=5))
            self.assertTrue(retired.wait(timeout=1), "joined an unrelated next load")
            if broadcast:
                self.assertEqual(linker.num_completed_loads(), 1)
            self.assertEqual(linker.pop_completed_load(), (["0"], False))
            self.assertNotIn(0, linker.load_batch_futures)
            self.assertFalse(linker.load_batch_futures[1].done())
        finally:
            release_current.set()
            release_next.set()
            worker.join(timeout=5)
            if waiter.ident is not None:
                waiter.join(timeout=5)
        self.assertFalse(worker.is_alive())

    def test_result_processing_scopes_the_barrier_and_skips_decode(self):
        for pp_size, extend in ((1, True), (1, False), (2, True)):
            with self.subTest(pp_size=pp_size, extend=extend):
                namespace = {
                    "get_parallel": lambda: SimpleNamespace(pp_size=pp_size),
                    "flush_trace_batch": Mock(),
                    "DisaggregationMode": DisaggregationMode,
                }
                method = _scheduler_method("process_batch_result", namespace)
                owner = Mock()
                owner.enable_unified_cache_external_linker = True
                owner.tree_cache.drain_linker_loads.return_value = []
                owner.disaggregation_mode = DisaggregationMode.NULL
                owner.enable_fpm = False
                mode = SimpleNamespace(
                    is_extend=lambda: extend, is_decode=lambda: not extend
                )
                batch = SimpleNamespace(
                    forward_mode=mode,
                    reqs=[],
                    hicache_consumer_index=7,
                    is_dllm=lambda: False,
                )
                method(owner, batch, Mock())
                if pp_size > 1:
                    owner.tree_cache.drain_linker_loads.assert_called_once_with(
                        wait_for_completion=True
                    )
                elif extend:
                    owner.tree_cache.linker.wait_for_load_batch.assert_called_once_with(
                        7
                    )
                    owner._mark_failed_linker_loads.assert_called_once_with(batch)
                    owner.tree_cache.drain_linker_loads.assert_not_called()
                else:
                    owner.tree_cache.linker.wait_for_load_batch.assert_not_called()
                    owner._mark_failed_linker_loads.assert_not_called()
                    owner.tree_cache.drain_linker_loads.assert_not_called()

    def test_next_batch_failure_is_retried_before_its_result_is_processed(self):
        namespace = {"logger": logging.getLogger(__name__)}
        method = _scheduler_method("_mark_failed_linker_loads", namespace)
        owner = Mock()
        owner._deferred_linker_rids = set()
        owner.tree_cache.drain_linker_loads.side_effect = [["next"], []]
        owner.tree_cache.has_outstanding_failed_linker_chains.return_value = False
        owner._iter_external_linker_candidates.return_value = []
        method(owner, Mock())
        self.assertEqual(owner._deferred_linker_rids, {"next"})
        req = SimpleNamespace(
            rid="next", external_kv_cache_released=False, time_stats=Mock()
        )
        owner.chunked_req = None
        owner._iter_external_linker_candidates.return_value = [req]
        method(owner, Mock())
        owner.abort_external_kv_request.assert_called_once()
        self.assertIs(owner.abort_external_kv_request.call_args.args[0], req)
        self.assertFalse(owner._deferred_linker_rids)

    def test_waiting_match_is_invalidated_without_aborting_its_request(self):
        namespace = {"logger": logging.getLogger(__name__)}
        method = _scheduler_method("_mark_failed_linker_loads", namespace)
        owner = Mock()
        owner._deferred_linker_rids = set()
        owner.tree_cache.drain_linker_loads.return_value = ["loading"]
        owner.tree_cache.has_outstanding_failed_linker_chains.return_value = True
        owner.tree_cache.is_on_failed_linker_chain.return_value = True
        waiting = SimpleNamespace(
            rid="waiting",
            last_node=7,
            prefix_len=8,
            external_kv_cache_released=False,
            time_stats=Mock(),
            kv=SimpleNamespace(holds_kv=False, holds_mamba=False),
        )
        loading = SimpleNamespace(
            rid="loading",
            last_node=7,
            external_kv_cache_released=False,
            time_stats=Mock(),
        )
        owner.chunked_req = None
        owner._iter_external_linker_candidates.return_value = [loading, waiting]
        method(owner, Mock())
        owner.abort_external_kv_request.assert_called_once()
        self.assertIs(owner.abort_external_kv_request.call_args.args[0], loading)
        owner.tree_cache.cancel_waiting_queue_prefetch.assert_called_once_with(
            "waiting"
        )
        self.assertIsNone(waiting.last_node)
        self.assertEqual(waiting.prefix_len, 0)

    def test_inflight_user_abort_preserves_its_category_and_timeout_message(self):
        namespace = {
            "logger": logging.getLogger(__name__),
            "DisaggregationMode": DisaggregationMode,
            "FINISH_ABORT": FINISH_ABORT,
            "HTTPStatus": HTTPStatus,
        }
        method = _scheduler_method("abort_request", namespace)
        for message in (None, "SLA timeout"):
            with self.subTest(message=message):
                req = SimpleNamespace(
                    rid="rid",
                    to_finish=None,
                    finished_reason=FINISH_LENGTH(0),
                    disagg_kv_sender=Mock(),
                    external_kv_abort_requested=False,
                    external_kv_checkpoint_on_abort=False,
                )
                owner = Mock()
                owner.chunked_req = owner.mm_receiver = owner.dllm_config = None
                owner.waiting_queue = []
                owner.disaggregation_mode = DisaggregationMode.PREFILL
                owner.disagg_prefill_bootstrap_queue.queue = []
                owner.disagg_prefill_inflight_queue = [req]
                owner.collect_inflight_reqs.return_value = []
                method(
                    owner,
                    SimpleNamespace(rid="rid", abort_all=False, abort_message=message),
                )
                self.assertIsInstance(req.to_finish, FINISH_ABORT)
                self.assertIsNone(req.to_finish.err_type)
                if message:
                    self.assertEqual(req.to_finish.message, message)
                    self.assertEqual(
                        req.to_finish.status_code, HTTPStatus.SERVICE_UNAVAILABLE
                    )
                req.disagg_kv_sender.abort.assert_called_once()
                self.assertTrue(req.external_kv_checkpoint_on_abort)


if __name__ == "__main__":
    unittest.main()
