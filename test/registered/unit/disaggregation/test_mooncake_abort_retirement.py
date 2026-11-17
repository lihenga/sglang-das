"""A logical transfer failure must wait for source ownership before freeing."""

import __future__

import ast
import logging
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.mem_cache.cleanup import CleanupOutcomeUnknown, run_cleanup_step
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Sender:
    def __init__(self, calls):
        self.calls, self.busy = calls, True

    def abort(self):
        self.calls.append("abort")

    def notify_drained_abort(self):
        assert not self.busy
        self.calls.append("notify")

    def failure_exception(self):
        self.calls.append("failure")
        raise self.failure


def _owner(calls, *, fail_release=False):
    path = Path(__file__).parents[4] / "python/sglang/srt/disaggregation/prefill.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "SchedulerDisaggregationPrefillMixin"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"abort_external_kv_request", "handle_inflight_transfer_failure"}
    ]

    def release(req, cache, *, checkpoint):
        assert checkpoint is False
        calls.append("kv")
        if fail_release:
            raise RuntimeError("opaque release outcome")
        req.kv.holds_kv = False

    namespace = {
        "FINISH_ABORT": FINISH_ABORT,
        "HTTPStatus": HTTPStatus,
        "EXTERNAL_KV_LOAD_ERR_TYPE": next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "EXTERNAL_KV_LOAD_ERR_TYPE"
                for t in n.targets
            )
        ),
        "DisaggregationMode": DisaggregationMode,
        "CleanupOutcomeUnknown": CleanupOutcomeUnknown,
        "run_cleanup_step": run_cleanup_step,
        "release_kv_cache": release,
        "maybe_release_metadata_buffer": lambda *args: calls.append("metadata"),
        "prepare_abort": Mock(),
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    owner = SimpleNamespace(
        _mooncake_cleanup_requests={},
        _external_kv_abort_rids=set(),
        _pending_chunked_abort_req=None,
        disaggregation_mode=DisaggregationMode.PREFILL,
        clear_pending_chunk_send=Mock(),
        _release_aborted_request=Mock(),
        req_to_metadata_buffer_idx_allocator=object(),
        tree_cache=SimpleNamespace(_reclaim_failed_linker_chain=Mock()),
        output_streamer=SimpleNamespace(
            stream_output=lambda *args: calls.append("response")
        ),
        prefill_abort_sender_terminal=lambda req: not req.disagg_kv_sender.busy,
        prefill_abort_forward_drained=lambda req: True,
        metrics_reporter=SimpleNamespace(enable_metrics=True),
        metrics_collector=Mock(),
    )
    for method in methods:
        setattr(owner, method.name, namespace[method.name].__get__(owner))
    return owner


def _request(calls, reason=None):
    req = SimpleNamespace(
        rid="rid",
        to_finish=None,
        finished_reason=reason,
        skip_radix_cache_insert=False,
        return_logprob=False,
        kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
        disagg_kv_sender=_Sender(calls),
        time_stats=Mock(),
        external_kv_cleanup_steps={},
    )
    for flag in (
        "abort_requested",
        "pending_chunk_cleared",
        "linker_released",
        "sender_abort_requested",
        "metadata_released",
        "cache_released",
        "finish_state_applied",
        "response_sent",
        "abort_response_via_chunked",
        "cleanup_done",
    ):
        setattr(req, "external_kv_" + flag, False)
    return req


class TestMooncakeAbortRetirement(CustomTestCase):
    def test_transport_root_cause_is_reported_once_and_keeps_its_error_category(self):
        for propagated in (False, True):
            with self.subTest(propagated=propagated):
                calls = []
                owner, req = _owner(calls), _request(calls, FINISH_LENGTH(0))
                req.bootstrap_room = 9
                error = RuntimeError("RDMA disconnected")
                error.is_from_another_rank = propagated
                req.disagg_kv_sender.failure = error
                with self.assertLogs(level="DEBUG") as logs:
                    self.assertFalse(owner.handle_inflight_transfer_failure(req))
                    self.assertFalse(owner.handle_inflight_transfer_failure(req))
                    req.disagg_kv_sender.busy = False
                    self.assertTrue(owner.handle_inflight_transfer_failure(req))
                self.assertEqual(len(logs.records), 1)
                self.assertEqual(
                    logs.records[0].levelname, "DEBUG" if propagated else "WARNING"
                )
                self.assertEqual(calls.count("failure"), 1)
                self.assertIn("RDMA disconnected", req.finished_reason.message)
                self.assertIsNone(req.finished_reason.err_type)
                owner.metrics_collector.increment_transfer_failed_reqs.assert_called_once()

    def test_external_load_failure_retains_the_donor_error_type(self):
        calls = []
        owner, req = _owner(calls), _request(calls)
        req.disagg_kv_sender.busy = False
        self.assertTrue(
            owner.abort_external_kv_request(
                req, forward_drained=True, sender_terminal=True
            )
        )
        self.assertEqual(req.finished_reason.err_type, "ExternalKVLoadError")

    def test_failed_transfer_overrides_an_earlier_prefill_length_finish(self):
        calls = []
        owner, req = _owner(calls), _request(calls, FINISH_LENGTH(0))
        self.assertFalse(
            owner.abort_external_kv_request(
                req, forward_drained=True, sender_terminal=True, emit_response=True
            )
        )
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertNotIn("metadata", calls)
        self.assertNotIn("kv", calls)
        req.disagg_kv_sender.busy = False
        self.assertTrue(
            owner.abort_external_kv_request(
                req, forward_drained=True, sender_terminal=True, emit_response=True
            )
        )
        self.assertEqual(calls.count("metadata"), 1)
        self.assertEqual(calls.count("kv"), 1)
        self.assertEqual(calls.count("response"), 1)
        self.assertTrue(
            owner.abort_external_kv_request(
                req, forward_drained=True, sender_terminal=True, emit_response=True
            )
        )
        self.assertEqual(calls.count("response"), 1)

    def test_unknown_release_outcome_is_retained_without_a_second_free(self):
        calls = []
        owner, req = _owner(calls, fail_release=True), _request(calls)
        req.disagg_kv_sender.busy = False
        with self.assertLogs(level="ERROR"):
            self.assertFalse(
                owner.abort_external_kv_request(
                    req, forward_drained=True, sender_terminal=True
                )
            )
        self.assertFalse(
            owner.abort_external_kv_request(
                req, forward_drained=True, sender_terminal=True
            )
        )
        self.assertEqual(calls.count("kv"), 1)
        self.assertIn(id(req), owner._mooncake_cleanup_requests)

    def test_existing_abort_reason_keeps_its_status(self):
        calls = []
        owner, req = _owner(calls), _request(calls)
        reason = FINISH_ABORT("client aborted", HTTPStatus.BAD_REQUEST)
        req.to_finish = reason
        req.disagg_kv_sender.busy = False
        self.assertTrue(
            owner.abort_external_kv_request(
                req, forward_drained=True, sender_terminal=True
            )
        )
        self.assertIs(req.finished_reason, reason)


if __name__ == "__main__":
    unittest.main()
