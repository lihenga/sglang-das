from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("success", [True, False])
def test_load_reports_l4_metrics_per_request_without_double_counting(success):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pending_load_tokens = {"first": 64, "second": 128}
    linker.storage = SimpleNamespace(store=Mock())
    collector = Mock(spec=StorageMetricsCollector)
    collector.labels = {}
    for name in (
        "prefetched_tokens_total",
        "l4_prefetched_tokens_total",
        "l4_prefetched_tokens_per_request",
        "l4_prefetch_duration_seconds",
    ):
        setattr(collector, name, Mock())
    collector.log_l4_prefetch.side_effect = (
        lambda *args: StorageMetricsCollector.log_l4_prefetch(collector, *args)
    )
    collector.log_prefetched_tokens.side_effect = (
        lambda tokens: StorageMetricsCollector.log_prefetched_tokens(collector, tokens)
    )
    linker.storage_metrics_collector = collector

    linker._finish_prefetch_metrics(["first", "second"], success, 0.25)

    assert not linker.pending_load_tokens
    assert collector.log_l4_prefetch.call_args_list == [
        call("mooncake", 64, 0.25, success),
        call("mooncake", 128, 0.25, success),
    ]
    assert (
        collector.l4_prefetch_duration_seconds.labels.call_args_list
        == [call(source="mooncake", result="success" if success else "failure")] * 2
    )
    if success:
        expected = [call(64), call(128)]
        assert (
            collector.prefetched_tokens_total.labels.return_value.inc.call_args_list
            == expected
        )
        assert (
            collector.l4_prefetched_tokens_total.labels.return_value.inc.call_args_list
            == expected
        )
        linker.storage.store.record_prefetched_tokens.assert_called_once_with(192)
    else:
        collector.prefetched_tokens_total.labels.assert_not_called()
        collector.l4_prefetched_tokens_total.labels.assert_not_called()
        linker.storage.store.record_prefetched_tokens.assert_not_called()


def test_metrics_failure_does_not_break_completed_load_cleanup():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pending_load_tokens = {"first": 64}
    linker.storage = SimpleNamespace(store=Mock())
    linker.storage_metrics_collector = Mock()
    linker.storage_metrics_collector.log_l4_prefetch.side_effect = RuntimeError(
        "metrics unavailable"
    )

    linker._finish_prefetch_metrics(["first"], True, 0.25)

    assert not linker.pending_load_tokens
    linker.storage.store.record_prefetched_tokens.assert_called_once_with(64)
