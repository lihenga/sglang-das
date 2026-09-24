"""A PD prefill request that fails bootstrap releases its cache-side state.

Before this fix only HiCache storage released it; with the unified-cache
external linker a waiting-queue prefetch taken at arrival kept its budget
slot, pinned session and hit markers after the request failed.
"""

import threading
import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (  # noqa: E402
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (  # noqa: E402
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid="r"):
    req = MagicMock()
    req.rid = rid
    req.req_pool_idx = None
    req.kv = None
    req.mamba_pool_idx = None
    req.disagg_kv_sender.failure_exception.side_effect = RuntimeError("boom")
    return req


def _scheduler(*, hicache, external_linker, tree_cache):
    s = types.SimpleNamespace(
        enable_hicache_storage=hicache,
        enable_unified_cache_external_linker=external_linker,
        tree_cache=tree_cache,
        ps=types.SimpleNamespace(tp_rank=0),
        clear_pending_chunk_send=MagicMock(),
        req_to_metadata_buffer_idx_allocator=MagicMock(),
        disagg_metadata_buffers=None,
        output_streamer=MagicMock(),
        metrics_reporter=types.SimpleNamespace(enable_metrics=False),
    )
    s._release_aborted_request = lambda rid: Scheduler._release_aborted_request(s, rid)
    return s


def _fail(s, req):
    with patch(
        "sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"
    ), patch("sglang.srt.disaggregation.prefill.prepare_abort"):
        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(s, req)


class TestBootstrapFailureRelease(CustomTestCase):
    def test_external_linker_releases_the_request(self):
        tree_cache = MagicMock()
        s = _scheduler(hicache=False, external_linker=True, tree_cache=tree_cache)
        _fail(s, _req("r"))
        tree_cache.release_aborted_request.assert_called_once_with("r")

    def test_hicache_still_releases_once(self):
        tree_cache = MagicMock()
        s = _scheduler(hicache=True, external_linker=False, tree_cache=tree_cache)
        _fail(s, _req("r"))
        tree_cache.release_aborted_request.assert_called_once_with("r")

    def test_plain_cache_is_untouched(self):
        tree_cache = MagicMock()
        s = _scheduler(hicache=False, external_linker=False, tree_cache=tree_cache)
        _fail(s, _req("r"))
        tree_cache.release_aborted_request.assert_not_called()


def _mooncake_linker(state):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_lock = threading.Lock()
    linker.session_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.pending_loads = {}
    linker.host_prefetch_entries = {
        "r": {
            "state": state,
            "cancelled": False,
            "session_rid": "s-r",
            "reserved_bytes": 1 << 20,
        }
    }
    linker._abort_prepared_load_now = MagicMock()
    return linker


class TestReleaseFreesThePrefetch(CustomTestCase):
    """What release_aborted_request does to a queued prefetch."""

    def _wrapper(self, cache_linker):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = MagicMock()
        wrapper.cache_linker = cache_linker
        hit = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h"], device_hit_len=0
        )
        wrapper.hit_markers = {"r": hit}
        wrapper.host_prefetch_hits = {"r": hit}
        wrapper.pending_loads = {}
        return wrapper

    def test_completed_prefetch_frees_its_slot_and_session(self):
        linker = _mooncake_linker("dfs_prefetched")
        wrapper = self._wrapper(linker)
        wrapper.release_request("r")
        self.assertEqual(linker.host_prefetch_entries, {})
        linker._abort_prepared_load_now.assert_called_once_with("s-r")
        self.assertEqual(wrapper.hit_markers, {})
        self.assertEqual(wrapper.host_prefetch_hits, {})

    def test_in_flight_read_is_only_marked_cancelled(self):
        linker = _mooncake_linker("reading")
        wrapper = self._wrapper(linker)
        wrapper.release_request("r")
        # The worker owns the session until its read ends, then drops it.
        self.assertTrue(linker.host_prefetch_entries["r"]["cancelled"])
        linker._abort_prepared_load_now.assert_not_called()


if __name__ == "__main__":
    unittest.main()
