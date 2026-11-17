"""Failed load pages cannot be republished by the next chunk's checkpoint."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFailedChunkOwnership(CustomTestCase):
    def setUp(self):
        scope = get_context().override_server_args(model_path="dummy")
        scope.install()
        self.addCleanup(scope.restore)
        pool = MHATokenToKVPool(
            size=16,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=16,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
                tree_core_backend="python",
            )
        )

    def _loading_chunk(self):
        req = Req("r", "", array("q", [1, 2, 3, 4, 5]), SamplingParams())
        slots = self.allocator.alloc(4)
        self.cache.req_to_token_pool.alloc([req])
        self.cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 4)), slots)
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 4
        req.full_untruncated_fill_ids = array("q", [1, 2, 3, 4, 5])
        req.extend_end, req.prefix_len = 4, 2
        result = self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2])),
                value=slots[:2],
            )
        )
        req.last_node = result.last_device_node
        req.lock = self.cache.lock(req.last_node)
        req.kv.cache_protected_len = req.kv.cache_inserted_len = 2
        transfer_lock = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = self.cache
        wrapper.pending_loads = {
            "backend": (req.last_node, transfer_lock, self.cache.tree_core.root_node.id)
        }
        wrapper.taken_loads = [(["backend"], False)]
        wrapper.load_request_rids = {"backend": req.rid}
        wrapper.request_ids = {req.rid: "backend"}
        wrapper.released_load_ids, wrapper.failed_chains = set(), {}
        # Storage is the external boundary; exercise the real tree/allocator path.
        wrapper.match = lambda key, request, result: result
        wrapper._skip_swa = False
        wrapper.pending_offloads = []
        self.offloads = []

        def offload(transfers):
            self.offloads.append(transfers)
            return False

        wrapper.cache_linker = SimpleNamespace(offload=offload)
        self.cache.linker = wrapper
        # Match production: the real wrapper enables external write-through.
        self.cache.tree_core.enable_external_cache_linker = True
        self.cache.write_through_threshold = 1
        self.cache.tree_core.backfill_missing_hash_values()
        self.cache.tree_core.mark_external_cache_stored_path(
            req.last_node, self.cache.tree_core.root_node.id
        )
        return req, slots

    def _checkpoint(self, req, pd):
        scheduler = SimpleNamespace(
            tree_cache=self.cache,
            disaggregation_mode=DisaggregationMode.PREFILL
            if pd
            else DisaggregationMode.NULL,
        )
        scheduler.checkpoint_disagg_prefill = (
            lambda r: SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(
                scheduler, r
            )
        )
        Scheduler.stash_chunked_request(scheduler, req)

    def test_verdict_before_next_checkpoint_never_republishes_failed_slots(self):
        for pd in (False, True):
            with self.subTest(pd=pd):
                req, slots = self._loading_chunk()
                self.assertEqual(
                    self.cache.linker.commit_completed_loads([False]), [req.rid]
                )
                self._checkpoint(req, pd)
                match = self.cache._match_tree(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
                )
                self.assertEqual(match.device_prefix_len, 0)
                self.assertTrue(req.skip_radix_cache_insert)
                self.cache.unlock(req.lock)
                req.lock = None
                self.cache._reclaim_failed_linker_chain(req.rid)
                self.allocator.free(slots[2:])
                self.cache.req_to_token_pool.free(req)
                self.assertEqual(self.allocator.available_size(), 16)
                self.assertEqual(self.allocator.free_pages.unique().numel(), 16)

    def test_pending_load_never_persists_continuation_until_success(self):
        req, _ = self._loading_chunk()
        self._checkpoint(req, False)
        self.assertEqual(req.prefix_len, 4)
        self.assertEqual(req.kv.cache_protected_len, 2)
        self.assertEqual(self.offloads, [])
        self.assertEqual(self.cache.linker.commit_completed_loads([True]), [])
        self._checkpoint(req, False)
        self.assertEqual(req.kv.cache_protected_len, 4)
        self.assertTrue(self.offloads)

    def test_shared_split_prefix_also_defers_publication(self):
        req, slots = self._loading_chunk()
        split = self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1])),
                value=slots[:1],
                prev_prefix_len=1,
                inserted_len=1,
            )
        )
        shared = SimpleNamespace(rid="other", last_node=split.last_device_node)
        self.assertFalse(self.cache.linker.can_checkpoint(shared))
        self.cache.linker.commit_completed_loads([True])
        self.assertTrue(self.cache.linker.can_checkpoint(shared))

    def test_checkpoint_before_verdict_reclaims_detached_descendants_once(self):
        req, slots = self._loading_chunk()
        # Reclaim an already-existing unpublished descendant (e.g. another
        # owner extended this chain before publication was guarded).
        self.cache.tree_core.enable_external_cache_linker = False
        result = self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=slots,
                prev_prefix_len=2,
                inserted_len=2,
            )
        )
        self.cache.unlock(req.lock)
        req.last_node = result.last_device_node
        req.lock = self.cache.lock(req.last_node)
        req.kv.cache_protected_len = req.kv.cache_inserted_len = 4
        self.cache.tree_core.enable_external_cache_linker = True
        self.assertEqual(self.cache.linker.commit_completed_loads([False]), [req.rid])
        self.cache.unlock(req.lock)
        req.lock = None
        self.cache._reclaim_failed_linker_chain(req.rid)
        self.assertFalse(self.cache.has_outstanding_failed_linker_chains())
        self.assertEqual(self.cache.tree_core.all_values_flatten().numel(), 0)
        self.assertEqual(self.allocator.available_size(), 16)
        self.assertEqual(self.allocator.free_pages.unique().numel(), 16)
        self.cache._reclaim_failed_linker_chain(req.rid)
        self.assertEqual(self.allocator.available_size(), 16)


if __name__ == "__main__":
    unittest.main()
