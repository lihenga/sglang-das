import unittest
from array import array
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    PreparedHostPrefetch,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeFailedChain(CustomTestCase):
    def make_core(self):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        root, child = [UnifiedTreeNode((ComponentType.FULL,)) for _ in range(2)]
        root.key = RadixKey(array("q"))
        child.key = RadixKey(array("q", [1, 2]))
        child.parent = root
        child.component_data[ComponentType.FULL].value = torch.tensor([4, 5])
        key = child.key.child_key(2)
        root.children[key] = child
        core.root_node = root
        core.page_size = 2
        core._node_arena = {root.id: root, child.id: child}
        core._detached_roots = {}
        core.full_host_duplicates = {}
        core.evictable_device_leaves = {child}
        core.evictable_host_leaves = set()
        core.components = ()
        return core, root, child, key

    def test_failed_chain_is_unreachable_before_its_last_owner_unlocks(self):
        core, root, child, key = self.make_core()
        child.component_data[ComponentType.FULL].lock_ref = 1
        self.assertEqual(core.detach_external_load_chain(child.id, root.id), [child.id])
        self.assertNotIn(key, root.children)
        self.assertTrue(core.is_on_detached_chain(child.id))
        self.assertTrue(core.holds_detached_node(child.id))
        result = core.invalidate_external_load_chain(child.id)
        self.assertFalse(result.is_dropped)
        self.assertFalse(result.device_frees)
        self.assertFalse(result.host_frees)

    def test_reclaiming_old_chain_cannot_remove_new_prefix_with_the_same_key(self):
        core, root, child, key = self.make_core()
        core.detach_external_load_chain(child.id, root.id)
        replacement = UnifiedTreeNode((ComponentType.FULL,))
        replacement.parent, replacement.key = root, child.key
        root.children[key] = replacement
        core._node_arena[replacement.id] = replacement
        core._remove_leaf_from_parent(child)
        self.assertIs(root.children[key], replacement)
        self.assertFalse(core.holds_detached_node(child.id))
        core._update_evictable_leaf_sets(child)
        self.assertNotIn(child, core.evictable_device_leaves)

    def test_retained_detached_allocations_remain_in_the_ownership_census(self):
        core, root, child, _ = self.make_core()
        core.device = "cpu"
        core.components_by_type = {ComponentType.FULL: object()}
        core.detach_external_load_chain(child.id, root.id)
        self.assertEqual(core.all_values_flatten().tolist(), [4, 5])
        self.assertEqual(
            core._all_component_values_flatten(ComponentType.FULL).tolist(), [4, 5]
        )

    def test_enqueue_rejection_detaches_before_unlocking_published_pages(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                events = []
                wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
                wrapper.pending_loads, wrapper.failed_chains = {}, {}
                backend = Mock()
                backend.load.return_value = False
                if raises:
                    backend.load.side_effect = RuntimeError("native enqueue failed")
                wrapper.cache_linker = backend

                def detach(endpoint, anchor):
                    events.append("detach")
                    return [endpoint]

                wrapper.cache = Mock()
                wrapper.cache.tree_core.detach_external_load_chain = detach
                wrapper.cache.dec_lock_ref.side_effect = lambda *args: events.append(
                    "unlock"
                )
                with self.assertRaises(RuntimeError):
                    wrapper._queue_load("rid", 7, [object()], anchor=3)
                self.assertEqual(events, ["detach", "unlock"])
                self.assertEqual(wrapper.failed_chains["rid"], [7])

    def test_attempt_reuse_keeps_old_ack_and_release_away_from_new_prefetch(self):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        backend = Mock()
        wrapper.cache_linker = backend
        wrapper._waiting_queue_prefetch_enabled = True
        wrapper.request_ids = {"rid": "new-attempt"}
        wrapper.hit_markers = {"rid": object()}
        wrapper.host_prefetch_hits = {}
        old = PreparedHostPrefetch(
            "old-attempt", True, (), backend, Mock(), request_rid="rid"
        )
        new = PreparedHostPrefetch(
            "new-attempt", True, (), backend, Mock(), request_rid="rid"
        )
        wrapper.pending_host_prefetch_submissions = {"rid": (new, object())}
        self.assertFalse(wrapper.complete_host_prefetch_submission(old, True))
        self.assertIs(wrapper.pending_host_prefetch_submissions["rid"][0], new)
        backend.cancel_host_prefetch.assert_called_once_with("old-attempt")
        backend.reset_mock()
        wrapper.release_request(CacheRequestHandle("rid", 0))
        backend.cancel_host_prefetch.assert_not_called()
        self.assertIn("rid", wrapper.hit_markers)

    def test_dispatched_release_retires_only_its_own_attempt_mapping(self):
        for current in ("old", "new"):
            with self.subTest(current=current):
                wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
                wrapper.pending_loads = {"old": (7, object(), 3)}
                wrapper.taken_loads = [(["old"], True)]
                wrapper.load_request_rids = {"old": "rid"}
                wrapper.released_load_ids = {"old"}
                wrapper.request_ids = {"rid": current}
                wrapper.cache = Mock()
                self.assertEqual(wrapper.commit_completed_loads([True]), [])
                self.assertEqual(
                    wrapper.request_ids, {} if current == "old" else {"rid": "new"}
                )
                self.assertFalse(wrapper.released_load_ids)

    def test_late_failed_completion_does_not_abort_a_new_attempt_with_the_same_rid(
        self,
    ):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.pending_loads = {"old": (7, object(), 3)}
        wrapper.taken_loads = [(["old"], False)]
        wrapper.load_request_rids = {"old": "rid"}
        wrapper.released_load_ids = {"old"}
        wrapper.request_ids = {"rid": "new"}
        wrapper.failed_chains = {}
        wrapper.cache = Mock()
        wrapper.cache.tree_core.detach_external_load_chain.return_value = [7]
        self.assertEqual(wrapper.commit_completed_loads([False]), [])
        self.assertEqual(wrapper.request_ids, {"rid": "new"})
        self.assertEqual(wrapper.failed_chains, {"rid": [7]})


if __name__ == "__main__":
    unittest.main()
