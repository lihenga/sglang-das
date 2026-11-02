"""Unit tests for external-linker load failure handling.

A failed remote read used to raise out of the model forward pass and take the
scheduler process down with it. These tests pin the replacement contract: the
failure travels as a value, the tree releases what the load pinned, and the
affected requests are reported so the scheduler can abort them.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest
from http import HTTPStatus
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import (
    EXTERNAL_KV_LOAD_ERR_TYPE,
    is_external_kv_load_failure,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    LayerWiseLoadCounter,
    UMBPDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore


class FakeLinker:
    """Stands in for a backend, replaying a scripted sequence of outcomes."""

    def __init__(self, completions):
        self._completions = list(completions)
        self.queued = {}
        self.offloaded = []

    def load(self, rid, transfers):
        self.queued[rid] = transfers
        return True

    def cancel_queued_load(self, rid):
        return self.queued.pop(rid, None) is not None

    def num_completed_loads(self):
        return len(self._completions)

    def pop_completed_load(self):
        return self._completions.pop(0)

    def offload(self, transfers):
        self.offloaded.append(transfers)
        return True

    def num_completed_offloads(self):
        return 0

    def reset(self):
        self._completions = []


class FakeKey:
    """A key whose child_key is the node id, so children map id -> node."""

    def __init__(self, token):
        self._token = token

    def child_key(self, page_size):
        return self._token


class FakeNode:
    def __init__(self, node_id, parent=None):
        self.id = node_id
        self.parent = parent
        self.external_cache_stored = True
        self.detached = False
        self.write_through_pending_id = None
        self.key = FakeKey(node_id)
        self.children = {}
        if parent is not None:
            parent.children[node_id] = self


class FakeCache:
    """Just the tree surface drain_loads touches."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.released = []
        self.locks = 0
        # No components: _offload_node then builds an empty transfer list, which
        # is enough to observe whether it decided to offload at all.
        self._components_tuple = ()

    def resolve_node_handle(self, node_id):
        return self.nodes[node_id]

    def inc_lock_ref(self, node_id):
        self.locks += 1

        class _Params:
            def to_dec_params(self):
                return ("dec", node_id)

        return _Params()

    def dec_lock_ref(self, node_id, params):
        self.locks -= 1
        self.released.append(node_id)


def _bare_core(arena=None, **overrides):
    """A UnifiedTreeCore with only what the detach path touches.

    Deliberately the real class: the linker delegates the whole cut to it, so
    stubbing it here would leave the cut untested everywhere.
    """
    core = UnifiedTreeCore.__new__(UnifiedTreeCore)
    core._node_arena = dict(arena or {})
    core._detached_roots = {}
    core.page_size = 1
    core.components = ()
    core.full_host_duplicates = {}
    # Deleting a node clears it from both leaf sets, so they have to exist.
    core.evictable_device_leaves = set()
    core.evictable_host_leaves = set()
    core.root_node = SimpleNamespace(id=-1, parent=None, children={})
    core._update_evictable_leaf_sets = lambda node: None
    for key, value in overrides.items():
        setattr(core, key, value)
    return core


def _make_wrapper(completions, chain_len=3):
    """A wrapper with a linker and a chain, bypassing backend construction."""
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    nodes = {}
    parent = None
    for node_id in range(chain_len):
        parent = FakeNode(node_id, parent)
        nodes[node_id] = parent
    wrapper.cache = FakeCache(nodes)
    wrapper.cache.tree_core = _bare_core(nodes)
    wrapper.cache_linker = FakeLinker(completions)
    wrapper._waiting_queue_prefetch_enabled = False
    wrapper.host_prefetch_hits = None
    wrapper.pending_host_prefetch_submissions = None
    wrapper.hit_markers = {}
    wrapper.pending_loads = {}
    wrapper.pending_offloads = []
    wrapper.failed_chains = {}
    wrapper.taken_loads = []
    return wrapper


def _drain(wrapper, finish_count):
    """take + commit on the local verdict, the way one rank sees it.

    The production path reduces the verdict across the attention group between
    these two calls; see TestLoadVerdictIsReduced.
    """
    successes = wrapper.take_completed_loads(finish_count)
    return wrapper.commit_completed_loads(successes)


class TestUnifiedCacheLinkerLoadFailure(CustomTestCase):
    def test_failed_batch_releases_locks_and_reports_rids(self):
        wrapper = _make_wrapper([(["rid-a", "rid-b"], False)])
        for rid in ("rid-a", "rid-b"):
            wrapper._queue_load(rid, 2, ["transfer"], anchor=0)
        self.assertEqual(wrapper.cache.locks, 2)

        failed = _drain(wrapper, 1)

        self.assertEqual(sorted(failed), ["rid-a", "rid-b"])
        # The load must not keep pinning the chain it can no longer fill.
        self.assertEqual(wrapper.cache.locks, 0)
        self.assertEqual(wrapper.pending_loads, {})

    def test_failed_batch_detaches_the_chain_it_published(self):
        """The published chain must never be offloaded back into the store."""
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        _drain(wrapper, 1)

        # Nodes 1 and 2 were published by this load; node 0 is the anchor the
        # request already had, and must be left alone.
        self.assertTrue(wrapper.cache.nodes[2].detached)
        self.assertTrue(wrapper.cache.nodes[1].detached)
        self.assertFalse(wrapper.cache.nodes[0].detached)
        self.assertEqual(wrapper.take_failed_chain("rid-a"), [2, 1])
        self.assertEqual(wrapper.take_failed_chain("rid-a"), [])

    def test_successful_batch_reports_nothing_and_keeps_the_chain(self):
        wrapper = _make_wrapper([(["rid-a"], True)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        failed = _drain(wrapper, 1)

        self.assertEqual(failed, [])
        self.assertEqual(wrapper.cache.locks, 0)
        self.assertTrue(wrapper.cache.nodes[2].external_cache_stored)
        self.assertFalse(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.failed_chains, {})

    def test_drain_tolerates_a_request_released_while_queued(self):
        """release_request cancels an unstarted load; its rid still comes back."""
        wrapper = _make_wrapper([(["rid-a", "rid-b"], False)])
        for rid in ("rid-a", "rid-b"):
            wrapper._queue_load(rid, 2, ["transfer"], anchor=0)
        wrapper.release_request("rid-a")
        self.assertEqual(wrapper.cache.locks, 1)

        failed = _drain(wrapper, 1)

        self.assertEqual(failed, ["rid-b"])
        self.assertEqual(wrapper.cache.locks, 0)

    def test_take_reports_the_local_verdict_without_touching_the_tree(self):
        """take must not detach or unlock -- the verdict is not final yet."""
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        successes = wrapper.take_completed_loads(1)

        self.assertEqual(successes, [False])
        # Still pinned and still on the tree until commit applies the verdict.
        self.assertEqual(wrapper.cache.locks, 1)
        self.assertEqual(wrapper.pending_loads.keys(), {"rid-a"})
        self.assertFalse(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.failed_chains, {})

    def test_commit_honours_a_verdict_the_rank_did_not_reach_itself(self):
        """A rank whose own get succeeded must still abort when a peer's failed.

        This is the divergence the MIN-reduce exists to stop: on hardware, one
        rank's batch_get missing a key made that rank abort while the other
        seven served the request, and the rank owning the output stream emitted
        KV that never arrived.
        """
        wrapper = _make_wrapper([(["rid-a"], True)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        self.assertEqual(wrapper.take_completed_loads(1), [True])
        failed = wrapper.commit_completed_loads([False])  # reduced from a peer

        self.assertEqual(failed, ["rid-a"])
        self.assertEqual(wrapper.cache.locks, 0)
        self.assertTrue(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.take_failed_chain("rid-a"), [2, 1])

    def test_commit_keeps_a_chain_when_the_group_agrees_it_landed(self):
        wrapper = _make_wrapper([(["rid-a"], True)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        self.assertEqual(wrapper.take_completed_loads(1), [True])
        self.assertEqual(wrapper.commit_completed_loads([True]), [])
        self.assertTrue(wrapper.cache.nodes[2].external_cache_stored)
        self.assertFalse(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.failed_chains, {})

    def test_reset_drops_batches_taken_but_never_committed(self):
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)
        wrapper.take_completed_loads(1)

        wrapper.reset()

        self.assertEqual(wrapper.taken_loads, [])

    def test_layer_counter_does_not_raise_into_the_forward(self):
        counter = LayerWiseLoadCounter(4)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.complete(index, 0)
        counter.fail(index, RuntimeError("transport reset"))

        # Every layer wait must return; raising here kills the engine.
        for layer in range(4):
            counter.wait_until(layer)

        self.assertNotIn(index, counter._futures)


class TestFailedLoadDoesNotPinTheForward(CustomTestCase):
    """A failed load must not keep the model forward's frames alive.

    fail() publishes one exception to every layer's future, and each layer's
    wait_until re-raises it. Python appends a traceback entry per raise, and
    each entry roots a frame chain reaching that layer's forward frame, keeping
    its locals -- that layer's activations -- alive. On DeepSeek-V4 that
    retained one full activation set per layer: +31.9 GiB on a single faulted
    forward at --chunked-prefill-size 2048, stacking across failures until the
    engine OOMed inside the attention kernel.
    """

    def test_the_published_failure_carries_no_traceback(self):
        counter = LayerWiseLoadCounter(8)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, RuntimeError("transport reset"))

        for layer in range(8):
            counter.wait_until(layer)
            published = counter._futures.get(index)
            if published is None:  # popped on the last layer
                break
            error = published[layer].exception()
            depth = 0
            tb = error.__traceback__
            while tb is not None:
                depth += 1
                tb = tb.tb_next
            self.assertEqual(
                depth,
                0,
                f"layer {layer}: traceback survived wait_until, so every frame "
                f"it reaches -- including that layer's forward -- stays alive",
            )

    def test_the_caller_keeps_its_own_exception_intact(self):
        """fail() must not strip the traceback of the exception handed to it:
        the loader thread logs it with logger.exception() right after."""
        counter = LayerWiseLoadCounter(4)
        index = counter.update_producer()
        counter.set_consumer(index)
        try:
            raise RuntimeError("transport reset")
        except RuntimeError as caller_error:
            counter.fail(index, caller_error)
            for layer in range(4):
                counter.wait_until(layer)
            self.assertIsNotNone(caller_error.__traceback__)

    def test_the_message_survives(self):
        counter = LayerWiseLoadCounter(2)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, ValueError("UMBP get failed for pool=swa"))

        published = counter._futures[index][0].exception()
        self.assertIn("UMBP get failed for pool=swa", str(published))
        self.assertIn("ValueError", str(published))


class TestUMBPLinkerCompletionChannel(CustomTestCase):
    def test_linker_is_concrete(self):
        """Adding a UnifiedCacheLinker method without implementing it here
        leaves UMBPDirectLinker abstract, and the mori backend then dies at
        construction with TypeError, nowhere near the cause."""
        self.assertFalse(
            inspect.isabstract(UMBPDirectLinker),
            "UMBPDirectLinker cannot be constructed while it leaves "
            "UnifiedCacheLinker methods unimplemented",
        )

    def _linker_with_queues(self, run_result):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker._load_queue = Queue()
        linker._completed_loads = Queue()
        linker._run_layer_wise_batch = run_result
        return linker

    # The load task carries the event guarding the batch's KV; the stubs below
    # replace the batch itself, so it only has to be there to be unpacked.
    _READY = SimpleNamespace(synchronize=lambda: None)

    def test_load_thread_publishes_failure(self):
        linker = self._linker_with_queues(lambda index, plans, ready_event: False)
        linker._load_queue.put((0, ["rid-a"], [], self._READY))
        linker._load_queue.put(None)
        linker._load_thread_func()

        self.assertEqual(linker.num_completed_loads(), 1)
        self.assertEqual(linker.pop_completed_load(), (["rid-a"], False))

    def test_load_thread_publishes_even_when_the_batch_raises(self):
        """The tree's locks hang on this batch coming back, however it ends."""

        def boom(index, plans, ready_event):
            raise RuntimeError("unexpected")

        linker = self._linker_with_queues(boom)
        linker._load_queue.put((0, ["rid-a"], [], self._READY))
        with self.assertRaises(RuntimeError):
            linker._load_thread_func()

        self.assertEqual(linker.pop_completed_load(), (["rid-a"], False))

    def test_load_thread_publishes_success(self):
        linker = self._linker_with_queues(lambda index, plans, ready_event: True)
        linker._load_queue.put((0, ["rid-a"], [], self._READY))
        linker._load_queue.put(None)
        linker._load_thread_func()

        self.assertEqual(linker.pop_completed_load(), (["rid-a"], True))


class TestInvalidateExternalLoadChain(CustomTestCase):
    """The tree-core guard: drop the chain only when nothing else owns it."""

    def _core(self, node, is_device_leaf=True):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core._node_arena = {node.id: node} if node is not None else {}
        core.root_node = object()
        core._is_device_leaf = lambda n: is_device_leaf
        core.deleted = []
        core._delete_unbacked_device_leaf = (
            lambda n, tracker, device_frees, host_frees: core.deleted.append(n.id)
        )
        return core

    def _node(self, **overrides):
        class _Component:
            host_lock_ref = 0

        class _Node:
            id = 7
            backuped = False
            write_through_pending_id = None
            load_back_pending_id = None
            component_data = (_Component(),)

        node = _Node()
        node.children = {}
        for key, value in overrides.items():
            setattr(node, key, value)
        return node

    def test_drops_an_unowned_chain(self):
        node = self._node()
        core = self._core(node)
        self.assertTrue(core.invalidate_external_load_chain(7).is_dropped)
        self.assertEqual(core.deleted, [7])

    def test_declines_when_the_node_is_gone(self):
        core = self._core(None)
        self.assertFalse(core.invalidate_external_load_chain(7).is_dropped)
        self.assertEqual(core.deleted, [])

    def test_declines_for_a_backed_up_or_in_flight_node(self):
        for field in ("backuped", "write_through_pending_id", "load_back_pending_id"):
            node = self._node(**{field: True})
            core = self._core(node)
            self.assertFalse(
                core.invalidate_external_load_chain(7).is_dropped,
                f"{field} should block the drop",
            )
            self.assertEqual(core.deleted, [])

    def test_declines_when_the_node_is_not_a_device_leaf(self):
        """Covers a since-adopted chain: locked, or grown a device child."""
        core = self._core(self._node(), is_device_leaf=False)
        self.assertFalse(core.invalidate_external_load_chain(7).is_dropped)
        self.assertEqual(core.deleted, [])

    def test_declines_for_any_child_not_only_a_device_bearing_one(self):
        """`_is_device_leaf` is not a child check, so it cannot be the one here.

        It only rejects children holding Full KV *on device*. A child that
        holds host-only KV, or none, leaves the node looking free -- and
        deleting it strands that child in the arena with a parent the tree no
        longer holds, where no walk from the root or a detached root reaches it.
        """
        node = self._node()
        node.children = {"tok": object()}
        core = self._core(node, is_device_leaf=True)

        self.assertFalse(core.invalidate_external_load_chain(7).is_dropped)
        self.assertEqual(core.deleted, [])



if __name__ == "__main__":
    unittest.main()
