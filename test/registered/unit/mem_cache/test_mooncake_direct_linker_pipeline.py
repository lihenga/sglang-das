import threading
import types
import unittest
from queue import Queue
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _stats():
    return {
        "lookup": 0,
        "lookup_pages": 0,
        "lookup_hit_pages": 0,
        "lookup_seconds": 0.0,
        "reserve_lock_seconds": 0.0,
        "reserve_lock_max_seconds": 0.0,
        "reserve_rpc_count": 0,
        "reserve_rpc_seconds": 0.0,
        "reserve_rpc_max_seconds": 0.0,
    }


class TestMooncakeDirectLinkerPipeline(CustomTestCase):
    def test_lookup_runs_on_worker_and_publishes_completion(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pp_size = 1
        linker.lookup_queue = Queue()
        linker.completed_lookups = Queue()
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers: transfers
        )
        linker.storage = types.SimpleNamespace(
            batch_exists_v2=Mock(
                return_value=types.SimpleNamespace(restorable_prefix_pages=[1, 2])
            )
        )
        linker.stats = _stats()
        linker.lookup_thread = threading.Thread(
            target=linker.lookup_thread_func, daemon=True
        )
        linker.lookup_thread.start()

        try:
            transfer = PoolTransfer(
                name=PoolName.KV,
                host_indices=torch.tensor([0, 1]),
                keys=["page-0", "page-1"],
            )
            self.assertIsNone(linker.lookup("req", [transfer]))
            linker.lookup_queue.join()
            self.assertEqual(linker.pop_completed_lookup(), ("req", [1, 2]))
        finally:
            linker.lookup_queue.put(None)
            linker.lookup_thread.join()

    def test_reservations_reference_count_shared_keys(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.read_plan_enabled = False
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers, **_kwargs: transfers
        )
        store = types.SimpleNamespace(
            batch_get_session_start=Mock(return_value=[0, 0]),
            batch_get_session_end=Mock(return_value=0),
        )
        linker.storage = types.SimpleNamespace(
            store=store,
            _get_hybrid_page_component_keys=lambda keys, _transfer: (keys, None),
            _tag_keys=lambda keys: [f"tagged:{key}" for key in keys],
        )
        linker.load_sessions = {}
        linker.load_session_refcounts = {}
        linker.load_session_lock = threading.Lock()
        linker._load_revalidation_groups = ()
        linker.stats = _stats()
        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([0, 1]),
            keys=["page-0", "page-1"],
        )

        self.assertTrue(linker.reserve_load("req-1", [transfer]))
        self.assertTrue(linker.reserve_load("req-2", [transfer]))
        store.batch_get_session_start.assert_called_once_with(
            ["tagged:page-0", "tagged:page-1"]
        )
        self.assertEqual(
            linker.load_session_refcounts,
            {"tagged:page-0": 2, "tagged:page-1": 2},
        )

        linker.release_load_reservation("req-1")
        store.batch_get_session_end.assert_not_called()
        linker.release_load_reservation("req-2")
        store.batch_get_session_end.assert_called_once()
        self.assertCountEqual(
            store.batch_get_session_end.call_args.args[0],
            ["tagged:page-0", "tagged:page-1"],
        )
        self.assertEqual(linker.load_session_refcounts, {})


if __name__ == "__main__":
    unittest.main()
