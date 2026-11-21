"""Preserve sparse restorable boundaries in direct-linker prefix lookup."""

import itertools
import types
import unittest
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakePrefixLookup(CustomTestCase):
    def test_trailing_windows_match_reference_with_holes(self):
        # Exhaustive short masks include partial head windows and an isolated
        # later hit, which a longest-contiguous-prefix shortcut would lose.
        for count in range(7):
            for present in itertools.product((False, True), repeat=count):
                for window in (1, 2, 4, 8):
                    for multiplier in (1, 2):
                        store = MooncakeStore.__new__(MooncakeStore)
                        store.mem_pool_host = types.SimpleNamespace(kv_buffer=None)
                        store._get_hybrid_page_component_keys = lambda keys, transfer: (
                            [f"{key}:{i}" for key in keys for i in range(multiplier)],
                            multiplier,
                        )
                        store._tag_keys = lambda keys: keys
                        # A page requires every physical component to exist.
                        store._batch_exist = lambda keys: [
                            (
                                int(present[int(key.split(":")[0])])
                                if key.endswith(":0")
                                else 1
                            )
                            for key in keys
                        ]
                        transfer = PoolTransfer(
                            name=PoolName.SWA,
                            keys=["window"] * window,
                            hit_policy=PoolHitPolicy.TRAILING_PAGES,
                        )
                        expected = [
                            end
                            for end in range(1, count + 1)
                            if all(present[max(0, end - window) : end])
                        ]
                        with self.subTest(
                            present=present, window=window, multiplier=multiplier
                        ):
                            result = store.batch_exists_v2(
                                [str(i) for i in range(count)], [transfer]
                            )
                            self.assertEqual(result.restorable_prefix_pages, expected)
                            self.assertEqual(
                                result.kv_hit_pages, max(expected, default=0)
                            )

    def test_prefix_mask_intersects_ranks_without_inventing_boundary(self):
        for local, remote, device_pages in [
            ([1, 3, 5], [1, 2, 4], 0),
            ([1, 3, 5], [2, 3, 5], 3),
            ([0, 1, 1, 3, 7, -1], [1, 3, 5], 0),
            ([], [1, 3, 5], 0),
            ([1, 3], [1, 3], 3),
        ]:
            with self.subTest(local=local, remote=remote, device_pages=device_pages):
                wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)

                def reduce(mask, op):
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    peer = torch.zeros_like(mask)
                    peer[remote] = 1
                    mask.copy_(torch.minimum(mask, peer))

                collective = Mock(side_effect=reduce)
                wrapper.cache = types.SimpleNamespace(
                    _all_reduce_attn_groups=collective
                )
                result = wrapper._sync_restorable_prefix(
                    local, num_pages=5, device_hit_pages=device_pages
                )
                expected = max(
                    (p for p in set(local) & set(remote) if device_pages < p <= 5),
                    default=0,
                )
                self.assertEqual(result, expected)
                collective.assert_called_once()


if __name__ == "__main__":
    unittest.main()
