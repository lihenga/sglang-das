"""Exercise direct-linker lookup against the real Mooncake storage interface."""

from unittest.mock import Mock

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("pp_size", [1, 2])
def test_lookup_requires_kv_indexer_and_checkpoint_on_every_pp_stage(pp_size):
    entries = [
        DevicePoolEntry(
            name=name,
            indices_from_pool=source,
            device_pool=None,
            components=[[torch.zeros((4, 1))]],
            layer_mapping={0: 0},
            page_size=1,
            rows_are_pages=True,
        )
        for name, source in (
            (PoolName.KV, PoolName.KV),
            (PoolName.INDEXER, PoolName.KV),
            (PoolName.MAMBA, PoolName.MAMBA),
        )
    ]
    group = DevicePoolGroup(entries, 1, 1, rank_replicated=False)
    storage = MooncakeStore.__new__(MooncakeStore)
    storage.mem_pool_host = group
    storage.registered_pools = group.entry_map
    storage.pp_rank, storage.pp_size = 0, pp_size
    storage.mla_suffix = "tp0_cp0_pp0"
    storage._tag_keys = lambda keys: keys
    existing = {
        f"{page}_tp0_cp0_pp{rank}_{pool}"
        for page in ("a", "b", "c")
        for rank in range(pp_size)
        for pool in ("k", "indexer", "mamba")
    }
    storage._batch_exist = Mock(
        side_effect=lambda keys: [int(k in existing) for k in keys]
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pool_group = group
    linker.storage = storage
    linker.stats = {"lookup": 0}
    transfers = [
        PoolTransfer(name=PoolName.KV, keys=["a", "b", "c"]),
        PoolTransfer(
            name=PoolName.MAMBA,
            keys=["c"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        ),
    ]

    assert linker.lookup("request", transfers) == [1, 2, 3]
    # Checkpoints are sparse: losing b's state must keep c restorable.
    existing.remove(f"b_tp0_cp0_pp{pp_size - 1}_mamba")
    assert linker.lookup("request", transfers) == [1, 3]
    # Index-K pages must be contiguous, unlike recurrent checkpoints.
    existing.remove(f"b_tp0_cp0_pp{pp_size - 1}_indexer")
    assert linker.lookup("request", transfers) == [1]
