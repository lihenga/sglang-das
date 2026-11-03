"""GLM's 45-layer CP=8 target, MTP and recurrent state survive packed L4 I/O."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.glm5_next import Glm5NextDSATokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    _build_glm5_next_device_pool_group,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _pool(rank, layers, offset=0):
    pool = Glm5NextDSATokenToKVPool.__new__(Glm5NextDSATokenToKVPool)
    pool.start_layer, pool.layer_num = 0, layers
    pool.layer_shard_rank, pool.layer_shard_size = rank, 8
    pool.layer_shard_rank_offset, pool.layer_shard_enabled = offset, True
    pool.page_size, pool.index_kpool = 64, 4
    pool.use_dsa, pool.use_fp8_index_k_cache = True, True
    pool.kv_buffer = [
        torch.full((192 if pool._is_layer_owned(i) else 0, 4), i + 1, dtype=torch.uint8)
        for i in range(layers)
    ]
    pool.index_k_with_scale_buffer = [
        torch.full((3 if pool._is_layer_owned(i) else 0, 8), i + 11, dtype=torch.uint8)
        for i in range(layers)
    ]
    return pool


@pytest.mark.parametrize("rank", range(8))
def test_target_draft_and_kda_replay_at_their_model_layers(monkeypatch, rank):
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_memory",
        lambda: SimpleNamespace(
            enable_unified_memory=False,
            enable_hisparse=False,
            unified_cache_external_linker_backend="mooncake",
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(attn_dcp_size=1),
    )
    full_layers = list(range(3, 45, 4))
    kda_layers = [i for i in range(45) if i not in full_layers]
    mamba_pool = SimpleNamespace(
        num_mamba_layers=len(kda_layers),
        mamba_cache=SimpleNamespace(
            temporal=torch.arange(len(kda_layers) * 4 * 4).reshape(
                len(kda_layers), 4, 4
            ),
            conv=[torch.ones((len(kda_layers), 4, 3))],
        ),
    )
    req_pool = SimpleNamespace(
        mamba_pool=mamba_pool, mamba_map=dict(zip(kda_layers, range(len(kda_layers))))
    )
    target, draft = _pool(rank, len(full_layers)), _pool(rank, 1, offset=7)
    cache = SimpleNamespace(
        start_layer=0,
        full_kv_pool=target,
        mamba_pool=mamba_pool,
        full_attention_layer_id_mapping=dict(zip(full_layers, range(len(full_layers)))),
    )
    params = SimpleNamespace(
        pp_size=1, req_to_token_pool=req_pool, mtp_draft_device_pools=(draft,)
    )
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
        group = _build_glm5_next_device_pool_group(cache, params, 64)
    assert group.num_layers == 45
    transfers = group.resolve_transfers(
        [
            PoolTransfer(
                name=PoolName.KV, keys=["page"], device_indices=torch.arange(64, 128)
            ),
            PoolTransfer(
                name=PoolName.MAMBA, keys=["page"], device_indices=torch.tensor([1])
            ),
        ]
    )
    for transfer in transfers:
        entry = group.entry_map[transfer.name]
        pointers, sizes = entry.get_page_buffer_meta(transfer.host_indices)
        original = b"".join(ctypes.string_at(p, n) for p, n in zip(pointers, sizes))
        for pointer, size in zip(pointers, sizes):
            ctypes.memset(pointer, 0, size)
        restored_layers = []
        for layer in range(group.num_layers):
            meta = entry.get_prepared_layer_range_meta(
                entry.prepare_locations(transfer.host_indices), layer
            )
            if meta is not None:
                restored_layers.append(layer)
                for ptrs, counts, offsets in zip(*meta):
                    for pointer, size, offset in zip(ptrs, counts, offsets):
                        ctypes.memmove(pointer, original[offset : offset + size], size)
        assert (
            b"".join(ctypes.string_at(p, n) for p, n in zip(pointers, sizes))
            == original
        )
        if transfer.name == PoolName.MAMBA:
            assert restored_layers == kda_layers
        else:
            expected = [
                layer
                for layer, i in cache.full_attention_layer_id_mapping.items()
                if target._is_layer_owned(i)
            ]
            if draft._is_layer_owned(0):
                expected = [0, *expected]
            assert restored_layers == expected
