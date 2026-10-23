"""Direct L3 must preserve target DSA and every plain MLA draft layer."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _build_dsa_device_pool_group,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    _DraftLayerLoadCounter,
)
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
    _can_pack_direct_linker_spark_dspark,
    _can_pack_hicache_mtp,
)
from sglang.srt.speculative import base_spec_worker
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue


def make_pools(device="cpu", dtype=torch.uint8):
    target = DSATokenToKVPool.__new__(DSATokenToKVPool)
    target.size = 128
    target.page_size = 64
    target.start_layer = 0
    target.layer_num = 60
    # Model CP layer splitting: only layers 1 and 12 have target KV here.
    target.kv_buffer = [
        torch.zeros((192 if layer in (1, 12) else 0, 1, 584),
                    dtype=torch.uint8, device=device)
        for layer in range(60)
    ]
    target.indexer_layer_id_to_index = {0: 0, 12: 1}
    target.index_key_cache = SimpleNamespace(buffer=[
        torch.zeros((0, 8320), dtype=torch.uint8, device=device),
        torch.zeros((3, 8320), dtype=torch.uint8, device=device),
    ])
    draft = MLATokenToKVPool.__new__(MLATokenToKVPool)
    draft.size = 128
    draft.page_size = 64
    draft.start_layer = 0
    draft.layer_num = 5
    draft.kv_buffer = [
        torch.zeros((192, 1, 576), dtype=dtype, device=device)
        for _ in range(5)
    ]
    return target, draft


def make_runners(target, draft):
    algorithm = SimpleNamespace(
        is_dspark=lambda: True, is_eagle=lambda: False, is_eagle3=lambda: False
    )
    target_runner = SimpleNamespace(
        spec_algorithm=algorithm, token_to_kv_pool=target
    )
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft,
        model_config=SimpleNamespace(
            num_nextn_predict_layers=1,
            hf_config=SimpleNamespace(
                architectures=["SparkV1DSparkDraftModel"], num_draft_layers=5
            ),
        ),
    )
    return target_runner, (draft_runner,)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.bfloat16])
def test_all_draft_layers_and_mixed_storage_bytes(dtype):
    target, draft = make_pools(dtype=dtype)
    group = _build_dsa_device_pool_group(target, 64, (draft,))
    assert group.num_layers == 65
    assert set(group.entry_map) == {PoolName.KV, PoolName.INDEXER}
    kv = group.entry_map[PoolName.KV]
    indexer = group.entry_map[PoolName.INDEXER]
    assert kv.layer_mapping == {1: 0, 12: 1, **{60 + i: 2 + i for i in range(5)}}
    assert indexer.layer_mapping == {12: 0}
    locations = kv.prepare_locations(torch.arange(64, 128))
    assert locations == [64]
    offset = 2 * 64 * 584
    for layer in range(5):
        pointers, sizes, offsets = kv.get_prepared_layer_range_meta(locations, 60 + layer)
        expected_size = 64 * 576 * draft.kv_buffer[layer].element_size()
        assert pointers == [[draft.kv_buffer[layer][64].data_ptr()]]
        assert sizes == [[expected_size]]
        assert offsets == [[offset]]
        assert indexer.get_prepared_layer_range_meta([1], 60 + layer) is None
        offset += expected_size
    pointers, sizes = kv.get_page_buffer_meta(torch.arange(64, 128))
    assert len(pointers) == 7
    assert sum(sizes) == offset


def test_plain_mla_only_opted_in_for_direct_linker():
    target, draft = make_pools()
    target_runner, runners = make_runners(target, draft)
    assert _can_pack_direct_linker_spark_dspark(target_runner, runners)
    # Host HiCache still cannot pack multi-layer draft buffers.
    assert not _can_pack_hicache_mtp(target_runner.spec_algorithm, runners)


@pytest.mark.parametrize("direct", [False, True])
def test_plan_keeps_host_sidecar_and_direct_packed_separate(monkeypatch, direct):
    target, draft = make_pools()
    target_runner, runners = make_runners(target, draft)
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        _draft_model_runners=lambda: runners,
    )
    monkeypatch.setattr(base_spec_worker, "get_memory", lambda: SimpleNamespace(
        enable_unified_cache_external_linker=direct, enable_hierarchical_cache=True
    ))
    plan = BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan.mode == (HiCacheDraftMode.PACKED if direct else HiCacheDraftMode.SIDECAR)
    assert plan.device_pools == (draft,)
    assert target_runner.mtp_draft_device_pools == ((draft,) if direct else ())


def test_existing_single_layer_dsa_mtp_plan_stays_packed(monkeypatch):
    target, _ = make_pools()
    draft, _ = make_pools()
    draft.layer_num = 1
    target_runner, runners = make_runners(target, draft)
    target_runner.spec_algorithm = SimpleNamespace(
        is_dspark=lambda: False, is_eagle=lambda: True, is_eagle3=lambda: False
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        _draft_model_runners=lambda: runners,
    )
    monkeypatch.setattr(base_spec_worker, "get_memory", lambda: SimpleNamespace(
        enable_unified_cache_external_linker=True, enable_hierarchical_cache=False
    ))
    plan = BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan.mode == HiCacheDraftMode.PACKED
    assert target_runner.mtp_draft_device_pools == (draft,)


@pytest.mark.parametrize("change", ["size", "page_size", "layer_num", "start_layer"])
def test_draft_geometry_mismatch_is_rejected(change):
    target, draft = make_pools()
    setattr(draft, change, getattr(draft, change) + 1)
    target_runner, runners = make_runners(target, draft)
    assert not _can_pack_direct_linker_spark_dspark(target_runner, runners)


def test_unknown_mla_subclasses_are_rejected():
    class CompressedPool(MLATokenToKVPool):
        pass

    target, draft = make_pools()
    draft.__class__ = CompressedPool
    target_runner, runners = make_runners(target, draft)
    assert not _can_pack_direct_linker_spark_dspark(target_runner, runners)
    with pytest.raises(ValueError, match="plain MLA"):
        _build_dsa_device_pool_group(target, 64, (draft,))


def test_unknown_architecture_is_rejected():
    target, draft = make_pools()
    target_runner, runners = make_runners(target, draft)
    runners[0].model_config.hf_config.architectures = ["OtherDraft"]
    assert not _can_pack_direct_linker_spark_dspark(target_runner, runners)


def test_all_pools_remain_required_for_lookup():
    target, draft = make_pools()
    group = _build_dsa_device_pool_group(target, 64, (draft,))
    resolved = group.resolve_transfers([
        PoolTransfer(name=PoolName.KV, keys=["page"], device_indices=torch.arange(64))
    ])
    assert {transfer.name for transfer in resolved} == {PoolName.KV, PoolName.INDEXER}
    assert all(transfer.keys == ["page"] for transfer in resolved)
    assert group.resolve_transfers([
        PoolTransfer(name=PoolName.INDEXER, keys=["page"])
    ]) == []


def test_draft_wait_counter_includes_last_layer():
    waited = []
    counter = _DraftLayerLoadCounter(
        SimpleNamespace(wait_until=waited.append), offset=60
    )
    for layer in range(5):
        counter.wait_until(layer)
    assert waited == [60, 61, 62, 63, 64]


def test_misaligned_pages_are_rejected():
    target, draft = make_pools()
    group = _build_dsa_device_pool_group(target, 64, (draft,))
    kv = group.entry_map[PoolName.KV]
    with pytest.raises(ValueError, match="aligned contiguous pages"):
        kv.prepare_locations(torch.arange(1, 65))
    with pytest.raises(ValueError, match="multiple of page_size"):
        kv.prepare_locations(torch.arange(63))


@pytest.mark.parametrize("algorithm", ["DSPARK", "STANDALONE_DSPARK", "EAGLE", None])
@pytest.mark.parametrize("hidden_wire", [False, True])
def test_prefix_policy_matches_dsv4_and_agrees_across_cp_ranks(algorithm, hidden_wire):
    caps = []
    for rank in range(8):
        sender_calls = []
        req = SimpleNamespace(
            pending_bootstrap=True, metadata_buffer_index=0,
            disagg_decode_prefix_len=64, bootstrap_room=1,
            origin_input_ids=list(range(192)), pd_hidden_max_prefix_len=None,
            time_stats=SimpleNamespace(set_bootstrap_done_time=lambda: None),
            disagg_kv_sender=SimpleNamespace(init=lambda *args: sender_calls.append(args)),
        )
        queue = SimpleNamespace(
            ensure_metadata_buffer=lambda req: True,
            _finalize_pd_hidden_bootstrap=lambda *args: True,
            kv_manager=SimpleNamespace(
                # Bootstrap metadata exists only on rank zero.
                req_to_pd_hidden_meta={1: {"hidden_len": 128}} if rank == 0 else {},
                kv_args=SimpleNamespace(state_types=[StateType.PD_HIDDEN] if hidden_wire else []),
            ),
            scheduler=SimpleNamespace(
                server_args=SimpleNamespace(speculative_algorithm=algorithm),
                token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
            ),
        )
        assert PrefillBootstrapQueue.finalize_bootstrap(queue, req)
        expected = 64 if hidden_wire and algorithm is not None and algorithm.endswith("DSPARK") else None
        assert req.pd_hidden_max_prefix_len == expected
        assert req.start_send_idx == 64 and not req.pending_bootstrap
        assert sender_calls == [(2, 0)]
        caps.append(req.pd_hidden_max_prefix_len)
    assert len(set(caps)) == 1
