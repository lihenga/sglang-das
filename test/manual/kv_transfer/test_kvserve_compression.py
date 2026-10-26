"""CPU tests for KVServe mapping and the real control/ack protocol.

Run standalone without importing SGLang's accelerator-dependent package init:
    python -m pytest test/manual/kv_transfer/test_kvserve_compression.py -q
The socket tests use a memory-copy engine, not a claimed RDMA/GPU validation.
"""

import ast
import ctypes
import dataclasses
import importlib.util
import json
import struct
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/disaggregation"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_source("kvserve_test_config", SOURCE / "kvserve_config.py")
adapter = load_source("kvserve_test_adapter", SOURCE / "mooncake/kvserve_adapter.py")
transport = load_source(
    "kvserve_test_transport", SOURCE / "mooncake/kvserve_transport.py"
)


def launch_args(**changes):
    args = dict(
        kvserve_compression="default",
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        max_running_requests=1,
        kv_cache_dtype="auto",
        kvserve_reserved_mb=1024,
        kvserve_rpc_timeout=2,
    )
    args.update(changes)
    return SimpleNamespace(**args)


def test_defaults_and_raw_are_distinct_and_disabled_has_no_dependency():
    assert config.resolve_kvserve_config(None) is None
    assert config.resolve_kvserve_config("default")["compression"] == {
        "enable-compression-pipeline": True,
        "logical_blocks_per_chunk": 64,
    }
    assert config.resolve_kvserve_config("raw") == {
        "raw_aggregate": {"logical_blocks_per_chunk": 64}
    }
    config.validate_kvserve_launch(launch_args(kvserve_compression=None, pp_size=8))
    config.validate_kvserve_launch(launch_args())


@pytest.mark.parametrize("requests", [None, 1, 2, 12, 64])
def test_request_admission_is_independent_of_transfer_slot_count(requests):
    config.validate_kvserve_launch(launch_args(max_running_requests=requests))


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize(
    "option", ["enable_unified_cache_external_linker", "enable_hierarchical_cache"]
)
def test_external_storage_can_coexist_with_kvserve(role, option):
    config.validate_kvserve_launch(
        launch_args(disaggregation_mode=role, max_running_requests=12, **{option: True})
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"disaggregation_mode": "null"},
        {"disaggregation_transfer_backend": "nixl"},
        {"pp_size": 2},
        {"dp_size": 2},
        {"dcp_size": 2},
        {"kv_cache_dtype": "fp8_e4m3"},
        {"enable_dp_attention": True},
        {"enable_prefill_cp": True},
        {"speculative_algorithm": "EAGLE"},
        {"enable_hisparse": True},
        {"enable_unified_memory": True},
        {"enable_page_major_kv_layout": True},
        {"disaggregation_decode_enable_radix_cache": True},
        {"disaggregation_decode_retraction_rebootstrap": True},
        {"kvserve_reserved_mb": 0},
        {"kvserve_rpc_timeout": 0},
        {"kvserve_rpc_timeout": float("inf")},
    ],
)
def test_unsupported_launches_fail_before_loading_model(changes):
    with pytest.raises(ValueError):
        config.validate_kvserve_launch(launch_args(**changes))


@pytest.mark.parametrize(
    "value",
    [
        '"default"',
        "[]",
        "{}",
        '{"enabled": true}',
        '{"compression": {}}',
        '{"compression": true}',
        '{"raw_aggregate": {}, "compression": {}}',
        '{"raw_aggregate": {"logical_blocks_per_chunk": 0}}',
        '{"raw_aggregate": {"logical_blocks_per_chunk": true}}',
        '{"raw_aggregate": {"slot_count": 9}}',
    ],
)
def test_invalid_profile_json_is_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        config.resolve_kvserve_config(value)


def test_explicit_profile_file(tmp_path):
    path = tmp_path / "profile.json"
    contents = {
        "compression": {"enable-compression-pipeline": True, "split_type": "layer"}
    }
    path.write_text(json.dumps(contents))
    assert config.resolve_kvserve_config(f"@{path}") == contents


@pytest.mark.parametrize("tokens", [1, 63, 64, 65, 127, 128])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_spark_tp2_ndh_roundtrip_preserves_heads_pages_and_partial_tail(
    tokens, tp_rank
):
    torch.manual_seed(7)
    page_size, dim, value_dim = 64, 256, 128
    pages = (tokens + page_size - 1) // page_size
    src_pages, dst_pages = [4, 1][:pages], [2, 5][:pages]
    # Two ranks each see their original two of the model's four KV heads.
    global_keys = torch.randn(7 * page_size, 4, dim).to(torch.bfloat16)
    global_values = torch.randn(7 * page_size, 4, value_dim).to(torch.bfloat16)
    keys = global_keys[:, tp_rank * 2 : tp_rank * 2 + 2]
    values = global_values[:, tp_rank * 2 : tp_rank * 2 + 2]
    packed = torch.empty(pages, 2, page_size, dim + value_dim, dtype=torch.bfloat16)
    source = adapter.page_token_indices(src_pages, page_size, keys.shape[0])
    destination = adapter.page_token_indices(dst_pages, page_size, keys.shape[0])[
        :tokens
    ]
    restored_keys, restored_values = torch.zeros_like(keys), torch.zeros_like(values)
    for original, restored, offset in [
        (keys, restored_keys, 0),
        (values, restored_values, dim),
    ]:
        adapter.gather_ndh(packed, original, source, page_size, offset)
        adapter.scatter_ndh(packed, restored, destination, page_size, offset)
        assert torch.equal(restored[destination], original[source[:tokens]])
        mask = torch.ones(restored.shape[0], dtype=torch.bool)
        mask[destination] = False
        assert torch.count_nonzero(restored[mask]) == 0


def test_hybrid_pool_uses_original_full_attention_layer_ids_only():
    ids = list(range(3, 36, 4))
    full = SimpleNamespace(
        k_buffer=[object() for _ in ids], v_buffer=[object() for _ in ids]
    )
    mapping = {i: (ids.index(i), False) if i in ids else (i, True) for i in range(36)}
    pool = SimpleNamespace(full_kv_pool=full, layers_mapping=mapping)
    selected, layer_ids, keys, _ = adapter.full_attention_buffers(pool)
    assert selected is full and layer_ids == ids and len(keys) == 9


@pytest.mark.parametrize("pages", [[], [-1], [1, 1], [100], [True], [1.5]])
def test_invalid_page_maps_fail_before_accelerator_submission(pages):
    with pytest.raises(ValueError):
        adapter.page_token_indices(pages, 64, 640)


class BufferAdapter:
    page_size = 2
    chunk_pages = 2
    profile = "spark-tp-rank-0-test-profile"
    max_u8_bytes = 64

    def __init__(self, slots=1):
        self.slot_count = slots
        self.buffers_by_slot = [
            {name: ctypes.create_string_buffer(64) for name in transport.REGIONS}
            for _ in range(slots)
        ]
        self.buffers = self.buffers_by_slot[0]
        self.restored = []
        self.restored_slots = []
        self.encode_calls = 0

    def validate_pages(self, pages):
        adapter.page_token_indices(pages, self.page_size, 512)

    def validate_sizes(self, part, tokens, sizes):
        assert sizes["u8_bytes"] == sizes["payload"]

    def receive_regions(self, slot_id=0):
        return {
            name: [ctypes.addressof(buf), 64]
            for name, buf in self.buffers_by_slot[slot_id].items()
        }

    def encode(self, pages, part, valid_tokens, slot_id=0):
        buffers = self.buffers_by_slot[slot_id]
        self.encode_calls += 1
        payload = bytes([part, valid_tokens, *pages])
        ctypes.memmove(ctypes.addressof(buffers["payload"]), payload, len(payload))
        result = {
            name: [ctypes.addressof(buffers[name]), 0] for name in transport.REGIONS
        }
        result["payload"][1] = len(payload)
        auxiliary = bytes([valid_tokens, part])
        tail = bytes([len(pages)])
        ctypes.memmove(ctypes.addressof(buffers["aux"]), auxiliary, len(auxiliary))
        ctypes.memmove(ctypes.addressof(buffers["tail"]), tail, len(tail))
        result["aux"][1] = len(auxiliary)
        result["tail"][1] = len(tail)
        result["u8_bytes"] = len(payload)
        return result

    def restore(self, pages, part, valid_tokens, sizes, slot_id=0):
        buffers = self.buffers_by_slot[slot_id]
        data = bytes(buffers["payload"][: sizes["payload"]])
        assert data[0] == part and data[1] == valid_tokens
        if sizes["aux"]:
            assert bytes(buffers["aux"][: sizes["aux"]]) == bytes([valid_tokens, part])
        if sizes["tail"]:
            assert bytes(buffers["tail"][: sizes["tail"]]) == bytes([len(pages)])
        self.restored_slots.append(slot_id)
        self.restored.append(
            (pages, part, valid_tokens, data[2:], threading.get_ident())
        )

    def raw_bytes(self, part, tokens):
        return tokens * 4


class CopyEngine:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def batch_transfer_sync(self, session, sources, destinations, lengths):
        self.calls += 1
        if self.fail:
            return -1
        for source, destination, length in zip(sources, destinations, lengths):
            ctypes.memmove(destination, source, length)
        return 0


def header(profile=None, **changes):
    result = dict(
        version=transport.PROTOCOL_VERSION,
        profile=profile or BufferAdapter.profile,
        request_id="request-7-0",
        room=7,
        pages=[2],
        part=0,
        valid_tokens=2,
        sizes=dict(payload=3, aux=0, tail=0, u8_bytes=3),
    )
    result.update(changes)
    return result


def only_job(service):
    assert len(service.jobs) == 1
    return next(iter(service.jobs.values()))


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 1),
        ("request_id", ""),
        ("request_id", 7),
        ("profile", "other-rank"),
        ("room", "7"),
        ("part", 2),
        ("part", True),
        ("pages", []),
        ("pages", [1, 1]),
        ("pages", [999]),
        ("valid_tokens", 0),
        ("valid_tokens", 3),
        ("sizes", dict(payload=65, aux=0, tail=0, u8_bytes=3)),
        ("sizes", dict(payload=3, aux=0, tail=0, u8_bytes=65)),
        ("sizes", dict(payload=-1, aux=0, tail=0, u8_bytes=3)),
    ],
)
def test_receiver_rejects_incompatible_or_out_of_bounds_descriptors(field, value):
    message = header()
    message[field] = value
    with pytest.raises(ValueError):
        transport.validate_header(message, BufferAdapter())


def test_receiver_credit_is_held_until_scheduler_restore_completes():
    buffers = BufferAdapter()
    service = transport.CompressionReceiver(buffers, lambda _: True, Mock(), 1)
    service.expect_pages(7, [2])
    ready = service.handle({"op": "prepare", "header": header()})
    assert service.handle({"op": "prepare", "header": header()}) == ready
    with pytest.raises(RuntimeError, match="stale"):
        service.handle({"op": "commit", "token": "old-token"})
    buffers.buffers["payload"][:3] = bytes([0, 2, 4])
    assert service.handle({"op": "commit", "token": ready["token"]}) == {
        "status": "pending"
    }
    assert not service.free_slots and not buffers.restored
    service.advance(7)
    reply = service.handle({"op": "status", "token": ready["token"]})
    assert reply == {"status": "done", "drained": True}
    assert not service.jobs and list(service.free_slots) == [0]
    assert buffers.restored[0][-1] == threading.get_ident()


def test_expired_receiver_slot_is_quarantined_not_reassigned():
    failure = Mock()
    service = transport.CompressionReceiver(
        BufferAdapter(), lambda _: True, failure, 0.01
    )
    service.expect_pages(7, [2])
    service.handle({"op": "prepare", "header": header()})
    job = only_job(service)
    job.granted_at -= 1
    service._expire_jobs()
    assert only_job(service) is job and service.poisoned
    failure.assert_called_once()
    with pytest.raises(RuntimeError, match="quarantined"):
        service.handle({"op": "prepare", "header": header()})


def test_aborted_room_never_scatters_into_released_cache():
    active = [True]
    buffers = BufferAdapter()
    service = transport.CompressionReceiver(buffers, lambda _: active[0], Mock(), 1)
    service.expect_pages(7, [2])
    service.handle({"op": "prepare", "header": header()})
    job = only_job(service)
    job.committed = True
    active[0] = False
    service.advance(7)
    assert not buffers.restored and job.done.is_set()
    assert job.error


def test_real_control_socket_chunks_and_ack_with_separate_peer_memory():
    producer, consumer = BufferAdapter(), BufferAdapter()
    engine = CopyEngine()
    service = transport.CompressionReceiver(consumer, lambda room: room == 7, Mock(), 2)
    service.expect_pages(7, [2, 5, 7])
    service.start("127.0.0.1")
    sender = transport.CompressionSender(producer, engine, 2)
    errors = []

    def send():
        try:
            sender.send(service.advertisement(), "peer", 7, [4, 1, 6], [2, 5, 7], 5)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=send)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while thread.is_alive() and time.monotonic() < deadline:
            service.advance(7)
            time.sleep(0.001)
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert engine.calls == 4
        assert service.is_complete(7)
        assert [(p, k, n, b) for p, k, n, b, _ in consumer.restored] == [
            ([2, 5], 0, 4, bytes([4, 1])),
            ([2, 5], 1, 4, bytes([4, 1])),
            ([7], 0, 1, bytes([6])),
            ([7], 1, 1, bytes([6])),
        ]
        assert all(record[-1] == threading.get_ident() for record in consumer.restored)
    finally:
        service.stop()


def test_receiver_rejects_unregistered_or_wrong_destination_ownership():
    service = transport.CompressionReceiver(BufferAdapter(), lambda _: True, Mock(), 1)
    with pytest.raises(RuntimeError, match="not registered"):
        service.handle({"op": "prepare", "header": header()})
    service.expect_pages(7, [3])
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        service.handle({"op": "prepare", "header": header()})
    assert not service.is_complete(7)


def test_clearing_room_holds_unfenced_slot_until_commit_without_cache_writes():
    service = transport.CompressionReceiver(BufferAdapter(), lambda _: True, Mock(), 1)
    service.expect_pages(7, [2])
    service.handle({"op": "prepare", "header": header()})
    job = only_job(service)
    service.forget(7)
    assert not job.done.is_set() and job.error and not service.poisoned
    assert not service.free_slots
    assert not service.is_complete(7)
    service.expect_pages(8, [2])
    # A late old commit proves DMA completion, but must never restore into
    # page 2, which is already owned by the new request.
    result = service.handle({"op": "commit", "token": job.token})
    assert result["status"] == "error" and result["drained"]
    assert not service.adapter.restored and list(service.free_slots) == [0]


def test_write_failure_never_reuses_source_slot():
    producer, consumer = BufferAdapter(), BufferAdapter()
    service = transport.CompressionReceiver(consumer, lambda _: True, Mock(), 0.1)
    service.expect_pages(7, [2])
    service.start("127.0.0.1")
    engine = CopyEngine(fail=True)
    sender = transport.CompressionSender(producer, engine, 1)
    try:
        with pytest.raises(RuntimeError, match="write failed"):
            sender.send(service.advertisement(), "peer", 7, [4], [2], 2)
        count = producer.encode_calls
        with pytest.raises(RuntimeError, match="quarantined"):
            sender.send(service.advertisement(), "peer", 7, [4], [2], 2)
        assert producer.encode_calls == count
        assert not consumer.restored
        assert service.jobs and service.poisoned
    finally:
        service.stop()


def test_decode_failure_quarantines_slot_and_cannot_report_completion():
    buffers = BufferAdapter()
    buffers.restore = Mock(side_effect=RuntimeError("corrupt compressed payload"))
    service = transport.CompressionReceiver(buffers, lambda _: True, Mock(), 1)
    service.expect_pages(7, [2])
    service.handle({"op": "prepare", "header": header()})
    job = only_job(service)
    job.committed = True
    service.advance(7)
    assert job.done.is_set() and service.poisoned
    assert not service.is_complete(7)


def test_unfenced_destination_write_escapes_poll_to_stop_worker():
    buffers = BufferAdapter()
    buffers.restore = Mock(side_effect=adapter.UnfencedCacheWrite("device failure"))
    service = transport.CompressionReceiver(buffers, lambda _: True, Mock(), 1)
    service.expect_pages(7, [2])
    service.handle({"op": "prepare", "header": header()})
    job = only_job(service)
    job.committed = True
    with pytest.raises(adapter.UnfencedCacheWrite):
        service.advance(7)
    assert job.done.is_set() and service.poisoned


@pytest.mark.parametrize("field", ["u8_bytes", "aux", "tail", "payload"])
def test_adapter_validates_exact_representation_byte_counts(field):
    instance = object.__new__(adapter.KVServeAdapter)
    instance.config = SimpleNamespace(uses_aggregate=False, uses_codec=False)
    instance.runtime = SimpleNamespace(
        data_bytes=lambda *_: 16,
        aux_bytes=lambda *_: 8,
        layout=SimpleNamespace(raw_tail_bytes=lambda *_: 4),
    )
    sizes = dict(payload=16, u8_bytes=16, aux=8, tail=4)
    instance.validate_sizes(0, 65, sizes)
    sizes[field] += 1
    with pytest.raises(ValueError):
        instance.validate_sizes(0, 65, sizes)


def extract_node(class_name, method=None, path=None):
    path = path or SOURCE / "mooncake/conn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    if method:
        node = next(
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method
        )
        # Exclude accelerator-dependent tracing from this CPU method harness.
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(module), str(path), "exec")


def test_actual_receiver_poll_advances_restore_before_reporting_success():
    poll = SimpleNamespace(Failed=0, WaitingForInput=2, Success=4)
    namespace = {"KVPoll": poll}
    exec(extract_node("MooncakeKVReceiver", "poll"), namespace)
    service = Mock()
    manager = SimpleNamespace(kvserve_receiver=service, check_status=lambda _: 4)
    receiver = SimpleNamespace(kv_mgr=manager, bootstrap_room=7, conclude_state=None)
    assert namespace["poll"](receiver) == 4
    service.advance.assert_called_once_with(7)


def test_actual_sender_requires_source_write_fence_before_publication():
    namespace = {}
    exec(extract_node("MooncakeKVSender", "send"), namespace)
    sender = SimpleNamespace(
        _source_event=None, kv_mgr=SimpleNamespace(kvserve=object())
    )
    with pytest.raises(RuntimeError, match="forward-stream event"):
        namespace["send"](sender, [1])


def test_actual_success_handler_rejects_raw_peer_without_compressed_pages():
    poll = SimpleNamespace(Failed=0, WaitingForInput=2, Success=4)
    namespace = {"KVPoll": poll}
    exec(extract_node("MooncakeKVManager", "_handle_prefill_status_message"), namespace)
    receiver = SimpleNamespace(is_complete=Mock(return_value=False))
    manager = SimpleNamespace(
        kvserve_receiver=receiver,
        request_status={7: 2},
        _fail_kvserve_room=Mock(),
    )
    namespace["_handle_prefill_status_message"](manager, [b"7", b"4", b"0"])
    manager._fail_kvserve_room.assert_called_once()


def test_registration_accepts_legacy_frames_and_compression_advertisement():
    namespace = {
        "dataclasses": dataclasses,
        "json": json,
        "struct": struct,
        "unpack_int_lists": lambda *_: [],
        "unpack_string_list": lambda *_: [],
    }
    exec(extract_node("KVArgsRegisterInfo"), namespace)
    frames = [b"None", b"127.0.0.1", b"5000", b"peer", b"", b"", b"", b"0", b"2", b"64"]
    frames += [b""] * 9
    assert namespace["KVArgsRegisterInfo"].from_zmq(frames).kvserve is None
    peer = dict(version=1, profile="test", endpoint="tcp://127.0.0.1:1234")
    assert (
        namespace["KVArgsRegisterInfo"]
        .from_zmq(frames + [json.dumps(peer).encode()])
        .kvserve
        == peer
    )


def test_receiver_slots_queue_fifo_and_a_cancelled_waiter_uses_no_credit():
    service = transport.CompressionReceiver(
        BufferAdapter(slots=2), lambda _: True, Mock(), 2
    )
    prepared = []
    headers = []
    for room in range(4):
        service.expect_pages(room, [room])
        h = header(room=room, pages=[room], request_id=f"request-{room}")
        headers.append(h)
        prepared.append(service.handle({"op": "prepare", "header": h}))
    assert [p["status"] for p in prepared] == ["ready", "ready", "queued", "queued"]
    assert prepared[0]["regions"] != prepared[1]["regions"]
    # Cancel a queued request, leaving the two active DMA leases untouched.
    assert service.handle(
        {"op": "cancel", "token": prepared[2]["token"], "drained": True}
    )["drained"]
    assert not service.poisoned and len(service.jobs) == 3
    # An explicitly drained cancellation can also return a granted slot.
    service.handle({"op": "cancel", "token": prepared[0]["token"], "drained": True})
    granted = service.handle({"op": "prepare", "header": headers[3]})
    assert granted["status"] == "ready" and granted["slot_id"] == prepared[0]["slot_id"]
    assert (
        service.handle({"op": "status", "token": prepared[2]["token"]})["status"]
        == "error"
    )


def test_receiver_queued_timeout_does_not_quarantine_other_active_slots():
    service = transport.CompressionReceiver(BufferAdapter(), lambda _: True, Mock(), 1)
    service.expect_pages(7, [2])
    service.expect_pages(8, [3])
    first = service.handle({"op": "prepare", "header": header()})
    queued = service.handle(
        {"op": "prepare", "header": header(room=8, pages=[3], request_id="queued")}
    )
    service.jobs[queued["token"]].created -= 2
    service._expire_jobs()
    assert not service.poisoned
    assert first["token"] in service.jobs
    assert service.handle({"op": "status", "token": queued["token"]})["drained"]


def test_receiver_rejects_aliasing_and_late_room_reuse():
    service = transport.CompressionReceiver(BufferAdapter(), lambda _: True, Mock(), 1)
    service.expect_pages(7, [2])
    with pytest.raises(RuntimeError, match="another room"):
        service.expect_pages(8, [2])
    service.forget(7)
    with pytest.raises(RuntimeError, match="registered twice"):
        service.expect_pages(7, [3])
    service.expect_pages(8, [2])


def test_receiver_request_ids_and_completed_tokens_cannot_reassign_a_slot():
    buffers = BufferAdapter()
    service = transport.CompressionReceiver(buffers, lambda _: True, Mock(), 1)
    service.expect_pages(7, [2])
    first_header = header()
    ready = service.handle({"op": "prepare", "header": first_header})
    with pytest.raises(RuntimeError, match="reused"):
        service.handle({"op": "prepare", "header": header(part=1)})
    buffers.buffers["payload"][:3] = bytes([0, 2, 4])
    service.handle({"op": "commit", "token": ready["token"]})
    service.advance(7)
    assert not service.free_slots  # Restore alone is not the ACK.
    result = service.handle({"op": "status", "token": ready["token"]})
    service.expect_pages(8, [3])
    newer = service.handle(
        {"op": "prepare", "header": header(room=8, pages=[3], request_id="new")}
    )
    assert service.handle({"op": "commit", "token": ready["token"]}) == result
    assert service.handle({"op": "prepare", "header": first_header}) == result
    assert list(service.jobs) == [newer["token"]] and not service.free_slots


@pytest.mark.parametrize(
    "source_slots,destination_slots", [(1, 2), (2, 1), (2, 2), (4, 2)]
)
def test_twelve_concurrent_sessions_with_independent_slots_and_no_cross_request_data(
    source_slots, destination_slots
):
    producer, consumer = BufferAdapter(source_slots), BufferAdapter(destination_slots)
    service = transport.CompressionReceiver(consumer, lambda _: True, Mock(), 10)
    engine = CopyEngine()
    sender = transport.CompressionSender(producer, engine, 10)
    for room in range(12):
        service.expect_pages(room, [3 * room, 3 * room + 1, 3 * room + 2])
    service.start("127.0.0.1")
    errors = []
    barrier = threading.Barrier(13)

    def send(room):
        try:
            barrier.wait(timeout=5)
            sender.send(
                service.advertisement(),
                "same-d-instance",
                room,
                [64 + 3 * room, 65 + 3 * room, 66 + 3 * room],
                [3 * room, 3 * room + 1, 3 * room + 2],
                5,
            )
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=send, args=(room,), daemon=True) for room in range(12)
    ]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=5)
        # Do not advance any GPU work yet: control must admit all available
        # credits even while the first commit is waiting for the scheduler.
        deadline = time.monotonic() + 5
        target = min(source_slots, destination_slots)
        while time.monotonic() < deadline:
            with service.lock:
                committed = [j for j in service.jobs.values() if j.committed]
                if len(committed) == target:
                    assert len({j.slot_id for j in committed}) == target
                    break
            time.sleep(0.001)
        else:
            pytest.fail("Control RPC blocked behind another room's commit")
        deadline = time.monotonic() + 15
        while any(t.is_alive() for t in threads) and time.monotonic() < deadline:
            for room in range(12):
                service.advance(room)
            time.sleep(0.001)
        assert all(not t.is_alive() for t in threads), "Concurrent transfer deadlocked"
        assert not errors
        assert all(service.is_complete(room) for room in range(12))
        assert len(consumer.restored) == 48
        for pages, part, tokens, source_pages, thread_id in consumer.restored:
            room = pages[0] // 3
            expected = bytes(64 + page for page in pages)
            assert source_pages == expected
            assert tokens == (4 if len(pages) == 2 else 1)
            assert thread_id == threading.get_ident() and part in (0, 1)
            assert room in range(12)
        assert not sender.sessions and not sender.credits.owned
        assert len(sender.credits.free) == source_slots
        assert len(service.free_slots) == destination_slots
        assert not service.jobs and not service.waiters
    finally:
        service.stop()
        for thread in threads:
            thread.join(1)


def test_cancel_after_native_write_drains_without_poisoning_unrelated_requests():
    producer, consumer = BufferAdapter(2), BufferAdapter(2)
    active = {7: True, 8: True}
    service = transport.CompressionReceiver(consumer, lambda _: True, Mock(), 2)
    service.expect_pages(7, [2])
    service.expect_pages(8, [3])
    service.start("127.0.0.1")
    engine = CopyEngine()
    original = engine.batch_transfer_sync

    def write_then_abort(*args):
        result = original(*args)
        active[7] = False
        return result

    engine.batch_transfer_sync = write_then_abort
    sender = transport.CompressionSender(producer, engine, 2, lambda room: active[room])
    errors = []
    try:
        with pytest.raises(transport.TransferCancelled):
            sender.send(service.advertisement(), "peer", 7, [4], [2], 2)
        assert not sender.poisoned and not service.poisoned
        assert not consumer.restored and not sender.credits.owned
        thread = threading.Thread(
            target=lambda: _send_capturing(sender, service, 8, [5], [3], 2, errors),
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 5
        while thread.is_alive() and time.monotonic() < deadline:
            service.advance(8)
            time.sleep(0.001)
        thread.join(1)
        assert not thread.is_alive() and not errors and service.is_complete(8)
    finally:
        service.stop()


def _send_capturing(sender, service, room, src, dst, tokens, errors):
    try:
        sender.send(service.advertisement(), "peer", room, src, dst, tokens)
    except Exception as error:
        errors.append(error)


def test_source_slot_cancel_and_timeout_never_leak_or_poison_credits():
    credits = transport.SlotCredits(1)
    slot = credits.acquire(1, lambda: True)
    with pytest.raises(transport.TransferCancelled):
        credits.acquire(1, lambda: False)
    with pytest.raises(TimeoutError):
        credits.acquire(0.01, lambda: True)
    assert not credits.poisoned and not credits.waiters and credits.owned == {slot}
    credits.release(slot)
    assert credits.acquire(1, lambda: True) == slot


def test_source_credit_waiters_are_fifo_and_quarantine_wakes_them():
    credits = transport.SlotCredits(1)
    initial = credits.acquire(1, lambda: True)
    received = []

    def acquire(index):
        slot = credits.acquire(2, lambda: True)
        received.append(index)
        credits.release(slot)

    threads = []
    for index in range(4):
        thread = threading.Thread(target=acquire, args=(index,), daemon=True)
        thread.start()
        threads.append(thread)
        deadline = time.monotonic() + 1
        while len(credits.waiters) != index + 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(credits.waiters) == index + 1
    credits.release(initial)
    for thread in threads:
        thread.join(3)
    assert received == [0, 1, 2, 3]
    held = credits.acquire(1, lambda: True)
    errors = []

    def wait_for_poison():
        try:
            credits.acquire(2, lambda: True)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=wait_for_poison, daemon=True)
    thread.start()
    credits.quarantine("uncertain DMA")
    thread.join(1)
    assert not thread.is_alive() and len(errors) == 1
    credits.release(held)
    assert not credits.free and credits.owned == {held}


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_queue_sharding_preserves_request_order_and_parallelizes_one_peer(
    enabled,
):
    namespace = {
        "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
        "KVPoll": SimpleNamespace(Failed=0),
        "TraceNullContext": lambda: object(),
        "TransferKVChunk": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    exec(extract_node("MooncakeKVManager", "add_transfer_request"), namespace)
    queues = [object() for _ in range(4)]
    published = []
    manager = SimpleNamespace(
        disaggregation_mode="prefill",
        request_status={room: 2 for room in range(12)},
        check_status=lambda _: 2,
        transfer_infos={room: {"same-peer:1234": object()} for room in range(12)},
        transfer_queues=queues,
        kvserve_sender=object() if enabled else None,
        _publish_source_chunk=lambda queue, chunk: published.append(
            (queue, chunk.room)
        ),
    )
    for _ in range(2):
        for room in range(12):
            namespace["add_transfer_request"](manager, room, [1], slice(0, 1), False)
    assert published[:12] == published[12:]
    assert len({id(queue) for queue, _ in published}) == (4 if enabled else 1)


def cpu_staging_adapter(monkeypatch, target=False):
    """Exercise the real adapter's slot/scratch code with a CPU raw runtime."""
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    instance = object.__new__(adapter.KVServeAdapter)
    instance.device = torch.device("cpu")
    instance.page_size, instance.chunk_pages = 2, 2
    instance.key_dim, instance.value_dim, instance.heads = 4, 4, 2
    instance._staging_lock = threading.Lock()
    instance.keys = [torch.arange(128).reshape(16, 2, 4).to(torch.bfloat16)]
    instance.values = [(instance.keys[0] + 128).clone()]
    if target:
        instance.keys[0].zero_()
        instance.values[0].zero_()
    instance.packed = {"layer": torch.empty(2, 2, 2, 8, dtype=torch.bfloat16)}
    event = SimpleNamespace(synchronize=lambda: None)
    runtime = SimpleNamespace(
        slots=[
            SimpleNamespace(
                arena_a=torch.empty(64, dtype=torch.uint8),
                aux=torch.empty(0, dtype=torch.uint8),
                raw_tail=torch.empty(0, dtype=torch.uint8),
            )
            for _ in range(2)
        ],
        streams=[SimpleNamespace(synchronize=Mock()) for _ in range(2)],
        aux_bytes=lambda *_: 0,
    )

    def encode(slot_id, groups, part, **kwargs):
        # Force overlap if the actual adapter forgets its shared-scratch lock.
        time.sleep(0.01)
        pages = len(groups[0])
        packed = instance.packed["layer"][:pages, :, :, part * 4 : part * 4 + 4]
        data = packed.contiguous().view(torch.uint8).flatten()
        runtime.slots[slot_id].arena_a[: data.numel()].copy_(data)
        return SimpleNamespace(
            event=event,
            codec_bytes=data.numel(),
            u8_bytes=data.numel(),
            raw_tail_addr=0,
            raw_tail_bytes=0,
        )

    def decode(slot_id, groups, part, codec_bytes, u8_bytes, **kwargs):
        data = runtime.slots[slot_id].arena_a[:codec_bytes].view(torch.bfloat16)
        instance.packed["layer"][: len(groups[0]), :, :, part * 4 : part * 4 + 4].copy_(
            data.reshape(len(groups[0]), 2, 2, 4)
        )
        return event

    runtime.encode, runtime.decode = encode, decode
    instance.runtime = runtime
    return instance


def test_actual_adapter_uses_independent_slots_and_fences_shared_scratch(monkeypatch):
    producer = cpu_staging_adapter(monkeypatch)
    consumer = cpu_staging_adapter(monkeypatch, target=True)
    mappings = [([4, 1], [2, 5]), ([6, 3], [0, 7])]
    for part in (0, 1):
        results = {}
        errors = []

        def encode(slot_id):
            try:
                results[slot_id] = producer.encode(
                    mappings[slot_id][0], part, 3, slot_id
                )
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=encode, args=(slot_id,)) for slot_id in (0, 1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        assert not errors and all(not t.is_alive() for t in threads)
        for source_slot, (src, dst) in enumerate(mappings):
            destination_slot = 1 - source_slot
            encoded = results[source_slot]
            assert (
                encoded["payload"][0]
                == producer.runtime.slots[source_slot].arena_a.data_ptr()
            )
            consumer.runtime.slots[destination_slot].arena_a.copy_(
                producer.runtime.slots[source_slot].arena_a
            )
            sizes = {name: encoded[name][1] for name in transport.REGIONS}
            sizes["u8_bytes"] = encoded["u8_bytes"]
            consumer.restore(dst, part, 3, sizes, destination_slot)
            source_indices = adapter.page_token_indices(src, 2, 16)[:3]
            destination_indices = adapter.page_token_indices(dst, 2, 16)[:3]
            source = (producer.keys if part == 0 else producer.values)[0]
            target = (consumer.keys if part == 0 else consumer.values)[0]
            assert torch.equal(source[source_indices], target[destination_indices])
        target = (consumer.keys if part == 0 else consumer.values)[0]
        untouched = [2, 3, 6, 7, 8, 9, 11, 12, 13, 15]
        assert torch.count_nonzero(target[untouched]) == 0
    assert all(s.synchronize.call_count == 2 for s in producer.runtime.streams)


def test_actual_adapter_never_reports_unfenced_reads_or_writes_as_completed(
    monkeypatch,
):
    instance = cpu_staging_adapter(monkeypatch)
    instance.runtime.streams[1].synchronize.side_effect = RuntimeError("device lost")
    with pytest.raises(adapter.UnfencedSourceRead):
        instance.encode([1], 0, 2, 1)
    with pytest.raises(adapter.UnfencedCacheWrite):
        instance.restore([2], 0, 2, dict(payload=32, u8_bytes=32, aux=0, tail=0), 1)


def test_compression_scratch_never_changes_raw_store_buffers_or_addresses(monkeypatch):
    instance = cpu_staging_adapter(monkeypatch)
    namespace = {"torch": torch}
    exec(
        extract_node(
            "DevicePoolEntry",
            path=ROOT / "python/sglang/srt/mem_cache/unified_cache_linker.py",
        ),
        namespace,
    )
    entry = namespace["DevicePoolEntry"](
        name="KV",
        indices_from_pool="KV",
        device_pool=instance,
        components=[instance.keys, instance.values],
        layer_mapping={3: 0},
        page_size=2,
        rows_are_pages=False,
    )
    indices = torch.tensor([8, 9, 2, 3])
    ptrs, sizes = entry.get_page_buffer_meta(indices)
    stored_bytes = [ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)]
    original = instance.runtime.encode

    def destructive_codec(*args, **kwargs):
        # A codec may alter the tensors it owns in place. Those must be the
        # staging tensors, never the canonical buffers registered with Store.
        instance.packed["layer"].add_(8)
        return original(*args, **kwargs)

    instance.runtime.encode = destructive_codec
    for slot in (0, 1):
        for part in (0, 1):
            instance.encode([4, 1], part, 3, slot)
    assert entry.get_page_buffer_meta(indices) == (ptrs, sizes)
    assert [
        ctypes.string_at(ptr, size) for ptr, size in zip(ptrs, sizes)
    ] == stored_bytes


def scheduler_result_method():
    namespace = {
        "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
        "flush_trace_batch": Mock(),
    }
    exec(
        extract_node(
            "Scheduler",
            "process_batch_result",
            ROOT / "python/sglang/srt/managers/scheduler.py",
        ),
        namespace,
    )
    return namespace["process_batch_result"]


def result_scheduler(*, enabled=True, external=True):
    mode = SimpleNamespace(
        is_extend=lambda: True,
        is_decode=lambda: False,
        is_prebuilt=lambda: False,
        is_idle=lambda: False,
    )
    batch = SimpleNamespace(forward_mode=mode, reqs=[], is_dllm=lambda: False)
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_unified_cache_external_linker=external,
            kvserve_compression="raw" if enabled else None,
        ),
        disaggregation_mode="prefill",
        tree_cache=SimpleNamespace(drain_linker_loads=Mock(return_value=[])),
        _mark_failed_linker_loads=Mock(),
        publish_load_snapshot=Mock(),
        process_batch_result_disagg_prefill=Mock(),
        batch_result_processor=Mock(),
        _record_step_counters=Mock(),
        metrics_reporter=Mock(),
        enable_fpm=False,
        _maybe_clear_mm_inputs=Mock(),
        maybe_send_health_check_signal=Mock(),
    )
    return scheduler, batch


@pytest.mark.parametrize("success", [False, True])
def test_storage_load_verdict_precedes_any_compression_publication(success):
    method = scheduler_result_method()
    scheduler, batch = result_scheduler()
    backend_started, allow_backend_finish = threading.Event(), threading.Event()
    published = []
    marks = []
    errors = []

    def drain(*, wait_for_completion=False):
        assert wait_for_completion
        backend_started.set()
        assert allow_backend_finish.wait(3)
        return [] if success else ["request"]

    def mark(_batch, *, failed_rids=None):
        marks.append(failed_rids)

    def publish(*_):
        assert marks
        if not marks[-1]:
            published.append("compressed-KV")

    scheduler.tree_cache.drain_linker_loads.side_effect = drain
    scheduler._mark_failed_linker_loads.side_effect = mark
    scheduler.process_batch_result_disagg_prefill.side_effect = publish

    def process():
        try:
            method(scheduler, batch, object())
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=process, daemon=True)
    thread.start()
    try:
        assert backend_started.wait(1)
        assert not published and not marks
        allow_backend_finish.set()
        thread.join(3)
        assert not thread.is_alive() and not errors
        assert marks == [set() if success else {"request"}]
        assert published == (["compressed-KV"] if success else [])
    finally:
        allow_backend_finish.set()
        thread.join(1)


@pytest.mark.parametrize(
    "external,enabled", [(False, True), (True, False), (False, False)]
)
def test_disabled_or_nonstorage_result_processing_keeps_existing_path(
    external, enabled
):
    scheduler, batch = result_scheduler(enabled=enabled, external=external)
    scheduler_result_method()(scheduler, batch, object())
    scheduler.tree_cache.drain_linker_loads.assert_not_called()
    if external:
        scheduler._mark_failed_linker_loads.assert_called_once_with(
            batch, failed_rids=None
        )
    else:
        scheduler._mark_failed_linker_loads.assert_not_called()


def test_precollected_storage_failure_consensus_is_not_replaced():
    scheduler, batch = result_scheduler()
    scheduler_result_method()(scheduler, batch, object(), linker_failed_rids={"failed"})
    scheduler.tree_cache.drain_linker_loads.assert_not_called()
    scheduler._mark_failed_linker_loads.assert_called_once_with(
        batch, failed_rids={"failed"}
    )


def test_storage_load_success_on_this_rank_still_obeys_failed_peer_verdict():
    namespace = {"torch": torch, "logger": Mock()}
    path = ROOT / "python/sglang/srt/mem_cache/unified_radix_cache.py"
    for name in ("drain_linker_loads", "_collect_failed_linker_loads"):
        exec(extract_node("UnifiedRadixCache", name, path), namespace)
    cache = SimpleNamespace(
        linker=SimpleNamespace(
            wait_for_pending_loads=Mock(),
            num_completed_loads=lambda: 1,
            take_completed_loads=lambda _: [True],
            commit_completed_loads=lambda successes: (
                [] if all(successes) else ["failed"]
            ),
        ),
        _failed_linker_rids=[],
    )

    def reduce_min(value, operation):
        assert operation == torch.distributed.ReduceOp.MIN
        if value.ndim == 1:
            value.zero_()  # The other TP rank failed its external KV read.

    cache._all_reduce_attn_groups = reduce_min
    cache._collect_failed_linker_loads = lambda n: namespace[
        "_collect_failed_linker_loads"
    ](cache, n)
    cache.drain_linker_loads = lambda **kwargs: namespace["drain_linker_loads"](
        cache, **kwargs
    )
    scheduler, batch = result_scheduler()
    scheduler.tree_cache = cache
    scheduler_result_method()(scheduler, batch, object())
    cache.linker.wait_for_pending_loads.assert_called_once()
    scheduler._mark_failed_linker_loads.assert_called_once_with(
        batch, failed_rids={"failed"}
    )


@pytest.mark.parametrize(
    "option", ["enable_unified_cache_external_linker", "enable_hierarchical_cache"]
)
def test_external_cache_prefix_reuse_is_deferred_to_post_forward_send(option):
    namespace = {
        "envs": SimpleNamespace(
            SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=SimpleNamespace(
                get=lambda: True
            )
        ),
    }
    exec(
        extract_node(
            "SchedulerDisaggregationPrefillMixin",
            "maybe_send_cached_prefix_chunk",
            SOURCE / "prefill.py",
        ),
        namespace,
    )
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            kvserve_compression="default",
            enable_unified_cache_external_linker=False,
        ),
        send_kv_chunk=Mock(),
    )
    setattr(scheduler.server_args, option, True)
    request = SimpleNamespace(prefix_indices=torch.arange(8))
    namespace["maybe_send_cached_prefix_chunk"](scheduler, request)
    scheduler.send_kv_chunk.assert_not_called()
    assert torch.equal(request.prefix_indices, torch.arange(8))


def test_nonstorage_cached_prefix_early_send_remains_enabled():
    namespace = {
        "envs": SimpleNamespace(
            SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=SimpleNamespace(
                get=lambda: True
            )
        ),
    }
    exec(
        extract_node(
            "SchedulerDisaggregationPrefillMixin",
            "maybe_send_cached_prefix_chunk",
            SOURCE / "prefill.py",
        ),
        namespace,
    )
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            kvserve_compression="raw", enable_unified_cache_external_linker=False
        ),
        enable_staging=False,
        enable_overlap=False,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=2),
        send_kv_chunk=Mock(),
    )
    request = SimpleNamespace(
        prefix_indices=torch.arange(8),
        host_hit_length=0,
        pending_bootstrap=False,
        start_send_idx=0,
    )
    namespace["maybe_send_cached_prefix_chunk"](scheduler, request)
    scheduler.send_kv_chunk.assert_called_once_with(
        request, last_chunk=False, end_idx=8
    )


@pytest.mark.parametrize("tokens", [0, 2, 64])
def test_direct_linker_hits_are_reported_as_external_storage_not_local_host(tokens):
    namespace = {}
    exec(
        extract_node(
            "UnifiedRadixCache",
            "init_load_back",
            ROOT / "python/sglang/srt/mem_cache/unified_radix_cache.py",
        ),
        namespace,
    )
    indices = torch.arange(tokens)
    req = SimpleNamespace(rid="request", storage_hit_length=0)
    linker = SimpleNamespace(has_hit=lambda _: True, load_back=lambda _: (indices, 17))
    cache = SimpleNamespace(buffer_pipeline=None, linker=linker)
    params = SimpleNamespace(best_match_node=1, mem_quota=None, req=req)
    result, node = namespace["init_load_back"](cache, params)
    assert result is indices and node == 17 and req.storage_hit_length == tokens
    batch_tree = ast.parse(
        (ROOT / "python/sglang/srt/managers/schedule_batch.py").read_text(
            encoding="utf-8"
        )
    )
    split = next(
        n
        for n in batch_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "split_cached_prefix_by_tier"
    )
    exec(
        compile(ast.Module(body=[split], type_ignores=[]), "cache_split", "exec"),
        namespace,
    )
    assert namespace["split_cached_prefix_by_tier"](
        tokens + 8, tokens, req.storage_hit_length
    ) == (
        8,
        0,
        tokens,
    )
    exec(
        extract_node(
            "SchedulerOutputStreamer",
            "get_cached_tokens_details",
            ROOT / "python/sglang/srt/managers/scheduler_components/output_streamer.py",
        ),
        namespace,
    )
    # D need not have a local Store backend to report P's storage-hit metadata.
    req.cached_tokens_device, req.cached_tokens_host, req.cached_tokens_storage = (
        8,
        0,
        tokens,
    )
    req.cached_tokens = 8 + tokens
    details = namespace["get_cached_tokens_details"](
        SimpleNamespace(enable_hicache_storage=lambda: False), req
    )
    assert details.get("storage", 0) == tokens and details["host"] == 0
