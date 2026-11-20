import ctypes
import threading
import types
import unittest
from queue import Queue
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeDirectLinkerSessions(CustomTestCase):
    def _layout_linker(self, *, packed=True, bad_result=False):
        linker = self._session_linker(prefetch=False)
        linker.read_layout_enabled = True
        linker.num_layers = 2
        buffers = [
            [torch.zeros((4, 2), dtype=torch.uint8) for _ in range(2)] for _ in range(2)
        ]
        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=buffers,
            layer_mapping={0: 0, 1: 1},
            page_size=1,
            rows_are_pages=True,
            packed=packed,
        )
        pool.get_prepared_layer_range_meta = Mock(
            side_effect=AssertionError("ReadLayout must expand addresses in C++")
        )
        linker.pools = {PoolName.KV: pool}
        linker.layer_done_counter = Mock()
        linker.storage._get_hybrid_page_component_keys = lambda keys, _: (
            keys if packed else [f"{key}:{i}" for key in keys for i in range(2)],
            1 if packed else 2,
        )
        linker.storage._tag_keys = lambda keys: keys
        linker.storage.store.batch_get_session_start.return_value = (
            [0, 0] if packed else [0, 0, 0, 0]
        )
        linker.storage.store.batch_get_session_end.return_value = 0
        linker.storage.store.all_get_sessions_memory.return_value = True
        linker.storage.store.batch_get_into_layouts.return_value = (
            [[-707]] if bad_result else [[4, 4]] if packed else [[2, 2, 2, 2]]
        )
        transfers = [
            [
                PoolTransfer(
                    name=PoolName.KV, keys=["a", "b"], host_indices=torch.tensor([3, 1])
                )
            ]
        ]
        native = types.SimpleNamespace(
            ReadComponent=lambda **kw: types.SimpleNamespace(**kw),
            ReadLayout=lambda **kw: types.SimpleNamespace(**kw),
            ReadKeyMapping=types.SimpleNamespace(
                PER_ROW="row", PER_ROW_COMPONENT="component"
            ),
        )
        return linker, transfers, native

    def test_read_layout_uses_native_strides_and_preserves_key_mapping(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                linker, transfers, native = self._layout_linker(packed=packed)
                with patch.dict("sys.modules", {"mooncake.store": native}):
                    self.assertTrue(linker.load_layer_wise(0, transfers))
                store = linker.storage.store
                self.assertEqual(store.batch_get_into_layouts.call_count, 2)
                store.batch_get_into_multi_buffer_ranges.assert_not_called()
                store.batch_get_session_end.assert_called_once()
                for layer, invocation in enumerate(
                    store.batch_get_into_layouts.call_args_list
                ):
                    layout = invocation.args[0][0]
                    self.assertEqual(layout.rows, [3, 1])
                    self.assertEqual(
                        layout.key_mapping, "row" if packed else "component"
                    )
                    self.assertEqual(
                        [c.source_offset for c in layout.components],
                        [layer * 2, 4 + layer * 2] if packed else [layer * 2] * 2,
                    )
                self.assertEqual(
                    linker.layer_done_counter.complete.call_args_list,
                    [call(0, 0), call(0, 1)],
                )

    def test_read_layout_failure_releases_sessions_without_exposing_layer(self):
        linker, transfers, native = self._layout_linker(bad_result=True)
        with patch.dict("sys.modules", {"mooncake.store": native}), self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="ERROR",
        ):
            self.assertFalse(linker.load_layer_wise(0, transfers))
        linker.storage.store.batch_get_session_end.assert_called_once()
        linker.layer_done_counter.fail.assert_called_once()
        linker.layer_done_counter.complete.assert_not_called()

    def test_read_layout_skips_unmapped_layers(self):
        linker, transfers, native = self._layout_linker()
        linker.pools[PoolName.KV].layer_mapping = {1: 1}
        with patch.dict("sys.modules", {"mooncake.store": native}):
            self.assertTrue(linker.load_layer_wise(0, transfers))
        linker.storage.store.batch_get_into_layouts.assert_called_once()
        self.assertEqual(
            linker.layer_done_counter.complete.call_args_list, [call(0, 0), call(0, 1)]
        )

    def _page_layout_linker(self, packed):
        linker = self._session_linker(memory=False, page_wise=True)
        linker.read_layout_enabled = True
        linker.num_layers = 3
        linker.layer_done_counter = LayerWiseLoadCounter(3)
        linker.layer_done_counter.update_producer()
        linker.pools = {}
        buffers = []
        expected_ranges = []
        transfers = []
        for name in (PoolName.KV, PoolName.SWA):
            pool = DevicePoolEntry.__new__(DevicePoolEntry)
            pool.packed = packed
            pool.layer_mapping = {0: [0, 1], 2: 2}
            pool.buffer_meta = []
            pool._component_offsets = []
            offset = 0
            for component in range(2):
                if not packed:
                    offset = 0
                meta, offsets = [], []
                for size in (2 + component, 3 + component, 4 + component):
                    buffer = ctypes.create_string_buffer(4 * size)
                    buffers.append(buffer)
                    meta.append((ctypes.addressof(buffer), size, size))
                    offsets.append(offset)
                    offset += size
                pool.buffer_meta.append(meta)
                pool._component_offsets.append(offsets)
            pool.prepare_locations = list
            pool.get_prepared_layer_range_meta = Mock(
                side_effect=AssertionError("ReadLayout must expand addresses in C++")
            )
            linker.pools[name] = pool
            transfers.append(
                PoolTransfer(name=name, keys=["a", "a"], host_indices=[1, 3])
            )
            for row in (1, 3):
                for component, (meta, offsets) in enumerate(
                    zip(pool.buffer_meta, pool._component_offsets)
                ):
                    key = f"{name}:a" if packed else f"{name}:a:{component}"
                    linker.objects[key] = bytes(range(64))
                    for (base, stride, size), offset in zip(meta, offsets):
                        expected_ranges.append((key, base + row * stride, size, offset))
        linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
            (
                [f"{transfer.name}:{key}" for key in keys]
                if packed
                else [f"{transfer.name}:{key}:{i}" for key in keys for i in range(2)]
            ),
            1 if packed else 2,
        )
        native = types.SimpleNamespace(
            ReadComponent=lambda **kw: types.SimpleNamespace(**kw),
            ReadLayout=lambda **kw: types.SimpleNamespace(**kw),
            ReadKeyMapping=types.SimpleNamespace(PER_ROW="row"),
        )
        actual_ranges = []

        def read(layouts):
            self.assertTrue(
                all(not f.done() for f in linker.layer_done_counter.futures[0])
            )
            results = []
            for layout in layouts:
                self.assertEqual(layout.key_mapping, "row")
                self.assertEqual(len(layout.keys), len(layout.rows))
                counts = []
                for key, row in zip(layout.keys, layout.rows):
                    self.assertIn(key, linker.active)
                    count = 0
                    for item in layout.components:
                        address = item.destination_base + row * item.destination_stride
                        actual_ranges.append(
                            (key, address, item.byte_count, item.source_offset)
                        )
                        data = linker.prefetched.get(key, linker.objects[key])
                        ctypes.memmove(
                            address,
                            data[
                                item.source_offset : item.source_offset
                                + item.byte_count
                            ],
                            item.byte_count,
                        )
                        count += item.byte_count
                    counts.append(count)
                results.append(counts)
            return results

        linker.storage.store.batch_get_into_layouts = Mock(side_effect=read)
        return linker, [transfers], native, buffers, expected_ranges, actual_ranges

    def test_page_layout_batches_all_layers_pools_and_duplicate_targets(self):
        for packed in (True, False):
            with self.subTest(packed=packed):
                linker, transfers, native, buffers, expected, actual = (
                    self._page_layout_linker(packed)
                )
                with patch.dict("sys.modules", {"mooncake.store": native}):
                    self.assertTrue(linker.load_layer_wise(0, transfers))
                store = linker.storage.store
                store.batch_get_into_layouts.assert_called_once()
                layouts = store.batch_get_into_layouts.call_args.args[0]
                self.assertEqual(len(layouts), 2 if packed else 4)
                store.batch_get_into_multi_buffer_ranges.assert_not_called()
                self.assertCountEqual(actual, expected)
                for key, address, size, offset in expected:
                    self.assertEqual(
                        ctypes.string_at(address, size),
                        linker.objects[key][offset : offset + size],
                    )
                self.assertTrue(
                    all(
                        f.done() and f.exception() is None
                        for f in linker.layer_done_counter.futures[0]
                    )
                )
                self.assertFalse(linker.active)

    def test_page_layout_reuses_claimed_prefetch_and_waits_for_session_cleanup(self):
        linker, transfers, native, buffers, expected, _ = self._page_layout_linker(
            False
        )
        linker.host_prefetch_enabled = True
        self.assertTrue(linker.prepare_load("prefetch", transfers[0]))
        linker.prefetched.update(linker.objects)
        linker.host_prefetch_entries["rid"] = {
            "state": "dfs_prefetched",
            "session_rid": "prefetch",
        }
        self.assertTrue(linker.revalidate_host_prefetch("rid"))
        self.assertTrue(linker.claim_ready_host_prefetch("rid"))
        linker.objects = dict.fromkeys(linker.objects, b"x" * 64)
        store = linker.storage.store
        end = store.batch_get_session_end.side_effect

        def end_checked(keys):
            self.assertTrue(
                all(not f.done() for f in linker.layer_done_counter.futures[0])
            )
            return end(keys)

        store.batch_get_session_end.side_effect = end_checked
        with patch.dict("sys.modules", {"mooncake.store": native}):
            self.assertTrue(linker.load_layer_wise(0, transfers, request_ids=["rid"]))
        store.batch_get_session_start.assert_not_called()
        store.batch_get_session_start_with_sources.assert_called_once()
        store.batch_get_session_refresh.assert_called_once()
        for _, address, size, offset in expected:
            self.assertEqual(
                ctypes.string_at(address, size),
                bytes(range(64))[offset : offset + size],
            )
        self.assertFalse(linker.prepared_load_sessions)
        self.assertFalse(linker.active)

    def test_page_layout_failures_wake_waiters_without_publishing_ready(self):
        for failure in (
            None,
            [],
            [[1]],
            [[9, 9], [12, 12], [9, 9], [11, 12]],
            "read",
            "cleanup",
            "injection",
        ):
            with self.subTest(failure=failure):
                linker, transfers, native, buffers, _, _ = self._page_layout_linker(
                    False
                )
                store = linker.storage.store
                if failure == "read":
                    store.batch_get_into_layouts.side_effect = RuntimeError(
                        "read failed"
                    )
                elif failure == "cleanup":
                    store.batch_get_session_end.side_effect = RuntimeError("end failed")
                elif failure != "injection":
                    store.batch_get_into_layouts.side_effect = None
                    store.batch_get_into_layouts.return_value = failure
                maybe_fail = Mock(
                    side_effect=(
                        RuntimeError("injected") if failure == "injection" else None
                    )
                )
                with (
                    patch.dict("sys.modules", {"mooncake.store": native}),
                    patch(
                        "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.arm_load_failure_injection",
                        return_value=maybe_fail,
                    ),
                    self.assertLogs(level="ERROR"),
                ):
                    self.assertFalse(linker.load_layer_wise(0, transfers))
                self.assertTrue(
                    all(
                        f.done() and f.exception()
                        for f in linker.layer_done_counter.futures[0]
                    )
                )
                store.batch_get_session_end.assert_called_once()
                if failure == "injection":
                    maybe_fail.assert_called_once_with("aggregated", "complete_page")
                    store.batch_get_into_layouts.assert_not_called()

    def test_range_layers_publish_before_next_layer_but_last_waits_for_cleanup(self):
        linker = self._session_linker()
        store = linker.storage.store
        read = store.batch_get_into_multi_buffer_ranges.side_effect
        end = store.batch_get_session_end.side_effect

        def read_checked(*args):
            futures = linker.layer_done_counter.futures[0]
            self.assertEqual(
                futures[0].done(),
                store.batch_get_into_multi_buffer_ranges.call_count == 2,
            )
            self.assertFalse(futures[-1].done())
            return read(*args)

        def end_checked(keys):
            self.assertTrue(linker.layer_done_counter.futures[0][0].done())
            self.assertFalse(linker.layer_done_counter.futures[0][-1].done())
            return end(keys)

        store.batch_get_into_multi_buffer_ranges.side_effect = read_checked
        store.batch_get_session_end.side_effect = end_checked
        self.assertTrue(linker.load_layer_wise(0, self._transfers()))
        self.assertTrue(linker.layer_done_counter.futures[0][-1].done())

    def _session_linker(self, *, prefetch=False, memory=True, page_wise=False):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_enabled = prefetch
        linker.enable_page_wise_load = page_wise
        # Adaptation must not consult this old threshold.
        linker.page_wise_load_threshold = 10000
        linker.read_layout_enabled = False
        linker.tp_rank = 0
        linker.num_layers = 2
        linker.session_lock = threading.Lock()
        linker.host_prefetch_lock = threading.Lock()
        linker.host_prefetch_entries = {}
        linker.prepared_load_sessions = {}
        linker.session_refcounts = {}
        linker.session_sources = {}
        linker.layer_done_counter = LayerWiseLoadCounter(2)
        linker.layer_done_counter.update_producer()
        linker.buffer = ctypes.create_string_buffer(64)
        linker.objects = {"a": b"abcdefghijklmnop", "b": b"ABCDEFGHIJKLMNOP"}
        linker.prefetched = {}
        linker.active = set()
        linker.events = []

        def start(keys):
            linker.active.update(keys)
            linker.events.append("start")
            return [0] * len(keys)

        def end(keys):
            linker.events.append("end")
            linker.active.difference_update(keys)
            for key in keys:
                linker.prefetched.pop(key, None)
            return 0

        def read(keys, pointers, sizes, offsets):
            linker.events.append("read")
            for key, addresses, counts, positions in zip(
                keys, pointers, sizes, offsets
            ):
                self.assertIn(key, linker.active)
                data = linker.prefetched.get(key, linker.objects[key])
                for address, count, offset in zip(addresses, counts, positions):
                    ctypes.memmove(address, data[offset : offset + count], count)
            return [sum(counts) for counts in sizes]

        def meta(locations, layer):
            return (
                [
                    [ctypes.addressof(linker.buffer) + row * 8 + layer * 4]
                    for row in locations
                ],
                [[4] for row in locations],
                [[layer * 8 + (row % 2) * 4] for row in locations],
            )

        pool = types.SimpleNamespace(
            prepare_locations=list, get_prepared_layer_range_meta=meta
        )
        linker.pools = {PoolName.KV: pool, PoolName.SWA: pool}
        linker.pool_group = types.SimpleNamespace(
            resolve_transfers=lambda transfers, **kw: transfers
        )

        def source(key):
            return (
                memory[key]
                if isinstance(memory, dict)
                else "memory" if memory else "dfs"
            )

        store = types.SimpleNamespace(
            batch_get_session_start=Mock(side_effect=start),
            batch_get_session_start_with_sources=Mock(
                side_effect=lambda keys: (
                    start(keys),
                    [source(key) for key in keys],
                )
            ),
            batch_get_session_refresh=Mock(side_effect=lambda keys: [0] * len(keys)),
            batch_get_session_end=Mock(side_effect=end),
            batch_get_into_multi_buffer_ranges=Mock(side_effect=read),
            all_get_sessions_memory=Mock(
                side_effect=lambda keys: bool(keys)
                and all(source(key) == "memory" for key in keys)
            ),
        )
        linker.storage = types.SimpleNamespace(
            store=store,
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=list,
            failed_get_cache=Mock(),
        )
        return linker

    def _transfers(self):
        return [[PoolTransfer(name=PoolName.KV, keys=["a", "b"], host_indices=[1, 2])]]

    def test_read_modes_route_to_one_helper(self):
        helpers = (
            "_load_layer_wise_ranges",
            "_load_with_read_layout",
            "_load_page_wise_ranges",
            "_load_page_wise_with_read_layout",
        )
        for page_wise in (False, True):
            for read_layout in (False, True):
                with self.subTest(page_wise=page_wise, read_layout=read_layout):
                    linker = self._session_linker(memory=False, page_wise=page_wise)
                    linker.read_layout_enabled = read_layout
                    for helper in helpers:
                        setattr(linker, helper, Mock())
                    self.assertTrue(linker.load_layer_wise(0, self._transfers()))
                    selected = helpers[2 * int(page_wise) + int(read_layout)]
                    for helper in helpers:
                        if helper == selected:
                            getattr(linker, helper).assert_called_once()
                        else:
                            getattr(linker, helper).assert_not_called()

    def test_selected_sources_choose_mode_without_threshold(self):
        for memory, page_wise, calls in (
            (True, True, 2),
            ({"a": "memory", "b": "dfs"}, True, 1),
            (False, False, 2),
        ):
            with self.subTest(memory=memory, page_wise=page_wise):
                linker = self._session_linker(memory=memory, page_wise=page_wise)
                store = linker.storage.store
                # The presence of other replicas is irrelevant: use this verdict.
                self.assertTrue(linker.load_layer_wise(0, self._transfers()))
                self.assertEqual(
                    store.batch_get_into_multi_buffer_ranges.call_count, calls
                )
                if page_wise:
                    store.all_get_sessions_memory.assert_called_once_with(["a", "b"])
                else:
                    store.all_get_sessions_memory.assert_not_called()
                self.assertEqual(bytes(linker.buffer)[8:24], b"efghmnopABCDIJKL")

    def test_unknown_source_uses_pagewise(self):
        linker = self._session_linker(page_wise=True)
        del linker.storage.store.all_get_sessions_memory
        self.assertTrue(linker.load_layer_wise(0, self._transfers()))
        self.assertEqual(
            linker.storage.store.batch_get_into_multi_buffer_ranges.call_count, 1
        )

    def test_duplicate_keys_keep_destinations_and_source_offsets(self):
        for page_wise in (False, True):
            with self.subTest(page_wise=page_wise):
                linker = self._session_linker(memory=False, page_wise=page_wise)
                transfers = [
                    [
                        PoolTransfer(
                            name=PoolName.KV, keys=["a", "a"], host_indices=[1, 2]
                        )
                    ]
                ]
                self.assertTrue(linker.load_layer_wise(0, transfers))
                linker.storage.store.batch_get_session_start.assert_called_once_with(
                    ["a"]
                )
                self.assertEqual(bytes(linker.buffer)[8:24], b"efghmnopabcdijkl")
                for (
                    invocation
                ) in (
                    linker.storage.store.batch_get_into_multi_buffer_ranges.call_args_list
                ):
                    self.assertEqual(invocation.args[0], ["a", "a"])

    def test_prefetched_session_is_claimed_and_data_survives_loading(self):
        linker = self._session_linker(prefetch=True, memory=False, page_wise=True)
        transfers = self._transfers()[0]
        self.assertTrue(linker.prepare_load("prefetch", transfers))
        linker.prefetched.update(linker.objects)
        linker.host_prefetch_entries["rid"] = {
            "state": "dfs_prefetched",
            "session_rid": "prefetch",
        }
        self.assertTrue(linker.revalidate_host_prefetch("rid"))
        self.assertTrue(linker.claim_ready_host_prefetch("rid"))
        linker.objects = dict.fromkeys(linker.objects, b"x" * 16)
        self.assertTrue(linker.load_layer_wise(0, [transfers], request_ids=["rid"]))
        self.assertEqual(bytes(linker.buffer)[8:24], b"efghmnopABCDIJKL")
        linker.storage.store.batch_get_session_start.assert_not_called()
        linker.storage.store.batch_get_session_start_with_sources.assert_called_once()
        linker.storage.store.batch_get_session_refresh.assert_called_once()
        self.assertFalse(linker.active)
        self.assertFalse(linker.prepared_load_sessions)

    def test_shared_key_ends_only_after_last_holder(self):
        linker = self._session_linker(prefetch=True)
        transfers = self._transfers()[0]
        self.assertTrue(linker.prepare_load("one", transfers))
        self.assertTrue(linker.prepare_load("two", transfers))
        self.assertTrue(linker.load_layer_wise(0, [transfers], request_ids=["one"]))
        linker.storage.store.batch_get_session_end.assert_not_called()
        self.assertEqual(linker.session_refcounts, {"a": 1, "b": 1})
        self.assertTrue(linker.abort_prepared_load("two"))
        linker.storage.store.batch_get_session_end.assert_called_once_with(["a", "b"])
        self.assertFalse(linker.active)

    def test_partial_start_cleans_successes_and_wakes_all_waiters(self):
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                linker = self._session_linker(prefetch=prefetch)

                def partial(keys):
                    linker.active.add("a")
                    return [0, -702]

                store = linker.storage.store
                store.batch_get_session_start.side_effect = partial
                if prefetch:
                    store.batch_get_session_start_with_sources.side_effect = (
                        lambda keys: (partial(keys), ["memory", "unknown"])
                    )
                    self.assertFalse(linker.prepare_load("rid", self._transfers()[0]))
                with self.assertLogs(level="ERROR"):
                    self.assertFalse(
                        linker.load_layer_wise(
                            0,
                            self._transfers(),
                            request_ids=["rid"] if prefetch else None,
                        )
                    )
                self.assertFalse(linker.active)
                store.batch_get_session_end.assert_called_once_with(["a"])
                self.assertTrue(
                    all(
                        f.done() and f.exception()
                        for f in linker.layer_done_counter.futures[0]
                    )
                )
                store.batch_get_into_multi_buffer_ranges.assert_not_called()

    def test_start_exception_releases_partial_sessions(self):
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                linker = self._session_linker(prefetch=prefetch)

                def start(keys):
                    linker.active.add("a")
                    raise RuntimeError("partial start exception")

                store = linker.storage.store
                store.batch_get_session_start.side_effect = start
                store.batch_get_session_start_with_sources.side_effect = start
                with self.assertLogs():
                    if prefetch:
                        self.assertFalse(
                            linker.prepare_load("rid", self._transfers()[0])
                        )
                    else:
                        self.assertFalse(linker.load_layer_wise(0, self._transfers()))
                self.assertFalse(linker.active)
                self.assertFalse(linker.session_refcounts)
                store.batch_get_session_end.assert_called_once_with(["a", "b"])

    def test_cancel_releases_only_its_own_reference(self):
        linker = self._session_linker(prefetch=True)
        transfers = self._transfers()[0]
        self.assertTrue(linker.prepare_load("cancelled", transfers))
        self.assertTrue(linker.prepare_load("live", transfers))
        linker.pending_loads = {"cancelled": transfers}
        self.assertTrue(linker.cancel_queued_load("cancelled"))
        linker.storage.store.batch_get_session_end.assert_not_called()
        self.assertTrue(linker.load_layer_wise(0, [transfers], request_ids=["live"]))
        linker.storage.store.batch_get_session_end.assert_called_once_with(["a", "b"])

    def test_device_event_failure_releases_prepared_sessions_and_reports_failure(self):
        linker = self._session_linker(prefetch=True)
        transfers = self._transfers()[0]
        self.assertTrue(linker.prepare_load("rid", transfers))
        linker.load_queue, linker.completed_loads = Queue(), Queue()
        event = Mock()
        event.synchronize.side_effect = RuntimeError("event failed")
        linker.load_queue.put((0, {"rid": transfers}, event))
        linker.load_queue.put(None)
        with self.assertLogs(level="ERROR"):
            linker.load_thread_func()
        self.assertEqual(linker.pop_completed_load(), (["rid"], False))
        self.assertFalse(linker.active)
        self.assertFalse(linker.prepared_load_sessions)
        self.assertTrue(
            all(f.exception() for f in linker.layer_done_counter.futures[0])
        )

    def test_read_failure_and_cleanup_failure_never_report_success(self):
        for prefetch in (False, True):
            for page_wise in (False, True):
                for failure in ("read", "end_code", "end_exception"):
                    with self.subTest(
                        prefetch=prefetch, page_wise=page_wise, failure=failure
                    ):
                        linker = self._session_linker(
                            prefetch=prefetch, memory=False, page_wise=page_wise
                        )
                        store = linker.storage.store
                        if prefetch:
                            self.assertTrue(
                                linker.prepare_load("rid", self._transfers()[0])
                            )
                        if failure == "read":
                            store.batch_get_into_multi_buffer_ranges.side_effect = (
                                RuntimeError("read failed")
                            )
                        elif failure == "end_code":
                            store.batch_get_session_end.side_effect = None
                            store.batch_get_session_end.return_value = -1
                        else:
                            store.batch_get_session_end.side_effect = RuntimeError(
                                "end failed"
                            )
                        with self.assertLogs(level="ERROR"):
                            self.assertFalse(
                                linker.load_layer_wise(
                                    0,
                                    self._transfers(),
                                    request_ids=["rid"] if prefetch else None,
                                )
                            )
                        futures = linker.layer_done_counter.futures[0]
                        self.assertTrue(all(f.done() for f in futures))
                        self.assertIsNotNone(futures[-1].exception())
                        if page_wise or failure == "read":
                            self.assertTrue(all(f.exception() for f in futures))
                        store.batch_get_session_end.assert_called_once()
                        self.assertFalse(linker.prepared_load_sessions)
                        self.assertFalse(linker.session_refcounts)

    def test_layer_ready_requires_all_pools_and_batch_completion_is_reported(self):
        linker = self._session_linker()
        linker.storage.store.batch_get_into_multi_buffer_ranges.side_effect = [
            [4, 4],
            [-1, -1],
        ]
        transfers = self._transfers()[0] + [
            PoolTransfer(name=PoolName.SWA, keys=["a", "b"], host_indices=[3, 4])
        ]
        linker.load_queue = Queue()
        linker.completed_loads = Queue()
        linker.load_queue.put((0, {"rid": transfers}, Mock()))
        linker.load_queue.put(None)
        with self.assertLogs(level="ERROR"):
            linker.load_thread_func()
        self.assertEqual(linker.pop_completed_load(), (["rid"], False))
        self.assertEqual(linker.load_queue.unfinished_tasks, 0)
        self.assertTrue(
            all(f.exception() for f in linker.layer_done_counter.futures[0])
        )

    def test_failure_is_deferred_past_model_forward(self):
        counter = LayerWiseLoadCounter(2)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, RuntimeError("read failed"))
        with self.assertLogs(level="ERROR") as logs:
            counter.wait_until(0)
            counter.wait_until(1)
        self.assertEqual(len(logs.output), 1)
        self.assertNotIn(index, counter.futures)

    def test_host_prefetch_submission_does_not_prepare_inline(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_enabled = True
        linker.host_prefetch_limit = 8
        linker.host_prefetch_generation = 0
        linker.host_prefetch_lock = threading.Lock()
        linker.host_prefetch_entries = {}
        linker.host_prefetch_queue = Queue()
        linker.prepare_load = Mock(
            side_effect=AssertionError(
                "session preparation must run on the prefetch worker"
            )
        )
        transfer = PoolTransfer(name=PoolName.KV, keys=["page-a"])

        self.assertTrue(linker.submit_host_prefetch("rid", [transfer]))
        linker.prepare_load.assert_not_called()
        self.assertEqual(linker.get_host_prefetch_status("rid"), "queued")
        queued_rid, queued_transfers = linker.host_prefetch_queue.get_nowait()
        self.assertEqual(queued_rid, "rid")
        self.assertEqual(queued_transfers, [transfer])
        linker.host_prefetch_queue.task_done()


if __name__ == "__main__":
    unittest.main()
