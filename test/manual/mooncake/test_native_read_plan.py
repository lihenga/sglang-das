"""CPU/TCP native contract smoke with a private in-process Master.

Use the matching wheel on PYTHONPATH, e.g. python -S this_file.py. This neither
connects to a deployed Master nor establishes GPU/RDMA/DFS behavior.
"""

import ctypes
import importlib.util
import os
import unittest
import uuid
from pathlib import Path

os.environ["MOONCAKE_ENABLE_DFS"] = "0"
os.environ["MOONCAKE_OFFLOAD_ENABLED"] = "false"
os.environ["MC_STORE_DFS_UNIFIED_BUFFER_SIZE_BYTES"] = "0"

from mooncake.store import MooncakeDistributedStore

_compat_path = (
    Path(__file__).parents[3]
    / "python/sglang/srt/mem_cache/storage/mooncake_store/compat.py"
)
_spec = importlib.util.spec_from_file_location("native_smoke_compat", _compat_path)
_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compat)
ReadComponent, ReadLayout, ReadKeyMapping = _compat.resolve_read_layout_api(
    MooncakeDistributedStore()
)


class TestNativeMooncakeReadPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MooncakeDistributedStore()
        cls.assert_native = cls.store.get_capabilities()["features"]
        assert cls.assert_native["read_plan"] and cls.assert_native["read_layout"]
        result = cls.store.setup(
            "127.0.0.1",
            "P2PHANDSHAKE",
            16 * 1024 * 1024,
            4 * 1024 * 1024,
            "tcp",
            "",
            "",
            enable_dfs_prefetch=False,
            enable_embedded_master=True,
        )
        assert result == 0, result

    def setUp(self):
        self.keys = [f"sglang-port-smoke-{uuid.uuid4().hex}" for _ in range(2)]
        self.payloads = [b"abcdefghijklmnop", b"0123456789ABCDEF"]
        for key, payload in zip(self.keys, self.payloads):
            self.assertEqual(self.store.put(key, payload), 0)
        self.destination = ctypes.create_string_buffer(32)
        self.address = ctypes.addressof(self.destination)
        self.assertIn(self.store.register_buffer(self.address, 32), (0, None))

    def tearDown(self):
        self.store.unregister_buffer(self.address)

    def layouts(self):
        return [
            (
                self.keys,
                [0, 1],
                True,
                [
                    [(self.address, 16, 8, 0)],
                    [(self.address + 8, 16, 8, 8)],
                ],
            )
        ]

    def test_read_plan_layerwise_and_pagewise(self):
        for page_wise in (False, True):
            with self.subTest(page_wise=page_wise):
                ctypes.memset(self.address, 0, 32)
                plan = self.store.create_read_plan(
                    self.layouts(),
                    2,
                    reuse_ranges=False,
                    page_wise=page_wise,
                    borrowed_sessions=False,
                    buffer_owners=[self.destination],
                )
                plan.run()
                plan.wait(0)
                plan.wait(1)
                self.assertEqual(self.destination.raw, b"".join(self.payloads))

    def test_read_layout_with_owned_session(self):
        self.assertEqual(list(self.store.batch_get_session_start(self.keys)), [0, 0])
        try:
            components = [
                ReadComponent(
                    destination_base=self.address,
                    destination_stride=16,
                    byte_count=8,
                    source_offset=0,
                ),
                ReadComponent(
                    destination_base=self.address + 8,
                    destination_stride=16,
                    byte_count=8,
                    source_offset=8,
                ),
            ]
            layout = ReadLayout(
                keys=self.keys,
                rows=[0, 1],
                key_mapping=ReadKeyMapping.PER_ROW,
                components=components,
            )
            self.assertEqual(
                list(self.store.batch_get_into_layouts([layout])), [[16, 16]]
            )
            self.assertEqual(self.destination.raw, b"".join(self.payloads))
        finally:
            self.store.batch_get_session_end(self.keys)

    def test_borrowed_session_refresh_and_adaptive_memory_source(self):
        codes, sources = self.store.batch_get_session_start_with_sources(self.keys)
        self.assertEqual(list(codes), [0, 0])
        self.assertEqual(len(sources), 2)
        try:
            self.assertEqual(
                list(self.store.batch_get_session_refresh(self.keys)), [0, 0]
            )
            os.environ["MOONCAKE_READ_PLAN_ADAPTIVE_SOURCE"] = "1"
            plan = self.store.create_read_plan(
                self.layouts(),
                2,
                page_wise=True,
                borrowed_sessions=True,
                buffer_owners=[self.destination],
            )
            plan.run()
            plan.wait(0)
            plan.wait(1)
            self.assertEqual(self.destination.raw, b"".join(self.payloads))
        finally:
            os.environ.pop("MOONCAKE_READ_PLAN_ADAPTIVE_SOURCE", None)
            self.store.batch_get_session_end(self.keys)


if __name__ == "__main__":
    unittest.main()
