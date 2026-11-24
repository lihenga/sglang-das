"""CPU namespace tests loading real methods without accelerator dependencies."""

import ast
import base64
import hashlib
import importlib.util
import logging
import time
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"
_CI_SPEC = importlib.util.spec_from_file_location(
    "ci_register", SRT.parent / "test/ci/ci_register.py"
)
_CI = importlib.util.module_from_spec(_CI_SPEC)
_CI_SPEC.loader.exec_module(_CI)
register_cpu_ci = _CI.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

logger = logging.getLogger(__name__)
_DEFINITIONS = dict(
    __name__=__name__,
    base64=base64,
    hashlib=hashlib,
    time=time,
    logger=logger,
    Sequence=Sequence,
    dataclass=dataclass,
    Enum=Enum,
)


def _load_class(path, name, methods=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.bases = []
        node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), _DEFINITIONS)
    return _DEFINITIONS[name]


PoolName = _load_class(SRT / "mem_cache/hicache_storage.py", "PoolName")
PoolHitPolicy = _load_class(SRT / "mem_cache/hicache_storage.py", "PoolHitPolicy")
PoolTransfer = _load_class(SRT / "mem_cache/hicache_storage.py", "PoolTransfer")
PoolTransferResult = _load_class(
    SRT / "mem_cache/hicache_storage.py", "PoolTransferResult"
)

MooncakeStore = _load_class(
    SRT / "mem_cache/storage/mooncake_store/mooncake_store.py",
    "MooncakeStore",
    {
        "_tag_keys",
        "_can_use_group_semantics",
        "_filter_group_ids",
        "_get_hybrid_page_component_keys",
        "_pack_multi_buffer_meta",
        "_get_mha_buffer_meta",
        "_get_mla_buffer_meta",
        "_batch_preprocess",
        "_batch_postprocess",
        "batch_set_v1",
        "batch_get_v1",
        "batch_set_v2",
        "_batch_set_v2_aggregated",
        "batch_get_v2",
        "_batch_io_v2",
        "batch_exists",
        "batch_exists_v2",
    },
)
MooncakeDirectLinker = _load_class(
    SRT / "mem_cache/storage/mooncake_store/mooncake_direct_linker.py",
    "MooncakeDirectLinker",
    {"_revalidate_load_local"},
)

RAW_PREFIX = (
    "dspark_ring_v1_-ai_data-models-300b_fast_main_w4a8_flexsmooth_ssz_dspark_w8a8_ssz_"
    "20261029_packed_tokenflow=1.2.2%2Bg7aaa223c_kvcache_dtype=float8_e4m3fn_"
    "kv_layout=pTYLeiQyGs09I_hiCKpgEw"
)
SHORT_PREFIX = "n1_iXR8weUmWB1rM4RK_0Fu8g"
PAGE_HASH = "c37c3d42ebd668bb96b469a8c397a322ef0fad0511d9befd3fb3d7a159960b7a"
RANK_SUFFIX = "tp0_cp0_pp0"


def _make_store(prefix=RAW_PREFIX):
    store = MooncakeStore.__new__(MooncakeStore)
    store.config_prefix = prefix
    store._namespace_prefix_cache = None
    return store


class TestMooncakeKeyNamespace(unittest.TestCase):
    def test_fixed_vector_and_independent_instances_preserve_object_keys(self):
        keys = [f"{PAGE_HASH}_{RANK_SUFFIX}_{kind}" for kind in ("k", "v")]
        expected = [f"{SHORT_PREFIX}_{key}" for key in keys]
        for store in (_make_store(), _make_store()):
            self.assertEqual(store._tag_keys(keys), expected)
            self.assertEqual(store.config_prefix, RAW_PREFIX)
        self.assertEqual(len(SHORT_PREFIX.encode("ascii")), 25)

    def test_plugin_fields_added_before_first_call_are_included(self):
        initial, plugin_fields = RAW_PREFIX.split("_tokenflow=", 1)
        store = _make_store(initial)
        store.config_prefix += "_tokenflow=" + plugin_fields
        self.assertEqual(store._tag_keys([PAGE_HASH]), [f"{SHORT_PREFIX}_{PAGE_HASH}"])

    def test_cache_reuses_only_the_latest_complete_prefix(self):
        store = _make_store()
        with patch.object(hashlib, "sha256", wraps=hashlib.sha256) as sha256:
            first = store._tag_keys([PAGE_HASH])
            cached = store._namespace_prefix_cache
            self.assertEqual(store._tag_keys([PAGE_HASH]), first)
            self.assertIs(store._namespace_prefix_cache, cached)
            sha256.assert_called_once_with(RAW_PREFIX.encode("utf-8"))

            store.config_prefix += "_new_layout"
            self.assertNotEqual(store._tag_keys([PAGE_HASH]), first)
            self.assertEqual(store._namespace_prefix_cache[0], store.config_prefix)
            self.assertEqual(len(store._namespace_prefix_cache), 2)
            self.assertEqual(sha256.call_count, 2)

            store.config_prefix = RAW_PREFIX
            self.assertEqual(store._tag_keys([PAGE_HASH]), first)
            self.assertEqual(sha256.call_count, 3)
        self.assertEqual(store.config_prefix, RAW_PREFIX)

    def test_none_returns_the_original_keys_without_hashing(self):
        store = _make_store(None)
        keys = [f"{PAGE_HASH}_{RANK_SUFFIX}_k"]
        with patch.object(hashlib, "sha256") as sha256:
            self.assertIs(store._tag_keys(keys), keys)
            sha256.assert_not_called()
        self.assertIsNone(store.config_prefix)

    def _io_store(self, *, is_mla=True):
        store = _make_store()
        store.is_mla_backend = is_mla
        store.should_split_heads = False
        store.mla_suffix = store.mha_suffix = RANK_SUFFIX
        store.pp_size = 1
        store._use_group_semantics = False
        store.enable_storage_metrics = False
        objects_per_page = 1 if is_mla else 2
        pool = SimpleNamespace(
            kv_buffer=object(),
            page_size=1,
            get_page_buffer_meta=lambda indices: (
                [1000] * (len(indices) * objects_per_page),
                [8] * (len(indices) * objects_per_page),
            ),
        )
        store.mem_pool_host = pool
        store.registered_pools = {PoolName.KV: pool, PoolName.INDEXER: pool}
        objects = set()

        def put(keys, *args):
            objects.update(keys)
            return [0] * len(keys)

        store._put_batch_zero_copy_impl = Mock(side_effect=put)
        store._get_batch_zero_copy_impl = Mock(
            side_effect=lambda keys, *args: [
                8 if key in objects else -1 for key in keys
            ]
        )
        store._batch_exist = Mock(
            side_effect=lambda keys: [int(key in objects) for key in keys]
        )
        return store

    def test_v1_write_read_and_query_agree_for_mha_and_mla(self):
        for is_mla in (False, True):
            with self.subTest(is_mla=is_mla):
                store = self._io_store(is_mla=is_mla)
                kinds = ("k",) if is_mla else ("k", "v")
                expected = [
                    f"{SHORT_PREFIX}_{PAGE_HASH}_{RANK_SUFFIX}_{kind}" for kind in kinds
                ]
                self.assertEqual(store.batch_set_v1([PAGE_HASH], [0]), [True])
                self.assertEqual(store.batch_get_v1([PAGE_HASH], [0]), [True])
                self.assertEqual(store.batch_exists([PAGE_HASH]), 1)
                self.assertEqual(
                    store._put_batch_zero_copy_impl.call_args.args[0], expected
                )
                self.assertEqual(
                    store._get_batch_zero_copy_impl.call_args.args[0], expected
                )
                self.assertEqual(store._batch_exist.call_args.args[0], expected)

    def test_v2_write_read_query_and_direct_linker_agree(self):
        store = self._io_store()
        transfers = [
            PoolTransfer(name=name, keys=[PAGE_HASH], host_indices=[0])
            for name in (PoolName.KV, PoolName.INDEXER)
        ]
        expected = [
            f"{SHORT_PREFIX}_{PAGE_HASH}_{RANK_SUFFIX}_{kind}"
            for kind in ("k", "indexer")
        ]
        results = {PoolName.KV: [True], PoolName.INDEXER: [True]}
        self.assertEqual(store.batch_set_v2(transfers), results)
        self.assertEqual(store._put_batch_zero_copy_impl.call_args.args[0], expected)
        self.assertEqual(store.batch_get_v2(transfers), results)
        self.assertEqual(
            [
                call.args[0][0]
                for call in store._get_batch_zero_copy_impl.call_args_list
            ],
            expected,
        )
        hits = store.batch_exists_v2([PAGE_HASH], transfers)
        self.assertEqual(hits.kv_hit_pages, 1)
        self.assertEqual(hits.extra_pool_hit_pages[PoolName.INDEXER], 1)
        self.assertEqual(
            [call.args[0][0] for call in store._batch_exist.call_args_list[-2:]],
            expected,
        )

        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.storage = store
        linker.pool_group = SimpleNamespace(
            resolve_transfers=Mock(return_value=transfers)
        )
        self.assertTrue(linker._revalidate_load_local(transfers))
        self.assertEqual(store._batch_exist.call_args.args[0], expected)
        self.assertEqual(store.config_prefix, RAW_PREFIX)


if __name__ == "__main__":
    unittest.main()
