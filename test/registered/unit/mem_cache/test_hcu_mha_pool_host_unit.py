"""HCU host-pool packed MTP allocation and transfer regression tests."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host import mha
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHostHCU
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_device_pool(layer_num=2, *, device="cpu", dtype=torch.float32, page_size=4):
    page_num, head_num, head_dim = 5, 2, 8
    k_buffer = [
        torch.zeros(page_num, head_num, page_size, head_dim, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    v_buffer = [
        torch.zeros(page_num, head_num, head_dim, page_size, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    return SimpleNamespace(
        layer_num=layer_num,
        layer_shard_enabled=False,
        start_layer=0,
        end_layer=layer_num,
        size=(page_num - 1) * page_size,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=head_dim,
        page_size=page_size,
        store_dtype=dtype,
        device=device,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=torch.tensor(
            [x.data_ptr() for x in k_buffer], dtype=torch.uint64, device=device
        ),
        v_data_ptrs=torch.tensor(
            [x.data_ptr() for x in v_buffer], dtype=torch.uint64, device=device
        ),
    )


def make_host(target, drafts=(), *, host_size=0, pin_memory=False):
    def allocate(shape, *, dtype, device, pin_memory, allocator):
        return torch.zeros(shape, dtype=dtype, device=device, pin_memory=pin_memory)

    with (
        mock.patch.dict(mha.ALLOC_MEMORY_FUNCS, {target.device: allocate}),
        mock.patch(
            "sglang.srt.mem_cache.pool_host.base.get_allocator_from_storage",
            return_value=None,
        ),
        mock.patch(
            "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
            return_value=8 * 1024**3,
        ),
    ):
        return MHATokenToKVPoolHostHCU(
            target,
            host_to_device_ratio=2,
            host_size=host_size,
            page_size=target.page_size,
            layout="layout_hcu",
            pin_memory=pin_memory,
            mtp_draft_device_pools=drafts,
            pool_label="kv_test",
        )


def direct_backup(**kw):
    page_size = kw["page_size"]
    for layer in range(len(kw["src_ptrs_k"])):
        for src, dst in zip(
            kw["src_indices"].tolist()[::page_size],
            kw["dst_indices"].tolist()[::page_size],
        ):
            index = kw["start_layer_id"] + layer
            for component in ("k", "v"):
                kw[f"dst_ptrs_{component}"][dst // page_size, index].copy_(
                    kw[f"src_ptrs_{component}"][index][src // page_size]
                )


def direct_load(**kw):
    page_size = kw["page_size"]
    index = kw["start_layer_id"]
    for src, dst in zip(
        kw["src_indices"].tolist()[::page_size],
        kw["dst_indices"].tolist()[::page_size],
    ):
        for component in ("k", "v"):
            kw[f"dst_ptrs_{component}"][index][dst // page_size].copy_(
                kw[f"src_ptrs_{component}"][src // page_size, index]
            )


class TestHCUHostPoolMTP(unittest.TestCase):
    def setUp(self):
        self.target = make_device_pool()
        self.drafts = (make_device_pool(1), make_device_pool(1))
        self.host = make_host(self.target, self.drafts)
        self.ops = ModuleType("sgl_kernel.kvcacheio")
        for name in (
            "transfer_kv_all_kernel_lf_pf_D2H_hcu",
            "transfer_kv_per_layer_kernel_pf_lf_H2D_hcu",
            "transfer_kv_all_direct_lf_pf_D2H_hcu",
            "transfer_kv_all_direct_pf_lf_H2D_hcu",
        ):
            setattr(self.ops, name, mock.Mock())
        patcher = mock.patch.dict(sys.modules, {"sgl_kernel.kvcacheio": self.ops})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_constructor_packs_all_draft_layers(self):
        host = self.host
        self.assertEqual(host.target_layer_num, 2)
        self.assertEqual(host.layer_num, 4)
        self.assertEqual(host.pool_label, "kv_test")
        self.assertEqual(host.k_buffer.shape, (9, 4, 2, 4, 8))
        self.assertEqual(host.v_buffer.shape, (9, 4, 2, 8, 4))
        self.assertEqual(host.size_per_token, 4 * 2 * 8 * 4 * 2)
        for component in ("k", "v"):
            buffers = [
                buf
                for pool in (self.target, *self.drafts)
                for buf in getattr(pool, f"{component}_buffer")
            ]
            self.assertEqual(
                getattr(host, f"packed_device_{component}_data_ptrs").tolist(),
                [buf.data_ptr() for buf in buffers],
            )
            for actual, expected in zip(
                getattr(host, f"packed_device_{component}_buffers"), buffers
            ):
                self.assertIs(actual, expected)

    def test_fixed_host_budget_accounts_for_draft_layers(self):
        budget = 16384
        plain = make_host(self.target, host_size=budget / 1e9)
        packed = make_host(self.target, self.drafts, host_size=budget / 1e9)
        for host in (plain, packed):
            expected_pages = int(budget // host.size_per_token) // host.page_size + 1
            self.assertEqual(host.page_num, expected_pages)
        self.assertLess(packed.size, plain.size)

    def test_no_mtp_uses_original_buffers(self):
        host = make_host(self.target)
        self.assertEqual(host.layer_num, self.target.layer_num)
        actual = host._resolve_device_transfer_buffers(self.target)
        expected = (
            self.target.k_data_ptrs,
            self.target.v_data_ptrs,
            self.target.k_buffer,
            self.target.v_buffer,
        )
        for a, b in zip(actual, expected):
            self.assertIs(a, b)

    def test_rejects_incompatible_draft_pools(self):
        cases = {
            "layer_num": 2,
            "layer_shard_enabled": True,
            "head_num": 3,
            "head_dim": 16,
            "v_head_dim": 16,
            "page_size": 8,
            "store_dtype": torch.float16,
            "device": "cuda",
            "size": self.target.size - 1,
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                draft = make_device_pool(1)
                setattr(draft, name, value)
                with self.assertRaises(ValueError):
                    make_host(self.target, (draft,))

    def test_kernel_backup_uses_packed_pointer_tables(self):
        indices = torch.arange(4)
        self.host.backup_from_device_all_layer(self.target, indices, indices, "kernel")
        call = self.ops.transfer_kv_all_kernel_lf_pf_D2H_hcu.call_args.kwargs
        self.assertIs(call["src_k"], self.host.packed_device_k_data_ptrs)
        self.assertIs(call["src_v"], self.host.packed_device_v_data_ptrs)
        self.assertEqual(call["layer_num"], 4)
        self.assertEqual(call["dst_layout_dim"], self.host.token_stride_size * 4)

    def test_kernel_load_maps_each_packed_tail_to_device_layer_zero(self):
        indices = torch.arange(4)
        for depth, draft in enumerate(self.drafts):
            self.host.load_to_device_per_layer(
                draft, indices, indices, 2 + depth, "kernel", is_draft=True
            )
            call = self.ops.transfer_kv_per_layer_kernel_pf_lf_H2D_hcu.call_args.kwargs
            self.assertEqual(call["layer_id"], 2 + depth)
            self.assertIs(call["src_k"], self.host.k_buffer)
            self.assertIs(call["dst_k"], draft.k_buffer[0])
            self.assertIs(call["dst_v"], draft.v_buffer[0])
            self.assertEqual(call["src_layout_dim"], self.host.token_stride_size * 4)

    def test_normal_load_accepts_false_and_skips_packed_tail(self):
        indices = torch.arange(4)
        self.host.load_to_device_per_layer(
            self.target, indices, indices, 1, "kernel", is_draft=False
        )
        op = self.ops.transfer_kv_per_layer_kernel_pf_lf_H2D_hcu
        self.assertEqual(op.call_args.kwargs["layer_id"], 1)
        self.assertIs(op.call_args.kwargs["dst_k"], self.target.k_buffer[1])
        for layer in (2, 3):
            self.host.load_to_device_per_layer(
                self.target, indices, indices, layer, "kernel"
            )
        self.assertEqual(op.call_count, 1)

    def test_direct_round_trip_with_reordered_pages(self):
        self.ops.transfer_kv_all_direct_lf_pf_D2H_hcu.side_effect = direct_backup
        self.ops.transfer_kv_all_direct_pf_lf_H2D_hcu.side_effect = direct_load
        for draft_count in (0, 1, 2):
            with self.subTest(draft_count=draft_count):
                host = make_host(self.target, self.drafts[:draft_count])
                pools = (self.target, *self.drafts[:draft_count])
                source_indices = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7])
                host_indices = torch.tensor([4, 5, 6, 7, 20, 21, 22, 23])
                dest_indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
                expected = []
                layer = 0
                for pool in pools:
                    for k, v in zip(pool.k_buffer, pool.v_buffer):
                        values = torch.arange(k.numel()).reshape(k.shape) + 1000 * layer
                        k.copy_(values)
                        v.copy_(-values.reshape(v.shape) - 1)
                        expected.append((k.clone(), v.clone()))
                        layer += 1
                host.backup_from_device_all_layer(
                    self.target, host_indices, source_indices, "direct"
                )
                layer = 0
                for pool in pools:
                    for k, v in zip(pool.k_buffer, pool.v_buffer):
                        for actual, saved in zip(
                            (host.k_buffer[:, layer], host.v_buffer[:, layer]),
                            expected[layer],
                        ):
                            torch.testing.assert_close(actual[1], saved[3])
                            torch.testing.assert_close(actual[5], saved[1])
                        k.zero_()
                        v.zero_()
                        layer += 1
                for layer in range(self.target.layer_num):
                    host.load_to_device_per_layer(
                        self.target, host_indices, dest_indices, layer, "direct"
                    )
                for depth, draft in enumerate(pools[1:]):
                    host.load_to_device_per_layer(
                        draft,
                        host_indices,
                        dest_indices,
                        2 + depth,
                        "direct",
                        is_draft=True,
                    )
                layer = 0
                for pool in pools:
                    for k, v in zip(pool.k_buffer, pool.v_buffer):
                        for actual, saved in zip((k, v), expected[layer]):
                            torch.testing.assert_close(actual[0], saved[3])
                            torch.testing.assert_close(actual[2], saved[1])
                            self.assertEqual(actual[[1, 3, 4]].count_nonzero(), 0)
                        layer += 1

    def test_storage_pages_and_metadata_include_draft_tail(self):
        host = self.host
        host.k_buffer.copy_(
            torch.arange(host.k_buffer.numel()).reshape(host.k_buffer.shape)
        )
        host.v_buffer.copy_(
            -torch.arange(host.v_buffer.numel()).reshape(host.v_buffer.shape)
        )
        original = tuple(x.clone() for x in host.get_data_page(4))
        host.set_from_flat_data_page(8, original)
        for actual, expected in zip(host.get_data_page(8), original):
            torch.testing.assert_close(actual, expected)
        dummy = host.get_dummy_flat_data_page()
        self.assertEqual(dummy[0].numel(), original[0].numel())
        indices = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        pointers, sizes = host.get_page_buffer_meta(indices)
        page_bytes = host.size_per_token * host.page_size // 2
        self.assertEqual(sizes, [page_bytes] * 4)
        self.assertEqual(
            pointers,
            [
                host.k_buffer.data_ptr() + page_bytes,
                host.v_buffer.data_ptr() + page_bytes,
                host.k_buffer.data_ptr() + 2 * page_bytes,
                host.v_buffer.data_ptr() + 2 * page_bytes,
            ],
        )


if __name__ == "__main__":
    unittest.main()
