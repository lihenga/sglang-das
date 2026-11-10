"""Manual HCU packed-MTP round trips through the real L2 engine and operators.

Run in an HCU container: python test/manual/test_hcu_mha_pool_host_mtp.py
Only small synthetic KV pools are allocated; no model server is started.
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "python"))

    import torch

    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine

    spec = importlib.util.spec_from_file_location(
        "hcu_host_test_helpers",
        args.repo / "test/registered/unit/mem_cache/test_hcu_mha_pool_host_unit.py",
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    logging.getLogger("sglang.srt.mem_cache.pool_host.base").setLevel(logging.WARNING)
    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    page_size = 64
    passed = 0

    def indices(pages, backend):
        return torch.tensor(
            [p * page_size + offset for p in pages for offset in range(page_size)],
            dtype=torch.int64,
            device=device if backend == "kernel" else "cpu",
        )

    def assert_bytes_equal(actual, expected):
        torch.testing.assert_close(
            actual.cpu().contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )

    for backend in ("direct", "kernel"):
        for dtype in (torch.bfloat16, torch.float8_e5m2, torch.uint8):
            for draft_count in (0, 1, 2):
                kwargs = dict(device=device, dtype=dtype, page_size=page_size)
                target = helpers.make_device_pool(2, **kwargs)
                drafts = tuple(
                    helpers.make_device_pool(1, **kwargs) for _ in range(draft_count)
                )
                host = helpers.make_host(target, drafts, pin_memory=True)
                pools = (target, *drafts)
                expected = []
                layer = 0
                for pool in pools:
                    for k, v in zip(pool.k_buffer, pool.v_buffer):
                        saved = []
                        for component, buffer in enumerate((k, v)):
                            raw = (
                                torch.arange(buffer.numel() * buffer.element_size())
                                + layer * 43
                                + component * 17
                            ).remainder(251).to(torch.uint8)
                            data = raw.view(dtype).reshape(buffer.shape)
                            buffer.copy_(data)
                            saved.append(data.clone())
                        expected.append(tuple(saved))
                        layer += 1

                source_indices = indices([3, 1], backend)
                host_indices = indices([1, 5], backend)
                dest_indices = indices([0, 2], backend)
                engine = L2TransferEngine(backend)
                completion = engine.submit_device_to_host(
                    [L2Transfer(host, target, host_indices, source_indices)]
                )
                completion.finish_event.synchronize()
                layer = 0
                for pool in pools:
                    for k, v in zip(pool.k_buffer, pool.v_buffer):
                        for actual, saved in zip(
                            (host.k_buffer[:, layer], host.v_buffer[:, layer]),
                            expected[layer],
                        ):
                            assert_bytes_equal(actual[1], saved[3])
                            assert_bytes_equal(actual[5], saved[1])
                        k.zero_()
                        v.zero_()
                        layer += 1

                # Exercise the production controller's packed-draft transfer mapping.
                controller = HybridCacheController.__new__(HybridCacheController)
                controller.layer_num = target.layer_num
                controller.has_mtp_draft = bool(drafts)
                controller.mtp_draft_device_pools = drafts
                layer_mapping = {i: i for i in range(host.layer_num)}
                controller.mem_pool_host = SimpleNamespace(
                    entry_map={
                        "kv": SimpleNamespace(
                            host_pool=host,
                            device_pool=target,
                            packed_draft_device_pools=drafts,
                        )
                    }
                )
                controller._l2_transfers = mock.Mock(
                    return_value=[
                        L2Transfer(
                            host,
                            target,
                            host_indices,
                            dest_indices,
                            layer_mapper=layer_mapping.get,
                        )
                    ]
                )
                transfers = controller._l2_load_transfers(host_indices, dest_indices)
                assert len(transfers) == 1 + draft_count
                done = []
                completion = engine.submit_host_to_device(
                    transfers, layer_num=target.layer_num, on_layer_done=done.append
                )
                completion.finish_event.synchronize()
                assert done == list(range(target.layer_num))
                layer = 0
                for pool in pools:
                    for k, v in zip(pool.k_buffer, pool.v_buffer):
                        for actual, saved in zip((k, v), expected[layer]):
                            restored = torch.zeros_like(saved)
                            restored[0].copy_(saved[3])
                            restored[2].copy_(saved[1])
                            assert_bytes_equal(actual, restored)
                        layer += 1
                passed += 1
                print(
                    f"PASS backend={backend} dtype={dtype} drafts={draft_count}",
                    flush=True,
                )

    print(f"PASS: {passed} real HCU round-trip cases", flush=True)


if __name__ == "__main__":
    main()
