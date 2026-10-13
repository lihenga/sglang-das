import gc
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.deepseek_v4_storage import allocate_dsv4_buffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_BACKEND = "MOONCAKE_DFS_DIRECT_IO_BACKEND"


class TestDSV4Storage(unittest.TestCase):
    def test_real_storage_regression(self):
        for backend in ("xds-required", "xds-auto"):
            with self.subTest(backend=backend), patch.dict(os.environ, {_BACKEND: backend}):
                tensor = allocate_dsv4_buffer(
                    (157098240,), dtype=torch.uint8, device="cpu", init="empty"
                )
                self.assertEqual(tensor.nbytes, 157098240)
                self.assertEqual(tensor.untyped_storage().nbytes(), 157102080)
                self.assertEqual(tensor.storage_offset(), 0)
                self.assertEqual(tensor.data_ptr(), tensor.untyped_storage().data_ptr())
                del tensor

    def test_dtype_layout_initialization_and_alias_lifetime(self):
        with patch.dict(os.environ, {_BACKEND: "xds-required"}):
            for dtype in (torch.uint8, torch.int8, torch.bfloat16, torch.float32):
                for init in ("zeros", "empty"):
                    with self.subTest(dtype=dtype, init=init):
                        tensor = allocate_dsv4_buffer(
                            (3, 8), dtype=dtype, device="cpu", init=init
                        )
                        self.assertEqual(tensor.shape, (3, 8))
                        self.assertEqual(tensor.stride(), (8, 1))
                        self.assertEqual(tensor.dtype, dtype)
                        self.assertEqual(tensor.nbytes, 24 * dtype.itemsize)
                        self.assertEqual(tensor.untyped_storage().nbytes(), 4096)
                        self.assertEqual(tensor.storage_offset(), 0)
                        if init == "zeros":
                            self.assertEqual(torch.count_nonzero(tensor).item(), 0)
                        kv, score = tensor[:, :4], tensor[:, 4:]
                        kv.fill_(3)
                        score.fill_(7)
                        self.assertTrue(torch.all(tensor[:, :4] == 3).item())
                        self.assertTrue(torch.all(tensor[:, 4:] == 7).item())
                        base = tensor.untyped_storage().data_ptr()
                        del tensor
                        gc.collect()
                        self.assertEqual(kv.untyped_storage().data_ptr(), base)
                        self.assertEqual(score.untyped_storage().data_ptr(), base)
                        self.assertEqual(score.untyped_storage().nbytes(), 4096)
                        self.assertTrue(torch.all(kv == 3).item())
                        self.assertTrue(torch.all(score == 7).item())

    def test_packed_dtype_alias_keeps_storage(self):
        with patch.dict(os.environ, {_BACKEND: "xds-required"}):
            tensor = allocate_dsv4_buffer(
                (3, 16), dtype=torch.uint8, device="cpu", init="zeros"
            )
            alias = tensor.view(torch.float32)
            self.assertEqual(alias.shape, (3, 4))
            self.assertEqual(alias.stride(), (4, 1))
            self.assertEqual(alias.data_ptr(), tensor.data_ptr())
            alias.fill_(1.0)
            self.assertTrue(torch.equal(tensor, alias.view(torch.uint8)))
            del tensor
            gc.collect()
            self.assertEqual(alias.untyped_storage().nbytes(), 4096)
            self.assertTrue(torch.all(alias == 1.0).item())

    def test_empty_stays_empty_and_zeros_stays_zeroed(self):
        # Spy on factories rather than assuming anything about uninitialized data.
        for init in ("empty", "zeros"):
            with patch.dict(os.environ, {_BACKEND: "xds-auto"}), patch(
                f"torch.{init}", wraps=getattr(torch, init)
            ) as factory:
                allocate_dsv4_buffer(
                    (3, 8), dtype=torch.float32, device="cpu", init=init
                )
                factory.assert_called_once_with(
                    (1024,), dtype=torch.float32, device="cpu"
                )

    def test_default_and_posix_unchanged(self):
        for backend in (None, "posix"):
            for init in ("empty", "zeros"):
                with self.subTest(backend=backend, init=init), patch.dict(os.environ):
                    os.environ.pop(_BACKEND, None)
                    if backend is not None:
                        os.environ[_BACKEND] = backend
                    with patch(f"torch.{init}", wraps=getattr(torch, init)) as factory:
                        tensor = allocate_dsv4_buffer(
                            (3, 8), dtype=torch.float32, device="cpu", init=init
                        )
                        factory.assert_called_once_with(
                            (3, 8), dtype=torch.float32, device="cpu"
                        )
                    self.assertEqual(tensor.untyped_storage().nbytes(), 96)
                    self.assertEqual(tensor.stride(), (8, 1))
                    self.assertIsNone(tensor._base)

    def test_already_aligned_and_zero_size(self):
        with patch.dict(os.environ, {_BACKEND: "xds-required"}):
            for size in (0, 4096, 8192):
                tensor = allocate_dsv4_buffer(
                    (size,), dtype=torch.uint8, device="cpu", init="zeros"
                )
                self.assertEqual(tensor.nbytes, size)
                self.assertEqual(tensor.untyped_storage().nbytes(), size)

    def test_gpu_misaligned_storage_is_diagnosed(self):
        # A CPU test must not depend on the host allocator's address alignment.
        storage = SimpleNamespace(data_ptr=lambda: 0x1100, nbytes=lambda: 4096)
        backing = SimpleNamespace(
            is_cuda=True, device="cuda:0", untyped_storage=lambda: storage
        )
        with patch.dict(os.environ, {_BACKEND: "xds-required"}), patch(
            "torch.empty", return_value=backing
        ):
            with self.assertRaisesRegex(RuntimeError, "storage_base=0x1100"):
                allocate_dsv4_buffer(
                    (24,), dtype=torch.uint8, device="cuda:0", init="empty"
                )


if __name__ == "__main__":
    unittest.main()
