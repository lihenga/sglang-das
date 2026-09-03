"""CPU-only regression tests for padded DSA TileLang output layout."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.attention.dsa_backend as dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_TOTAL_ROWS = 10
_VALID_ROWS = 5
_NUM_HEADS = 2
_VALUE_DIM = 3
_HEAD_DIM = 4


class _DraftExtendMode:
    def is_decode_or_idle(self):
        return False

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return True


def _make_forward_batch(valid_rows):
    original_batch_size = int(valid_rows > 0)
    return SimpleNamespace(
        forward_mode=_DraftExtendMode(),
        _original_forward_mode=None,
        _original_batch_size=original_batch_size,
        _original_num_tokens=valid_rows,
        forward_metadata_planned_bs=original_batch_size,
        forward_metadata_planned_num_tokens=valid_rows,
    )


def _make_backend(tilelang_output):
    return SimpleNamespace(
        forward_metadata=SimpleNamespace(),
        dsa_decode_impl="tilelang",
        dsa_prefill_impl="tilelang",
        use_mha=False,
        token_to_kv_pool=SimpleNamespace(
            get_key_buffer=MagicMock(return_value=torch.empty(1))
        ),
        get_topk_transform_method=MagicMock(return_value=object()),
        use_fused_topk=True,
        _get_fused_topk_page_table=MagicMock(
            return_value=torch.zeros((_TOTAL_ROWS, 2), dtype=torch.int32)
        ),
        hisparse_coordinator=None,
        hcu_fp8_kv_dequant_bf16=False,
        _forward_tilelang=MagicMock(return_value=tilelang_output),
    )


def _run_tilelang(method, backend, valid_rows):
    layer = SimpleNamespace(
        is_cross_attention=False,
        layer_id=0,
        tp_q_head_num=_NUM_HEADS,
        v_head_dim=_VALUE_DIM,
        head_dim=_HEAD_DIM,
        scaling=1.0,
    )
    q = torch.zeros((_TOTAL_ROWS, _NUM_HEADS * _VALUE_DIM))
    q_rope = torch.zeros(
        (_TOTAL_ROWS, _NUM_HEADS * (_HEAD_DIM - _VALUE_DIM))
    )
    with patch.object(dsa_backend, "_is_hcu", True):
        return method(
            backend,
            q,
            None,
            None,
            layer,
            _make_forward_batch(valid_rows),
            q_rope=q_rope,
        )


class TestDSATileLangRepad(CustomTestCase):
    _METHODS = (
        dsa_backend.DeepseekSparseAttnBackend.forward_extend,
        dsa_backend.DeepseekSparseAttnBackend.forward_decode,
    )

    def test_repad_preserves_prefix_and_zeros_tail(self):
        expected_prefix = torch.arange(
            _VALID_ROWS * _NUM_HEADS * _VALUE_DIM, dtype=torch.float32
        ).view(1, _VALID_ROWS, _NUM_HEADS, _VALUE_DIM)

        for method in self._METHODS:
            with self.subTest(method=method.__name__):
                backend = _make_backend(expected_prefix)
                output = _run_tilelang(method, backend, _VALID_ROWS)

                self.assertEqual(
                    output.shape,
                    (1, _TOTAL_ROWS, _NUM_HEADS, _VALUE_DIM),
                )
                self.assertTrue(torch.equal(output[:, :_VALID_ROWS], expected_prefix))
                self.assertEqual(output[:, _VALID_ROWS:].count_nonzero().item(), 0)
                call = backend._forward_tilelang.call_args.kwargs
                self.assertEqual(call["q_all"].shape[0], _VALID_ROWS)
                self.assertEqual(call["page_table_1"].shape[0], _VALID_ROWS)

    def test_zero_valid_rows_keeps_four_dimensional_abi(self):
        unused_output = torch.empty((1, 0, _NUM_HEADS, _VALUE_DIM))

        for method in self._METHODS:
            with self.subTest(method=method.__name__):
                backend = _make_backend(unused_output)
                output = _run_tilelang(method, backend, 0)

                self.assertEqual(
                    output.shape,
                    (1, _TOTAL_ROWS, _NUM_HEADS, _VALUE_DIM),
                )
                self.assertEqual(output.count_nonzero().item(), 0)
                backend._forward_tilelang.assert_not_called()


if __name__ == "__main__":
    unittest.main()
