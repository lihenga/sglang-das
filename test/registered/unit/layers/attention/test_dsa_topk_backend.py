import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_deterministic_unfused_topk_canonical_short_rows():
    topk = 8
    width = 2 * topk
    lengths = torch.tensor([0, 1, topk - 1, topk, topk + 1], dtype=torch.int32)
    row_starts = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    scores = torch.arange(width, dtype=torch.float32).repeat(lengths.numel(), 1)

    with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
        batched = DSATopKBackend.TORCH.topk_func(
            scores, lengths, topk, row_starts=row_starts
        )
        standalone = DSATopKBackend.TORCH.topk_func(
            torch.zeros((1, topk)), torch.tensor([topk]), topk
        )

    expected_short = torch.tensor(
        [
            [-1] * topk,
            [0] + [-1] * (topk - 1),
            list(range(topk - 1)) + [-1],
            list(range(topk)),
        ],
        dtype=torch.int32,
    )
    canonical = torch.arange(topk, dtype=torch.int32)

    assert batched.shape == (5, topk)
    assert batched.dtype == torch.int32
    assert torch.equal(batched[:4], expected_short)
    assert torch.equal(standalone[0], canonical)
    assert torch.equal(batched[3], standalone[0])
    assert torch.equal(
        torch.sort(batched[4]).values,
        torch.arange(1, topk + 1, dtype=torch.int32),
    )
