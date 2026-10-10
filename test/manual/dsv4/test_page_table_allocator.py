"""Compare allocator states with the page table off/on on CPU and HIP.

Run in the HCU SGLang environment:
    python test/manual/dsv4/test_page_table_allocator.py

Checks allocation order, duplicate and dummy-page frees, deferred frees,
shared SWA mappings, and the terminal -1 sentinel against the original path.
The GPU tests use an idle device and should not overlap a throughput test.
"""

import json
import random
import time

import torch

from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    _page_membership,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator


def paged(ps, sort, dev, use_page_table, n=64):
    a = PagedTokenToKVPoolAllocator(n * ps, ps, torch.int64, dev, None, sort)
    a.use_page_table = use_page_table
    return a


def same(a, b):
    for attr in ["free_pages", "release_pages"]:
        assert torch.equal(getattr(a, attr), getattr(b, attr)), attr
    assert a.available_size() == b.available_size()


def swa(ps, sort, dev, use_page_table):
    a = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
    a.page_size = ps
    a.device = dev
    a.dtype = torch.int64
    a.need_sort = sort
    a._size_full = 128 * ps
    a._size_swa = 64 * ps
    a.full_attn_allocator = paged(ps, sort, dev, use_page_table, 128)
    a.swa_attn_allocator = paged(ps, sort, dev, use_page_table, 64)
    a.full_attn_allocator.alloc(128 * ps)
    a.swa_attn_allocator.alloc(64 * ps)
    a.is_not_in_free_group = True
    a.free_group = []
    a.swa_free_group = []
    # Multiple full slots alias each SWA page, including dummy and sentinel slots.
    m = torch.arange(128 * ps + ps, device=dev, dtype=torch.int64) % (64 * ps) + ps
    m[::17] = 0
    a.full_to_swa_index_mapping = torch.cat([m, torch.tensor([-1], device=dev)])
    return a


def run_checks():
    started = time.time()
    results = {}
    rng = random.Random(42)
    for dev in ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []):
        cases = 0
        for n in [1, 8, 64, 1048, 6988]:
            for _ in range(20):
                q = torch.tensor(
                    [rng.randrange(-1, n + 1) for _ in range(257)], device=dev
                )
                m = torch.tensor(
                    [rng.randrange(n + 1) for _ in range(rng.randrange(1, 100))],
                    device=dev,
                )
                empty = m[:0]
                assert torch.equal(_page_membership(q, n, m, empty), torch.isin(q, m))
                cases += 1
        for ps in [8, 256]:
            for sort in [False, True]:
                a, b = [paged(ps, sort, dev, bit) for bit in [False, True]]
                live = []
                for _ in range(100):
                    if not live or rng.random() < 0.5:
                        n = rng.randrange(1, 9) * ps
                        x = a.alloc(n)
                        y = b.alloc(n)
                        assert (x is None) == (y is None)
                        if x is not None:
                            assert torch.equal(x, y)
                            live.append(x)
                    else:
                        x = live.pop(rng.randrange(len(live)))
                        part = x[:: rng.randrange(1, ps + 1)]
                        ids = torch.cat(
                            [part, part, torch.zeros(2, dtype=torch.int64, device=dev)]
                        )
                        a.free(ids)
                        b.free(ids)
                        a.free(ids)
                        b.free(ids)
                    same(a, b)
                    cases += 1
                for alloc in [a, b]:
                    alloc.clear()
                    alloc.alloc(4 * ps)
                    x = torch.arange(ps, 3 * ps, device=dev)
                    alloc.free_group_begin()
                    alloc.free(x)
                    x.fill_(4 * ps)
                    alloc.free_group_end()
                same(a, b)
                a, b = [swa(ps, sort, dev, bit) for bit in [False, True]]
                for i in range(60):
                    ids = torch.tensor(
                        [
                            rng.randrange(1, 128 * ps)
                            for _ in range(rng.randrange(1, 80))
                        ],
                        device=dev,
                    )
                    if i % 3 == 0:
                        a.free_group_begin()
                        b.free_group_begin()
                    a.free_swa(ids)
                    b.free_swa(ids)
                    if i % 3 == 0:
                        a.free_group_end()
                        b.free_group_end()
                    same(a.swa_attn_allocator, b.swa_attn_allocator)
                    assert torch.equal(
                        a.full_to_swa_index_mapping, b.full_to_swa_index_mapping
                    )
                    assert a.full_to_swa_index_mapping[-1].item() == -1
                    assert torch.equal(
                        a.swa_attn_allocator.alloc(ps), b.swa_attn_allocator.alloc(ps)
                    )
                    cases += 1
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        results[dev] = {"state_and_membership_checks": cases}
    results["seconds"] = time.time() - started
    return results


def test_allocator_equivalence():
    run_checks()


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
