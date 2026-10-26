"""Send serial/concurrent boundary-length requests through the PD router.

This checks serving plumbing, not task accuracy. Reported latency is complete
request latency, not TTFT. Use the normal serving benchmark for TTFT/throughput.
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://10.113.46.17:30320")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[63, 64, 65, 4096, 16383, 16384, 16385],
    )
    parser.add_argument("--token-id", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--require-storage-hit",
        action="store_true",
        help="Fail unless at least one response reports external KV storage hits.",
    )
    parser.add_argument(
        "--token-step",
        type=int,
        default=1,
        help="Vary each request's token ID to help detect cross-request data mixups.",
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.repeat < 1 or any(n < 1 for n in args.lengths):
        parser.error("concurrency, repeat, and all lengths must be positive")
    lengths = args.lengths * args.repeat

    def send(item):
        index, length = item
        token_id = args.token_id + index * args.token_step
        request = urllib.request.Request(
            args.url.rstrip("/") + "/generate",
            data=json.dumps(
                {
                    "input_ids": [token_id] * length,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.max_new_tokens,
                        "ignore_eos": True,
                    },
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - start
        info = result.get("meta_info", {})
        if info.get("prompt_tokens") != length or info.get("completion_tokens", 0) <= 0:
            raise RuntimeError(
                f"Invalid generate response for {length} tokens: {result}"
            )
        if (info.get("finish_reason") or {}).get("type") in ("abort", "error"):
            raise RuntimeError(f"Request failed: {result}")
        row = dict(
            request_index=index,
            input_tokens=length,
            token_id=token_id,
            latency_s=elapsed,
            response=result,
        )
        print(
            json.dumps(
                dict(
                    request_index=index,
                    input_tokens=length,
                    latency_s=elapsed,
                    completion_tokens=info["completion_tokens"],
                    cached_tokens=info.get("cached_tokens", 0),
                    cached_tokens_details=info.get("cached_tokens_details"),
                )
            ),
            flush=True,
        )
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(send, enumerate(lengths)))
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.require_storage_hit and not any(
        (row["response"].get("meta_info", {}).get("cached_tokens_details") or {}).get(
            "storage", 0
        )
        > 0
        for row in results
    ):
        raise RuntimeError(
            "No external KV storage hit was reported; local cache hits alone "
            "do not validate Store/DFS reuse. Results were saved for inspection."
        )
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if [r["input_tokens"] for r in reference] != lengths:
            raise RuntimeError("Reference and current input lengths differ")
        for old, new in zip(reference, results):
            if old.get("token_id", args.token_id) != new["token_id"]:
                raise RuntimeError("Reference and current request token IDs differ")
            for field in ("text", "output_ids"):
                if old["response"].get(field) != new["response"].get(field):
                    raise RuntimeError(
                        f"Raw baseline output differs at {new['input_tokens']} tokens ({field})"
                    )
        print(
            "Raw-path outputs match the reference. This does not evaluate lossy-compression quality."
        )


if __name__ == "__main__":
    main()
