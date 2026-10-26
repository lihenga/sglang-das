"""Validate the concurrent HTTP probe against a local serving stub, not a GPU."""

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def test_concurrent_http_smoke_keeps_input_output_identity_and_reference(
    monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "kvserve_smoke_test", root / "examples/kvserve/smoke_requests.py"
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    lock = threading.Lock()
    counts = {"active": 0, "maximum": 0, "storage": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                counts["active"] += 1
                counts["maximum"] = max(counts["maximum"], counts["active"])
            time.sleep(0.02)
            tokens = payload["input_ids"]
            result = {
                "text": f"{len(tokens)}:{tokens[0]}",
                "output_ids": [tokens[0]],
                "meta_info": {
                    "prompt_tokens": len(tokens),
                    "completion_tokens": 1,
                    "cached_tokens_details": {
                        "device": len(tokens),
                        "host": 0,
                        "storage": counts["storage"],
                    },
                },
            }
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            with lock:
                counts["active"] -= 1

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        common = [
            "smoke_requests.py",
            "--url",
            f"http://127.0.0.1:{server.server_port}",
            "--lengths",
            "1",
            "2",
            "3",
            "4",
            "--repeat",
            "3",
            "--concurrency",
            "4",
        ]
        baseline, current = tmp_path / "off.json", tmp_path / "raw.json"
        monkeypatch.setattr(sys, "argv", common + ["--output", str(baseline)])
        smoke.main()
        monkeypatch.setattr(
            sys,
            "argv",
            common + ["--output", str(current), "--reference", str(baseline)],
        )
        smoke.main()
        rows = json.loads(current.read_text(encoding="utf-8"))
        assert [r["request_index"] for r in rows] == list(range(12))
        assert [r["input_tokens"] for r in rows] == [1, 2, 3, 4] * 3
        assert [r["token_id"] for r in rows] == list(range(1000, 1012))
        assert counts["maximum"] > 1
        probe = common + ["--output", str(current), "--require-storage-hit"]
        monkeypatch.setattr(sys, "argv", probe)
        with pytest.raises(RuntimeError, match="No external KV storage hit"):
            smoke.main()
        counts["storage"] = 1
        monkeypatch.setattr(sys, "argv", probe)
        smoke.main()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
