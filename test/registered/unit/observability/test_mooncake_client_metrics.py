import json
import logging
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.observability.mooncake_client_metrics import (
    CLIENT_LABELS,
    MAX_SNAPSHOT_BYTES,
    SNAPSHOT_PATH,
    MooncakeClientCollector,
    MooncakeClientMetricsReporter,
    SnapshotQuotaExceeded,
    add_snapshot_route,
    expire_snapshots,
    receive_snapshot,
    snapshot_expiry,
    start_client_metrics_reporter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

METRICS = """# HELP mooncake_test_bytes Bytes read
# TYPE mooncake_test_bytes counter
mooncake_test_bytes 12
# HELP mooncake_test_latency Seconds per read
# TYPE mooncake_test_latency histogram
mooncake_test_latency_bucket{le="1"} 1
mooncake_test_latency_bucket{le="+Inf"} 2
mooncake_test_latency_count 2
mooncake_test_latency_sum 3
# HELP mooncake_test_buffer Buffer usage
# TYPE mooncake_test_buffer gauge
mooncake_test_buffer{kind="read"} 0
"""


def payload(index=1, sequence=1, metrics=METRICS):
    client_id = f"{index:032x}"
    return {
        "client_id": client_id,
        "sequence": sequence,
        "labels": dict(
            zip(CLIENT_LABELS, (client_id, f"worker-{index % 2}", "10", str(index)))
        ),
        "metrics": metrics,
    }


from sglang.srt.runtime_context import get_context, get_serving
from sglang.test.test_utils import CustomTestCase


class TestMooncakeClientMetrics(CustomTestCase):
    def test_successful_report_access_logs_are_silent_without_configuration(self):
        from sglang.srt.utils.common import (
            UvicornAccessLogFilter,
            _configure_uvicorn_access_log_filter,
        )

        config = {"handlers": {"access": {}}, "loggers": {"uvicorn.access": {}}}
        _configure_uvicorn_access_log_filter(config)
        name = "sglang_uvicorn_access_path_filter"
        self.assertIn(name, config["handlers"]["access"]["filters"])
        self.assertIn(name, config["loggers"]["uvicorn.access"]["filters"])
        options = dict(config["filters"][name])
        options.pop("()")
        access_filter = UvicornAccessLogFilter(**options)
        for method, path, status, visible in (
            ("POST", SNAPSHOT_PATH, 204, False),
            ("POST", SNAPSHOT_PATH + "?worker=1", 204, False),
            ("POST", SNAPSHOT_PATH, 400, True),
            ("POST", SNAPSHOT_PATH, 401, True),
            ("POST", SNAPSHOT_PATH, 500, True),
            ("POST", SNAPSHOT_PATH + "/other", 204, True),
            ("POST", "/generate", 204, True),
            ("POST", "/metrics", 200, True),
            ("GET", SNAPSHOT_PATH, 204, False),
            ("GET", SNAPSHOT_PATH, 401, True),
        ):
            with self.subTest(path=path, status=status):
                record = logging.LogRecord(
                    "uvicorn.access",
                    logging.INFO,
                    __file__,
                    0,
                    '%s - "%s %s HTTP/%s" %d',
                    ("127.0.0.1:1234", method, path, "1.1", status),
                    None,
                )
                self.assertEqual(access_filter.filter(record), visible)

    def test_configured_access_log_prefixes_still_apply(self):
        from sglang.srt.utils.common import UvicornAccessLogFilter

        record = logging.LogRecord(
            "uvicorn.access",
            logging.INFO,
            __file__,
            0,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:1234", "GET", "/metrics", "1.1", 200),
            None,
        )
        self.assertFalse(UvicornAccessLogFilter(["/metrics"]).filter(record))

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_metrics=True, api_key="test-key"
        )
        override.install()
        self.addCleanup(override.restore)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.environment = patch.dict(
            os.environ, {"PROMETHEUS_MULTIPROC_DIR": self.directory.name}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def scrape(self):
        registry = CollectorRegistry()
        registry.register(MooncakeClientCollector())
        text = generate_latest(registry).decode()
        return {
            sample.name: [
                candidate
                for family in text_string_to_metric_families(text)
                for candidate in family.samples
                if candidate.name == sample.name
            ]
            for family in text_string_to_metric_families(text)
            for sample in family.samples
        }

    def test_eight_clients_across_hosts_keep_counters_gauges_and_histograms(self):
        for index in range(1, 9):
            receive_snapshot(payload(index))
        samples = self.scrape()
        self.assertEqual(len(samples["mooncake_test_bytes_total"]), 8)
        self.assertEqual(sum(s.value for s in samples["mooncake_test_bytes_total"]), 96)
        self.assertEqual(len(samples["mooncake_test_latency_bucket"]), 16)
        self.assertEqual(
            sum(s.value for s in samples["mooncake_test_latency_count"]), 16
        )
        self.assertEqual(sum(s.value for s in samples["mooncake_test_latency_sum"]), 24)
        self.assertEqual(len(samples["mooncake_test_buffer"]), 8)
        self.assertEqual(
            {s.labels["kind"] for s in samples["mooncake_test_buffer"]}, {"read"}
        )
        self.assertEqual({s.value for s in samples["sglang:mooncake_client_up"]}, {1})

    def test_new_snapshot_replaces_old_and_out_of_order_report_is_ignored(self):
        receive_snapshot(payload(sequence=2))
        receive_snapshot(payload(sequence=1, metrics=METRICS.replace("12", "99")))
        self.assertEqual(self.scrape()["mooncake_test_bytes_total"][0].value, 12)
        receive_snapshot(payload(sequence=3, metrics=METRICS.replace("12", "24")))
        self.assertEqual(self.scrape()["mooncake_test_bytes_total"][0].value, 24)

    def test_expired_and_unavailable_clients_do_not_export_stale_metrics(self):
        with patch("time.monotonic", return_value=time.monotonic() - 31):
            receive_snapshot(payload(1))
        receive_snapshot(payload(2, metrics=None))
        receive_snapshot(payload(3))
        samples = self.scrape()
        self.assertEqual(len(samples["mooncake_test_bytes_total"]), 1)
        self.assertEqual(
            sorted(s.value for s in samples["sglang:mooncake_client_up"]), [0, 0, 1]
        )
        receive_snapshot(payload(1, sequence=2))
        self.assertEqual(len(self.scrape()["mooncake_test_bytes_total"]), 2)

    def test_invalid_payload_does_not_replace_healthy_snapshot(self):
        receive_snapshot(payload())
        for invalid in (
            {**payload(), "client_id": "../../bad"},
            {**payload(), "sequence": True},
            {**payload(), "metrics": 1},
            {**payload(), "labels": {}},
            payload(metrics="# TYPE sglang_requests counter\nsglang_requests 1\n"),
            payload(metrics='mooncake_test{mooncake_client="collision"} 1\n'),
            payload(metrics="not a valid metric\n"),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    receive_snapshot(invalid)
        self.assertEqual(self.scrape()["mooncake_test_bytes_total"][0].value, 12)

    def test_corrupt_client_file_does_not_break_healthy_clients(self):
        receive_snapshot(payload())
        (Path(self.directory.name) / "mooncake_clients" / "broken.json").write_text("{")
        self.assertEqual(len(self.scrape()["mooncake_test_bytes_total"]), 1)

    def test_http_receive_route_and_metrics_share_snapshot(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from sglang.srt.utils.common import add_prometheus_middleware

        app = FastAPI()
        add_prometheus_middleware(app)
        add_snapshot_route(app)
        with TestClient(app) as client:
            self.assertEqual(client.get(SNAPSHOT_PATH).status_code, 204)
            self.assertEqual(
                list(Path(self.directory.name).glob("mooncake_clients/*.json")), []
            )
            self.assertEqual(
                client.post(SNAPSHOT_PATH, json=payload()).status_code, 204
            )
            response = client.get("/metrics")
            self.assertEqual(response.status_code, 200)
            self.assertIn("mooncake_test_bytes_total", response.text)
            self.assertEqual(client.post(SNAPSHOT_PATH, json=[]).status_code, 400)
            self.assertEqual(client.post(SNAPSHOT_PATH, content="{").status_code, 400)
            with patch(
                "sglang.srt.observability.mooncake_client_metrics.MAX_SNAPSHOT_CLIENTS",
                1,
            ):
                self.assertEqual(
                    client.post(SNAPSHOT_PATH, json=payload(2)).status_code, 429
                )
            self.assertEqual(
                client.post(
                    SNAPSHOT_PATH, content=b"x" * (MAX_SNAPSHOT_BYTES + 1)
                ).status_code,
                413,
            )

    def test_metrics_only_server_does_not_expose_snapshot_writes(self):
        """Dummy health servers share the collector, never the ingestion route."""
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from sglang.srt.utils.common import add_prometheus_middleware

        app = FastAPI()
        add_prometheus_middleware(app)
        with TestClient(app) as client:
            self.assertEqual(client.get("/metrics").status_code, 200)
            self.assertEqual(client.get(SNAPSHOT_PATH).status_code, 404)
            self.assertEqual(
                client.post(SNAPSHOT_PATH, json=payload()).status_code, 404
            )
        self.assertEqual(list(Path(self.directory.name).rglob("*.json")), [])

    def test_http_body_limit_stops_unknown_length_stream_before_eof(self):
        """Chunked/unknown-length input must be rejected before reading the tail."""
        import asyncio

        from fastapi import FastAPI

        async def check(headers):
            app = FastAPI()
            add_snapshot_route(app)
            reads = 0
            responses = []
            chunk_size = 64 * 1024

            async def receive():
                nonlocal reads
                reads += 1
                if reads > MAX_SNAPSHOT_BYTES // chunk_size + 1:
                    self.fail("Receiver continued reading after crossing 8 MiB")
                return {
                    "type": "http.request",
                    "body": b"x" * chunk_size,
                    "more_body": True,
                }

            async def send(message):
                responses.append(message)

            await app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "scheme": "http",
                    "method": "POST",
                    "path": SNAPSHOT_PATH,
                    "raw_path": SNAPSHOT_PATH.encode(),
                    "root_path": "",
                    "query_string": b"",
                    "headers": headers,
                    "client": ("127.0.0.1", 1234),
                    "server": ("127.0.0.1", 80),
                },
                receive,
                send,
            )
            self.assertEqual(responses[0]["status"], 413)
            self.assertEqual(reads, MAX_SNAPSHOT_BYTES // chunk_size + 1)

        for headers in ([], [(b"transfer-encoding", b"chunked")]):
            with self.subTest(headers=headers):
                asyncio.run(check(headers))
        self.assertEqual(list(Path(self.directory.name).rglob("*.json")), [])

    def test_quota_rejects_new_clients_but_allows_existing_client_updates(self):
        """A full client registry must still accept replacement snapshots."""
        with patch(
            "sglang.srt.observability.mooncake_client_metrics.MAX_SNAPSHOT_CLIENTS", 1
        ):
            receive_snapshot(payload())
            with self.assertRaises(SnapshotQuotaExceeded):
                receive_snapshot(payload(2))
            receive_snapshot(payload(sequence=2, metrics=METRICS.replace("12", "24")))
        self.assertEqual(self.scrape()["mooncake_test_bytes_total"][0].value, 24)

    def test_ingest_with_spare_capacity_does_not_read_other_client_snapshots(self):
        """Normal publication must not parse every client's metrics under the lock."""
        receive_snapshot(payload(1))
        receive_snapshot(payload(2))
        with patch.object(
            Path,
            "read_text",
            side_effect=AssertionError("Ingest read an unrelated client snapshot"),
        ):
            receive_snapshot(payload(3))
        self.assertEqual(len(self.scrape()["mooncake_test_bytes_total"]), 3)

    def test_quota_pressure_expires_old_snapshots_and_retries_publication(self):
        """Quota pressure reclaims stale clients and tmp files without a scrape."""
        for limit_name in ("MAX_SNAPSHOT_CLIENTS", "MAX_SNAPSHOT_TOTAL_BYTES"):
            with (
                self.subTest(limit=limit_name),
                tempfile.TemporaryDirectory() as directory,
                patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": directory}),
            ):
                with patch("time.monotonic", return_value=time.monotonic() - 301):
                    receive_snapshot(payload(1))
                snapshots = Path(directory) / "mooncake_clients"
                old = snapshots / f"{1:032x}.json"
                orphan = snapshots / "tmp-orphan"
                orphan.write_bytes(b"abandoned")
                os.utime(orphan, (time.time() - 301, time.time() - 301))
                limit = (
                    1
                    if limit_name == "MAX_SNAPSHOT_CLIENTS"
                    else old.stat().st_size + 128
                )
                with patch(
                    "sglang.srt.observability.mooncake_client_metrics." + limit_name,
                    limit,
                ):
                    receive_snapshot(payload(2))
                self.assertFalse(old.exists())
                self.assertFalse(orphan.exists())
                samples = self.scrape()["mooncake_test_bytes_total"]
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0].labels["mooncake_client"], f"{2:032x}")

    def test_expired_client_can_restart_sequence_before_background_sweep(self):
        """Removing full sweeps must preserve expiry of the client being replaced."""
        with patch("time.monotonic", return_value=time.monotonic() - 301):
            receive_snapshot(payload(sequence=2))
        receive_snapshot(payload(sequence=1, metrics=METRICS.replace("12", "24")))
        self.assertEqual(self.scrape()["mooncake_test_bytes_total"][0].value, 24)

    def test_byte_quota_counts_old_target_and_temporary_files(self):
        """Publication must fit before the old target can be atomically replaced."""
        receive_snapshot(payload())
        directory = Path(self.directory.name) / "mooncake_clients"
        target = next(directory.glob("*.json"))
        before = target.read_bytes()
        orphan = directory / "tmp-orphan"
        orphan.write_bytes(b"x" * 4096)
        for budget in (len(before) + 4096, len(before) + 1):
            with (
                self.subTest(budget=budget),
                patch(
                    "sglang.srt.observability.mooncake_client_metrics.MAX_SNAPSHOT_TOTAL_BYTES",
                    budget,
                ),
            ):
                with self.assertRaises(SnapshotQuotaExceeded):
                    receive_snapshot(payload(sequence=2))
                self.assertEqual(target.read_bytes(), before)
            orphan.unlink(missing_ok=True)
        self.assertEqual(list(directory.glob("tmp*")), [])

    def test_file_quota_counts_abandoned_temporary_files(self):
        """An orphan cannot bypass the file-count limit while awaiting expiry."""
        receive_snapshot(payload())
        directory = Path(self.directory.name) / "mooncake_clients"
        (directory / "tmp-orphan").write_bytes(b"abandoned")
        with patch(
            "sglang.srt.observability.mooncake_client_metrics.MAX_SNAPSHOT_CLIENTS", 1
        ):
            with self.assertRaises(SnapshotQuotaExceeded):
                receive_snapshot(payload(sequence=2))
        self.assertEqual(len(list(directory.iterdir())), 3)  # target, orphan, lock

    def test_concurrent_receivers_share_client_quota(self):
        """Separate receiver threads must serialize their check and publication."""
        from concurrent.futures import ThreadPoolExecutor

        def receive(index):
            try:
                receive_snapshot(payload(index))
                return True
            except SnapshotQuotaExceeded:
                return False

        with (
            patch(
                "sglang.srt.observability.mooncake_client_metrics.MAX_SNAPSHOT_CLIENTS",
                1,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            self.assertEqual(sorted(executor.map(receive, (1, 2))), [False, True])
        self.assertEqual(len(list(Path(self.directory.name).rglob("*.json"))), 1)

    def test_quota_uses_serialized_bytes_including_json_escaping(self):
        """Non-ASCII labels consume more disk bytes than their UTF-8 metric text."""
        value = payload(metrics='mooncake_test{reason="é"} 1\n')
        with (
            patch("time.monotonic", return_value=100.0),
            patch("time.time", return_value=100.0),
        ):
            receive_snapshot(value)
            path = next(Path(self.directory.name).rglob("*.json"))
            serialized_size = path.stat().st_size
            path.unlink()
            with patch(
                "sglang.srt.observability.mooncake_client_metrics.MAX_SNAPSHOT_TOTAL_BYTES",
                serialized_size - 1,
            ):
                with self.assertRaises(SnapshotQuotaExceeded):
                    receive_snapshot(value)
        self.assertEqual(list(Path(self.directory.name).rglob("tmp*")), [])
        self.assertEqual(list(Path(self.directory.name).rglob("*.json")), [])

    def test_background_expiry_does_not_require_metrics_scrapes(self):
        """An idle receiver reclaims expired clients and abandoned temporary files."""
        import asyncio

        with patch("time.monotonic", return_value=time.monotonic() - 301):
            receive_snapshot(payload())
        directory = Path(self.directory.name) / "mooncake_clients"
        orphan = directory / "tmp-orphan"
        orphan.write_bytes(b"abandoned")
        os.utime(orphan, (time.time() - 301, time.time() - 301))

        async def check():
            async with snapshot_expiry():

                async def wait_for_expiry():
                    while list(directory.glob("*.json")) or orphan.exists():
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_for_expiry(), timeout=2)

        asyncio.run(check())
        receive_snapshot(payload(sequence=2))
        expire_snapshots()
        self.assertTrue(next(directory.glob("*.json")).exists())

    def test_reporter_retries_failure_and_sends_cumulative_snapshot(self):
        store = Mock()
        store.get_client_metrics.return_value = METRICS
        args = SimpleNamespace(api_key="test-key", ssl_verify=lambda: True)
        with patch("threading.Thread"):
            reporter = MooncakeClientMetricsReporter(
                store, "http://127.0.0.1:1234", args, 7
            )
        reporter._stopped.wait = Mock(side_effect=[False, True])
        reporter._stopped.is_set = Mock(side_effect=[False, False, False, True])
        session = Mock()
        session.headers = {}
        session.get.return_value = SimpleNamespace(status_code=204)
        session.post.side_effect = [
            OSError("not ready"),
            SimpleNamespace(status_code=204),
        ]
        with patch("requests.Session") as create_session:
            create_session.return_value.__enter__.return_value = session
            reporter._run()
        self.assertFalse(session.trust_env)
        self.assertEqual(session.headers["Authorization"], "Bearer test-key")
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(
            [c.kwargs["json"]["sequence"] for c in session.post.call_args_list], [1, 2]
        )
        self.assertEqual(session.post.call_args.kwargs["json"]["metrics"], METRICS)
        self.assertEqual(
            session.post.call_args.kwargs["json"]["labels"]["mooncake_rank"], "7"
        )
        reporter.close()
        reporter._thread.join.assert_called_once()

    def make_reporter(self):
        scope = get_serving().override(ssl_ca_certs="ca.pem")
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        store = Mock()
        store.get_client_metrics.return_value = METRICS
        args = SimpleNamespace(api_key="test-key", ssl_verify=lambda: "ca.pem")
        with patch("threading.Thread"):
            reporter = MooncakeClientMetricsReporter(
                store, "https://127.0.0.1:1234", args, 7
            )
        reporter._stopped.wait = Mock(return_value=False)
        session = Mock()
        session.headers = {}
        session.get.return_value = SimpleNamespace(status_code=204)
        return reporter, store, session

    def run_reports(self, reporter, session, results):
        results = iter(results)

        def post(*args, **kwargs):
            result, last = next(results)
            if last:
                reporter._stopped.set()
            if isinstance(result, Exception):
                raise result
            return SimpleNamespace(status_code=result)

        session.post.side_effect = post
        with patch("requests.Session") as create_session:
            create_session.return_value.__enter__.return_value = session
            reporter._run()

    def test_reporter_waits_for_receiver_without_collecting_or_posting(self):
        reporter, store, session = self.make_reporter()

        def get(*args, **kwargs):
            store.get_client_metrics.assert_not_called()
            session.post.assert_not_called()
            if session.get.call_count <= 2:
                raise requests.ConnectionError("connection refused")
            if session.get.call_count == 3:
                raise requests.Timeout("lost readiness response")
            return SimpleNamespace(status_code=204)

        session.get.side_effect = get
        with patch("sglang.srt.observability.mooncake_client_metrics.logger") as log:
            self.run_reports(reporter, session, [(204, True)])
        self.assertEqual(session.get.call_count, 4)
        self.assertEqual(session.post.call_count, 1)
        store.get_client_metrics.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in reporter._stopped.wait.call_args_list],
            [1, 1, 1, 5],
        )
        self.assertEqual(session.get.call_args.kwargs["verify"], "ca.pem")
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])
        self.assertEqual(session.headers["Authorization"], "Bearer test-key")
        log.warning.assert_not_called()
        log.info.assert_called_once()

    def test_reporter_can_stop_while_waiting_for_receiver(self):
        reporter, store, session = self.make_reporter()
        session.get.side_effect = requests.ConnectionError("connection refused")
        reporter._stopped.wait.side_effect = lambda _: reporter._stopped.set() or True
        with patch("requests.Session") as create_session:
            create_session.return_value.__enter__.return_value = session
            reporter._run()
        store.get_client_metrics.assert_not_called()
        session.post.assert_not_called()
        reporter.close()
        reporter._thread.join.assert_called_once()

    def test_startup_wait_warns_after_grace_and_keeps_retrying(self):
        reporter, store, session = self.make_reporter()
        session.get.side_effect = [
            *[requests.ConnectionError("connection refused") for _ in range(7)],
            SimpleNamespace(status_code=204),
        ]
        with (
            patch(
                "sglang.srt.observability.mooncake_client_metrics.time.monotonic",
                side_effect=[0, 0, 299, 300, 301, 359, 360, 361],
            ),
            patch("sglang.srt.observability.mooncake_client_metrics.logger") as log,
        ):
            self.assertTrue(reporter._wait_for_receiver(session))
        self.assertEqual(log.warning.call_count, 2)
        self.assertEqual(session.get.call_count, 8)
        store.get_client_metrics.assert_not_called()
        session.post.assert_not_called()

    def test_auth_route_and_tls_errors_are_not_hidden_by_startup_grace(self):
        for result in (
            SimpleNamespace(status_code=401),
            SimpleNamespace(status_code=404),
            requests.exceptions.SSLError("bad certificate"),
        ):
            with self.subTest(result=result):
                reporter, _, session = self.make_reporter()
                session.get.side_effect = [
                    result,
                    result,
                    SimpleNamespace(status_code=204),
                ]
                with (
                    patch(
                        "sglang.srt.observability.mooncake_client_metrics.time.monotonic",
                        side_effect=[0, 0, 1],
                    ),
                    patch(
                        "sglang.srt.observability.mooncake_client_metrics.logger"
                    ) as log,
                ):
                    self.assertTrue(reporter._wait_for_receiver(session))
                log.warning.assert_called_once()
                self.assertEqual(
                    [call.args[0] for call in reporter._stopped.wait.call_args_list],
                    [5, 5],
                )

    def test_runtime_refusals_do_not_return_to_startup_wait(self):
        reporter, _, session = self.make_reporter()
        with patch("sglang.srt.observability.mooncake_client_metrics.logger") as log:
            self.run_reports(
                reporter,
                session,
                [
                    (204, False),
                    (requests.ConnectionError("connection refused"), False),
                    (requests.ConnectionError("connection refused"), False),
                    (204, True),
                ],
            )
        session.get.assert_called_once()
        self.assertEqual(session.post.call_count, 4)
        log.warning.assert_called_once()
        self.assertEqual(log.info.call_count, 2)
        self.assertEqual(log.info.call_args.args[1], 2)
        self.assertTrue(
            all(call.args == (5,) for call in reporter._stopped.wait.call_args_list)
        )

    def test_runtime_warning_frequency_and_success_reset_are_preserved(self):
        for outcomes, expected in (
            ([requests.ConnectionError("refused")] * 24, 3),
            ([requests.ConnectionError("refused")] * 12 + [204, 500], 3),
        ):
            with self.subTest(expected=expected, attempts=len(outcomes)):
                reporter, _, session = self.make_reporter()
                with patch(
                    "sglang.srt.observability.mooncake_client_metrics.logger"
                ) as log:
                    self.run_reports(
                        reporter,
                        session,
                        [
                            (result, i == len(outcomes) - 1)
                            for i, result in enumerate(outcomes)
                        ],
                    )
                self.assertEqual(log.warning.call_count, expected)
                session.get.assert_called_once()
                self.assertEqual(session.post.call_count, len(outcomes))

    def test_http_snapshot_route_uses_existing_api_key_auth(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from sglang.srt.utils.auth import add_api_key_middleware
        from sglang.srt.utils.common import add_prometheus_middleware

        app = FastAPI()
        add_prometheus_middleware(app)
        add_snapshot_route(app)
        add_api_key_middleware(app, api_key="test-key", admin_api_key=None)
        with TestClient(app) as client:
            self.assertEqual(client.get(SNAPSHOT_PATH).status_code, 401)
            self.assertEqual(
                client.get(
                    SNAPSHOT_PATH, headers={"Authorization": "Bearer test-key"}
                ).status_code,
                204,
            )
            self.assertEqual(
                client.post(SNAPSHOT_PATH, json=payload()).status_code, 401
            )
            self.assertEqual(
                client.post(
                    SNAPSHOT_PATH,
                    json=payload(),
                    headers={"Authorization": "Bearer test-key"},
                ).status_code,
                204,
            )
            self.assertEqual(client.get("/metrics").status_code, 200)

    def test_old_package_warns_without_starting_reporter(self):
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(
                start_client_metrics_reporter(
                    object(), SimpleNamespace(enable_metrics=True), "host:80", 0
                )
            )

    def test_legacy_grpc_uses_http_sidecar(self):
        args = SimpleNamespace(
            enable_metrics=True,
            ssl_certfile=None,
            smg_grpc_mode=True,
            grpc_mode=False,
            smg_http_sidecar_port=30002,
            port=30000,
        )
        with (
            patch(
                "sglang.srt.observability.mooncake_client_metrics.MooncakeClientMetricsReporter"
            ) as reporter,
            get_context().override_server_args(
                enable_metrics=True,
                smg_grpc_mode=True,
                port=30000,
                smg_http_sidecar_port=30002,
            ),
        ):
            start_client_metrics_reporter(Mock(), args, "[::1]:30000", 0)
            self.assertEqual(reporter.call_args.args[1], "http://[::1]:30002")

    def test_grpc_sidecar_receives_authenticated_snapshots(self):
        import asyncio

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from sglang.srt.entrypoints.grpc_server import _add_metrics_routes

        async def check():
            app = web.Application(client_max_size=MAX_SNAPSHOT_BYTES)
            get_context().override(
                "metrics-sidecar-test", api_key="key", admin_api_key=None
            )
            _add_metrics_routes(app)
            async with TestClient(TestServer(app)) as client:
                response = await client.get(SNAPSHOT_PATH)
                self.assertEqual(response.status, 401)
                response = await client.get(
                    SNAPSHOT_PATH, headers={"Authorization": "Bearer key"}
                )
                self.assertEqual(response.status, 204)
                self.assertEqual(
                    list(Path(self.directory.name).glob("mooncake_clients/*.json")), []
                )
                response = await client.post(SNAPSHOT_PATH, json=payload())
                self.assertEqual(response.status, 401)
                response = await client.post(
                    SNAPSHOT_PATH,
                    json=payload(),
                    headers={"Authorization": "Bearer key"},
                )
                self.assertEqual(response.status, 204)
                with patch(
                    "sglang.srt.observability.mooncake_client_metrics.MAX_SNAPSHOT_CLIENTS",
                    1,
                ):
                    response = await client.post(
                        SNAPSHOT_PATH,
                        json=payload(2),
                        headers={"Authorization": "Bearer key"},
                    )
                    self.assertEqual(response.status, 429)
                response = await client.get("/metrics")
                self.assertEqual(response.status, 200)
                self.assertIn("mooncake_test_bytes_total", await response.text())

        asyncio.run(check())

    def test_grpc_sidecar_limits_chunked_body_before_json_parsing(self):
        """The route cap must work even without aiohttp's whole-body size guard."""
        import asyncio

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from sglang.srt.entrypoints.grpc_server import _add_metrics_routes

        async def check():
            observed = {}

            @web.middleware
            async def record_framing(request, handler):
                observed["length"] = request.content_length
                observed["encoding"] = request.headers.get("Transfer-Encoding")
                return await handler(request)

            app = web.Application(client_max_size=0, middlewares=[record_framing])
            _add_metrics_routes(app)

            async def chunks():
                yield json.dumps(payload()).encode()
                for _ in range(MAX_SNAPSHOT_BYTES // (64 * 1024)):
                    yield b" " * (64 * 1024)

            async with TestClient(TestServer(app)) as client:
                response = await client.post(
                    SNAPSHOT_PATH,
                    data=chunks(),
                    chunked=True,
                    headers={"Authorization": "Bearer test-key"},
                )
                self.assertEqual(response.status, 413)
                self.assertIsNone(observed["length"])
                self.assertEqual(observed["encoding"], "chunked")

        asyncio.run(check())
        self.assertEqual(list(Path(self.directory.name).rglob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
