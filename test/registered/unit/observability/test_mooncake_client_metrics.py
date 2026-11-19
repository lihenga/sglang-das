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
    receive_snapshot,
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


class TestMooncakeClientMetrics(unittest.TestCase):
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
            self.assertEqual(
                client.post(
                    SNAPSHOT_PATH, content=b"x" * (MAX_SNAPSHOT_BYTES + 1)
                ).status_code,
                413,
            )

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
        with patch(
            "sglang.srt.observability.mooncake_client_metrics.MooncakeClientMetricsReporter"
        ) as reporter:
            start_client_metrics_reporter(Mock(), args, "[::1]:30000", 0)
            self.assertEqual(reporter.call_args.args[1], "http://[::1]:30002")

    def test_grpc_sidecar_receives_authenticated_snapshots(self):
        import asyncio

        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from sglang.srt.entrypoints.grpc_server import _add_metrics_routes

        async def check():
            app = web.Application(client_max_size=MAX_SNAPSHOT_BYTES)
            _add_metrics_routes(app, SimpleNamespace(api_key="key", admin_api_key=None))
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
                response = await client.get("/metrics")
                self.assertEqual(response.status, 200)
                self.assertIn("mooncake_test_bytes_total", await response.text())

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
