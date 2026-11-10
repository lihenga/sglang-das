"""Forward embedded Mooncake metrics to the serving process, including remote ranks."""

import asyncio
import json
import logging
import os
import re
import socket
import tempfile
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = "/internal/mooncake_client_metrics"
REPORT_INTERVAL = 5
REPORT_TIMEOUT = 30
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
CLIENT_LABELS = ("mooncake_client", "mooncake_host", "mooncake_pid", "mooncake_rank")


def _snapshot_directory():
    # All HTTP/tokenizer workers already share this per-server directory.
    directory = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"]) / "mooncake_clients"
    directory.mkdir(exist_ok=True)
    return directory


def _parse_metrics(text):
    from prometheus_client.parser import text_string_to_metric_families

    metrics = list(text_string_to_metric_families(text))
    for metric in metrics:
        if not metric.name.startswith("mooncake_"):
            raise ValueError("Expected Mooncake client metric names")
        for sample in metric.samples:
            if any(label in sample.labels for label in CLIENT_LABELS):
                raise ValueError("Client metric uses a reserved source label")
    return metrics


def receive_snapshot(payload):
    """Validate before atomically publishing to other HTTP workers."""
    client_id = payload.get("client_id")
    if not isinstance(client_id, str) or not re.fullmatch(r"[0-9a-f]{32}", client_id):
        raise ValueError("Invalid client ID")
    sequence = payload.get("sequence")
    if type(sequence) is not int or sequence <= 0:
        raise ValueError("Invalid snapshot sequence")
    labels = payload.get("labels")
    if (
        not isinstance(labels, dict)
        or set(labels) != set(CLIENT_LABELS)
        or labels["mooncake_client"] != client_id
        or any(
            not isinstance(value, str) or len(value) > 256 for value in labels.values()
        )
    ):
        raise ValueError("Invalid client labels")
    text = payload.get("metrics")
    if text is not None:
        if not isinstance(text, str) or len(text.encode()) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Invalid client metrics")
        _parse_metrics(text)
    snapshot = {
        "labels": labels,
        "metrics": text,
        "received": time.monotonic(),
        "timestamp": time.time(),
        "sequence": sequence,
    }
    directory = _snapshot_directory()
    fd, temporary = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(snapshot, stream)
        import fcntl

        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            target = directory / f"{client_id}.json"
            if target.exists():
                previous = json.loads(target.read_text())
                if previous["sequence"] >= sequence:
                    return
                if previous["labels"] != labels:
                    raise ValueError("Client identity changed")
            os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class MooncakeClientCollector:
    def collect(self):
        from prometheus_client.core import GaugeMetricFamily

        up = GaugeMetricFamily(
            "sglang:mooncake_client_up",
            "Embedded client metrics available and reported within 30 seconds",
            labels=CLIENT_LABELS,
        )
        reported = GaugeMetricFamily(
            "sglang:mooncake_client_last_report_timestamp_seconds",
            "Time the serving process last received a client metrics snapshot",
            labels=CLIENT_LABELS,
        )
        families = {}
        for path in _snapshot_directory().glob("*.json"):
            try:
                snapshot = json.loads(path.read_text())
                age = time.monotonic() - snapshot["received"]
                if age > 300:
                    import fcntl

                    with (path.parent / ".lock").open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        if (
                            json.loads(path.read_text())["sequence"]
                            == snapshot["sequence"]
                        ):
                            path.unlink(missing_ok=True)
                    continue
                labels = snapshot["labels"]
                values = [labels[name] for name in CLIENT_LABELS]
                available = age <= REPORT_TIMEOUT and snapshot["metrics"] is not None
                metrics = _parse_metrics(snapshot["metrics"]) if available else []
                if any(
                    metric.name in families
                    and families[metric.name].type != metric.type
                    for metric in metrics
                ):
                    raise ValueError("Conflicting client metric types")
                up.add_metric(values, int(available))
                reported.add_metric(values, snapshot["timestamp"])
                for metric in metrics:
                    metric.samples = [
                        sample._replace(labels={**sample.labels, **labels})
                        for sample in metric.samples
                    ]
                    if metric.name in families:
                        families[metric.name].samples.extend(metric.samples)
                    else:
                        families[metric.name] = metric
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("Cannot read Mooncake client metrics snapshot %s", path)
        yield up
        yield reported
        yield from families.values()


def add_snapshot_route(app):
    from starlette.requests import Request
    from starlette.responses import Response

    async def receive(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_SNAPSHOT_BYTES:
                return Response(status_code=413)
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Expected a snapshot object")
            await asyncio.to_thread(receive_snapshot, payload)
        except (ValueError, TypeError):
            return Response(status_code=400)
        return Response(status_code=204)

    app.add_route(SNAPSHOT_PATH, receive, methods=["POST"])


class MooncakeClientMetricsReporter:
    def __init__(self, store, url, server_args, rank):
        self._store = store
        self._url = url + SNAPSHOT_PATH
        self._verify = server_args.ssl_verify()
        self._api_key = server_args.api_key
        client_id = uuid.uuid4().hex
        self._labels = dict(
            zip(
                CLIENT_LABELS,
                (client_id, socket.gethostname(), str(os.getpid()), str(rank)),
            )
        )
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="mooncake-client-metrics", daemon=True
        )
        self._thread.start()

    def _run(self):
        import requests

        with requests.Session() as session:
            session.trust_env = False
            if self._api_key:
                session.headers["Authorization"] = f"Bearer {self._api_key}"
            failures = 0
            sequence = 0
            while not self._stopped.is_set():
                try:
                    metrics = self._store.get_client_metrics()
                    sequence += 1
                    response = session.post(
                        self._url,
                        json={
                            "client_id": self._labels["mooncake_client"],
                            "labels": self._labels,
                            "metrics": metrics,
                            "sequence": sequence,
                        },
                        timeout=(2, 5),
                        verify=self._verify,
                        allow_redirects=False,
                    )
                    if response.status_code != 204:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    failures = 0
                except Exception as error:
                    failures += 1
                    if failures == 1 or failures % 12 == 0:
                        logger.warning(
                            "Mooncake client metrics report failed: %s", error
                        )
                self._stopped.wait(REPORT_INTERVAL)

    def close(self):
        self._stopped.set()
        self._thread.join()


def start_client_metrics_reporter(store, server_args, service_endpoint, rank):
    if not server_args.enable_metrics:
        return None
    if not callable(getattr(store, "get_client_metrics", None)):
        logger.warning(
            "Embedded Mooncake metrics forwarding needs a Mooncake package with "
            "get_client_metrics support; rebuild/install the matching package."
        )
        return None
    scheme = "https" if server_args.ssl_certfile else "http"
    if server_args.smg_grpc_mode or server_args.grpc_mode:
        from sglang.srt.utils.network import NetworkAddress

        endpoint = NetworkAddress.parse(service_endpoint)
        port = server_args.smg_http_sidecar_port or server_args.port + 1
        service_endpoint = NetworkAddress(endpoint.host, port).to_host_port_str()
        scheme = "http"
    return MooncakeClientMetricsReporter(
        store, f"{scheme}://{service_endpoint}", server_args, rank
    )
