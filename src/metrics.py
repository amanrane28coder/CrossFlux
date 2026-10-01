"""Dependency-free Prometheus text exporter for the Python simulation."""
from __future__ import annotations

import os
import re
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping

from src.config import METRICS_ENABLED_ENV, METRICS_HOST_ENV, METRICS_PORT_ENV

_METRIC_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class PrometheusRegistry:
    """Thread-safe registry with strict names and escaped label values."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._definitions: dict[str, tuple[str, str]] = {}
        self._samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    def _register(self, name: str, kind: str, help_text: str) -> None:
        if not _METRIC_NAME.fullmatch(name):
            raise ValueError(f"invalid Prometheus metric name: {name!r}")
        if kind not in {"counter", "gauge"}:
            raise ValueError(f"unsupported metric type: {kind!r}")
        definition = (kind, help_text)
        previous = self._definitions.get(name)
        if previous is not None and previous != definition:
            raise ValueError(f"metric {name!r} registered with conflicting metadata")
        self._definitions[name] = definition

    @staticmethod
    def _label_key(labels: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
        pairs = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        if any(not _LABEL_NAME.fullmatch(key) for key, _ in pairs):
            raise ValueError("invalid Prometheus label name")
        return pairs

    def increment(
        self,
        name: str,
        help_text: str,
        amount: float = 1.0,
        *,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("counter increments must be finite and non-negative")
        key = (name, self._label_key(labels))
        with self._lock:
            self._register(name, "counter", help_text)
            self._samples[key] = self._samples.get(key, 0.0) + amount

    def set_gauge(
        self,
        name: str,
        help_text: str,
        value: float,
        *,
        labels: Mapping[str, object] | None = None,
    ) -> None:
        if not math.isfinite(value):
            raise ValueError("gauge values must be finite")
        key = (name, self._label_key(labels))
        with self._lock:
            self._register(name, "gauge", help_text)
            self._samples[key] = float(value)

    def render(self) -> str:
        with self._lock:
            definitions = dict(self._definitions)
            samples = dict(self._samples)
        lines: list[str] = []
        for name in sorted(definitions):
            kind, help_text = definitions[name]
            safe_help = help_text.replace("\\", "\\\\").replace("\n", "\\n")
            lines.extend((f"# HELP {name} {safe_help}", f"# TYPE {name} {kind}"))
            for (sample_name, labels), value in sorted(samples.items()):
                if sample_name != name:
                    continue
                label_text = ""
                if labels:
                    escaped = [f'{key}="{_escape_label(val)}"' for key, val in labels]
                    label_text = "{" + ",".join(escaped) + "}"
                lines.append(f"{name}{label_text} {value:.12g}")
        return "\n".join(lines) + ("\n" if lines else "")


class _MetricsHandler(BaseHTTPRequestHandler):
    registry: PrometheusRegistry

    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_error(404)
            return
        body = self.registry.render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        # Keep the endpoint quiet; application logs already record lifecycle.
        return


class MetricsServer:
    def __init__(self, registry: PrometheusRegistry, host: str, port: int) -> None:
        handler = type("BoundMetricsHandler", (_MetricsHandler,), {"registry": registry})
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def metrics_server_from_env(registry: PrometheusRegistry) -> MetricsServer | None:
    enabled = os.environ.get(METRICS_ENABLED_ENV, "false").strip().lower()
    if enabled not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{METRICS_ENABLED_ENV} must be true or false")
    if enabled not in {"true", "1", "yes"}:
        return None

    host = os.environ.get(METRICS_HOST_ENV, "127.0.0.1")
    if not host.strip():
        raise ValueError(f"{METRICS_HOST_ENV} must not be empty")
    try:
        port = int(os.environ.get(METRICS_PORT_ENV, "9108"))
    except ValueError as exc:
        raise ValueError(f"{METRICS_PORT_ENV} must be an integer port") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{METRICS_PORT_ENV} must be between 1 and 65535")
    return MetricsServer(registry, host, port)
