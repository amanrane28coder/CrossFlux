import math
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from src.metrics import MetricsServer, PrometheusRegistry, metrics_server_from_env


def test_registry_renders_typed_metrics_and_escapes_labels():
    registry = PrometheusRegistry()
    registry.increment(
        "crossflux_feed_messages_total", "Messages received.", 2,
        labels={"venue": 'kraken"\nwest'},
    )
    registry.set_gauge(
        "crossflux_pnl_account_units", "Simulated PnL.", -2.5,
    )

    output = registry.render()
    assert "# TYPE crossflux_feed_messages_total counter" in output
    assert 'crossflux_feed_messages_total{venue="kraken\\"\\nwest"} 2' in output
    assert "# TYPE crossflux_pnl_account_units gauge" in output
    assert "crossflux_pnl_account_units -2.5" in output


def test_registry_rejects_invalid_samples():
    registry = PrometheusRegistry()
    with pytest.raises(ValueError, match="non-negative"):
        registry.increment("crossflux_count_total", "counter", -1)
    with pytest.raises(ValueError, match="finite"):
        registry.set_gauge("crossflux_value", "gauge", math.inf)
    with pytest.raises(ValueError, match="metric name"):
        registry.set_gauge("not valid", "bad name", 1)


def test_metrics_server_serves_only_metrics_endpoint():
    registry = PrometheusRegistry()
    registry.set_gauge("crossflux_up", "Exporter test gauge.", 1)
    try:
        server = MetricsServer(registry, "127.0.0.1", 0)
    except PermissionError:
        pytest.skip("sandbox policy does not allow binding a loopback test server")
    server.start()
    host, port = server._server.server_address
    try:
        with urlopen(f"http://{host}:{port}/metrics", timeout=2) as response:
            assert response.status == 200
            assert "crossflux_up 1" in response.read().decode()
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://{host}:{port}/", timeout=2)
        assert error.value.code == 404
    finally:
        server.close()


def test_metrics_server_configuration_is_opt_in_and_validated(monkeypatch):
    assert metrics_server_from_env(PrometheusRegistry()) is None
    monkeypatch.setenv("CROSSFLUX_METRICS_ENABLED", "true")
    monkeypatch.setenv("CROSSFLUX_METRICS_PORT", "abc")
    with pytest.raises(ValueError, match="integer port"):
        metrics_server_from_env(PrometheusRegistry())
    monkeypatch.setenv("CROSSFLUX_METRICS_PORT", "0")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        metrics_server_from_env(PrometheusRegistry())
