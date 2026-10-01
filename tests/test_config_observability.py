import json
import logging

import pytest

from src.config import RiskSettings
from src.observability import JsonFormatter


def test_risk_settings_require_positive_capital():
    with pytest.raises(ValueError, match="CROSSFLUX_INITIAL_CAPITAL"):
        RiskSettings.from_env(environ={})

    with pytest.raises(ValueError, match="greater than zero"):
        RiskSettings.from_env(environ={"CROSSFLUX_INITIAL_CAPITAL": "0"})


def test_risk_settings_validate_finite_values_and_drawdown_range():
    with pytest.raises(ValueError, match="finite"):
        RiskSettings.from_env(environ={
            "CROSSFLUX_INITIAL_CAPITAL": "nan",
            "CROSSFLUX_MAX_DRAWDOWN_PCT": "5",
        })

    with pytest.raises(ValueError, match="0.0..100.0"):
        RiskSettings.from_env(environ={
            "CROSSFLUX_INITIAL_CAPITAL": "100000",
            "CROSSFLUX_MAX_DRAWDOWN_PCT": "101",
        })


def test_risk_settings_use_validated_environment_and_defaults():
    settings = RiskSettings.from_env(environ={"CROSSFLUX_INITIAL_CAPITAL": "100000"})
    assert settings.initial_capital == 100000
    assert settings.max_drawdown_pct == 5

    settings = RiskSettings.from_env(
        initial_capital=50000,
        max_drawdown_pct=2.5,
        environ={},
    )
    assert settings.initial_capital == 50000
    assert settings.max_drawdown_pct == 2.5


def test_json_formatter_preserves_context_and_exception_without_overrides():
    try:
        raise RuntimeError("reconcile failed")
    except RuntimeError:
        record = logging.LogRecord(
            name="crossflux.orders",
            level=logging.ERROR,
            pathname=__file__,
            lineno=10,
            msg="order state unknown",
            args=(),
            exc_info=__import__("sys").exc_info(),
        )
    record.venue = "binance"
    record.order_id = "client-123"
    record.level = "forged"

    payload = json.loads(JsonFormatter().format(record))
    assert payload["logger"] == "crossflux.orders"
    assert payload["level"] == "ERROR"
    assert payload["message"] == "order state unknown"
    assert payload["venue"] == "binance"
    assert payload["order_id"] == "client-123"
    assert "RuntimeError: reconcile failed" in payload["exception"]
