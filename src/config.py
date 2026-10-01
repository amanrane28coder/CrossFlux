"""Validated Python-side environment configuration.

This module centralizes the names and parsing rules for settings shared across
Python entry points. C++ reads the same documented variable names directly.
Secrets are intentionally not part of this configuration surface yet because
the current C++ order path is simulated.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

INITIAL_CAPITAL_ENV = "CROSSFLUX_INITIAL_CAPITAL"
MAX_DRAWDOWN_ENV = "CROSSFLUX_MAX_DRAWDOWN_PCT"
FEE_PRESET_ENV = "CROSSFLUX_FEE_PRESET"
OBI_PROFILE_ENV = "CROSSFLUX_OBI_PROFILE"
FRICTION_PRESET_ENV = "CROSSFLUX_FRICTION_PRESET"
LOG_LEVEL_ENV = "CROSSFLUX_LOG_LEVEL"
LOG_FORMAT_ENV = "CROSSFLUX_LOG_FORMAT"
METRICS_ENABLED_ENV = "CROSSFLUX_METRICS_ENABLED"
METRICS_HOST_ENV = "CROSSFLUX_METRICS_HOST"
METRICS_PORT_ENV = "CROSSFLUX_METRICS_PORT"


def _finite_float(name: str, raw: str, *, minimum: float, maximum: float | None = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be finite and {bounds}")
    return value


@dataclass(frozen=True, slots=True)
class RiskSettings:
    initial_capital: float
    max_drawdown_pct: float

    @classmethod
    def from_env(
        cls,
        *,
        initial_capital: float | None = None,
        max_drawdown_pct: float | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "RiskSettings":
        env = os.environ if environ is None else environ
        if initial_capital is None:
            raw_capital = env.get(INITIAL_CAPITAL_ENV)
            if raw_capital is None:
                raise ValueError(
                    f"Set {INITIAL_CAPITAL_ENV} in the PnL quote currency before starting."
                )
            initial_capital = _finite_float(INITIAL_CAPITAL_ENV, raw_capital, minimum=0.0)
        else:
            initial_capital = _finite_float(INITIAL_CAPITAL_ENV, initial_capital, minimum=0.0)
        if initial_capital <= 0.0:
            raise ValueError(f"{INITIAL_CAPITAL_ENV} must be greater than zero")

        if max_drawdown_pct is None:
            raw_drawdown = env.get(MAX_DRAWDOWN_ENV, "5.0")
            max_drawdown_pct = _finite_float(
                MAX_DRAWDOWN_ENV, raw_drawdown, minimum=0.0, maximum=100.0
            )
        else:
            max_drawdown_pct = _finite_float(
                MAX_DRAWDOWN_ENV, max_drawdown_pct, minimum=0.0, maximum=100.0
            )
        if max_drawdown_pct <= 0.0:
            raise ValueError(f"{MAX_DRAWDOWN_ENV} must be greater than zero and at most 100")

        return cls(float(initial_capital), float(max_drawdown_pct))
