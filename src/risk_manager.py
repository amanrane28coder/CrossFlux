from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class OrderCheckResult:
    approved: bool
    reason: str = ""
    guard: str = ""


class MaxPositionGuard:
    """Rejects orders exceeding a fixed per-trade base-currency limit."""

    def __init__(self, max_trade_qty: float = 0.01) -> None:
        if max_trade_qty <= 0:
            raise ValueError("max_trade_qty must be > 0")
        self._max = max_trade_qty

    @property
    def max_trade_qty(self) -> float:
        return self._max

    def check(self, qty: float) -> OrderCheckResult:
        if qty <= 0:
            return OrderCheckResult(False, f"qty must be > 0, got {qty}", "MaxPositionGuard")
        if qty > self._max:
            return OrderCheckResult(False, f"qty {qty} exceeds max {self._max}", "MaxPositionGuard")
        return OrderCheckResult(True)


class MicrosecondCooldownGuard:
    """Enforces a minimum elapsed time between consecutive trades."""

    def __init__(self, cooldown_us: int = 1_000_000) -> None:
        if cooldown_us <= 0:
            raise ValueError("cooldown_us must be > 0")
        self._cooldown_us = cooldown_us
        self._last_ns: int = 0

    @property
    def cooldown_us(self) -> int:
        return self._cooldown_us

    def check(self) -> OrderCheckResult:
        now = time.perf_counter_ns()
        elapsed_ns = now - self._last_ns
        if elapsed_ns < self._cooldown_us * 1_000:
            remaining_us = (self._cooldown_us * 1_000 - elapsed_ns) // 1_000
            return OrderCheckResult(
                False, f"cooldown active: {remaining_us}µs remaining", "MicrosecondCooldownGuard"
            )
        return OrderCheckResult(True)

    def record_trade(self) -> None:
        self._last_ns = time.perf_counter_ns()

    def reset(self) -> None:
        self._last_ns = 0


class GlobalKillSwitch:
    """Halts the system when total drawdown from peak PnL exceeds a threshold."""

    def __init__(self, max_drawdown_pct: float = 5.0) -> None:
        if max_drawdown_pct <= 0:
            raise ValueError("max_drawdown_pct must be > 0")
        self._max_drawdown_pct = max_drawdown_pct
        self._cumulative_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._tripped: bool = False

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct

    @property
    def cumulative_pnl(self) -> float:
        return self._cumulative_pnl

    @property
    def peak_pnl(self) -> float:
        return self._peak_pnl

    def drawdown_pct(self) -> float:
        if self._peak_pnl <= 0.0:
            if self._cumulative_pnl < 0.0:
                return 100.0
            return 0.0
        return max(0.0, (self._peak_pnl - self._cumulative_pnl) / abs(self._peak_pnl) * 100.0)

    def is_tripped(self) -> bool:
        return self._tripped

    def record_trade(self, pnl: float) -> None:
        if self._tripped:
            return
        self._cumulative_pnl += pnl
        if self._cumulative_pnl > self._peak_pnl:
            self._peak_pnl = self._cumulative_pnl
        dd = self.drawdown_pct()
        if dd >= self._max_drawdown_pct:
            self._tripped = True

    def reset(self) -> None:
        self._cumulative_pnl = 0.0
        self._peak_pnl = 0.0
        self._tripped = False


class RiskManager:
    """Orchestrates all three risk guards before an order is executed."""

    def __init__(
        self,
        max_trade_qty: float = 0.01,
        cooldown_us: int = 1_000_000,
        max_drawdown_pct: float = 5.0,
    ) -> None:
        self.position_guard = MaxPositionGuard(max_trade_qty)
        self.cooldown_guard = MicrosecondCooldownGuard(cooldown_us)
        self.kill_switch = GlobalKillSwitch(max_drawdown_pct)

    def check_order(self, qty: float) -> OrderCheckResult:
        if self.kill_switch.is_tripped():
            return OrderCheckResult(False, "kill switch is tripped", "GlobalKillSwitch")

        result = self.position_guard.check(qty)
        if not result.approved:
            return result

        result = self.cooldown_guard.check()
        if not result.approved:
            return result

        return OrderCheckResult(True)

    def record_trade(self, pnl: float) -> None:
        self.cooldown_guard.record_trade()
        self.kill_switch.record_trade(pnl)

    def is_kill_switched(self) -> bool:
        return self.kill_switch.is_tripped()

    def drawdown_pct(self) -> float:
        return self.kill_switch.drawdown_pct()

    def cumulative_pnl(self) -> float:
        return self.kill_switch.cumulative_pnl

    def reset(self) -> None:
        self.cooldown_guard.reset()
        self.kill_switch.reset()
