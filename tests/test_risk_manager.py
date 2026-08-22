from __future__ import annotations

import time

from src.risk_manager import (
    GlobalKillSwitch,
    MaxPositionGuard,
    MicrosecondCooldownGuard,
    RiskManager,
)


def test_max_position_guard() -> None:
    g = MaxPositionGuard(0.01)
    assert g.max_trade_qty == 0.01
    assert g.check(0.005).approved
    assert not g.check(0.02).approved
    assert "exceeds" in g.check(0.02).reason
    assert not g.check(0.0).approved
    assert not g.check(-1.0).approved


def test_cooldown_guard() -> None:
    g = MicrosecondCooldownGuard(cooldown_us=100_000)  # 100ms
    assert g.cooldown_us == 100_000
    assert g.check().approved
    g.record_trade()
    assert not g.check().approved
    assert "cooldown" in g.check().reason
    time.sleep(0.15)
    assert g.check().approved
    g.reset()
    assert g.check().approved


def test_kill_switch_basic() -> None:
    ks = GlobalKillSwitch(max_drawdown_pct=10.0)
    assert not ks.is_tripped()
    assert ks.drawdown_pct() == 0.0
    ks.record_trade(100.0)
    assert ks.cumulative_pnl == 100.0
    assert ks.peak_pnl == 100.0
    assert ks.drawdown_pct() == 0.0
    ks.record_trade(-90.0)
    assert ks.cumulative_pnl == 10.0
    assert ks.peak_pnl == 100.0
    assert ks.drawdown_pct() == 90.0
    assert ks.is_tripped()


def test_kill_switch_reset() -> None:
    ks = GlobalKillSwitch(max_drawdown_pct=5.0)
    ks.record_trade(50.0)
    ks.record_trade(-60.0)
    assert ks.is_tripped()
    ks.reset()
    assert not ks.is_tripped()
    assert ks.cumulative_pnl == 0.0
    assert ks.peak_pnl == 0.0


def test_kill_switch_zero_peak() -> None:
    ks = GlobalKillSwitch(max_drawdown_pct=10.0)
    assert ks.drawdown_pct() == 0.0
    ks.record_trade(-5.0)
    assert ks.drawdown_pct() > 0.0


def test_risk_manager_full_flow() -> None:
    rm = RiskManager(max_trade_qty=0.01, cooldown_us=200_000, max_drawdown_pct=50.0)
    r = rm.check_order(0.005)
    assert r.approved
    assert r.guard == ""
    rm.record_trade(10.0)
    r = rm.check_order(0.005)
    assert not r.approved
    assert "cooldown" in r.reason
    time.sleep(0.25)
    r = rm.check_order(0.005)
    assert r.approved
    rm.record_trade(-50.0)
    assert rm.is_kill_switched()
    r = rm.check_order(0.005)
    assert not r.approved
    assert "kill switch" in r.reason
    assert rm.drawdown_pct() > 0


def test_risk_manager_exceed_position() -> None:
    rm = RiskManager(max_trade_qty=0.01)
    r = rm.check_order(0.02)
    assert not r.approved
    assert "exceeds" in r.reason
    assert r.guard == "MaxPositionGuard"


def test_risk_manager_reset() -> None:
    rm = RiskManager(max_drawdown_pct=10.0)
    rm.record_trade(50.0)
    rm.record_trade(-100.0)
    assert rm.is_kill_switched()
    rm.reset()
    assert not rm.is_kill_switched()
    assert rm.cumulative_pnl() == 0.0
    assert rm.drawdown_pct() == 0.0
    r = rm.check_order(0.01)
    assert r.approved
