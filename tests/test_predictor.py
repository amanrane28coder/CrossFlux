"""
tests/test_predictor.py
=======================
Validation suite for the Signal Aggregator — Phase 4.

Test inventory (9 tests)
------------------------
  1. test_arbitrage_signal_creation          – Valid ArbitrageSignal construction
  2. test_arbitrage_signal_invalid           – p_execute out-of-range; empty action
  3. test_aggregator_emits_valid_signal      – Both gates pass → 1 signal
  4. test_aggregator_filters_low_delta       – Gate 1 fails (|delta| too small)
  5. test_aggregator_filters_high_latency    – Gate 2 fails (p_execute too low)
  6. test_aggregator_both_filters_required   – Mixed states → only 1 fires
  7. test_aggregator_action_label_buy_b      – Positive delta → BUY_B_SELL_A label
  8. test_aggregator_action_label_buy_a      – Negative delta → BUY_A_SELL_B label
  9. test_aggregator_empty_input             – Empty states list → empty output

Fixture strategy
----------------
All tests use synthetic MarketState objects constructed with precise per-level
volumes that produce deterministic OBI values.  No I/O or network calls needed.

Volume → OBI relationship used in fixtures:
    bid_vol ≫ ask_vol → OBI ≈ +1.0  (bid-heavy snapshot)
    ask_vol ≫ bid_vol → OBI ≈ -1.0  (ask-heavy snapshot)
    bid_vol = ask_vol → OBI  = 0.0  (balanced snapshot)

Python 3.10+ required.
"""

from __future__ import annotations

from typing import List

import pytest

from src.models import ArbitrageSignal, MarketState, OrderBookSnapshot, PriceLevel
from src.predictor import SignalAggregator


# ===========================================================================
# Shared fixture helpers
# ===========================================================================


def _make_snapshot(
    exchange_id: str,
    timestamp_ms: int,
    bid_volume: float,
    ask_volume: float,
    base_bid: float = 49_999.0,
    base_ask: float = 50_001.0,
) -> OrderBookSnapshot:
    """Build a one-level snapshot with precisely controlled bid/ask volumes.

    OBI of the resulting snapshot:
        (bid_volume - ask_volume) / (bid_volume + ask_volume)
    """
    return OrderBookSnapshot(
        exchange_id=exchange_id,
        timestamp_ms=timestamp_ms,
        bids=[PriceLevel(price=base_bid, volume=bid_volume)],
        asks=[PriceLevel(price=base_ask, volume=ask_volume)],
    )


def _make_state(
    timestamp_ms: int,
    bid_vol_a: float,
    ask_vol_a: float,
    bid_vol_b: float,
    ask_vol_b: float,
    exchange_a: str = "binance",
    exchange_b: str = "kraken",
) -> MarketState:
    """Convenience factory: build a two-venue MarketState from raw volumes."""
    snap_a = _make_snapshot(exchange_a, timestamp_ms, bid_vol_a, ask_vol_a)
    snap_b = _make_snapshot(exchange_b, timestamp_ms, bid_vol_b, ask_vol_b)
    return MarketState(
        timestamp_ms=timestamp_ms,
        venues={exchange_a: snap_a, exchange_b: snap_b},
    )


def _make_aggregator(
    delta_threshold: float = 0.3,
    min_p_execute: float = 0.80,
    latency_mu: float = 3.5,
    latency_sigma: float = 0.3,
    alpha_lifetime_ms: float = 50.0,
) -> SignalAggregator:
    """Convenience factory for a standard Binance/Kraken aggregator."""
    return SignalAggregator(
        exchange_a="binance",
        exchange_b="kraken",
        latency_mu=latency_mu,
        latency_sigma=latency_sigma,
        alpha_lifetime_ms=alpha_lifetime_ms,
        delta_threshold=delta_threshold,
        min_p_execute=min_p_execute,
    )


# ===========================================================================
# Test 1 — ArbitrageSignal: valid construction
# ===========================================================================


def test_arbitrage_signal_creation() -> None:
    """ArbitrageSignal must be constructible and its fields must be accessible."""
    sig = ArbitrageSignal(
        timestamp_ms=1_700_000_001_000,
        obi_delta=0.75,
        p_execute=0.93,
        action="BUY_KRAKEN_SELL_BINANCE",
    )

    assert sig.timestamp_ms == 1_700_000_001_000
    assert sig.obi_delta == 0.75
    assert sig.p_execute == 0.93
    assert sig.action == "BUY_KRAKEN_SELL_BINANCE"

    # frozen=True: mutation must raise
    with pytest.raises((AttributeError, TypeError)):
        sig.action = "INVALID"  # type: ignore[misc]


# ===========================================================================
# Test 2 — ArbitrageSignal: invalid field values raise ValueError
# ===========================================================================


def test_arbitrage_signal_invalid() -> None:
    """ArbitrageSignal must reject p_execute outside [0, 1] and empty action."""
    # p_execute > 1.0
    with pytest.raises(ValueError, match="p_execute must be in"):
        ArbitrageSignal(
            timestamp_ms=1_700_000_001_000,
            obi_delta=0.5,
            p_execute=1.5,
            action="BUY_KRAKEN_SELL_BINANCE",
        )

    # p_execute < 0.0
    with pytest.raises(ValueError, match="p_execute must be in"):
        ArbitrageSignal(
            timestamp_ms=1_700_000_001_000,
            obi_delta=0.5,
            p_execute=-0.1,
            action="BUY_KRAKEN_SELL_BINANCE",
        )

    # empty action
    with pytest.raises(ValueError, match="action must not be empty"):
        ArbitrageSignal(
            timestamp_ms=1_700_000_001_000,
            obi_delta=0.5,
            p_execute=0.90,
            action="",
        )


# ===========================================================================
# Test 3 — Aggregator: emits exactly one signal when both gates pass
# ===========================================================================


def test_aggregator_emits_valid_signal() -> None:
    """When both gates pass, exactly one ArbitrageSignal is emitted.

    Construction:
        binance: bid_vol=100, ask_vol=1  → OBI ≈ +0.980  (bid-heavy)
        kraken:  bid_vol=1,   ask_vol=1  → OBI  =  0.0   (balanced)
        delta   = 0.980 - 0.0 = +0.980  > 0.3  ✓  Gate 1 passes
        p_exec  = lognorm.cdf(50, s=0.3, scale=exp(3.5)) ≈ 0.93  > 0.80  ✓  Gate 2 passes
    """
    state = _make_state(
        timestamp_ms=1_700_000_001_000,
        bid_vol_a=100.0, ask_vol_a=1.0,   # binance: strongly bid-heavy
        bid_vol_b=1.0,   ask_vol_b=1.0,   # kraken:  balanced
    )
    agg = _make_aggregator()
    signals = agg.evaluate([state])

    assert len(signals) == 1, f"Expected 1 signal, got {len(signals)}"
    sig = signals[0]
    assert sig.timestamp_ms == 1_700_000_001_000
    assert sig.obi_delta > 0.3
    assert sig.p_execute > 0.80
    assert sig.action != ""


# ===========================================================================
# Test 4 — Aggregator: filters states where |delta| ≤ threshold (Gate 1 fail)
# ===========================================================================


def test_aggregator_filters_low_delta() -> None:
    """States with |OBI delta| ≤ 0.3 must be silently filtered (Gate 1).

    Construction:
        binance: bid_vol=1.0, ask_vol=1.0  → OBI = 0.0
        kraken:  bid_vol=1.0, ask_vol=1.0  → OBI = 0.0
        delta = 0.0  ≤ 0.3  → Gate 1 FAIL → no signal
    """
    state = _make_state(
        timestamp_ms=1_700_000_002_000,
        bid_vol_a=1.0, ask_vol_a=1.0,
        bid_vol_b=1.0, ask_vol_b=1.0,
    )
    agg = _make_aggregator()
    signals = agg.evaluate([state])

    assert len(signals) == 0, (
        f"Expected 0 signals for balanced books (delta=0.0), got {len(signals)}"
    )


# ===========================================================================
# Test 5 — Aggregator: filters states where p_execute ≤ min_p (Gate 2 fail)
# ===========================================================================


def test_aggregator_filters_high_latency() -> None:
    """States that pass Gate 1 but have p_execute ≤ min_p must be filtered.

    Approach: configure an aggregator with extreme jitter (sigma=3.0) so that
    the pre-computed p_execute ≈ 0.42, well below min_p_execute=0.80.
    The delta is deliberately large (> 0.3) so only Gate 2 is the blocker.
    """
    state = _make_state(
        timestamp_ms=1_700_000_003_000,
        bid_vol_a=100.0, ask_vol_a=1.0,   # delta >> 0.3 → Gate 1 passes
        bid_vol_b=1.0,   ask_vol_b=1.0,
    )
    # sigma=3.0 produces very low p_execute even for generous alpha
    agg = _make_aggregator(latency_sigma=3.0, min_p_execute=0.80)
    signals = agg.evaluate([state])

    # Verify the aggregator's pre-computed p is actually below the threshold
    assert agg._p_execute <= 0.80, (
        f"Test precondition failed: expected p_execute ≤ 0.80, got {agg._p_execute:.4f}"
    )
    assert len(signals) == 0, (
        f"Expected 0 signals when p_execute={agg._p_execute:.4f} ≤ 0.80, "
        f"got {len(signals)}"
    )


# ===========================================================================
# Test 6 — Aggregator: only states clearing BOTH gates emit signals
# ===========================================================================


def test_aggregator_both_filters_required() -> None:
    """Mixed batch: only states passing both gates produce signals.

    Batch (3 states):
      t=1000: delta ≈ 0.0   → Gate 1 FAIL → no signal
      t=2000: delta ≈ +0.98 → Gate 1 PASS → Gate 2 PASS (low jitter) → SIGNAL
      t=3000: delta ≈ +0.98 → Gate 1 PASS → Gate 2 FAIL (extreme jitter) → no signal

    We use two separate aggregators to simulate the Gate 2 pass/fail cases,
    then verify the counts independently.
    """
    state_low_delta = _make_state(
        timestamp_ms=1_000,
        bid_vol_a=1.0, ask_vol_a=1.0,   # balanced → delta ≈ 0
        bid_vol_b=1.0, ask_vol_b=1.0,
    )
    state_high_delta = _make_state(
        timestamp_ms=2_000,
        bid_vol_a=100.0, ask_vol_a=1.0,  # bid-heavy → delta > 0.3
        bid_vol_b=1.0,   ask_vol_b=1.0,
    )

    # Aggregator with good latency params: Gate 2 passes
    agg_good = _make_aggregator(latency_sigma=0.3)
    sigs_good = agg_good.evaluate([state_low_delta, state_high_delta])

    # Only t=2000 clears both gates
    assert len(sigs_good) == 1, (
        f"Expected 1 signal from good-latency aggregator, got {len(sigs_good)}"
    )
    assert sigs_good[0].timestamp_ms == 2_000

    # Aggregator with terrible latency params: Gate 2 fails even for high delta
    agg_bad = _make_aggregator(latency_sigma=3.0)
    sigs_bad = agg_bad.evaluate([state_low_delta, state_high_delta])

    assert len(sigs_bad) == 0, (
        f"Expected 0 signals from bad-latency aggregator, got {len(sigs_bad)}"
    )


# ===========================================================================
# Test 7 — Aggregator: action label for positive delta (buy B, sell A)
# ===========================================================================


def test_aggregator_action_label_buy_b_sell_a() -> None:
    """Positive delta (A bid-heavy, B balanced) must yield BUY_KRAKEN_SELL_BINANCE.

    OBI_A = (100-1)/(100+1) ≈ +0.980  (bid-heavy)
    OBI_B = 0.0                         (balanced)
    delta = +0.980 > 0  → BUY exchange_b, SELL exchange_a
    """
    state = _make_state(
        timestamp_ms=1_700_000_010_000,
        bid_vol_a=100.0, ask_vol_a=1.0,
        bid_vol_b=1.0,   ask_vol_b=1.0,
    )
    agg = _make_aggregator()
    signals = agg.evaluate([state])

    assert len(signals) == 1
    action = signals[0].action
    assert "BUY_KRAKEN" in action, f"Expected BUY_KRAKEN in action, got '{action}'"
    assert "SELL_BINANCE" in action, f"Expected SELL_BINANCE in action, got '{action}'"


# ===========================================================================
# Test 8 — Aggregator: action label for negative delta (buy A, sell B)
# ===========================================================================


def test_aggregator_action_label_buy_a_sell_b() -> None:
    """Negative delta (B bid-heavy, A balanced) must yield BUY_BINANCE_SELL_KRAKEN.

    OBI_A = 0.0                         (balanced)
    OBI_B = (100-1)/(100+1) ≈ +0.980   (bid-heavy)
    delta = 0.0 - 0.980 = -0.980 < 0  → BUY exchange_a, SELL exchange_b
    """
    state = _make_state(
        timestamp_ms=1_700_000_011_000,
        bid_vol_a=1.0,   ask_vol_a=1.0,
        bid_vol_b=100.0, ask_vol_b=1.0,
    )
    agg = _make_aggregator()
    signals = agg.evaluate([state])

    assert len(signals) == 1
    action = signals[0].action
    assert "BUY_BINANCE" in action, f"Expected BUY_BINANCE in action, got '{action}'"
    assert "SELL_KRAKEN" in action, f"Expected SELL_KRAKEN in action, got '{action}'"


# ===========================================================================
# Test 9 — Aggregator: empty input list returns empty output
# ===========================================================================


def test_aggregator_empty_input() -> None:
    """evaluate([]) must return an empty list without raising."""
    agg = _make_aggregator()
    signals = agg.evaluate([])

    assert signals == [], f"Expected [], got {signals}"
    assert isinstance(signals, list)
