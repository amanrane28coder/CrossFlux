"""
tests/test_bindings.py
======================
Integration test for the arbitrage_engine C++ Python bindings (Phase 8).

Verifies:
  1. Module imports successfully.
  2. PriceLevel construction and field access.
  3. make_order_book_snapshot() with single and multi-level books.
  4. OrderBookSnapshot properties (exchange_id, bids, asks, mid_price, spread).
  5. make_market_tick() construction.
  6. SignalAggregator construction and property access.
  7. evaluate() with one strongly imbalanced tick → 1 signal emitted.
  8. evaluate() with one balanced tick → 0 signals emitted.
  9. evaluate() with empty list → 0 signals emitted.
 10. ArbitrageSignal fields: timestamp_ms, obi_delta, p_execute, action (str).
 11. action_enum == TradeAction.BUY_B_SELL_A.
 12. p_execute numerical cross-check against Python prototype.

Usage
-----
    python3 tests/test_bindings.py

Requires the compiled extension module ``arbitrage_engine`` (built by
cpp_engine/CMakeLists.txt into the project root). Without it this file skips
rather than erroring: it is collected by pytest because of its name, but it is a
standalone script with its own harness, so a hard ImportError here would show up
as a collection error and mask the rest of the suite.
"""

from __future__ import annotations

import math
import pathlib
import sys

# ── Add project root to sys.path so arbitrage_engine.so is importable ────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

_MISSING = (
    "arbitrage_engine extension module not built. Build it with:\n"
    "    cmake -S . -B build && cmake --build build\n"
    "which writes arbitrage_engine*.so to the project root."
)

try:
    import arbitrage_engine as ae
except ImportError:
    if __name__ == "__main__":
        # Run directly: say why and exit 0, since "not built" is not a failure
        # of the bindings themselves.
        print(f"SKIP: {_MISSING}", file=sys.stderr)
        raise SystemExit(0)
    import pytest

    pytest.skip(_MISSING, allow_module_level=True)

# ── Minimal test harness ─────────────────────────────────────────────────────
_failures: list[str] = []
_total: list[int] = [0]


def expect(condition: bool, name: str, detail: str = "") -> None:
    _total[0] += 1
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f" — {detail}" if detail else "")
        print(msg, file=sys.stderr)
        _failures.append(name)


def expect_near(actual: float, expected: float, tol: float, name: str) -> None:
    _total[0] += 1
    diff = abs(actual - expected)
    if diff <= tol:
        print(f"  PASS  {name}  (got={actual:.8f}, expected={expected:.8f})")
    else:
        msg = (f"  FAIL  {name}  — got={actual:.8f}, expected={expected:.8f}, "
               f"|diff|={diff:.2e} > tol={tol:.2e}")
        print(msg, file=sys.stderr)
        _failures.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TS = 1_700_000_000_000   # fixed test timestamp (ms)


def make_imbalanced_tick() -> ae.MarketTick:
    """Exchange A: bid-heavy (OBI ≈ +0.98).  Exchange B: balanced (OBI = 0).
    Expected delta ≈ +0.98 > 0.3  →  Gate 1 PASS.
    """
    snap_a = ae.make_order_book_snapshot(
        TS, "binance",
        [ae.PriceLevel(50_100.0, 100.0)],   # bid-heavy and high price
        [ae.PriceLevel(50_102.0,   1.0)],
    )
    snap_b = ae.make_order_book_snapshot(
        TS, "kraken",
        [ae.PriceLevel(49_898.0, 1.0)],     # balanced and low price
        [ae.PriceLevel(49_900.0, 1.0)],
    )
    return ae.make_market_tick(TS, snap_a, snap_b)


def make_balanced_tick() -> ae.MarketTick:
    """Both venues balanced: OBI = 0 on each → delta = 0 → Gate 1 FAIL."""
    snap_a = ae.make_order_book_snapshot(
        TS + 1, "binance",
        [ae.PriceLevel(49_999.0, 1.0)],
        [ae.PriceLevel(50_001.0, 1.0)],
    )
    snap_b = ae.make_order_book_snapshot(
        TS + 1, "kraken",
        [ae.PriceLevel(49_999.0, 1.0)],
        [ae.PriceLevel(50_001.0, 1.0)],
    )
    return ae.make_market_tick(TS + 1, snap_a, snap_b)


# ─────────────────────────────────────────────────────────────────────────────
# Test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_price_level() -> None:
    pl = ae.PriceLevel(49_999.0, 5.5)
    expect(pl.price  == 49_999.0, "PriceLevel.price")
    expect(pl.volume == 5.5,      "PriceLevel.volume")
    expect("PriceLevel" in repr(pl), "PriceLevel.__repr__")


def test_order_book_snapshot() -> None:
    bids = [ae.PriceLevel(49_999.0, 10.0), ae.PriceLevel(49_998.0, 5.0)]
    asks = [ae.PriceLevel(50_001.0,  2.0), ae.PriceLevel(50_002.0, 1.0)]
    snap = ae.make_order_book_snapshot(TS, "binance", bids, asks)

    expect(snap.timestamp_ms == TS,         "Snapshot.timestamp_ms")
    expect(snap.exchange_id  == "binance",  "Snapshot.exchange_id")
    expect(snap.bid_depth    == 2,          "Snapshot.bid_depth")
    expect(snap.ask_depth    == 2,          "Snapshot.ask_depth")

    # bids / asks returned as Python lists clipped to valid depth
    expect(len(snap.bids) == 2,             "Snapshot.bids length")
    expect(len(snap.asks) == 2,             "Snapshot.asks length")
    expect(snap.bids[0].price == 49_999.0,  "Snapshot.bids[0].price")
    expect(snap.asks[0].price == 50_001.0,  "Snapshot.asks[0].price")

    expect_near(snap.mid_price, 50_000.0, 1e-9, "Snapshot.mid_price")
    expect_near(snap.spread,        2.0,   1e-9, "Snapshot.spread")


def test_market_tick() -> None:
    tick = make_imbalanced_tick()
    expect(tick.timestamp_ms == TS,           "MarketTick.timestamp_ms")
    expect(tick.snap_a.exchange_id == "binance", "MarketTick.snap_a.exchange_id")
    expect(tick.snap_b.exchange_id == "kraken",  "MarketTick.snap_b.exchange_id")
    expect("MarketTick" in repr(tick),        "MarketTick.__repr__")


def test_signal_aggregator_construction() -> None:
    agg = ae.SignalAggregator("binance", "kraken", 3.5, 0.4, 50.0)
    expect(agg.exchange_a       == "binance", "Aggregator.exchange_a")
    expect(agg.exchange_b       == "kraken",  "Aggregator.exchange_b")
    expect_near(agg.delta_threshold, 0.3,  1e-12, "Aggregator.delta_threshold")
    expect_near(agg.min_p_execute,   0.80, 1e-12, "Aggregator.min_p_execute")
    # Cross-check p_execute against the Python latency model
    from src.latency_model import calculate_execution_probability
    py_p = calculate_execution_probability(50.0, 3.5, 0.4)
    expect_near(agg.p_execute, py_p, 1e-6, "Aggregator.p_execute vs Python")


def test_evaluate_fires_signal() -> None:
    agg  = ae.SignalAggregator("binance", "kraken", 3.5, 0.4, 50.0, 0.3, 0.8, 0.0)
    tick = make_imbalanced_tick()
    sigs = agg.evaluate([tick])

    expect(len(sigs) == 1,                   "evaluate: 1 signal from imbalanced tick")

    sig = sigs[0]
    expect(sig.timestamp_ms == TS,           "Signal.timestamp_ms")
    expect_near(sig.obi_delta, 99.0/101.0,   1e-6, "Signal.obi_delta ≈ +0.9802")
    expect(sig.p_execute > 0.80,             "Signal.p_execute > 0.80")
    expect(sig.action == "BUY_B_SELL_A",     "Signal.action str")
    expect(sig.action_enum == ae.TradeAction.BUY_B_SELL_A, "Signal.action_enum")
    expect("ArbitrageSignal" in repr(sig),   "Signal.__repr__")


def test_evaluate_no_signal_balanced() -> None:
    agg  = ae.SignalAggregator("binance", "kraken", 3.5, 0.4, 50.0)
    tick = make_balanced_tick()
    sigs = agg.evaluate([tick])
    expect(len(sigs) == 0, "evaluate: 0 signals from balanced tick (Gate 1 fail)")


def test_evaluate_empty_list() -> None:
    agg  = ae.SignalAggregator("binance", "kraken", 3.5, 0.4, 50.0)
    sigs = agg.evaluate([])
    expect(len(sigs) == 0, "evaluate: 0 signals from empty tick list")


def test_evaluate_mixed_batch() -> None:
    """50 imbalanced + 50 balanced ticks → exactly 50 signals."""
    agg   = ae.SignalAggregator("binance", "kraken", 3.5, 0.4, 50.0)
    ticks = []
    for i in range(100):
        ticks.append(make_imbalanced_tick() if i % 2 == 0 else make_balanced_tick())
    sigs = agg.evaluate(ticks)
    expect(len(sigs) == 50, f"evaluate: 50 signals from mixed batch (got {len(sigs)})")


def test_module_constants() -> None:
    expect(ae.DEFAULT_DEPTH           == 10,  "MODULE.DEFAULT_DEPTH")
    expect_near(ae.DEFAULT_DELTA_THRESHOLD, 0.3,  1e-12, "MODULE.DEFAULT_DELTA_THRESHOLD")
    expect_near(ae.DEFAULT_MIN_P_EXECUTE,   0.80, 1e-12, "MODULE.DEFAULT_MIN_P_EXECUTE")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Argus Phase 8 Binding Tests ===\n")

    test_price_level()
    test_order_book_snapshot()
    test_market_tick()
    test_signal_aggregator_construction()
    test_evaluate_fires_signal()
    test_evaluate_no_signal_balanced()
    test_evaluate_empty_list()
    test_evaluate_mixed_batch()
    test_module_constants()

    print()
    if not _failures:
        print(f"All {_total[0]} tests PASSED.\n")
        sys.exit(0)
    else:
        print(f"{len(_failures)} / {_total[0]} tests FAILED: {_failures}\n",
              file=sys.stderr)
        sys.exit(1)
