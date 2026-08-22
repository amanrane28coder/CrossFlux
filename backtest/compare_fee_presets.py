#!/usr/bin/env python3
"""Run the real-data backtest under every fee preset and report side by side.

Point of this script
--------------------
The fee schedule is not a tuning parameter for this strategy -- it decides the
sign of the result. Reporting a single preset's number in isolation is how a
backtest ends up quoting the most flattering assumption by accident, so this
runner always prints all of them together.

Usage:
    python -m backtest.compare_fee_presets            # every preset
    python -m backtest.compare_fee_presets retail zero
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src import fees

# The Backtester's built-in defaults (data/raw/binance_btcusdt.csv.gz,
# data/raw/kraken_xbtusd.csv.gz) do not match the filenames actually present in
# data/raw/, so a bare Backtester() silently falls through to the SYNTHETIC
# generator. Synthetic results would make this whole comparison meaningless, so
# resolve the real files here and refuse to run without them.
_RAW = _ROOT / "data" / "raw"


def resolve_real_data() -> tuple[Path, Path]:
    def one(pattern: str, label: str) -> Path:
        hits = sorted(p for p in _RAW.glob(pattern) if p.name.endswith(".csv.gz"))
        if not hits:
            raise FileNotFoundError(
                f"No {label} book-snapshot file matching {pattern!r} in {_RAW}. "
                f"Present: {[p.name for p in _RAW.iterdir()] or 'nothing'}. "
                "Refusing to fall back to synthetic data -- a fee comparison on "
                "generated prices tells you nothing."
            )
        return hits[-1]

    return one("binance_book_snapshot_5_*", "binance"), one("kraken_book_snapshot_5_*", "kraken")


def run_one(preset: str, binance_path: Path, kraken_path: Path) -> dict:
    """Run the backtest under one preset in a clean module state."""
    # The fee model is read through fees.active() at call time, but Backtester
    # and its module-level state are cheap to rebuild, and rebuilding avoids any
    # chance of a cached rate leaking between presets.
    fees.set_active(preset)
    model = fees.active()

    from backtest.engine import Backtester

    t0 = time.perf_counter()
    bt = Backtester(binance_path=binance_path, kraken_path=kraken_path)
    result = bt.run()
    elapsed = time.perf_counter() - t0

    trades = result.trades
    filled = [t for t in trades if getattr(t, "status", "filled") == "filled"]
    rejected = [t for t in trades if getattr(t, "status", "filled") == "rejected"]

    total_fees = sum(t.fee for t in filled)
    net_pnl = sum(t.pnl_net for t in filled)
    gross_pnl = net_pnl + total_fees
    wins = [t for t in filled if t.pnl_net > 0]

    # Per-trade economics. Absolute PnL is not comparable across presets because
    # the gate admits a different number of trades under each one, so express
    # edge per unit of notional traded.
    notional = sum(t.qty * (t.buy_price + t.sell_price) / 2.0 for t in filled)
    gross_bps = (gross_pnl / notional * 10_000.0) if notional else 0.0
    net_bps = (net_pnl / notional * 10_000.0) if notional else 0.0

    return {
        "preset": preset,
        "round_trip_bps": model.round_trip_bps("binance", "kraken"),
        "signals": len(trades),
        "filled": len(filled),
        "rejected": len(rejected),
        "gross_pnl": gross_pnl,
        "fees": total_fees,
        "net_pnl": net_pnl,
        "return_pct": result.total_return_pct,
        "win_rate": (len(wins) / len(filled) * 100.0) if filled else 0.0,
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown_pct,
        "notional": notional,
        "gross_bps_per_trade": gross_bps,
        "net_bps_per_trade": net_bps,
        "net_usd_per_trade": (net_pnl / len(filled)) if filled else 0.0,
        "secs": elapsed,
    }


def main(argv: list[str]) -> int:
    wanted = [a for a in argv[1:] if not a.startswith("-")] or list(fees.PRESETS)
    for name in wanted:
        fees.get_preset(name)  # fail fast on a typo before a long run

    binance_path, kraken_path = resolve_real_data()
    print(f"binance: {binance_path.name}")
    print(f"kraken : {kraken_path.name}")
    print()
    print(fees.comparison_table())
    print()

    rows = []
    for name in wanted:
        print(f"[{name}] running...", flush=True)
        rows.append(run_one(name, binance_path, kraken_path))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print()
    print("=" * 112)
    print("REAL-DATA BACKTEST BY FEE PRESET")
    print("=" * 112)
    hdr = (f"{'preset':<14}{'rt_bps':>8}{'signals':>10}{'rejected':>10}"
           f"{'gross_pnl':>13}{'fees':>12}{'net_pnl':>13}{'return%':>10}{'win%':>7}{'sharpe':>8}")
    print(hdr)
    print("-" * 112)
    for r in rows:
        print(f"{r['preset']:<14}{r['round_trip_bps']:>8.1f}{r['signals']:>10,}"
              f"{r['rejected']:>10,}{r['gross_pnl']:>13,.2f}{r['fees']:>12,.2f}"
              f"{r['net_pnl']:>13,.2f}{r['return_pct']:>10.3f}{r['win_rate']:>7.1f}"
              f"{r['sharpe']:>8.2f}")
    print()
    print("PER-TRADE ECONOMICS (absolute PnL is not comparable -- the gate admits a")
    print("different trade count under each preset)")
    print("-" * 112)
    print(f"{'preset':<14}{'rt_bps':>8}{'filled':>10}{'notional':>16}"
          f"{'gross_bps':>11}{'net_bps':>10}{'net_$/trade':>13}{'trades/s':>10}")
    print("-" * 112)
    for r in rows:
        print(f"{r['preset']:<14}{r['round_trip_bps']:>8.1f}{r['filled']:>10,}"
              f"{r['notional']:>16,.0f}{r['gross_bps_per_trade']:>11.2f}"
              f"{r['net_bps_per_trade']:>10.2f}{r['net_usd_per_trade']:>13.4f}"
              f"{r['filled'] / 86400.0:>10.2f}")
    print("-" * 112)
    print()
    print("How to read this")
    print("  * Raising fees does NOT turn the strategy negative -- it just makes it trade")
    print("    less. The entry gate is `margin > fee`, so any trade that could not cover")
    print("    the fee is never taken. The strategy is unfalsifiable by fee assumption:")
    print("    every preset is positive by construction, and the ranking only reflects how")
    print("    permissive the gate is.")
    print("  * win% is meaningless for the same reason: the gate (margin > fee) and the")
    print("    booked PnL (margin - fee) are the same inequality, so it is ~97-99% for any")
    print("    signal whatsoever.")
    print("  * gross_bps is the mean gross edge per unit of notional. It RISES as fees rise")
    print("    because the gate is selecting progressively wider spreads -- not because the")
    print("    signal got better.")
    print("  * trades/s is a feasibility check, not a result. A number well above ~1 means")
    print("    the backtest assumes a round trip on both venues faster than either venue")
    print("    could realistically be hit, with capital pre-positioned on both sides.")

    out = _ROOT / "backtest" / "fee_preset_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
