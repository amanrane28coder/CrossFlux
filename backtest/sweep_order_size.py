#!/usr/bin/env python3
"""Task #30/#31: sweep order size through the real backtester and print the curve.

Runs the MODIFIED backtest/engine.py (pending-order buffer + VWAP walk + adverse
fills booked) against the real Tardis CSVs, at each size in the sweep, and prints
the numbers the independent measurement script produced so the two can be
compared row by row.

Paths are passed explicitly on purpose: a bare Backtester() silently generates
synthetic data because _DEFAULT_BINANCE/_DEFAULT_KRAKEN point at filenames that
do not exist in this repo.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root, not backtest/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import Backtester            # noqa: E402
from src.friction import get_preset               # noqa: E402
from src.fees import active as active_fees        # noqa: E402

BINANCE = ROOT / "data/raw/binance_book_snapshot_5_2024-03-01_BTCUSDT.csv.gz"
KRAKEN = ROOT / "data/raw/kraken_book_snapshot_5_2024-03-01_XBT-USD.csv.gz"

# The order-size curve this produces replaces the one from the throwaway
# measurement script, which read only the first 400,000 rows of each file. That
# truncates binance at 13.86 h but kraken at 7.58 h, and the ffill in its
# alignment then holds a single frozen kraken snapshot across the final 45% of
# its sample. Its unfilled% and win% are artefacts of that frozen book.


def one(qty: float, friction_name: str) -> dict:
    for path in (BINANCE, KRAKEN):
        if not path.exists():
            raise SystemExit(f"missing real data file: {path}")
    t0 = time.time()
    bt = Backtester(qty=qty, friction=friction_name,
                    binance_path=BINANCE, kraken_path=KRAKEN)
    res = bt.run()
    fills = res.filled_trades
    n = len(fills)
    adverse = res.adverse_selection_fills
    collapsed = res.spread_collapse_fills
    requested = sum(t.qty for t in fills)
    matched = sum(t.matched_qty for t in fills)
    pnl = sum(t.pnl_net for t in fills)
    # Edge in bps on the notional actually arbitraged, so partial fills cannot
    # inflate it the way charging PnL to t.qty used to.
    notional = sum(t.matched_qty * t.buy_price for t in fills)
    return {
        "qty": qty,
        "friction": friction_name,
        "signals": len(res.trades),
        "fills": n,
        "win_rate_pct": 100.0 * sum(1 for t in fills if t.pnl_net > 0) / n if n else 0.0,
        "adverse_fills": len(adverse),
        "adverse_pct": 100.0 * len(adverse) / n if n else 0.0,
        "adverse_pnl": sum(t.pnl_net for t in adverse),
        "spread_collapsed": len(collapsed),
        "legged_fills": sum(1 for t in fills if t.residual_qty > 1e-12),
        "legging_cost": sum(t.legging_cost for t in fills),
        "unfilled_pct": 100.0 * (1.0 - matched / requested) if requested else 0.0,
        "fees": sum(t.fee for t in fills),
        "pnl_net": pnl,
        "mean_edge_bps": 1e4 * pnl / notional if notional else 0.0,
        "worst_fill": min((t.pnl_net for t in fills), default=0.0),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qty", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--friction", nargs="+", default=["stress"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"fees   : {active_fees().name}")
    rows = []
    for name in args.friction:
        fr = get_preset(name)
        print(f"\nfriction: {fr.name}  latency={fr.latency_ms:.0f} ms/leg  "
              f"jitter={fr.jitter_log_sigma:.2f}  legging={fr.legging_cost_bps:.1f} bps")
        for qty in args.qty:
            r = one(qty, name)
            rows.append(r)
            print(f"  qty={qty:>5} BTC  fills={r['fills']:>7}  "
                  f"win={r['win_rate_pct']:5.1f}%  adverse={r['adverse_pct']:5.1f}%  "
                  f"unfilled={r['unfilled_pct']:5.2f}%  legged={r['legged_fills']:>7}  "
                  f"edge={r['mean_edge_bps']:6.2f} bps  pnl=${r['pnl_net']:,.0f}  "
                  f"[{r['seconds']}s]")

    if args.out:
        out = Path(args.out)
        prior = json.loads(out.read_text()) if out.exists() else []
        keep = [p for p in prior
                if not any(p["qty"] == r["qty"] and p["friction"] == r["friction"]
                           for r in rows)]
        out.write_text(json.dumps(keep + rows, indent=2))
        print("\nwrote", out, f"({len(keep) + len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
