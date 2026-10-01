"""Post-hoc quote-currency sensitivity for the Binance BTC/USDT vs Kraken XBT/USD run.

This diagnostic translates modeled USDT cash flows at a constant scenario rate.
It does not replace an observed, time-aligned USDT/USD book and does not rerun
signal selection at each rate. Use it to measure how sensitive the existing
selected fills are to quote-currency basis, not as a performance estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backtest.engine import Backtester, Trade, VENUE_A, VENUE_B
from src.fees import active as active_fees


@dataclass(frozen=True)
class CurrencyPnlComponents:
    """Net USD-denominated and USDT-denominated contributions before conversion."""

    usd: float
    usdt: float

    def at_usdt_usd(self, rate: float) -> float:
        if rate <= 0:
            raise ValueError("USDT/USD rate must be > 0")
        return self.usd + self.usdt * rate


def currency_pnl_components(trade: Trade, fees=None) -> CurrencyPnlComponents:
    """Split a modeled trade's net cash flow into USD and USDT components.

    Assumes the model's per-venue fees are charged in that venue's quote asset.
    Legging costs are assigned to the venue that over-filled, matching the
    simulator's residual-unwind accounting.
    """
    if trade.status != "filled":
        return CurrencyPnlComponents(0.0, 0.0)
    if {trade.buy_venue, trade.sell_venue} != {VENUE_A, VENUE_B}:
        raise ValueError("sensitivity is defined for one Binance and one Kraken leg")
    fee_model = fees or active_fees()

    usd = 0.0
    usdt = 0.0

    # Matched-leg proceeds and costs.
    if trade.sell_venue == VENUE_A:
        usdt += trade.sell_price * trade.matched_qty
    else:
        usd += trade.sell_price * trade.matched_qty
    if trade.buy_venue == VENUE_A:
        usdt -= trade.buy_price * trade.matched_qty
    else:
        usd -= trade.buy_price * trade.matched_qty

    # Fees are applied to each venue's actual filled quantity, including any
    # unhedged residual, as in the execution model.
    buy_fee = (trade.buy_filled_qty * trade.buy_price
               * fee_model.taker_rate(trade.buy_venue))
    sell_fee = (trade.sell_filled_qty * trade.sell_price
                * fee_model.taker_rate(trade.sell_venue))
    if trade.buy_venue == VENUE_A:
        usdt -= buy_fee
    else:
        usd -= buy_fee
    if trade.sell_venue == VENUE_A:
        usdt -= sell_fee
    else:
        usd -= sell_fee

    if trade.residual_qty > 0 and trade.legging_cost:
        residual_venue = (
            trade.buy_venue if trade.buy_filled_qty > trade.sell_filled_qty
            else trade.sell_venue
        )
        if residual_venue == VENUE_A:
            usdt -= trade.legging_cost
        else:
            usd -= trade.legging_cost

    return CurrencyPnlComponents(usd, usdt)


def summarize(trades: Iterable[Trade], rates: list[float]) -> list[str]:
    rows = [t for t in trades if t.status == "filled"]
    fees = active_fees()
    groups = [("ALL", rows)] + [
        (label.upper(), [t for t in rows if t.split == label])
        for label in ("train", "test")
    ]
    lines = [
        "Post-hoc quote-currency sensitivity (conditional on original signals)",
        f"Fee preset: {fees.name}; filled trades: {len(rows):,}",
        "Rate values are scenarios, not historical USDT/USD observations.",
        "Signals and entry gates are not recalculated at each scenario rate.",
    ]
    for label, group in groups:
        components = [currency_pnl_components(t, fees) for t in group]
        usd = sum(c.usd for c in components)
        usdt = sum(c.usdt for c in components)
        breakeven = -usd / usdt if abs(usdt) > 1e-12 else float("nan")
        lines.append(f"\n{label}: USD component={usd:+,.2f}; USDT component={usdt:+,.2f}")
        lines.append(f"  break-even USDT/USD={breakeven:.8f} (if finite)")
        for rate in rates:
            pnl = usd + usdt * rate
            lines.append(f"  at {rate:.6f} USD/USDT: translated net PnL={pnl:+,.2f} USD")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rates", type=float, nargs="+",
        default=[0.995, 0.999, 0.9995, 1.0, 1.0005, 1.001, 1.005],
        help="constant USD per USDT scenario rates; these are not observed FX data",
    )
    args = parser.parse_args()
    if any(rate <= 0 for rate in args.rates):
        parser.error("all rates must be positive")

    root = Path(__file__).resolve().parents[1]
    backtester = Backtester(
        exchange_a="binance", exchange_b="kraken",
        latency_mu=3.5, latency_sigma=0.4, alpha_lifetime_ms=50.0,
        delta_threshold=0.65, min_p_execute=0.80, min_spread_pct=0.0012,
        qty=0.01, batch_size=10_000,
        binance_path=root / "data/raw/binance_book_snapshot_5_2024-03-01_BTCUSDT.csv.gz",
        kraken_path=root / "data/raw/kraken_book_snapshot_5_2024-03-01_XBT-USD.csv.gz",
        generator_kwargs={"duration_s": 625, "seed": 42},
        use_synthetic=False, max_quote_age_ms=250,
    )
    result = backtester.run()
    print("\n".join(summarize(result.trades, args.rates)))


if __name__ == "__main__":
    main()
