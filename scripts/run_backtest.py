#!/usr/bin/env python3
"""
scripts/run_backtest.py
=======================
Phase 9 runner: executes the event-driven backtester and renders the
performance visualisation.

Output
------
  Console : BacktestResult.summary() — key metrics table
  File    : backtest_results.png — 3-panel performance chart:
              Panel 1 (top)    : Cumulative equity curve (USD)
              Panel 2 (middle) : Per-trade PnL bars (green = profit, red = loss)
              Panel 3 (bottom) : Rolling drawdown (shaded area)

Usage
-----
    cd "HFT trader"
    python3 scripts/run_backtest.py                 # requires real files in data/raw/
    python3 scripts/run_backtest.py --synthetic      # explicit synthetic demo

Configuration
-------------
Edit the Backtester() constructor call below to tune simulation parameters.
Real-data runs require both Tardis book files in data/raw/. Synthetic mode must be requested explicitly.
"""

from __future__ import annotations

import sys
import pathlib
import logging
import argparse

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works in all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Resolve project root ──────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.engine import Backtester, INITIAL_CAPITAL, BacktestResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_backtest")

OUTPUT_PATH = _ROOT / "backtest_results.png"


# ─────────────────────────────────────────────────────────────────────────────
# Plot builder
# ─────────────────────────────────────────────────────────────────────────────

def _to_datetime_index(series: pd.Series) -> pd.Series:
    """Convert integer ms timestamps to datetime index for clean x-axis labels."""
    dt_index = pd.to_datetime(series.index, unit="ms", utc=True)
    return pd.Series(series.values, index=dt_index, name=series.name)


def plot_results(result: BacktestResult, output_path: pathlib.Path) -> None:
    """Render and save a 3-panel performance chart to output_path."""

    # ── Style ─────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    ACCENT   = "#00d4aa"   # teal — primary signal colour
    RED_CLR  = "#ff4b6e"
    GREEN_CLR= "#00d4aa"
    GOLD     = "#f5c542"
    BG       = "#0d0d1a"
    PANEL_BG = "#12122a"

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(
        "Cross-Venue Arbitrage Predictor — Phase 9 Backtest",
        fontsize=18, fontweight="bold", color="white", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[3, 2, 1.5],
        hspace=0.08,
    )

    ax1 = fig.add_subplot(gs[0])   # equity curve
    ax2 = fig.add_subplot(gs[1])   # per-trade PnL bars
    ax3 = fig.add_subplot(gs[2])   # rolling drawdown

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")

    # ── Panel 1: Equity Curve ─────────────────────────────────────────────
    ec = _to_datetime_index(result.equity_curve)

    ax1.plot(ec.index, ec.values, color=ACCENT, linewidth=1.5, zorder=3, label="Equity")
    ax1.axhline(INITIAL_CAPITAL, color="#444466", linestyle="--", linewidth=1,
                label=f"Starting capital ${INITIAL_CAPITAL:,.0f}")

    # Fill above/below starting capital
    ax1.fill_between(
        ec.index, INITIAL_CAPITAL, ec.values,
        where=(ec.values >= INITIAL_CAPITAL),
        alpha=0.15, color=GREEN_CLR, zorder=2,
    )
    ax1.fill_between(
        ec.index, INITIAL_CAPITAL, ec.values,
        where=(ec.values < INITIAL_CAPITAL),
        alpha=0.25, color=RED_CLR, zorder=2,
    )

    ax1.set_ylabel("Portfolio Equity (USD)", color="white", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.tick_params(colors="white", labelbottom=False)
    ax1.grid(axis="y", color="#1e1e3a", linewidth=0.6, zorder=1)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.3)

    # Annotations: final equity + return
    final_eq = float(ec.iloc[-1])
    ret_str  = f"{result.total_return_pct:+.2f}%"
    ax1.annotate(
        f"  Final: ${final_eq:,.2f}  ({ret_str})",
        xy=(ec.index[-1], final_eq),
        fontsize=10, color=ACCENT if final_eq >= INITIAL_CAPITAL else RED_CLR,
        ha="right",
    )

    # Metric box top-right
    metrics_text = (
        f"Sharpe: {result.sharpe_ratio:.3f}\n"
        f"MDD:   {result.max_drawdown_pct:.2f}%\n"
        f"Trades: {len(result.trades):,}\n"
        f"Win %:  {result.win_rate*100:.1f}%"
    )
    ax1.text(
        0.99, 0.05, metrics_text,
        transform=ax1.transAxes,
        fontsize=9, color="white", alpha=0.85,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a3a", alpha=0.7),
    )

    # ── Panel 2: Per-Trade PnL Bars ───────────────────────────────────────
    if result.trades:
        trade_ts  = pd.to_datetime([t.timestamp_ms for t in result.trades], unit="ms", utc=True)
        trade_pnl = np.array([t.pnl_net for t in result.trades])
        colours   = [GREEN_CLR if p > 0 else RED_CLR for p in trade_pnl]

        # Bar width: proportional to sim duration (avoid invisible bars)
        bar_width = max(
            (trade_ts[-1] - trade_ts[0]).total_seconds() / len(trade_ts) * 0.7,
            3600,
        )
        bar_width_td = pd.Timedelta(seconds=bar_width)

        # Downsample for bar rendering if trade count is huge to avoid matplotlib hanging
        if len(result.trades) > 5000:
            indices = np.linspace(0, len(result.trades) - 1, 5000, dtype=int)
            plot_ts = trade_ts[indices]
            plot_pnl = trade_pnl[indices]
            plot_colours = [colours[i] for i in indices]
        else:
            plot_ts = trade_ts
            plot_pnl = trade_pnl
            plot_colours = colours

        ax2.bar(plot_ts, plot_pnl, width=bar_width_td, color=plot_colours, alpha=0.8, zorder=3)
        ax2.axhline(0, color="#444466", linewidth=0.8)

        # Rolling mean PnL overlay (computed on all trades, but plotted against trade_ts)
        if len(trade_pnl) >= 20:
            roll_mean = pd.Series(trade_pnl).rolling(20, min_periods=1).mean().values
            # Downsample rolling mean plot if too large to keep plotting fast
            if len(trade_pnl) > 5000:
                indices = np.linspace(0, len(trade_pnl) - 1, 5000, dtype=int)
                ax2.plot(trade_ts[indices], roll_mean[indices], color=GOLD, linewidth=1.2,
                         alpha=0.9, label="20-trade rolling mean", zorder=4)
            else:
                ax2.plot(trade_ts, roll_mean, color=GOLD, linewidth=1.2,
                         alpha=0.9, label="20-trade rolling mean", zorder=4)
            ax2.legend(loc="upper left", fontsize=8, framealpha=0.3)

    ax2.set_ylabel("Trade PnL (USD)", color="white", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:+.2f}"))
    ax2.tick_params(colors="white", labelbottom=False)
    ax2.grid(axis="y", color="#1e1e3a", linewidth=0.6, zorder=1)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────
    equity_vals = ec.values.astype(float)
    peak_curve  = np.maximum.accumulate(equity_vals)
    drawdown_pct = (equity_vals - peak_curve) / peak_curve * 100.0  # <= 0

    ax3.fill_between(ec.index, drawdown_pct, 0, alpha=0.6, color=RED_CLR, zorder=3)
    ax3.plot(ec.index, drawdown_pct, color=RED_CLR, linewidth=0.8, zorder=4)
    ax3.axhline(0, color="#444466", linewidth=0.8)
    ax3.axhline(
        -result.max_drawdown_pct,
        color=GOLD, linewidth=1, linestyle=":",
        label=f"MDD = {result.max_drawdown_pct:.2f}%",
    )

    ax3.set_ylabel("Drawdown (%)", color="white", fontsize=11)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax3.tick_params(colors="white")
    ax3.tick_params(axis="x", labelrotation=20)
    ax3.grid(axis="y", color="#1e1e3a", linewidth=0.6, zorder=1)
    ax3.legend(loc="lower left", fontsize=8, framealpha=0.3)

    # ── X-axis date formatting on bottom panel ────────────────────────────
    import matplotlib.dates as mdates
    hours = result.sim_duration_s / 3600
    if hours <= 6:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    elif hours <= 48:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    else:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nChart saved → {output_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CrossFlux backtest")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run generated demo data explicitly instead of requiring real book files",
    )
    args = parser.parse_args()

    print("\n" + "═" * 50)
    print("  Cross-Venue Arbitrage Predictor — Phase 9")
    print("═" * 50 + "\n")

    bt = Backtester(
        exchange_a        = "binance",
        exchange_b        = "kraken",
        latency_mu        = 3.5,
        latency_sigma     = 0.4,
        alpha_lifetime_ms = 50.0,
        delta_threshold   = 0.65,
        min_p_execute     = 0.80,
        min_spread_pct    = 0.0012,
        qty               = 0.01,
        batch_size        = 10_000,
        # ── Real Tardis data (2024-03-01, book_snapshot_5, depth=5) ────────
        binance_path     = pathlib.Path("data/raw/binance_book_snapshot_5_2024-03-01_BTCUSDT.csv.gz"),
        kraken_path      = pathlib.Path("data/raw/kraken_book_snapshot_5_2024-03-01_XBT-USD.csv.gz"),
        generator_kwargs = {"duration_s": 625, "seed": 42},
        use_synthetic   = args.synthetic,
    )


    result = bt.run()
    plot_results(result, OUTPUT_PATH)
    
    # ── Phase 12: Export Data for Analytics Dashboard ──────────────────────
    print("Exporting execution logs for the Analytics Dashboard...")
    
    # Export trades log
    trades_path = _ROOT / "data" / "trades_log.csv"
    import dataclasses
    trades_df = pd.DataFrame([dataclasses.asdict(t) for t in result.trades])
    trades_df.to_csv(trades_path, index=False)
    print(f"Trades log saved → {trades_path}")
    
    # Export equity curve
    equity_path = _ROOT / "data" / "equity_log.csv"
    ec_df = _to_datetime_index(result.equity_curve).to_frame(name="equity")
    ec_df.to_csv(equity_path, index=True, index_label="timestamp")
    print(f"Equity log saved → {equity_path}")

