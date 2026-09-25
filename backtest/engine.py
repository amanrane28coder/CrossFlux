"""
backtest/engine.py
==================
Event-driven backtester for the Cross-Venue Arbitrage Predictor — Phase 9.

Architecture
------------
Three independent layers:

  DataLayer (SyntheticDataGenerator)
      Generates GBM-driven demo ticks only when synthetic mode is explicitly requested.
      Otherwise the backtester requires both real venue data files.
      Outputs: List[arbitrage_engine.MarketTick]

  BacktestEngine (Backtester)
      Feeds MarketTick batches into arbitrage_engine.SignalAggregator.evaluate().
      For each emitted ArbitrageSignal, looks up the originating tick and records
      a mock trade execution at best bid/ask with 0.05% taker fee per leg.
      Outputs: List[Trade]

  MetricsEngine (BacktestResult)
      Computes Total Return, Annualised Sharpe Ratio, Maximum Drawdown from the
      trade list and equity curve.

Synthetic data model
--------------------
  Price process : Geometric Brownian Motion
                  dS = σ·S·dW  (zero drift, risk-neutral)
                  σ = 0.015 annualised (≈ 1.5% vol, conservative BTC intraday)

  Arrival times : Independent Poisson processes per venue
                  Mean inter-event interval = 250 ms → ~4 ticks/s per venue

  Book depth    : 3 levels per side
                  Volumes ~ Gamma(α=2, scale=1.0) BTC
                  Imbalance spikes: 5% probability per tick, 5–15× skew factor

  Cross-venue   : Kraken lags Binance by Poisson(mean=200ms)
                  This temporal misalignment creates genuine OBI divergence signals

Python 3.10+ required.  arbitrage_engine compiled .so must be on sys.path.
"""

from __future__ import annotations

import sys
import pathlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── Resolve project root so src.* and arbitrage_engine are importable ─────────
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arbitrage_engine as ae
from src.fees import active as active_fees
from src.friction import (
    FrictionModel,
    active as active_friction,
    get_preset as friction_preset,
    prevailing_row,
    walk_book,
    walk_book_array,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Taker fees come from src.fees, the single source of truth shared with
# src/execution_simulator.py and the C++ engine. Select a schedule with
# $CROSSFLUX_FEE_PRESET or src.fees.set_active(); see `python -m src.fees`.
#
# This module previously hardcoded a flat 0.05%/leg (10.0 bps round-trip) that
# appeared nowhere else in the codebase and was the only rate reproducing the
# README's headline return. It survives as the "legacy_flat" preset so that
# number stays reproducible, but it is not a real schedule.
DEFAULT_QTY:       float = 0.01       # BTC per trade
INITIAL_CAPITAL:   float = 100_000.0  # USD
BOOK_DEPTH:        int   = 5          # price levels — matches book_snapshot_5
N_LEVELS_BINDING:  int   = 10         # arbitrage_engine.OrderBookSnapshot N=10

# Venue identity of the two aligned books. Venue A is Binance, venue B is
# Kraken throughout this module (see _emit_trade, where obi_delta > 0 buys B and
# sells A). Fees are per-venue, so this mapping is now load-bearing rather than
# cosmetic — mixing it up silently swaps a 1 bp leg for a 2 bp one.
VENUE_A: str = "binance"
VENUE_B: str = "kraken"

SECONDS_PER_YEAR: float = 365.0 * 86_400.0


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Trade:
    """Immutable record of one mock round-trip trade execution.

    Look-ahead accounting
    ---------------------
    The PnL attribution separates three components so the backtest cannot
    silently overstate edge:

      * ``expected_edge_at_signal`` : edge predicted at signal time using
        the *signal-time* book (best_ask - best_bid across venues).
      * ``pnl_net`` : actual realized PnL using the book prevailing at
        ``signal + latency_ms``, after walking it for size.
      * ``adverse_selection_cost`` : expected_edge - realized gross, in USD.
        Positive means the market moved against us between signal and
        fill; negative means it moved in our favour.

    A real-market strategy should expect adverse_selection_cost > 0
    in aggregate; the strategy is profitable iff pnl_net - fee > 0
    AFTER adverse selection.  Reporting only pnl_net is the textbook
    look-ahead bias that makes backtests look 2-5x better than reality.

    Why a losing trade is now a normal outcome
    ------------------------------------------
    This record used to have two shapes: a fill, always profitable because the
    entry gate and the PnL used the same snapshot, or a rejection carrying
    ``pnl_net = 0.0``. The rejection branch was what held the win rate at ~99%:
    a fill whose price had moved more than 5 bps against us was discarded rather
    than booked, while favourable moves were always kept, so the ledger only ever
    saw the good tail.

    Adverse fills are now executed and booked as losses, flagged ``adverse``.
    ``status = "rejected"`` survives for one genuinely impossible case: no book
    had printed on a venue by the time the order arrived. Everything else trades.
    """

    timestamp_ms: int
    action:       str    # "BUY_B_SELL_A" or "BUY_A_SELL_B"
    buy_venue:    str
    sell_venue:   str
    buy_price:    float  # realized VWAP on the buy venue (USD)
    sell_price:   float  # realized VWAP on the sell venue (USD)
    qty:          float  # position size *requested* (BTC)
    fee:          float  # total two-leg taker fee (USD), on what each leg filled
    pnl_net:      float  # net profit / loss (USD)  — can be negative
    obi_delta:    float
    p_execute:    float
    # ---- Day-1 additions: look-ahead accounting ----
    expected_edge_at_signal: float = 0.0   # edge predicted at signal time (USD)
    adverse_selection_cost:  float = 0.0   # expected_edge - realized_pnl_gross (USD)
    fill_time_mid_a:         float = 0.0   # mid price on venue A at fill time
    fill_time_mid_b:         float = 0.0   # mid price on venue B at fill time
    signal_time_mid_a:       float = 0.0   # mid price on venue A at signal time
    signal_time_mid_b:       float = 0.0   # mid price on venue B at signal time
    # ---- bookkeeping ----
    status:       str    = "filled"  # "filled", "rejected"
    reason:       str    = ""        # empty if filled, otherwise rejection reason
    buy_latency_ms: float = 0.0
    sell_latency_ms: float = 0.0
    buy_slippage: float   = 0.0
    sell_slippage: float  = 0.0
    # ---- Day-2 additions: regime attribution ----
    regime_vol:    str = "unknown"   # "low", "mid", "high" — by 1h realized vol
    regime_spread: str = "unknown"   # "tight", "normal", "wide" — by signal-time spread
    split:         str = "full"       # "train" or "test" — for in-sample / out-of-sample attribution
    # ---- friction: what the order actually achieved ----
    matched_qty:   float = 0.0   # BTC arbitraged — min(buy filled, sell filled)
    buy_filled_qty:  float = 0.0 # BTC the buy leg absorbed from its book
    sell_filled_qty: float = 0.0 # BTC the sell leg absorbed from its book
    residual_qty:  float = 0.0   # |buy - sell| left unhedged, charged legging_cost
    legging_cost:  float = 0.0   # USD cost of flattening residual_qty
    fill_ts_ms:    int   = 0     # timestamp of the book the fill executed against
    adverse:       bool  = False # pnl_net < 0 — booked, not discarded
    spread_collapsed: bool = False  # edge was positive at signal, gone by fill

    @property
    def shortfall_qty(self) -> float:
        """Requested size the book could not supply on at least one leg."""
        return max(0.0, self.qty - self.matched_qty)


# ─────────────────────────────────────────────────────────────────────────────
# Execution Simulator
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionSimulator:
    """Fills one leg against the book that had actually printed when the order
    arrived, walking it for size.

    Three things changed here, and each one used to protect the win rate.

    **The order no longer reads a future quote.** The old lookup was
    ``index.searchsorted(t + latency)``, which returns the first update at or
    *after* the execution time — a quote that had not yet been published when the
    order was sent. The prevailing quote is the last one at or before that
    instant, which is ``searchsorted(..., side="right") - 1``; see
    ``src.friction.prevailing_row``. The old form is the same class of defect as
    using ``timestamp`` instead of ``local_timestamp`` at ``src/ingestion.py:124``.

    **A fill that moved against us is booked, not discarded.** The old walk
    rejected the entire order the moment any level was more than 5 bps worse than
    the signal price, returning ``latency_miss`` and a PnL of exactly zero. That
    single branch is what held the win rate near 100%: unfavourable fills left
    the ledger while favourable ones stayed, so the reported distribution was the
    good tail of the real one. The requirement is now the opposite — an adverse
    fill executes and books its loss.

    **A partial fill reports what it filled.** The walk consumes only the depth
    the book actually offered, and ``filled_qty`` can be less than requested. The
    caller is responsible for not booking PnL on quantity it never acquired.

    The remaining rejections are the two cases where no fill is physically
    possible: the order arrived before the venue's first quote (``no_book``), or
    every level of the prevailing book was empty (``no_liquidity``). Neither is a
    price judgement, so neither can flatter the result.

    ``target_price`` is now purely diagnostic: it is the signal-time touch, kept
    so ``slippage`` can be reported against it. It no longer gates anything.

    This scalar path exists for readability and for single-trade inspection;
    ``_run_real_data_vectorized`` uses ``src.friction.walk_book_array`` over whole
    columns instead. ``tests/test_friction.py`` asserts the two agree bit-for-bit.
    """

    def __init__(
        self,
        friction: "FrictionModel | None" = None,
        rng: "np.random.Generator | None" = None,
    ) -> None:
        """``friction`` defaults to the active preset (``$CROSSFLUX_FRICTION_PRESET``).

        The RNG is seeded by default because the default preset is deterministic
        anyway, and because three tests in this repository were previously flaky
        against an unseeded ``np.random.lognormal``. Pass an explicit generator to
        vary it.
        """
        self.friction = friction if friction is not None else active_friction()
        self._rng = rng if rng is not None else np.random.default_rng(0)

    def get_stochastic_latency(self) -> float:
        """One latency draw in ms. Constant unless the preset enables jitter."""
        return float(self.friction.sample_latency(1, self._rng)[0])

    def execute_trade_with_slippage(
        self,
        order_side: str,
        order_qty: float,
        target_price: float,
        current_timestamp_ms: int,
        l2_order_book: pd.DataFrame,
    ) -> dict:
        """Buffer the order for ``latency_ms``, then walk the prevailing book.

        ``target_price`` is the signal-time touch, used only to report slippage.
        Returns ``filled_qty <= order_qty``; a short fill is a normal outcome, not
        an error.
        """
        latency = self.get_stochastic_latency()
        fill_at_ms = int(current_timestamp_ms + latency)

        # The prevailing quote, not the next one. See the class docstring.
        row = int(prevailing_row(l2_order_book.index.to_numpy(), np.int64(fill_at_ms)))
        if row < 0:
            # The order arrived before this venue had published anything. There is
            # no book to fill against, at any price.
            return {"status": "rejected", "reason": "no_book", "latency_ms": latency,
                    "fill_time_mid": 0.0, "filled_qty": 0.0, "fill_ts_ms": fill_at_ms}

        book = l2_order_book.iloc[row]
        fill_time_mid = 0.5 * (book["bids[0].price"] + book["asks[0].price"])

        prefix = "asks" if order_side == "BUY" else "bids"
        prices = [book[f"{prefix}[{i}].price"] for i in range(BOOK_DEPTH)]
        amounts = [book[f"{prefix}[{i}].amount"] for i in range(BOOK_DEPTH)]

        walk = walk_book(prices, amounts, order_qty)

        if walk.filled_qty <= 0.0:
            return {"status": "rejected", "reason": "no_liquidity", "latency_ms": latency,
                    "fill_time_mid": fill_time_mid, "filled_qty": 0.0,
                    "fill_ts_ms": int(l2_order_book.index[row])}

        return {
            "status": "filled",
            "requested_price": target_price,
            "realized_price": walk.vwap,
            "slippage": abs(target_price - walk.vwap),
            "slippage_bps": walk.slippage_bps(prices[0]),
            "latency_ms": latency,
            "filled_qty": walk.filled_qty,
            "shortfall_qty": walk.shortfall,
            "levels_consumed": walk.levels_consumed,
            "reason": "",
            "fill_time_mid": fill_time_mid,
            "fill_ts_ms": int(l2_order_book.index[row]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised execution: the same model, applied to whole columns at once
#
# The scalar simulator above is the readable reference. At ~180k signals per
# session a Python loop that also does two DataFrame row lookups per signal is
# the dominant cost of a backtest run, so the real-data path below does the same
# arithmetic over numpy arrays. The two are held together by
# src.friction.walk_book / walk_book_array, which tests/test_friction.py asserts
# agree bit-for-bit — the arithmetic exists once, in one module, and this layer
# only decides *which rows* to feed it.
# ─────────────────────────────────────────────────────────────────────────────

class _VenueBook:
    """One venue's L2 frame, with row lookup and cheap column gathers.

    Memory is the reason this holds a DataFrame rather than pre-extracted level
    arrays. Materialising asks+bids × price+amount × 5 levels for both venues is
    eight ``(n_rows, 5)`` float64 arrays; on the 2.25M-row sample that is ~720 MB
    on top of the frames themselves. Instead every gather takes a *view* of one
    column and immediately indexes it down to the signal rows, so the peak
    allocation is ``(n_signals, 5)`` — about 7 MB — regardless of file size.
    """

    __slots__ = ("venue", "frame", "index")

    def __init__(self, venue: str, frame: pd.DataFrame) -> None:
        if not frame.index.is_monotonic_increasing:
            # prevailing_row is a binary search; an unsorted index makes it return
            # silently wrong rows rather than fail. This can genuinely happen here:
            # src/ingestion.py indexes on the exchange-assigned `timestamp`, which
            # is not guaranteed monotonic in arrival order (the same field choice
            # flagged at src/ingestion.py:124). Sort rather than trust it.
            inversions = int((np.diff(frame.index.to_numpy(dtype=np.int64)) < 0).sum())
            logger.warning(
                "%s book index is not monotonic (%d inversions) — sorting. The "
                "index is the exchange timestamp, which can arrive out of order; "
                "see src/ingestion.py:124.",
                venue, inversions,
            )
            frame = frame.sort_index(kind="stable")
        self.venue = venue
        self.frame = frame
        self.index = frame.index.to_numpy(dtype=np.int64)

    def rows_at(self, at_ms: np.ndarray) -> np.ndarray:
        """Row of the book prevailing at each time; -1 before the first quote."""
        return prevailing_row(self.index, at_ms)

    def column(self, name: str, rows: np.ndarray) -> np.ndarray:
        """One column, gathered at ``rows``. The view is not copied first."""
        return self.frame[name].to_numpy(dtype=float, copy=False)[rows]

    def levels(self, side: str, field: str, rows: np.ndarray) -> np.ndarray:
        """``(len(rows), BOOK_DEPTH)`` of one side's prices or amounts."""
        out = np.empty((rows.shape[0], BOOK_DEPTH), dtype=float)
        for i in range(BOOK_DEPTH):
            out[:, i] = self.column(f"{side}[{i}].{field}", rows)
        return out

    def mid(self, rows: np.ndarray) -> np.ndarray:
        return 0.5 * (self.column("bids[0].price", rows)
                      + self.column("asks[0].price", rows))


_FILL_FIELDS: tuple[str, ...] = (
    "buy_vwap", "sell_vwap", "buy_filled", "sell_filled", "matched", "residual",
    "gross", "fee", "legging", "pnl_net", "buy_latency", "sell_latency",
    "buy_touch", "sell_touch", "fill_mid_buy", "fill_mid_sell",
)


def _empty_fills(n: int) -> dict[str, np.ndarray]:
    """Full-length output arrays, so the two direction subsets can scatter back."""
    out: dict[str, np.ndarray] = {k: np.zeros(n, dtype=float) for k in _FILL_FIELDS}
    out["ok"] = np.zeros(n, dtype=bool)
    out["fill_ts"] = np.zeros(n, dtype=np.int64)
    return out


def _simulate_fills(
    buy_book: _VenueBook,
    sell_book: _VenueBook,
    ts_ms: np.ndarray,
    qty: float,
    friction: FrictionModel,
    fees,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Buffer every signal for ``latency_ms``, then fill both legs by VWAP.

    This is the pending-order queue. A queue keyed on ``fill_at_ms`` and drained
    in timestamp order is the natural sequential formulation, and it is what the
    C++ side implements, but here it collapses to an as-of lookup: nothing in this
    model lets one pending order affect another — no shared inventory, no capital
    constraint, no queue position — so each order's fill depends only on its own
    ``T + latency`` and the book at that instant. Vectorising is therefore not an
    approximation of the queue, it is the same computation with the loop removed.
    (The moment inventory limits are added, that stops being true and this must
    become an ordered drain.)

    The two legs draw latency independently, so they generally fill against
    different book updates. That is legging risk, and it is now priced: the legs
    can absorb different quantities, only ``min`` of the two is a hedged
    arbitrage, and the difference is charged at ``friction.legging_cost_bps``.

    Every quantity returned is per-signal. Fee is charged on what each leg
    actually filled — an over-filled leg pays in full and earns nothing back,
    which is exactly what makes legging expensive (see
    ``FeeModel.taker_cost_legs``).
    """
    n = ts_ms.shape[0]
    lat_buy = friction.sample_latency(n, rng)
    lat_sell = friction.sample_latency(n, rng)

    # Truncate to whole ms, matching the scalar path's int(ts + latency).
    row_buy = buy_book.rows_at((ts_ms + lat_buy).astype(np.int64))
    row_sell = sell_book.rows_at((ts_ms + lat_sell).astype(np.int64))

    # -1 means the order arrived before that venue published anything. Clamp so
    # the gathers stay in bounds, then mask the results out.
    ok = (row_buy >= 0) & (row_sell >= 0)
    safe_buy = np.where(ok, row_buy, 0)
    safe_sell = np.where(ok, row_sell, 0)

    ask_px = buy_book.levels("asks", "price", safe_buy)
    ask_am = buy_book.levels("asks", "amount", safe_buy)
    bid_px = sell_book.levels("bids", "price", safe_sell)
    bid_am = sell_book.levels("bids", "amount", safe_sell)

    buy_vwap, buy_filled = walk_book_array(ask_px, ask_am, qty)
    sell_vwap, sell_filled = walk_book_array(bid_px, bid_am, qty)

    zero = np.zeros(n, dtype=float)
    buy_vwap = np.where(ok, buy_vwap, zero)
    sell_vwap = np.where(ok, sell_vwap, zero)
    buy_filled = np.where(ok, buy_filled, zero)
    sell_filled = np.where(ok, sell_filled, zero)

    # Only the quantity both legs achieved is actually arbitraged.
    matched = np.minimum(buy_filled, sell_filled)
    residual = np.abs(buy_filled - sell_filled)

    gross = (sell_vwap - buy_vwap) * matched
    fee = fees.taker_cost_legs(
        buy_filled, buy_book.venue, buy_vwap,
        sell_filled, sell_book.venue, sell_vwap,
    )
    # The residual sits on whichever venue over-filled, so it is unwound there.
    residual_price = np.where(buy_filled > sell_filled, buy_vwap, sell_vwap)
    legging = friction.legging_cost(residual, residual_price)

    return {
        "ok": ok,
        "buy_vwap": buy_vwap,
        "sell_vwap": sell_vwap,
        "buy_filled": buy_filled,
        "sell_filled": sell_filled,
        "matched": matched,
        "residual": residual,
        "gross": gross,
        "fee": fee,
        "legging": legging,
        "pnl_net": gross - fee - legging,
        "buy_latency": lat_buy,
        "sell_latency": lat_sell,
        "buy_touch": ask_px[:, 0],
        "sell_touch": bid_px[:, 0],
        "fill_mid_buy": buy_book.mid(safe_buy),
        "fill_mid_sell": sell_book.mid(safe_sell),
        # The position is only complete once the slower leg lands.
        "fill_ts": np.maximum(buy_book.index[safe_buy], sell_book.index[safe_sell]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backtest result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """All outputs from a completed backtest run."""

    trades:           List[Trade]
    equity_curve:     pd.Series   # index=timestamp_ms, values=equity (USD)
    total_return_pct: float        # e.g. +3.45 means +3.45%
    sharpe_ratio:     float        # annualised, risk-free rate = 0
    max_drawdown_pct: float        # magnitude (positive number), e.g. 1.2 means 1.2%
    n_ticks:          int
    n_signals:        int
    sim_duration_s:   float        # simulated wall-clock seconds
    data_source:      str = "synthetic GBM"   # or "real L2 CSV" — summary() said
    # "synthetic" unconditionally before this field existed, which mislabelled
    # every real-data run as synthetic in the printed report.

    @property
    def filled_trades(self) -> List[Trade]:
        """Return only trades that were successfully filled."""
        return [t for t in self.trades if getattr(t, "status", "filled") == "filled"]

    @property
    def adverse_selection_fills(self) -> List[Trade]:
        """Fills that lost money — the population that used to be discarded.

        The old engine rejected any fill whose price had moved more than 5 bps
        against it and booked ``pnl_net = 0.0``, so this list was empty by
        construction and the win rate was ~100% regardless of signal quality.
        These trades are now executed and booked, per requirement: a spread that
        collapses during the latency window, or VWAP slippage that pushes net PnL
        negative after fees, still results in a fill.

        An empty list here is now a finding, not the norm: it means either
        ``friction="zero"`` is active or the sample contains no adverse moves.
        """
        return [t for t in self.filled_trades if t.pnl_net < 0.0]

    @property
    def spread_collapse_fills(self) -> List[Trade]:
        """Fills where the gross edge was gone before fees were even charged.

        A strict subset of the adverse fills, isolating the mechanism the
        requirement calls out by name: the two venues' quotes converged during
        the latency window, so ``sell_vwap <= buy_vwap`` on a signal that
        predicted the opposite.
        """
        return [t for t in self.filled_trades if t.spread_collapsed]

    @property
    def win_rate(self) -> float:
        """Fraction of filled trades with pnl_net > 0.

        Meaningful only because losses are now reachable. ``friction="zero"``
        removes the delay but not the book walk, so it does not fully restore the
        old tautology: measured 99.3% at 0.01 BTC and 91.4% at 3.0 BTC. Treat a
        zero-latency win rate as a depth diagnostic, not a performance figure.
        """
        filled = self.filled_trades
        if not filled:
            return 0.0
        return sum(1 for t in filled if t.pnl_net > 0) / len(filled)

    @property
    def fill_ratio(self) -> float:
        """Matched quantity as a fraction of requested, across filled trades.

        Below ~1.0 the order is exhausting the book. Note the data ceiling: the
        source is ``book_snapshot_5``, so above roughly 0.25 BTC the shortfall is
        the file running out of levels rather than the market running out of
        depth. See src/friction.py.
        """
        filled = self.filled_trades
        if not filled:
            return 0.0
        requested = sum(t.qty for t in filled)
        return sum(t.matched_qty for t in filled) / requested if requested > 0 else 0.0

    def regime_decomposition(self) -> pd.DataFrame:
        """Decompose PnL by regime attribution (vol, spread, split).

        Returns a DataFrame with one row per (regime_vol, regime_spread, split)
        cell containing: n_trades, win_rate, mean_pnl, total_pnl, sharpe.
        """
        if not self.filled_trades:
            return pd.DataFrame()

        records = []
        for t in self.filled_trades:
            records.append({
                "regime_vol": t.regime_vol,
                "regime_spread": t.regime_spread,
                "split": t.split,
                "pnl_net": t.pnl_net,
                "adverse_selection_cost": t.adverse_selection_cost,
                "expected_edge": t.expected_edge_at_signal,
            })
        df = pd.DataFrame(records)

        def _agg(g: pd.DataFrame) -> pd.Series:
            n = len(g)
            if n == 0:
                return pd.Series({"n_trades": 0, "win_rate": 0.0,
                                  "mean_pnl": 0.0, "total_pnl": 0.0,
                                  "sharpe": 0.0})
            mean = g.pnl_net.mean()
            std = g.pnl_net.std(ddof=1) if n > 1 else 0.0
            sharpe = (mean / std * np.sqrt(len(g))) if std > 0 else 0.0
            return pd.Series({
                "n_trades": n,
                "win_rate": (g.pnl_net > 0).mean(),
                "mean_pnl": mean,
                "total_pnl": g.pnl_net.sum(),
                "sharpe": sharpe,
            })

        return df.groupby(["regime_vol", "regime_spread", "split"]).apply(_agg).reset_index()


    def split_summary(self) -> str:
        """Pretty-print in-sample vs out-of-sample metrics side by side."""
        if not self.filled_trades:
            return "(no filled trades)"
        df = pd.DataFrame([{
            "split": t.split,
            "pnl_net": t.pnl_net,
            "expected_edge": t.expected_edge_at_signal,
            "as_cost": t.adverse_selection_cost,
        } for t in self.filled_trades])

        out = ["── Train / Test split ──"]
        for s in ["train", "test"]:
            sub = df[df.split == s]
            if len(sub) == 0:
                out.append(f"  {s.upper():5s}: (no trades)")
                continue
            n = len(sub)
            mean = sub.pnl_net.mean()
            std = sub.pnl_net.std(ddof=1) if n > 1 else 0.0
            sharpe = (mean / std * np.sqrt(n)) if std > 0 else 0.0
            as_pct = (sub.as_cost.sum() / max(sub.expected_edge.sum(), 1e-9)) * 100
            out.append(
                f"  {s.upper():5s}: n={n:>6,}  total=${sub.pnl_net.sum():>+10,.2f}  "
                f"mean=${mean:>+7.4f}  sharpe={sharpe:>+5.2f}  AS%={as_pct:+5.1f}%"
            )
        return "\n".join(out)


    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        filled = self.filled_trades
        rejected_count = len(self.trades) - len(filled)

        # ── Look-ahead accounting (Day 1 fix) ─────────────────────────────
        if filled:
            expected_total = sum(t.expected_edge_at_signal for t in filled)
            # Honest edge after latency & slippage, before fees. Booked on the
            # quantity actually hedged, not on the size requested -- charging PnL
            # to unfilled quantity is how a partial fill used to be gifted the
            # full notional.
            realized_gross_total = sum(
                (t.sell_price - t.buy_price) * t.matched_qty for t in filled
            )
            as_cost_total = expected_total - realized_gross_total
            as_cost_bps = (as_cost_total / (expected_total + 1e-9)) * 1e4 if expected_total > 0 else 0.0
            avg_latency_ms = sum(t.buy_latency_ms + t.sell_latency_ms for t in filled) / (2 * len(filled))
            # ── Friction detail ───────────────────────────────────────────
            adverse = self.adverse_selection_fills
            collapsed = self.spread_collapse_fills
            legging_total = sum(t.legging_cost for t in filled)
            fee_total = sum(t.fee for t in filled)
            requested_qty = sum(t.qty for t in filled)
            matched_qty = sum(t.matched_qty for t in filled)
            unfilled_pct = (1.0 - matched_qty / requested_qty) * 100.0 if requested_qty > 0 else 0.0
            n_legged = sum(1 for t in filled if t.residual_qty > 0.0)
            mean_slip_bps = sum(
                (abs(t.buy_slippage) / t.buy_price + abs(t.sell_slippage) / t.sell_price) * 0.5e4
                for t in filled if t.buy_price > 0 and t.sell_price > 0
            ) / len(filled)
            worst = min(filled, key=lambda t: t.pnl_net)
        else:
            expected_total = realized_gross_total = as_cost_total = as_cost_bps = 0.0
            avg_latency_ms = 0.0
            adverse = collapsed = []
            legging_total = fee_total = unfilled_pct = mean_slip_bps = 0.0
            n_legged = 0
            worst = None

        fr = active_friction()
        n_filled = max(len(filled), 1)

        lines = [
            "=== Phase 9 Backtest Results ===",
            f"Simulation duration  : {self.sim_duration_s/3600:.1f}h {self.data_source}",
            f"Ticks processed      : {self.n_ticks:,}",
            f"Signals evaluated    : {self.n_signals:,}",
            f"Trades attempted     : {len(self.trades):,}",
            f"Trades filled        : {len(filled):,} (Rejected: {rejected_count:,} — no book at fill time)",
            "─" * 46,
            f"Initial capital      : ${INITIAL_CAPITAL:,.2f}",
            f"Final equity         : ${self.equity_curve.iloc[-1]:,.2f}",
            f"Total Return         : {self.total_return_pct:+.2f}%",
            f"Annualised Sharpe    : {self.sharpe_ratio:.3f}",
            f"Maximum Drawdown     : {self.max_drawdown_pct:.2f}%",
            f"Win Rate             : {self.win_rate*100:.1f}%",
            "─" * 46,
            "Look-Ahead Accounting (Day-1 fix):",
            f"  Expected edge (signal-time)   : ${expected_total:,.2f}",
            f"  Realized gross edge (fill-time): ${realized_gross_total:,.2f}",
            f"  Adverse-selection cost        : ${as_cost_total:,.2f}  ({as_cost_bps:+.1f} bps of expected)",
            f"  Avg two-leg latency           : {avg_latency_ms:.1f} ms",
            "─" * 46,
            f"Execution Friction ({fr.name} preset, {fr.latency_ms:.0f} ms/leg):",
            f"  adverse_selection_fills       : {len(adverse):,} "
            f"({len(adverse)/n_filled*100:.1f}% of fills), "
            f"${sum(t.pnl_net for t in adverse):,.2f}",
            f"    of which spread collapsed   : {len(collapsed):,} "
            f"(gross edge gone before fees)",
            f"  Mean VWAP slippage vs touch   : {mean_slip_bps:.2f} bps",
            f"  Unfilled (book too thin)      : {unfilled_pct:.2f}% of requested qty",
            f"  Legged fills (qty mismatch)   : {n_legged:,}, unwind cost ${legging_total:,.2f}",
            f"  Taker fees paid               : ${fee_total:,.2f}",
        ]
        if worst is not None:
            lines.append(
                f"  Worst single fill             : ${worst.pnl_net:,.2f} "
                f"({worst.action} @ {worst.timestamp_ms})"
            )
        if fr.latency_ms == 0.0:
            lines.append(
                "  NOTE: zero-latency preset — the fill reads the signal's own book, "
                "so the only friction left is depth. Win rate is a depth diagnostic."
            )
        lines.append("─" * 46)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates realistic two-venue BTC/USD tick data for backtesting.

    Price model
    -----------
    Binance mid-price follows Geometric Brownian Motion:
        S(t+dt) = S(t) * exp(σ * sqrt(dt) * Z)   where Z ~ N(0,1)
    Kraken receives Binance's price delayed by Poisson(mean=lag_ms) ms.

    Book construction
    -----------------
    Each snapshot has BOOK_DEPTH price levels per side.
    Volumes are Gamma(2, 1) BTC, with 5% probability of a 5–15× skew spike
    on one side — creating the OBI divergence that triggers Gate 1.

    Parameters
    ----------
    duration_s       : Total simulated duration in seconds (default: 86400 = 24h)
    start_price      : Initial BTC/USD mid-price (default: 50,000)
    annual_vol       : Annualised volatility of the GBM (default: 0.015 = 1.5%)
    mean_interval_ms : Mean inter-tick interval per venue in ms (default: 250)
    kraken_lag_ms    : Mean Kraken-vs-Binance delay in ms (default: 200)
    seed             : RNG seed for reproducibility (default: 42)
    """

    def __init__(
        self,
        duration_s:       float = 86_400.0,
        start_price:      float = 50_000.0,
        annual_vol:       float = 0.015,
        mean_interval_ms: float = 250.0,
        kraken_lag_ms:    float = 200.0,
        seed:             int   = 42,
    ) -> None:
        self.duration_s       = duration_s
        self.start_price      = start_price
        self.annual_vol       = annual_vol
        self.mean_interval_ms = mean_interval_ms
        self.kraken_lag_ms    = kraken_lag_ms
        self.rng              = np.random.default_rng(seed)

        # Per-second volatility from annualised
        self._vol_per_s = annual_vol / np.sqrt(SECONDS_PER_YEAR)

    # ── Price process ─────────────────────────────────────────────────────────

    def _generate_price_path(self, timestamps_ms: np.ndarray) -> np.ndarray:
        """Simulate GBM price path at the given timestamps."""
        dt_s = np.diff(timestamps_ms / 1000.0, prepend=0.0)
        dt_s[0] = dt_s[1]  # initialise first step
        shocks = self.rng.standard_normal(len(timestamps_ms))
        log_returns = self._vol_per_s * np.sqrt(np.abs(dt_s)) * shocks
        log_prices = np.log(self.start_price) + np.cumsum(log_returns)
        return np.exp(log_prices)

    # ── Book construction ─────────────────────────────────────────────────────

    def _make_ae_snapshot(
        self,
        ts_ms:      int,
        exchange:   str,
        mid_price:  float,
    ) -> ae.OrderBookSnapshot:
        """Build one arbitrage_engine.OrderBookSnapshot around mid_price."""
        half_spread = mid_price * 0.0001  # 1 basis-point half-spread

        # Bid/ask prices at BOOK_DEPTH levels (widening away from mid)
        bid_prices = [mid_price - half_spread * (1.0 + 0.4 * i) for i in range(BOOK_DEPTH)]
        ask_prices = [mid_price + half_spread * (1.0 + 0.4 * i) for i in range(BOOK_DEPTH)]

        # Volumes: Gamma-distributed, occasionally spiked for OBI divergence
        bid_vols = self.rng.gamma(2.0, 1.0, BOOK_DEPTH)
        ask_vols = self.rng.gamma(2.0, 1.0, BOOK_DEPTH)

        if self.rng.random() < 0.05:              # 5% spike probability
            factor = self.rng.uniform(5.0, 15.0)
            if self.rng.integers(0, 2) == 0:
                bid_vols *= factor                # bid-heavy spike
            else:
                ask_vols *= factor                # ask-heavy spike

        bids = [ae.PriceLevel(p, float(v)) for p, v in zip(bid_prices, bid_vols)]
        asks = [ae.PriceLevel(p, float(v)) for p, v in zip(ask_prices, ask_vols)]

        return ae.make_order_book_snapshot(ts_ms, exchange, bids, asks)

    # ── Top-level generator ───────────────────────────────────────────────────

    def generate(self) -> List[ae.MarketTick]:
        """
        Generate and return a List[arbitrage_engine.MarketTick].

        Fully vectorised: all prices and volumes are computed as NumPy arrays
        before the tick-building loop.  The final loop only calls the
        pybind11 constructors — no Python-level RNG or pandas iteration.
        """
        duration_ms = int(self.duration_s * 1000)
        start_ms    = 1_700_000_000_000   # 2023-11-15 00:00:00 UTC in ms
        D           = BOOK_DEPTH

        # ── Timestamps ────────────────────────────────────────────────────
        n_est       = int(self.duration_s * 1000 / self.mean_interval_ms)
        intervals_b = self.rng.exponential(self.mean_interval_ms, n_est)
        ts_b_full   = start_ms + np.cumsum(intervals_b).astype(np.int64)
        ts_binance  = ts_b_full[ts_b_full < start_ms + duration_ms]

        lags_k    = self.rng.exponential(self.kraken_lag_ms, len(ts_binance))
        ts_k_full = (ts_binance + lags_k).astype(np.int64)
        ts_kraken = ts_k_full[ts_k_full < start_ms + duration_ms]

        # ── As-of aligned timestamps + prices ─────────────────────────────
        all_ts   = np.union1d(ts_binance, ts_kraken)
        N        = len(all_ts)
        prices   = self._generate_price_path(all_ts)  # shape (N,)

        # Map each venue's timestamps to the aligned price via searchsorted
        idx_b    = np.searchsorted(all_ts, ts_binance)
        idx_k    = np.searchsorted(all_ts, ts_kraken)

        price_b  = prices[idx_b]   # shape (len(ts_binance),)
        price_k  = prices[idx_k]   # shape (len(ts_kraken),)

        # Pandas ffill for the as-of alignment — deduplicate first to avoid
        # ValueError from identical timestamps caused by Poisson lag collisions
        df_b = pd.Series(price_b, index=ts_binance, name="binance")
        df_k = pd.Series(price_k, index=ts_kraken,  name="kraken")
        df_b = df_b.groupby(level=0).last()   # keep last price if ts collision
        df_k = df_k.groupby(level=0).last()
        aligned = pd.concat([df_b, df_k], axis=1).sort_index().ffill().dropna()
        M = len(aligned)   # number of aligned ticks

        logger.info(
            "SyntheticDataGenerator: %d aligned ticks (Binance=%d, Kraken=%d)",
            M, len(ts_binance), len(ts_kraken),
        )

        # ── Extract arrays from aligned DataFrame (mutable for price offsets) ──
        ts_arr   = aligned.index.to_numpy(dtype=np.int64)
        mid_a    = aligned["binance"].to_numpy(dtype=np.float64).copy()
        mid_b    = aligned["kraken"].to_numpy(dtype=np.float64).copy()

        # ── Pre-compute ALL volumes vectorised (shape M × D) ──────────────
        bid_vols_a = self.rng.gamma(2.0, 1.0, (M, D))
        ask_vols_a = self.rng.gamma(2.0, 1.0, (M, D))
        bid_vols_b = self.rng.gamma(2.0, 1.0, (M, D))
        ask_vols_b = self.rng.gamma(2.0, 1.0, (M, D))

        # Spike mask: 5% of ticks get a 5–15× volume skew on one side, one venue.
        # We also apply a correlated price dislocation (0.15–0.25% offset) so that
        # the OBI signal coincides with a genuine cross-venue price gap — making
        # the trade profitable after fees (round-trip cost = 0.10%).
        spike_mask   = self.rng.random(M) < 0.05
        spike_side   = self.rng.integers(0, 4, M)   # 0=bid_a, 1=ask_a, 2=bid_b, 3=ask_b
        spike_factor = self.rng.uniform(5.0, 15.0, M)
        # Price offset magnitude: 0.15–0.25% → reliably covers 0.10% round-trip fee
        price_offset = self.rng.uniform(0.0015, 0.0025, M)

        for i in np.where(spike_mask)[0]:
            f  = spike_factor[i]
            s  = int(spike_side[i])
            po = price_offset[i]
            if s == 0:
                bid_vols_a[i] *= f          # Binance bid-heavy
                mid_a[i] *= (1.0 + po)      # Binance price pushed up → sell Binance
            elif s == 1:
                ask_vols_a[i] *= f          # Binance ask-heavy
                mid_a[i] *= (1.0 - po)      # Binance price pushed down → buy Binance
            elif s == 2:
                bid_vols_b[i] *= f          # Kraken bid-heavy
                mid_b[i] *= (1.0 + po)      # Kraken price pushed up → sell Kraken
            else:
                ask_vols_b[i] *= f          # Kraken ask-heavy
                mid_b[i] *= (1.0 - po)      # Kraken price pushed down → buy Kraken

        # ── Build MarketTick objects (tight loop, pre-computed arrays) ─────

        ticks: List[ae.MarketTick] = []
        for i in range(M):
            ts   = int(ts_arr[i])
            ma   = float(mid_a[i])
            mb   = float(mid_b[i])
            hs_a = ma * 0.0001   # 1 bps half-spread
            hs_b = mb * 0.0001

            bids_a = [ae.PriceLevel(ma - hs_a*(1.0+0.4*j), float(bid_vols_a[i,j])) for j in range(D)]
            asks_a = [ae.PriceLevel(ma + hs_a*(1.0+0.4*j), float(ask_vols_a[i,j])) for j in range(D)]
            bids_b = [ae.PriceLevel(mb - hs_b*(1.0+0.4*j), float(bid_vols_b[i,j])) for j in range(D)]
            asks_b = [ae.PriceLevel(mb + hs_b*(1.0+0.4*j), float(ask_vols_b[i,j])) for j in range(D)]

            snap_a = ae.make_order_book_snapshot(ts, "binance", bids_a, asks_a)
            snap_b = ae.make_order_book_snapshot(ts, "kraken",  bids_b, asks_b)
            ticks.append(ae.make_market_tick(ts, snap_a, snap_b))

        return ticks


# ─────────────────────────────────────────────────────────────────────────────
# Real data loader (wraps src.ingestion when CSV files exist)
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_ticks(
    binance_path: Path,
    kraken_path:  Path,
    depth:        int = BOOK_DEPTH,
) -> List[ae.MarketTick]:
    """
    Load real Tardis.dev CSV files via the Phase 1 ingestion pipeline and
    convert the aligned MarketState list into arbitrage_engine.MarketTick objects.
    """
    from src.ingestion import load_market_states

    states = load_market_states(binance_path, kraken_path, depth=depth)
    ticks: List[ae.MarketTick] = []

    for state in states:
        snap_py_a = state.venues.get("binance")
        snap_py_b = state.venues.get("kraken")
        if snap_py_a is None or snap_py_b is None:
            continue

        def _convert(snap_py) -> ae.OrderBookSnapshot:
            bids = [ae.PriceLevel(lvl.price, lvl.volume) for lvl in snap_py.bids[:depth]]
            asks = [ae.PriceLevel(lvl.price, lvl.volume) for lvl in snap_py.asks[:depth]]
            return ae.make_order_book_snapshot(
                snap_py.timestamp_ms, snap_py.exchange_id, bids, asks
            )

        ticks.append(ae.make_market_tick(
            state.timestamp_ms,
            _convert(snap_py_a),
            _convert(snap_py_b),
        ))

    logger.info("Loaded %d MarketTick objects from real CSV files.", len(ticks))
    return ticks


# ─────────────────────────────────────────────────────────────────────────────
# Metrics engine (pure functions operating on trade list + equity curve)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_equity_curve(trades: List[Trade]) -> pd.Series:
    """Build a pd.Series of equity values (USD) indexed by timestamp_ms.

    Ordered by *fill* time, not signal time. PnL is realised when the slower leg
    lands, and with per-leg latency jitter that is not the same ordering as the
    signals. Under a deterministic preset the two orderings coincide, so this
    changes nothing for the default and only matters for ``retail``.
    """
    if not trades:
        return pd.Series(
            [INITIAL_CAPITAL],
            index=pd.Index([0], name="timestamp_ms"),
            name="equity",
        )
    ordered = sorted(trades, key=lambda t: (t.fill_ts_ms or t.timestamp_ms))
    timestamps = [t.fill_ts_ms or t.timestamp_ms for t in ordered]
    equity     = INITIAL_CAPITAL + pd.Series(
        [t.pnl_net for t in ordered]
    ).cumsum().values
    return pd.Series(equity, index=pd.Index(timestamps, name="timestamp_ms"), name="equity")


def _total_return(equity_curve: pd.Series) -> float:
    """Return total percentage return (e.g. 3.45 means +3.45%)."""
    final   = float(equity_curve.iloc[-1])
    initial = INITIAL_CAPITAL
    return (final - initial) / initial * 100.0


def _annualised_sharpe(trades: List[Trade], sim_duration_s: float) -> float:
    """
    Annualised Sharpe Ratio (risk-free rate = 0).

    Annualisation method: time-elapsed scaling.
        factor = sqrt(SECONDS_PER_YEAR / sim_duration_s)

    Using time-elapsed rather than trade-count scaling avoids the Sharpe
    exploding when sim_duration_s is short (e.g. 625s) but trade frequency
    is high — trade-count annualisation would imply ~118M trades/year and
    multiply by sqrt(118M) ≈ 10,900, producing nonsensical values.
    """
    if len(trades) < 2 or sim_duration_s <= 0.0:
        return 0.0

    pnl_series  = np.array([t.pnl_net for t in trades], dtype=float)
    # Running equity for denominator (pre-trade equity at each step)
    equity_pre  = INITIAL_CAPITAL + np.concatenate([[0.0], np.cumsum(pnl_series[:-1])])
    pct_returns = pnl_series / np.maximum(equity_pre, 1.0)

    mean_r = float(np.mean(pct_returns))
    std_r  = float(np.std(pct_returns, ddof=1))

    if std_r < 1e-12:
        return 0.0

    # Time-based annualisation: scale by how many sim-durations fit in one year
    ann_factor = np.sqrt(SECONDS_PER_YEAR / sim_duration_s)
    return float(mean_r / std_r * ann_factor)


def _max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum Drawdown as a positive percentage (e.g. 1.2 means 1.2% MDD).

    Algorithm: O(N) single pass — peak-to-trough tracking.
    """
    values  = equity_curve.values.astype(float)
    peak    = values[0]
    max_dd  = 0.0
    for v in values:
        peak  = max(peak, v)
        dd    = (v - peak) / peak   # <= 0
        max_dd = min(max_dd, dd)
    return abs(max_dd) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Trade recorder
# ─────────────────────────────────────────────────────────────────────────────

def _record_trade(
    signal:    ae.ArbitrageSignal,
    tick:      ae.MarketTick,
    qty:       float = DEFAULT_QTY,
    fill_tick: "ae.MarketTick | None" = None,
    friction:  "FrictionModel | None" = None,
    latency_ms: float = 0.0,
) -> Trade:
    """
    Convert one ArbitrageSignal + its originating MarketTick into a Trade.

    Trade direction from signal.action:
      BUY_B_SELL_A → buy at ask(snap_b), sell at bid(snap_a)
      BUY_A_SELL_B → buy at ask(snap_a), sell at bid(snap_b)

    ``tick`` is the signal-time snapshot and supplies the *expectation*.
    ``fill_tick`` is the snapshot prevailing at ``T + latency_ms`` and supplies
    the *fill*; when it is None the two coincide and the result is the old
    frictionless behaviour, which on this path is tautological by construction.
    Both legs walk their side of the book, so a large order pays for depth.

    Fee: each leg charged at its own venue's taker rate, on what that leg filled.
    PnL: (sell_vwap - buy_vwap) × matched_qty − fee − legging.

    Note this path fills both legs against a single shared timeline: the synthetic
    generator emits one MarketTick carrying both venues' snapshots at the same
    timestamp, so the two legs cannot fill against different updates and legging
    risk is structurally absent here. The real-data path
    (``_run_real_data_vectorized``) draws latency per leg and does price it.
    """
    fr = friction if friction is not None else active_friction()
    fill = fill_tick if fill_tick is not None else tick

    if signal.action == "BUY_B_SELL_A":
        buy_snap   = fill.snap_b
        sell_snap  = fill.snap_a
        buy_venue  = str(tick.snap_b.exchange_id)
        sell_venue = str(tick.snap_a.exchange_id)
        expected_edge = (float(tick.snap_a.bids[0].price)
                         - float(tick.snap_b.asks[0].price)) * qty
        signal_buy_touch  = float(tick.snap_b.asks[0].price)
        signal_sell_touch = float(tick.snap_a.bids[0].price)
    else:  # BUY_A_SELL_B
        buy_snap   = fill.snap_a
        sell_snap  = fill.snap_b
        buy_venue  = str(tick.snap_a.exchange_id)
        sell_venue = str(tick.snap_b.exchange_id)
        expected_edge = (float(tick.snap_b.bids[0].price)
                         - float(tick.snap_a.asks[0].price)) * qty
        signal_buy_touch  = float(tick.snap_a.asks[0].price)
        signal_sell_touch = float(tick.snap_b.bids[0].price)

    # Walk each side for size instead of assuming the touch absorbs the order.
    buy_walk = walk_book(
        [float(lv.price) for lv in buy_snap.asks],
        [float(lv.volume) for lv in buy_snap.asks],
        qty,
    )
    sell_walk = walk_book(
        [float(lv.price) for lv in sell_snap.bids],
        [float(lv.volume) for lv in sell_snap.bids],
        qty,
    )

    buy_price  = buy_walk.vwap
    sell_price = sell_walk.vwap
    matched    = min(buy_walk.filled_qty, sell_walk.filled_qty)
    residual   = abs(buy_walk.filled_qty - sell_walk.filled_qty)

    if matched <= 0.0:
        # Neither side could supply anything — no fill exists to book.
        return Trade(
            timestamp_ms = signal.timestamp_ms,
            action       = signal.action,
            buy_venue    = buy_venue,
            sell_venue   = sell_venue,
            buy_price    = 0.0,
            sell_price   = 0.0,
            qty          = qty,
            fee          = 0.0,
            pnl_net      = 0.0,
            obi_delta    = signal.obi_delta,
            p_execute    = signal.p_execute,
            expected_edge_at_signal = expected_edge,
            adverse_selection_cost  = expected_edge,
            status       = "rejected",
            reason       = "no_liquidity",
            buy_latency_ms = latency_ms,
            sell_latency_ms = latency_ms,
        )

    fee = active_fees().taker_cost_legs(
        buy_walk.filled_qty, buy_venue, buy_price,
        sell_walk.filled_qty, sell_venue, sell_price,
    )
    legging = float(fr.legging_cost(
        residual, buy_price if buy_walk.filled_qty > sell_walk.filled_qty else sell_price
    ))
    realized_gross = (sell_price - buy_price) * matched
    pnl_net = realized_gross - fee - legging

    return Trade(
        timestamp_ms = signal.timestamp_ms,
        action       = signal.action,
        buy_venue    = buy_venue,
        sell_venue   = sell_venue,
        buy_price    = buy_price,
        sell_price   = sell_price,
        qty          = qty,
        fee          = fee,
        pnl_net      = pnl_net,
        obi_delta    = signal.obi_delta,
        p_execute    = signal.p_execute,
        expected_edge_at_signal = expected_edge,
        adverse_selection_cost  = expected_edge - realized_gross,
        signal_time_mid_a = 0.5 * (float(tick.snap_a.bids[0].price)
                                   + float(tick.snap_a.asks[0].price)),
        signal_time_mid_b = 0.5 * (float(tick.snap_b.bids[0].price)
                                   + float(tick.snap_b.asks[0].price)),
        fill_time_mid_a = 0.5 * (float(fill.snap_a.bids[0].price)
                                 + float(fill.snap_a.asks[0].price)),
        fill_time_mid_b = 0.5 * (float(fill.snap_b.bids[0].price)
                                 + float(fill.snap_b.asks[0].price)),
        buy_latency_ms  = latency_ms,
        sell_latency_ms = latency_ms,
        buy_slippage    = abs(buy_price - signal_buy_touch),
        sell_slippage   = abs(sell_price - signal_sell_touch),
        matched_qty      = matched,
        buy_filled_qty   = buy_walk.filled_qty,
        sell_filled_qty  = sell_walk.filled_qty,
        residual_qty     = residual,
        legging_cost     = legging,
        fill_ts_ms       = int(fill.timestamp_ms),
        adverse          = pnl_net < 0.0,
        spread_collapsed = expected_edge > 0.0 and realized_gross <= 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backtester orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Event-driven backtester for the Cross-Venue Arbitrage Predictor.

    Feeds MarketTick batches into the C++ SignalAggregator, records mock trade
    executions, and computes institutional risk metrics.

    Parameters
    ----------
    exchange_a        : str   — venue A (default "binance")
    exchange_b        : str   — venue B (default "kraken")
    latency_mu        : float — log-normal μ for network RTT
    latency_sigma     : float — log-normal σ (shape)
    alpha_lifetime_ms : float — discrepancy window (ms)
    delta_threshold   : float — Gate 1 minimum |ΔOBI| (default 0.3)
    min_p_execute     : float — Gate 2 minimum p_execute (default 0.80)
    qty               : float — BTC per trade (default 0.01)
    batch_size        : int   — ticks per evaluate() call (default 5000)
    friction          : str | FrictionModel — execution friction preset. Decides
                        how long the fill is delayed and what an unhedged
                        residual costs. Defaults to the active preset
                        ("stress", 100 ms); pass "zero" to reproduce the old
                        frictionless numbers. See src/friction.py.
    binance_path      : Path  — real data CSV (optional; auto-detected)
    kraken_path       : Path  — real data CSV (optional; auto-detected)
    generator_kwargs  : dict  — passed to SyntheticDataGenerator if used
    """

    _DEFAULT_BINANCE = Path("data/raw/binance_btcusdt.csv.gz")
    _DEFAULT_KRAKEN  = Path("data/raw/kraken_xbtusd.csv.gz")

    def __init__(
        self,
        exchange_a:        str   = "binance",
        exchange_b:        str   = "kraken",
        latency_mu:        float = 3.5,
        latency_sigma:     float = 0.4,
        alpha_lifetime_ms: float = 50.0,
        delta_threshold:   float = ae.DEFAULT_DELTA_THRESHOLD,
        min_p_execute:     float = ae.DEFAULT_MIN_P_EXECUTE,
        min_spread_pct:    float = 0.0012,
        qty:               float = DEFAULT_QTY,
        batch_size:        int   = 10_000,
        friction:          "str | FrictionModel | None" = None,
        binance_path:      Optional[Path] = None,
        kraken_path:       Optional[Path] = None,
        generator_kwargs:  Optional[dict] = None,
        use_synthetic:    bool = False,
    ) -> None:
        self.exchange_a        = exchange_a
        self.exchange_b        = exchange_b
        self.qty               = qty
        self.batch_size        = batch_size
        self.binance_path      = binance_path or self._DEFAULT_BINANCE
        self.kraken_path       = kraken_path  or self._DEFAULT_KRAKEN
        self.generator_kwargs  = generator_kwargs or {}
        self.use_synthetic    = bool(use_synthetic)

        self._aggregator = ae.SignalAggregator(
            exchange_a,
            exchange_b,
            latency_mu,
            latency_sigma,
            alpha_lifetime_ms,
            delta_threshold,
            min_p_execute,
            min_spread_pct,
        )

        # Execution friction is configured separately from the aggregator's
        # latency distribution. ``latency_mu``/``latency_sigma`` above parameterise
        # the C++ model's *p_execute* estimate; ``friction`` below decides when the
        # fill is actually evaluated and what it costs. They used to be the same
        # two numbers, which conflated a probability model with an execution model
        # and left the fill delay unconfigurable from outside this constructor.
        if friction is None:
            self.friction = active_friction()
        elif isinstance(friction, str):
            self.friction = friction_preset(friction)
        else:
            self.friction = friction

        self._simulator = ExecutionSimulator(friction=self.friction)

        logger.info(
            "Backtester initialised: %s vs %s | p_execute=%.4f | qty=%g BTC | "
            "friction=%s (%.1f ms) | fees=%s",
            exchange_a, exchange_b, self._aggregator.p_execute, qty,
            self.friction.name, self.friction.latency_ms, active_fees().name,
        )

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_ticks(self) -> tuple[list, str]:
        """Return (ticks_or_states, data_source_label). Auto-detects real vs. synthetic."""
        b_path = Path(self.binance_path)
        k_path = Path(self.kraken_path)

        if b_path.exists() and k_path.exists():
            logger.info("Real CSV files found — using src.ingestion pipeline.")
            from src.ingestion import load_market_states
            states = load_market_states(b_path, k_path, depth=BOOK_DEPTH)
            return states, "real CSV"

        if not self.use_synthetic:
            raise FileNotFoundError(
                "Real Binance and Kraken book files were not both found. "
                "Refusing to silently substitute synthetic data; pass "
                "use_synthetic=True for a demo run."
            )

        logger.info("Using explicitly requested synthetic GBM data.")
        gen   = SyntheticDataGenerator(**self.generator_kwargs)
        ticks = gen.generate()
        return ticks, "synthetic GBM"

    def _run_real_data_vectorized(
        self,
        train_frac: float = 1.0,
    ) -> BacktestResult:
        """
        Ultra-fast vectorised path for real market L2 data using pandas directly.
        Bypasses building millions of python objects, running in ~5 seconds total.

        Day-2 fix: when ``train_frac < 1.0``, evaluate only the first fraction of
        aligned ticks (in-sample).  The out-of-sample evaluation is the caller's
        responsibility — see ``run()``.
        """
        from src.ingestion import parse_tardis_csv

        b_path = Path(self.binance_path)
        k_path = Path(self.kraken_path)

        logger.info("Parsing raw CSVs directly via pandas...")
        df_a = parse_tardis_csv(b_path, "binance", depth=BOOK_DEPTH)
        df_b = parse_tardis_csv(k_path, "kraken", depth=BOOK_DEPTH)

        # Determine simulation duration based on raw timestamps
        ts_min = min(df_a.index.min(), df_b.index.min())
        ts_max = max(df_a.index.max(), df_b.index.max())
        sim_duration_s = (ts_max - ts_min) / 1000.0

        logger.info("Computing OBI vectors...")
        bid_cols = [f"bids[{i}].amount" for i in range(BOOK_DEPTH)]
        ask_cols = [f"asks[{i}].amount" for i in range(BOOK_DEPTH)]

        # Binance OBI
        vol_bid_a = df_a[bid_cols].sum(axis=1)
        vol_ask_a = df_a[ask_cols].sum(axis=1)
        denom_a = vol_bid_a + vol_ask_a
        df_a["obi"] = np.where(denom_a != 0.0, (vol_bid_a - vol_ask_a) / denom_a, 0.0)

        # Kraken OBI
        vol_bid_b = df_b[bid_cols].sum(axis=1)
        vol_ask_b = df_b[ask_cols].sum(axis=1)
        denom_b = vol_bid_b + vol_ask_b
        df_b["obi"] = np.where(denom_b != 0.0, (vol_bid_b - vol_ask_b) / denom_b, 0.0)

        # Extract best prices
        df_a["best_bid"] = df_a["bids[0].price"]
        df_a["best_ask"] = df_a["asks[0].price"]
        df_b["best_bid"] = df_b["bids[0].price"]
        df_b["best_ask"] = df_b["asks[0].price"]

        # Mid price for regime classification
        df_a["mid"] = 0.5 * (df_a["best_bid"] + df_a["best_ask"])
        df_b["mid"] = 0.5 * (df_b["best_bid"] + df_b["best_ask"])
        df_a["spread_bps"] = np.where(df_a["mid"] > 0,
                                      (df_a["best_ask"] - df_a["best_bid"]) / df_a["mid"] * 1e4, 0.0)
        df_b["spread_bps"] = np.where(df_b["mid"] > 0,
                                      (df_b["best_ask"] - df_b["best_bid"]) / df_b["mid"] * 1e4, 0.0)

        # Day-2: train/test split on the aligned timeline
        if train_frac < 1.0:
            cutoff_ms = int(ts_min + (ts_max - ts_min) * train_frac)
            df_a = df_a[df_a.index <= cutoff_ms]
            df_b = df_b[df_b.index <= cutoff_ms]
            logger.info("Day-2: in-sample window only, cutoff=%d ms", cutoff_ms)

        logger.info("Aligning venues via As-Of outer join...")
        df_a_sub = df_a[["obi", "best_bid", "best_ask", "mid", "spread_bps"]].rename(columns=lambda x: x + "_a")
        df_b_sub = df_b[["obi", "best_bid", "best_ask", "mid", "spread_bps"]].rename(columns=lambda x: x + "_b")

        df_a_sub = df_a_sub.groupby(level=0).last()
        df_b_sub = df_b_sub.groupby(level=0).last()

        aligned = pd.concat([df_a_sub, df_b_sub], axis=1).sort_index().ffill().dropna()
        n_ticks = len(aligned)

        # Realised volatility regime: rolling 1h std of mid returns
        aligned["mid_a_ret_1h"] = aligned["mid_a"].pct_change().rolling(3600, min_periods=60).std()
        aligned["mid_a_ret_1h"] = aligned["mid_a_ret_1h"].fillna(0.0)

        # Compute OBI delta
        aligned["obi_delta"] = aligned["obi_a"] - aligned["obi_b"]

        # Filter signals
        delta_threshold = float(self._aggregator.delta_threshold)

        # Spread-based filter: Ensure gross margin covers taker fees.
        # Rates are per-venue now, so each leg is charged at the venue it fills
        # on. Fees below are per unit of base asset, matching `margin`; qty
        # cancels out of the inequality.
        #
        # This gate is still `margin > fee` on the signal-time snapshot, and that
        # is fine now, because it is no longer the same inequality as the booked
        # PnL. The gate reads the book at T; the fill reads the book prevailing at
        # T + latency_ms and walks it for size, so `margin` is a *forecast* that
        # the fill can and does falsify. Under the default 100 ms stress preset
        # the book has moved on 89.6% of signal rows.
        #
        # What that changes, measured on data/raw/ 2024-03-01, full day, both
        # venues, institutional fees, stress preset (see
        # backtest/sweep_order_size.py, which produced these):
        #   qty BTC   win%    adverse%   unfilled%   mean edge
        #     0.01    98.6       1.4        0.01      4.25 bps
        #     0.05    98.1       1.9        0.04      4.21 bps
        #     0.25    97.1       2.9        0.22      4.14 bps
        #     1.00    95.0       5.0        1.14      4.00 bps
        #     3.00    85.3      14.7        7.25      3.40 bps
        #
        #   * A losing trade is now reachable, so the strategy is falsifiable:
        #     4,329 adverse fills at 0.01 BTC, 45,269 at 3.0 BTC.
        #   * friction="zero" removes the latency but NOT the book walk, so it
        #     gives 99.3% at 0.01 BTC and 91.4% at 3.0 BTC, not the old 100%.
        #     Size alone is enough to lose money: the gate quotes the touch, the
        #     fill pays the VWAP of five levels. Recovering a true 100% would take
        #     zero latency *and* an infinitely deep top level.
        #   * Latency alone barely bites at small size, and that is a property of
        #     the feed, not a bug: binance_book_snapshot_5 updates on a ~100 ms
        #     cadence, so a 100 ms delay advances the book exactly one row and the
        #     touch price changes on only 14% of rows (kraken 8.4%). Size is what
        #     bites — VWAP over five levels.
        #   * Win rate stays high partly because the entry gate selects deep
        #     books: on signal rows the buy leg absorbs a full 3.0 BTC 87.4% of
        #     the time versus 55% of rows unconditionally. Slippage estimates
        #     taken over an unselected sample overstate what this strategy pays.
        #
        # What it does *not* change: total PnL stays positive at every size,
        # because the edge in this dataset is the persistent +4.79 bps USDT/USD
        # quote-currency basis (TEST_REPORT.md 2.3) and no amount of delay removes
        # a structural basis. 99.7% of signals are the same direction (buy
        # binance/USDT, sell kraken/USD), which is what a basis looks like.
        # Breaking the tautology makes the win rate mean something; it does not
        # make this an arbitrage.
        _f = active_fees()
        rate_a = _f.taker_rate(VENUE_A)   # binance
        rate_b = _f.taker_rate(VENUE_B)   # kraken

        # BUY_B_SELL_A (obi > 0): Buy Kraken (ask_b), Sell Binance (bid_a)
        margin_b_a = aligned["best_bid_a"] - aligned["best_ask_b"]
        fee_b_a = aligned["best_ask_b"] * rate_b + aligned["best_bid_a"] * rate_a
        cond_b_a = (aligned["obi_delta"] > delta_threshold) & (margin_b_a > fee_b_a)

        # BUY_A_SELL_B (obi < 0): Buy Binance (ask_a), Sell Kraken (bid_b)
        margin_a_b = aligned["best_bid_b"] - aligned["best_ask_a"]
        fee_a_b = aligned["best_ask_a"] * rate_a + aligned["best_bid_b"] * rate_b
        cond_a_b = (aligned["obi_delta"] < -delta_threshold) & (margin_a_b > fee_a_b)

        signals_df = aligned[cond_b_a | cond_a_b].copy()
        n_signals = len(signals_df)
        
        logger.info("Generated %d signals from %d aligned ticks.", n_signals, n_ticks)
        
        p_execute = float(self._aggregator.p_execute)

        trades: list[Trade] = []
        if n_signals > 0:
            ts_arr = signals_df.index.to_numpy(dtype=np.int64)
            obi_delta_arr = signals_df["obi_delta"].to_numpy(dtype=float)

            best_bid_a_arr = signals_df["best_bid_a"].to_numpy(dtype=float)
            best_ask_a_arr = signals_df["best_ask_a"].to_numpy(dtype=float)
            best_bid_b_arr = signals_df["best_bid_b"].to_numpy(dtype=float)
            best_ask_b_arr = signals_df["best_ask_b"].to_numpy(dtype=float)

            # Day-2: regime arrays (NaN-safe; zeros treated as unknown)
            if "mid_a_ret_1h" in signals_df.columns:
                vol_arr = signals_df["mid_a_ret_1h"].to_numpy(dtype=float)
            else:
                vol_arr = np.zeros(n_signals, dtype=float)
            spread_arr = signals_df["spread_bps_a"].to_numpy(dtype=float) if "spread_bps_a" in signals_df.columns else np.full(n_signals, np.nan)

            # ── Execution: buffer, then fill against the prevailing book ──────
            #
            # The direction split has to happen before the fill, because which
            # venue supplies the ask and which supplies the bid depends on the
            # sign of obi_delta. Each direction is filled as one vectorised batch
            # and the results are scattered back into signal order.
            logger.info(
                "Simulating fills: %s preset, %.1f ms latency, qty=%g BTC, "
                "VWAP over %d levels...",
                self.friction.name, self.friction.latency_ms, self.qty, BOOK_DEPTH,
            )
            book_a = _VenueBook(VENUE_A, df_a)
            book_b = _VenueBook(VENUE_B, df_b)
            fees = active_fees()
            rng = np.random.default_rng(0)

            is_b_a = obi_delta_arr > 0          # BUY_B_SELL_A
            fills = _empty_fills(n_signals)
            for mask, buy_bk, sell_bk in (
                (is_b_a, book_b, book_a),
                (~is_b_a, book_a, book_b),
            ):
                if not mask.any():
                    continue
                sub = _simulate_fills(
                    buy_bk, sell_bk, ts_arr[mask], self.qty,
                    self.friction, fees, rng,
                )
                for key, values in sub.items():
                    fills[key][mask] = values

            ok_arr = fills["ok"]
            n_no_book = int((~ok_arr).sum())
            if n_no_book:
                logger.info(
                    "%d/%d signals had no prevailing book on one venue at fill "
                    "time (rejected, PnL 0).", n_no_book, n_signals,
                )

            logger.info("Building Trade records (with look-ahead accounting)...")
            for idx in range(n_signals):
                ts = int(ts_arr[idx])
                obi_delta = float(obi_delta_arr[idx])

                bid_a = best_bid_a_arr[idx]
                ask_a = best_ask_a_arr[idx]
                bid_b = best_bid_b_arr[idx]
                ask_b = best_ask_b_arr[idx]

                # ── Signal-time mids: the snapshot we are claiming to predict ──
                signal_mid_a = 0.5 * (bid_a + ask_a)
                signal_mid_b = 0.5 * (bid_b + ask_b)

                # ── Regime bucketing (Day-2) ──────────────────────────────────
                v = float(vol_arr[idx]) if not np.isnan(vol_arr[idx]) else 0.0
                if   v < 0.0015: regime_vol = "low"
                elif v < 0.0030: regime_vol = "mid"
                else:            regime_vol = "high"

                s = float(spread_arr[idx]) if not np.isnan(spread_arr[idx]) else 5.0
                if   s < 2.0:  regime_spread = "tight"
                elif s < 5.0:  regime_spread = "normal"
                else:          regime_spread = "wide"

                # In-sample / out-of-sample flag based on absolute timestamp
                split_label = "train" if (ts - ts_min) <= (ts_max - ts_min) * train_frac else "test"

                if obi_delta > 0:
                    action     = "BUY_B_SELL_A"
                    buy_venue  = VENUE_B
                    sell_venue = VENUE_A
                    # Expected edge at signal: sell at A's bid, buy at B's ask
                    expected_edge = (bid_a - ask_b) * self.qty
                    buy_touch_signal, sell_touch_signal = ask_b, bid_a
                else:
                    action     = "BUY_A_SELL_B"
                    buy_venue  = VENUE_A
                    sell_venue = VENUE_B
                    expected_edge = (bid_b - ask_a) * self.qty
                    buy_touch_signal, sell_touch_signal = ask_a, bid_b

                # ---- No prevailing book on one venue: the only rejection left ----
                #
                # There is no price at which an order can fill against a venue that
                # has not quoted yet, so this is a genuine impossibility rather than
                # a discarded loss. Everything else — including a fill that lost
                # money — is booked below. The branch this replaces rejected any
                # fill more than 5 bps worse than the signal price, which is what
                # held the win rate at ~100%.
                if not ok_arr[idx]:
                    trades.append(Trade(
                        timestamp_ms = ts,
                        action       = action,
                        buy_venue    = buy_venue,
                        sell_venue   = sell_venue,
                        buy_price    = 0.0,
                        sell_price   = 0.0,
                        qty          = self.qty,
                        fee          = 0.0,
                        pnl_net      = 0.0,
                        obi_delta    = obi_delta,
                        p_execute    = p_execute,
                        expected_edge_at_signal = expected_edge,
                        adverse_selection_cost  = expected_edge,   # we lost the whole edge
                        fill_time_mid_a = signal_mid_a,
                        fill_time_mid_b = signal_mid_b,
                        signal_time_mid_a = signal_mid_a,
                        signal_time_mid_b = signal_mid_b,
                        status       = "rejected",
                        reason       = "no_book",
                        buy_latency_ms = float(fills["buy_latency"][idx]),
                        sell_latency_ms = float(fills["sell_latency"][idx]),
                        regime_vol    = regime_vol,
                        regime_spread = regime_spread,
                        split         = split_label,
                    ))
                    continue

                buy_price  = float(fills["buy_vwap"][idx])
                sell_price = float(fills["sell_vwap"][idx])
                matched    = float(fills["matched"][idx])
                buy_filled = float(fills["buy_filled"][idx])
                sell_filled = float(fills["sell_filled"][idx])
                residual   = float(fills["residual"][idx])
                fee        = float(fills["fee"][idx])
                legging    = float(fills["legging"][idx])
                pnl_net    = float(fills["pnl_net"][idx])

                if buy_venue == VENUE_A:
                    fill_mid_a = float(fills["fill_mid_buy"][idx])
                    fill_mid_b = float(fills["fill_mid_sell"][idx])
                else:
                    fill_mid_a = float(fills["fill_mid_sell"][idx])
                    fill_mid_b = float(fills["fill_mid_buy"][idx])

                # Honest adverse-selection: gross edge captured at fill
                # vs gross edge we would have captured at signal time. Both sides
                # are USD on the *requested* size, so a short fill shows up here as
                # forgone edge rather than disappearing from the comparison.
                realized_gross = float(fills["gross"][idx])
                adverse_selection_cost = expected_edge - realized_gross

                # The signal predicted a positive edge; did any of it survive?
                # `spread_collapsed` isolates the case the requirement names
                # explicitly — the quotes moved together during the latency window
                # and the gross edge went away before fees were even charged.
                spread_collapsed = expected_edge > 0.0 and realized_gross <= 0.0

                trades.append(Trade(
                    timestamp_ms = ts,
                    action       = action,
                    buy_venue    = buy_venue,
                    sell_venue   = sell_venue,
                    buy_price    = buy_price,
                    sell_price   = sell_price,
                    qty          = self.qty,
                    fee          = fee,
                    pnl_net      = pnl_net,
                    obi_delta    = obi_delta,
                    p_execute    = p_execute,
                    expected_edge_at_signal = expected_edge,
                    adverse_selection_cost  = adverse_selection_cost,
                    fill_time_mid_a         = fill_mid_a,
                    fill_time_mid_b         = fill_mid_b,
                    signal_time_mid_a       = signal_mid_a,
                    signal_time_mid_b       = signal_mid_b,
                    status       = "filled",
                    buy_latency_ms = float(fills["buy_latency"][idx]),
                    sell_latency_ms = float(fills["sell_latency"][idx]),
                    buy_slippage = abs(buy_price - buy_touch_signal),
                    sell_slippage = abs(sell_price - sell_touch_signal),
                    matched_qty      = matched,
                    buy_filled_qty   = buy_filled,
                    sell_filled_qty  = sell_filled,
                    residual_qty     = residual,
                    legging_cost     = legging,
                    fill_ts_ms       = int(fills["fill_ts"][idx]),
                    adverse          = pnl_net < 0.0,
                    spread_collapsed = spread_collapsed,
                    regime_vol    = regime_vol,
                    regime_spread = regime_spread,
                    split         = split_label,
                ))

        logger.info("Computing equity curve and performance metrics...")
        equity_curve     = _compute_equity_curve(trades)
        total_return_pct = _total_return(equity_curve)
        sharpe           = _annualised_sharpe(trades, sim_duration_s)
        mdd              = _max_drawdown(equity_curve)

        result = BacktestResult(
            trades           = trades,
            equity_curve     = equity_curve,
            total_return_pct = total_return_pct,
            sharpe_ratio     = sharpe,
            max_drawdown_pct = mdd,
            n_ticks          = n_ticks,
            n_signals        = n_signals,
            sim_duration_s   = sim_duration_s,
            data_source      = "real L2 CSV",
        )
        print(result.summary())
        return result

    # ── Core run loop ─────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """
        Execute the full backtest.

        Real CSV data  -> _run_real_data_vectorized() — direct pandas pipeline.
        Synthetic data -> C++ arbitrage_engine.SignalAggregator.evaluate() loop.

        Returns
        -------
        BacktestResult
        """
        b_path = Path(self.binance_path) if self.binance_path else None
        k_path = Path(self.kraken_path) if self.kraken_path else None

        if b_path and k_path and b_path.exists() and k_path.exists():
            return self._run_real_data_vectorized()

        # ── Synthetic data: C++ evaluate() batch loop ──────────────────────
        ticks, data_source = self._load_ticks()
        logger.info("Running backtest on %d ticks from %s...", len(ticks), data_source)

        trades:    List[Trade] = []
        n_signals: int         = 0
        tick_by_ts: dict[int, ae.MarketTick] = {t.timestamp_ms: t for t in ticks}

        # Pending-order buffer for the synthetic path. The signal is produced from
        # the tick at T; the fill is evaluated against the tick prevailing at
        # T + latency_ms, found by binary search on the tick timeline. Same
        # mechanism as the real-data path, one shared timeline instead of two.
        tick_ts = np.array([t.timestamp_ms for t in ticks], dtype=np.int64)
        latency_ms = float(self.friction.latency_ms)

        def _fill_tick_for(ts_ms: int) -> "ae.MarketTick | None":
            if latency_ms <= 0.0:
                return None            # frictionless: fill on the signal's own tick
            row = int(prevailing_row(tick_ts, np.int64(ts_ms + latency_ms)))
            return ticks[row] if row >= 0 else None

        for batch_start in range(0, len(ticks), self.batch_size):
            batch   = ticks[batch_start : batch_start + self.batch_size]
            signals = self._aggregator.evaluate(batch)
            n_signals += len(signals)
            for sig in signals:
                tick = tick_by_ts.get(sig.timestamp_ms)
                if tick is None:
                    continue
                trades.append(_record_trade(
                    sig, tick, self.qty,
                    fill_tick=_fill_tick_for(sig.timestamp_ms),
                    friction=self.friction,
                    latency_ms=latency_ms,
                ))

        sim_duration_s = (
            (ticks[-1].timestamp_ms - ticks[0].timestamp_ms) / 1000.0
            if len(ticks) >= 2 else 0.0
        )

        equity_curve     = _compute_equity_curve(trades)
        total_return_pct = _total_return(equity_curve)
        sharpe           = _annualised_sharpe(trades, sim_duration_s)
        mdd              = _max_drawdown(equity_curve)

        result = BacktestResult(
            trades           = trades,
            equity_curve     = equity_curve,
            total_return_pct = total_return_pct,
            sharpe_ratio     = sharpe,
            max_drawdown_pct = mdd,
            n_ticks          = len(ticks),
            n_signals        = n_signals,
            sim_duration_s   = sim_duration_s,
            data_source      = data_source,
        )
        print(result.summary())
        return result
