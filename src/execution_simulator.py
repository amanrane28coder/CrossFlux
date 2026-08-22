"""Synthetic cross-venue fill simulator: book walking, fees, latency, rejection.

Where the costs are charged, and the one rule that matters
---------------------------------------------------------
A raw spread is charged three things, in this order:

  entry gate    At signal time, on the touch prices: the spread must beat fees by
                ``MIN_PROFIT_NET_BPS``. This is a *forecast* the fill is allowed
                to falsify.
  latency       The order arrives ``latency_ms`` later, and the book it arrives at
                has moved against it by ``latency_ms / 1000 * 0.01`` on each side.
  book walk     Each leg pays the VWAP of the levels it consumes, not the touch.

**Nothing is re-gated after the latency move.** That is the whole point. This
module used to reject three times *after* friction had been applied -- on a
collapsed spread, on a spread that fell below the minimum, and on a negative
net PnL -- and each rejection booked ``pnl_net = 0.0``. Recomputing the edge once
the costs are known and then declining the trades that came out badly keeps
exactly the winners, which is the tautology TEST_REPORT.md 2.1 documents; the
`fees_exceed_profit` reject was that defect in its purest form, an `if net_pnl <=
0: don't count it`. The C++ path removed the same three gates
(``cpp_engine/src/execution_manager.cpp``, "No gate here") and this now matches
it: whatever friction leaves is booked, negative included, flagged ``adverse``.

The rejections that remain are all "no trade happened", never "the trade went
badly": the signal-time gate, no liquidity to size against, and a book that
filled nothing at any price.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.fees import active as active_fees

# Taker fees are no longer defined here. They live in src/fees.py, shared with
# backtest/engine.py and the C++ engine, and are selected by preset
# ($CROSSFLUX_FEE_PRESET or src.fees.set_active). The rates this module used to
# hardcode -- binance 0.0004 / kraken 0.0010 -- are now the "retail" preset.
#
# Read rates through active_fees() at call time rather than caching them at
# import: set_active() is meant to work mid-process for preset sweeps, and a
# module-level snapshot would silently ignore it.
LATENCY_MEAN_MS = 30.0
LATENCY_STD_MS = 5.0
MIN_PROFIT_NET_BPS = 0.5

@dataclass
class FillResult:
    vwap: float
    filled_qty: float
    levels_consumed: int
    slippage_bps: float
    fees_paid: float
    partial: bool

@dataclass
class ExecutionReport:
    buy_fill: Optional[FillResult]
    sell_fill: Optional[FillResult]
    gross_pnl: float
    net_pnl: float
    total_slippage_bps: float
    total_fees: float
    rejected: bool
    reject_reason: str = ""
    latency_ms: float = 0.0
    # Booked at a loss: entered on a positive edge, came back negative because the
    # spread moved inside the latency window or the book walk cost more than the
    # edge was worth. `rejected` is False for these -- the trade happened. The
    # name of `reject_reason` is a misnomer for them and is kept anyway, because
    # the C++ SimOrder labels booked adverse fills through the same field and the
    # same CSV column ("adverse_fill" / "adverse_fill_spread_collapsed").
    adverse: bool = False


def walk_book(
    levels: list,
    side: str,
    qty: float,
    venue: str = "binance",
) -> FillResult:
    if qty <= 0 or not levels:
        return FillResult(0.0, 0.0, 0, 0.0, 0.0, False)

    remaining = qty
    total_cost = 0.0
    total_filled = 0.0
    levels_consumed = 0

    for level in levels:
        if remaining <= 1e-10:
            break
        price = level.price
        vol = level.volume
        if price <= 0 or vol <= 1e-10:
            levels_consumed += 1
            continue
        take = min(remaining, vol)
        total_cost += take * price
        total_filled += take
        remaining -= take
        levels_consumed += 1

    if total_filled <= 1e-10:
        return FillResult(0.0, 0.0, levels_consumed, 0.0, 0.0, False)

    vwap = total_cost / total_filled
    best_price = levels[0].price
    slippage_bps = abs(vwap - best_price) / best_price * 10000.0 if best_price > 0 else 0.0
    fee_rate = active_fees().taker_rate(venue)
    fees_paid = total_cost * fee_rate

    return FillResult(
        vwap=vwap,
        filled_qty=total_filled,
        levels_consumed=levels_consumed,
        slippage_bps=slippage_bps,
        fees_paid=fees_paid,
        partial=remaining > 1e-10,
    )


def simulate_latency(buy_price: float, sell_price: float) -> Tuple[float, float, float, float]:
    """Draw a latency and move both touch prices against us by it.

    Returns ``(latency_ms, moved_buy, moved_sell, drift)`` where ``drift`` is the
    fractional move applied to each side.

    It no longer returns a ``spread_ok`` flag. The flag's only consumer was a
    rejection that discarded any order whose spread collapsed during the window,
    and returning it invites that rejection back. A collapsed spread is a losing
    trade, not a cancelled one.
    """
    latency_ms = max(0.0, random.gauss(LATENCY_MEAN_MS, LATENCY_STD_MS))
    drift = (latency_ms / 1000.0) * 0.01
    return latency_ms, buy_price * (1.0 + drift), sell_price * (1.0 - drift), drift


@dataclass(frozen=True)
class _Level:
    """A price level with a shifted price. Deliberately not the caller's object.

    ``drift_book`` must not mutate the books it is handed: they are the caller's
    market data, they are usually shared between the two actions, and a fill that
    silently edited them would make every later call read a book that had already
    been moved once per trade.
    """

    price: float
    volume: float


def drift_book(levels: list, factor: float) -> List[_Level]:
    """The book as it looks after the latency window: every level shifted.

    A parallel shift, volumes untouched. The latency model here has exactly one
    parameter -- elapsed time -- and says nothing about how depth reshapes, so
    reshaping it would be inventing information. This is weaker than the
    backtester and the C++ engine, which read the *actual* prevailing book at
    T + latency out of the feed; a synthetic simulator has no future book to read.

    Shifting before walking keeps the two costs separable: the vwaps carry the
    latency move, and ``slippage_bps`` -- measured against the shifted touch --
    stays a pure book-walk cost. Measuring slippage against the pre-latency touch
    would fold the drift into it and count it twice.
    """
    return [_Level(l.price * factor, l.volume) for l in levels]


def simulate_cross_venue_fill(
    bids_a: list,
    asks_a: list,
    bids_b: list,
    asks_b: list,
    action,
    qty: float,
    venue_a: str = "binance",
    venue_b: str = "kraken",
) -> ExecutionReport:
    # Normalise action — may be int (0/1) or str ("BUY_A_SELL_B"/"BUY_B_SELL_A")
    if isinstance(action, str):
        action = 0 if action == "BUY_A_SELL_B" else 1
    if action == 0:
        buy_venue = venue_a
        sell_venue = venue_b
        buy_book = asks_a
        sell_book = bids_b
        buy_price = buy_book[0].price if buy_book else 0.0
        sell_price = sell_book[0].price if sell_book else 0.0
    else:
        buy_venue = venue_b
        sell_venue = venue_a
        buy_book = asks_b
        sell_book = bids_a
        buy_price = buy_book[0].price if buy_book else 0.0
        sell_price = sell_book[0].price if sell_book else 0.0

    gross_spread_bps = (sell_price - buy_price) / buy_price * 10000.0 if buy_price > 0 else 0.0
    total_fee_rate = active_fees().round_trip_rate(buy_venue, sell_venue)
    net_spread_bps = gross_spread_bps - total_fee_rate * 10000.0

    # Reject if net spread after fees is too small
    if net_spread_bps < MIN_PROFIT_NET_BPS:
        return ExecutionReport(
            None, None, 0.0, 0.0, 0.0, 0.0,
            rejected=True, reject_reason=f"net_spread_{net_spread_bps:.2f}bps_below_min_{MIN_PROFIT_NET_BPS}bps",
            latency_ms=0.0,
        )

    # Simulate latency. The prices move against us during the window, and from
    # here on the *drifted* book is the only book: walking the signal-time book
    # after computing a drift would make latency free, which is how this function
    # behaved when the drift was used for nothing but the gates below it.
    latency_ms, moved_buy, moved_sell, drift = simulate_latency(buy_price, sell_price)
    fill_buy_book = drift_book(buy_book, 1.0 + drift)
    fill_sell_book = drift_book(sell_book, 1.0 - drift)

    # No gate here, and none after the fill either. The edge was checked above on
    # the signal-time touch and the order went out on the strength of it. Three
    # rejections used to live in this gap -- spread_collapsed_during_latency,
    # spread_below_min_after_latency, fees_exceed_profit -- and all three booked
    # pnl_net = 0.0, so the ledger kept only the trades friction happened to
    # spare. Requirement #4: the trade must still execute and book the loss.

    available_buy = sum(l.volume for l in fill_buy_book)
    available_sell = sum(l.volume for l in fill_sell_book)
    buy_qty = min(qty, available_buy)
    sell_qty = min(qty, available_sell)
    # Hedge on the smaller side. Taking the larger would book gross on a size one
    # venue never supplied.
    used_qty = min(buy_qty, sell_qty)
    if used_qty <= 1e-10:
        return ExecutionReport(
            None, None, 0.0, 0.0, 0.0, 0.0,
            rejected=True, reject_reason="no_liquidity",
            latency_ms=latency_ms,
        )

    buy_fill = walk_book(fill_buy_book, "buy", used_qty, buy_venue)
    sell_fill = walk_book(fill_sell_book, "sell", used_qty, sell_venue)

    # Nothing filled on either leg. This is the one "not booked" case left after
    # the fill, and it is not a trade that went badly -- it is a book that could
    # not supply the order at any price, e.g. every level non-positive. Named to
    # match the C++ path's SimOrder; it was previously mislabelled "partial_fill",
    # which describes the opposite condition.
    if buy_fill.filled_qty <= 1e-10 or sell_fill.filled_qty <= 1e-10:
        return ExecutionReport(
            buy_fill, sell_fill, 0.0, 0.0,
            buy_fill.slippage_bps + sell_fill.slippage_bps,
            buy_fill.fees_paid + sell_fill.fees_paid,
            rejected=True, reject_reason="no_liquidity_at_fill",
            latency_ms=latency_ms,
        )

    hedged = min(buy_fill.filled_qty, sell_fill.filled_qty)
    gross_pnl = (sell_fill.vwap - buy_fill.vwap) * hedged
    total_fees = buy_fill.fees_paid + sell_fill.fees_paid
    net_pnl = gross_pnl - total_fees
    total_slippage = buy_fill.slippage_bps + sell_fill.slippage_bps

    # Adverse selection: entered on a positive edge, came back negative. Booked.
    # `spread_collapsed` distinguishes the case the requirement names -- the move
    # inside the latency window turned the spread over on its own -- from a spread
    # that survived but could not cover the walk and the fees. Same two labels the
    # C++ path writes.
    adverse = net_pnl < 0.0
    reason = ""
    if adverse:
        reason = ("adverse_fill_spread_collapsed"
                  if moved_sell - moved_buy <= 0.0 < sell_price - buy_price
                  else "adverse_fill")

    return ExecutionReport(
        buy_fill=buy_fill,
        sell_fill=sell_fill,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        total_slippage_bps=total_slippage,
        total_fees=total_fees,
        rejected=False,
        reject_reason=reason,
        latency_ms=latency_ms,
        adverse=adverse,
    )
