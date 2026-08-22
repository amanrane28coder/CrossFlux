"""
tests/test_engine_friction.py
=============================
Tests for the execution-friction path added to ``backtest/engine.py``: the
pending-order buffer, per-venue VWAP fills, and adverse fills being *booked*
rather than discarded.

These use hand-built two-row order books rather than the real CSVs, so every
expected number is computed by hand in the test itself. ``tests/test_friction.py``
already covers ``src/friction.py`` in isolation; this file covers the wiring —
that the engine reads the *prevailing* book at ``T + latency``, charges each leg
at its own venue, and lets a trade lose money.

The regression that motivated most of it: the previous engine rejected any fill
worse than 5 bps from the signal price and booked ``pnl_net = 0``. That kept
favourable fills and dropped unfavourable ones, so the ledger held only the good
tail and the win rate was 100% by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import (
    BOOK_DEPTH,
    VENUE_A,
    VENUE_B,
    Backtester,
    Trade,
    _compute_equity_curve,
    _empty_fills,
    _FILL_FIELDS,
    _simulate_fills,
    _VenueBook,
)
from src.fees import active as active_fees
from src.friction import FrictionModel, get_preset


# ── helpers ──────────────────────────────────────────────────────────────────

def make_frame(rows: dict[int, tuple[list[tuple[float, float]], list[tuple[float, float]]]]
               ) -> pd.DataFrame:
    """Build a book frame from {ts_ms: (bids, asks)} with (price, amount) pairs.

    Missing levels are zero-padded, which is what a five-level Tardis snapshot
    does when the venue publishes fewer levels. src.friction treats a zero
    amount as absent, not as depth.
    """
    index, data = [], []
    for ts, (bids, asks) in rows.items():
        index.append(ts)
        rec: dict[str, float] = {}
        for i in range(BOOK_DEPTH):
            bp, ba = bids[i] if i < len(bids) else (0.0, 0.0)
            ap, aa = asks[i] if i < len(asks) else (0.0, 0.0)
            rec[f"bids[{i}].price"] = bp
            rec[f"bids[{i}].amount"] = ba
            rec[f"asks[{i}].price"] = ap
            rec[f"asks[{i}].amount"] = aa
        data.append(rec)
    return pd.DataFrame(data, index=pd.Index(index, name="ts_ms"))


FLAT = [(100.0, 10.0)], [(101.0, 10.0)]      # bids, asks — deep and static


# ── _VenueBook: the as-of lookup ─────────────────────────────────────────────

def test_rows_at_returns_prevailing_not_next_quote() -> None:
    """Look-ahead check. searchsorted without the -1 gives the *next* quote."""
    book = _VenueBook(VENUE_A, make_frame({1000: FLAT, 2000: FLAT, 3000: FLAT}))
    got = book.rows_at(np.array([1000, 1500, 1999, 2000, 9999], dtype=np.int64))
    assert list(got) == [0, 0, 0, 1, 2]


def test_rows_at_is_negative_before_the_first_quote() -> None:
    book = _VenueBook(VENUE_A, make_frame({1000: FLAT, 2000: FLAT}))
    got = book.rows_at(np.array([0, 999], dtype=np.int64))
    assert list(got) == [-1, -1], "a fill before the book opens must not resolve"


def test_venue_book_sorts_a_shuffled_index() -> None:
    """src/ingestion.py never sorts its index, so the guard has to be here."""
    frame = make_frame({3000: FLAT, 1000: FLAT, 2000: FLAT})
    assert not frame.index.is_monotonic_increasing
    book = _VenueBook(VENUE_A, frame)
    assert book.index.tolist() == [1000, 2000, 3000]
    assert book.rows_at(np.array([2500], dtype=np.int64))[0] == 1


def test_levels_are_gathered_in_book_order() -> None:
    book = _VenueBook(VENUE_A, make_frame(
        {1000: ([(100.0, 1.0), (99.0, 2.0)], [(101.0, 3.0), (102.0, 4.0)])}))
    px = book.levels("asks", "price", np.array([0]))
    am = book.levels("asks", "amount", np.array([0]))
    assert px.shape == (1, BOOK_DEPTH)
    assert px[0, 0] == 101.0 and px[0, 1] == 102.0
    assert am[0, 0] == 3.0 and am[0, 1] == 4.0


# ── _simulate_fills: the pending-order buffer ────────────────────────────────

def sim(buy_frame, sell_frame, ts, qty, latency_ms, legging_bps=0.0):
    fr = FrictionModel(name="t", latency_ms=latency_ms, jitter_log_sigma=0.0,
                       legging_cost_bps=legging_bps)
    return _simulate_fills(
        _VenueBook(VENUE_A, buy_frame), _VenueBook(VENUE_B, sell_frame),
        np.asarray(ts, dtype=np.int64), qty, fr, active_fees(),
        np.random.default_rng(0),
    )


def test_zero_latency_reads_the_signals_own_book() -> None:
    frame = make_frame({1000: ([(100.0, 5.0)], [(101.0, 5.0)]),
                        1100: ([(1.0, 5.0)], [(999.0, 5.0)])})
    out = sim(frame, frame, [1000], qty=1.0, latency_ms=0.0)
    assert out["buy_vwap"][0] == 101.0
    assert out["sell_vwap"][0] == 100.0


def test_latency_moves_the_fill_to_the_later_book() -> None:
    """The whole point: at T+100 the 1100 row is prevailing, so it is what fills."""
    buy = make_frame({1000: ([(100.0, 5.0)], [(101.0, 5.0)]),
                      1100: ([(100.0, 5.0)], [(105.0, 5.0)])})
    sell = make_frame({1000: FLAT, 1100: FLAT})
    out = sim(buy, sell, [1000], qty=1.0, latency_ms=100.0)
    assert out["buy_vwap"][0] == 105.0, "fill must use the book prevailing at T+100"
    assert out["buy_touch"][0] == 105.0


def test_a_fill_the_gate_would_not_have_taken_is_still_booked() -> None:
    """Adverse selection. The old code rejected this and booked pnl_net = 0."""
    buy = make_frame({1000: ([(100.0, 5.0)], [(100.5, 5.0)]),
                      1100: ([(100.0, 5.0)], [(120.0, 5.0)])})
    sell = make_frame({1000: FLAT, 1100: FLAT})
    out = sim(buy, sell, [1000], qty=1.0, latency_ms=100.0)
    assert out["ok"][0], "an adverse move is a fill, not a rejection"
    assert out["matched"][0] == 1.0
    assert out["pnl_net"][0] < 0.0
    # gross = (sell 100 - buy 120) * 1 = -20, before fees.
    assert out["gross"][0] == -20.0


def test_vwap_walks_levels_and_the_gate_price_is_not_the_fill_price() -> None:
    buy = make_frame({1000: ([(100.0, 9.0)], [(101.0, 1.0), (102.0, 1.0), (110.0, 8.0)])})
    sell = make_frame({1000: ([(100.0, 10.0)], [(101.0, 10.0)])})
    out = sim(buy, sell, [1000], qty=2.0, latency_ms=0.0)
    assert out["buy_touch"][0] == 101.0
    assert out["buy_vwap"][0] == 101.5      # (1*101 + 1*102) / 2
    assert out["buy_filled"][0] == 2.0


def test_partial_fill_matches_the_smaller_leg_and_charges_the_residual() -> None:
    """Legging risk: the over-filled leg is a naked position, not free size."""
    buy = make_frame({1000: ([(100.0, 9.0)], [(101.0, 3.0)])})     # can buy 3
    sell = make_frame({1000: ([(100.0, 1.0)], [(101.0, 9.0)])})    # can sell 1
    out = sim(buy, sell, [1000], qty=3.0, latency_ms=0.0, legging_bps=10.0)
    assert out["buy_filled"][0] == 3.0
    assert out["sell_filled"][0] == 1.0
    assert out["matched"][0] == 1.0
    assert out["residual"][0] == 2.0
    # residual unwinds at the over-filling venue's price: 2 * 101 * 10bps
    assert np.isclose(out["legging"][0], 2.0 * 101.0 * 10e-4)


def test_fee_is_charged_on_what_each_leg_filled_not_on_the_match() -> None:
    """An over-filled leg pays its fee in full and earns nothing back."""
    buy = make_frame({1000: ([(100.0, 9.0)], [(101.0, 3.0)])})
    sell = make_frame({1000: ([(100.0, 1.0)], [(101.0, 9.0)])})
    out = sim(buy, sell, [1000], qty=3.0, latency_ms=0.0)
    fees = active_fees()
    expected = fees.taker_cost_legs(3.0, VENUE_A, 101.0, 1.0, VENUE_B, 100.0)
    assert np.isclose(out["fee"][0], float(expected))
    on_match_only = fees.taker_cost_legs(1.0, VENUE_A, 101.0, 1.0, VENUE_B, 100.0)
    assert out["fee"][0] > float(on_match_only)


def test_no_book_on_one_venue_is_the_only_rejection_left() -> None:
    buy = make_frame({5000: FLAT})
    sell = make_frame({1000: FLAT})
    out = sim(buy, sell, [1000], qty=1.0, latency_ms=100.0)
    assert not out["ok"][0]
    assert out["pnl_net"][0] == 0.0


def test_empty_fills_covers_every_field_simulate_returns() -> None:
    """Scatter-back would silently drop a field added to one and not the other."""
    frame = make_frame({1000: FLAT})
    out = sim(frame, frame, [1000], qty=1.0, latency_ms=0.0)
    assert set(out) == set(_FILL_FIELDS) | {"ok", "fill_ts"}
    assert set(_empty_fills(3)) == set(out)


def test_fill_ts_is_the_later_of_the_two_legs() -> None:
    buy = make_frame({1000: FLAT, 1050: FLAT})
    sell = make_frame({1000: FLAT, 1090: FLAT})
    out = sim(buy, sell, [1000], qty=1.0, latency_ms=100.0)
    assert out["fill_ts"][0] == 1090


# ── Trade / BacktestResult bookkeeping ───────────────────────────────────────

def a_trade(**kw) -> Trade:
    base = dict(timestamp_ms=1000, action="BUY_A_SELL_B", buy_venue=VENUE_A,
                sell_venue=VENUE_B, buy_price=100.0, sell_price=101.0, qty=1.0,
                fee=0.0, pnl_net=1.0, obi_delta=0.5, p_execute=0.9)
    base.update(kw)
    return Trade(**base)


def test_shortfall_is_requested_minus_matched() -> None:
    t = a_trade(qty=3.0, matched_qty=1.25)
    assert t.shortfall_qty == 1.75
    assert a_trade(qty=1.0, matched_qty=1.0).shortfall_qty == 0.0


def test_equity_curve_is_ordered_by_fill_time_not_signal_time() -> None:
    """Two signals can fill out of order once each leg has its own latency."""
    early_signal_late_fill = a_trade(timestamp_ms=1000, fill_ts_ms=9000, pnl_net=-5.0)
    late_signal_early_fill = a_trade(timestamp_ms=8000, fill_ts_ms=8100, pnl_net=+2.0)
    curve = _compute_equity_curve([early_signal_late_fill, late_signal_early_fill])
    assert list(curve.index) == [8100, 9000]
    # +2 lands first, so the running total dips only at the end.
    assert curve.iloc[0] > curve.iloc[1]


def test_a_trade_with_no_fill_ts_falls_back_to_the_signal_ts() -> None:
    curve = _compute_equity_curve([a_trade(timestamp_ms=4242, fill_ts_ms=0)])
    assert list(curve.index) == [4242]


# ── Backtester wiring ────────────────────────────────────────────────────────

def test_friction_argument_accepts_a_name_a_model_or_none() -> None:
    assert Backtester(friction="colocated").friction.name == "colocated"
    custom = FrictionModel(name="mine", latency_ms=7.0, jitter_log_sigma=0.0,
                           legging_cost_bps=1.0)
    assert Backtester(friction=custom).friction is custom
    assert Backtester().friction.name == get_preset("stress").name


def test_simulator_shares_the_backtesters_friction_model() -> None:
    bt = Backtester(friction="retail")
    assert bt._simulator.friction is bt.friction


def test_stochastic_latency_is_fixed_without_jitter_and_spread_with_it() -> None:
    fixed = Backtester(friction="stress")._simulator
    assert {fixed.get_stochastic_latency() for _ in range(20)} == {100.0}
    jittered = Backtester(friction="retail")._simulator
    assert len({jittered.get_stochastic_latency() for _ in range(20)}) > 1


