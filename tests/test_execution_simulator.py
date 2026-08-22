"""Tests for the execution simulator: book walking, fees, latency, adverse fills.

What gates and what merely costs (read this before changing a fixture)
---------------------------------------------------------------------
``simulate_cross_venue_fill`` charges three costs against a raw spread, and only
the first of them can stop a trade:

* **The entry gate** — at signal time, on the touch prices, the spread must beat
  the ``active()`` round-trip fee rate by ``MIN_PROFIT_NET_BPS`` (0.5 bps). On
  the default institutional preset that is 3.0 bps of fees, so the gate opens at
  a raw spread of 3.5 bps. This is the *only* rejection an otherwise well-formed
  book can produce.
* **Latency** — the order arrives ``LATENCY_MEAN_MS`` (30 ms) later, and the book
  it arrives at has moved against it by ``latency_ms / 1000 * 0.01`` on each
  side: ~6 bps of squeeze at 30 ms. It is charged to the fill, never gated.
* **The book walk** — each leg pays the VWAP of the levels it consumes.

The gate is a forecast, and these tests pin both of its failure modes: a spread
that clears 3.5 bps and still loses money (``test_books_the_loss_when_the_spread
_collapses``, at 3.6 bps), and one that clears it and wins (10 bps). Three
rejections used to sit *after* the friction — on a collapsed spread, on a spread
below the minimum, and on a negative net PnL — and the tests below were written
against them, asserting ``rejected`` where they now assert a booked loss. Those
rejections were the tautology: they recomputed the edge once the costs were known
and dropped whatever came out badly, so the ledger kept only winners.

Two consequences worth knowing before reading an assertion:

* **No fill prices at the touch.** ``buy_fill.vwap`` is the *drifted* touch, not
  the signal-time one. An earlier version asserted ``== 50000.0`` exactly, which
  was only true because the drift was computed and then thrown away — latency
  gated trades but never cost one anything.
* **``slippage_bps`` is drift-invariant.** It is measured against the drifted
  touch, so it stays a pure book-walk cost; a parallel shift scales the VWAP and
  the touch by the same factor and cancels. ``test_slippage_is_drift_invariant``
  pins that exactly, which is what lets the two costs be read separately.

Latency is stochastic (``random.gauss``), so every test that needs a definite
outcome runs inside ``fixed_latency()``, which pins the jitter to zero. Without
it these tests pass or fail on the RNG.
"""

import contextlib
from types import SimpleNamespace

from src import execution_simulator as ex
from src import fees
from src.execution_simulator import walk_book, simulate_cross_venue_fill, FillResult

def pl(price: float, vol: float):
    return SimpleNamespace(price=price, volume=vol)


# The fractional move each side of the book makes over the default 30 ms window:
# 30 / 1000 * 0.01 = 3 bps per side, 6 bps round trip. Written out rather than
# imported or recomputed from ``simulate_latency`` -- a test that derives its
# expectation from the code under test asserts only that the code is consistent
# with itself, which is true of any drift, including none.
DRIFT_30MS = 0.0003


@contextlib.contextmanager
def fixed_latency(mean_ms: float | None = None):
    """Make latency exactly ``mean_ms`` with no jitter, then restore.

    ``simulate_latency`` draws from ``random.gauss(LATENCY_MEAN_MS,
    LATENCY_STD_MS)`` unseeded. Setting the standard deviation to zero is
    preferable to seeding: it removes the RNG from the assertion entirely rather
    than depending on a particular Mersenne Twister sequence, and it states the
    latency the test assumes in the test itself.
    """
    old_mean, old_std = ex.LATENCY_MEAN_MS, ex.LATENCY_STD_MS
    if mean_ms is not None:
        ex.LATENCY_MEAN_MS = mean_ms
    ex.LATENCY_STD_MS = 0.0
    try:
        yield
    finally:
        ex.LATENCY_MEAN_MS, ex.LATENCY_STD_MS = old_mean, old_std


def test_walk_book_single_level():
    levels = [pl(50000.0, 2.0)]
    r = walk_book(levels, "buy", 1.0)
    assert abs(r.vwap - 50000.0) < 1e-6
    assert abs(r.filled_qty - 1.0) < 1e-6
    assert r.levels_consumed == 1
    assert r.slippage_bps == 0.0
    assert not r.partial


def test_walk_book_multi_level():
    levels = [pl(50000.0, 1.0), pl(50010.0, 1.0), pl(50020.0, 1.0)]
    r = walk_book(levels, "buy", 2.5)
    assert abs(r.vwap - 50008.0) < 1e-6  # (1*50000 + 1*50010 + 0.5*50020) / 2.5
    assert abs(r.filled_qty - 2.5) < 1e-6
    assert r.levels_consumed == 3
    assert not r.partial  # total volume 3.0 > 2.5


def test_walk_book_exact_match():
    levels = [pl(50000.0, 1.0), pl(50010.0, 2.0)]
    r = walk_book(levels, "sell", 3.0)
    expected_vwap = (1 * 50000 + 2 * 50010) / 3.0
    assert abs(r.vwap - expected_vwap) < 1e-6
    assert abs(r.filled_qty - 3.0) < 1e-6
    assert not r.partial


def test_walk_book_zero_qty():
    r = walk_book([pl(50000.0, 1.0)], "buy", 0.0)
    assert abs(r.filled_qty) < 1e-6


def test_walk_book_empty():
    r = walk_book([], "buy", 1.0)
    assert abs(r.filled_qty) < 1e-6


def test_walk_book_slippage():
    levels = [pl(50000.0, 1.0), pl(50020.0, 1.0)]
    r = walk_book(levels, "buy", 2.0)
    vwap = (50000 + 50020) / 2.0
    assert abs(r.vwap - vwap) < 1e-6
    assert r.slippage_bps > 0


def test_simulate_cross_venue_action_0():
    """action=0 buys venue A's ask and sells venue B's bid."""
    bids_a = [pl(49990.0, 5.0)]
    asks_a = [pl(50000.0, 3.0)]          # buy here
    bids_b = [pl(50150.0, 2.0)]          # sell here -- 30 bps above the ask
    asks_b = [pl(50160.0, 5.0)]
    with fixed_latency():
        r = simulate_cross_venue_fill(
            bids_a, asks_a, bids_b, asks_b, action=0, qty=2.0
        )
    assert not r.rejected, f"unexpected rejection: {r.reject_reason}"
    assert r.buy_fill is not None
    assert r.sell_fill is not None
    # Single-level books, so there is no walk: the VWAP is the touch price the
    # order *arrived* at, which is the signal-time touch moved by DRIFT_30MS.
    # Asserting 50000.0 here would pass only if the drift were discarded, which
    # is how this function behaved until latency was made to cost something.
    assert abs(r.buy_fill.vwap - 50000.0 * (1.0 + DRIFT_30MS)) < 1e-9
    assert abs(r.sell_fill.vwap - 50150.0 * (1.0 - DRIFT_30MS)) < 1e-9
    assert r.buy_fill.vwap > 50000.0, "buying got dearer during the window"
    assert r.sell_fill.vwap < 50150.0, "selling got cheaper during the window"
    assert r.buy_fill.slippage_bps == 0.0, "one level cannot slip"
    assert r.sell_fill.slippage_bps == 0.0
    # 30 bps of raw spread survives 6 bps of squeeze and 3 bps of fees.
    # Both must hold: gross_pnl > 0 alone would still pass if fees were dropped.
    assert r.gross_pnl > 0
    assert r.net_pnl > 0
    assert not r.adverse
    assert r.reject_reason == "", "a booked winner carries no label"
    assert r.total_fees > 0, "institutional preset is 3 bps, not free"
    assert r.net_pnl == r.gross_pnl - r.total_fees
    assert r.latency_ms == ex.LATENCY_MEAN_MS


def test_simulate_cross_venue_action_1():
    """action=1 is the mirror image: buy venue B's ask, sell venue A's bid."""
    bids_a = [pl(50150.0, 3.0)]          # sell here
    asks_a = [pl(50160.0, 5.0)]
    bids_b = [pl(49990.0, 5.0)]
    asks_b = [pl(50000.0, 2.0)]          # buy here
    with fixed_latency():
        r = simulate_cross_venue_fill(
            bids_a, asks_a, bids_b, asks_b, action=1, qty=1.0
        )
    assert not r.rejected, f"unexpected rejection: {r.reject_reason}"
    assert r.buy_fill is not None
    assert r.sell_fill is not None
    assert abs(r.buy_fill.vwap - 50000.0 * (1.0 + DRIFT_30MS)) < 1e-9
    assert abs(r.sell_fill.vwap - 50150.0 * (1.0 - DRIFT_30MS)) < 1e-9
    assert r.net_pnl > 0
    assert not r.adverse


def test_simulate_multilevel_slippage():
    """Walking past level 1 on both sides must register as slippage."""
    bids_a = [pl(50150.0, 1.0), pl(50140.0, 1.0)]     # sell side, 2 levels
    asks_a = [pl(50200.0, 5.0)]
    bids_b = [pl(49900.0, 5.0)]
    asks_b = [pl(50000.0, 1.0), pl(50010.0, 1.0)]     # buy side, 2 levels
    with fixed_latency():
        r = simulate_cross_venue_fill(
            bids_a, asks_a, bids_b, asks_b, action=1, qty=2.0
        )
    assert not r.rejected, f"unexpected rejection: {r.reject_reason}"
    assert r.buy_fill is not None and r.sell_fill is not None
    assert r.buy_fill.levels_consumed > 1
    assert r.sell_fill.levels_consumed > 1
    assert r.total_slippage_bps > 0
    # VWAP must be worse than the touch on both sides -- that is what slippage
    # means here, and it is the thing a "walk the book" implementation gets
    # wrong by silently filling everything at level 1. Compared against the
    # *drifted* touch on purpose: against the signal-time touch, the latency
    # move alone would satisfy both of these even if the walk were skipped, so
    # the pre-drift comparison this test used to make no longer isolates it.
    assert r.buy_fill.vwap > 50000.0 * (1.0 + DRIFT_30MS)
    assert r.sell_fill.vwap < 50150.0 * (1.0 - DRIFT_30MS)


def test_slippage_is_drift_invariant():
    """``slippage_bps`` must measure the walk only, with the latency taken out.

    The drift is a parallel shift, so it scales the VWAP and the touch by the
    same factor and cancels out of ``|vwap - touch| / touch`` exactly. That is
    the property that lets the two costs be read separately: the VWAPs carry the
    latency move and ``slippage_bps`` carries the walk. Measuring slippage
    against the pre-latency touch instead would fold the drift in and count it
    twice, and the discrepancy would grow with latency -- so this compares 1 ms
    against 200 ms, where the drift differs by a factor of 200.
    """
    def outcome(latency_ms: float):
        with fixed_latency(latency_ms):
            return simulate_cross_venue_fill(
                [pl(50150.0, 1.0), pl(50140.0, 1.0)], [pl(50200.0, 5.0)],
                [pl(49900.0, 5.0)], [pl(50000.0, 1.0), pl(50010.0, 1.0)],
                action=1, qty=2.0,
            )

    fast, slow = outcome(1.0), outcome(200.0)
    assert not fast.rejected and not slow.rejected
    assert fast.buy_fill.slippage_bps > 0 and fast.sell_fill.slippage_bps > 0
    assert abs(fast.buy_fill.slippage_bps - slow.buy_fill.slippage_bps) < 1e-9
    assert abs(fast.sell_fill.slippage_bps - slow.sell_fill.slippage_bps) < 1e-9
    # And the fills themselves are not invariant, or the drift is inert.
    assert slow.buy_fill.vwap > fast.buy_fill.vwap
    assert slow.sell_fill.vwap < fast.sell_fill.vwap


def test_books_the_loss_when_the_spread_collapses():
    """A spread that clears the gate and then collapses must fill, at a loss.

    This is the original 4 bps action=0 fixture, and it used to assert
    ``r.rejected``. 4 bps clears the entry gate -- 3 bps of institutional fees
    leaves 1 bps, above the 0.5 bps floor -- and then loses to 6 bps of latency
    squeeze. The old code rejected it here and booked ``pnl_net = 0.0``, which is
    how a ledger ends up holding only the trades friction happened to spare.

    The order went out on the strength of a forecast. The forecast was wrong.
    That is a loss, not a cancellation.
    """
    bids_a = [pl(50010.0, 5.0)]
    asks_a = [pl(50000.0, 3.0)]
    bids_b = [pl(50020.0, 2.0)]          # 4 bps above the ask
    asks_b = [pl(50005.0, 5.0)]
    with fixed_latency():
        r = simulate_cross_venue_fill(
            bids_a, asks_a, bids_b, asks_b, action=0, qty=2.0
        )
    assert not r.rejected, f"the trade must happen: {r.reject_reason}"
    assert r.adverse, "and it must be flagged as an adverse fill"
    assert r.buy_fill is not None and r.sell_fill is not None
    assert r.net_pnl < 0.0, "the loss has to reach the ledger, not round to zero"
    assert r.gross_pnl < 0.0, "6 bps of squeeze against a 4 bps spread"
    assert r.total_fees > 0.0, "and the fees are paid on top of it"
    assert r.net_pnl == r.gross_pnl - r.total_fees
    # The move inside the window turned the spread over on its own, which is the
    # narrower of the two labels. Same string the C++ path writes.
    assert r.reject_reason == "adverse_fill_spread_collapsed"


def test_entry_gate_still_rejects_below_the_minimum():
    """The one rejection left for a well-formed book: the signal-time gate.

    3 bps of raw spread cannot beat 3 bps of fees, let alone by the 0.5 bps
    floor, so no order is ever sent. Nothing filled, so nothing is booked -- and
    that is categorically different from the adverse fill above, which is why
    ``adverse`` must be false and the fills absent rather than empty.
    """
    bids_a = [pl(50010.0, 5.0)]
    asks_a = [pl(50000.0, 3.0)]
    bids_b = [pl(50015.0, 2.0)]          # 3 bps above the ask
    asks_b = [pl(50005.0, 5.0)]
    with fixed_latency():
        r = simulate_cross_venue_fill(
            bids_a, asks_a, bids_b, asks_b, action=0, qty=2.0
        )
    assert r.rejected
    assert not r.adverse
    assert r.reject_reason.startswith("net_spread"), r.reject_reason
    assert r.buy_fill is None and r.sell_fill is None
    assert r.net_pnl == 0.0
    assert r.total_fees == 0.0, "an order that never went out pays no fees"
    assert r.latency_ms == 0.0, "and waits for nothing"


def test_entry_gate_boundary():
    """Pin where the gate opens, and that clearing it is not a promise.

    The gate is fees plus ``MIN_PROFIT_NET_BPS``, and those live in two different
    places; nothing else in the suite would notice if either moved. On the
    institutional preset it opens at 3.5 bps of raw spread.

    3.4 and 3.6 rather than 3.5 on both sides: the round-trip rate is
    ``0.0001 + 0.0002``, which in binary is a hair above 0.0003, so a raw 3.5 bps
    lands at 0.49999999999999956 net and rejects. Pinning that would be pinning
    the float, not the model.

    An earlier version of this test asserted the break-even spread was ~9.51 bps,
    because latency and the book walk used to gate as well. They do not any more,
    so the boundary moved down to the gate's own arithmetic -- and the two
    assertions at the end are the point: 3.6 bps opens the gate and loses money,
    10 bps opens it and makes money. The gate is a forecast either way.
    """
    def outcome(gross_bps: float):
        buy = 50000.0
        sell = buy * (1.0 + gross_bps / 10000.0)
        with fixed_latency():
            return simulate_cross_venue_fill(
                [pl(sell, 5.0)], [pl(buy + 1000.0, 5.0)],
                [pl(sell - 1000.0, 5.0)], [pl(buy, 5.0)],
                action=1, qty=1.0,
            )

    assert outcome(2.0).rejected, "2 bps cannot cover 3 bps of fees"
    assert outcome(3.4).rejected, "3.4 bps leaves 0.4, below the 0.5 floor"

    thin = outcome(3.6)
    assert not thin.rejected, f"3.6 bps leaves 0.6, above it: {thin.reject_reason}"
    assert thin.adverse, "but 0.6 bps of forecast edge cannot pay 6 bps of drift"
    assert thin.net_pnl < 0.0

    wide = outcome(10.0)
    assert not wide.rejected, f"10 bps clears drift and fees: {wide.reject_reason}"
    assert not wide.adverse
    assert wide.net_pnl > 0.0


def test_both_fee_paths_read_the_active_preset():
    """The entry gate and the booked PnL must charge fees from the same preset.

    Fees enter the simulator twice, by different routes: the gate subtracts
    ``active().round_trip_rate(buy, sell)`` from the spread in bps, and
    ``walk_book`` charges ``active().taker_rate(venue)`` against each leg's
    notional. Nothing in the code forces those to agree, and a hardcoded rate in
    either place would be invisible -- the simulator would still produce
    plausible fills, just gated on one fee schedule and booked on another.

    Two spreads are needed now that only the gate can reject. 2 bps separates the
    presets at the gate: it clears zero's and fails institutional's. 20 bps clears
    every preset's gate, so all three reach ``walk_book`` and the fees they book
    can be compared directly. The 7 bps single case this test used to rely on
    stopped separating anything once the post-fill rejections came out.
    """
    def outcome(preset: str, gross_bps: float):
        previous = fees.active()
        try:
            fees.set_active(preset)
            buy = 50000.0
            sell = buy * (1.0 + gross_bps / 10000.0)
            with fixed_latency():
                return simulate_cross_venue_fill(
                    [pl(sell, 5.0)], [pl(buy + 1000.0, 5.0)],
                    [pl(sell - 1000.0, 5.0)], [pl(buy, 5.0)],
                    action=1, qty=1.0,
                )
        finally:
            fees.set_active(previous)

    # The gate read the preset: 2 bps survives with no fees, dies with 3 bps.
    assert outcome("institutional", 2.0).rejected, (
        "3 bps of fees should close the gate on a 2 bps spread"
    )
    free_thin = outcome("zero", 2.0)
    assert not free_thin.rejected, f"zero-fee gate should open: {free_thin.reject_reason}"

    # Past the gate, the booking read the same preset rather than a rate of its
    # own. Note the zero preset still loses here: it removes the fees, not the
    # 6 bps of latency drift, so a 2 bps spread is underwater either way.
    assert free_thin.total_fees == 0.0, (
        f"zero preset still booked {free_thin.total_fees} in fees -- walk_book is "
        f"not reading src.fees"
    )
    assert free_thin.net_pnl == free_thin.gross_pnl
    assert free_thin.adverse and free_thin.net_pnl < 0.0

    # And on a spread wide enough for every gate, the booked fees rank by preset.
    free = outcome("zero", 20.0)
    charged = outcome("institutional", 20.0)
    charging = outcome("retail", 20.0)
    for r, name in ((free, "zero"), (charged, "institutional"), (charging, "retail")):
        assert not r.rejected, f"20 bps should clear the {name} gate: {r.reject_reason}"
    assert charging.total_fees > charged.total_fees > free.total_fees == 0.0
    assert charging.net_pnl < charged.net_pnl < free.net_pnl
    assert charging.net_pnl < charging.gross_pnl
    # Identical books, so the gross is a fee-independent constant. If it moves
    # with the preset, a fee is being charged inside the VWAP instead of beside it.
    assert abs(charging.gross_pnl - free.gross_pnl) < 1e-9
    assert abs(charged.gross_pnl - free.gross_pnl) < 1e-9


def test_walk_book_fees_scale_with_venue_rate():
    """A more expensive venue must cost more on an identical fill."""
    previous = fees.active()
    try:
        fees.set_active("retail")          # binance 4 bps, kraken 10 bps
        levels = [pl(50000.0, 2.0)]
        cheap = walk_book(levels, "buy", 1.0, "binance")
        dear = walk_book(levels, "buy", 1.0, "kraken")
        assert dear.fees_paid > cheap.fees_paid > 0
        # Rates are 0.0004 and 0.0010, so the ratio is exactly 2.5.
        assert abs(dear.fees_paid / cheap.fees_paid - 2.5) < 1e-9
        # Fee is charged on notional, not on quantity.
        assert abs(cheap.fees_paid - 50000.0 * 0.0004) < 1e-9
    finally:
        fees.set_active(previous)


def test_latency_squeeze_scales_with_latency():
    """Longer latency must cost strictly more, and enough of it must lose money.

    This is the test the whole change turns on, and it is worth being precise
    about why. The squeeze is proportional to latency, so the booked PnL must be
    monotonically decreasing in it. The version of this test that asserted
    ``slow.rejected`` could not distinguish a working latency model from an inert
    one: the drift was computed, used to decide whether to reject, and then
    *discarded* -- the fill walked the signal-time book. All three outcomes below
    booked an identical PnL, and the test passed on the rejection alone.

    So the assertion is the ordering, not the rejection. The gate sees only the
    signal-time touch, which is the same 30 bps in all three cases, so latency
    cannot reject anything here -- 200 ms books the loss instead.
    """
    def outcome(latency_ms: float):
        with fixed_latency(latency_ms):
            return simulate_cross_venue_fill(
                [pl(50150.0, 5.0)], [pl(51000.0, 5.0)],
                [pl(49000.0, 5.0)], [pl(50000.0, 5.0)],
                action=1, qty=1.0,
            )

    fast = outcome(1.0)
    normal = outcome(30.0)
    slow = outcome(200.0)

    for r, name, ms in ((fast, "1 ms", 1.0), (normal, "30 ms", 30.0),
                        (slow, "200 ms", 200.0)):
        assert not r.rejected, f"{name} must not gate on latency: {r.reject_reason}"
        assert r.latency_ms == ms, f"{name}: fixed_latency did not pin the draw"

    # Strictly decreasing. Equality here is the inert-drift bug returning.
    assert fast.net_pnl > normal.net_pnl > slow.net_pnl

    assert fast.net_pnl > 0 and normal.net_pnl > 0
    assert not fast.adverse and not normal.adverse

    # 200 ms is 20 bps a side: 40 bps of squeeze against a 30 bps spread.
    assert slow.net_pnl < 0, "200 ms of drift must eat a 30 bps spread"
    assert slow.gross_pnl < 0, "and eat it before fees, not because of them"
    assert slow.adverse
    assert slow.reject_reason == "adverse_fill_spread_collapsed"



def test_walk_book_partial_fill():
    levels = [pl(50000.0, 1.0)]
    r = walk_book(levels, "buy", 2.0)
    assert r.partial
    assert abs(r.filled_qty - 1.0) < 1e-6


def test_drift_is_a_parallel_shift_and_leaves_the_caller_s_book_alone():
    """Two calls on the *same* book objects must give the same answer.

    ``drift_book`` returns shifted copies. Shifting in place would be invisible in
    any single-call test and wrong in the one way that matters: the caller's books
    are market data, both actions are evaluated against the same snapshot, and the
    backtester reuses them. An in-place shift would move the book once per
    evaluation, so the second action would price against an already-drifted book
    and every later signal would compound the error.

    The parallel-shift property is checked here too: every level moves by the same
    factor, so the *ratio* between the touch and a deeper level is preserved.
    Shifting only the touch would leave the walk's cost depending on latency,
    which is what ``test_slippage_is_drift_invariant`` measures from outside.
    """
    asks = [pl(50000.0, 1.0), pl(50010.0, 1.0)]
    bids = [pl(50150.0, 1.0), pl(50140.0, 1.0)]
    prices_before = [l.price for l in asks + bids]

    def once():
        with fixed_latency():
            return simulate_cross_venue_fill(
                bids, [pl(50200.0, 5.0)], [pl(49900.0, 5.0)], asks,
                action=1, qty=2.0,
            )

    first, second = once(), once()
    assert [l.price for l in asks + bids] == prices_before, (
        "simulate_cross_venue_fill mutated the books it was handed"
    )
    assert first.buy_fill.vwap == second.buy_fill.vwap
    assert first.sell_fill.vwap == second.sell_fill.vwap
    assert first.net_pnl == second.net_pnl

    # Parallel: the level-to-level ratio survives the shift exactly.
    shifted = ex.drift_book(asks, 1.0 + DRIFT_30MS)
    assert abs(shifted[1].price / shifted[0].price
               - asks[1].price / asks[0].price) < 1e-15
    assert [l.volume for l in shifted] == [l.volume for l in asks], (
        "the model has one parameter -- elapsed time -- and says nothing about depth"
    )


def test_the_two_adverse_labels_name_different_causes():
    """A loss to the book walk is ``adverse_fill``, not ``..._spread_collapsed``.

    Both are booked, so the distinction is only in the label -- which is exactly
    why it needs a test: nothing downstream would break if the two were merged,
    and the adverse-fill counts in TEST_REPORT would silently stop telling latency
    apart from depth.

    Here the spread survives the window comfortably (30 bps raw, 6 bps of drift),
    and the loss comes entirely from thin depth at the touch: 0.1 BTC at 50000
    and the rest 500 higher.
    """
    with fixed_latency():
        r = simulate_cross_venue_fill(
            [pl(50150.0, 5.0)], [pl(51000.0, 5.0)],
            [pl(49000.0, 5.0)], [pl(50000.0, 0.1), pl(50500.0, 5.0)],
            action=1, qty=1.0,
        )
    assert not r.rejected, f"the trade must happen: {r.reject_reason}"
    assert r.adverse and r.net_pnl < 0.0
    assert r.reject_reason == "adverse_fill", (
        "the spread never turned over -- 30 bps raw against 6 bps of drift"
    )
    assert r.buy_fill.levels_consumed == 2
    assert r.buy_fill.slippage_bps > 0.0, "the walk is what cost the money"
    assert r.sell_fill.slippage_bps == 0.0, "the sell side never left the touch"


def test_a_book_that_fills_nothing_is_not_an_adverse_fill():
    """The last no-fill guard: a malformed book, not a trade that went badly.

    ``no_liquidity_at_fill`` fires when both books have volume to size against but
    neither can supply it at a usable price -- here a zero-volume touch above a
    negative-priced level, which ``walk_book`` skips. Nonsense market data, and
    the point of the test is the classification: nothing traded, so ``net_pnl`` is
    zero and ``adverse`` is false. If a future edit ever routes a real loss
    through this branch, the loss disappears from the ledger, which is the defect
    this whole change removed.

    It was previously labelled ``partial_fill``, which describes the opposite
    condition -- a book that filled *some* of the order. That case is booked.

    One honest asymmetry, asserted here so it is on the record rather than
    discovered later: the buy leg *did* fill, and this path books its fees
    against a zero PnL. A one-legged fill leaves a real position and the model
    treats it as no trade.

    An earlier version of this docstring claimed the C++ path "makes the same
    simplification, so parity holds". It does not, and the claim is withdrawn.
    ``price_resolved`` takes its ``no_liquidity_at_fill`` early return only when
    ``hedged`` *and* ``residual`` are both zero, so one filled leg falls through
    and is booked: gross 0 on nothing hedged, fees on the leg that filled, the
    residual charged at ``legging_cost``, ``filled = true``, ``adverse = true``.
    The Python guard is an ``or``, so the same shape is declined here.

    The divergence is unreachable rather than harmless, and it is worth being
    exact about why, because "unreachable" is the sort of thing that stops being
    true. Reaching it needs positive volume at a non-positive price -- an
    all-zero-volume book takes the earlier ``no_liquidity`` branch -- which the
    Tardis feeds do not contain and which ``crossflux::PriceLevel``'s validating
    two-argument constructor will not construct, so no parity harness can even
    submit it (see ``cpp_engine/tests/parity_walk_book.cpp:54``). Separately, this
    synchronous path sizes both legs to the same ``used_qty`` and walks them
    together, so it cannot leg *genuinely*; real legging comes from the async
    path's independent per-leg latency, and that path charges it. TEST_REPORT 5
    carries both halves as a known limitation.
    """
    with fixed_latency():
        r = simulate_cross_venue_fill(
            [pl(50150.0, 0.0), pl(-5.0, 5.0)], [pl(51000.0, 5.0)],
            [pl(49000.0, 5.0)], [pl(50000.0, 5.0)],
            action=1, qty=1.0,
        )
    assert r.rejected
    assert r.reject_reason == "no_liquidity_at_fill"
    assert not r.adverse
    assert r.net_pnl == 0.0 and r.gross_pnl == 0.0
    assert r.sell_fill is not None and r.sell_fill.filled_qty == 0.0
    assert r.buy_fill.filled_qty > 0.0, "the asymmetry described above"


def test_hedged_quantity_is_the_smaller_leg():
    """Gross PnL must be booked on the size *both* venues supplied.

    Whichever leg fills less is the hedged quantity; the remainder is an unhedged
    position, not a profit. Tested in both orientations on purpose. A fixture
    where only the sell leg is short cannot tell ``min(buy, sell)`` apart from
    ``sell_filled_qty`` -- an earlier version of this test had exactly that hole,
    and a mutation replacing the min with the sell leg passed all 20 tests.
    """
    def outcome(short_leg: str):
        thin = [pl(50000.0, 0.4), pl(0.0, 5.0)]     # fills 0.4: junk level below
        deep = [pl(50000.0, 5.0)]
        sell_thin = [pl(50150.0, 0.4), pl(0.0, 5.0)]
        sell_deep = [pl(50150.0, 5.0)]
        with fixed_latency():
            return simulate_cross_venue_fill(
                sell_thin if short_leg == "sell" else sell_deep,
                [pl(51000.0, 5.0)],
                [pl(49000.0, 5.0)],
                thin if short_leg == "buy" else deep,
                action=1, qty=1.0,
            )

    for short_leg, other in (("buy", "sell"), ("sell", "buy")):
        r = outcome(short_leg)
        assert not r.rejected, f"{short_leg}: unexpected rejection {r.reject_reason}"
        legs = {"buy": r.buy_fill, "sell": r.sell_fill}
        assert abs(legs[short_leg].filled_qty - 0.4) < 1e-12, short_leg
        assert abs(legs[other].filled_qty - 1.0) < 1e-12, short_leg
        expect = (r.sell_fill.vwap - r.buy_fill.vwap) * 0.4
        assert abs(r.gross_pnl - expect) < 1e-9, (
            f"{short_leg} leg is short: gross must be booked on 0.4, "
            f"got {r.gross_pnl} vs {expect}"
        )
        assert r.net_pnl == r.gross_pnl - r.total_fees
        # Fees follow what each leg actually traded, so the unhedged 0.6 is not
        # free -- the deeper leg pays on its full notional. Correct asymmetry:
        # the coin changed hands whether or not it ended up hedged.
        assert legs[other].fees_paid > 0.0 and legs[short_leg].fees_paid > 0.0
