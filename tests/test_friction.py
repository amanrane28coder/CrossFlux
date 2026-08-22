"""Tests for src/friction.py: the VWAP walk, the latency model, and the
prevailing-book lookup that replaces the old look-ahead fill.

What these tests are guarding
-----------------------------
``walk_book`` exists twice on purpose -- once as a readable loop that mirrors the
C++ ``crossflux::walk_book``, once vectorised over whole DataFrames as
``walk_book_array``. Two implementations of the same arithmetic is exactly the
setup that let three divergent fee models coexist in this repository, so the
central test here is that they agree **bit-for-bit**, not approximately.

That is achievable and was not free. Written the obvious way, with
``remaining -= take``, the loop disagreed with the cumulative-sum form by
~2.8e-14: repeated subtraction rounds at every level, ``np.cumsum`` does not.
The loop now accumulates a running total instead, which matches. See
``test_scalar_and_vector_agree_bit_for_bit``, and
``test_exactness_limit_is_where_it_is_documented`` which pins the *reason* the
guarantee holds so that widening the book past 8 levels fails loudly rather than
silently degrading.
"""

import math

import numpy as np
import pytest

from src.friction import (
    EPS_QTY,
    PRESETS,
    BookWalk,
    FrictionModel,
    active,
    comparison_table,
    get_preset,
    prevailing_row,
    set_active,
    walk_book,
    walk_book_array,
)

# A book small enough to compute by hand, so the expected values below are
# derived independently of the implementation rather than copied out of it.
PX = [100.0, 101.0, 102.0]
AM = [1.0, 2.0, 5.0]


# ─────────────────────────────────────────────────────────────────────────────
# The walk itself
# ─────────────────────────────────────────────────────────────────────────────

def test_fill_inside_the_touch_pays_the_touch():
    w = walk_book(PX, AM, 0.5)
    assert w.vwap == 100.0
    assert w.filled_qty == 0.5
    assert w.levels_consumed == 1
    assert not w.partial


def test_walk_spans_levels_and_weights_by_depth():
    # 1.0 @ 100 + 1.5 @ 101 = 251.5 over 2.5 => 100.6
    w = walk_book(PX, AM, 2.5)
    assert w.vwap == pytest.approx((1.0 * 100.0 + 1.5 * 101.0) / 2.5, abs=1e-15)
    assert w.filled_qty == 2.5
    assert w.levels_consumed == 2
    assert w.notional == pytest.approx(251.5, abs=1e-12)


def test_size_costs_more_than_the_touch():
    """The whole point of walking: a bigger order fills worse, monotonically."""
    vwaps = [walk_book(PX, AM, q).vwap for q in (0.5, 1.5, 3.0, 6.0)]
    assert vwaps == sorted(vwaps)
    assert vwaps[0] == 100.0
    assert vwaps[-1] > 100.0


def test_order_larger_than_the_book_fills_partially_and_says_so():
    w = walk_book(PX, AM, 50.0)
    assert w.filled_qty == 8.0             # 1 + 2 + 5, all the book holds
    assert w.shortfall == pytest.approx(42.0, abs=1e-12)
    assert w.partial
    # The VWAP that *is* reported must describe only what filled.
    assert w.vwap == pytest.approx((100.0 + 2 * 101.0 + 5 * 102.0) / 8.0, abs=1e-13)


def test_zero_and_empty_requests_fill_nothing():
    for w in (walk_book(PX, AM, 0.0),
              walk_book(PX, AM, -1.0),
              walk_book([], [], 1.0)):
        assert w.filled_qty == 0.0
        assert w.vwap == 0.0
        assert w.notional == 0.0


def test_zero_padded_levels_are_not_depth():
    """Snapshot files pad absent levels with zeros; they must supply nothing.

    This is the same trap that made a C++ loop bound unfalsifiable -- see the
    t16 note in cpp_engine/tests/test_signals.cpp. A padded level that counted
    as depth would silently improve every large fill.
    """
    assert walk_book([0.0, 0.0], [0.0, 0.0], 1.0).filled_qty == 0.0
    w = walk_book([100.0, 0.0, 102.0], [1.0, 9.0, 3.0], 2.0)
    assert w.filled_qty == 2.0
    assert w.vwap == pytest.approx(101.0, abs=1e-13)   # 1@100 + 1@102, not 9@0


def test_walk_does_not_reorder_the_book():
    """A real order takes levels as the exchange presents them.

    Sorting would silently turn a crossed or malformed book into a better fill
    than the market offered.
    """
    w = walk_book([102.0, 100.0], [1.0, 1.0], 2.0)
    assert w.vwap == pytest.approx(101.0, abs=1e-13)
    assert w.notional == pytest.approx(202.0, abs=1e-12)


def test_bookwalk_slippage_is_unsigned():
    """Buy-above and sell-below are both costs and must not cancel in a mean."""
    buy = BookWalk(vwap=101.0, filled_qty=1.0, levels_consumed=2,
                   notional=101.0, requested_qty=1.0)
    sell = BookWalk(vwap=99.0, filled_qty=1.0, levels_consumed=2,
                    notional=99.0, requested_qty=1.0)
    assert buy.slippage_bps(100.0) == pytest.approx(100.0)
    assert sell.slippage_bps(100.0) == pytest.approx(100.0)
    assert BookWalk(0.0, 0.0, 0, 0.0, 1.0).slippage_bps(100.0) == 0.0
    assert buy.slippage_bps(0.0) == 0.0        # no touch price, no claim


# ─────────────────────────────────────────────────────────────────────────────
# The two implementations must not drift
# ─────────────────────────────────────────────────────────────────────────────

def _random_books(rows: int, levels: int, seed: int):
    rng = np.random.default_rng(seed)
    px = np.sort(rng.uniform(90.0, 110.0, size=(rows, levels)), axis=1)
    am = rng.gamma(2.0, 1.0, size=(rows, levels))
    am[rng.random(am.shape) < 0.25] = 0.0      # padded levels
    px[rng.random(px.shape) < 0.05] = 0.0      # padded prices
    return px, am


def test_scalar_and_vector_agree_bit_for_bit():
    """Asserted with ``==``, deliberately.

    A tolerance here would hide exactly the kind of drift this test exists to
    catch, and the FMA-contraction finding in signals.hpp showed that ~1 ULP
    gaps are a real signal about the code rather than unavoidable noise.
    """
    px, am = _random_books(2000, 5, seed=7)
    for qty in (0.001, 0.01, 0.5, 2.0, 7.0, 40.0):
        vec_vwap, vec_filled = walk_book_array(px, am, qty)
        for i in range(px.shape[0]):
            ref = walk_book(px[i], am[i], qty)
            assert vec_vwap[i] == ref.vwap, (
                f"row {i} qty {qty}: vector {vec_vwap[i]!r} != loop {ref.vwap!r}"
            )
            assert vec_filled[i] == ref.filled_qty


def test_exactness_limit_is_where_it_is_documented():
    """The bit-exact guarantee rests on numpy reducing a narrow axis sequentially.

    Past 8 levels numpy switches to pairwise summation and the two forms diverge
    by ~1 ULP. That is documented in walk_book_array; this test pins it, so that
    widening BOOK_DEPTH past 8 breaks a test with an explanation attached
    instead of quietly weakening the guarantee above.
    """
    px, am = _random_books(400, 13, seed=11)
    vec_vwap, _ = walk_book_array(px, am, 12.0)
    gaps = [abs(vec_vwap[i] - walk_book(px[i], am[i], 12.0).vwap)
            for i in range(px.shape[0])]
    assert max(gaps) > 0.0, (
        "13-level divergence no longer happens -- the documented limit in "
        "walk_book_array is now wrong and should be updated, not ignored."
    )
    # Still far too small to move any gate (thresholds here are ~1e-3).
    assert max(gaps) < 1e-11


def test_vector_accepts_per_row_quantities():
    px, am = _random_books(300, 5, seed=3)
    scalar_vwap, scalar_filled = walk_book_array(px, am, 2.0)
    array_vwap, array_filled = walk_book_array(px, am, np.full(px.shape[0], 2.0))
    assert np.array_equal(scalar_vwap, array_vwap)
    assert np.array_equal(scalar_filled, array_filled)

    mixed = np.where(np.arange(px.shape[0]) % 2 == 0, 0.5, 5.0)
    mixed_vwap, _ = walk_book_array(px, am, mixed)
    for i in (0, 1, 2, 3):
        assert mixed_vwap[i] == walk_book(px[i], am[i], mixed[i]).vwap


def test_vector_rejects_mismatched_input():
    px, am = _random_books(10, 5, seed=1)
    with pytest.raises(ValueError, match="same shape"):
        walk_book_array(px, am[:, :3], 1.0)
    with pytest.raises(ValueError, match="2-D"):
        walk_book_array(px[0], am[0], 1.0)
    with pytest.raises(ValueError, match="rows"):
        walk_book_array(px, am, np.ones(4))


# ─────────────────────────────────────────────────────────────────────────────
# Prevailing book: the fix for the look-ahead fill
# ─────────────────────────────────────────────────────────────────────────────

def test_prevailing_row_takes_the_last_quote_at_or_before():
    """This is the whole pending-order mechanism, and the direction matters.

    Plain ``searchsorted`` returns the *next* update at or after the target --
    a quote that had not printed when the order arrived. That is look-ahead, of
    the same kind as the timestamp/local_timestamp defect in src/ingestion.py.
    """
    idx = np.array([0, 100, 250, 900], dtype=np.int64)
    got = prevailing_row(idx, np.array([0, 99, 100, 101, 250, 899, 10_000]))
    assert list(got) == [0, 0, 1, 1, 2, 2, 3]

    # Contrast with the look-ahead form, so the difference is on the record.
    lookahead = np.searchsorted(idx, np.array([99, 101, 899]))
    assert list(lookahead) == [1, 2, 3]        # jumps to quotes not yet seen


def test_prevailing_row_picks_the_last_of_a_duplicated_timestamp():
    """Matches the groupby(level=0).last() used to align the two venues."""
    idx = np.array([0, 100, 100, 100, 250], dtype=np.int64)
    assert list(prevailing_row(idx, np.array([100, 150]))) == [3, 3]


def test_prevailing_row_signals_before_the_first_quote():
    idx = np.array([500, 600], dtype=np.int64)
    assert prevailing_row(idx, np.array([499]))[0] == -1


def test_zero_latency_reads_the_signals_own_book():
    """The 'zero' preset must reproduce the pre-friction behaviour exactly."""
    idx = np.array([0, 100, 250], dtype=np.int64)
    at = idx + get_preset("zero").latency_ms
    assert list(prevailing_row(idx, at)) == [0, 1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# Latency model
# ─────────────────────────────────────────────────────────────────────────────

def test_default_is_deterministic_and_needs_no_seed():
    """Three tests in this suite were previously flaky against random.gauss.

    A stochastic default also makes preset comparisons incommensurable: two runs
    would differ for reasons unrelated to the change under test.
    """
    fr = get_preset("stress")
    assert fr.deterministic
    assert fr.latency_ms == 100.0
    a = fr.sample_latency(1000)
    b = fr.sample_latency(1000)
    assert np.array_equal(a, b)
    assert np.all(a == 100.0)


def test_jitter_is_multiplicative_so_the_median_is_recoverable():
    """latency_ms must mean something in the output.

    The previous ``base + lognormal(mu, sigma)`` form reported a median of
    ``base + exp(mu)``, so the configured number appeared nowhere in the result.
    """
    fr = FrictionModel("j", latency_ms=100.0, jitter_log_sigma=0.4)
    draws = fr.sample_latency(400_000, np.random.default_rng(0))
    assert np.median(draws) == pytest.approx(100.0, rel=0.01)
    assert draws.min() > 0.0                       # cannot go negative
    assert np.percentile(draws, 99) == pytest.approx(fr.latency_quantile(0.99), rel=0.02)


def test_latency_quantile_matches_sampling():
    fr = get_preset("retail")
    draws = fr.sample_latency(400_000, np.random.default_rng(5))
    for p in (0.5, 0.9, 0.99):
        assert np.percentile(draws, p * 100) == pytest.approx(
            fr.latency_quantile(p), rel=0.03
        )


def test_latency_quantile_of_a_fixed_model_is_the_fixed_value():
    fr = get_preset("colocated")
    assert fr.latency_quantile(0.01) == 5.0
    assert fr.latency_quantile(0.99) == 5.0
    with pytest.raises(ValueError):
        fr.latency_quantile(1.0)


def test_erfinv_agrees_with_erf():
    from src.friction import _erfinv
    for y in (-0.999, -0.5, -1e-9, 0.0, 1e-9, 0.25, 0.9, 0.999999):
        assert math.erf(_erfinv(y)) == pytest.approx(y, abs=1e-14)
    with pytest.raises(ValueError):
        _erfinv(1.0)


def test_sample_latency_handles_empty_request():
    assert get_preset("stress").sample_latency(0).shape == (0,)


# ─────────────────────────────────────────────────────────────────────────────
# Legging cost
# ─────────────────────────────────────────────────────────────────────────────

def test_legging_cost_charges_the_unhedged_residual():
    fr = get_preset("stress")                       # 5.0 bps
    assert fr.legging_cost(0.0, 60_000.0) == 0.0
    assert fr.legging_cost(1.0, 60_000.0) == pytest.approx(30.0)
    # Direction of the mismatch is irrelevant: either way a position must go.
    assert fr.legging_cost(-1.0, 60_000.0) == pytest.approx(30.0)


def test_zero_legging_cost_recovers_the_old_costless_behaviour():
    """TEST_REPORT.md 4 records legging risk as having been modeled as costless.

    Keeping that reachable is what makes the charge auditable rather than an
    unexplained new number.
    """
    assert get_preset("zero").legging_cost(5.0, 60_000.0) == 0.0


def test_legging_cost_vectorises():
    fr = get_preset("stress")
    got = fr.legging_cost(np.array([0.0, 1.0, 2.0]), 60_000.0)
    assert np.allclose(got, [0.0, 30.0, 60.0])


# ─────────────────────────────────────────────────────────────────────────────
# Preset plumbing
# ─────────────────────────────────────────────────────────────────────────────

def test_every_preset_is_self_consistent():
    for name, fr in PRESETS.items():
        assert fr.name == name, f"{name!r} carries the wrong name {fr.name!r}"
        assert fr.latency_ms >= 0.0
        assert fr.legging_cost_bps >= 0.0
        assert fr.description.strip(), f"{name!r} has no description"
        assert fr.describe()


def test_nonsense_configurations_are_refused():
    with pytest.raises(ValueError, match="negative"):
        FrictionModel("bad", latency_ms=-1.0)
    with pytest.raises(ValueError, match="jitter"):
        FrictionModel("bad", latency_ms=10.0, jitter_log_sigma=-0.1)
    with pytest.raises(ValueError, match="rebate"):
        FrictionModel("bad", latency_ms=10.0, legging_cost_bps=-1.0)


def test_unknown_preset_names_the_alternatives():
    with pytest.raises(KeyError, match="stress"):
        get_preset("nope")


def test_set_active_round_trips():
    before = active()
    try:
        assert set_active("colocated").latency_ms == 5.0
        assert active().name == "colocated"
        custom = FrictionModel("custom", latency_ms=42.0)
        assert set_active(custom) is custom
        assert active() is custom
    finally:
        set_active(before)
    assert active() is before


def test_with_latency_derives_without_mutating():
    base = get_preset("stress")
    derived = base.with_latency(250.0)
    assert derived.latency_ms == 250.0
    assert base.latency_ms == 100.0
    assert derived.legging_cost_bps == base.legging_cost_bps
    assert "250" in derived.name


def test_comparison_table_lists_every_preset():
    table = comparison_table()
    for name in PRESETS:
        assert name in table


def test_eps_qty_is_below_anything_tradeable():
    """Guards the tolerance itself: 1e-12 BTC must be far under a real minimum."""
    assert EPS_QTY < 1e-8
    assert walk_book(PX, AM, 1e-5).filled_qty == 1e-5
