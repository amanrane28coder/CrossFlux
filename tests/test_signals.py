"""
tests/test_signals.py
=====================
Validation suite for the OBI signal engine — Phase 2.

Test inventory
--------------
Unweighted OBI (7 tests)
  1. test_obi_bid_skewed              – Bid-dominated book → OBI near +1.0
  2. test_obi_ask_skewed              – Ask-dominated book → OBI near -1.0
  3. test_obi_balanced_book           – Equal volumes → OBI == 0.0
  4. test_obi_zero_denominator        – All-zero volumes → OBI == 0.0 sentinel
  5. test_obi_depth_truncation        – Only top-N levels consumed, not whole book
  6. test_obi_fewer_levels_than_depth – Sparse book uses available levels; no error
  7. test_obi_invalid_depth           – depth=0 raises ValueError

Weighted multi-level OBI (13 tests)
  8. test_flat_profile_reproduces_calculate_obi – the baseline claim, to 8 ULP
  9. test_weighted_obi_stays_bounded            – |wOBI| < 1 over random books
 10. test_weighted_obi_bound_holds_for_all_profiles
 10b. test_weighted_obi_saturates_when_one_side_is_negligible – the bound is closed
 11. test_decay_weights_favour_the_touch        – weighting actually changes the answer
 12. test_l1_only_ignores_depth
 13. test_weighted_obi_zero_denominator
 14. test_asymmetric_book_uses_shallower_side   – documented depth convention
 15. test_scaling_weights_is_a_no_op            – normalization invariant
 16. test_unknown_profile_is_an_error           – no silent fallback
 17. test_negative_weight_rejected              – the bound's precondition
 18. test_delta_threshold_still_filters         – guards the pass-through failure
 19. test_weighted_delta_within_assert_bounds   – legal input to calculate_obi_delta

Why the weighted tests exist
----------------------------
The weighted signal is normalized specifically so it stays a legal input to
``calculate_obi_delta`` (which asserts [-1, 1]) and so ``DEFAULT_DELTA_THRESHOLD
= 0.3`` keeps filtering. An unnormalized implementation still "works" — it
produces numbers and fires trades — while silently turning the entry gate into a
pass-through. Tests 9, 10, 18 and 19 exist to make that failure loud.

C++/Python parity is *not* tested here: it needs a compiler and lives in
tests/test_bindings.py, which skips when the extension module is absent.

All tests use synthetic fixture data; no I/O, no network calls.

Mathematical boundary reference:
    OBI(bid_vol=100, ask_vol=1) = 99/101 ≈ 0.9802  → near +1.0
    OBI(bid_vol=1, ask_vol=100) = -99/101 ≈ -0.9802 → near -1.0

Python 3.10+ required.
"""

from __future__ import annotations

import random
import sys
from typing import List

import pytest

from src import obi_weights
from src.models import OrderBookSnapshot, PriceLevel
from src.obi_weights import (
    PROFILES,
    WeightProfile,
    from_decay,
    get_profile,
    weighted_obi_from_volumes,
)
from src.signals import calculate_obi, calculate_obi_delta, calculate_weighted_obi


# ===========================================================================
# Shared fixture factory
# ===========================================================================


def _make_obi_snapshot(
    bid_volumes: List[float],
    ask_volumes: List[float],
    exchange_id: str = "test",
    timestamp_ms: int = 1_700_000_000_000,
    base_bid_price: float = 49_999.0,
    base_ask_price: float = 50_001.0,
) -> OrderBookSnapshot:
    """Build a synthetic ``OrderBookSnapshot`` with precise per-level volumes.

    Prices are generated mechanically (best bid descending by 1.0 per level,
    best ask ascending by 1.0 per level) to satisfy the non-crossed invariant.
    The caller controls *volumes* exactly for OBI testing.

    Parameters
    ----------
    bid_volumes:
        Volumes for each bid level, index 0 = best bid.
    ask_volumes:
        Volumes for each ask level, index 0 = best ask.
    """
    if not bid_volumes:
        raise ValueError("_make_obi_snapshot: bid_volumes must not be empty.")
    if not ask_volumes:
        raise ValueError("_make_obi_snapshot: ask_volumes must not be empty.")

    bids = [
        PriceLevel(price=base_bid_price - i * 1.0, volume=vol)
        for i, vol in enumerate(bid_volumes)
    ]
    asks = [
        PriceLevel(price=base_ask_price + i * 1.0, volume=vol)
        for i, vol in enumerate(ask_volumes)
    ]
    return OrderBookSnapshot(
        exchange_id=exchange_id,
        timestamp_ms=timestamp_ms,
        bids=bids,
        asks=asks,
    )


# ===========================================================================
# Test 1 — Bid-skewed book: OBI near +1.0
# ===========================================================================


def test_obi_bid_skewed() -> None:
    """A heavily bid-dominated book must return an OBI value near +1.0.

    Construction:
        bid_vol = 100.0,  ask_vol = 1.0
        OBI = (100 - 1) / (100 + 1) = 99/101 ≈ 0.9802
    """
    snap = _make_obi_snapshot(
        bid_volumes=[100.0],
        ask_volumes=[1.0],
    )
    result = calculate_obi(snap, depth=5)

    expected = (100.0 - 1.0) / (100.0 + 1.0)  # ≈ 0.9802
    assert result > 0.9, f"Expected OBI > 0.9 for bid-skewed book, got {result:.6f}"
    assert abs(result - expected) < 1e-9, (
        f"OBI formula mismatch: expected {expected:.9f}, got {result:.9f}"
    )


# ===========================================================================
# Test 2 — Ask-skewed book: OBI near -1.0
# ===========================================================================


def test_obi_ask_skewed() -> None:
    """A heavily ask-dominated book must return an OBI value near -1.0.

    Construction:
        bid_vol = 1.0,  ask_vol = 100.0
        OBI = (1 - 100) / (1 + 100) = -99/101 ≈ -0.9802
    """
    snap = _make_obi_snapshot(
        bid_volumes=[1.0],
        ask_volumes=[100.0],
    )
    result = calculate_obi(snap, depth=5)

    expected = (1.0 - 100.0) / (1.0 + 100.0)  # ≈ -0.9802
    assert result < -0.9, f"Expected OBI < -0.9 for ask-skewed book, got {result:.6f}"
    assert abs(result - expected) < 1e-9, (
        f"OBI formula mismatch: expected {expected:.9f}, got {result:.9f}"
    )


# ===========================================================================
# Test 3 — Balanced book: OBI == 0.0
# ===========================================================================


def test_obi_balanced_book() -> None:
    """A perfectly balanced book (equal bid/ask volumes) must return OBI == 0.0.

    Construction:
        bid_vol = ask_vol = 5.0 + 3.0 + 2.0 = 10.0
        OBI = (10 - 10) / (10 + 10) = 0.0
    """
    volumes = [5.0, 3.0, 2.0]
    snap = _make_obi_snapshot(bid_volumes=volumes, ask_volumes=volumes)
    result = calculate_obi(snap, depth=5)

    assert result == 0.0, f"Expected OBI == 0.0 for balanced book, got {result}"


# ===========================================================================
# Test 4 — Zero denominator: illiquid book returns 0.0 sentinel
# ===========================================================================


def test_obi_zero_denominator() -> None:
    """A book where all volumes are 0.0 must return 0.0 without raising.

    This tests the ZeroDivisionError guard: denominator = 0.0 + 0.0 = 0.0.
    """
    snap = _make_obi_snapshot(
        bid_volumes=[0.0, 0.0, 0.0],
        ask_volumes=[0.0, 0.0, 0.0],
    )
    # Must not raise; must return the safe sentinel
    result = calculate_obi(snap, depth=5)
    assert result == 0.0, (
        f"Expected 0.0 sentinel for zero-volume book, got {result}"
    )


# ===========================================================================
# Test 5 — Depth truncation: only top-N levels are consumed
# ===========================================================================


def test_obi_depth_truncation() -> None:
    """calculate_obi with depth=2 must ignore levels beyond index 1.

    Book (5 levels per side):
        Bids:  [10.0, 10.0,  50.0, 50.0, 50.0]  ← top-2 sum = 20.0
        Asks:  [10.0, 10.0, 100.0, 100.0, 100.0] ← top-2 sum = 20.0

    With depth=2 → OBI = (20 - 20) / (20 + 20) = 0.0  (balanced at top-2)
    With depth=5 → OBI = (170 - 320) / (170 + 320) ≈ -0.306  (ask-heavy overall)

    The test verifies depth=2 returns 0.0, proving deeper levels are excluded.
    """
    snap = _make_obi_snapshot(
        bid_volumes=[10.0, 10.0, 50.0, 50.0, 50.0],
        ask_volumes=[10.0, 10.0, 100.0, 100.0, 100.0],
    )

    obi_depth2 = calculate_obi(snap, depth=2)
    obi_depth5 = calculate_obi(snap, depth=5)

    # At depth=2: balanced top-of-book → 0.0
    assert obi_depth2 == 0.0, (
        f"Expected OBI=0.0 at depth=2 (balanced top-2), got {obi_depth2}"
    )

    # At depth=5: ask-heavy overall → negative
    assert obi_depth5 < 0.0, (
        f"Expected OBI < 0 at depth=5 (ask-heavy full book), got {obi_depth5}"
    )

    # The two values must differ, proving depth truncation is active
    assert obi_depth2 != obi_depth5, (
        "OBI at depth=2 and depth=5 should differ but are equal — "
        "depth truncation may not be working."
    )


# ===========================================================================
# Test 6 — Fewer levels than depth: graceful degradation
# ===========================================================================


def test_obi_fewer_levels_than_depth() -> None:
    """A book with only 2 levels per side must compute correctly with depth=10.

    No error should be raised; the function uses all available levels.

    Construction (2 levels per side):
        bid_vol = 3.0 + 7.0 = 10.0
        ask_vol = 2.0 + 4.0 = 6.0
        OBI = (10 - 6) / (10 + 6) = 4/16 = 0.25
    """
    snap = _make_obi_snapshot(
        bid_volumes=[3.0, 7.0],
        ask_volumes=[2.0, 4.0],
    )
    result = calculate_obi(snap, depth=10)  # depth >> available levels

    expected = (10.0 - 6.0) / (10.0 + 6.0)  # = 0.25
    assert abs(result - expected) < 1e-9, (
        f"OBI with sparse book (depth=10, 2 levels): expected {expected}, got {result}"
    )
    assert result > 0.0, "Expected positive OBI for bid-heavy sparse book."


# ===========================================================================
# Test 7 — Invalid depth raises ValueError
# ===========================================================================


def test_obi_invalid_depth() -> None:
    """depth=0 is a programming error and must raise ValueError immediately."""
    snap = _make_obi_snapshot(
        bid_volumes=[1.0],
        ask_volumes=[1.0],
    )
    with pytest.raises(ValueError, match="strictly positive integer"):
        calculate_obi(snap, depth=0)


# ===========================================================================
# Weighted multi-level OBI
#
# The unweighted tests above pass volumes straight to calculate_obi. The
# weighted ones mostly go through weighted_obi_from_volumes, which takes raw
# sequences: it is the same arithmetic without the snapshot-construction
# overhead, and it is the function the C++ parity check compares against.
# Tests that specifically exercise the snapshot path use calculate_weighted_obi.
# ===========================================================================


# ---------------------------------------------------------------------------
# Test 8 — flat weighting reproduces the original signal
# ---------------------------------------------------------------------------

# Measured, not guessed: the largest divergence over 65,000 randomised books
# (uniform, log-uniform across 1e-6..1e4, near-identical sides, and books with a
# single 1e16 level to force cancellation) was 3 x eps. Eight leaves headroom for
# a different libm or a wider depth without being loose enough to hide a real
# reordering, which would show up in the 1e-3 range or larger.
FLAT_PARITY_TOL = 8 * sys.float_info.epsilon


def test_flat_profile_reproduces_calculate_obi() -> None:
    """The "flat" profile must track calculate_obi to within a few ULP.

    This is the load-bearing claim behind using flat as the baseline in a
    weighted-vs-unweighted comparison. If it drifts, every such comparison is
    measuring the change in arithmetic as well as the change in weights, and the
    two are no longer separable.

    Why this is a tolerance and not ``==``
    --------------------------------------
    An earlier version of this test asserted exact equality, on the reasoning
    that equal weights cancel algebraically so both functions do the same
    additions in the same order. The second half of that is false, and this test
    caught it: ``calculate_obi`` sums each side and then subtracts,
    ``weighted_obi_from_volumes`` subtracts per level and accumulates
    (``num += w * (bid[i] - ask[i])``). Same value in exact arithmetic, up to a
    few ULP apart in binary.

    The per-level form is deliberately kept, for two reasons that both outrank
    bit-parity with the older function:

    * It is the *more* accurate of the two under cancellation. For a book with a
      1e16 level on both sides, summing first rounds the smaller levels away
      entirely, while subtracting per level preserves them.
    * It is what the C++ implementation does, so it is what keeps the
      cross-language parity check in test_bindings.py meaningful. Rewriting
      Python to match a legacy Python function would trade a guarantee that
      spans both engines for one that does not.

    So the assertion is a hard numeric bound rather than an equality, and the
    bound is small enough that no gate decision can turn on it: delta_threshold
    is 0.3 and the divergence is ~1e-16.
    """
    books = [
        ([100.0, 50.0, 25.0, 12.0, 6.0], [1.0, 2.0, 4.0, 8.0, 16.0]),
        ([1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
        ([0.5, 0.25, 0.125, 0.0625, 0.03125], [3.0, 1.0, 0.5, 0.25, 0.125]),
        ([2.7182818, 3.1415926, 1.4142135, 1.7320508, 2.2360679],
         [0.5772156, 1.6180339, 0.3010299, 0.6931471, 1.2020569]),
    ]
    flat = get_profile("flat")

    for bids, asks in books:
        snap = _make_obi_snapshot(bid_volumes=bids, ask_volumes=asks)
        plain = calculate_obi(snap, depth=5)
        weighted = calculate_weighted_obi(snap, profile=flat)
        assert abs(weighted - plain) <= FLAT_PARITY_TOL, (
            f"flat profile diverged from calculate_obi by more than "
            f"{FLAT_PARITY_TOL:.2e} on\n"
            f"  bids={bids}\n  asks={asks}\n"
            f"  calculate_obi          = {plain!r}\n"
            f"  calculate_weighted_obi = {weighted!r}\n"
            f"  difference             = {weighted - plain!r}\n"
            f"A gap this large is a reordering or a weighting bug, not rounding."
        )
        # Direction is the part downstream code branches on, so assert it
        # separately and exactly. Skipped for books balanced to within the
        # tolerance, where "the sign" is not a meaningful property of either.
        if abs(plain) > FLAT_PARITY_TOL:
            assert (weighted > 0) == (plain > 0), (
                f"flat profile flipped the sign of the signal on "
                f"bids={bids} asks={asks}: {plain!r} -> {weighted!r}"
            )


# ---------------------------------------------------------------------------
# Tests 9 & 10 — the [-1, 1] bound, which everything downstream assumes
# ---------------------------------------------------------------------------


def test_weighted_obi_stays_bounded() -> None:
    """Randomised books must never produce |wOBI| >= 1.0.

    The bound is what makes the output a legal input to calculate_obi_delta and
    keeps delta_threshold meaningful. It holds by construction — non-negative
    weights times non-negative volumes — so a failure here means the
    implementation stopped normalizing, which is the exact bug this design was
    chosen to avoid.

    Strict inequality is correct *for this magnitude window* and not in general.
    The draw spans 10 decades (1e-6..1e4), and saturation to exactly ±1.0 needs
    about 16 — enough for one side to disappear below the other side's ULP. See
    test_weighted_obi_saturates_when_one_side_is_negligible for that case; the
    interval itself is closed.
    """
    rng = random.Random(20260821)
    profile = get_profile("decay_50")

    for trial in range(2000):
        n = rng.randint(1, 5)
        # Log-uniform magnitudes: real books span several orders of magnitude,
        # and a uniform draw would only ever exercise same-scale levels.
        bids = [10.0 ** rng.uniform(-6, 4) for _ in range(n)]
        asks = [10.0 ** rng.uniform(-6, 4) for _ in range(n)]
        val = weighted_obi_from_volumes(bids, asks, profile)
        assert -1.0 < val < 1.0, (
            f"trial {trial}: wOBI={val!r} escaped (-1, 1)\n"
            f"  bids={bids}\n  asks={asks}"
        )


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_weighted_obi_bound_holds_for_all_profiles(profile_name: str) -> None:
    """Every registered profile respects the bound, including future additions.

    Parametrised over PROFILES rather than a hardcoded list so a newly added
    profile is covered automatically instead of quietly untested.
    """
    rng = random.Random(hash(profile_name) & 0xFFFF)
    profile = get_profile(profile_name)

    for _ in range(500):
        n = rng.randint(1, 5)
        bids = [rng.uniform(0.0, 1000.0) for _ in range(n)]
        asks = [rng.uniform(0.0, 1000.0) for _ in range(n)]
        val = weighted_obi_from_volumes(bids, asks, profile)
        assert -1.0 <= val <= 1.0, (
            f"profile {profile_name!r}: wOBI={val!r} out of bounds\n"
            f"  bids={bids}\n  asks={asks}"
        )


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_weighted_obi_saturates_when_one_side_is_negligible(
    profile_name: str,
) -> None:
    """The bound is closed: ±1.0 is reachable, not merely approached.

    Two routes, both asserted here:

      * An empty side. numerator == denominator by construction, no rounding.
      * A side below the other's ULP. ``1e9 + 1e-9 == 1e9`` in double precision,
        so the two accumulators end up holding the same double and the quotient
        is exactly 1.0. This needs ~16 orders of magnitude between the sides,
        which no real BTC book has.

    This is not a defect: ``calculate_obi_delta``'s assert is inclusive, so a
    saturated signal is still legal. It is recorded because the docstrings used
    to claim an open interval (-1, 1), and t10 in cpp_engine/tests/
    test_signals.cpp failed on exactly this when written to match that claim.
    """
    profile = get_profile(profile_name)

    empty_asks = weighted_obi_from_volumes([5.0, 4.0, 3.0, 2.0, 1.0],
                                           [0.0, 0.0, 0.0, 0.0, 0.0], profile)
    empty_bids = weighted_obi_from_volumes([0.0, 0.0, 0.0, 0.0, 0.0],
                                           [5.0, 4.0, 3.0, 2.0, 1.0], profile)
    assert empty_asks == 1.0, f"{profile_name}: empty ask side gave {empty_asks!r}"
    assert empty_bids == -1.0, f"{profile_name}: empty bid side gave {empty_bids!r}"

    big, tiny = [1e9] * 5, [1e-9] * 5
    assert big[0] + tiny[0] == big[0], "premise broken: 1e-9 is no longer below the ULP of 1e9"
    assert weighted_obi_from_volumes(big, tiny, profile) == 1.0
    assert weighted_obi_from_volumes(tiny, big, profile) == -1.0


# ---------------------------------------------------------------------------
# Tests 11 & 12 — weighting must actually change the answer
# ---------------------------------------------------------------------------


def test_decay_weights_favour_the_touch() -> None:
    """A book that is bid-heavy at the touch and ask-heavy in depth must flip sign.

    This is the whole point of weighting, and it is the test that would fail if
    the weight vector were ignored — a bug that leaves every bound and parity
    check passing, because an unweighted result is still perfectly valid output.

    Book:
        bids = [100,  1,  1,  1,  1]   bid pressure concentrated at L1
        asks = [  1, 80, 80, 80, 80]   ask pressure sitting in depth

    flat     counts all levels equally -> depth dominates -> negative
    decay_50 discounts depth 0.5x per level -> the touch wins -> positive
    """
    bids = [100.0, 1.0, 1.0, 1.0, 1.0]
    asks = [1.0, 80.0, 80.0, 80.0, 80.0]

    flat = weighted_obi_from_volumes(bids, asks, get_profile("flat"))
    decay = weighted_obi_from_volumes(bids, asks, get_profile("decay_50"))

    assert flat < 0.0, f"expected flat weighting to read ask-heavy, got {flat}"
    assert decay > 0.0, f"expected decay_50 to read bid-heavy, got {decay}"
    assert decay > flat, (
        "decay_50 must weight the touch more heavily than flat does; "
        f"got decay={decay} <= flat={flat}"
    )


def test_l1_only_ignores_depth() -> None:
    """l1_only must depend on level 0 alone.

    Included because the upgrade request assumed top-of-book was the engine's
    existing behaviour. It was not, and keeping l1_only honest is what makes the
    ablation meaningful.
    """
    prof = get_profile("l1_only")
    base = weighted_obi_from_volumes([10.0, 0.0], [5.0, 0.0], prof)

    for deep_bid, deep_ask in ((0.0, 0.0), (999.0, 1.0), (1.0, 999.0), (500.0, 500.0)):
        val = weighted_obi_from_volumes([10.0, deep_bid], [5.0, deep_ask], prof)
        assert val == base, (
            f"l1_only changed with level-2 volumes ({deep_bid}, {deep_ask}): "
            f"{val} != {base}"
        )

    expected = (10.0 - 5.0) / (10.0 + 5.0)
    assert abs(base - expected) < 1e-12, f"expected {expected}, got {base}"


# ---------------------------------------------------------------------------
# Tests 13 & 14 — degenerate and asymmetric books
# ---------------------------------------------------------------------------


def test_weighted_obi_zero_denominator() -> None:
    """A zero-volume book returns the same 0.0 sentinel as calculate_obi.

    Also covers an empty side and a zero-weight prefix, which reach the guard by
    a different route: depth collapses to 0, so nothing is accumulated.
    """
    prof = get_profile("decay_50")
    assert weighted_obi_from_volumes([0.0, 0.0], [0.0, 0.0], prof) == 0.0
    assert weighted_obi_from_volumes([], [1.0, 2.0], prof) == 0.0
    assert weighted_obi_from_volumes([1.0, 2.0], [], prof) == 0.0

    # A profile whose only weight is zero cannot be constructed (validated in
    # WeightProfile.__post_init__), which is itself the guarantee being relied on.
    with pytest.raises(ValueError, match="all weights are zero"):
        WeightProfile(name="dead", weights=(0.0, 0.0))


def test_asymmetric_book_uses_shallower_side() -> None:
    """Depth is min(len(bids), len(asks), profile.depth) — the shallower side wins.

    This convention is deliberate and differs from ``calculate_obi``, which
    slices each side independently. It exists so Python matches the C++
    implementation exactly; see the note in calculate_weighted_obi's docstring.
    Pinning it here means a future "fix" toward calculate_obi's convention breaks
    a test that explains itself, rather than silently desynchronising the two
    languages.
    """
    prof = get_profile("flat")

    # 3 bid levels, 1 ask level -> only level 0 participates.
    got = weighted_obi_from_volumes([10.0, 99.0, 99.0], [4.0], prof)
    expected = (10.0 - 4.0) / (10.0 + 4.0)
    assert abs(got - expected) < 1e-12, (
        f"asymmetric book consumed more than the shallower side: "
        f"got {got}, expected {expected} (level 0 only)"
    )

    # Deeper bid levels must not move the result at all.
    assert weighted_obi_from_volumes([10.0, 1e9, 1e9], [4.0], prof) == got

    # Profile depth is the third term in the min: l1_only truncates a full book.
    full = weighted_obi_from_volumes([10.0, 10.0], [4.0, 4.0], get_profile("l1_only"))
    assert abs(full - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests 15, 16, 17 — the invariants and error contract of a profile
# ---------------------------------------------------------------------------


def test_scaling_weights_is_a_no_op() -> None:
    """Multiplying every weight by a constant must not change the result.

    Normalization implies weights are relative, which is what lets a profile be
    written in whatever units read most clearly. If this fails, the weights are
    leaking into the output scale and the (-1, 1) bound is not what it claims.
    """
    bids = [7.0, 3.0, 2.0, 1.0, 0.5]
    asks = [2.0, 8.0, 1.0, 4.0, 0.25]
    base = get_profile("decay_50")

    reference = weighted_obi_from_volumes(bids, asks, base)
    for k in (0.001, 0.5, 2.0, 1000.0, 1e6):
        scaled = WeightProfile(name=f"scaled_{k}",
                               weights=tuple(w * k for w in base.weights))
        got = weighted_obi_from_volumes(bids, asks, scaled)
        assert abs(got - reference) < 1e-12, (
            f"scaling weights by {k} changed the signal: {got} != {reference}"
        )


def test_unknown_profile_is_an_error() -> None:
    """An unrecognised profile name raises instead of falling back to the default.

    Silent fallback is the dangerous behaviour: a typo in
    ``CROSSFLUX_OBI_PROFILE`` would produce a complete set of plausible results
    for a signal nobody selected, and nothing on the page would say so. The C++
    side calls std::abort() in the same situation (obi_config.hpp).
    """
    with pytest.raises(KeyError) as excinfo:
        get_profile("decay_5o")          # letter o for zero
    message = str(excinfo.value)
    assert "decay_5o" in message, "error should name the bad input"
    assert "decay_50" in message, "error should list the valid profiles"

    # Case-insensitivity is intentional and matched by the C++ `lowered()` helper.
    assert get_profile("DECAY_50") is get_profile("decay_50")


def test_negative_weight_rejected() -> None:
    """Negative weights are rejected at construction.

    Non-negativity is the precondition for the (-1, 1) bound: it is what makes
    |numerator| <= denominator hold term by term. Allowing a negative weight
    would break the bound far from the construction site, in an assert inside
    calculate_obi_delta, which is a much harder thing to diagnose.
    """
    with pytest.raises(ValueError, match="negative weight"):
        WeightProfile(name="bad", weights=(1.0, -0.5))

    with pytest.raises(ValueError, match="weights cannot be empty"):
        WeightProfile(name="empty", weights=())

    with pytest.raises(ValueError, match="exceeds MAX_LEVELS"):
        WeightProfile(name="deep", weights=tuple(1.0 for _ in range(obi_weights.MAX_LEVELS + 1)))

    # from_decay guards its own inputs rather than deferring to WeightProfile.
    with pytest.raises(ValueError, match="ratio must be in"):
        from_decay(1.5)


# ---------------------------------------------------------------------------
# Tests 18 & 19 — the reason normalization is not optional
# ---------------------------------------------------------------------------


def test_delta_threshold_still_filters() -> None:
    """The 0.3 threshold must reject a meaningful share of realistic books.

    This is the regression test for the failure that motivated normalizing. An
    unnormalized weighted sum carries volume units, so on BTC books |delta| is
    routinely in the hundreds and every tick clears 0.3 — the entry gate becomes
    a pass-through. Nothing errors; the engine simply looks busier while
    filtering nothing, and the win rate stays high because of the separate
    fee-gate tautology.

    The assertion is deliberately loose (some rejection, some acceptance) because
    the exact pass rate depends on the synthetic book distribution and would make
    this a brittle test of the fixture rather than of the gate.
    """
    from src.predictor import DEFAULT_DELTA_THRESHOLD

    rng = random.Random(4242)
    profile = get_profile("decay_50")
    passed = 0
    total = 3000

    for _ in range(total):
        def side() -> list[float]:
            return [rng.uniform(0.0, 50.0) for _ in range(5)]

        obi_a = weighted_obi_from_volumes(side(), side(), profile)
        obi_b = weighted_obi_from_volumes(side(), side(), profile)
        if abs(calculate_obi_delta(obi_a, obi_b)) > DEFAULT_DELTA_THRESHOLD:
            passed += 1

    rate = passed / total
    assert 0.01 < rate < 0.95, (
        f"gate pass rate {rate:.1%} at threshold {DEFAULT_DELTA_THRESHOLD}. "
        "Near 100% means the signal is no longer normalized and the gate has "
        "degraded into a pass-through; near 0% means it rejects everything."
    )


def test_weighted_delta_within_assert_bounds() -> None:
    """A weighted delta must be a legal input to calculate_obi_delta.

    calculate_obi_delta validates [-1, 1] on both inputs (mirroring the C++
    assert at signals.hpp:173-175) and returns a value in [-2, 2]. Feeding it
    weighted OBI is the entire integration point, so it is worth asserting
    directly rather than inferring it from the bound test.
    """
    rng = random.Random(99)
    for profile_name in sorted(PROFILES):
        profile = get_profile(profile_name)
        for _ in range(300):
            def side() -> list[float]:
                return [rng.uniform(0.0, 100.0) for _ in range(5)]

            a = weighted_obi_from_volumes(side(), side(), profile)
            b = weighted_obi_from_volumes(side(), side(), profile)
            delta = calculate_obi_delta(a, b)      # raises if out of range
            assert -2.0 <= delta <= 2.0, (
                f"{profile_name}: delta={delta} outside [-2, 2]"
            )
