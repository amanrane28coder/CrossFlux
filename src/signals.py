"""
src/signals.py
==============
Statistical feature generation layer for the Cross-Venue Arbitrage Predictor.

Phase 2: Order Book Imbalance (OBI) Signal
Phase 3: Cross-Venue OBI Delta Signal
------------------------------------------
OBI quantifies the directional pressure of resting liquidity across the top
``depth`` levels of a limit order book.  It is a primary signal for detecting
*ghost liquidity* — phantom orders placed to create a false impression of depth
on one side of the book without genuine execution intent.

Formula
-------
                 V_bid(d)  -  V_ask(d)
    OBI(d)  =  ─────────────────────────
                 V_bid(d)  +  V_ask(d)

where:
    V_bid(d) = Σ bids[i].volume  for i in 0 … min(d, len(bids)) - 1
    V_ask(d) = Σ asks[i].volume  for i in 0 … min(d, len(asks)) - 1

Signal range
------------
    OBI ≈ +1.0  →  Heavily bid-side; bullish pressure; possible ghost asks.
    OBI ≈ -1.0  →  Heavily ask-side; bearish pressure; possible ghost bids.
    OBI ≈  0.0  →  Balanced book; no dominant directional imbalance.
    OBI =  0.0  →  Degenerate / illiquid book (denominator = 0); safe sentinel.

Python 3.10+ required.
"""

from __future__ import annotations

import logging

from src.models import OrderBookSnapshot
from src.obi_weights import WeightProfile
from src.obi_weights import active as active_profile
from src.obi_weights import weighted_obi_from_volumes

logger = logging.getLogger(__name__)


def calculate_obi(snapshot: OrderBookSnapshot, depth: int = 5) -> float:
    """Calculate the Order Book Imbalance (OBI) across the top ``depth`` levels.

    The limit order book is treated as a step function: volumes are summed
    as-is at each level; no weighting, interpolation, or price-distance
    adjustment is applied.

    Parameters
    ----------
    snapshot:
        A valid, non-crossed ``OrderBookSnapshot``.
    depth:
        Number of price levels to consume from each side of the book.
        If the book has fewer than ``depth`` levels on either side, all
        available levels are used — no padding, no error.
        Must be a strictly positive integer.

    Returns
    -------
    float
        OBI scalar in the open interval ``(-1.0, 1.0)``, or exactly ``0.0``
        for a degenerate / illiquid book where total volume is zero.

    Raises
    ------
    ValueError
        If ``depth <= 0``.  This is a programming error, not a data condition.

    Examples
    --------
    >>> from src.models import PriceLevel, OrderBookSnapshot
    >>> snap = OrderBookSnapshot(
    ...     exchange_id="binance",
    ...     timestamp_ms=1_700_000_000_000,
    ...     bids=[PriceLevel(49_999.0, 10.0), PriceLevel(49_998.0, 5.0)],
    ...     asks=[PriceLevel(50_001.0, 1.0),  PriceLevel(50_002.0, 0.5)],
    ... )
    >>> calculate_obi(snap, depth=2)  # (15.0 - 1.5) / (15.0 + 1.5)
    0.8181818181818182
    """
    if depth <= 0:
        raise ValueError(
            f"calculate_obi: depth must be a strictly positive integer, got {depth!r}."
        )

    # --- Slice to requested depth (handles books with fewer levels gracefully) ---
    bid_levels = snapshot.bids[:depth]
    ask_levels = snapshot.asks[:depth]

    # --- Aggregate volumes per side ---
    bid_vol: float = sum(lvl.volume for lvl in bid_levels)
    ask_vol: float = sum(lvl.volume for lvl in ask_levels)

    denominator: float = bid_vol + ask_vol

    # --- Zero-denominator guard: completely illiquid / zero-volume book ---
    # Return 0.0 as a safe sentinel rather than raising ZeroDivisionError.
    # This covers: both sides empty, all levels have volume=0.0, etc.
    if denominator == 0.0:
        logger.debug(
            "[%s] calculate_obi: zero-volume book at %d ms — returning 0.0 sentinel.",
            snapshot.exchange_id,
            snapshot.timestamp_ms,
        )
        return 0.0

    obi: float = (bid_vol - ask_vol) / denominator

    logger.debug(
        "[%s] OBI(depth=%d) = %.6f  (bid_vol=%.4f, ask_vol=%.4f) at %d ms",
        snapshot.exchange_id,
        depth,
        obi,
        bid_vol,
        ask_vol,
        snapshot.timestamp_ms,
    )

    return obi


def calculate_weighted_obi(
    snapshot: OrderBookSnapshot,
    profile: WeightProfile | None = None,
) -> float:
    """Calculate multi-level OBI with per-level weights.

    Discounts resting depth by distance from the touch: size five ticks away
    says less about immediate pressure than size at the front of the queue, and
    is cheaper to spoof.

    Formula
    -------
    ::

                     Σ w[i] · (bids[i].volume − asks[i].volume)
        wOBI  =     ─────────────────────────────────────────────
                     Σ w[i] · (bids[i].volume + asks[i].volume)

    This is **not MLOFI**. MLOFI (Cont–Kukanov–Stoikov and its multi-level
    extensions) is built from the *change* in depth between consecutive book
    updates and measures order *flow*; this measures a static snapshot of
    resting depth. See ``src/obi_weights.py`` for the full distinction and why
    real MLOFI is currently blocked on data rather than code.

    Normalizing by weighted total volume keeps the result in ``(-1.0, 1.0)``,
    matching ``calculate_obi``, so the output is a legal input to
    ``calculate_obi_delta`` and the engine's ``delta_threshold`` stays
    calibrated. The bound holds by construction — weights and volumes are both
    non-negative — so no clamping is needed.

    Depth convention differs from ``calculate_obi``
    ----------------------------------------------
    This function consumes ``min(len(bids), len(asks), profile.depth)`` levels:
    the shallower book side wins. ``calculate_obi`` above instead slices each
    side independently, which is also what the C++ ``calculate_obi`` does *not*
    do — the two languages already disagree on asymmetric books. That divergence
    is pre-existing and left alone here because changing it would move published
    backtest numbers. This function deliberately follows the C++ convention so
    Python and C++ agree, verified to one ULP across 411 cases including
    asymmetric and degenerate books.

    Parameters
    ----------
    snapshot:
        A valid ``OrderBookSnapshot``.
    profile:
        Weight profile from ``src.obi_weights``. Defaults to the active profile
        (``$CROSSFLUX_OBI_PROFILE``, else ``flat``).

    Returns
    -------
    float
        Weighted OBI in ``(-1.0, 1.0)``, or exactly ``0.0`` for a degenerate
        book — the same sentinel ``calculate_obi`` uses.

    Examples
    --------
    >>> from src.models import PriceLevel, OrderBookSnapshot
    >>> from src.obi_weights import get_profile

    A book with a large ask resting at level 2 — the case where the choice of
    profile actually matters:

    >>> snap = OrderBookSnapshot(
    ...     exchange_id="binance",
    ...     timestamp_ms=1_700_000_000_000,
    ...     bids=[PriceLevel(49_999.0, 10.0), PriceLevel(49_998.0, 5.0)],
    ...     asks=[PriceLevel(50_001.0, 1.0),  PriceLevel(50_002.0, 20.0)],
    ... )

    Flat weighting lets that deep ask dominate, and reports ask-side pressure:

    >>> round(calculate_weighted_obi(snap, get_profile("flat")), 6)
    -0.166667

    Halving per level discounts it enough to flip the sign to mildly bid-side:

    >>> round(calculate_weighted_obi(snap, get_profile("decay_50")), 6)
    0.06383

    Top-of-book only ignores it altogether, and sees a strongly bid-heavy book:

    >>> round(calculate_weighted_obi(snap, get_profile("l1_only")), 6)
    0.818182

    Three profiles, three different signs of conclusion from one book. Which is
    right is an empirical question this repo cannot yet answer — see the caveat
    in ``src/obi_weights.py`` about the entry gate.

    Note that a *proportional* book, where each level has the same bid:ask
    ratio, gives an identical result under every profile — reweighting cannot
    change a ratio that is constant across levels:

    >>> flat_snap = OrderBookSnapshot(
    ...     exchange_id="binance",
    ...     timestamp_ms=1_700_000_000_000,
    ...     bids=[PriceLevel(49_999.0, 10.0), PriceLevel(49_998.0, 5.0)],
    ...     asks=[PriceLevel(50_001.0, 1.0),  PriceLevel(50_002.0, 0.5)],
    ... )
    >>> calculate_weighted_obi(flat_snap, get_profile("flat"))
    0.8181818181818182
    >>> calculate_weighted_obi(flat_snap, get_profile("decay_50"))
    0.8181818181818182

    Worth knowing when a profile sweep shows no difference: the books may be
    proportional rather than the weighting inert.
    """
    prof = profile if profile is not None else active_profile()

    depth: int = min(len(snapshot.bids), len(snapshot.asks), prof.depth)

    bid_volumes = [lvl.volume for lvl in snapshot.bids[:depth]]
    ask_volumes = [lvl.volume for lvl in snapshot.asks[:depth]]

    # Delegated so exactly one implementation of the arithmetic exists. The
    # accumulation order there matches cpp_engine/include/signals.hpp term for
    # term, which is what keeps the two languages in agreement.
    wobi: float = weighted_obi_from_volumes(bid_volumes, ask_volumes, prof)

    if wobi == 0.0 and depth == 0:
        logger.debug(
            "[%s] calculate_weighted_obi: no usable levels at %d ms "
            "(bids=%d, asks=%d, profile=%s) — returning 0.0 sentinel.",
            snapshot.exchange_id,
            snapshot.timestamp_ms,
            len(snapshot.bids),
            len(snapshot.asks),
            prof.name,
        )
        return 0.0

    logger.debug(
        "[%s] wOBI(profile=%s, depth=%d) = %.6f at %d ms",
        snapshot.exchange_id,
        prof.name,
        depth,
        wobi,
        snapshot.timestamp_ms,
    )

    return wobi


def calculate_obi_delta(obi_exchange_a: float, obi_exchange_b: float) -> float:
    """Calculate the cross-venue OBI delta — the directional divergence between two venues.

    This is the primary composite signal for detecting cross-venue arbitrage
    opportunities driven by ghost liquidity.  A large positive delta means
    exchange A is heavily bid-side while exchange B is heavily ask-side — a
    classic pattern preceding a cross-venue price correction.

    Formula
    -------
    ::

        Δ_OBI(A, B) = OBI_A − OBI_B   ∈ [−2.0, +2.0]

    Unlike ``MarketState.spread_between()`` (a price-level metric), OBI delta
    is a *volume-pressure* metric and captures ghost liquidity dynamics that
    price-only signals miss entirely.

    Parameters
    ----------
    obi_exchange_a:
        OBI of the first venue, as returned by ``calculate_obi()``.
        Must be in the closed interval ``[-1.0, 1.0]``.
    obi_exchange_b:
        OBI of the second venue, as returned by ``calculate_obi()``.
        Must be in the closed interval ``[-1.0, 1.0]``.

    Returns
    -------
    float
        OBI delta in ``[-2.0, +2.0]``.  Positive means exchange A is more
        bid-heavy than exchange B; negative means the reverse.

    Raises
    ------
    ValueError
        If either OBI input is outside ``[-1.0, 1.0]``.  A value outside
        this range indicates a corrupted upstream signal, not a data edge case.

    Examples
    --------
    >>> calculate_obi_delta(obi_exchange_a=0.8, obi_exchange_b=-0.6)
    1.4
    >>> calculate_obi_delta(obi_exchange_a=0.0, obi_exchange_b=0.0)
    0.0
    """
    _OBI_BOUNDS = (-1.0, 1.0)

    for label, value in (("obi_exchange_a", obi_exchange_a), ("obi_exchange_b", obi_exchange_b)):
        if not (_OBI_BOUNDS[0] <= value <= _OBI_BOUNDS[1]):
            raise ValueError(
                f"calculate_obi_delta: {label}={value!r} is outside the valid "
                f"OBI range [{_OBI_BOUNDS[0]}, {_OBI_BOUNDS[1]}]. "
                f"This indicates a corrupted upstream signal."
            )

    delta: float = obi_exchange_a - obi_exchange_b

    logger.debug(
        "OBI delta = %.6f  (A=%.6f, B=%.6f)",
        delta,
        obi_exchange_a,
        obi_exchange_b,
    )

    return delta
