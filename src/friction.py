"""Microstructural friction: the delay between signal and fill, and the price
paid for size.

Why this module exists
----------------------
The backtest's entry gate is ``margin > fee`` and its booked PnL was
``margin - fee``, both evaluated on the *same* order-book snapshot. A trade was
therefore admitted if and only if it was profitable, and the reported win rate
measured the gate's arithmetic rather than any predictive skill. Fee presets
could not falsify it (see ``src/fees.py``); nothing could, because signal and
execution read the same row of the same DataFrame.

Two mechanisms decouple them, and both are needed:

  latency   The fill is evaluated against the book prevailing at
            ``T + latency_ms``, not the book that produced the signal. If the
            spread collapsed in between, the trade still executes -- at the
            worse price -- and books a loss.

  VWAP      Top-of-book does not absorb an arbitrary notional. The order walks
            down the levels, paying each one's price for its own depth, so the
            realized price degrades with size.

Measured effect on ``data/raw/`` 2024-03-01 (307,881 signals, both directions,
institutional fees, ``stress`` preset -- 100 ms per leg):

    qty BTC   win%   adverse%   unfilled%   mean edge    net PnL
       0.01   98.6        1.4        0.01    4.25 bps     $0.08M
       0.10   97.7        2.3        0.08    4.19 bps     $0.80M
       0.25   97.1        2.9        0.22    4.14 bps     $1.98M
       1.00   95.0        5.0        1.14    4.00 bps     $7.57M
       3.00   85.3       14.7        7.25    3.40 bps    $18.12M
       5.00   68.8       31.2       20.14    2.22 bps    $16.96M

Regenerate with ``python3 backtest/sweep_order_size.py``. Every figure above is
re-derived from its JSON by ``backtest/check_report_numbers.py``, which exists
because an earlier version of this table was wrong for a week: it came from a
script that read a truncated slice of the day and claimed 93.3% at 0.01 BTC
falling to 59.8% at 3.0, with 54.96% unfilled. If those numbers turn up
anywhere again, they are not a variant measurement -- they are the bug.

Three things that table does say, and one it does not:

  * Losing trades are reachable at every size, so the strategy is falsifiable.
    That is the point of the whole module. The old engine booked 99.0% winners
    at every size, the residual 1% being a slippage-tolerance artifact rather
    than a loss.
  * Mean edge decays with size. That is slippage being paid rather than assumed.
  * **Net PnL peaks at 3.0 BTC and falls at 5.0** -- a capacity limit. The old
    engine had none: doubling size doubled PnL exactly.
  * It does *not* say the strategy works. Net PnL stays positive at every size,
    and that is not a validation: the edge in this dataset is the persistent
    +4.79 bps USDT/USD quote-currency basis (``TEST_REPORT.md`` 2.3), and a
    100 ms delay does not remove a structural basis. Breaking the tautology
    makes the win rate meaningful; it does not make the strategy an arbitrage.

One caveat on the large sizes. Above roughly 1 BTC the unfilled fraction is an
artifact of the data, not of the market: ``book_snapshot_5`` carries five levels,
so a large order runs out of book because the file stops, not because liquidity
stops. Slippage measured at 3-5 BTC is a lower bound on fill quality and an
upper bound on cost, and the location of the PnL peak is therefore a property of
a five-level feed. Real depth needs the full incremental L2 feed.

Latency is deterministic by default. Three tests were previously flaky against
an unseeded ``random.gauss``, and a stochastic default makes fee-preset and
weight-profile comparisons incommensurable -- two runs differ for reasons that
have nothing to do with the change under test. Jitter is opt-in per preset.

Usage
-----
    from src.friction import active, get_preset, set_active, walk_book

    fr = active()                          # env-or-default preset
    fr.latency_ms                           # 100.0

    set_active("colocated")                 # switch at runtime
    for name in PRESETS: ...                # sweep every preset

Select a preset without editing code:

    CROSSFLUX_FRICTION_PRESET=colocated python -m backtest.engine

Print the presets side by side:

    python -m src.friction
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

ENV_VAR = "CROSSFLUX_FRICTION_PRESET"
DEFAULT_PRESET = "stress"

# Quantities below this are treated as zero. Book amounts are in BTC and the
# smallest meaningful trade is ~1e-5 BTC, so 1e-12 is comfortably below anything
# real while still absorbing float accumulation error in a cumulative sum.
EPS_QTY: float = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Walking the book
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BookWalk:
    """Result of consuming ``qty`` from one side of one book."""

    vwap:            float  # volume-weighted average fill price (0.0 if nothing filled)
    filled_qty:      float  # how much the book could actually supply
    levels_consumed: int    # how many price levels were touched
    notional:        float  # vwap * filled_qty, i.e. cash exchanged
    requested_qty:   float  # what was asked for, so shortfall is visible here

    @property
    def shortfall(self) -> float:
        """Quantity the book could not supply. Zero on a complete fill."""
        return max(0.0, self.requested_qty - self.filled_qty)

    @property
    def partial(self) -> bool:
        return self.shortfall > EPS_QTY

    def slippage_bps(self, touch_price: float) -> float:
        """Cost of size, in bps against the top-of-book price.

        Unsigned: a buy filling above the touch and a sell filling below it are
        both a cost, and reporting them with opposite signs invites them to
        cancel in an average.
        """
        if touch_price <= 0.0 or self.filled_qty <= EPS_QTY:
            return 0.0
        return abs(self.vwap - touch_price) / touch_price * 1e4


def walk_book(
    prices: Sequence[float],
    amounts: Sequence[float],
    qty: float,
) -> BookWalk:
    """Consume ``qty`` from a book side, cheapest level first.

    This is the reference implementation: a plain loop, in the same shape as the
    C++ ``crossflux::walk_book``, so the two can be read side by side and are
    asserted equal by ``cpp_engine/tests/run_parity_check.py``. The vectorised
    ``walk_book_array`` below must agree with it exactly, which
    ``tests/test_friction.py`` checks -- when a fast path and a readable path
    disagree, the readable one is right.

    ``prices`` must be ordered outward from the touch (ascending for asks,
    descending for bids); the walk does not sort, because a real order does not
    get to reorder the book. Levels with non-positive price or amount are
    skipped, since exchange snapshots pad absent levels with zeros.

    Returns ``vwap = 0.0`` when nothing fills. That is a sentinel, not a price:
    check ``filled_qty`` before using it.

    Note the shape of the outstanding-quantity update: ``qty - taken`` against a
    running total, rather than the more obvious ``remaining -= take``. The two
    are algebraically identical and differ in floating point -- repeated
    subtraction rounds at every level, while a running total rounds the same way
    ``np.cumsum`` does. Written the obvious way, this function disagreed with
    ``walk_book_array`` by ~2.8e-14, and closing that gap is worth one unusual
    line: a fast path that merely *nearly* matches its reference is a fast path
    nobody can check.
    """
    if qty <= EPS_QTY or len(prices) == 0:
        return BookWalk(0.0, 0.0, 0, 0.0, max(0.0, float(qty)))

    want = float(qty)
    taken = 0.0
    notional = 0.0
    consumed = 0

    for price, amount in zip(prices, amounts):
        if want - taken <= EPS_QTY:
            break
        consumed += 1
        if price <= 0.0 or amount <= EPS_QTY:
            continue
        take = min(want - taken, float(amount))
        notional += take * float(price)
        taken += take

    if taken <= EPS_QTY:
        return BookWalk(0.0, 0.0, consumed, 0.0, want)

    return BookWalk(
        vwap=notional / taken,
        filled_qty=taken,
        levels_consumed=consumed,
        notional=notional,
        requested_qty=want,
    )


def walk_book_array(
    prices: np.ndarray,
    amounts: np.ndarray,
    qty: "float | np.ndarray",
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``walk_book`` over many books at once.

    ``prices`` and ``amounts`` are ``(n_rows, n_levels)``; ``qty`` is a scalar or
    an ``(n_rows,)`` array, so each row may request a different size. Returns
    ``(vwap, filled_qty)``, each ``(n_rows,)``.

    The arithmetic is identical to the loop, expressed cumulatively: level *i*
    supplies whatever of the order is still outstanding when it is reached,
    capped by its own depth. With ``prev`` the depth available strictly above
    level *i*, that is ``clip(qty - prev, 0, amount_i)`` -- zero once the order
    is complete, the full level while still deep in the queue, and a partial
    slice exactly once.

    Agreement with ``walk_book`` is bit-for-bit at ``BOOK_DEPTH = 5``, and
    ``tests/test_friction.py`` asserts it with ``==`` rather than a tolerance.
    That guarantee has a stated limit: it holds because ``np.sum`` reduces a
    narrow axis sequentially, matching the loop's accumulation order. Beyond
    8 levels numpy switches to pairwise summation and the two diverge by ~1 ULP
    (verified: equal at 5 levels, unequal at 13). The divergence cannot change a
    trade decision -- it is ~1e-16 relative against gates in the 1e-3 range --
    but if the book is ever widened past 8 levels, the exact-equality test will
    fail and should be relaxed to a bound, not deleted.
    """
    prices = np.asarray(prices, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if prices.shape != amounts.shape:
        raise ValueError(
            f"prices {prices.shape} and amounts {amounts.shape} must have the "
            "same shape (n_rows, n_levels)."
        )
    if prices.ndim != 2:
        raise ValueError(f"expected 2-D (n_rows, n_levels), got {prices.ndim}-D.")

    want = np.asarray(qty, dtype=float).reshape(-1, 1)
    if want.shape[0] not in (1, prices.shape[0]):
        raise ValueError(
            f"qty has {want.shape[0]} rows but there are {prices.shape[0]} books."
        )

    # Zero out padded / crossed levels so they neither supply depth nor price.
    usable = np.where((prices > 0.0) & (amounts > EPS_QTY), amounts, 0.0)

    csum = np.cumsum(usable, axis=1)
    prev = np.concatenate([np.zeros((usable.shape[0], 1)), csum[:, :-1]], axis=1)
    take = np.clip(want - prev, 0.0, usable)

    filled = take.sum(axis=1)
    notional = (take * prices).sum(axis=1)

    vwap = np.where(filled > EPS_QTY, notional / np.where(filled > EPS_QTY, filled, 1.0), 0.0)
    filled = np.where(filled > EPS_QTY, filled, 0.0)
    return vwap, filled


def prevailing_row(index: np.ndarray, at_ms: np.ndarray) -> np.ndarray:
    """Row of the last book update at or before each time in ``at_ms``.

    This is the whole of "buffer the order until ``T + latency_ms``": the fill
    reads the most recent quote that had actually printed by then.

    ``side="right"`` then ``-1`` is load-bearing. ``searchsorted`` without it
    returns the *next* update at or after the target -- a quote that did not
    exist when the order arrived -- which is the look-ahead this replaces. On a
    duplicated timestamp it also picks the last update at that instant, matching
    the ``groupby(level=0).last()`` used to align the venues.

    Returns -1 where the time precedes the first update, so callers must guard.
    """
    return np.searchsorted(index, at_ms, side="right") - 1


# ─────────────────────────────────────────────────────────────────────────────
# Friction model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FrictionModel:
    """How long a fill takes, and what an unhedged leg costs to clean up."""

    name: str
    latency_ms: float
    jitter_log_sigma: float = 0.0
    legging_cost_bps: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.latency_ms < 0.0:
            raise ValueError(
                f"FrictionModel {self.name!r}: latency_ms={self.latency_ms!r} is "
                "negative, which would fill the order before the signal existed."
            )
        if self.jitter_log_sigma < 0.0:
            raise ValueError(
                f"FrictionModel {self.name!r}: jitter_log_sigma cannot be negative."
            )
        if self.legging_cost_bps < 0.0:
            raise ValueError(
                f"FrictionModel {self.name!r}: legging_cost_bps cannot be negative "
                "-- unwinding an unwanted position is not a rebate."
            )

    # ── latency ──────────────────────────────────────────────────────────────
    @property
    def deterministic(self) -> bool:
        return self.jitter_log_sigma == 0.0

    def sample_latency(
        self,
        n: int,
        rng: "np.random.Generator | None" = None,
    ) -> np.ndarray:
        """``n`` latencies in ms, one per order leg.

        With ``jitter_log_sigma == 0`` every draw is exactly ``latency_ms``, so a
        run is reproducible without a seed and two runs differ only by the thing
        being compared.

        Otherwise ``latency_ms * exp(sigma * Z)``: multiplicative, so the
        *median* is exactly ``latency_ms`` and the tail is right-skewed the way
        network delay actually is. sigma is in log space, not milliseconds --
        sigma=0.4 puts p99 at about 2.5x the median, sigma=0.8 at about 6.4x.
        An additive Gaussian would be symmetric and can go negative; the
        previous ``base + lognormal(mu, sigma)`` form made ``latency_ms``
        unrecoverable from the output, since the reported median was
        ``base + exp(mu)``.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)
        if self.deterministic:
            return np.full(n, float(self.latency_ms), dtype=float)
        gen = rng if rng is not None else np.random.default_rng()
        return self.latency_ms * np.exp(self.jitter_log_sigma * gen.standard_normal(n))

    def latency_quantile(self, p: float) -> float:
        """Latency at probability ``p``, for stating the tail without sampling."""
        if not 0.0 < p < 1.0:
            raise ValueError(f"p must be in (0, 1), got {p!r}.")
        if self.deterministic:
            return float(self.latency_ms)
        # Inverse standard normal via the erf inverse relation.
        z = math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
        return float(self.latency_ms * math.exp(self.jitter_log_sigma * z))

    # ── unhedged residual ────────────────────────────────────────────────────
    def legging_cost(
        self,
        residual_qty: "float | np.ndarray",
        price: "float | np.ndarray",
    ) -> "float | np.ndarray":
        """Cost of flattening a position one leg acquired and the other did not.

        When the buy and sell legs fill different quantities the difference is a
        naked position that has to be closed, crossing a spread to do it. The
        backtest previously charged nothing for this -- ``TEST_REPORT.md`` 4,
        "legging risk is modeled as costless".

        ``legging_cost_bps`` is an assumption, not a measurement: roughly one
        spread crossing on the venue that over-filled. Set it to 0.0 to recover
        the old costless behaviour and see how much of the result depended on it.
        """
        return np.abs(residual_qty) * np.asarray(price) * (self.legging_cost_bps / 1e4)

    # ── derivation / display ─────────────────────────────────────────────────
    def with_latency(self, latency_ms: float) -> "FrictionModel":
        """Copy at a different latency, for sweeps."""
        return replace(self, name=f"{self.name}@{latency_ms:g}ms", latency_ms=float(latency_ms))

    def describe(self) -> str:
        if self.deterministic:
            lat = f"{self.latency_ms:>6.1f}ms fixed"
        else:
            lat = (f"{self.latency_ms:>6.1f}ms median, p99="
                   f"{self.latency_quantile(0.99):.0f}ms")
        return f"{self.name:<14} {lat:<32} legging={self.legging_cost_bps:.1f}bps"


def _erfinv(y: float) -> float:
    """Inverse error function, Newton-refined from Winitzki's approximation.

    Written out rather than imported because ``scipy`` is not a dependency of
    this module and ``statistics.NormalDist`` is only used for the quantile,
    which needs to work under the same minimal environment as the rest of src/.
    Accurate to ~1e-15 after two Newton steps, which is far beyond what a
    latency quantile needs.
    """
    if y <= -1.0 or y >= 1.0:
        raise ValueError(f"erfinv domain is (-1, 1), got {y!r}.")
    if y == 0.0:
        return 0.0
    a = 0.147
    ln1my2 = math.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a) + ln1my2 / 2.0
    x = math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln1my2 / a) - t1), y)
    # The initial approximation is accurate in the center but loses precision
    # in the tails, where erf's derivative is small. Iterate to a residual bound
    # instead of assuming two Newton steps suffice for every p in (0, 1).
    for _ in range(8):
        err = math.erf(x) - y
        if abs(err) <= 2e-16:
            break
        derivative = 2.0 / math.sqrt(math.pi) * math.exp(-x * x)
        if derivative == 0.0:
            break
        x -= err / derivative
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Presets
#
# latency_ms is one-way, per leg, and drawn independently for the buy and sell
# leg -- so the two halves of one arbitrage do not fill at the same instant, and
# the resulting quantity mismatch is charged at legging_cost_bps.
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, FrictionModel] = {
    "stress": FrictionModel(
        name="stress",
        latency_ms=100.0,
        legging_cost_bps=5.0,
        description=(
            "The default, and the one to quote. 100 ms is a deliberately harsh "
            "round number that happens to land almost exactly on Binance's "
            "median inter-quote gap in this sample (100.0 ms), so the fill reads "
            "roughly one snapshot past the signal -- the book has genuinely moved "
            "on 89.6% of rows. On its own that is worth about 0.7 points of win "
            "rate at 0.01 BTC (99.3% under `zero`, 98.6% here); size is what "
            "actually bites, and by 5 BTC the same preset gives 68.8%."
        ),
    ),
    "colocated": FrictionModel(
        name="colocated",
        latency_ms=5.0,
        legging_cost_bps=5.0,
        description=(
            "Best case for a co-located taker: 5 ms one-way. Below Kraken's "
            "6.4 ms median inter-quote gap, so the fill usually sees the same "
            "book as the signal and friction comes almost entirely from size. "
            "Useful for isolating VWAP from latency."
        ),
    ),
    "retail": FrictionModel(
        name="retail",
        latency_ms=250.0,
        jitter_log_sigma=0.5,
        legging_cost_bps=10.0,
        description=(
            "Retail path over the public internet: 250 ms median with a heavy "
            "right tail (p99 about 800 ms) and a wider unwind cost. The only "
            "preset with jitter enabled, so results move between runs unless the "
            "RNG is seeded -- pass a seed when comparing anything."
        ),
    ),
    "zero": FrictionModel(
        name="zero",
        latency_ms=0.0,
        legging_cost_bps=0.0,
        description=(
            "Frictionless *latency*: the fill reads the signal's own book. "
            "Diagnostic only, and it does not restore the old tautological 100% "
            "-- it gives 99.3% at 0.01 BTC and 91.4% at 3.0, because removing "
            "the delay does not remove the book walk, and size alone loses money "
            "when the gate quotes the touch and the fill pays the VWAP of five "
            "levels. Use it to isolate the cost of latency, not to reproduce the "
            "old result. Never report a return from it."
        ),
    ),
}


def get_preset(name: str) -> FrictionModel:
    try:
        return PRESETS[str(name).lower()]
    except KeyError:
        raise KeyError(
            f"Unknown friction preset {name!r}. Available: {', '.join(sorted(PRESETS))}."
        ) from None


_active: FrictionModel | None = None


def active() -> FrictionModel:
    """Currently selected model: explicit ``set_active`` > ``$CROSSFLUX_FRICTION_PRESET`` > default."""
    global _active
    if _active is not None:
        return _active
    _active = get_preset(os.environ.get(ENV_VAR, DEFAULT_PRESET))
    return _active


def set_active(model: "str | FrictionModel") -> FrictionModel:
    """Select a preset by name, or install a derived model, for this process."""
    global _active
    _active = model if isinstance(model, FrictionModel) else get_preset(model)
    return _active


def comparison_table() -> str:
    """Every preset side by side, so a result is never quoted from one alone."""
    lines = [
        "Friction presets (latency is one-way, per leg)",
        "─" * 74,
    ]
    for name in sorted(PRESETS):
        lines.append("  " + PRESETS[name].describe())
    lines += [
        "─" * 74,
        f"active: {active().name}   (${ENV_VAR} or src.friction.set_active)",
        "",
        "'zero' removes the latency but not the book walk, so it does not restore",
        "the old tautological 100% -- nothing here does. Any preset decouples the",
        "fill from the signal, which is the point -- see the module docstring for",
        "the measured effect by size.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(comparison_table())
    print()
    for nm in sorted(PRESETS):
        print(f"{nm}:\n  {PRESETS[nm].description}\n")
