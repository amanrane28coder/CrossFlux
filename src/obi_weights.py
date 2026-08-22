"""Single source of truth for multi-level OBI weighting profiles.

Why this module exists
----------------------
The order book imbalance signal originally summed volume across levels with no
weighting: every level counted the same, whether it sat at the touch or five
ticks away. Weighting deeper levels less is a reasonable idea -- resting size far
from the mid is less informative about immediate pressure, and easier to spoof.

This module holds the weight vectors, so a profile can be swapped without
editing code in two languages. It follows the same pattern as ``src/fees.py``:
Python is the source of truth, the C++ header is generated from it, and an
unknown profile name is a hard error rather than a silent fallback.

    python -m src.obi_weights --emit-cpp-header

What this is NOT
----------------
This is *not* MLOFI, despite that name being attached to the request that
prompted it. MLOFI -- multi-level order flow imbalance, following Cont, Kukanov
and Stoikov and its multi-level extensions -- is computed from the *change* in
depth between consecutive book updates:

    OFI_i(t) = ΔBidDepth_i(t) - ΔAskDepth_i(t)

It measures order *flow*: arrivals, cancellations and executions. What this
module weights is a *static snapshot* of resting depth. The correct name for
that is weighted multi-level OBI, and calling it MLOFI in an interview would not
survive a follow-up question.

Real MLOFI is currently blocked on data, not on code. It needs successive book
deltas; ``data/raw/`` holds only 5-level snapshots, and the
``binance_incremental_book_L2`` download is still an unconfirmed partial file.

Normalization is not optional
-----------------------------
The weighted signal is normalized by the weighted *total* volume, keeping it in
[-1, 1] exactly like unweighted OBI. This is load-bearing, not stylistic:

  * ``calculate_obi_delta`` asserts both inputs lie in [-1, 1]
    (cpp_engine/include/signals.hpp:173-175). An unnormalized sum in volume
    units trips that assert in Debug builds.
  * ``DEFAULT_DELTA_THRESHOLD = 0.3`` is calibrated against a bounded delta in
    [-2, 2]. Against raw BTC volume sums, 0.3 is satisfied by nearly every tick,
    which silently converts the entry gate into a pass-through -- the engine
    would look far more active while actually filtering nothing.

Because weights are non-negative and volumes are non-negative, the numerator is
bounded in absolute value by the denominator, so the range is guaranteed by
construction rather than by clamping.

The interval is closed and both endpoints are reachable: an empty book side
gives numerator == denominator, and so does a side whose weighted volume falls
below the other side's ULP (``1e9 + 1e-9 == 1e9``, so the two accumulators end
up as the same double). The second route needs ~16 orders of magnitude between
the sides, which no real book has, but the downstream assert is inclusive for
this reason and the bound tests assert ``<=``, not ``<``.

A caveat on measurement
-----------------------
Switching profiles changes which ticks fire, but the backtest cannot currently
tell you whether any profile is *better*. The entry gate is ``margin > fee`` and
the booked PnL is ``margin - fee`` against the same fee model, so every signal
reports a 97-99% win rate regardless of quality. See ``src/fees.py``. Treat a
profile comparison as a description of selectivity until that is fixed.

Usage
-----
    from src.obi_weights import active, get_profile, set_active

    prof = active()                        # env-or-default profile
    w = prof.weights                       # (1.0, 1.0, 1.0, 1.0, 1.0) for flat

    set_active("decay_50")                 # switch at runtime
    for name in PROFILES: ...              # sweep every profile

Select a profile without editing code:

    CROSSFLUX_OBI_PROFILE=decay_50 python -m backtest.engine
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "WeightProfile", "PROFILES", "DEFAULT_PROFILE", "MAX_LEVELS",
    "get_profile", "active", "set_active", "comparison_table", "emit_cpp_header",
]

DEFAULT_PROFILE = "flat"
ENV_VAR = "CROSSFLUX_OBI_PROFILE"

# Matches OrderBookSnapshot<N> default N=10 in cpp_engine/include/models.hpp.
# A profile may be shorter; trailing levels then carry zero weight.
MAX_LEVELS = 10


@dataclass(frozen=True)
class WeightProfile:
    """Per-level weights for the multi-level OBI signal.

    ``weights[i]`` scales level ``i``, where ``i = 0`` is the touch (best bid /
    best ask). Levels beyond ``len(weights)`` carry zero weight, so the length
    of the vector *is* the effective depth.

    Weights are relative. Scaling the whole vector by a constant leaves the
    normalized signal unchanged, so ``(1.0, 0.5)`` and ``(2.0, 1.0)`` are the
    same profile in effect.
    """

    name: str
    weights: tuple[float, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(
                f"WeightProfile {self.name!r}: weights cannot be empty. "
                "At least the touch level must carry weight."
            )
        if len(self.weights) > MAX_LEVELS:
            raise ValueError(
                f"WeightProfile {self.name!r}: {len(self.weights)} weights exceeds "
                f"MAX_LEVELS={MAX_LEVELS} (OrderBookSnapshot<10>). Deeper levels "
                "cannot be read from the snapshot type."
            )
        for i, w in enumerate(self.weights):
            if w < 0.0:
                raise ValueError(
                    f"WeightProfile {self.name!r}: negative weight {w!r} at level {i}. "
                    "Negative weights break the [-1, 1] normalization bound that "
                    "calculate_obi_delta's assert depends on."
                )
        if sum(self.weights) <= 0.0:
            raise ValueError(
                f"WeightProfile {self.name!r}: all weights are zero, which makes the "
                "signal identically 0.0."
            )

    @property
    def depth(self) -> int:
        """Effective number of levels consumed."""
        return len(self.weights)

    def weight(self, level: int) -> float:
        """Weight for ``level``, or 0.0 beyond the profile's depth."""
        if level < 0:
            raise ValueError(f"weight: level must be >= 0, got {level!r}.")
        return self.weights[level] if level < len(self.weights) else 0.0

    def padded(self) -> tuple[float, ...]:
        """Weights zero-padded to MAX_LEVELS, for fixed-size C++ arrays."""
        return self.weights + (0.0,) * (MAX_LEVELS - len(self.weights))

    def normalized(self) -> tuple[float, ...]:
        """Weights rescaled so the touch level is 1.0, for display only."""
        head = self.weights[0]
        if head == 0.0:
            return self.weights
        return tuple(w / head for w in self.weights)

    def describe(self) -> str:
        ws = " ".join(f"{w:.4g}" for w in self.weights)
        return f"{self.name:<10} depth={self.depth}  [{ws}]"


# ─────────────────────────────────────────────────────────────────────────────
# Profiles
#
# DEFAULT_PROFILE is "flat" deliberately. Flat weighting is what the engine has
# always done, so the default keeps every existing backtest number reproducible.
# A weighting scheme should have to be asked for, not arrive silently in a
# result someone already published.
# ─────────────────────────────────────────────────────────────────────────────

PROFILES: dict[str, WeightProfile] = {
    "flat": WeightProfile(
        name="flat",
        weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        description=(
            "Every level counts equally across the top 5 -- the engine's original "
            "behaviour and the default. Reproduces calculate_obi() exactly for any "
            "book with at most 5 valid levels per side, which covers all of "
            "data/raw/ (Tardis book_snapshot_5). Keep as the baseline for any "
            "weighted-vs-unweighted comparison."
        ),
    ),
    "decay_50": WeightProfile(
        name="decay_50",
        weights=(1.0, 0.5, 0.25, 0.125, 0.0625),
        description=(
            "Exponential decay, ratio 0.5 per level. The profile the upgrade "
            "request specified. Aggressive: level 5 carries 6.25% of the touch's "
            "weight, so the signal is dominated by the top two levels and behaves "
            "close to l1_only on most books."
        ),
    ),
    "decay_75": WeightProfile(
        name="decay_75",
        weights=(1.0, 0.75, 0.5625, 0.421875, 0.31640625),
        description=(
            "Exponential decay, ratio 0.75 per level. Gentler alternative to "
            "decay_50: still discounts depth, but level 5 retains 32% weight, so "
            "deeper liquidity continues to matter. Worth comparing against "
            "decay_50 before assuming faster decay is better."
        ),
    ),
    "l1_only": WeightProfile(
        name="l1_only",
        weights=(1.0,),
        description=(
            "Top of book only. Included as a genuine ablation: the upgrade request "
            "assumed this was the engine's existing behaviour, and it was not. "
            "Selecting it lets you measure what depth actually contributes instead "
            "of assuming it."
        ),
    ),
}


def get_profile(name: str) -> WeightProfile:
    try:
        return PROFILES[str(name).lower()]
    except KeyError:
        raise KeyError(
            f"Unknown OBI weight profile {name!r}. "
            f"Available: {', '.join(sorted(PROFILES))}."
        ) from None


_active: WeightProfile | None = None


def active() -> WeightProfile:
    """Selected profile: explicit ``set_active`` > ``$CROSSFLUX_OBI_PROFILE`` > default."""
    global _active
    if _active is not None:
        return _active
    _active = get_profile(os.environ.get(ENV_VAR, DEFAULT_PROFILE))
    return _active


def set_active(profile: "str | WeightProfile") -> WeightProfile:
    """Select a profile by name, or install a custom one, for this process."""
    global _active
    if isinstance(profile, WeightProfile):
        _active = profile
    else:
        _active = get_profile(profile)
    return _active


def from_decay(ratio: float, depth: int = 5, name: str | None = None) -> WeightProfile:
    """Build an exponential-decay profile: ``weights[i] = ratio ** i``.

    Provided for sweeps over the decay ratio. Not registered in PROFILES -- a
    result should name a registered profile, or state the ratio explicitly.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(
            f"from_decay: ratio must be in (0.0, 1.0], got {ratio!r}. "
            "A ratio above 1.0 would weight deep liquidity more than the touch."
        )
    if not (1 <= depth <= MAX_LEVELS):
        raise ValueError(f"from_decay: depth must be in [1, {MAX_LEVELS}], got {depth!r}.")
    return WeightProfile(
        name=name or f"decay({ratio:g})x{depth}",
        weights=tuple(ratio ** i for i in range(depth)),
        description=f"Generated: weights[i] = {ratio:g} ** i, depth={depth}.",
    )


def weighted_obi_from_volumes(
    bid_volumes: Sequence[float],
    ask_volumes: Sequence[float],
    profile: WeightProfile | None = None,
) -> float:
    """Normalized weighted OBI from raw volume sequences.

    Kept here, free of any snapshot type, so the identical arithmetic can be
    reused by tests, the backtester and the parity check against C++ without
    constructing an OrderBookSnapshot.

    Depth consumed is ``min(len(bid_volumes), len(ask_volumes), profile.depth)``
    -- the shallower side wins, matching the C++ implementation. Returns 0.0 for
    a degenerate book, matching calculate_obi's sentinel.
    """
    prof = profile if profile is not None else active()
    depth = min(len(bid_volumes), len(ask_volumes), prof.depth)

    num = 0.0
    den = 0.0
    for i in range(depth):
        w = prof.weights[i]
        num += w * (bid_volumes[i] - ask_volumes[i])
        den += w * (bid_volumes[i] + ask_volumes[i])

    if den == 0.0:
        return 0.0
    return num / den


def comparison_table() -> str:
    """Every profile's weights side by side, so a result names its profile."""
    rows = [
        "profile      depth        L1      L2      L3      L4      L5   share(L1+L2)",
        "-" * 76,
    ]
    for name in ("flat", "decay_75", "decay_50", "l1_only"):
        p = PROFILES[name]
        cells = "".join(f"{p.weight(i):>8.4g}" for i in range(5))
        total = sum(p.weights)
        share = (p.weight(0) + p.weight(1)) / total * 100.0
        marker = "  <- default" if name == DEFAULT_PROFILE else ""
        rows.append(f"{name:<12} {p.depth:>5} {cells} {share:>13.1f}%{marker}")
    rows += [
        "-" * 76,
        "share(L1+L2) is the fraction of total weight in the top two levels: how",
        "much of the signal is really top-of-book. decay_50 puts 77% there, against",
        "flat's 40%, so it sits closer to l1_only than to flat despite reading five",
        "levels. Worth knowing before reading a decay_50 result as a depth signal.",
        "",
        "Weights are relative; scaling a profile by a constant does not change the",
        "normalized signal. Switching profiles changes selectivity, and cannot",
        "currently be read as an improvement -- see the module docstring.",
    ]
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# C++ header generation
#
# The C++ engine cannot import this module, so its weights are generated from
# it. Editing cpp_engine/include/obi_config.hpp by hand reintroduces exactly the
# two-language drift this module exists to prevent -- the same drift that let
# three different fee models coexist before src/fees.py.
# ─────────────────────────────────────────────────────────────────────────────

_CPP_TEMPLATE = """// GENERATED FILE -- DO NOT EDIT BY HAND.
//
// Regenerate with:  python -m src.obi_weights --emit-cpp-header
// Source of truth:  src/obi_weights.py
//
// Per-level weights for the multi-level OBI signal. NOT MLOFI: this weights a
// static depth snapshot, whereas MLOFI is built from successive book deltas.
// See src/obi_weights.py for why the distinction matters.
#ifndef CROSSFLUX_OBI_CONFIG_HPP
#define CROSSFLUX_OBI_CONFIG_HPP

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>

namespace crossflux {{
namespace obi {{

// Matches OrderBookSnapshot<N> default N=10 in models.hpp.
inline constexpr std::size_t kMaxLevels = {max_levels};

// weights[i] scales level i (i == 0 is the touch). Entries beyond `depth` are
// zero, so `depth` is the effective number of levels consumed.
//
// Weights are non-negative by construction (validated in src/obi_weights.py).
// That guarantee is what bounds the normalized signal to [-1, 1], which in turn
// is what lets calculate_obi_delta's [-1, 1] assert hold.
struct Profile {{
    std::string_view                     name;
    std::array<double, kMaxLevels>       weights;
    std::size_t                          depth;

    // Non-owning view of just the meaningful weights.
    [[nodiscard]] constexpr std::span<const double> span() const noexcept {{
        return std::span<const double>{{weights.data(), depth}};
    }}
}};

{profiles}

inline constexpr Profile kDefaultProfile = {default_ref};

// src/obi_weights.py lowercases the requested name; match that here so both
// sides accept exactly the same inputs.
inline std::string lowered(const char* s) {{
    std::string out{{s}};
    for (char& c : out) {{
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }}
    return out;
}}

// Profile selection mirrors src/obi_weights.py: $CROSSFLUX_OBI_PROFILE, else
// the default. Resolved once on first use; not intended to change mid-run.
//
// An unrecognised name aborts rather than falling back. src/obi_weights.py
// raises KeyError in the same situation. Falling back would resolve a typo to
// the default profile and silently report results for a signal nobody selected.
inline const Profile& active() {{
    static const Profile& selected = []() -> const Profile& {{
        const char* env = std::getenv("CROSSFLUX_OBI_PROFILE");
        if (env != nullptr && *env != '\\0') {{
            const std::string want = lowered(env);
{dispatch}
            std::fprintf(stderr,
                "[obi] FATAL: unknown CROSSFLUX_OBI_PROFILE=\\"%s\\". "
                "Valid profiles: {profile_names}. "
                "Refusing to fall back -- a typo would silently report results "
                "for an unselected signal. See src/obi_weights.py.\\n", env);
            std::abort();
        }}
        return kDefaultProfile;
    }}();
    return selected;
}}

// Convenience: weights of the active profile, ready for calculate_weighted_obi.
inline std::span<const double> active_weights() {{
    return active().span();
}}

}}  // namespace obi
}}  // namespace crossflux

#endif  // CROSSFLUX_OBI_CONFIG_HPP
"""


def _cpp_ident(name: str) -> str:
    return "k" + "".join(part.capitalize() for part in name.split("_"))


def emit_cpp_header(path: "str | os.PathLike[str] | None" = None) -> str:
    """Write cpp_engine/include/obi_config.hpp from PROFILES. Returns the path."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "cpp_engine", "include", "obi_config.hpp")
    path = os.path.normpath(str(path))

    profiles, dispatch = [], []
    for name, p in PROFILES.items():
        ident = _cpp_ident(name)
        ws = ", ".join(f"{w:.10g}" for w in p.padded())
        profiles.append(
            f'inline constexpr Profile {ident}{{"{name}", {{{ws}}}, {p.depth}}};'
        )
        dispatch.append(f'            if (want == "{name}") return {ident};')

    body = _CPP_TEMPLATE.format(
        max_levels=MAX_LEVELS,
        profiles="\n".join(profiles),
        default_ref=_cpp_ident(DEFAULT_PROFILE),
        dispatch="\n".join(dispatch),
        profile_names=", ".join(PROFILES),
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


if __name__ == "__main__":
    import sys

    if "--emit-cpp-header" in sys.argv:
        print(f"wrote {emit_cpp_header()}")
    else:
        print(comparison_table())
        print()
        for name in PROFILES:
            print(f"  {PROFILES[name].describe()}")
