"""Single source of truth for exchange trading fees.

Why this module exists
---------------------
Three divergent fee models used to coexist in this repository:

    backtest/engine.py:69                      flat 0.05% per leg  -> 10.0 bps round-trip
    src/execution_simulator.py:7-9             binance 0.04% / kraken 0.10% -> 14.0 bps
    cpp_engine/src/execution_manager.cpp:9-10  the same 14.0 bps

The flat 10 bps model was the only one that reproduced the README's headline
return, and it was used nowhere else in the codebase.

Fee choice matters here, but not in the way you would expect. The entry gate in
backtest/engine.py is `margin > fee` and the booked PnL is `margin - fee`, using
the same fee model, so a trade that cannot cover its fee is never taken.
Consequence: raising fees does not make the strategy lose money, it makes it
trade less. Measured on data/raw/ 2024-03-01 BTCUSDT vs XBT-USD:

    preset          round-trip   trades    gross_bps   net_bps   return%
    zero                0.0bps  540,985       4.84      4.84    +162.4
    institutional       3.0bps  307,447       7.28      4.28     +81.9
    legacy_flat        10.0bps   58,664      13.42      3.42     +12.6
    retail             14.0bps   22,300      15.79      1.78      +2.5

Every preset is positive by construction, and `gross_bps` rises in lockstep with
the fee because the gate is selecting progressively wider spreads -- not because
the signal improved. `net_bps` is just `gross_bps - round_trip`. So the fee
preset sets the strategy's *selectivity*, and the headline return mostly measures
how many trades the gate let through.

This is why no preset's return should be quoted on its own, and why the fee work
does not substitute for breaking the gate/PnL tautology. Until the gate uses a
criterion independent of the PnL formula, the backtest cannot report a loss under
any fee schedule.

Any code path that charges a fee must read it from here, and any reported
result should name the preset it used.

Usage
-----
    from src.fees import active, get_preset, set_active

    fees = active()                       # env-or-default preset
    cost = fees.taker_cost(qty=0.01, buy_venue="binance", buy_price=50_000.0,
                           sell_venue="kraken", sell_price=50_010.0)

    set_active("retail")                  # switch at runtime
    for name in PRESETS: ...              # sweep every preset

Select a preset without editing code via the environment:

    CROSSFLUX_FEE_PRESET=retail python -m backtest.engine

Regenerate the C++ side after changing any rate here:

    python -m src.fees --emit-cpp-header
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Mapping

__all__ = [
    "FeeModel", "PRESETS", "DEFAULT_PRESET",
    "get_preset", "active", "set_active", "comparison_table", "emit_cpp_header",
]

DEFAULT_PRESET = "institutional"
ENV_VAR = "CROSSFLUX_FEE_PRESET"


@dataclass(frozen=True)
class FeeModel:
    """Per-venue fee schedule.

    Rates are fractions of notional, not percentages and not basis points:
    0.0001 == 0.01% == 1.0 bps.

    An unmodelled venue resolves to ``fallback_taker``, which defaults to the
    *most expensive* modelled venue. That direction is deliberate -- a missing
    venue should never make a strategy look cheaper than it is, which is exactly
    how the old ``VENUE_FEES.get(venue, 0.0004)`` fallbacks let divergence in.
    """

    name: str
    taker: Mapping[str, float]
    description: str = ""
    maker: Mapping[str, float] = field(default_factory=dict)
    _fallback_taker: float | None = None

    def __post_init__(self) -> None:
        if not self.taker:
            raise ValueError(f"FeeModel {self.name!r}: taker schedule cannot be empty.")
        for venue, rate in self.taker.items():
            if rate < 0.0:
                raise ValueError(
                    f"FeeModel {self.name!r}: negative taker rate {rate!r} for {venue!r}. "
                    "Rebates are not supported; model them explicitly if needed."
                )
            if rate > 0.01:
                raise ValueError(
                    f"FeeModel {self.name!r}: taker rate {rate!r} for {venue!r} exceeds 1%. "
                    "Rates are fractions of notional (0.0001 == 1 bp), not percentages."
                )

    # ── rates ────────────────────────────────────────────────────────────────
    @property
    def fallback_taker(self) -> float:
        if self._fallback_taker is not None:
            return self._fallback_taker
        return max(self.taker.values())

    def taker_rate(self, venue: str) -> float:
        return self.taker.get(str(venue).lower(), self.fallback_taker)

    def maker_rate(self, venue: str) -> float:
        """Maker rates are intentionally unmodelled.

        Both legs of a cross-venue arbitrage cross the spread, so no leg ever
        earns the maker rate. Populate ``maker`` in PRESETS only if passive
        quoting is actually implemented -- quoting a maker rate for a taker
        strategy understates cost.
        """
        try:
            return self.maker[str(venue).lower()]
        except KeyError:
            raise KeyError(
                f"FeeModel {self.name!r} does not model maker fees for {venue!r}. "
                "This strategy takes on both legs, so the maker rate never applies. "
                "See FeeModel.maker_rate."
            ) from None

    # ── costs ────────────────────────────────────────────────────────────────
    def taker_cost(
        self,
        qty: float,
        buy_venue: str,
        buy_price: float,
        sell_venue: str,
        sell_price: float,
    ) -> float:
        """Total two-leg taker fee in quote currency, both legs the same size.

        Charged on each leg's own notional at that leg's own venue rate. The
        legacy ``TAKER_FEE_RATE * qty * (buy_price + sell_price)`` form is the
        special case where both venues share one rate.
        """
        return self.taker_cost_legs(qty, buy_venue, buy_price,
                                    qty, sell_venue, sell_price)

    def taker_cost_legs(
        self,
        buy_qty: float,
        buy_venue: str,
        buy_price: float,
        sell_qty: float,
        sell_venue: str,
        sell_price: float,
    ) -> float:
        """Two-leg taker fee when the legs filled *different* quantities.

        Once fills can be partial, the legs of one arbitrage no longer match:
        the buy side may absorb 1.0 BTC while the sell side manages 0.6. Fees
        are owed on what each leg actually executed, not on the matched
        quantity -- an over-filled leg costs its fee in full and earns no
        offsetting revenue, which is precisely what makes legging expensive.

        ``taker_cost`` is the equal-quantity special case. Accepts scalars or
        numpy arrays; the expression is pure arithmetic.
        """
        return (
            buy_qty * self.taker_rate(buy_venue) * buy_price
            + sell_qty * self.taker_rate(sell_venue) * sell_price
        )

    def round_trip_rate(self, buy_venue: str, sell_venue: str) -> float:
        """Sum of both legs' rates -- the break-even gross spread as a fraction."""
        return self.taker_rate(buy_venue) + self.taker_rate(sell_venue)

    def round_trip_bps(self, buy_venue: str, sell_venue: str) -> float:
        return self.round_trip_rate(buy_venue, sell_venue) * 10_000.0

    # ── derivation ───────────────────────────────────────────────────────────
    def with_rates(self, **venue_rates: float) -> "FeeModel":
        """Copy with specific venue taker rates replaced, for sensitivity runs."""
        merged = dict(self.taker)
        merged.update({k.lower(): float(v) for k, v in venue_rates.items()})
        suffix = ",".join(f"{k}={v}" for k, v in sorted(venue_rates.items()))
        return replace(self, name=f"{self.name}+({suffix})", taker=merged)

    def describe(self) -> str:
        legs = "  ".join(f"{v}={r * 1e4:.1f}bps" for v, r in sorted(self.taker.items()))
        rt = self.round_trip_bps("binance", "kraken")
        return f"{self.name:<14} {legs}   round-trip={rt:.1f}bps"


# ─────────────────────────────────────────────────────────────────────────────
# Presets
#
# Rates are fractions of notional: 0.0001 == 0.01% == 1.0 bps.
# These are not verified against any live account -- confirm against your own
# volume tier before treating a backtest as tradeable. Published schedules
# change and vary by tier, asset and settlement currency.
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, FeeModel] = {
    "institutional": FeeModel(
        name="institutional",
        taker={"binance": 0.0001, "kraken": 0.0002},
        description=(
            "VIP / institutional tier. 3.0 bps round-trip. The most permissive "
            "preset with a real fee schedule: the gate admits ~307k trades on the "
            "2024-03-01 sample and reports +81.9%, versus +2.5% under retail. That "
            "spread is selectivity, not alpha -- see the module docstring."
        ),
    ),
    "retail": FeeModel(
        name="retail",
        taker={"binance": 0.0004, "kraken": 0.0010},
        description=(
            "Standard non-VIP taker rates as previously hardcoded in "
            "execution_simulator.py and execution_manager.cpp. 14.0 bps "
            "round-trip. The strictest realistic preset: ~22k trades, +2.5%, and "
            "only 1.78 bps net per trade -- thin enough that any unmodelled cost "
            "(funding, withdrawal, rebalancing, partial fills) plausibly erases it."
        ),
    ),
    "zero": FeeModel(
        name="zero",
        taker={"binance": 0.0, "kraken": 0.0},
        description=(
            "Frictionless. Diagnostic only: isolates signal quality from cost, and "
            "shows the unfiltered edge is just 4.84 bps gross per trade. Never "
            "report a return from this preset as a result."
        ),
    ),
    "legacy_flat": FeeModel(
        name="legacy_flat",
        taker={"binance": 0.0005, "kraken": 0.0005},
        description=(
            "The flat 0.05%/leg constant formerly at backtest/engine.py:69, kept "
            "only so historical README numbers stay reproducible. Not a real "
            "schedule -- Kraken's taker fee is not 5 bps."
        ),
    ),
}


def get_preset(name: str) -> FeeModel:
    try:
        return PRESETS[str(name).lower()]
    except KeyError:
        raise KeyError(
            f"Unknown fee preset {name!r}. Available: {', '.join(sorted(PRESETS))}."
        ) from None


_active: FeeModel | None = None


def active() -> FeeModel:
    """Currently selected model: explicit ``set_active`` > ``$CROSSFLUX_FEE_PRESET`` > default."""
    global _active
    if _active is not None:
        return _active
    _active = get_preset(os.environ.get(ENV_VAR, DEFAULT_PRESET))
    return _active


def set_active(model: "str | FeeModel") -> FeeModel:
    """Select a preset by name, or install a derived FeeModel, for this process."""
    global _active
    _active = model if isinstance(model, FeeModel) else get_preset(model)
    return _active


def comparison_table() -> str:
    """Every preset's round-trip cost, so a result is never quoted from one alone.

    Deliberately does not label any preset "survives" or "fails". An earlier
    version did, based on comparing round-trip fee against a mean gross edge --
    that comparison is invalid here, because the entry gate re-selects trades
    against whatever fee is active, so the mean gross edge is a function of the
    fee rather than a fixed property of the market. Measured trade counts are
    shown instead; they are what actually changes.
    """
    # Measured on data/raw/ 2024-03-01 BTCUSDT vs XBT-USD via
    # `python -m backtest.compare_fee_presets`. Indicative, not a contract.
    measured = {
        "institutional": (307_447, 81.9),
        "legacy_flat":   (58_664, 12.6),
        "retail":        (22_300, 2.5),
        "zero":          (540_985, 162.4),
    }
    rows = [
        "preset          binance    kraken   round-trip     trades   return%",
        "-" * 66,
    ]
    for name in ("institutional", "legacy_flat", "retail", "zero"):
        m = PRESETS[name]
        trades, ret = measured[name]
        marker = "  <- default" if name == DEFAULT_PRESET else ""
        rows.append(
            f"{name:<14} {m.taker_rate('binance') * 1e4:>6.1f}bps "
            f"{m.taker_rate('kraken') * 1e4:>7.1f}bps "
            f"{m.round_trip_bps('binance', 'kraken'):>10.1f}bps "
            f"{trades:>10,} {ret:>8.1f}{marker}"
        )
    rows += [
        "-" * 66,
        "Fees set selectivity, not profitability: the gate is `margin > fee`, so a",
        "higher fee removes trades rather than causing losses. Every preset is",
        "positive by construction. Do not read the return column as alpha.",
    ]
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# C++ header generation
#
# The C++ engine cannot import this module, so its rates are generated from it.
# Editing cpp_engine/include/fee_config.hpp by hand reintroduces exactly the
# drift this module exists to prevent.
# ─────────────────────────────────────────────────────────────────────────────

_CPP_TEMPLATE = """// GENERATED FILE -- DO NOT EDIT BY HAND.
//
// Regenerate with:  python -m src.fees --emit-cpp-header
// Source of truth:  src/fees.py
//
// Editing this file by hand reintroduces the divergent-fee-model bug it exists
// to prevent. Change the rates in src/fees.py and regenerate.
#ifndef CROSSFLUX_FEE_CONFIG_HPP
#define CROSSFLUX_FEE_CONFIG_HPP

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace crossflux {{
namespace fees {{

// Rates are fractions of notional: 0.0001 == 0.01% == 1.0 bps.
struct Schedule {{
    std::string_view name;
    double binance_taker;
    double kraken_taker;
    double fallback_taker;   // conservative: the most expensive modelled venue
}};

{schedules}

inline constexpr Schedule kDefaultSchedule = {default_ref};

// src/fees.py lowercases the requested preset name; match that here so the two
// sides accept exactly the same inputs.
inline std::string lowered(const char* s) {{
    std::string out{{s}};
    for (char& c : out) {{
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }}
    return out;
}}

// Preset selection mirrors src/fees.py: $CROSSFLUX_FEE_PRESET, else the default.
// Resolved once on first use; not intended to change mid-run.
//
// An unrecognised preset name aborts rather than falling back. src/fees.py
// raises KeyError in the same situation, and silently falling back would resolve
// a typo to the cheapest schedule -- i.e. fail in the direction that flatters
// the result. A misconfigured fee assumption is not a recoverable condition.
inline const Schedule& active() {{
    static const Schedule& selected = []() -> const Schedule& {{
        const char* env = std::getenv("CROSSFLUX_FEE_PRESET");
        if (env != nullptr && *env != '\\0') {{
            const std::string want = lowered(env);
{dispatch}
            std::fprintf(stderr,
                "[fees] FATAL: unknown CROSSFLUX_FEE_PRESET=\\"%s\\". "
                "Valid presets: {preset_names}. "
                "Refusing to fall back -- a typo would silently pick a cheaper "
                "fee schedule. See src/fees.py.\\n", env);
            std::abort();
        }}
        return kDefaultSchedule;
    }}();
    return selected;
}}

inline double taker_fee_for(std::string_view exchange) {{
    const Schedule& s = active();
    if (exchange == "binance") return s.binance_taker;
    if (exchange == "kraken")  return s.kraken_taker;
    return s.fallback_taker;
}}

}}  // namespace fees
}}  // namespace crossflux

#endif  // CROSSFLUX_FEE_CONFIG_HPP
"""


def _cpp_ident(name: str) -> str:
    return "k" + "".join(part.capitalize() for part in name.split("_"))


def emit_cpp_header(path: "str | os.PathLike[str] | None" = None) -> str:
    """Write cpp_engine/include/fee_config.hpp from PRESETS. Returns the path."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "cpp_engine", "include", "fee_config.hpp")
    path = os.path.normpath(str(path))

    schedules, dispatch = [], []
    for name, m in PRESETS.items():
        ident = _cpp_ident(name)
        schedules.append(
            f'inline constexpr Schedule {ident}{{"{name}", '
            f"{m.taker_rate('binance'):.6f}, {m.taker_rate('kraken'):.6f}, "
            f"{m.fallback_taker:.6f}}};"
        )
        dispatch.append(f'            if (want == "{name}") return {ident};')

    body = _CPP_TEMPLATE.format(
        schedules="\n".join(schedules),
        default_ref=_cpp_ident(DEFAULT_PRESET),
        dispatch="\n".join(dispatch),
        preset_names=", ".join(PRESETS),
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
        for name in PRESETS:
            print(f"  {PRESETS[name].describe()}")
