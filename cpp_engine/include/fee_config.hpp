// GENERATED FILE -- DO NOT EDIT BY HAND.
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

namespace crossflux {
namespace fees {

// Rates are fractions of notional: 0.0001 == 0.01% == 1.0 bps.
struct Schedule {
    std::string_view name;
    double binance_taker;
    double kraken_taker;
    double fallback_taker;   // conservative: the most expensive modelled venue
};

inline constexpr Schedule kInstitutional{"institutional", 0.000100, 0.000200, 0.000200};
inline constexpr Schedule kRetail{"retail", 0.000400, 0.001000, 0.001000};
inline constexpr Schedule kZero{"zero", 0.000000, 0.000000, 0.000000};
inline constexpr Schedule kLegacyFlat{"legacy_flat", 0.000500, 0.000500, 0.000500};

inline constexpr Schedule kDefaultSchedule = kInstitutional;

// src/fees.py lowercases the requested preset name; match that here so the two
// sides accept exactly the same inputs.
inline std::string lowered(const char* s) {
    std::string out{s};
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

// Preset selection mirrors src/fees.py: $CROSSFLUX_FEE_PRESET, else the default.
// Resolved once on first use; not intended to change mid-run.
//
// An unrecognised preset name aborts rather than falling back. src/fees.py
// raises KeyError in the same situation, and silently falling back would resolve
// a typo to the cheapest schedule -- i.e. fail in the direction that flatters
// the result. A misconfigured fee assumption is not a recoverable condition.
inline const Schedule& active() {
    static const Schedule& selected = []() -> const Schedule& {
        const char* env = std::getenv("CROSSFLUX_FEE_PRESET");
        if (env != nullptr && *env != '\0') {
            const std::string want = lowered(env);
            if (want == "institutional") return kInstitutional;
            if (want == "retail") return kRetail;
            if (want == "zero") return kZero;
            if (want == "legacy_flat") return kLegacyFlat;
            std::fprintf(stderr,
                "[fees] FATAL: unknown CROSSFLUX_FEE_PRESET=\"%s\". "
                "Valid presets: institutional, retail, zero, legacy_flat. "
                "Refusing to fall back -- a typo would silently pick a cheaper "
                "fee schedule. See src/fees.py.\n", env);
            std::abort();
        }
        return kDefaultSchedule;
    }();
    return selected;
}

inline double taker_fee_for(std::string_view exchange) {
    const Schedule& s = active();
    if (exchange == "binance") return s.binance_taker;
    if (exchange == "kraken")  return s.kraken_taker;
    return s.fallback_taker;
}

}  // namespace fees
}  // namespace crossflux

#endif  // CROSSFLUX_FEE_CONFIG_HPP
