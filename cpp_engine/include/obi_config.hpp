// GENERATED FILE -- DO NOT EDIT BY HAND.
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

namespace crossflux {
namespace obi {

// Matches OrderBookSnapshot<N> default N=10 in models.hpp.
inline constexpr std::size_t kMaxLevels = 10;

// weights[i] scales level i (i == 0 is the touch). Entries beyond `depth` are
// zero, so `depth` is the effective number of levels consumed.
//
// Weights are non-negative by construction (validated in src/obi_weights.py).
// That guarantee is what bounds the normalized signal to [-1, 1], which in turn
// is what lets calculate_obi_delta's [-1, 1] assert hold.
struct Profile {
    std::string_view                     name;
    std::array<double, kMaxLevels>       weights;
    std::size_t                          depth;

    // Non-owning view of just the meaningful weights.
    [[nodiscard]] constexpr std::span<const double> span() const noexcept {
        return std::span<const double>{weights.data(), depth};
    }
};

inline constexpr Profile kFlat{"flat", {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}, 5};
inline constexpr Profile kDecay50{"decay_50", {1, 0.5, 0.25, 0.125, 0.0625, 0, 0, 0, 0, 0}, 5};
inline constexpr Profile kDecay75{"decay_75", {1, 0.75, 0.5625, 0.421875, 0.31640625, 0, 0, 0, 0, 0}, 5};
inline constexpr Profile kL1Only{"l1_only", {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1};

inline constexpr Profile kDefaultProfile = kFlat;

// src/obi_weights.py lowercases the requested name; match that here so both
// sides accept exactly the same inputs.
inline std::string lowered(const char* s) {
    std::string out{s};
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

// Profile selection mirrors src/obi_weights.py: $CROSSFLUX_OBI_PROFILE, else
// the default. Resolved once on first use; not intended to change mid-run.
//
// An unrecognised name aborts rather than falling back. src/obi_weights.py
// raises KeyError in the same situation. Falling back would resolve a typo to
// the default profile and silently report results for a signal nobody selected.
inline const Profile& active() {
    static const Profile& selected = []() -> const Profile& {
        const char* env = std::getenv("CROSSFLUX_OBI_PROFILE");
        if (env != nullptr && *env != '\0') {
            const std::string want = lowered(env);
            if (want == "flat") return kFlat;
            if (want == "decay_50") return kDecay50;
            if (want == "decay_75") return kDecay75;
            if (want == "l1_only") return kL1Only;
            std::fprintf(stderr,
                "[obi] FATAL: unknown CROSSFLUX_OBI_PROFILE=\"%s\". "
                "Valid profiles: flat, decay_50, decay_75, l1_only. "
                "Refusing to fall back -- a typo would silently report results "
                "for an unselected signal. See src/obi_weights.py.\n", env);
            std::abort();
        }
        return kDefaultProfile;
    }();
    return selected;
}

// Convenience: weights of the active profile, ready for calculate_weighted_obi.
inline std::span<const double> active_weights() {
    return active().span();
}

}  // namespace obi
}  // namespace crossflux

#endif  // CROSSFLUX_OBI_CONFIG_HPP
