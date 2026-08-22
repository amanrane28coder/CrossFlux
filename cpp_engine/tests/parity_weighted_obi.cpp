// Cross-language parity harness for the weighted OBI signal.
//
// Reads books from stdin as exact hex doubles, prints one weighted OBI per
// (book, profile) back as an exact hex double. The Python side generates the
// books and compares the output bit-for-bit against src/obi_weights.py.
//
// Hex floats (%a / float.hex()) both ways: a decimal round-trip would make a
// real 1-ULP divergence indistinguishable from the formatting, which is exactly
// the size of disagreement this harness exists to detect.
//
// Do not run this by hand -- cpp_engine/tests/run_parity_check.py builds it with
// the project's real flags, generates the books, and interprets the result:
//
//   python3 cpp_engine/tests/run_parity_check.py
//
// Why it exists: the two implementations are transcriptions of each other, and
// -ffp-contract=fast (GCC's default) fuses `num += w * (b - a)` into an FMA on
// the C++ side only. That put 339 of 3756 values ~1 ULP apart, all of them on
// decay_75 -- the one profile whose weights are not powers of two, hence the one
// profile where that multiply rounds at all. The fix lives in the CMakeLists;
// this harness is what keeps the fix honest.

#include <array>
#include <cstdio>
#include <span>
#include <vector>

#include "models.hpp"
#include "obi_config.hpp"
#include "signals.hpp"

namespace {

constexpr std::size_t kN = 5;

struct Book {
    std::array<double, kN> bid_vol{};
    std::array<double, kN> ask_vol{};
    int bid_depth = 0;
    int ask_depth = 0;
};

crossflux::OrderBookSnapshot<kN> make_snapshot(const Book& b) {
    crossflux::OrderBookSnapshot<kN> s{};
    s.timestamp_ms = 1;
    s.exchange_id[0] = 'x';
    s.exchange_id[1] = '\0';
    // Prices are irrelevant to the signal but must be sane (descending bids,
    // ascending asks, non-crossed) so nothing downstream trips an assert.
    for (std::size_t i = 0; i < kN; ++i) {
        s.bids[i].price  = 100.0 - static_cast<double>(i);
        s.bids[i].volume = b.bid_vol[i];
        s.asks[i].price  = 101.0 + static_cast<double>(i);
        s.asks[i].volume = b.ask_vol[i];
    }
    s.bid_depth = static_cast<uint8_t>(b.bid_depth);
    s.ask_depth = static_cast<uint8_t>(b.ask_depth);
    return s;
}

}  // namespace

int main() {
    const std::array<const crossflux::obi::Profile*, 4> profiles{
        &crossflux::obi::kFlat,
        &crossflux::obi::kDecay50,
        &crossflux::obi::kDecay75,
        &crossflux::obi::kL1Only,
    };

    Book b;
    // Line format: bid_depth ask_depth b0..b4 a0..a4   (all volumes as %a)
    while (std::scanf("%d %d", &b.bid_depth, &b.ask_depth) == 2) {
        for (std::size_t i = 0; i < kN; ++i) {
            if (std::scanf("%lf", &b.bid_vol[i]) != 1) return 2;
        }
        for (std::size_t i = 0; i < kN; ++i) {
            if (std::scanf("%lf", &b.ask_vol[i]) != 1) return 2;
        }
        const auto snap = make_snapshot(b);
        for (const auto* p : profiles) {
            const double v = crossflux::calculate_weighted_obi(snap, p->span());
            // Profile::name is a string_view, which is NOT null-terminated in
            // general -- "%s" reads past the end. Print it with an explicit
            // length instead.
            std::printf("%.*s %a\n", static_cast<int>(p->name.size()),
                        p->name.data(), v);
        }
    }
    return 0;
}
