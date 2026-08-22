// Cross-language parity harness for the VWAP book walk.
//
// Reads one book side per line from stdin as exact hex doubles, prints the whole
// BookWalk back as exact hex doubles. The Python side generates the cases and
// compares the output bit-for-bit against src/friction.py's walk_book.
//
// Hex floats (%a / float.hex()) both ways: a decimal round-trip would make a real
// 1-ULP divergence indistinguishable from the formatting, which is the size of
// disagreement this harness exists to detect.
//
// Do not run this by hand -- cpp_engine/tests/run_parity_check.py builds it with
// the project's real flags, generates the cases, and interprets the result:
//
//   python3 cpp_engine/tests/run_parity_check.py
//
// Why it exists, and why bit-exact rather than close
// -------------------------------------------------
// The fill price is the number the whole friction result rests on: the sweep in
// backtest/friction_report.html is a claim about how PnL decays with size, and
// that decay is entirely the vwap moving away from the touch. Two
// implementations produce it -- this one prices the C++ engine's fills,
// src/friction.py prices the backtester's -- and a report that quietly mixed
// them would be comparing two different strategies.
//
// A tolerance would not do. walk_book's loop is written in a deliberately
// unusual shape: `want - taken` against a running total rather than
// `remaining -= take`, because repeated subtraction rounds at every level while
// a running total rounds the way np.cumsum does. Written the obvious way, the
// two sides sat ~2.8e-14 apart -- comfortably inside any tolerance anyone would
// pick, and the exact class of drift that makes a fast path unauditable. Only
// bit-for-bit agreement can tell that fix from a coincidence.
//
// Unlike the weighted-OBI harness, nothing here is expected to diverge under
// -ffp-contract=fast: the walk's inner step is a multiply-accumulate
// (`notional += take * price`) that FMA *can* fuse, so the contrast build is a
// real question rather than a formality. Whichever way it comes out, the answer
// is reported rather than assumed.

#include <array>
#include <cstdio>
#include <span>

#include "friction.hpp"
#include "models.hpp"

namespace {

constexpr std::size_t kN = 5;

}  // namespace

int main() {
    // Deliberately assigned field by field rather than braced: PriceLevel's
    // two-argument constructor validates (price > 0), and half the point of this
    // harness is the zero-padded and non-positive levels walk_book has to skip.
    // Those are reachable only through the default constructor.
    std::array<crossflux::PriceLevel, kN> levels{};

    int    depth = 0;
    double qty   = 0.0;
    double touch = 0.0;

    // Line format: depth qty touch p0..p4 v0..v4   (depth as %d, rest as %a)
    while (std::scanf("%d %lf %lf", &depth, &qty, &touch) == 3) {
        for (std::size_t i = 0; i < kN; ++i) {
            if (std::scanf("%lf", &levels[i].price) != 1) return 2;
        }
        for (std::size_t i = 0; i < kN; ++i) {
            if (std::scanf("%lf", &levels[i].volume) != 1) return 2;
        }
        if (depth < 0 || static_cast<std::size_t>(depth) > kN) return 3;

        // The span is sliced to `depth`, so a level past the stated depth is not
        // merely zero-volume -- it is not visible to the walk at all. Python
        // slices its sequences the same way.
        const std::span<const crossflux::PriceLevel> side{
            levels.data(), static_cast<std::size_t>(depth)};
        const crossflux::friction::BookWalk w = crossflux::friction::walk_book(side, qty);

        // vwap, filled, notional, requested, slippage, levels_consumed.
        // requested_qty is printed too: it is the only field that survives a
        // no-fill, and it carries walk_book's max(0, qty) clamp on a negative
        // request.
        std::printf("%a %a %a %a %a %zu\n",
                    w.vwap, w.filled_qty, w.notional, w.requested_qty,
                    w.slippage_bps(touch), w.levels_consumed);
    }
    return 0;
}
