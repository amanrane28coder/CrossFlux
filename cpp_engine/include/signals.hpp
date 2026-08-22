/**
 * @file    signals.hpp
 * @brief   Header-only OBI and cross-venue delta signal functions.
 *
 * Phase 6: High-speed C++20 translation of src/signals.py.
 *
 * Design principles
 * -----------------
 *  constexpr throughout
 *      Both functions are constexpr-eligible, meaning the compiler MAY evaluate
 *      them entirely at compile time when fed constant inputs (e.g. in unit
 *      tests or static initialisers).  At runtime they compile to tight
 *      register-level arithmetic with no heap, no virtual dispatch, no RTTI.
 *
 *  Header-only (no .cpp translation unit)
 *      All functions are inline and template-parameterised.  The linker sees
 *      only one instantiation per N; the compiler can inline them at every
 *      call site without LTO.
 *
 *  Loop optimisation via template depth N
 *      OrderBookSnapshot<N> bakes the array size into the type.  The accumulation
 *      loop over [0, min(bid_depth, ask_depth)) has a compile-time upper bound of
 *      N.  For N ≤ 8, Clang/GCC auto-vectorises with SIMD (ARM: faddp / vaddpd).
 *      For N ≤ 4, the loop is fully unrolled to straight-line FMA pairs.
 *      No manual #pragma unroll or intrinsics are needed.
 *
 *  [[unlikely]] on the zero-denominator guard
 *      The C++20 [[unlikely]] attribute marks the guard branch as cold.
 *      This keeps the branch predictor focused on the hot path and avoids a
 *      misprediction penalty on every non-degenerate book.
 *
 *  noexcept + assert for calculate_obi_delta
 *      The Python function raises ValueError for out-of-range OBI inputs.
 *      In C++ the function is noexcept: throwing on the HFT hot path violates
 *      latency SLAs and unwinds the stack.  Instead, a debug-only assert fires
 *      immediately during development; in Release (-DNDEBUG) it compiles away.
 *
 *  Zero-denominator guard uses == 0.0 (not epsilon)
 *      The denominator is a sum of non-negative volumes.  There are no
 *      cancellation paths, so IEEE 754 guarantees the sum is exactly +0.0
 *      when all volumes are zero.  An epsilon guard would be wrong here —
 *      it would suppress legitimate near-zero-but-non-zero imbalance signals.
 *
 * Python parity table
 * -------------------
 *   Python                           C++
 *   ──────────────────────────────── ──────────────────────────────────────────
 *   calculate_obi(snap, depth=5)     calculate_obi<N>(snap)
 *                                      — depth = min(bid_depth, ask_depth)
 *   calculate_obi_delta(a, b)        calculate_obi_delta(a, b) noexcept
 *   ValueError on depth <= 0        N >= 1 enforced by OrderBookSnapshot<N>
 *   ValueError on bad OBI range     assert() in Debug; removed in Release
 *
 * C++20 required.
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <algorithm>    // std::min
#include <span>         // std::span — weight vector view, non-owning

#include "models.hpp"

namespace crossflux {

// ─────────────────────────────────────────────────────────────────────────────
// calculate_obi<N>
//
// Compute the Order Book Imbalance across all valid levels of a snapshot.
//
// Formula:
//                V_bid  -  V_ask
//   OBI  =  ─────────────────────   ∈ [-1.0, 1.0]
//                V_bid  +  V_ask
//
// where V_bid = Σ bids[i].volume  for i in [0, min(bid_depth, ask_depth))
//       V_ask = Σ asks[i].volume  for i in [0, min(bid_depth, ask_depth))
//
// The loop bound is min(bid_depth, ask_depth) — the number of valid levels
// on the shallower side.  Slots beyond that depth hold sentinel PriceLevels
// with volume = 0.0 and are never touched.
//
// Degenerate / illiquid book:
//   If bid_vol + ask_vol == 0.0 (all volumes are zero, or both depths are
//   zero), the function returns 0.0 as a safe sentinel — identical to the
//   Python implementation's ZeroDivisionError guard.
//
// Template parameter:
//   N — compile-time book depth from OrderBookSnapshot<N>.  Controls loop
//       unrolling and SIMD vectorisation decisions made by the compiler.
//
// @param  snap   Const reference to a valid OrderBookSnapshot<N>.
//                Must have been constructed via make_order_book_snapshot()
//                to ensure the invariants hold.
// @return OBI scalar in [-1.0, 1.0], or 0.0 for a degenerate book.  The
//         interval is closed: an empty side gives exactly ±1.0, and so does a
//         side whose total volume is below the other side's ULP.
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N>
[[nodiscard]] constexpr double
calculate_obi(const OrderBookSnapshot<N>& snap) noexcept
{
    // ── Determine loop bound ──────────────────────────────────────────────
    // Consume only the valid levels on the shallower side.
    // Cast to size_t for indexing arithmetic — bid_depth/ask_depth are uint8_t.
    const std::size_t depth = std::min(
        static_cast<std::size_t>(snap.bid_depth),
        static_cast<std::size_t>(snap.ask_depth)
    );

    // ── Two-accumulator reduction ─────────────────────────────────────────
    // Separate bid and ask accumulators allow the compiler to issue two
    // independent FMA chains, doubling the effective throughput on
    // superscalar pipelines.  On ARM NEON, Clang fuses these into
    // paired vector loads + faddp instructions for N <= 8.
    double bid_vol = 0.0;
    double ask_vol = 0.0;

    for (std::size_t i = 0; i < depth; ++i) {
        bid_vol += snap.bids[i].volume;
        ask_vol += snap.asks[i].volume;
    }

    // ── Zero-denominator guard ────────────────────────────────────────────
    // == 0.0 is exact here: denominator is a sum of non-negative doubles.
    // No cancellation path exists, so IEEE 754 guarantees +0.0 output when
    // all volumes are zero.  [[unlikely]] keeps this branch cold.
    const double denom = bid_vol + ask_vol;
    if (denom == 0.0) [[unlikely]] {
        return 0.0;
    }

    return (bid_vol - ask_vol) / denom;
}


// ─────────────────────────────────────────────────────────────────────────────
// calculate_weighted_obi
//
// Multi-level OBI with a per-level weight vector.  Discounts resting depth by
// distance from the touch: size sitting five ticks away says less about
// immediate pressure than size at the front of the queue, and is cheaper to
// spoof.
//
// Formula:
//                Σ w[i] · (bid[i].volume − ask[i].volume)
//   wOBI  =     ──────────────────────────────────────────
//                Σ w[i] · (bid[i].volume + ask[i].volume)
//
// NOT MLOFI.  The upgrade request that prompted this function called it MLOFI,
// but MLOFI (Cont–Kukanov–Stoikov and its multi-level extensions) is built from
// the *change* in depth between consecutive book updates:
//
//   OFI_i(t) = ΔBidDepth_i(t) − ΔAskDepth_i(t)
//
// That measures order flow — arrivals, cancels, fills.  This measures a static
// snapshot of resting depth.  Both are useful; they are not the same statistic,
// and the name is worth getting right.  Real MLOFI needs incremental L2 deltas,
// which data/raw/ does not currently contain (5-level snapshots only).
//
// Why normalized:
//   Dividing by weighted *total* volume keeps the result in [-1.0, 1.0], the
//   same range as calculate_obi.  This is load-bearing, not cosmetic:
//     · calculate_obi_delta asserts its inputs are in [-1.0, 1.0].  A raw
//       weighted sum carries volume units and trips that assert in Debug.
//     · SignalAggregator's delta_threshold_ (default 0.3, predictor.hpp) is
//       calibrated against a bounded delta in [-2.0, 2.0].  Against raw BTC
//       volume sums, 0.3 is cleared by nearly every tick — the entry gate
//       silently degrades into a pass-through, and the engine looks far busier
//       while actually filtering nothing.
//
//   The bound holds by construction, with no clamping: weights are
//   non-negative (validated in src/obi_weights.py) and volumes are
//   non-negative, so |numerator| <= denominator term by term.
//
//   The interval is closed, and both endpoints are reachable.  An empty book
//   side gives numerator == denominator, hence exactly ±1.0.  So does a side
//   whose weighted volume falls below the other side's ULP — 1e9 + 1e-9 == 1e9
//   in double precision, so num and den become the same double.  That second
//   route needs roughly sixteen orders of magnitude between the sides and no
//   real book reaches it, but it is why calculate_obi_delta's assert is
//   inclusive and why t10 in cpp_engine/tests/test_signals.cpp asserts a closed
//   bound rather than a strict one.
//
// Relationship to calculate_obi:
//   With every weight equal and weights.size() >= min(bid_depth, ask_depth),
//   this reduces to calculate_obi exactly — the common factor cancels.  The
//   "flat" profile is therefore the honest baseline for a weighted-vs-unweighted
//   comparison.  Caveat: "flat" has depth 5, so on a book with more than 5
//   valid levels per side it consumes fewer levels than calculate_obi and the
//   two diverge.  Every dataset in data/raw/ is book_snapshot_5, so they agree
//   there.
//
// Depth consumed:
//   min(bid_depth, ask_depth, weights.size()) — the shallowest of the two book
//   sides and the weight vector.  Using the shallower *book* side matches
//   calculate_obi's existing convention.  Note that src/signals.py's
//   calculate_obi slices each side independently, so Python and C++ already
//   disagree on asymmetric books; the Python mirror of *this* function follows
//   the C++ convention deliberately, so the two agree here.
//
// Accumulation order matches src/obi_weights.py term for term, and under the
// project's build flags the two implementations agree *bit for bit* — that is
// what cpp_engine/tests/run_parity_check.py asserts, comparing hex doubles with
// ==, over 3,756 values.
//
// That equality depends on -ffp-contract=off, which every CMake target sets.
// Do not drop it: under GCC's default -ffp-contract=fast, `num += w * (b - a)`
// contracts into a single FMA, rounding once where Python rounds twice.  The
// measured cost is 339 of those 3,756 values off by ~1 ULP, all of them on the
// decay_75 profile — the only shipped preset whose weights are not powers of
// two, hence the only one where that multiply rounds at all.  Too small to move
// a trade decision against a 0.3 threshold, but it means the backtester and the
// live engine compute different numbers from identical input.
//
// Template parameter:
//   N — compile-time book depth from OrderBookSnapshot<N>.
//
// @param  snap     Const reference to a valid OrderBookSnapshot<N>.
// @param  weights  Per-level weights, weights[0] being the touch.  Typically
//                  crossflux::obi::active_weights() from the generated
//                  obi_config.hpp.  Must be non-negative.
// @return Weighted OBI in [-1.0, 1.0], or 0.0 for a degenerate book or an
//         empty weight vector.  Closed interval — see "Why normalized" above.
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N>
[[nodiscard]] constexpr double
calculate_weighted_obi(const OrderBookSnapshot<N>& snap,
                       std::span<const double> weights) noexcept
{
    // ── Determine loop bound ──────────────────────────────────────────────
    // Three-way min: the shallower book side, and the weight vector length.
    const std::size_t depth = std::min(
        std::min(static_cast<std::size_t>(snap.bid_depth),
                 static_cast<std::size_t>(snap.ask_depth)),
        weights.size()
    );

    // ── Two-accumulator reduction ─────────────────────────────────────────
    // Grouped as w*(b-a) and w*(b+a) rather than four separate products: half
    // the multiplies, and it mirrors src/obi_weights.py term for term so the
    // parity check is comparing the same arithmetic.
    double num = 0.0;
    double den = 0.0;

    for (std::size_t i = 0; i < depth; ++i) {
        const double w = weights[i];
        const double b = snap.bids[i].volume;
        const double a = snap.asks[i].volume;
        num += w * (b - a);
        den += w * (b + a);
    }

    // ── Zero-denominator guard ────────────────────────────────────────────
    // den is a sum of non-negative products (non-negative weights x
    // non-negative volumes), so == 0.0 is exact — no cancellation path exists.
    // Covers a zero-volume book, depth == 0, and an all-zero weight prefix.
    if (den == 0.0) [[unlikely]] {
        return 0.0;
    }

    return num / den;
}


// ─────────────────────────────────────────────────────────────────────────────
// calculate_obi_delta
//
// Compute the cross-venue OBI delta — the directional divergence between two
// venues.  This is the primary composite signal for detecting cross-venue
// arbitrage opportunities driven by ghost liquidity.
//
// Formula:
//   Δ_OBI(A, B) = OBI_A − OBI_B   ∈ [-2.0, +2.0]
//
// Sign convention:
//   +  →  Exchange A is bid-heavy relative to B (buy B, sell A)
//   −  →  Exchange B is bid-heavy relative to A (buy A, sell B)
//
// Unlike a price-spread metric, OBI delta is a volume-pressure metric that
// captures ghost liquidity dynamics invisible to price-only signals.
//
// Parameters:
//   obi_a  — OBI of the first venue.  Must be in [-1.0, 1.0].
//             (Enforced by assert() in Debug builds; removed in Release.)
//   obi_b  — OBI of the second venue.  Must be in [-1.0, 1.0].
//
// Returns:
//   OBI delta in [-2.0, +2.0].
//
// noexcept rationale:
//   Throwing on the HFT hot path unwinds the stack and flushes the branch
//   predictor.  Debug-mode assert() catches contract violations during
//   development; Release builds compile asserts away entirely (-DNDEBUG).
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] constexpr double
calculate_obi_delta(double obi_a, double obi_b) noexcept
{
    // Debug-only contract checks — compiled out in Release (-DNDEBUG)
    assert(obi_a >= -1.0 && obi_a <= 1.0
           && "calculate_obi_delta: obi_a out of [-1.0, 1.0] range");
    assert(obi_b >= -1.0 && obi_b <= 1.0
           && "calculate_obi_delta: obi_b out of [-1.0, 1.0] range");

    return obi_a - obi_b;
}

} // namespace crossflux
