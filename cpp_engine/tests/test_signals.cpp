/**
 * @file    test_signals.cpp
 * @brief   Standalone verification suite for calculate_obi<N>,
 *          calculate_weighted_obi<N> and calculate_obi_delta.
 *
 * Zero external dependencies.
 *
 * Test inventory
 * --------------
 *   t1_bid_skewed       bid_vol >> ask_vol (1 level)           → OBI ≈ +0.9802
 *   t2_ask_skewed       ask_vol >> bid_vol (1 level)           → OBI ≈ -0.9802
 *   t3_balanced         bid_vol == ask_vol                     → OBI =  0.0
 *   t4_zero_volume      all volumes = 0.0                      → OBI =  0.0 (sentinel)
 *   t5_multi_level      3-level bids (1,2,3) vs 3-level asks   → OBI = (6-3)/(6+3) = 0.3333...
 *   t6_sparse_book      bid_depth=2, ask_depth=3, N=10         → consumes min(2,3)=2 levels
 *   t7_delta_positive   obi_a=+0.8, obi_b=-0.6                → delta = +1.4
 *   t8_delta_zero       obi_a == obi_b                         → delta =  0.0
 *   t9_flat_reproduces_obi   flat weights == calculate_obi, to 8 ULP
 *   t10_weighted_bounded     |wOBI| <= 1; == ±1 on an empty side or a >1e16 ratio
 *   t11_decay_favours_touch  steeper decay ⇒ the touch outvotes the deep levels
 *   t12_l1_only_ignores_depth  l1_only is blind to levels 2..5
 *   t13_weighted_degenerate  zero book and empty weight span → 0.0 sentinel
 *   t14_weighted_sparse_book weighted path honours min(bid_depth, ask_depth)
 *   t15_profiles_distinct    the four presets are not the same signal
 *   t16_weight_span_length_bounds_depth  a short span must truncate the loop
 *
 * The weighted cases (t9–t16) mirror tests/test_signals.py one for one. That is
 * deliberate: the two implementations are checked against each other bit-for-bit
 * by cpp_engine/tests/parity_weighted_obi.cpp, and a shared expectation that is
 * wrong on both sides would sail through that harness. These tests pin the
 * arithmetic to hand-derivable numbers so parity means "both right", not merely
 * "both the same".
 *
 * Whether these tests check anything at all is itself checked:
 * cpp_engine/tests/run_mutation_check.py corrupts calculate_weighted_obi five
 * ways and confirms the suite notices each one. t16 exists because one of those
 * mutations initially survived.
 *
 * Harness
 * -------
 * Uses a local EXPECT_NEAR macro that writes PASS/FAIL to stdout/stderr and
 * increments a failure counter.  Returns exit code 0 if all tests pass,
 * non-zero otherwise — compatible with cmake's ctest(1).
 *
 * Build command (direct clang++, no cmake required):
 *   clang++ -std=c++20 -Wall -Wextra -O2 \
 *       -I cpp_engine/include \
 *       cpp_engine/tests/test_signals.cpp \
 *       -o /tmp/test_signals && /tmp/test_signals
 *
 * C++20 required.
 */

#include <array>
#include <cassert>
#include <cmath>      // std::abs
#include <cstdio>     // std::printf, std::fprintf
#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <limits>     // std::numeric_limits<double>::epsilon
#include <string_view>

#include "models.hpp"
#include "obi_config.hpp"
#include "signals.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Test harness — zero external dependencies
// ─────────────────────────────────────────────────────────────────────────────

static int g_failures = 0;
static int g_total    = 0;

/// Assert that |actual - expected| <= tol.  Prints PASS or FAIL to stdout/stderr.
#define EXPECT_NEAR(actual, expected, tol, name)                             \
    do {                                                                      \
        ++g_total;                                                            \
        const double _a = static_cast<double>(actual);                       \
        const double _e = static_cast<double>(expected);                     \
        const double _t = static_cast<double>(tol);                          \
        if (std::abs(_a - _e) <= _t) {                                       \
            std::printf("  PASS  %-30s  got=%.8f  expected=%.8f\n",          \
                        name, _a, _e);                                        \
        } else {                                                              \
            std::fprintf(stderr,                                              \
                "  FAIL  %-30s  got=%.8f  expected=%.8f  tol=%.2e\n",        \
                name, _a, _e, _t);                                            \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

/// Assert exact equality for doubles where exactness is expected (e.g. 0.0).
#define EXPECT_EQ(actual, expected, name) EXPECT_NEAR(actual, expected, 0.0, name)

/// Assert a boolean property (an ordering, a bound). Routed through the same
/// counters so a failing property cannot be lost next to the numeric checks;
/// prints got=0.00000000 expected=1.00000000 when it fails.
#define EXPECT_TRUE(cond, name) EXPECT_EQ((cond) ? 1.0 : 0.0, 1.0, name)


// ─────────────────────────────────────────────────────────────────────────────
// Fixture helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using namespace crossflux;

/// Build a one-level snapshot with precisely controlled bid/ask volumes.
/// OBI of the result = (bid_vol - ask_vol) / (bid_vol + ask_vol).
[[nodiscard]] OrderBookSnapshot<10>
make_one_level(double bid_vol, double ask_vol,
               double bid_px = 49'999.0, double ask_px = 50'001.0)
{
    std::array<PriceLevel, 10> bids{};
    std::array<PriceLevel, 10> asks{};
    bids[0] = PriceLevel{bid_px, bid_vol};
    asks[0] = PriceLevel{ask_px, ask_vol};

    return make_order_book_snapshot<10>(
        1'700'000'000'000ULL, "binance",
        bids, asks,
        /*bid_depth=*/1, /*ask_depth=*/1);
}

/// Build a snapshot with up to 3 explicitly valued bid and ask levels.
[[nodiscard]] OrderBookSnapshot<10>
make_three_level(
    double b0, double b1, double b2,
    double a0, double a1, double a2)
{
    std::array<PriceLevel, 10> bids{};
    std::array<PriceLevel, 10> asks{};
    bids[0] = PriceLevel{49'999.0, b0};
    bids[1] = PriceLevel{49'998.0, b1};
    bids[2] = PriceLevel{49'997.0, b2};
    asks[0] = PriceLevel{50'001.0, a0};
    asks[1] = PriceLevel{50'002.0, a1};
    asks[2] = PriceLevel{50'003.0, a2};

    return make_order_book_snapshot<10>(
        1'700'000'000'000ULL, "kraken",
        bids, asks,
        /*bid_depth=*/3, /*ask_depth=*/3);
}

/// Build a snapshot with five explicitly valued bid and ask levels.
/// Five is the depth every shipped profile uses, and the depth of every
/// book_snapshot_5 file in data/raw/, so this is the shape the weighted tests
/// actually care about.
[[nodiscard]] OrderBookSnapshot<10>
make_five_level(const std::array<double, 5>& bid_vols,
                const std::array<double, 5>& ask_vols,
                uint8_t bid_depth = 5, uint8_t ask_depth = 5)
{
    std::array<PriceLevel, 10> bids{};
    std::array<PriceLevel, 10> asks{};
    for (std::size_t i = 0; i < 5; ++i) {
        // Prices only have to be sane and uncrossed; the signal ignores them.
        bids[i] = PriceLevel{49'999.0 - static_cast<double>(i), bid_vols[i]};
        asks[i] = PriceLevel{50'001.0 + static_cast<double>(i), ask_vols[i]};
    }
    return make_order_book_snapshot<10>(
        1'700'000'000'003ULL, "binance",
        bids, asks, bid_depth, ask_depth);
}

} // anonymous namespace


// ─────────────────────────────────────────────────────────────────────────────
// Test cases
// ─────────────────────────────────────────────────────────────────────────────

/// t1 — Heavily bid-skewed book: bid_vol=100, ask_vol=1
/// Expected OBI = (100 - 1) / (100 + 1) = 99 / 101 ≈ 0.98019801...
static void t1_bid_skewed()
{
    const auto snap = make_one_level(/*bid_vol=*/100.0, /*ask_vol=*/1.0);
    const double obi = calculate_obi(snap);
    EXPECT_NEAR(obi, 99.0 / 101.0, 1e-12, "t1_bid_skewed");
}

/// t2 — Heavily ask-skewed book: bid_vol=1, ask_vol=100
/// Expected OBI = (1 - 100) / (1 + 100) = -99 / 101 ≈ -0.98019801...
static void t2_ask_skewed()
{
    const auto snap = make_one_level(/*bid_vol=*/1.0, /*ask_vol=*/100.0);
    const double obi = calculate_obi(snap);
    EXPECT_NEAR(obi, -99.0 / 101.0, 1e-12, "t2_ask_skewed");
}

/// t3 — Perfectly balanced book: bid_vol == ask_vol
/// Expected OBI = 0.0 (exact)
static void t3_balanced()
{
    const auto snap = make_one_level(/*bid_vol=*/50.0, /*ask_vol=*/50.0);
    const double obi = calculate_obi(snap);
    EXPECT_EQ(obi, 0.0, "t3_balanced");
}

/// t4 — Completely illiquid / zero-volume book (degenerate case)
/// bid_vol=0, ask_vol=0 → denominator=0 → safe sentinel 0.0
/// This exercises the [[unlikely]] zero-denominator guard.
static void t4_zero_volume()
{
    // Build a snapshot with zero volumes manually — PriceLevel default constructor
    // sets volume=0.0, but validated constructor requires price > 0.
    // We use volume=0.0 with a valid price by using the default-then-assign pattern.
    std::array<PriceLevel, 10> bids{};
    std::array<PriceLevel, 10> asks{};
    // Assign valid prices but zero volumes
    bids[0] = PriceLevel{49'999.0, 0.0};
    asks[0] = PriceLevel{50'001.0, 0.0};

    const auto snap = make_order_book_snapshot<10>(
        1'700'000'000'001ULL, "binance",
        bids, asks, 1, 1);

    const double obi = calculate_obi(snap);
    EXPECT_EQ(obi, 0.0, "t4_zero_volume");
}

/// t5 — Multi-level book: bids=(1,2,3) asks=(1,1,1) — 3 levels each
/// bid_vol = 1+2+3 = 6, ask_vol = 1+1+1 = 3
/// Expected OBI = (6-3)/(6+3) = 3/9 = 0.33333...
static void t5_multi_level()
{
    const auto snap = make_three_level(
        1.0, 2.0, 3.0,   // bid volumes
        1.0, 1.0, 1.0);  // ask volumes

    const double obi = calculate_obi(snap);
    EXPECT_NEAR(obi, 1.0 / 3.0, 1e-12, "t5_multi_level");
}

/// t6 — Sparse book: bid_depth=2 < ask_depth=3, N=10
/// Only min(2,3)=2 levels consumed from each side.
/// bids=(1,2,3), asks=(1,1,1)  →  only first 2 levels used
/// bid_vol=1+2=3, ask_vol=1+1=2  → OBI=(3-2)/(3+2)=1/5=0.2
static void t6_sparse_book()
{
    std::array<PriceLevel, 10> bids{};
    std::array<PriceLevel, 10> asks{};
    bids[0] = PriceLevel{49'999.0, 1.0};
    bids[1] = PriceLevel{49'998.0, 2.0};
    bids[2] = PriceLevel{49'997.0, 3.0};  // NOT consumed (bid_depth=2)
    asks[0] = PriceLevel{50'001.0, 1.0};
    asks[1] = PriceLevel{50'002.0, 1.0};
    asks[2] = PriceLevel{50'003.0, 1.0};  // NOT consumed (ask_depth > bid_depth=2)

    const auto snap = make_order_book_snapshot<10>(
        1'700'000'000'002ULL, "kraken",
        bids, asks,
        /*bid_depth=*/2, /*ask_depth=*/3);

    // depth = min(2, 3) = 2 → bid_vol=1+2=3, ask_vol=1+1=2 → OBI=1/5
    const double obi = calculate_obi(snap);
    EXPECT_NEAR(obi, 0.2, 1e-12, "t6_sparse_book");
}

/// t7 — Delta: positive divergence (A bid-heavy, B ask-heavy)
/// obi_a=+0.8, obi_b=-0.6 → delta = 0.8 - (-0.6) = +1.4
static void t7_delta_positive()
{
    const double delta = calculate_obi_delta(0.8, -0.6);
    EXPECT_NEAR(delta, 1.4, 1e-12, "t7_delta_positive");
}

/// t8 — Delta: zero divergence (identical venues)
/// For any equal value v: calculate_obi_delta(v, v) must equal 0.0 exactly.
static void t8_delta_zero()
{
    // The label is per-case rather than one shared literal. The harness prints
    // the name it is handed, so five identical "t8_delta_zero" lines made a
    // failure impossible to attribute to an input.
    struct Case { double v; const char* name; };
    static constexpr Case cases[] = {
        {-1.0, "t8_delta_zero[v=-1]"},
        {-0.5, "t8_delta_zero[v=-0.5]"},
        { 0.0, "t8_delta_zero[v=0]"},
        { 0.5, "t8_delta_zero[v=+0.5]"},
        { 1.0, "t8_delta_zero[v=+1]"},
    };
    for (const auto& c : cases) {
        EXPECT_EQ(calculate_obi_delta(c.v, c.v), 0.0, c.name);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Weighted OBI (t9–t15)
//
// Mirrors tests/test_signals.py. Naming note: this is a weighted static depth
// imbalance, not MLOFI — see the comment above calculate_weighted_obi.
// ─────────────────────────────────────────────────────────────────────────────

/// t9 — The baseline claim: flat weights reduce to calculate_obi.
///
/// bids=(3,1,4,1,5) → 14, asks=(2,7,1,8,2) → 20, so OBI = -6/34 = -0.176470...
/// The common weight cancels out of numerator and denominator, so this is an
/// algebraic identity, not an approximation.
///
/// Why a tolerance and not ==:
///   calculate_obi sums each side and subtracts once; calculate_weighted_obi
///   subtracts per level and sums the differences. Same value in exact
///   arithmetic, different rounding in floating point. The largest divergence
///   measured over 65,000 randomised books (including books built to force
///   cancellation) was 3 x epsilon; 8 leaves headroom for a wider depth or a
///   different libm without being loose enough to hide a real reordering, which
///   would show up around 1e-3 or larger. Same constant as
///   FLAT_PARITY_TOL in tests/test_signals.py.
static void t9_flat_reproduces_obi()
{
    constexpr double kFlatParityTol = 8.0 * std::numeric_limits<double>::epsilon();

    const auto snap     = make_five_level({3.0, 1.0, 4.0, 1.0, 5.0},
                                         {2.0, 7.0, 1.0, 8.0, 2.0});
    const double plain    = calculate_obi(snap);
    const double weighted = calculate_weighted_obi(snap, obi::kFlat.span());

    EXPECT_NEAR(plain, -6.0 / 34.0, 1e-12, "t9_plain_obi_value");
    EXPECT_NEAR(weighted, plain, kFlatParityTol, "t9_flat_reproduces_obi");
}

/// t10 — The bound that everything downstream depends on.
///
/// calculate_obi_delta asserts its inputs are within [-1, 1], and
/// SignalAggregator's 0.3 threshold is calibrated against that range.
///
/// The bound is CLOSED, not open. Two ways to land exactly on ±1:
///   1. One side of the book is empty — numerator == denominator by
///      construction, no rounding involved.
///   2. One side is smaller than the other side's ULP. 1e9 + 1e-9 == 1e9 in
///      double precision, so num and den are the same double and the quotient
///      is exactly 1.0. This needs ~16 orders of magnitude between the two
///      sides, which no real BTC book has, but it is reachable and it is why
///      calculate_obi_delta's assert is inclusive. This test originally claimed
///      strict interiority and failed here on l1_only, which is how the closed
///      bound was found; the doc comments in signals.hpp said (-1.0, 1.0) and
///      have been corrected.
static void t10_weighted_bounded()
{
    const std::array<const obi::Profile*, 4> profiles{
        &obi::kFlat, &obi::kDecay50, &obi::kDecay75, &obi::kL1Only};

    // Lopsided but within realistic magnitudes (9 decades): strictly inside.
    const auto lopsided = make_five_level({1.0e5, 1.0e-4, 5.0, 0.25, 40.0},
                                          {1.0e-4, 1.0e5, 0.5, 90.0, 1.0});
    for (const auto* p : profiles) {
        const double v = calculate_weighted_obi(lopsided, p->span());
        EXPECT_TRUE(std::abs(v) < 1.0, "t10_strictly_inside");
        // Cheap proof the value is usable as a delta input at all.
        EXPECT_NEAR(calculate_obi_delta(v, 0.0), v, 0.0, "t10_usable_as_delta");
    }

    // Empty ask side: numerator == denominator, so exactly +1.0.
    const auto bids_only = make_five_level({5.0, 4.0, 3.0, 2.0, 1.0},
                                           {0.0, 0.0, 0.0, 0.0, 0.0});
    const auto asks_only = make_five_level({0.0, 0.0, 0.0, 0.0, 0.0},
                                           {5.0, 4.0, 3.0, 2.0, 1.0});
    for (const auto* p : profiles) {
        EXPECT_EQ(calculate_weighted_obi(bids_only, p->span()), 1.0,
                  "t10_bids_only_is_plus_one");
        EXPECT_EQ(calculate_weighted_obi(asks_only, p->span()), -1.0,
                  "t10_asks_only_is_minus_one");
    }

    // Saturation by rounding: the ask side is below the ULP of the bid side.
    const auto swamped = make_five_level({1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9},
                                         {1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9});
    for (const auto* p : profiles) {
        const double v = calculate_weighted_obi(swamped, p->span());
        EXPECT_EQ(v, 1.0, "t10_saturates_at_plus_one");
        // Still a legal delta input — the closed bound is what the assert wants.
        EXPECT_TRUE(v >= -1.0 && v <= 1.0, "t10_saturated_still_in_bounds");
    }
}

/// t11 — The reason the weighting exists.
///
/// Book: bids=(100,1,1,1,1) asks=(1,60,60,60,60). Size at the touch is bid-side;
/// the size behind it is ask-side and much larger in total. Flat weights call
/// this book ask-heavy. The steeper the decay, the more the touch outvotes the
/// depth, until decay_50 calls it bid-heavy:
///
///   flat      (104-241)/345                        = -0.397101...
///   decay_75  tail weight 2.05078125               = -0.097285...
///   decay_50  (99 - 59*0.9375)/(101 + 61*0.9375)
///             = 43.6875/158.1875                   = +0.276175...
///
/// The ordering flat < decay_75 < decay_50 is the property under test; a broken
/// weight vector or a mis-indexed loop breaks the ordering, not just a value.
static void t11_decay_favours_touch()
{
    const auto snap = make_five_level({100.0, 1.0, 1.0, 1.0, 1.0},
                                      {1.0, 60.0, 60.0, 60.0, 60.0});

    const double flat  = calculate_weighted_obi(snap, obi::kFlat.span());
    const double d75   = calculate_weighted_obi(snap, obi::kDecay75.span());
    const double d50   = calculate_weighted_obi(snap, obi::kDecay50.span());

    EXPECT_NEAR(flat, -137.0 / 345.0, 1e-12, "t11_flat_says_ask_heavy");
    EXPECT_NEAR(d50, 43.6875 / 158.1875, 1e-12, "t11_decay50_says_bid_heavy");
    EXPECT_TRUE(flat < d75 && d75 < d50,
                "t11_steeper_decay_is_more_bid_heavy");
}

/// t12 — l1_only has depth 1, so levels 2..5 must not reach the result.
/// Both books share bids[0]=2 / asks[0]=6 → (2-6)/(2+6) = -0.5 exactly, and
/// differ everywhere below. The flat readings are asserted to differ, otherwise
/// "l1_only ignored the depth" would be vacuously true for two identical books.
static void t12_l1_only_ignores_depth()
{
    const auto deep_bids = make_five_level({2.0, 999.0, 999.0, 999.0, 999.0},
                                           {6.0, 1.0, 1.0, 1.0, 1.0});
    const auto deep_asks = make_five_level({2.0, 0.0, 0.0, 0.0, 0.0},
                                           {6.0, 500.0, 500.0, 500.0, 500.0});

    EXPECT_EQ(calculate_weighted_obi(deep_bids, obi::kL1Only.span()), -0.5,
              "t12_l1_only_deep_bids");
    EXPECT_EQ(calculate_weighted_obi(deep_asks, obi::kL1Only.span()), -0.5,
              "t12_l1_only_deep_asks");

    // Vacuity guard: the two books really are different below the touch.
    const double flat_a = calculate_weighted_obi(deep_bids, obi::kFlat.span());
    const double flat_b = calculate_weighted_obi(deep_asks, obi::kFlat.span());
    EXPECT_TRUE(std::abs(flat_a - flat_b) > 1.0,
                "t12_books_differ_below_touch");
}

/// t13 — Degenerate inputs hit the zero-denominator sentinel, not a NaN.
/// Three routes to den == 0: no volume anywhere, an empty weight span, and a
/// weight prefix that is all zeros. (bid_depth == 0 is unreachable through the
/// validated factory, which requires depth >= 1 — the empty span covers the same
/// branch.)
static void t13_weighted_degenerate()
{
    const auto empty_book = make_five_level({0.0, 0.0, 0.0, 0.0, 0.0},
                                            {0.0, 0.0, 0.0, 0.0, 0.0});
    EXPECT_EQ(calculate_weighted_obi(empty_book, obi::kFlat.span()), 0.0,
              "t13_zero_volume_book");

    const auto real_book = make_five_level({3.0, 1.0, 4.0, 1.0, 5.0},
                                           {2.0, 7.0, 1.0, 8.0, 2.0});
    EXPECT_EQ(calculate_weighted_obi(real_book, std::span<const double>{}), 0.0,
              "t13_empty_weight_span");

    static constexpr std::array<double, 3> zero_weights{0.0, 0.0, 0.0};
    EXPECT_EQ(calculate_weighted_obi(
                  real_book, std::span<const double>{zero_weights}),
              0.0, "t13_all_zero_weights");
}

/// t14 — The weighted path uses the same depth convention as calculate_obi:
/// min(bid_depth, ask_depth, weights.size()). With bid_depth=2 and ask_depth=3
/// only two levels are consumed, so with decay_50 weights (1, 0.5):
///   num = 1*(1-1) + 0.5*(2-1) = 0.5
///   den = 1*(1+1) + 0.5*(2+1) = 3.5   →  1/7 = 0.142857...
/// Level 3 is present on the ask side and must not appear in the result.
static void t14_weighted_sparse_book()
{
    const auto snap = make_five_level({1.0, 2.0, 3.0, 0.0, 0.0},
                                      {1.0, 1.0, 1.0, 0.0, 0.0},
                                      /*bid_depth=*/2, /*ask_depth=*/3);
    EXPECT_NEAR(calculate_weighted_obi(snap, obi::kDecay50.span()),
                1.0 / 7.0, 1e-12, "t14_weighted_sparse_book");

    // The unweighted signal on the same book, for the record: (3-2)/(3+2)=0.2.
    EXPECT_NEAR(calculate_obi(snap), 0.2, 1e-12, "t14_plain_sparse_book");
}

/// t15 — The whole weighted suite would pass vacuously if the four presets had
/// collapsed to the same weight vector (a bad obi_config.hpp regeneration would
/// do exactly that, since the header is generated from src/obi_weights.py).
/// One probe book, four distinct readings — the same guard the parity harness
/// ends with, and the same book, so the numbers are comparable.
static void t15_profiles_distinct()
{
    const auto snap = make_five_level({100.0, 1.0, 1.0, 1.0, 1.0},
                                      {1.0, 80.0, 80.0, 80.0, 80.0});
    const std::array<const obi::Profile*, 4> profiles{
        &obi::kFlat, &obi::kDecay50, &obi::kDecay75, &obi::kL1Only};

    std::array<double, 4> vals{};
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        vals[i] = calculate_weighted_obi(snap, profiles[i]->span());
    }

    int collisions = 0;
    for (std::size_t i = 0; i < vals.size(); ++i) {
        for (std::size_t j = i + 1; j < vals.size(); ++j) {
            if (vals[i] == vals[j]) ++collisions;
        }
    }
    EXPECT_EQ(collisions, 0, "t15_profiles_distinct");

    // Pinned values, so a silent weight edit fails here and not only in parity.
    EXPECT_NEAR(vals[0], -0.5105882352941177, 1e-12, "t15_flat");
    EXPECT_NEAR(vals[1],  0.1409395973154362, 1e-12, "t15_decay_50");
    EXPECT_NEAR(vals[2], -0.2358988607946652, 1e-12, "t15_decay_75");
    EXPECT_NEAR(vals[3],  99.0 / 101.0,       1e-12, "t15_l1_only");
}

/// t16 — The weights.size() term of the three-way min, pinned independently.
///
/// This test exists because a mutation survived. Deleting `weights.size()` from
/// the min in calculate_weighted_obi left t9–t15 all green: every shipped
/// Profile zero-pads its weight storage out to kMaxLevels, so reading past the
/// span's end lands on a 0.0 and the value comes out unchanged. The bound was
/// therefore doing real work — it is what keeps the read in bounds for a caller
/// who passes a short span over a densely populated array — while being
/// unfalsifiable by any test that only uses the presets.
///
/// So: a 5-entry weight array with large values at levels 3-5, viewed as a span
/// of length 2. Depth must come out as 2.
///   num = 1*(1-5) + 0.5*(2-4) = -5
///   den = 1*(1+5) + 0.5*(2+4) =  9   →  -5/9 = -0.5555...
/// Ignoring the bound would read the 7.0s and give +0.2741 instead — a defined,
/// in-bounds read of the parent array, so the mutation is caught by value rather
/// than by tripping undefined behaviour.
static void t16_weight_span_length_bounds_depth()
{
    static constexpr std::array<double, 5> weights{1.0, 0.5, 7.0, 7.0, 7.0};
    const std::span<const double> first_two{weights.data(), 2};

    const auto snap = make_five_level({1.0, 2.0, 3.0, 4.0, 5.0},
                                      {5.0, 4.0, 3.0, 2.0, 1.0});

    EXPECT_NEAR(calculate_weighted_obi(snap, first_two), -5.0 / 9.0, 1e-12,
                "t16_span_length_bounds_depth");

    // Same book, same array, full span: proof the deep weights are not inert.
    EXPECT_NEAR(calculate_weighted_obi(snap, std::span<const double>{weights}),
                37.0 / 135.0, 1e-12, "t16_full_span_reads_deep_levels");
}


// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Runner
//
// With no arguments, runs everything. With a substring argument, runs only the
// tests whose name contains it:
//
//     ./test_signals            # all of them
//     ./test_signals t16        # just the weight-span bound
//     ./test_signals weighted   # t9-t16
//
// The filter exists because one test failing is not the only way a suite can
// stop: a test that reads out of bounds takes the process down with it, and
// every test after it goes unevaluated. run_mutation_check.py uses the filter to
// ask a single test for its verdict when an earlier one crashes the run.
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const std::string_view filter = (argc > 1) ? argv[1] : std::string_view{};

    const struct { void (*fn)(); std::string_view name; } tests[] = {
        {t1_bid_skewed,                     "t1_bid_skewed"},
        {t2_ask_skewed,                     "t2_ask_skewed"},
        {t3_balanced,                       "t3_balanced"},
        {t4_zero_volume,                    "t4_zero_volume"},
        {t5_multi_level,                    "t5_multi_level"},
        {t6_sparse_book,                    "t6_sparse_book"},
        {t7_delta_positive,                 "t7_delta_positive"},
        {t8_delta_zero,                     "t8_delta_zero"},
        {nullptr,                           "-- weighted OBI --"},
        {t9_flat_reproduces_obi,            "t9_flat_reproduces_obi"},
        {t10_weighted_bounded,              "t10_weighted_bounded"},
        {t11_decay_favours_touch,           "t11_decay_favours_touch"},
        {t12_l1_only_ignores_depth,         "t12_l1_only_ignores_depth"},
        {t13_weighted_degenerate,           "t13_weighted_degenerate"},
        {t14_weighted_sparse_book,          "t14_weighted_sparse_book"},
        {t15_profiles_distinct,             "t15_profiles_distinct"},
        {t16_weight_span_length_bounds_depth,
                                            "t16_weight_span_length_bounds_depth"},
    };

    std::printf("\n=== CrossFlux Signal Tests ===\n\n");
    if (!filter.empty()) {
        std::printf("  (filter: \"%.*s\")\n\n",
                    static_cast<int>(filter.size()), filter.data());
    }

    int selected = 0;
    for (const auto& t : tests) {
        if (t.fn == nullptr) {                       // section header
            if (filter.empty()) {
                std::printf("\n  %.*s\n",
                            static_cast<int>(t.name.size()), t.name.data());
            }
            continue;
        }
        if (!filter.empty() && t.name.find(filter) == std::string_view::npos) {
            continue;
        }
        ++selected;
        t.fn();
    }

    std::printf("\n");

    // An empty selection must not look like success: a typo'd filter would
    // otherwise print "All 0 tests PASSED" and exit 0.
    if (selected == 0) {
        std::fprintf(stderr, "No test matched \"%.*s\".\n\n",
                     static_cast<int>(filter.size()), filter.data());
        return EXIT_FAILURE;
    }

    if (g_failures == 0) {
        std::printf("All %d tests PASSED.\n\n", g_total);
        return EXIT_SUCCESS;
    } else {
        std::fprintf(stderr, "%d / %d tests FAILED.\n\n", g_failures, g_total);
        return EXIT_FAILURE;
    }
}
