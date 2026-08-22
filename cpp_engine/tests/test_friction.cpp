/**
 * @file    test_friction.cpp
 * @brief   Standalone verification suite for friction.hpp and the asynchronous
 *          execution path in execution_manager.hpp.
 *
 * Zero external dependencies (no gtest, no cmake).
 *
 * What this file is defending
 * --------------------------
 * TEST_REPORT.md §2.1 documents how the backtest came to report a 100% win rate
 * on 307,881 trades: the entry gate and the PnL calculation read the same static
 * snapshot, so `margin > fee` at entry and `margin - fee` at booking could not
 * disagree. Every trade was profitable by construction.
 *
 * The fix is not "add a cost". A cost that is followed by a re-check of the edge
 * reproduces the bug in a more expensive costume, because rejecting the trades
 * that friction ruined keeps exactly the ones it spared. The fix is that a
 * signal at T books nothing at T, each leg is priced later against whatever book
 * has printed by then, and whatever that arithmetic produces is booked —
 * including a loss, with no branch anywhere capable of declining one.
 *
 * So the load-bearing tests here are the ones that assert a *loss*. If e3 and e5
 * ever start passing while asserting a profit, or start reporting zero fills, the
 * tautology is back.
 *
 * Test inventory
 * --------------
 * walk_book — VWAP, the "don't assume the touch absorbs the order" half
 *   w1_single_level          qty inside level 0                  → vwap == touch, 1 level
 *   w2_walks_two_levels      0.6@100.0 + 0.4@100.5              → vwap == 100.2 exactly
 *   w3_walks_three_levels    consumes outward from the touch     → vwap == 100.6
 *   w4_partial_fill          book thinner than the order         → shortfall, partial()
 *   w5_zero_padding_no_depth zero-volume padding invents nothing → filled unchanged
 *   w6_levels_examined       levels_consumed counts examined     → 5, not 3
 *   w7_zero_qty_sentinel     qty 0                               → vwap 0.0 sentinel
 *   w8_negative_qty          qty < 0                             → sentinel, requested 0
 *   w9_empty_book            no levels                           → sentinel
 *   w10_bids_walk_downward   walk_bids consumes the highest bid first
 *   w11_slippage_unsigned    both sides report a positive cost
 *   w12_vwap_bounded         touch <= vwap <= worst level touched
 *   w13_dead_book            priced levels, zero volume          → sentinel, not a price
 *
 * FrictionModel — the configurable latency half
 *   f1_presets_match_python  the four presets equal src/friction.py's PRESETS
 *   f2_deterministic_draw    zero jitter → exactly latency_ms, rng untouched
 *   f3_jitter_median         retail: median of 1001 draws ≈ 250 ms
 *   f4_quantile_median       quantile(0.5) == latency_ms under jitter
 *   f5_retail_p99            retail p99 ≈ 800 ms (the documented claim)
 *   f6_quantile_monotone     p ↑ ⇒ latency ↑
 *   f7_legging_cost          |residual| * price * bps/1e4, sign-symmetric
 *   f8_valid_rejects_negative
 *   f9_get_preset_by_name
 *   f10_unknown_preset_reports_failure   a typo must not silently mean "zero"
 *
 * PendingOrderQueue — the buffering half
 *   q1_nothing_due_yet       drain before the fill time books nothing
 *   q2_books_when_due        drain at the fill time books once
 *   q3_one_leg_is_not_enough
 *   q4_fill_order_not_submission_order   a later signal's leg can resolve first
 *   q5_tie_break_deterministic           equal times → seq, then buy before sell
 *   q6_resolve_sees_leg_time
 *   q7_overflow_counted_and_refused
 *   q8_slots_recycled
 *   q9_reentrant_submit_refused
 *   q10_rewound_time_clamped
 *   q11_flush_to_infinity
 *   q12_zero_capacity_clamped
 *   q13_resolved_order_arithmetic        hedged/residual/legged/complete/empty
 *   q14_ready_at_and_leg_gap
 *
 * SimulatedExecutor — the two halves together, end to end
 *   e1_submit_books_nothing         a signal at T books nothing at T
 *   e2_fill_prices_are_vwap         booked PnL matches the hand-derived VWAPs
 *   e3_spread_collapse_books_loss   ← requirement #4, first clause
 *   e5_size_alone_books_loss        ← requirement #4, second clause
 *   e4_size_degrades_fill
 *   e6_partial_fill_charges_legging
 *   e7_no_liquidity_is_not_a_fill
 *   e8_gate_reads_signal_book_only
 *   e9_latency_defers_the_fill
 *   e10_sync_path_books_loss_and_flags_clocks
 *
 * Whether these tests check anything is itself checked, two ways:
 *
 *   cpp_engine/tests/run_mutation_check.py  breaks the implementation on purpose
 *     and requires this suite to notice. Five edits target this code: filling at
 *     the touch instead of walking (F1), dropping the drain's fill-time ordering
 *     (F2), hedging on the larger leg instead of the smaller (F3), and — the one
 *     that matters — reinstating the profitability gate at fill time (F4), which
 *     is the original tautology. F0 is a no-op control that must survive.
 *     Measured: F1 fails 25 checks, F2 fails 2, F3 fails 2, F4 fails 8.
 *
 *   cpp_engine/tests/run_parity_check.py    asserts walk_book returns the *same
 *     double* as src/friction.py's reference loop over 1,370 cases × 6 fields,
 *     via cpp_engine/tests/parity_walk_book.cpp. Value tests in this file pin
 *     hand-derived decimals; that harness owns the bit-exactness claim.
 *
 * Build command (direct g++/clang++, no cmake required):
 *   g++ -std=c++20 -Wall -Wextra -O2 -ffp-contract=off \
 *       -I cpp_engine/include \
 *       cpp_engine/tests/test_friction.cpp \
 *       cpp_engine/src/execution_manager.cpp \
 *       -o /tmp/test_friction && /tmp/test_friction
 *
 * -ffp-contract=off matches what the project ships and what the parity harness
 * builds with. Without it GCC fuses `notional += take * price` into an FMA, which
 * moves the vwap by 1 ULP — invisible in this file's tolerances, fatal to the
 * bit-exact claim above.
 *
 * C++20 required (std::span, designated initialisers, <numbers>-era constexpr).
 */

#include <algorithm>   // std::sort
#include <array>
#include <cmath>       // std::abs, std::isinf
#include <cstdio>      // std::printf, std::fprintf
#include <cstdlib>     // EXIT_SUCCESS, EXIT_FAILURE
#include <fstream>     // the CSV guard below
#include <limits>
#include <memory>      // std::make_shared
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "execution_manager.hpp"
#include "fee_config.hpp"
#include "friction.hpp"
#include "models.hpp"
#include "risk.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Harness — deliberately the same three macros as test_signals.cpp.
//
// A second assertion vocabulary in the same test directory is a tax on every
// future reader, so this file borrows rather than improves.
// ─────────────────────────────────────────────────────────────────────────────

static int g_failures = 0;
static int g_total    = 0;

#define EXPECT_NEAR(actual, expected, tol, name)                              \
    do {                                                                      \
        ++g_total;                                                            \
        const double _a = static_cast<double>(actual);                        \
        const double _e = static_cast<double>(expected);                      \
        const double _t = static_cast<double>(tol);                           \
        if (std::abs(_a - _e) <= _t) {                                        \
            std::printf("  PASS  %-38s  got=%.8f  expected=%.8f\n",           \
                        name, _a, _e);                                        \
        } else {                                                              \
            std::fprintf(stderr,                                              \
                "  FAIL  %-38s  got=%.8f  expected=%.8f  tol=%.2e\n",         \
                name, _a, _e, _t);                                            \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

/// Exact equality, for the places where exactness is the claim (0.0 sentinels,
/// counters, and the VWAPs that src/friction.py must reproduce bit for bit).
#define EXPECT_EQ(actual, expected, name) EXPECT_NEAR(actual, expected, 0.0, name)

/// A boolean property, routed through the same counters so it cannot be lost
/// among the numeric checks.
#define EXPECT_TRUE(cond, name) EXPECT_EQ((cond) ? 1.0 : 0.0, 1.0, name)

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using namespace crossflux;
using friction::BookWalk;
using friction::PendingOrder;
using friction::PendingOrderQueue;

using Book = OrderBookSnapshot<5>;

/// Build a 5-level snapshot from explicit levels, depth taken from the counts.
///
/// Goes through the validating factory rather than aggregate-initialising the
/// struct, so a fixture that accidentally describes a crossed or unsorted book
/// throws here instead of quietly producing a nonsense expectation.
[[nodiscard]] Book make_book(std::string_view ex, uint64_t ts,
                             std::vector<PriceLevel> bid_levels,
                             std::vector<PriceLevel> ask_levels)
{
    std::array<PriceLevel, 5> bids{};
    std::array<PriceLevel, 5> asks{};
    for (std::size_t i = 0; i < bid_levels.size() && i < 5; ++i) bids[i] = bid_levels[i];
    for (std::size_t i = 0; i < ask_levels.size() && i < 5; ++i) asks[i] = ask_levels[i];
    return make_order_book_snapshot<5>(
        ts, ex, bids, asks,
        static_cast<uint8_t>(bid_levels.size()),
        static_cast<uint8_t>(ask_levels.size()));
}

/// A signal that clears every risk guard, so a test that expects a rejection is
/// testing the gate it means to test.
[[nodiscard]] ArbitrageSignal make_signal(uint64_t ts, TradeAction action,
                                          double obi = 0.5)
{
    ArbitrageSignal s{};
    s.timestamp_ms = ts;
    s.obi_delta    = obi;
    s.p_execute    = 1.0;
    s.action       = action;
    return s;
}

/// SimulatedExecutor's constructor truncates /tmp/live_orders.csv, which is the
/// file the live dashboard tails. Running the test suite would otherwise wipe a
/// running demo's feed — a side effect no test is entitled to. Slurped here and
/// written back on the way out.
class CsvGuard {
public:
    CsvGuard() {
        std::ifstream in("/tmp/live_orders.csv", std::ios::binary);
        if (!in) { return; }
        std::ostringstream buf;
        buf << in.rdbuf();
        saved_ = buf.str();
        had_   = true;
    }
    ~CsvGuard() {
        if (!had_) { return; }
        std::ofstream out("/tmp/live_orders.csv", std::ios::binary | std::ios::trunc);
        out << saved_;
    }
    CsvGuard(const CsvGuard&) = delete;
    CsvGuard& operator=(const CsvGuard&) = delete;

private:
    std::string saved_;
    bool        had_{false};
};

[[nodiscard]] std::shared_ptr<CircuitBreaker> fresh_breaker() {
    return std::make_shared<CircuitBreaker>();
}

/// An executor configured for testing: no cooldown, so a test can book several
/// fills in market time without the second one being swallowed by the guard.
[[nodiscard]] SimulatedExecutor make_executor(
    friction::FrictionModel model = friction::preset_stress(),
    double min_profit_bps = 2.0)
{
    return SimulatedExecutor(fresh_breaker(), "binance", "kraken",
                             /*max_position=*/5.0, /*cooldown_ms=*/0.0,
                             min_profit_bps, /*slippage_model_bps=*/0.5,
                             model);
}

const double kFeeBuy  = fees::taker_fee_for("binance");
const double kFeeSell = fees::taker_fee_for("kraken");

/// The three-level ask side used by most of the VWAP tests, with the last two
/// slots left as an exchange snapshot leaves them.
/// Cumulative: 0.6 @ 100.0, then 1.0 @ 100.2 avg, then 2.0 @ 100.6 avg.
///
/// The padding is `PriceLevel{}` rather than `{0.0, 0.0}` because PriceLevel's
/// two-argument constructor rejects a non-positive price — the zero-filled slot
/// is reachable only through the default constructor, which is exactly the state
/// walk_book has to recognise as "not a level".
const std::array<PriceLevel, 5> kAsks{{
    {100.0, 0.6}, {100.5, 0.4}, {101.0, 1.0}, PriceLevel{}, PriceLevel{}
}};

[[nodiscard]] std::span<const PriceLevel> asks(std::size_t depth) {
    return std::span<const PriceLevel>{kAsks.data(), depth};
}

// ─────────────────────────────────────────────────────────────────────────────
// walk_book — "do not assume the top of book can absorb our entire notional"
//
// Exactness policy: single-level fills are asserted with EXPECT_EQ because
// notional/taken is exact there. Multi-level VWAPs use a 1e-12 tolerance — not
// because the arithmetic is uncertain, but because pinning a hand-typed decimal
// to the last bit would be testing the compiler's constant folding rather than
// the walk. Bit-exact agreement with src/friction.py is a separate claim, owned
// by cpp_engine/tests/parity_walk_book.cpp.
// ─────────────────────────────────────────────────────────────────────────────

void w1_single_level()
{
    const BookWalk w = friction::walk_book(asks(3), 0.5);
    EXPECT_EQ(w.vwap, 100.0, "w1_vwap_is_touch");
    EXPECT_EQ(w.filled_qty, 0.5, "w1_filled");
    EXPECT_EQ(static_cast<double>(w.levels_consumed), 1.0, "w1_one_level");
    EXPECT_EQ(w.notional, 50.0, "w1_notional");
    EXPECT_EQ(w.shortfall(), 0.0, "w1_no_shortfall");
    EXPECT_TRUE(!w.partial(), "w1_not_partial");
}

void w2_walks_two_levels()
{
    // The whole point of the exercise: asking for 1.0 when the touch holds 0.6
    // does not fill 1.0 at the touch.
    const BookWalk w = friction::walk_book(asks(3), 1.0);
    EXPECT_NEAR(w.vwap, 100.2, 1e-12, "w2_vwap_blended");
    EXPECT_EQ(w.filled_qty, 1.0, "w2_filled_in_full");
    EXPECT_EQ(static_cast<double>(w.levels_consumed), 2.0, "w2_two_levels");
    EXPECT_NEAR(w.notional, 100.2, 1e-12, "w2_notional");
    EXPECT_NEAR(w.vwap * w.filled_qty, w.notional, 0.0, "w2_notional_consistent");
    EXPECT_TRUE(w.vwap > 100.0, "w2_worse_than_touch");
}

void w3_walks_three_levels()
{
    const BookWalk w = friction::walk_book(asks(3), 2.0);
    EXPECT_NEAR(w.vwap, 100.6, 1e-12, "w3_vwap_blended");
    EXPECT_EQ(w.filled_qty, 2.0, "w3_filled_in_full");
    EXPECT_EQ(static_cast<double>(w.levels_consumed), 3.0, "w3_three_levels");
}

void w4_partial_fill()
{
    // Three levels hold 2.0 in total. Asking for 3.0 gets 2.0 and a shortfall —
    // not an error, and not a rejection: this is what a real IOC does.
    const BookWalk w = friction::walk_book(asks(3), 3.0);
    EXPECT_EQ(w.filled_qty, 2.0, "w4_filled_what_existed");
    EXPECT_EQ(w.requested_qty, 3.0, "w4_requested_preserved");
    EXPECT_EQ(w.shortfall(), 1.0, "w4_shortfall");
    EXPECT_TRUE(w.partial(), "w4_partial");
    EXPECT_NEAR(w.vwap, 100.6, 1e-12, "w4_vwap_of_what_filled");
}

void w5_zero_padding_no_depth()
{
    // Exchange snapshots pad absent levels with zeros. Treating a zero-volume
    // level as depth would invent liquidity that is not there, and would do it
    // silently — the fill would simply look better than it was.
    const BookWalk padded = friction::walk_book(asks(5), 3.0);
    const BookWalk tight  = friction::walk_book(asks(3), 3.0);
    EXPECT_EQ(padded.filled_qty, tight.filled_qty, "w5_padding_adds_no_liquidity");
    EXPECT_EQ(padded.vwap, tight.vwap, "w5_padding_does_not_move_vwap");
    EXPECT_EQ(padded.shortfall(), 1.0, "w5_shortfall_survives_padding");
}

void w6_levels_examined()
{
    // levels_consumed counts levels *examined*, which is the documented meaning
    // and is not the same as levels that supplied anything. Asserted explicitly
    // because the two readings differ by exactly the zero padding, and a reader
    // who assumes the other one would misreport book depth usage.
    EXPECT_EQ(static_cast<double>(friction::walk_book(asks(5), 3.0).levels_consumed),
              5.0, "w6_examined_includes_padding");
    EXPECT_EQ(static_cast<double>(friction::walk_book(asks(3), 3.0).levels_consumed),
              3.0, "w6_examined_without_padding");
    // Stops as soon as the order is full rather than scanning the rest.
    EXPECT_EQ(static_cast<double>(friction::walk_book(asks(5), 0.6).levels_consumed),
              1.0, "w6_stops_when_filled");
}

void w7_zero_qty_sentinel()
{
    const BookWalk w = friction::walk_book(asks(3), 0.0);
    EXPECT_EQ(w.vwap, 0.0, "w7_vwap_sentinel");
    EXPECT_EQ(w.filled_qty, 0.0, "w7_nothing_filled");
    EXPECT_EQ(w.notional, 0.0, "w7_no_notional");
    EXPECT_EQ(static_cast<double>(w.levels_consumed), 0.0, "w7_no_levels_touched");
    EXPECT_TRUE(!w.partial(), "w7_zero_request_is_not_partial");
    // A vwap of 0.0 is "nothing filled", not "filled at zero". Every consumer
    // must read filled_qty first; subtracting this as a price would invent a
    // 100% spread and book a fictional profit the size of the notional.
    EXPECT_EQ(w.slippage_bps(100.0), 0.0, "w7_sentinel_reports_no_slippage");
}

void w8_negative_qty()
{
    const BookWalk w = friction::walk_book(asks(3), -1.0);
    EXPECT_EQ(w.vwap, 0.0, "w8_vwap_sentinel");
    EXPECT_EQ(w.filled_qty, 0.0, "w8_nothing_filled");
    EXPECT_EQ(w.requested_qty, 0.0, "w8_request_clamped_to_zero");
    EXPECT_EQ(w.shortfall(), 0.0, "w8_no_shortfall_from_negative");
}

void w9_empty_book()
{
    const BookWalk w = friction::walk_book(std::span<const PriceLevel>{}, 1.0);
    EXPECT_EQ(w.vwap, 0.0, "w9_vwap_sentinel");
    EXPECT_EQ(w.filled_qty, 0.0, "w9_nothing_filled");
    EXPECT_EQ(w.requested_qty, 1.0, "w9_request_preserved");
    EXPECT_TRUE(w.partial(), "w9_fully_unfilled_is_partial");
}

void w10_bids_walk_downward()
{
    const Book b = make_book("kraken", 1'700'000'000'000ULL,
                             {{99.5, 0.5}, {99.0, 0.5}, {98.0, 2.0}},
                             {{100.0, 1.0}});
    // Selling 1.0 takes the best bid first and then a worse one: 0.5 @ 99.5
    // plus 0.5 @ 99.0.
    const BookWalk w = friction::walk_bids(b, 1.0);
    EXPECT_NEAR(w.vwap, 99.25, 1e-12, "w10_vwap_blended_downward");
    EXPECT_EQ(w.filled_qty, 1.0, "w10_filled");
    EXPECT_TRUE(w.vwap < b.bids[0].price, "w10_worse_than_touch");
    // The ask side of the same book is untouched by a sell.
    EXPECT_NEAR(friction::walk_asks(b, 1.0).vwap, 100.0, 1e-12, "w10_asks_independent");
}

void w11_slippage_unsigned()
{
    const Book b = make_book("kraken", 1'700'000'000'000ULL,
                             {{99.5, 0.5}, {99.0, 0.5}},
                             {{100.0, 0.6}, {100.5, 0.4}});
    const BookWalk buy  = friction::walk_asks(b, 1.0);
    const BookWalk sell = friction::walk_bids(b, 1.0);
    // Buying above the touch: 100.2 vs 100.0 → 20 bps.
    EXPECT_NEAR(buy.slippage_bps(100.0), 20.0, 1e-9, "w11_buy_slippage");
    // Selling below it: 99.25 vs 99.5 → 25.13 bps. Reported positive, because
    // both are costs and opposite signs would let them cancel in an average.
    EXPECT_NEAR(sell.slippage_bps(99.5), 25.1256281407, 1e-9, "w11_sell_slippage");
    EXPECT_TRUE(sell.slippage_bps(99.5) > 0.0, "w11_sell_cost_is_positive");
    EXPECT_EQ(buy.slippage_bps(0.0), 0.0, "w11_no_touch_no_slippage");
}

void w12_vwap_bounded()
{
    // A blended price can never beat the touch, nor be worse than the last level
    // the order actually reached. Both bounds are properties of a walk that
    // consumes outward, so violating either means the loop is not doing that.
    for (const double q : {0.1, 0.6, 0.7, 1.0, 1.5, 2.0, 5.0}) {
        const BookWalk w = friction::walk_book(asks(3), q);
        if (w.filled_qty <= friction::kEpsQty) { continue; }
        EXPECT_TRUE(w.vwap >= 100.0 && w.vwap <= 101.0,
                    "w12_vwap_within_book");
    }
}

void w13_dead_book()
{
    // Priced levels with no volume. A book like this has quotes but nothing
    // behind them, and the answer must be the sentinel rather than a price.
    const std::array<PriceLevel, 3> dead{{{100.0, 0.0}, {100.5, 0.0}, {101.0, 0.0}}};
    const BookWalk w = friction::walk_book(
        std::span<const PriceLevel>{dead.data(), dead.size()}, 1.0);
    EXPECT_EQ(w.vwap, 0.0, "w13_vwap_sentinel");
    EXPECT_EQ(w.filled_qty, 0.0, "w13_nothing_filled");
    EXPECT_EQ(static_cast<double>(w.levels_consumed), 3.0, "w13_all_examined");
    EXPECT_TRUE(w.partial(), "w13_partial");
}

// ─────────────────────────────────────────────────────────────────────────────
// FrictionModel — the configurable latency parameter
//
// The presets are asserted against src/friction.py's PRESETS by value rather
// than by round-tripping a config file. Two implementations of the same model
// that disagree on what "stress" means would produce two different reports under
// the same label, which is worse than either being wrong on its own.
// ─────────────────────────────────────────────────────────────────────────────

void f1_presets_match_python()
{
    const auto stress = friction::preset_stress();
    EXPECT_EQ(stress.latency_ms, 100.0, "f1_stress_latency");
    EXPECT_EQ(stress.jitter_log_sigma, 0.0, "f1_stress_no_jitter");
    EXPECT_EQ(stress.legging_cost_bps, 5.0, "f1_stress_legging");
    EXPECT_TRUE(stress.name == "stress", "f1_stress_name");

    const auto colo = friction::preset_colocated();
    EXPECT_EQ(colo.latency_ms, 5.0, "f1_colocated_latency");
    EXPECT_EQ(colo.jitter_log_sigma, 0.0, "f1_colocated_no_jitter");
    EXPECT_EQ(colo.legging_cost_bps, 5.0, "f1_colocated_legging");

    const auto retail = friction::preset_retail();
    EXPECT_EQ(retail.latency_ms, 250.0, "f1_retail_latency");
    EXPECT_EQ(retail.jitter_log_sigma, 0.5, "f1_retail_jitter");
    EXPECT_EQ(retail.legging_cost_bps, 10.0, "f1_retail_legging");

    const auto zero = friction::preset_zero();
    EXPECT_EQ(zero.latency_ms, 0.0, "f1_zero_latency");
    EXPECT_EQ(zero.legging_cost_bps, 0.0, "f1_zero_legging");
    EXPECT_TRUE(zero.deterministic(), "f1_zero_deterministic");

    // The four are not the same model. A refactor that collapsed them would make
    // every preset-labelled result identical and every comparison vacuous.
    EXPECT_TRUE(stress.latency_ms != colo.latency_ms &&
                colo.latency_ms != retail.latency_ms &&
                retail.latency_ms != zero.latency_ms,
                "f1_presets_distinct");
}

void f2_deterministic_draw()
{
    const auto m = friction::preset_stress();
    EXPECT_TRUE(m.deterministic(), "f2_stress_is_deterministic");

    std::mt19937_64 used(42);
    std::mt19937_64 untouched(42);
    bool all_exact = true;
    for (int i = 0; i < 100; ++i) {
        all_exact = all_exact && (m.sample_latency(used) == 100.0);
    }
    EXPECT_TRUE(all_exact, "f2_every_draw_is_exactly_latency_ms");

    // A deterministic preset must not consume the generator. If it did, adding a
    // zero-jitter leg would shift every later draw, and two runs that differ only
    // in the number of rejected signals would produce different latencies.
    EXPECT_TRUE(used() == untouched(), "f2_generator_not_consumed");
}

void f3_jitter_median()
{
    // Multiplicative jitter: latency_ms * exp(sigma * Z). The median of that is
    // exactly latency_ms, which is the reason for preferring it to the earlier
    // additive form where latency_ms was neither the mean nor the median of
    // anything. Seeded, so this is a fixed number rather than a coin flip.
    const auto m = friction::preset_retail();
    EXPECT_TRUE(!m.deterministic(), "f3_retail_has_jitter");

    std::mt19937_64 rng(20260822);
    std::vector<double> draws;
    draws.reserve(1001);
    for (int i = 0; i < 1001; ++i) { draws.push_back(m.sample_latency(rng)); }
    std::sort(draws.begin(), draws.end());

    EXPECT_NEAR(draws[500], 250.0, 25.0, "f3_sample_median_near_250");
    EXPECT_TRUE(draws.front() < 100.0, "f3_left_tail_exists");
    EXPECT_TRUE(draws.back() > 700.0, "f3_right_tail_is_heavy");
    EXPECT_TRUE(draws.front() > 0.0, "f3_latency_never_negative");
    // Right-skewed, so the mean sits above the median. If these ever come out
    // equal the distribution has become symmetric and the tail is gone.
    double sum = 0.0;
    for (const double d : draws) { sum += d; }
    EXPECT_TRUE(sum / static_cast<double>(draws.size()) > draws[500],
                "f3_mean_above_median");
}

void f4_quantile_median()
{
    EXPECT_NEAR(friction::preset_retail().latency_quantile(0.5), 250.0, 1e-9,
                "f4_jittered_median_is_latency_ms");
    // With no jitter every quantile is the same single value.
    const auto m = friction::preset_stress();
    EXPECT_EQ(m.latency_quantile(0.01), 100.0, "f4_deterministic_p01");
    EXPECT_EQ(m.latency_quantile(0.99), 100.0, "f4_deterministic_p99");
}

void f5_retail_p99()
{
    // The number src/friction.py's preset description quotes to the reader:
    // "250 ms median with a heavy right tail (p99 about 800 ms)". If the model
    // changes, the prose is wrong, and this is what says so.
    EXPECT_NEAR(friction::preset_retail().latency_quantile(0.99), 800.0, 1.0,
                "f5_retail_p99_about_800ms");
}

void f6_quantile_monotone()
{
    const auto m = friction::preset_retail();
    double prev = -1.0;
    for (const double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        const double q = m.latency_quantile(p);
        EXPECT_TRUE(q > prev, "f6_quantile_increasing");
        prev = q;
    }
}

void f7_legging_cost()
{
    const auto m = friction::preset_stress();   // 5 bps
    // 0.6 BTC left naked at 100.5, unwound at 5 bps.
    EXPECT_NEAR(m.legging_cost(0.6, 100.5), 0.6 * 100.5 * (5.0 / 1e4), 0.0,
                "f7_cost_formula");
    // The residual is a quantity mismatch, and which side over-filled does not
    // change what it costs to flatten it.
    EXPECT_EQ(m.legging_cost(-0.6, 100.5), m.legging_cost(0.6, 100.5),
              "f7_sign_symmetric");
    EXPECT_EQ(m.legging_cost(0.0, 100.5), 0.0, "f7_no_residual_no_cost");
    EXPECT_EQ(friction::preset_zero().legging_cost(0.6, 100.5), 0.0,
              "f7_zero_preset_charges_nothing");
    // Scales linearly in both arguments.
    EXPECT_NEAR(m.legging_cost(1.2, 100.5), 2.0 * m.legging_cost(0.6, 100.5), 1e-15,
                "f7_linear_in_qty");
}

void f8_valid_rejects_negative()
{
    // Named locals rather than braced temporaries inside the macro: the commas in
    // an initialiser list would be read as macro argument separators.
    const friction::FrictionModel neg_latency{"bad", -1.0, 0.0, 0.0};
    const friction::FrictionModel neg_jitter{"bad", 100.0, -0.1, 0.0};
    const friction::FrictionModel neg_legging{"bad", 100.0, 0.0, -5.0};
    const friction::FrictionModel all_zero{"edge", 0.0, 0.0, 0.0};

    EXPECT_TRUE(friction::preset_stress().valid(), "f8_preset_is_valid");
    EXPECT_TRUE(!neg_latency.valid(), "f8_negative_latency_invalid");
    EXPECT_TRUE(!neg_jitter.valid(), "f8_negative_jitter_invalid");
    EXPECT_TRUE(!neg_legging.valid(), "f8_negative_legging_invalid");
    EXPECT_TRUE(all_zero.valid(), "f8_all_zero_is_valid");
}

void f9_get_preset_by_name()
{
    bool ok = false;
    EXPECT_EQ(friction::get_preset("stress", &ok).latency_ms, 100.0, "f9_stress");
    EXPECT_TRUE(ok, "f9_stress_ok");
    EXPECT_EQ(friction::get_preset("colocated", &ok).latency_ms, 5.0, "f9_colocated");
    EXPECT_EQ(friction::get_preset("retail", &ok).latency_ms, 250.0, "f9_retail");
    EXPECT_EQ(friction::get_preset("zero", &ok).latency_ms, 0.0, "f9_zero");
    EXPECT_TRUE(ok, "f9_zero_ok");
}

void f10_unknown_preset_reports_failure()
{
    // A typo must not silently select the frictionless model. That would hand
    // back the configuration which produces the tautological 100% win rate, in
    // response to a misspelling, and the report would carry the label the caller
    // asked for rather than the model it got.
    bool ok = true;
    const auto m = friction::get_preset("strss", &ok);
    EXPECT_TRUE(!ok, "f10_unknown_name_reports_not_ok");
    EXPECT_EQ(m.latency_ms, 0.0, "f10_falls_back_to_zero_model");
    EXPECT_TRUE(friction::get_preset("", &ok).latency_ms == 0.0, "f10_empty_name");
    EXPECT_TRUE(!ok, "f10_empty_name_reports_not_ok");
    // The out-parameter is optional and omitting it must not crash.
    EXPECT_EQ(friction::get_preset("nope").latency_ms, 0.0, "f10_no_out_param");
}

// ─────────────────────────────────────────────────────────────────────────────
// PendingOrderQueue — "do not book the PnL immediately"
//
// The unit of scheduling is the leg, not the order. Under a jittered preset the
// buy leg of a later signal can resolve before the sell leg of an earlier one, so
// insertion order is not fill order; a FIFO would price legs against the wrong
// books and would do it silently, because every fill would still look plausible.
// q4 is the test that would catch that.
// ─────────────────────────────────────────────────────────────────────────────

/// A pending order whose two legs are due at the given absolute times.
[[nodiscard]] PendingOrder mk_order(double ts, double buy_at, double sell_at,
                                    double qty = 1.0)
{
    PendingOrder o{};
    o.signal_ts_ms    = ts;
    o.qty             = qty;
    o.signal_edge_bps = 50.0;
    o.obi_delta       = 0.5;
    o.buy  = friction::PendingLeg{buy_at,  100.0, friction::Venue::kA};
    o.sell = friction::PendingLeg{sell_at, 100.5, friction::Venue::kB};
    return o;
}

/// A complete fill of `qty` at `px`.
[[nodiscard]] BookWalk full_walk(double qty, double px) {
    return BookWalk{px, qty, 1, px * qty, qty};
}

/// A fill that got `got` of the `req` it asked for.
[[nodiscard]] BookWalk short_walk(double req, double got, double px) {
    return BookWalk{px, got, 3, px * got, req};
}

/// The resolve callback most queue tests want: fill everything at a flat price,
/// so any difference in the booked result comes from the scheduling rather than
/// from the prices.
const auto fill_everything = [](const PendingOrder& p, friction::Leg leg, double) {
    return full_walk(p.qty, leg == friction::Leg::kBuy ? 100.0 : 100.5);
};

void q1_nothing_due_yet()
{
    PendingOrderQueue q(8);
    EXPECT_TRUE(q.submit(mk_order(1000.0, 1100.0, 1100.0)), "q1_submitted");
    EXPECT_EQ(q.next_due_ms(), 1100.0, "q1_next_due");
    EXPECT_EQ(static_cast<double>(q.in_flight()), 1.0, "q1_in_flight");
    EXPECT_EQ(static_cast<double>(q.legs_pending()), 2.0, "q1_two_legs_queued");

    // One millisecond early is early. This is the assertion that a signal at T
    // books nothing at T.
    const std::size_t n = q.drain(1099.0, fill_everything,
                                  [](const friction::ResolvedOrder&) {});
    EXPECT_EQ(static_cast<double>(n), 0.0, "q1_nothing_booked");
    EXPECT_EQ(static_cast<double>(q.resolved_legs()), 0.0, "q1_no_legs_resolved");
    EXPECT_EQ(static_cast<double>(q.in_flight()), 1.0, "q1_still_in_flight");
}

void q2_books_when_due()
{
    PendingOrderQueue q(8);
    (void)q.submit(mk_order(1000.0, 1100.0, 1100.0));
    int booked = 0;
    const std::size_t n = q.drain(1100.0, fill_everything,
                                  [&](const friction::ResolvedOrder&) { ++booked; });
    EXPECT_EQ(static_cast<double>(n), 1.0, "q2_one_booked");
    EXPECT_EQ(static_cast<double>(booked), 1.0, "q2_callback_ran_once");
    EXPECT_EQ(static_cast<double>(q.resolved_legs()), 2.0, "q2_both_legs_resolved");
    EXPECT_EQ(static_cast<double>(q.in_flight()), 0.0, "q2_slot_released");
    EXPECT_TRUE(q.empty(), "q2_queue_empty");
    EXPECT_TRUE(std::isinf(q.next_due_ms()), "q2_next_due_infinite_when_idle");
}

void q3_one_leg_is_not_enough()
{
    PendingOrderQueue q(8);
    (void)q.submit(mk_order(1000.0, 1100.0, 1200.0));
    EXPECT_EQ(static_cast<double>(q.drain(1100.0, fill_everything,
              [](const friction::ResolvedOrder&) {})), 0.0, "q3_half_filled_not_booked");
    EXPECT_EQ(static_cast<double>(q.resolved_legs()), 1.0, "q3_one_leg_resolved");
    EXPECT_EQ(static_cast<double>(q.in_flight()), 1.0, "q3_order_still_open");
    EXPECT_EQ(q.next_due_ms(), 1200.0, "q3_second_leg_still_due");
    EXPECT_EQ(static_cast<double>(q.drain(1200.0, fill_everything,
              [](const friction::ResolvedOrder&) {})), 1.0, "q3_booked_on_second_leg");
}

void q4_fill_order_not_submission_order()
{
    PendingOrderQueue q(8);
    // Order 0 is submitted first but its legs land at 300 and 400.
    // Order 1 is submitted second and its buy leg lands at 100 — before either
    // of order 0's. Under jitter this is ordinary, not exotic.
    (void)q.submit(mk_order(0.0, 300.0, 400.0));
    (void)q.submit(mk_order(0.0, 100.0, 500.0));

    std::vector<int>    resolved;   // seq * 10 + leg
    std::vector<double> times;
    std::vector<int>    booked;

    q.drain(std::numeric_limits<double>::infinity(),
            [&](const PendingOrder& p, friction::Leg leg, double at_ms) {
                resolved.push_back(static_cast<int>(p.seq) * 10 +
                                   (leg == friction::Leg::kBuy ? 0 : 1));
                times.push_back(at_ms);
                return full_walk(p.qty, 100.0);
            },
            [&](const friction::ResolvedOrder& r) {
                booked.push_back(static_cast<int>(r.order.seq));
            });

    EXPECT_EQ(static_cast<double>(resolved.size()), 4.0, "q4_four_legs_resolved");
    const std::vector<int>    want_resolved{10, 0, 1, 11};
    const std::vector<double> want_times{100.0, 300.0, 400.0, 500.0};
    EXPECT_TRUE(resolved == want_resolved, "q4_legs_resolve_in_time_order");
    EXPECT_TRUE(times == want_times, "q4_leg_times_ascending");
    // Order 0 completes at 400, order 1 not until 500, so order 0 books first
    // even though its first leg resolved second.
    const std::vector<int> want_booked{0, 1};
    EXPECT_TRUE(booked == want_booked, "q4_books_when_second_leg_lands");
}

void q5_tie_break_deterministic()
{
    // Identical fill times. The order still has to be reproducible or a rerun of
    // the same backtest produces a different number, so ties break on sequence
    // and then buy before sell.
    PendingOrderQueue q(8);
    (void)q.submit(mk_order(0.0, 500.0, 500.0));
    (void)q.submit(mk_order(0.0, 500.0, 500.0));

    std::vector<int> resolved;
    q.drain(500.0,
            [&](const PendingOrder& p, friction::Leg leg, double) {
                resolved.push_back(static_cast<int>(p.seq) * 10 +
                                   (leg == friction::Leg::kBuy ? 0 : 1));
                return full_walk(p.qty, 100.0);
            },
            [](const friction::ResolvedOrder&) {});
    const std::vector<int> want{0, 1, 10, 11};
    EXPECT_TRUE(resolved == want, "q5_seq_then_leg");
}

void q6_resolve_sees_leg_time()
{
    // Each leg is handed its own fill time, which is what lets the caller look up
    // the book prevailing then. Handing both legs the order's completion time
    // would price the early leg against a book from the future.
    PendingOrderQueue q(4);
    (void)q.submit(mk_order(600.0, 700.0, 900.0));
    std::vector<double> times;
    q.drain(1000.0,
            [&](const PendingOrder& p, friction::Leg leg, double at_ms) {
                times.push_back(at_ms);
                return full_walk(p.qty, leg == friction::Leg::kBuy ? 100.0 : 100.5);
            },
            [](const friction::ResolvedOrder&) {});
    const std::vector<double> want{700.0, 900.0};
    EXPECT_TRUE(times == want, "q6_each_leg_gets_its_own_time");
}

void q7_overflow_counted_and_refused()
{
    // A full buffer must refuse loudly. Silently overwriting a slot would drop a
    // trade from the PnL and leave the win rate looking better for it.
    PendingOrderQueue q(2);
    EXPECT_EQ(static_cast<double>(q.capacity()), 2.0, "q7_capacity");
    EXPECT_TRUE(q.submit(mk_order(0.0, 100.0, 100.0)), "q7_first_accepted");
    EXPECT_TRUE(q.submit(mk_order(0.0, 100.0, 100.0)), "q7_second_accepted");
    EXPECT_TRUE(!q.submit(mk_order(0.0, 100.0, 100.0)), "q7_third_refused");
    EXPECT_EQ(static_cast<double>(q.overflowed()), 1.0, "q7_overflow_counted");
    EXPECT_EQ(static_cast<double>(q.submitted()), 2.0, "q7_submitted_excludes_refused");
    EXPECT_EQ(static_cast<double>(q.in_flight()), 2.0, "q7_in_flight");
}

void q8_slots_recycled()
{
    PendingOrderQueue q(2);
    (void)q.submit(mk_order(0.0, 100.0, 100.0));
    (void)q.submit(mk_order(0.0, 200.0, 200.0));
    EXPECT_EQ(static_cast<double>(q.drain(std::numeric_limits<double>::infinity(),
              fill_everything, [](const friction::ResolvedOrder&) {})), 2.0,
              "q8_both_booked");
    EXPECT_EQ(static_cast<double>(q.in_flight()), 0.0, "q8_drained");
    // Same two slots, reused. A leaked slot would show up here as a refusal.
    EXPECT_TRUE(q.submit(mk_order(300.0, 400.0, 400.0)), "q8_slot_reused_once");
    EXPECT_TRUE(q.submit(mk_order(300.0, 400.0, 400.0)), "q8_slot_reused_twice");
    EXPECT_EQ(static_cast<double>(q.overflowed()), 0.0, "q8_no_overflow");
    EXPECT_EQ(static_cast<double>(q.submitted()), 4.0, "q8_four_submitted_total");
}

void q9_reentrant_submit_refused()
{
    // With a zero-latency preset a booking callback could submit a leg that is
    // already due and the drain loop would never terminate. Refused and counted
    // rather than accepted-and-deferred, because a step budget would silently
    // drop work instead of telling the caller.
    PendingOrderQueue q(8);
    (void)q.submit(mk_order(1000.0, 1100.0, 1100.0));
    bool inner = true;
    q.drain(1100.0, fill_everything,
            [&](const friction::ResolvedOrder&) {
                inner = q.submit(mk_order(1100.0, 1100.0, 1100.0));
            });
    EXPECT_TRUE(!inner, "q9_reentrant_submit_returns_false");
    EXPECT_EQ(static_cast<double>(q.refused_reentrant()), 1.0, "q9_counted");
    EXPECT_EQ(static_cast<double>(q.booked()), 1.0, "q9_outer_order_still_booked");
    // And the queue is usable again once the drain has finished.
    EXPECT_TRUE(q.submit(mk_order(1100.0, 1200.0, 1200.0)), "q9_accepts_after_drain");
}

void q10_rewound_time_clamped()
{
    // Market data that goes backwards (a late row, an out-of-order feed) must not
    // strand an order that is already due: the clamp keeps time monotone and the
    // counter says it happened rather than hiding it.
    PendingOrderQueue q(4);
    (void)q.submit(mk_order(600.0, 700.0, 700.0));
    EXPECT_EQ(static_cast<double>(q.drain(1000.0, fill_everything,
              [](const friction::ResolvedOrder&) {})), 1.0, "q10_first_booked");
    EXPECT_EQ(static_cast<double>(q.rewound()), 0.0, "q10_no_rewind_yet");

    (void)q.submit(mk_order(750.0, 800.0, 800.0));
    EXPECT_EQ(static_cast<double>(q.drain(500.0, fill_everything,
              [](const friction::ResolvedOrder&) {})), 1.0,
              "q10_rewound_drain_still_books");
    EXPECT_EQ(static_cast<double>(q.rewound()), 1.0, "q10_rewind_counted");
    EXPECT_TRUE(q.empty(), "q10_nothing_stranded");
}

void q11_flush_to_infinity()
{
    PendingOrderQueue q(8);
    for (int i = 0; i < 3; ++i) {
        (void)q.submit(mk_order(0.0, 1e9 + i, 1e9 + 2 * i));
    }
    EXPECT_EQ(static_cast<double>(q.drain(std::numeric_limits<double>::infinity(),
              fill_everything, [](const friction::ResolvedOrder&) {})), 3.0,
              "q11_flush_books_everything");
    EXPECT_TRUE(q.empty(), "q11_empty_after_flush");
    EXPECT_EQ(static_cast<double>(q.in_flight()), 0.0, "q11_no_slots_held");
}

void q12_zero_capacity_clamped()
{
    // A zero-capacity queue would refuse every order and report a flat PnL of
    // zero, which reads as "no signals" rather than "misconfigured".
    PendingOrderQueue q(0);
    EXPECT_EQ(static_cast<double>(q.capacity()), 1.0, "q12_clamped_to_one");
    EXPECT_TRUE(q.submit(mk_order(0.0, 100.0, 100.0)), "q12_one_fits");
    EXPECT_TRUE(!q.submit(mk_order(0.0, 100.0, 100.0)), "q12_second_refused");
}

void q13_resolved_order_arithmetic()
{
    // Asymmetric fills: the buy venue supplied 0.4 of the 1.0 asked for, the sell
    // venue supplied all of it. Only the matched 0.4 is a hedge; the 0.6 is a
    // naked position and the class must not describe it as an arbitrage.
    friction::ResolvedOrder r{};
    r.order = mk_order(1000.0, 1100.0, 1100.0);
    r.buy   = short_walk(1.0, 0.4, 100.0);
    r.sell  = full_walk(1.0, 100.5);
    EXPECT_EQ(r.hedged_qty(), 0.4, "q13_hedged_is_the_minimum");
    EXPECT_NEAR(r.residual_qty(), 0.6, 1e-15, "q13_residual_is_the_difference");
    EXPECT_TRUE(r.legged(), "q13_legged");
    EXPECT_TRUE(!r.complete(), "q13_not_complete");
    EXPECT_TRUE(!r.empty(), "q13_not_empty");

    // Symmetric full fill.
    friction::ResolvedOrder ok{};
    ok.order = mk_order(1000.0, 1100.0, 1100.0);
    ok.buy   = full_walk(1.0, 100.0);
    ok.sell  = full_walk(1.0, 100.5);
    EXPECT_EQ(ok.hedged_qty(), 1.0, "q13_full_hedged");
    EXPECT_EQ(ok.residual_qty(), 0.0, "q13_full_no_residual");
    EXPECT_TRUE(!ok.legged(), "q13_full_not_legged");
    EXPECT_TRUE(ok.complete(), "q13_full_complete");

    // Both legs reached a book that could not supply anything. Note that these
    // are real walk results (requested 1.0, filled 0.0), not default-constructed
    // ones: complete() asks "was either leg short of what it requested", and a
    // default BookWalk requested nothing, so it would answer yes.
    friction::ResolvedOrder none{};
    none.order = mk_order(1000.0, 1100.0, 1100.0);
    none.buy   = short_walk(1.0, 0.0, 0.0);
    none.sell  = short_walk(1.0, 0.0, 0.0);
    EXPECT_EQ(none.hedged_qty(), 0.0, "q13_empty_hedged");
    EXPECT_EQ(none.residual_qty(), 0.0, "q13_empty_residual");
    EXPECT_TRUE(none.empty(), "q13_empty");
    EXPECT_TRUE(!none.legged(), "q13_empty_is_not_legged");
    EXPECT_TRUE(!none.complete(), "q13_empty_is_not_complete");
}

void q14_ready_at_and_leg_gap()
{
    // An order is done when its *slower* leg lands, and the gap between the two
    // is how long the position sat half-on. Both are read off whichever leg is
    // later, so neither may assume buy comes first.
    const PendingOrder late_sell = mk_order(1000.0, 1100.0, 1400.0);
    EXPECT_EQ(late_sell.ready_at_ms(), 1400.0, "q14_ready_is_max");
    EXPECT_EQ(late_sell.leg_gap_ms(), 300.0, "q14_gap_positive_direction");

    const PendingOrder late_buy = mk_order(1000.0, 1400.0, 1100.0);
    EXPECT_EQ(late_buy.ready_at_ms(), 1400.0, "q14_ready_is_max_either_way");
    EXPECT_EQ(late_buy.leg_gap_ms(), 300.0, "q14_gap_is_unsigned");

    const PendingOrder together = mk_order(1000.0, 1100.0, 1100.0);
    EXPECT_EQ(together.leg_gap_ms(), 0.0, "q14_no_gap_without_jitter");
}

// ─────────────────────────────────────────────────────────────────────────────
// SimulatedExecutor — both halves, end to end
//
// These are the tests the whole exercise is for. e3 and e5 assert that the engine
// books a *loss*, one for each clause of the requirement:
//
//   e3   the spread collapsed inside the latency window
//   e5   nothing moved at all, and walking the book for size cost more than the
//        edge was worth
//
// If either starts reporting zero fills, or a profit, the trade is being declined
// after the fact and the tautology is back.
//
// Fees are read from fee_config.hpp rather than hardcoded, so these expectations
// hold under all three fee presets. What is being pinned is the *composition* —
// gross on the hedged quantity only, fees on each leg's own notional, legging on
// the residual — not any particular schedule's numbers.
// ─────────────────────────────────────────────────────────────────────────────

/// Signal-time books with a 60 bps cross-venue edge and depth to spare.
[[nodiscard]] Book deep_a(uint64_t ts) {
    return make_book("binance", ts, {{99.9, 5.0}}, {{100.0, 5.0}});
}
[[nodiscard]] Book deep_b(uint64_t ts) {
    return make_book("kraken", ts, {{100.5, 5.0}}, {{100.6, 5.0}});
}

void e1_submit_books_nothing()
{
    auto ex = make_executor();
    const Book a = deep_a(1000), b = deep_b(1000);
    const auto sig = make_signal(1000, TradeAction::BUY_A_SELL_B);

    EXPECT_TRUE(ex.submit(sig, a, b, 1.0), "e1_accepted");
    // The claim: at T the order exists and the PnL does not.
    EXPECT_EQ(static_cast<double>(ex.pending_orders()), 1.0, "e1_buffered");
    EXPECT_EQ(static_cast<double>(ex.filled_orders()), 0.0, "e1_nothing_filled_at_T");
    EXPECT_EQ(ex.realized_pnl(), 0.0, "e1_no_pnl_at_T");
    EXPECT_EQ(static_cast<double>(ex.total_orders()), 1.0, "e1_signal_counted");
    EXPECT_EQ(static_cast<double>(ex.dropped_orders()), 0.0, "e1_nothing_dropped");

    EXPECT_EQ(static_cast<double>(ex.resolve_due(1099.0, a, b)), 0.0,
              "e1_one_ms_early_is_early");
    EXPECT_EQ(ex.realized_pnl(), 0.0, "e1_still_no_pnl");
    EXPECT_EQ(static_cast<double>(ex.resolve_due(1100.0, a, b)), 1.0,
              "e1_books_at_T_plus_latency");
    EXPECT_EQ(static_cast<double>(ex.pending_orders()), 0.0, "e1_buffer_drained");
}

void e2_fill_prices_are_vwap()
{
    auto ex = make_executor();
    // 1.0 BTC against 0.6 at the touch on both venues, so both legs blend.
    const Book a = make_book("binance", 1000, {{99.9, 5.0}},
                             {{100.0, 0.6}, {100.5, 0.4}});
    const Book b = make_book("kraken", 1000, {{100.6, 0.6}, {100.4, 0.4}},
                             {{100.7, 5.0}});
    EXPECT_TRUE(ex.submit(make_signal(1000, TradeAction::BUY_A_SELL_B), a, b, 1.0),
                "e2_accepted");
    EXPECT_EQ(static_cast<double>(ex.resolve_due(1100.0, a, b)), 1.0, "e2_booked");

    const double buy_vwap  = 0.6 * 100.0 + 0.4 * 100.5;   // 100.2, /1.0
    const double sell_vwap = 0.6 * 100.6 + 0.4 * 100.4;   // 100.52
    const double gross     = (sell_vwap - buy_vwap) * 1.0;
    const double fee       = buy_vwap * kFeeBuy + sell_vwap * kFeeSell;

    EXPECT_NEAR(ex.realized_pnl(), gross - fee, 1e-9, "e2_pnl_from_vwaps");
    EXPECT_EQ(static_cast<double>(ex.filled_orders()), 1.0, "e2_one_fill");
    EXPECT_EQ(static_cast<double>(ex.legged_fills()), 0.0, "e2_fully_hedged");
    EXPECT_EQ(static_cast<double>(ex.adverse_selection_fills()), 0.0, "e2_not_adverse");
    EXPECT_TRUE(ex.realized_pnl() > 0.0, "e2_this_one_is_a_winner");
    // The fill did not get the touch price on either side. Booking at the touch
    // is the assumption the requirement names.
    EXPECT_TRUE(buy_vwap > 100.0 && sell_vwap < 100.6, "e2_worse_than_touch");
}

void e3_spread_collapse_books_loss()
{
    // Requirement #4, first clause. Signal sees a 50 bps edge; 100 ms later the
    // sell venue's bid has dropped a full point and the trade is under water. It
    // executes anyway, because by then it is already in the market.
    auto ex = make_executor();
    const Book a_sig = deep_a(1000), b_sig = deep_b(1000);
    EXPECT_TRUE(ex.submit(make_signal(1000, TradeAction::BUY_A_SELL_B),
                          a_sig, b_sig, 1.0), "e3_accepted_on_positive_edge");

    const Book a_fill = deep_a(1100);
    const Book b_fill = make_book("kraken", 1100, {{99.0, 5.0}}, {{99.1, 5.0}});

    EXPECT_EQ(static_cast<double>(ex.resolve_due(1100.0, a_fill, b_fill)), 1.0,
              "e3_booked_not_rejected");
    EXPECT_EQ(static_cast<double>(ex.filled_orders()), 1.0, "e3_counted_as_a_fill");
    EXPECT_EQ(static_cast<double>(ex.adverse_selection_fills()), 1.0,
              "e3_logged_as_adverse_selection");
    EXPECT_EQ(static_cast<double>(ex.legged_fills()), 0.0, "e3_both_legs_filled");
    EXPECT_TRUE(ex.realized_pnl() < 0.0, "e3_booked_a_loss");

    const double expect = (99.0 - 100.0) * 1.0
                        - (100.0 * kFeeBuy + 99.0 * kFeeSell);
    EXPECT_NEAR(ex.realized_pnl(), expect, 1e-9, "e3_loss_is_the_full_move");
    // And the loss reached the risk manager. A loss the circuit breaker never
    // sees cannot trip it, which is the failure mode worth guarding.
    EXPECT_NEAR(ex.risk_mgr()->get_cumulative_pnl(), ex.realized_pnl(), 0.0,
                "e3_loss_recorded_against_the_breaker");
}

void e4_size_degrades_fill()
{
    // Same book, same latency, same everything but size. Five levels deep on each
    // side, and the only difference between the two runs is how far down the order
    // has to reach.
    const Book a = make_book("binance", 1000, {{99.9, 5.0}},
                             {{100.0, 0.5}, {100.5, 0.5}, {101.0, 5.0}});
    const Book b = make_book("kraken", 1000,
                             {{100.6, 0.5}, {100.4, 0.5}, {100.0, 5.0}},
                             {{100.7, 5.0}});
    const auto sig = make_signal(1000, TradeAction::BUY_A_SELL_B);

    auto small = make_executor();
    (void)small.submit(sig, a, b, 0.5);
    (void)small.resolve_due(1100.0, a, b);

    auto big = make_executor();
    (void)big.submit(sig, a, b, 2.0);
    (void)big.resolve_due(1100.0, a, b);

    EXPECT_EQ(static_cast<double>(small.filled_orders()), 1.0, "e4_small_filled");
    EXPECT_EQ(static_cast<double>(big.filled_orders()), 1.0, "e4_big_filled");
    EXPECT_TRUE(small.realized_pnl() / 0.5 > big.realized_pnl() / 2.0,
                "e4_per_unit_pnl_falls_with_size");
    EXPECT_TRUE(small.realized_pnl() > 0.0, "e4_small_is_profitable");
}

void e5_size_alone_books_loss()
{
    // Requirement #4, second clause, and the cleaner of the two results: the book
    // does not move at all between signal and fill — the same two snapshots are
    // passed to submit() and resolve_due(). The loss is entirely the cost of
    // asking for 2.0 BTC when 0.5 sits at the touch.
    const Book a = make_book("binance", 1000, {{99.9, 5.0}},
                             {{100.0, 0.5}, {100.5, 0.5}, {101.0, 5.0}});
    const Book b = make_book("kraken", 1000,
                             {{100.6, 0.5}, {100.4, 0.5}, {100.0, 5.0}},
                             {{100.7, 5.0}});
    auto ex = make_executor();
    EXPECT_TRUE(ex.submit(make_signal(1000, TradeAction::BUY_A_SELL_B), a, b, 2.0),
                "e5_accepted_on_60bps_touch_edge");
    EXPECT_EQ(static_cast<double>(ex.resolve_due(1100.0, a, b)), 1.0, "e5_booked");

    const double buy_vwap  = (0.5 * 100.0 + 0.5 * 100.5 + 1.0 * 101.0) / 2.0;  // 100.625
    const double sell_vwap = (0.5 * 100.6 + 0.5 * 100.4 + 1.0 * 100.0) / 2.0;  // 100.25
    const double gross     = (sell_vwap - buy_vwap) * 2.0;                     // -0.75
    const double fee       = (buy_vwap * 2.0) * kFeeBuy + (sell_vwap * 2.0) * kFeeSell;

    EXPECT_TRUE(gross < 0.0, "e5_fixture_really_does_invert_the_edge");
    EXPECT_EQ(static_cast<double>(ex.adverse_selection_fills()), 1.0, "e5_adverse");
    EXPECT_TRUE(ex.realized_pnl() < 0.0, "e5_booked_a_loss_on_a_static_book");
    EXPECT_NEAR(ex.realized_pnl(), gross - fee, 1e-9, "e5_loss_is_the_vwap_gap");
}

void e6_partial_fill_charges_legging()
{
    // The buy venue can only supply 0.4 of the 1.0; the sell venue fills all of
    // it. Only 0.4 is hedged, 0.6 is naked, and the naked part is charged rather
    // than quietly credited or dropped.
    const Book a = make_book("binance", 1000, {{99.9, 5.0}}, {{100.0, 0.4}});
    const Book b = deep_b(1000);
    auto ex = make_executor();
    EXPECT_TRUE(ex.submit(make_signal(1000, TradeAction::BUY_A_SELL_B), a, b, 1.0),
                "e6_accepted");
    EXPECT_EQ(static_cast<double>(ex.resolve_due(1100.0, a, b)), 1.0, "e6_booked");

    EXPECT_EQ(static_cast<double>(ex.legged_fills()), 1.0, "e6_logged_as_legged");
    EXPECT_EQ(static_cast<double>(ex.filled_orders()), 1.0, "e6_counted_as_a_fill");

    const double gross   = (100.5 - 100.0) * 0.4;                  // hedged only
    const double fee     = 40.0 * kFeeBuy + 100.5 * kFeeSell;      // each leg's own
    const double legging = 0.6 * 100.5 * (5.0 / 1e4);              // stress: 5 bps
    EXPECT_NEAR(ex.realized_pnl(), gross - fee - legging, 1e-9, "e6_pnl_decomposition");
    // Fees on the matched 0.4 rather than on each leg's own fill would be cheaper
    // by this much, and a fill that over-fills one side would look free.
    EXPECT_TRUE(fee > (40.0 * kFeeBuy + 40.2 * kFeeSell) || kFeeSell == 0.0,
                "e6_sell_leg_pays_for_all_of_what_it_filled");
}

void e7_no_liquidity_is_not_a_fill()
{
    // The order was live and reached a book with quotes but no volume behind
    // them. Nothing filled, so there is no PnL to book — but it is not a
    // rejection either, and it has its own counter so it cannot be mistaken for
    // one.
    auto ex = make_executor();
    const Book a_sig = deep_a(1000), b_sig = deep_b(1000);
    EXPECT_TRUE(ex.submit(make_signal(1000, TradeAction::BUY_A_SELL_B),
                          a_sig, b_sig, 1.0), "e7_accepted");
    const Book a_dead = make_book("binance", 1100, {{99.9, 0.0}}, {{100.0, 0.0}});
    const Book b_dead = make_book("kraken", 1100, {{100.5, 0.0}}, {{100.6, 0.0}});
    EXPECT_EQ(static_cast<double>(ex.resolve_due(1100.0, a_dead, b_dead)), 1.0,
              "e7_order_resolved");
    EXPECT_EQ(static_cast<double>(ex.filled_orders()), 0.0, "e7_not_a_fill");
    EXPECT_EQ(static_cast<double>(ex.empty_fills()), 1.0, "e7_counted_separately");
    EXPECT_EQ(ex.realized_pnl(), 0.0, "e7_no_pnl");
    EXPECT_EQ(static_cast<double>(ex.adverse_selection_fills()), 0.0,
              "e7_not_adverse_selection");
    EXPECT_EQ(static_cast<double>(ex.pending_orders()), 0.0, "e7_slot_released");
}

void e8_gate_reads_signal_book_only()
{
    // The one surviving gate. It is a decision about whether to *try*, taken from
    // the book at T, and it is configurable rather than hardcoded — the same
    // signal is refused at 2 bps and accepted at 0.5 bps.
    const Book a = make_book("binance", 1000, {{99.9, 5.0}}, {{100.0, 5.0}});
    const Book b = make_book("kraken", 1000, {{100.01, 5.0}}, {{100.02, 5.0}});
    const auto sig = make_signal(1000, TradeAction::BUY_A_SELL_B);   // edge 1 bp

    auto strict = make_executor(friction::preset_stress(), 2.0);
    EXPECT_TRUE(!strict.submit(sig, a, b, 1.0), "e8_refused_below_threshold");
    EXPECT_EQ(static_cast<double>(strict.pending_orders()), 0.0, "e8_nothing_buffered");
    EXPECT_EQ(static_cast<double>(strict.total_orders()), 1.0, "e8_signal_still_counted");

    auto loose = make_executor(friction::preset_stress(), 0.5);
    EXPECT_TRUE(loose.submit(sig, a, b, 1.0), "e8_accepted_above_threshold");
    EXPECT_EQ(static_cast<double>(loose.pending_orders()), 1.0, "e8_buffered");
}

void e9_latency_defers_the_fill()
{
    // Requirement #1, demonstrated rather than asserted: the configured latency
    // is what decides when the fill happens.
    const Book a = deep_a(1000), b = deep_b(1000);
    const auto sig = make_signal(1000, TradeAction::BUY_A_SELL_B);

    auto zero = make_executor(friction::preset_zero());
    EXPECT_EQ(zero.latency_ms(), 0.0, "e9_zero_preset_latency");
    (void)zero.submit(sig, a, b, 1.0);
    EXPECT_EQ(static_cast<double>(zero.resolve_due(1000.0, a, b)), 1.0,
              "e9_zero_latency_fills_at_T");

    auto stress = make_executor(friction::preset_stress());
    EXPECT_EQ(stress.latency_ms(), 100.0, "e9_stress_preset_latency");
    (void)stress.submit(sig, a, b, 1.0);
    EXPECT_EQ(static_cast<double>(stress.resolve_due(1000.0, a, b)), 0.0,
              "e9_stress_does_not_fill_at_T");
    EXPECT_EQ(static_cast<double>(stress.resolve_due(1100.0, a, b)), 1.0,
              "e9_stress_fills_at_T_plus_100");

    // The frictionless preset reads the signal's own book, which is the
    // configuration that produced the tautological win rate. Kept reachable on
    // purpose, and it must stay profitable here — that is what makes e3 and e5
    // evidence about friction rather than about a broken fixture.
    EXPECT_TRUE(zero.realized_pnl() > 0.0, "e9_zero_friction_still_wins");
}

void e10_sync_path_books_loss_and_flags_clocks()
{
    // The synchronous path has no depth to walk and no later book, so latency can
    // only be a price haircut there. What it must not do is re-check the edge and
    // reject: slippage of 2 bps a side against a 1 bp raw spread is a loss, and a
    // loss is what gets booked.
    SimulatedExecutor ex(fresh_breaker(), "binance", "kraken",
                         /*max_position=*/5.0, /*cooldown_ms=*/0.0,
                         /*min_profit_bps=*/0.5, /*slippage_model_bps=*/2.0);
    const auto sig = make_signal(1000, TradeAction::BUY_A_SELL_B);
    EXPECT_TRUE(!ex.mixed_clocks(), "e10_single_clock_initially");

    // ask_a = 100.0, bid_b = 100.01 → a 1 bp raw spread, which clears the 0.5 bp
    // gate and is then comfortably eaten by 4 bps of round-trip slippage.
    EXPECT_TRUE(ex.evaluate_and_execute(sig, 99.9, 100.0, 100.01, 100.02),
                "e10_executed_not_rejected");
    EXPECT_EQ(static_cast<double>(ex.filled_orders()), 1.0, "e10_counted_as_a_fill");
    EXPECT_TRUE(ex.realized_pnl() < 0.0, "e10_sync_path_books_a_loss");
    EXPECT_EQ(static_cast<double>(ex.adverse_selection_fills()), 1.0,
              "e10_logged_as_adverse_selection");

    // Now use the async path on the same executor. The two stamp fills from
    // different clocks — market time and wall clock — so the cooldown stops
    // meaning anything, and that is reported rather than silently tolerated.
    const Book a = deep_a(1000), b = deep_b(1000);
    (void)ex.submit(sig, a, b, 1.0);
    (void)ex.resolve_due(1100.0, a, b);
    EXPECT_TRUE(ex.mixed_clocks(), "e10_mixed_clocks_detected");
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Runner. Same shape as test_signals.cpp: a table with nullptr as a section
// header, an argv substring filter, and an empty selection treated as failure so
// a typo'd filter cannot print "All 0 tests PASSED" and exit 0.
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const std::string_view filter = (argc > 1) ? argv[1] : std::string_view{};

    // SimulatedExecutor's constructor truncates the dashboard's feed file.
    // Restored on the way out; see CsvGuard.
    const CsvGuard csv_guard;

    const struct { void (*fn)(); std::string_view name; } tests[] = {
        {nullptr,                        "-- walk_book (VWAP) --"},
        {w1_single_level,                "w1_single_level"},
        {w2_walks_two_levels,            "w2_walks_two_levels"},
        {w3_walks_three_levels,          "w3_walks_three_levels"},
        {w4_partial_fill,                "w4_partial_fill"},
        {w5_zero_padding_no_depth,       "w5_zero_padding_no_depth"},
        {w6_levels_examined,             "w6_levels_examined"},
        {w7_zero_qty_sentinel,           "w7_zero_qty_sentinel"},
        {w8_negative_qty,                "w8_negative_qty"},
        {w9_empty_book,                  "w9_empty_book"},
        {w10_bids_walk_downward,         "w10_bids_walk_downward"},
        {w11_slippage_unsigned,          "w11_slippage_unsigned"},
        {w12_vwap_bounded,               "w12_vwap_bounded"},
        {w13_dead_book,                  "w13_dead_book"},

        {nullptr,                        "-- FrictionModel (latency) --"},
        {f1_presets_match_python,        "f1_presets_match_python"},
        {f2_deterministic_draw,          "f2_deterministic_draw"},
        {f3_jitter_median,               "f3_jitter_median"},
        {f4_quantile_median,             "f4_quantile_median"},
        {f5_retail_p99,                  "f5_retail_p99"},
        {f6_quantile_monotone,           "f6_quantile_monotone"},
        {f7_legging_cost,                "f7_legging_cost"},
        {f8_valid_rejects_negative,      "f8_valid_rejects_negative"},
        {f9_get_preset_by_name,          "f9_get_preset_by_name"},
        {f10_unknown_preset_reports_failure,
                                         "f10_unknown_preset_reports_failure"},

        {nullptr,                        "-- PendingOrderQueue (buffering) --"},
        {q1_nothing_due_yet,             "q1_nothing_due_yet"},
        {q2_books_when_due,              "q2_books_when_due"},
        {q3_one_leg_is_not_enough,       "q3_one_leg_is_not_enough"},
        {q4_fill_order_not_submission_order,
                                         "q4_fill_order_not_submission_order"},
        {q5_tie_break_deterministic,     "q5_tie_break_deterministic"},
        {q6_resolve_sees_leg_time,       "q6_resolve_sees_leg_time"},
        {q7_overflow_counted_and_refused, "q7_overflow_counted_and_refused"},
        {q8_slots_recycled,              "q8_slots_recycled"},
        {q9_reentrant_submit_refused,    "q9_reentrant_submit_refused"},
        {q10_rewound_time_clamped,       "q10_rewound_time_clamped"},
        {q11_flush_to_infinity,          "q11_flush_to_infinity"},
        {q12_zero_capacity_clamped,      "q12_zero_capacity_clamped"},
        {q13_resolved_order_arithmetic,  "q13_resolved_order_arithmetic"},
        {q14_ready_at_and_leg_gap,       "q14_ready_at_and_leg_gap"},

        {nullptr,                        "-- SimulatedExecutor (end to end) --"},
        {e1_submit_books_nothing,        "e1_submit_books_nothing"},
        {e2_fill_prices_are_vwap,        "e2_fill_prices_are_vwap"},
        {e3_spread_collapse_books_loss,  "e3_spread_collapse_books_loss"},
        {e4_size_degrades_fill,          "e4_size_degrades_fill"},
        {e5_size_alone_books_loss,       "e5_size_alone_books_loss"},
        {e6_partial_fill_charges_legging, "e6_partial_fill_charges_legging"},
        {e7_no_liquidity_is_not_a_fill,  "e7_no_liquidity_is_not_a_fill"},
        {e8_gate_reads_signal_book_only, "e8_gate_reads_signal_book_only"},
        {e9_latency_defers_the_fill,     "e9_latency_defers_the_fill"},
        {e10_sync_path_books_loss_and_flags_clocks,
                                         "e10_sync_path_books_loss_and_flags_clocks"},
    };

    std::printf("\n=== CrossFlux Friction / Async Execution Tests ===\n");
    if (!filter.empty()) {
        std::printf("\n  (filter: \"%.*s\")\n",
                    static_cast<int>(filter.size()), filter.data());
    }

    int selected = 0;
    for (const auto& t : tests) {
        if (t.fn == nullptr) {
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

    if (selected == 0) {
        std::fprintf(stderr, "No test matched \"%.*s\".\n\n",
                     static_cast<int>(filter.size()), filter.data());
        return EXIT_FAILURE;
    }

    if (g_failures == 0) {
        std::printf("All %d checks PASSED (%d tests).\n\n", g_total, selected);
        return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "%d / %d checks FAILED.\n\n", g_failures, g_total);
    return EXIT_FAILURE;
}
