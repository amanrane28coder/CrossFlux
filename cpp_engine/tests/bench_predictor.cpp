/**
 * @file    bench_predictor.cpp
 * @brief   Throughput benchmark: 100,000 MarketTick<10> objects through SignalAggregator.
 *
 * Phase 7: Performance validation of the C++ decision core.
 *
 * Methodology
 * -----------
 *  1. PRE-BUILD phase (not timed):
 *       Generate 100,000 MarketTick<10> objects into a std::vector.
 *       Alternate volumes to produce ~50% Gate 1 pass rate, simulating
 *       a realistically mixed live data stream (not all-pass or all-fail).
 *
 *  2. BENCHMARK phase (timed with std::chrono::steady_clock):
 *       Call aggregator.evaluate(ticks) once.
 *       Measure wall-clock nanoseconds from first to last tick processed.
 *
 *  3. REPORT phase:
 *       Print total µs, per-tick ns, signal count, and throughput in M ticks/s.
 *
 * Design notes
 * ------------
 *  - steady_clock is monotonic and nanosecond-resolution on macOS arm64.
 *  - The result vector is passed to std::fprintf to prevent the compiler
 *    from eliminating the evaluate() call as dead code.
 *  - The pre-build phase includes a p_execute cross-check against the
 *    known Python reference value to validate numerical correctness.
 *
 * Expected results (Apple M-series, Release -O3 -march=native):
 *   Per-tick latency : ~50–200 ns
 *   Throughput       : ~5–20 million ticks/s
 *
 * C++20 required.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "models.hpp"
#include "predictor.hpp"

using namespace crossflux;
using Clock = std::chrono::steady_clock;


// ─────────────────────────────────────────────────────────────────────────────
// Fixture helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Build a single-level OrderBookSnapshot<10> with controlled bid/ask volumes.
[[nodiscard]] OrderBookSnapshot<10>
make_snap(const char* exchange, uint64_t ts, double bid_vol, double ask_vol)
{
    std::array<PriceLevel, 10> bids{};
    std::array<PriceLevel, 10> asks{};
    bids[0] = PriceLevel{49'999.0, bid_vol};
    asks[0] = PriceLevel{50'001.0, ask_vol};
    return make_order_book_snapshot<10>(
        ts, exchange, bids, asks, /*bid_depth=*/1, /*ask_depth=*/1);
}

/**
 * Generate N_TICKS MarketTick<10> objects with an alternating volume pattern
 * designed to yield ~50% Gate 1 pass rate.
 *
 * Even ticks  (i % 2 == 0):
 *   snap_a: bid=100, ask=1   → OBI_A ≈ +0.98  (bid-heavy)
 *   snap_b: bid=1,   ask=1   → OBI_B =  0.0   (balanced)
 *   delta = +0.98 > 0.3      → Gate 1 PASS
 *
 * Odd ticks (i % 2 == 1):
 *   snap_a: bid=1, ask=1     → OBI_A = 0.0   (balanced)
 *   snap_b: bid=1, ask=1     → OBI_B = 0.0   (balanced)
 *   delta = 0.0 <= 0.3       → Gate 1 FAIL
 */
[[nodiscard]] std::vector<MarketTick<10>>
build_ticks(std::size_t n_ticks)
{
    std::vector<MarketTick<10>> ticks;
    ticks.reserve(n_ticks);

    for (std::size_t i = 0; i < n_ticks; ++i) {
        const uint64_t ts = 1'700'000'000'000ULL + static_cast<uint64_t>(i);
        MarketTick<10> tick{};
        tick.timestamp_ms = ts;

        if (i % 2 == 0) {
            // Strongly imbalanced → |delta| ≈ 0.98 → Gate 1 PASS
            tick.snap_a = make_snap("binance", ts, 100.0, 1.0);
            tick.snap_b = make_snap("kraken",  ts,   1.0, 1.0);
        } else {
            // Balanced on both sides → delta = 0.0 → Gate 1 FAIL
            tick.snap_a = make_snap("binance", ts, 1.0, 1.0);
            tick.snap_b = make_snap("kraken",  ts, 1.0, 1.0);
        }

        ticks.push_back(tick);
    }

    return ticks;
}

} // anonymous namespace


// ─────────────────────────────────────────────────────────────────────────────
// Numerical cross-check against Python reference
// ─────────────────────────────────────────────────────────────────────────────

static void cross_check_p_execute(const SignalAggregator& agg)
{
    // Python reference: calculate_execution_probability(50.0, 3.5, 0.4)
    // i.e. scipy.stats.lognorm.cdf(50.0, s=0.4, scale=exp(3.5)) = 0.84850850
    // Verified against Python output; tolerance 1e-6 for cross-language comparison.
    const double python_ref = 0.84850850;
    const double cpp_val    = agg.p_execute();
    const double diff       = std::abs(cpp_val - python_ref);

    std::fprintf(stderr, "[CrossCheck] C++ p_execute=%.8f  Python_ref=%.8f  |diff|=%.2e  %s\n",
        cpp_val, python_ref, diff,
        (diff < 1e-6) ? "PASS" : "FAIL");
}


// ─────────────────────────────────────────────────────────────────────────────
// Main benchmark
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    constexpr std::size_t N_TICKS = 100'000;

    std::fprintf(stderr, "\n=== Argus Phase 7 Benchmark ===\n\n");

    // ── Construct aggregator (erf called here — not in hot loop) ──────────
    SignalAggregator agg{
        "binance", "kraken",
        /*latency_mu=*/    3.5,
        /*latency_sigma=*/ 0.4,
        /*alpha_ms=*/      50.0,
        /*delta_thresh=*/  0.3,
        /*min_p_execute=*/ 0.80
    };

    // ── Numerical cross-check ─────────────────────────────────────────────
    cross_check_p_execute(agg);
    std::fprintf(stderr, "\n");

    // ── Pre-build tick batch (NOT timed) ──────────────────────────────────
    std::fprintf(stderr, "Building %zu market ticks... ", N_TICKS);
    const auto ticks = build_ticks(N_TICKS);
    std::fprintf(stderr, "done.\n\n");

    // ── TIMED BENCHMARK ───────────────────────────────────────────────────
    const auto t0 = Clock::now();

    auto signals = agg.evaluate(ticks);

    const auto t1 = Clock::now();
    // ── END TIMED BENCHMARK ───────────────────────────────────────────────

    // ── Compute metrics ───────────────────────────────────────────────────
    const auto elapsed_ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const auto elapsed_us  = elapsed_ns / 1'000;
    const double per_tick_ns  = static_cast<double>(elapsed_ns) / static_cast<double>(N_TICKS);
    const double throughput_m = (static_cast<double>(N_TICKS) / static_cast<double>(elapsed_ns)) * 1'000.0;

    // ── Report ────────────────────────────────────────────────────────────
    std::fprintf(stderr, "─────────────────────────────────────────\n");
    std::fprintf(stderr, "Ticks evaluated   : %zu\n", N_TICKS);
    std::fprintf(stderr, "Signals emitted   : %zu  (%.1f%% pass rate)\n",
        signals.size(),
        100.0 * static_cast<double>(signals.size()) / static_cast<double>(N_TICKS));
    std::fprintf(stderr, "Total time        : %lld µs\n", (long long)elapsed_us);
    std::fprintf(stderr, "Per-tick latency  : %.1f ns\n", per_tick_ns);
    std::fprintf(stderr, "Throughput        : %.2f million ticks/s\n", throughput_m);
    std::fprintf(stderr, "─────────────────────────────────────────\n\n");

    // Use the signals vector to prevent dead-code elimination.
    // Print the first signal's timestamp as a spot-check.
    if (!signals.empty()) {
        const auto& s = signals.front();
        std::fprintf(stderr, "First signal: ts=%llu  delta=%+.4f  p=%.4f  action=%s\n\n",
            (unsigned long long)s.timestamp_ms,
            s.obi_delta,
            s.p_execute,
            s.action == TradeAction::BUY_B_SELL_A ? "BUY_B_SELL_A" : "BUY_A_SELL_B");
    }

    return EXIT_SUCCESS;
}
