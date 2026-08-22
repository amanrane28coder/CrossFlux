/**
 * @file    predictor.hpp
 * @brief   Decision core: MarketTick aggregation struct and SignalAggregator.
 *
 * Phase 7: Low-latency C++20 translation of src/predictor.py.
 *
 * Architecture
 * ------------
 *  MarketTick<N>
 *      Stack-allocated replacement for Python's MarketState (which used a
 *      Dict[str, OrderBookSnapshot] heap map).  Holds two OrderBookSnapshot<N>
 *      objects directly by value — adjacent in memory, loaded together into
 *      cache, zero hash-lookup latency.
 *
 *  SignalAggregator
 *      Mirrors Python's SignalAggregator class with one critical optimisation:
 *      p_execute (the log-normal CDF result) is pre-computed ONCE in the
 *      constructor and stored as a plain double.  The hot evaluate() loop
 *      never calls erf, log, or any transcendental function — it is pure
 *      integer comparisons and FMA arithmetic.
 *
 * Hot-path cost per MarketTick<10> (Release build, arm64):
 *   calculate_obi × 2    →  ~20 FMA + 2 divs
 *   calculate_obi_delta  →  1 sub
 *   Gate 1 compare       →  1 FABS + 1 FCMP + branch
 *   Gate 2 compare       →  1 FCMP + branch  (p_execute_ is a register load)
 *   ArbitrageSignal emit →  32-byte struct write (if fired)
 *   ─────────────────────────────────────────────────────
 *   Total transcendental calls in hot loop: 0
 *
 * C++20 required.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>      // std::memcpy, std::memset
#include <limits>       // std::numeric_limits — NaN when weighted delta is unused
#include <span>         // std::span — view over the active weight profile
#include <stdexcept>    // std::invalid_argument
#include <string_view>
#include <vector>

#include "models.hpp"
#include "signals.hpp"
#include "obi_config.hpp"   // generated from src/obi_weights.py

namespace crossflux {

// ─────────────────────────────────────────────────────────────────────────────
// SignalAggregator
//
// Iterates over a batch of MarketTick<N> objects and emits ArbitrageSignal
// events for every tick that clears both threshold gates:
//
//   Gate 1 (volume pressure) : |obi_delta| > delta_threshold
//   Gate 2 (latency risk)    : p_execute_  > min_p_execute
//
// p_execute_ is pre-computed ONCE in the constructor via the log-normal CDF:
//
//   P(L < alpha) = 0.5 * (1 + erf((ln(alpha) - mu) / (sigma * sqrt(2))))
//
// The hot evaluate() loop never calls erf, log, or sqrt — it is a pure
// floating-point comparison against the stored p_execute_ scalar.
// ─────────────────────────────────────────────────────────────────────────────

class SignalAggregator {
public:
    /// Default thresholds — match the Python predictor.py constants.
    static constexpr double kDefaultDeltaThreshold = 0.3;
    static constexpr double kDefaultMinPExecute    = 0.80;

    /**
     * Construct a fully configured SignalAggregator.
     *
     * Pre-computes p_execute via the log-normal CDF (std::erf called once here).
     *
     * @param exchange_a        Canonical ID of venue A (e.g. "binance"). Max 15 chars.
     * @param exchange_b        Canonical ID of venue B (e.g. "kraken").  Max 15 chars.
     * @param latency_mu        Log-normal μ: mean of ln(latency). exp(mu) = median RTT (ms).
     * @param latency_sigma     Log-normal σ: shape parameter. Must be > 0.
     * @param alpha_lifetime_ms Expected discrepancy lifetime (ms). Must be > 0.
     * @param delta_threshold   Gate 1 minimum |Δ_OBI|.  Default: 0.3.
     * @param min_p_execute     Gate 2 minimum execution probability. Default: 0.80.
     *
     * @throws std::invalid_argument  for any invalid parameter.
     */
    SignalAggregator(
        std::string_view exchange_a,
        std::string_view exchange_b,
        double           latency_mu,
        double           latency_sigma,
        double           alpha_lifetime_ms,
        double           delta_threshold = kDefaultDeltaThreshold,
        double           min_p_execute   = kDefaultMinPExecute,
        double           min_spread_pct  = 0.0012
    );

    /**
     * Evaluate a batch of MarketTick<N> objects and emit ArbitrageSignals.
     *
     * The hot path per tick:
     *   1. calculate_obi(snap_a) and calculate_obi(snap_b)
     *   2. delta = calculate_obi_delta(obi_a, obi_b)
     *   3. Gate 1: if |delta| <= delta_threshold_ → skip
     *   4. Gate 2: if p_execute_ <= min_p_execute_ → skip
     *   5. Emit ArbitrageSignal with pre-resolved TradeAction
     *
     * @param ticks  Const reference to batch of MarketTick<N>. May be empty.
     * @return       Vector of emitted ArbitrageSignal objects (may be empty).
     */
    template <std::size_t N = 10>
    [[nodiscard]] std::vector<ArbitrageSignal>
    evaluate(const std::vector<MarketTick<N>>& ticks) const noexcept;

    /// Return the pre-computed execution probability (for diagnostics/tests).
    [[nodiscard]] double p_execute() const noexcept { return p_execute_; }

    /// Return the configured delta threshold.
    [[nodiscard]] double delta_threshold() const noexcept { return delta_threshold_; }

    /// Return the configured minimum p_execute.
    [[nodiscard]] double min_p_execute() const noexcept { return min_p_execute_; }

    /// Return the configured minimum spread percentage.
    [[nodiscard]] double min_spread_pct() const noexcept { return min_spread_pct_; }

    /// Name of the active OBI weight profile (from the generated obi_config.hpp).
    [[nodiscard]] std::string_view obi_profile() const noexcept { return profile_name_; }

    /// True if the weighted delta — not the unweighted one — drives Gate 1.
    [[nodiscard]] bool gate_on_weighted() const noexcept { return gate_on_weighted_; }

    /// True if weighted_obi_delta is populated on emitted signals.
    [[nodiscard]] bool report_weighted() const noexcept { return report_weighted_; }

    /// Active per-level weights (empty view if the profile is unused).
    [[nodiscard]] std::span<const double> weights() const noexcept { return weights_; }

    /**
     * Choose whether the weighted delta drives the entry gate.
     *
     * Defaults to true when the active profile is anything other than "flat",
     * and false for "flat". That default means selecting a profile actually
     * changes the strategy rather than silently doing nothing, while the
     * out-of-the-box configuration still reproduces published results exactly.
     *
     * Call with true and a flat profile to gate on the weighted path
     * deliberately — useful for confirming the two agree.
     *
     * @param on  true to gate on weighted_obi_delta, false for obi_delta.
     */
    void set_gate_on_weighted(bool on) noexcept {
        gate_on_weighted_ = on;
        // Gating on a value that is never computed would emit nothing at all,
        // since NaN fails Gate 1. Keep the two flags consistent.
        if (on) { report_weighted_ = true; }
    }

    /// Enable or disable computing weighted_obi_delta for reporting.
    /// Ignored (forced on) while gate_on_weighted() is true.
    void set_report_weighted(bool on) noexcept {
        report_weighted_ = on || gate_on_weighted_;
    }

    /// Return exchange A ID as a string_view (zero-copy).
    [[nodiscard]] std::string_view exchange_a() const noexcept {
        return std::string_view{exchange_a_};
    }

    /// Return exchange B ID as a string_view (zero-copy).
    [[nodiscard]] std::string_view exchange_b() const noexcept {
        return std::string_view{exchange_b_};
    }

private:
    // Fixed-width char arrays — no heap, no SSO complexity
    char   exchange_a_[16];
    char   exchange_b_[16];
    double delta_threshold_;
    double min_p_execute_;
    double min_spread_pct_;
    double p_execute_;          ///< Pre-computed lognorm CDF result. Used in every Gate 2 check.

    // ── Weight profile state ──────────────────────────────────────────────
    // weights_ is a non-owning view into the constexpr arrays in the generated
    // obi_config.hpp. Those have static storage duration, so the view stays
    // valid for the program's lifetime and copying a SignalAggregator is safe.
    std::span<const double> weights_;
    std::string_view        profile_name_;
    bool                    gate_on_weighted_;
    bool                    report_weighted_;
};


// ─────────────────────────────────────────────────────────────────────────────
// evaluate<N>() — template method defined in header (must be visible to callers)
//
// This is the hot path.  Every line is written for minimal instruction count.
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N>
[[nodiscard]] std::vector<ArbitrageSignal>
SignalAggregator::evaluate(const std::vector<MarketTick<N>>& ticks) const noexcept
{
    std::vector<ArbitrageSignal> signals;
    signals.reserve(ticks.size() / 4);   // Heuristic: ~25% signal rate avoids realloc

    for (const auto& tick : ticks) {
        // ── Step 1 & 2: OBI per venue ─────────────────────────────────────
        const double obi_a = calculate_obi(tick.snap_a);
        const double obi_b = calculate_obi(tick.snap_b);

        // ── Step 3: cross-venue delta ─────────────────────────────────────
        const double delta = calculate_obi_delta(obi_a, obi_b);

        // ── Step 3b: same delta under the active weight profile ───────────
        // Computed unconditionally, even when it does not drive the gate, so
        // every emitted signal carries both readings and the dashboard can show
        // them side by side. That comparison is the whole reason the profile
        // mechanism exists, and it costs ~20 flops on a path that already does
        // two full book reductions.
        //
        // Skipped entirely when the profile is flat AND flat does not gate:
        // the result would duplicate `delta` to within one ULP, so there is
        // nothing to compare. weighted_obi_delta then stays NaN, which the
        // dashboard reads as "not computed" rather than "balanced book".
        double wdelta = std::numeric_limits<double>::quiet_NaN();
        if (report_weighted_) {
            const double wobi_a = calculate_weighted_obi(tick.snap_a, weights_);
            const double wobi_b = calculate_weighted_obi(tick.snap_b, weights_);
            wdelta = calculate_obi_delta(wobi_a, wobi_b);
        }

        // Which reading drives the entry gate. Defaults to the unweighted
        // delta, so the default configuration reproduces every previously
        // published backtest number bit for bit — including on books deeper
        // than the profile, where flat-weighted and unweighted genuinely differ
        // (a 10-level book with deep bid size gives 0.667 unweighted versus
        // 0.000 under a 5-level flat profile).
        const double gate_delta = gate_on_weighted_ ? wdelta : delta;

        // ── Gate 1: volume-pressure filter ────────────────────────────────
        // Evaluated first: most ticks fail here, so we avoid the p_execute
        // comparison entirely for those ticks (branch predictor trained cold).
        // std::abs is constexpr on doubles in C++20.
        //
        // Written as a pass-through range test, so a NaN gate_delta falls
        // through to `continue` rather than emitting: NaN fails both
        // comparisons, making the condition false. That is the safe direction —
        // a misconfiguration that leaves gate_delta undefined produces no
        // trades instead of trading on garbage.
        if (!(gate_delta < -delta_threshold_ || gate_delta > delta_threshold_)) [[likely]] {
            continue;
        }

        // ── Gate 2: latency-risk filter ───────────────────────────────────
        // p_execute_ is a pre-computed scalar stored at construction.
        // This is a single FCMP instruction — zero transcendental calls.
        if (p_execute_ <= min_p_execute_) [[unlikely]] {
            continue;
        }

        // ── Emit signal ───────────────────────────────────────────────────
        // Direction follows whichever delta gated, so action and gate_delta
        // never disagree about which venue is bid-heavy.
        const TradeAction action = (gate_delta > 0.0)
            ? TradeAction::BUY_B_SELL_A
            : TradeAction::BUY_A_SELL_B;

        // ── Gate 3: Spread friction filter ────────────────────────────────
        const double buy_price = (action == TradeAction::BUY_B_SELL_A)
            ? tick.snap_b.asks[0].price
            : tick.snap_a.asks[0].price;
        const double sell_price = (action == TradeAction::BUY_B_SELL_A)
            ? tick.snap_a.bids[0].price
            : tick.snap_b.bids[0].price;

        const double spread_pct = (sell_price - buy_price) / buy_price;
        if (spread_pct < min_spread_pct_) [[unlikely]] {
            continue;
        }

        signals.emplace_back(
            tick.timestamp_ms,
            delta,          // always the unweighted baseline, whatever gated
            p_execute_,
            action,
            wdelta          // NaN when not computed; see report_weighted_
        );
    }

    return signals;
}

} // namespace crossflux