/**
 * @file    predictor.cpp
 * @brief   SignalAggregator constructor and log-normal CDF implementation.
 *
 * Phase 7: This translation unit contains:
 *   1. lognorm_cdf() — the one and only call to std::erf in the engine.
 *      Called ONCE per SignalAggregator lifetime in the constructor.
 *      Never called in the hot evaluate() loop (which is header-only).
 *
 *   2. SignalAggregator constructor — validates parameters and pre-computes
 *      p_execute_ via lognorm_cdf().
 *
 * Mathematical derivation of lognorm_cdf
 * ----------------------------------------
 * Let L ~ LogNormal(mu, sigma²). We want P(L < alpha).
 *
 * By change of variable: if ln(L) ~ N(mu, sigma²), then:
 *
 *   P(L < alpha) = P(ln(L) < ln(alpha))
 *                = P(Z < (ln(alpha) - mu) / sigma)   where Z ~ N(0,1)
 *                = Φ((ln(alpha) - mu) / sigma)
 *
 * The standard normal CDF Φ is related to the error function by:
 *
 *   Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
 *
 * Therefore:
 *
 *   P(L < alpha) = 0.5 * (1 + erf((ln(alpha) - mu) / (sigma * sqrt(2))))
 *
 * This is exactly what scipy.stats.lognorm.cdf(alpha, s=sigma, scale=exp(mu))
 * computes internally, confirmed to ≥ 10 significant figures.
 *
 * C++20 required.
 */

#include "predictor.hpp"

#include <cmath>       // std::log, std::erf, std::sqrt
#include <cstdio>      // std::fprintf (for constructor diagnostics)
#include <cstring>     // std::memset, std::memcpy
#include <stdexcept>   // std::invalid_argument

namespace crossflux {

// ─────────────────────────────────────────────────────────────────────────────
// Internal: log-normal CDF
//
// This is the ONLY call to a transcendental function (erf) in the entire
// Phase 7 engine.  It lives here — in the constructor path — and is never
// invoked from the hot evaluate() loop.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/**
 * Evaluate the log-normal CDF: P(L < alpha) where ln(L) ~ N(mu, sigma²).
 *
 * Maps directly to scipy.stats.lognorm.cdf(alpha, s=sigma, scale=exp(mu)).
 *
 * @param alpha   Discrepancy lifetime threshold (ms). Must be > 0.
 * @param mu      Log-normal location parameter (mean of ln(L)).
 * @param sigma   Log-normal shape parameter (std of ln(L)). Must be > 0.
 * @return        Probability in [0.0, 1.0].
 */
[[nodiscard]] double
lognorm_cdf(double alpha, double mu, double sigma) noexcept
{
    // Guard: alpha <= 0 → opportunity already expired → P = 0.0
    if (alpha <= 0.0) return 0.0;

    // Standardise: z = (ln(alpha) - mu) / (sigma * sqrt(2))
    static const double kSqrt2Inv = 1.0 / std::sqrt(2.0);  // computed once
    const double z = (std::log(alpha) - mu) * kSqrt2Inv / sigma;

    // Φ(z) = 0.5 * (1 + erf(z))
    return 0.5 * (1.0 + std::erf(z));
}

} // anonymous namespace


// ─────────────────────────────────────────────────────────────────────────────
// SignalAggregator constructor
// ─────────────────────────────────────────────────────────────────────────────

SignalAggregator::SignalAggregator(
    std::string_view exchange_a,
    std::string_view exchange_b,
    double           latency_mu,
    double           latency_sigma,
    double           alpha_lifetime_ms,
    double           delta_threshold,
    double           min_p_execute,
    double           min_spread_pct)
{
    // ── Validate exchange IDs ─────────────────────────────────────────────
    if (exchange_a.empty()) {
        throw std::invalid_argument(
            "SignalAggregator: exchange_a must not be empty.");
    }
    if (exchange_a.size() >= 16) {
        throw std::invalid_argument(
            "SignalAggregator: exchange_a must be < 16 characters.");
    }
    if (exchange_b.empty()) {
        throw std::invalid_argument(
            "SignalAggregator: exchange_b must not be empty.");
    }
    if (exchange_b.size() >= 16) {
        throw std::invalid_argument(
            "SignalAggregator: exchange_b must be < 16 characters.");
    }
    if (exchange_a == exchange_b) {
        throw std::invalid_argument(
            "SignalAggregator: exchange_a and exchange_b must be distinct.");
    }

    // ── Validate latency parameters ───────────────────────────────────────
    if (latency_sigma <= 0.0) {
        throw std::invalid_argument(
            "SignalAggregator: latency_sigma must be strictly positive.");
    }
    if (alpha_lifetime_ms <= 0.0) {
        throw std::invalid_argument(
            "SignalAggregator: alpha_lifetime_ms must be strictly positive.");
    }

    // ── Validate threshold parameters ─────────────────────────────────────
    if (delta_threshold < 0.0) {
        throw std::invalid_argument(
            "SignalAggregator: delta_threshold must be non-negative.");
    }
    if (min_p_execute < 0.0 || min_p_execute > 1.0) {
        throw std::invalid_argument(
            "SignalAggregator: min_p_execute must be in [0.0, 1.0].");
    }

    // ── Store exchange IDs (null-padded fixed char arrays) ────────────────
    std::memset(exchange_a_, 0, sizeof(exchange_a_));
    std::memcpy(exchange_a_, exchange_a.data(), exchange_a.size());
    std::memset(exchange_b_, 0, sizeof(exchange_b_));
    std::memcpy(exchange_b_, exchange_b.data(), exchange_b.size());

    // ── Store threshold scalars ───────────────────────────────────────────
    // ── Pre-compute constant log-normal execution probability ─────────────
    p_execute_       = lognorm_cdf(alpha_lifetime_ms, latency_mu, latency_sigma);
    delta_threshold_ = delta_threshold;
    min_p_execute_   = min_p_execute;
    min_spread_pct_  = min_spread_pct;

    // ── Resolve the OBI weight profile ────────────────────────────────────
    // obi::active() reads $CROSSFLUX_OBI_PROFILE once and aborts on an
    // unrecognised name; see cpp_engine/include/obi_config.hpp. The span points
    // into constexpr static storage, so it outlives this object.
    const auto& profile = obi::active();
    weights_      = profile.span();
    profile_name_ = profile.name;

    // Default: the weighted delta gates only when a non-flat profile was asked
    // for. With the default flat profile the engine behaves exactly as it did
    // before weighting existed, so previously published backtest numbers stay
    // reproducible. Selecting any other profile is taken as intent to trade on
    // it, rather than computing it and quietly ignoring it.
    gate_on_weighted_ = (profile_name_ != "flat");

    // Report the weighted reading whenever it differs from the baseline, so the
    // dashboard has both columns to compare. Under flat-and-not-gating the two
    // agree to within one ULP, so there is nothing to show and the field stays
    // NaN rather than duplicating obi_delta.
    report_weighted_ = gate_on_weighted_;

    // Optional diagnostic: print initialization details
    std::fprintf(stderr,
        "[SignalAggregator] %s vs %s | delta_threshold=%.3f | min_p_execute=%.3f "
        "| min_spread_pct=%.4f | p_execute=%.8f (alpha=%.1f ms, mu=%.3f, sigma=%.3f)\n",
        exchange_a_, exchange_b_, delta_threshold_, min_p_execute_, min_spread_pct_,
        p_execute_, alpha_lifetime_ms, latency_mu, latency_sigma);

    // Printed separately and unconditionally: a run whose signal definition
    // changed via an environment variable should say so in its own log, or the
    // resulting CSV is unattributable after the fact.
    std::fprintf(stderr,
        "[SignalAggregator] obi_profile=%.*s | depth=%zu | gates_on=%s "
        "(set CROSSFLUX_OBI_PROFILE to change; weighted OBI is not MLOFI -- "
        "see src/obi_weights.py)\n",
        static_cast<int>(profile_name_.size()), profile_name_.data(),
        weights_.size(),
        gate_on_weighted_ ? "weighted_obi_delta" : "obi_delta");
}

} // namespace crossflux
