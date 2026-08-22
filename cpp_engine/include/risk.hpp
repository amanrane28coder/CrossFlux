#pragma once

#include <atomic>
#include <cstdint>

namespace crossflux {

/**
 * @brief Circuit breaker for hard risk controls.
 *
 * Designed to be called from the hot execution path. Uses atomics for thread-safe
 * updates between the background execution thread and the main monitoring thread.
 */
class CircuitBreaker {
public:
    // Configurable thresholds with sensible defaults
    double max_daily_loss_;
    int max_consecutive_timeouts_;
    int max_consecutive_slippage_;

    /**
     * Construct a CircuitBreaker with configurable thresholds.
     *
     * @param max_daily_loss      Maximum daily loss before tripping (negative value, e.g. -1000.0)
     * @param max_consecutive_timeouts  Maximum consecutive timeouts before tripping
     * @param max_consecutive_slippage  Maximum consecutive slippage events before tripping
     */
    CircuitBreaker(
        double max_daily_loss = -5000.0,
        int max_consecutive_timeouts = 3,
        int max_consecutive_slippage = 5
    ) : max_daily_loss_(max_daily_loss),
        max_consecutive_timeouts_(max_consecutive_timeouts),
        max_consecutive_slippage_(max_consecutive_slippage) {}

    /** @return true if trading should be halted immediately. */
    bool is_tripped() const noexcept {
        return tripped_.load(std::memory_order_acquire);
    }

    /** Record the net PnL of a completed trade. Trips if cumulative loss exceeds threshold. */
    void record_trade_pnl(double pnl_net) noexcept {
        double current = cumulative_pnl_.load(std::memory_order_relaxed);
        while (!cumulative_pnl_.compare_exchange_weak(current, current + pnl_net,
               std::memory_order_release, std::memory_order_relaxed));

        if (cumulative_pnl_.load(std::memory_order_acquire) <= max_daily_loss_) {
            trip();
        }
    }

    /** Record an API timeout. Trips if consecutive timeouts hit threshold. */
    void record_timeout() noexcept {
        if (++consecutive_timeouts_ >= max_consecutive_timeouts_) {
            trip();
        }
    }

    /** Reset consecutive timeout counter upon a successful API response. */
    void reset_timeouts() noexcept {
        consecutive_timeouts_.store(0, std::memory_order_relaxed);
    }

    /** Record if a trade suffered excessive slippage (> 1.5x spread). Trips if consecutive. */
    void record_slippage(bool exceeded) noexcept {
        if (exceeded) {
            if (++consecutive_slippage_ >= max_consecutive_slippage_) {
                trip();
            }
        } else {
            consecutive_slippage_.store(0, std::memory_order_relaxed);
        }
    }

    /** Get cumulative PnL for monitoring. */
    double get_cumulative_pnl() const noexcept {
        return cumulative_pnl_.load(std::memory_order_acquire);
    }

private:
    void trip() noexcept {
        tripped_.store(true, std::memory_order_release);
    }

    std::atomic<bool> tripped_{false};
    std::atomic<double> cumulative_pnl_{0.0};
    std::atomic<int> consecutive_timeouts_{0};
    std::atomic<int> consecutive_slippage_{0};
};

} // namespace crossflux
