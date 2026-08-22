#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>        // std::shared_ptr<CircuitBreaker> below
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "fee_config.hpp"
#include "friction.hpp"
#include "models.hpp"
#include "risk.hpp"

namespace crossflux {

struct SimOrder {
    uint64_t    timestamp_ms;
    uint64_t    signal_timestamp_ms;
    double      obi_delta;
    double      fill_price_buy;
    double      fill_price_sell;
    double      fill_qty;
    double      spread_pct;
    double      gross_pnl;
    double      net_pnl;
    double      fees_paid;
    double      slippage_bps;
    double      latency_ms;
    bool        filled;
    std::string buy_exchange;
    std::string sell_exchange;
    std::string reject_reason;

    // ── added when execution was decoupled from signalling ──────────────────
    // The fields above cannot express a partly-hedged trade: one `fill_qty` and
    // one `slippage_bps` assume both legs got the same size at symmetric cost.
    // Once each leg fills against its own book at its own time, they don't.
    double filled_qty_buy{0.0};    ///< What the buy venue actually supplied.
    double filled_qty_sell{0.0};   ///< What the sell venue actually supplied.
    double residual_qty{0.0};      ///< |buy - sell|: the naked leftover, not an arbitrage.
    double legging_cost{0.0};      ///< Cost of flattening that leftover.
    double signal_edge_bps{0.0};   ///< Edge at T, which is what justified the trade.
    double realized_edge_bps{0.0}; ///< Edge at the fill. The gap between these two is the finding.
    double leg_gap_ms{0.0};        ///< How long the position sat half-on.
    bool   adverse{false};         ///< Booked at a loss. See adverse_selection_fills().
};

/// Simulated taker execution.
///
/// Two ways in, and they differ in the thing that matters:
///
///   submit() + resolve_due()   Asynchronous. A signal at T buys nothing at T;
///                              it is buffered, and each leg is priced later
///                              against whatever book has printed by then, by
///                              walking it for the full size. This is the path
///                              that can lose money.
///
///   evaluate_and_execute()     Synchronous, and kept only for the existing
///                              scalar-price callers. It has no book depth and
///                              no future book, so it cannot represent either
///                              mechanism; see the note on its declaration.
///
/// Neither path may decline a trade because it turned out unprofitable. A gate
/// that re-reads the edge after applying friction and rejects when the edge is
/// gone keeps exactly the winners, which is how the backtest came to report a
/// 100% win rate on 307,881 trades.
class SimulatedExecutor {
public:
    SimulatedExecutor(
        std::shared_ptr<CircuitBreaker> risk_mgr,
        std::string_view exchange_a,
        std::string_view exchange_b,
        double max_position = 5.0,
        double cooldown_ms = 10000.0,
        double min_profit_bps = 2.0,
        double slippage_model_bps = 0.5,
        friction::FrictionModel friction_model = friction::preset_stress(),
        uint64_t latency_seed = 0x5CF1'0000'C0FFEEull,
        std::size_t pending_capacity = 4096
    );

    /// Buffer an order for later fill. Books nothing, returns true if buffered.
    ///
    /// `qty` is per leg. The signal's own books are read here for two things
    /// only: the touch prices, which are recorded so slippage can be attributed
    /// afterwards, and the pre-trade edge. No fill price is taken from them —
    /// that is the whole point.
    ///
    /// Latency is drawn per leg, so the two halves resolve at different times
    /// under a jittered preset and the quantity mismatch that leaves behind is
    /// charged at the model's legging cost.
    template <std::size_t N>
    bool submit(
        const ArbitrageSignal& signal,
        const OrderBookSnapshot<N>& book_a,
        const OrderBookSnapshot<N>& book_b,
        double qty
    );

    /// Fill and book every leg due at or before `now_ms`, against the books given.
    ///
    /// Returns the number of orders booked. Some of them will be losses; that is
    /// the intended behaviour and there is no code path here that can refuse one.
    ///
    /// The book passed in is the *most recent* one, so a leg due at t is priced
    /// against a book stamped at or after t — one tick of lag in a live engine,
    /// and none at all in a backtest that calls this on every row. It is never a
    /// look-ahead: a quote that had not printed by t cannot be reached from here,
    /// because the caller has not seen it either.
    template <std::size_t N>
    std::size_t resolve_due(
        double now_ms,
        const OrderBookSnapshot<N>& book_a,
        const OrderBookSnapshot<N>& book_b
    );

    /// Synchronous path for callers holding scalar prices rather than books.
    ///
    /// Retained because the Python bindings and the live dispatcher call it, and
    /// honestly labelled: with one price per side there is no depth to walk and
    /// no later book to fill against, so `latency_ms` can only act as a price
    /// haircut. That haircut is always adverse, which is the mirror image of the
    /// bug it replaced — it cannot produce a favourable surprise either. Treat
    /// its win rate as uninformative and use submit()/resolve_due() for anything
    /// being measured.
    bool evaluate_and_execute(
        const ArbitrageSignal& signal,
        double bid_price_a, double ask_price_a,
        double bid_price_b, double ask_price_b
    );

    double current_position() const noexcept { return position_.load(std::memory_order_acquire); }
    double realized_pnl() const noexcept { return pnl_.load(std::memory_order_acquire); }
    int total_orders() const noexcept { return total_orders_.load(std::memory_order_acquire); }
    int filled_orders() const noexcept { return filled_orders_.load(std::memory_order_acquire); }
    uint64_t last_trade_ms() const noexcept { return last_trade_ms_.load(std::memory_order_acquire); }
    double max_position_limit() const noexcept { return max_position_; }
    double cooldown_ms() const noexcept { return cooldown_ms_; }
    double min_profit_bps() const noexcept { return min_profit_bps_; }
    bool is_cooldown_active(uint64_t now_ms) const noexcept;
    std::shared_ptr<CircuitBreaker> risk_mgr() const noexcept { return risk_mgr_; }

    /// Trades booked at a negative net PnL: the spread collapsed inside the
    /// latency window, or walking the book for size cost more than the edge.
    /// Reported rather than rejected, because a strategy's adverse-selection rate
    /// is the number worth knowing and a rejection would hide it.
    int adverse_selection_fills() const noexcept {
        return adverse_fills_.load(std::memory_order_acquire);
    }

    /// Orders where the two legs filled different quantities, leaving a residual.
    int legged_fills() const noexcept {
        return legged_fills_.load(std::memory_order_acquire);
    }

    /// Orders that reached a book too thin to supply the requested size at all.
    int empty_fills() const noexcept {
        return empty_fills_.load(std::memory_order_acquire);
    }

    const friction::FrictionModel& friction_model() const noexcept { return friction_; }
    double latency_ms() const noexcept { return friction_.latency_ms; }

    /// Orders buffered and not yet filled.
    std::size_t pending_orders() const noexcept { return pending_.in_flight(); }

    /// Orders dropped because the buffer was full. Non-zero means the reported
    /// PnL is missing trades and the capacity needs raising.
    uint64_t dropped_orders() const noexcept { return pending_.overflowed(); }

    /// True if both the async and the sync path have been used on this executor.
    ///
    /// They keep time differently — the async path stamps fills in market time
    /// from `signal.timestamp_ms`, the sync path in wall-clock from
    /// system_clock::now() — so `last_trade_ms_` and therefore the cooldown
    /// becomes meaningless once both have run. Surfaced rather than prevented,
    /// because the sync path exists only for callers that cannot supply books and
    /// forbidding the combination outright would break them.
    bool mixed_clocks() const noexcept {
        return paths_used_.load(std::memory_order_acquire) == 0b11;
    }

private:
    bool check_risk_guards(const ArbitrageSignal& signal, uint64_t now_ms) noexcept;
    void log_order(const SimOrder& order) noexcept;

    /// Shared tail of both paths: account for a fill and write it down. Called
    /// with the trade already priced, so it has no opportunity to reject one.
    void book_fill(SimOrder& order) noexcept;

    /// Turn a resolved order into a priced SimOrder.
    ///
    /// Arithmetic only, no decision: fees are charged on what each leg actually
    /// filled, the residual is charged the legging cost, and whether net PnL comes
    /// out negative is simply read off the result. Independent of the book depth,
    /// so it lives in the .cpp rather than being instantiated per N.
    SimOrder price_resolved(const friction::ResolvedOrder& r) const;

    std::shared_ptr<CircuitBreaker> risk_mgr_;
    std::string exchange_a_;
    std::string exchange_b_;

    const double max_position_;
    const double cooldown_ms_;
    const double min_profit_bps_;
    const double slippage_model_bps_;

    friction::FrictionModel friction_;
    friction::PendingOrderQueue pending_;
    /// Seeded, not std::random_device: a backtest that cannot be replayed cannot
    /// be debugged, and three Python tests were already flaky against an unseeded
    /// draw. Only touched when the preset has jitter.
    std::mt19937_64 latency_rng_;
    std::mutex pending_mutex_;

    std::atomic<double> position_{0.0};
    std::atomic<double> pnl_{0.0};
    std::atomic<int> total_orders_{0};
    std::atomic<int> filled_orders_{0};
    std::atomic<int> adverse_fills_{0};
    std::atomic<int> legged_fills_{0};
    std::atomic<int> empty_fills_{0};
    std::atomic<uint64_t> last_trade_ms_{0};
    /// Bit 0: the sync path has run. Bit 1: the async path has run. See
    /// mixed_clocks().
    std::atomic<int> paths_used_{0};

    std::mutex csv_mutex_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Template definitions
//
// In the header because the book depth N is a template parameter and the engine
// instantiates it at more than one width (5 for the snapshot feeds, 10 for the
// incremental one).
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N>
bool SimulatedExecutor::submit(
    const ArbitrageSignal& signal,
    const OrderBookSnapshot<N>& book_a,
    const OrderBookSnapshot<N>& book_b,
    double qty
) {
    const uint64_t now = signal.timestamp_ms;
    total_orders_.fetch_add(1, std::memory_order_acq_rel);
    paths_used_.fetch_or(0b10, std::memory_order_acq_rel);

    if (!check_risk_guards(signal, now)) {
        SimOrder rejected{};
        rejected.timestamp_ms = now;
        rejected.signal_timestamp_ms = signal.timestamp_ms;
        rejected.obi_delta = signal.obi_delta;
        rejected.filled = false;
        rejected.reject_reason = "risk_guard";
        log_order(rejected);
        return false;
    }

    // Which venue we buy from, and the touch price on each side at T.
    const bool buy_a = (signal.action == TradeAction::BUY_A_SELL_B);
    const OrderBookSnapshot<N>& buy_book  = buy_a ? book_a : book_b;
    const OrderBookSnapshot<N>& sell_book = buy_a ? book_b : book_a;
    const double touch_buy  = buy_book.ask_depth  > 0 ? buy_book.asks[0].price  : 0.0;
    const double touch_sell = sell_book.bid_depth > 0 ? sell_book.bids[0].price : 0.0;

    if (touch_buy <= 0.0 || touch_sell <= 0.0) {
        SimOrder rejected{};
        rejected.timestamp_ms = now;
        rejected.signal_timestamp_ms = signal.timestamp_ms;
        rejected.obi_delta = signal.obi_delta;
        rejected.filled = false;
        rejected.reject_reason = "invalid_price";
        log_order(rejected);
        return false;
    }

    const double edge_bps = (touch_sell - touch_buy) / touch_buy * 10000.0;
    // The one gate that stays a gate. It reads the book at T only, and it is a
    // decision about whether to *try* — not a preview of the outcome, which is
    // still unknown here. Nothing downstream re-checks it.
    if (edge_bps < min_profit_bps_) {
        SimOrder rejected{};
        rejected.timestamp_ms = now;
        rejected.signal_timestamp_ms = signal.timestamp_ms;
        rejected.obi_delta = signal.obi_delta;
        rejected.signal_edge_bps = edge_bps;
        rejected.filled = false;
        rejected.reject_reason = "spread_below_min";
        log_order(rejected);
        return false;
    }

    friction::PendingOrder order{};
    order.signal_ts_ms    = static_cast<double>(signal.timestamp_ms);
    order.qty             = qty;
    order.signal_edge_bps = edge_bps;
    order.obi_delta       = signal.obi_delta;

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        const double lat_buy  = friction_.sample_latency(latency_rng_);
        const double lat_sell = friction_.sample_latency(latency_rng_);
        order.buy  = friction::PendingLeg{order.signal_ts_ms + lat_buy,  touch_buy,
                                         buy_a ? friction::Venue::kA : friction::Venue::kB};
        order.sell = friction::PendingLeg{order.signal_ts_ms + lat_sell, touch_sell,
                                         buy_a ? friction::Venue::kB : friction::Venue::kA};
        if (pending_.submit(order)) {
            return true;
        }
    }

    SimOrder dropped{};
    dropped.timestamp_ms = now;
    dropped.signal_timestamp_ms = signal.timestamp_ms;
    dropped.obi_delta = signal.obi_delta;
    dropped.signal_edge_bps = edge_bps;
    dropped.filled = false;
    dropped.reject_reason = "pending_queue_full";
    log_order(dropped);
    return false;
}

template <std::size_t N>
std::size_t SimulatedExecutor::resolve_due(
    double now_ms,
    const OrderBookSnapshot<N>& book_a,
    const OrderBookSnapshot<N>& book_b
) {
    std::vector<SimOrder> booked;

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_.drain(
            now_ms,
            // Price one leg. A buy takes from the asks, a sell hits the bids;
            // each walks its own venue's book for the full size, so the price
            // depends on how much is being asked for.
            [&](const friction::PendingOrder& p, friction::Leg leg, double) {
                const friction::Venue v = (leg == friction::Leg::kBuy) ? p.buy.venue
                                                                       : p.sell.venue;
                const OrderBookSnapshot<N>& bk =
                    (v == friction::Venue::kA) ? book_a : book_b;
                return (leg == friction::Leg::kBuy) ? friction::walk_asks(bk, p.qty)
                                                    : friction::walk_bids(bk, p.qty);
            },
            [&](const friction::ResolvedOrder& r) {
                booked.push_back(price_resolved(r));
            });
    }

    // Booked outside the queue lock: log_order takes the CSV mutex and
    // record_trade_pnl reaches into the circuit breaker, and holding the pending
    // lock across either would put the submit path behind file I/O.
    for (SimOrder& o : booked) {
        book_fill(o);
    }
    return booked.size();
}

} // namespace crossflux
