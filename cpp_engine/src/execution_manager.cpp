#include "execution_manager.hpp"
#include "execution.hpp"
#include "fee_config.hpp"
#include <cmath>
#include <iostream>
#include <random>

namespace crossflux {

// Taker fees now come from fee_config.hpp, which is generated from src/fees.py
// so the C++ engine, backtest/engine.py and src/execution_simulator.py cannot
// drift apart. The rates formerly hardcoded here (binance 0.0004 /
// kraken 0.0010) are the "retail" preset; select a schedule with
// $CROSSFLUX_FEE_PRESET. Regenerate: python -m src.fees --emit-cpp-header
//
// These two constants and the generator below belong to the *synchronous* path
// only. They are not the friction model: an additive normal is symmetric, can go
// negative before the clamp, and has no relationship to the per-leg latency the
// async path draws from FrictionModel. Kept unchanged so the old scalar callers
// behave exactly as before rather than silently changing numbers underneath them.
static constexpr double LATENCY_MEAN_MS = 40.0;
static constexpr double LATENCY_STD_MS = 10.0;

static thread_local std::mt19937_64 rng_{std::random_device{}()};

static double sample_latency_ms() {
    std::normal_distribution<double> dist(LATENCY_MEAN_MS, LATENCY_STD_MS);
    return std::max(0.0, dist(rng_));
}

static double taker_fee_for(std::string_view exchange) {
    return fees::taker_fee_for(exchange);
}

static void atomic_add(std::atomic<double>& atom, double val) noexcept {
    double expected = atom.load(std::memory_order_relaxed);
    while (!atom.compare_exchange_weak(expected, expected + val,
           std::memory_order_release, std::memory_order_relaxed)) {}
}

// ─── SimulatedExecutor ──────────────────────────────────────────────────────

SimulatedExecutor::SimulatedExecutor(
    std::shared_ptr<CircuitBreaker> risk_mgr,
    std::string_view exchange_a,
    std::string_view exchange_b,
    double max_position,
    double cooldown_ms,
    double min_profit_bps,
    double slippage_model_bps,
    friction::FrictionModel friction_model,
    uint64_t latency_seed,
    std::size_t pending_capacity
)
    : risk_mgr_(std::move(risk_mgr))
    , exchange_a_(exchange_a)
    , exchange_b_(exchange_b)
    , max_position_(max_position)
    , cooldown_ms_(cooldown_ms)
    , min_profit_bps_(min_profit_bps)
    , slippage_model_bps_(slippage_model_bps)
    , friction_(friction_model.valid() ? friction_model : friction::preset_stress())
    , pending_(pending_capacity)
    , latency_rng_(latency_seed)
{
    if (!friction_model.valid()) {
        // Falling back to `zero` on a bad model would hand the caller the
        // frictionless configuration — the one that produces the tautological
        // win rate — in response to a typo. Fall back to the harsh preset and
        // say so instead.
        std::cerr << "[crossflux] friction model '" << friction_model.name
                  << "' is invalid (latency_ms=" << friction_model.latency_ms
                  << ", jitter=" << friction_model.jitter_log_sigma
                  << ", legging_bps=" << friction_model.legging_cost_bps
                  << "); using 'stress' instead\n";
    }
    std::ofstream csv("/tmp/live_orders.csv");
    csv << "timestamp_ms,signal_timestamp_ms,obi_delta,"
        << "buy_exchange,sell_exchange,fill_price_buy,fill_price_sell,"
        << "fill_qty,spread_pct,gross_pnl,net_pnl,fees_paid,slippage_bps,latency_ms,filled,reject_reason,"
        // Appended, never inserted: dashboard/app.py reads this file by column
        // name, and dashboard/seed_demo_feed.py writes it positionally.
        << "filled_qty_buy,filled_qty_sell,residual_qty,legging_cost,"
        << "signal_edge_bps,realized_edge_bps,leg_gap_ms,adverse\n";
}

SimOrder SimulatedExecutor::price_resolved(const friction::ResolvedOrder& r) const {
    const std::string& buy_ex =
        (r.order.buy.venue == friction::Venue::kA) ? exchange_a_ : exchange_b_;
    const std::string& sell_ex =
        (r.order.sell.venue == friction::Venue::kA) ? exchange_a_ : exchange_b_;

    const double hedged   = r.hedged_qty();
    const double residual = r.residual_qty();

    SimOrder o{};
    o.timestamp_ms        = static_cast<uint64_t>(r.order.ready_at_ms());
    o.signal_timestamp_ms = static_cast<uint64_t>(r.order.signal_ts_ms);
    o.obi_delta           = r.order.obi_delta;
    o.buy_exchange        = buy_ex;
    o.sell_exchange       = sell_ex;
    o.fill_price_buy      = r.buy.vwap;
    o.fill_price_sell     = r.sell.vwap;
    o.fill_qty            = hedged;
    o.filled_qty_buy      = r.buy.filled_qty;
    o.filled_qty_sell     = r.sell.filled_qty;
    o.residual_qty        = residual;
    o.signal_edge_bps     = r.order.signal_edge_bps;
    o.leg_gap_ms          = r.order.leg_gap_ms();
    o.latency_ms          = r.order.ready_at_ms() - r.order.signal_ts_ms;

    // Slippage as the sum of two unsigned costs — buying above the touch and
    // selling below it are both losses and must not be allowed to net out.
    o.slippage_bps = r.buy.slippage_bps(r.order.buy.signal_price)
                   + r.sell.slippage_bps(r.order.sell.signal_price);

    // Nothing filled on either side. Not a rejection: the order was live and
    // reached a book that could not supply it at any price in the five levels
    // this feed carries.
    if (hedged <= friction::kEpsQty && residual <= friction::kEpsQty) {
        o.filled = false;
        o.reject_reason = "no_liquidity_at_fill";
        return o;
    }

    // Gross is earned on the hedged quantity only; the residual is not an
    // arbitrage, it is a naked position, and it is charged rather than credited.
    // Guarded on hedged > 0 because a vwap of 0.0 is the "nothing filled"
    // sentinel, and subtracting it as a price would invent a spread of 100%.
    o.gross_pnl = (hedged > friction::kEpsQty)
                ? (r.sell.vwap - r.buy.vwap) * hedged
                : 0.0;

    // Fees on what each leg actually filled, not on the matched quantity: a venue
    // charges for the size it executed regardless of whether the other side got
    // there. This is where an over-fill on one leg becomes expensive twice.
    o.fees_paid = r.buy.notional  * taker_fee_for(buy_ex)
                + r.sell.notional * taker_fee_for(sell_ex);

    // Flattening the residual, at the price of whichever venue over-filled.
    const double unwind_px = (r.buy.filled_qty > r.sell.filled_qty) ? r.buy.vwap
                                                                    : r.sell.vwap;
    o.legging_cost = friction_.legging_cost(residual, unwind_px);

    o.realized_edge_bps = (r.buy.vwap > 0.0 && hedged > friction::kEpsQty)
                        ? (r.sell.vwap - r.buy.vwap) / r.buy.vwap * 10000.0
                        : 0.0;
    o.spread_pct = o.realized_edge_bps / 10000.0;
    o.net_pnl    = o.gross_pnl - o.fees_paid - o.legging_cost;
    o.filled     = true;

    // Adverse selection, and the whole reason this is recorded rather than
    // rejected: the trade was entered on a positive edge and came back negative,
    // because the spread moved inside the latency window or because walking the
    // book for this size cost more than the edge was worth. There is no branch
    // above that could have declined it.
    o.adverse = o.net_pnl < 0.0;
    if (o.adverse) {
        o.reject_reason = r.legged() ? "adverse_fill_legged" : "adverse_fill";
    } else if (r.legged()) {
        o.reject_reason = "legged_fill";
    }
    return o;
}

void SimulatedExecutor::book_fill(SimOrder& order) noexcept {
    if (!order.filled) {
        if (order.reject_reason == "no_liquidity_at_fill") {
            empty_fills_.fetch_add(1, std::memory_order_acq_rel);
        }
        log_order(order);
        return;
    }

    risk_mgr_->record_trade_pnl(order.net_pnl);
    atomic_add(pnl_, order.net_pnl);
    filled_orders_.fetch_add(1, std::memory_order_acq_rel);
    if (order.adverse) {
        adverse_fills_.fetch_add(1, std::memory_order_acq_rel);
    }
    if (order.residual_qty > friction::kEpsQty) {
        legged_fills_.fetch_add(1, std::memory_order_acq_rel);
    }
    // `position_` deliberately stays at zero. Charging legging_cost models
    // flattening the residual immediately, so no exposure carries forward — which
    // means the max_position guard in check_risk_guards is inert on this path.
    // Recorded in TEST_REPORT.md §5 rather than papered over with a number that
    // would only look like risk tracking.
    last_trade_ms_.store(order.timestamp_ms, std::memory_order_release);
    log_order(order);
}

bool SimulatedExecutor::is_cooldown_active(uint64_t now_ms) const noexcept {
    uint64_t last = last_trade_ms_.load(std::memory_order_acquire);
    if (last == 0) return false;
    return (now_ms - last) < static_cast<uint64_t>(cooldown_ms_);
}

bool SimulatedExecutor::check_risk_guards(
    const ArbitrageSignal& signal, uint64_t now_ms
) noexcept {
    if (risk_mgr_->is_tripped()) return false;
    if (is_cooldown_active(now_ms)) return false;
    double pos = position_.load(std::memory_order_acquire);
    if (std::abs(pos) >= max_position_) return false;
    if (signal.p_execute < 0.80) return false;
    return true;
}

bool SimulatedExecutor::evaluate_and_execute(
    const ArbitrageSignal& signal,
    double bid_price_a, double ask_price_a,
    double bid_price_b, double ask_price_b
) {
    uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    total_orders_.fetch_add(1, std::memory_order_acq_rel);
    paths_used_.fetch_or(0b01, std::memory_order_acq_rel);

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

    double buy_price, sell_price;
    std::string buy_ex, sell_ex;
    const double qty = 0.01;

    if (signal.action == TradeAction::BUY_A_SELL_B) {
        buy_price = ask_price_a;
        sell_price = bid_price_b;
        buy_ex = exchange_a_;
        sell_ex = exchange_b_;
    } else {
        buy_price = ask_price_b;
        sell_price = bid_price_a;
        buy_ex = exchange_b_;
        sell_ex = exchange_a_;
    }

    if (buy_price <= 0 || sell_price <= 0) {
        SimOrder rejected{};
        rejected.timestamp_ms = now;
        rejected.signal_timestamp_ms = signal.timestamp_ms;
        rejected.obi_delta = signal.obi_delta;
        rejected.filled = false;
        rejected.reject_reason = "invalid_price";
        log_order(rejected);
        return false;
    }

    double raw_spread_bps = (sell_price - buy_price) / buy_price * 10000.0;

    if (raw_spread_bps < min_profit_bps_) {
        SimOrder rejected{};
        rejected.timestamp_ms = now;
        rejected.signal_timestamp_ms = signal.timestamp_ms;
        rejected.obi_delta = signal.obi_delta;
        rejected.filled = false;
        rejected.reject_reason = "spread_below_min";
        log_order(rejected);
        return false;
    }

    // Latency, such as it can be represented from scalar prices: a haircut
    // proportional to the delay rather than a re-read of a later book. Always
    // adverse and never favourable, which is the opposite bias to the one this
    // replaced but a bias all the same. Use submit()/resolve_due() to measure
    // anything.
    double latency_ms = sample_latency_ms();
    double latency_jitter = (latency_ms / 1000.0) * 0.01;
    buy_price *= (1.0 + latency_jitter);
    sell_price *= (1.0 - latency_jitter);

    // No gate here. The edge was checked at signal time above and the order went
    // out on the strength of it; re-checking after friction and rejecting would
    // keep exactly the trades that happened to stay profitable, which is the
    // tautology TEST_REPORT.md §2.1 documents. Whatever the haircut leaves is
    // what gets booked, negative included.
    const double slippage_bps = slippage_model_bps_;
    const double fill_buy  = buy_price  * (1.0 + slippage_bps / 10000.0);
    const double fill_sell = sell_price * (1.0 - slippage_bps / 10000.0);
    const double net_spread_bps = ((fill_sell - fill_buy) / fill_buy) * 10000.0;

    SimOrder order{};
    order.timestamp_ms = now;
    order.signal_timestamp_ms = signal.timestamp_ms;
    order.obi_delta = signal.obi_delta;
    order.fill_price_buy = fill_buy;
    order.fill_price_sell = fill_sell;
    order.fill_qty = qty;
    order.filled_qty_buy = qty;
    order.filled_qty_sell = qty;
    order.spread_pct = net_spread_bps / 10000.0;
    order.buy_exchange = buy_ex;
    order.sell_exchange = sell_ex;
    order.gross_pnl = (fill_sell - fill_buy) * qty;
    order.fees_paid = fill_buy * qty * taker_fee_for(buy_ex)
                    + fill_sell * qty * taker_fee_for(sell_ex);
    order.net_pnl = order.gross_pnl - order.fees_paid;
    order.slippage_bps = slippage_bps;
    order.latency_ms = latency_ms;
    order.signal_edge_bps = raw_spread_bps;
    order.realized_edge_bps = net_spread_bps;
    order.filled = true;
    order.adverse = order.net_pnl < 0.0;
    if (order.adverse) {
        // The case the requirement names: positive edge at signal, negative by
        // fill. Booked, counted, and visible in adverse_selection_fills().
        order.reject_reason = (raw_spread_bps > 0.0 && net_spread_bps <= 0.0)
                            ? "adverse_fill_spread_collapsed"
                            : "adverse_fill";
    }

    book_fill(order);
    return true;
}

void SimulatedExecutor::log_order(const SimOrder& order) noexcept {
    try {
        std::lock_guard<std::mutex> lock(csv_mutex_);
        std::ofstream csv("/tmp/live_orders.csv", std::ios::app);
        csv << order.timestamp_ms << ","
            << order.signal_timestamp_ms << ","
            << order.obi_delta << ","
            << order.buy_exchange << ","
            << order.sell_exchange << ","
            << order.fill_price_buy << ","
            << order.fill_price_sell << ","
            << order.fill_qty << ","
            << order.spread_pct << ","
            << order.gross_pnl << ","
            << order.net_pnl << ","
            << order.fees_paid << ","
            << order.slippage_bps << ","
            << order.latency_ms << ","
            << (order.filled ? "1" : "0") << ","
            << order.reject_reason << ","
            << order.filled_qty_buy << ","
            << order.filled_qty_sell << ","
            << order.residual_qty << ","
            << order.legging_cost << ","
            << order.signal_edge_bps << ","
            << order.realized_edge_bps << ","
            << order.leg_gap_ms << ","
            << (order.adverse ? "1" : "0") << "\n";
    } catch (...) {
    }
}

// ─── SimulatedOrderDispatcher ───────────────────────────────────────────────

SimulatedOrderDispatcher::SimulatedOrderDispatcher(
    std::shared_ptr<CircuitBreaker> risk_mgr,
    std::string_view exchange_a,
    std::string_view exchange_b,
    double max_position,
    double cooldown_ms,
    double min_profit_bps,
    double slippage_model_bps
)
    : OrderDispatcher(risk_mgr)
    , executor_(risk_mgr, exchange_a, exchange_b,
                max_position, cooldown_ms,
                min_profit_bps, slippage_model_bps)
{
}

bool SimulatedOrderDispatcher::execute(const DispatchSignal& ds) noexcept {
    return executor_.evaluate_and_execute(
        ds.signal,
        ds.bid_price_a, ds.ask_price_a,
        ds.bid_price_b, ds.ask_price_b
    );
}

ExecutionResult SimulatedOrderDispatcher::execute_buy(
    const std::string& exchange,
    double price,
    double qty,
    OrderType type
) noexcept {
    if (risk_mgr_->is_tripped()) {
        return {false, 0.0, 0.0, "Circuit breaker tripped."};
    }
    std::string t_str = (type == OrderType::FOK) ? "FOK" : "IOC";
    std::cout << "[SimExec] -> BUY " << qty << " @ " << price
              << " " << t_str << " on " << exchange << std::endl;
    return {true, price, qty, ""};
}

ExecutionResult SimulatedOrderDispatcher::execute_sell(
    const std::string& exchange,
    double price,
    double qty,
    OrderType type
) noexcept {
    if (risk_mgr_->is_tripped()) {
        return {false, 0.0, 0.0, "Circuit breaker tripped."};
    }
    std::string t_str = (type == OrderType::FOK) ? "FOK" : "IOC";
    std::cout << "[SimExec] -> SELL " << qty << " @ " << price
              << " " << t_str << " on " << exchange << std::endl;
    return {true, price, qty, ""};
}

double SimulatedOrderDispatcher::current_position() const noexcept {
    return executor_.current_position();
}

double SimulatedOrderDispatcher::realized_pnl() const noexcept {
    return executor_.realized_pnl();
}

int SimulatedOrderDispatcher::total_orders() const noexcept {
    return executor_.total_orders();
}

int SimulatedOrderDispatcher::filled_orders() const noexcept {
    return executor_.filled_orders();
}

uint64_t SimulatedOrderDispatcher::last_trade_ms() const noexcept {
    return executor_.last_trade_ms();
}

double SimulatedOrderDispatcher::max_position_limit() const noexcept {
    return executor_.max_position_limit();
}

double SimulatedOrderDispatcher::cooldown_ms() const noexcept {
    return executor_.cooldown_ms();
}

double SimulatedOrderDispatcher::min_profit_bps() const noexcept {
    return executor_.min_profit_bps();
}

bool SimulatedOrderDispatcher::is_cooldown_active(uint64_t now_ms) const noexcept {
    return executor_.is_cooldown_active(now_ms);
}

} // namespace crossflux
