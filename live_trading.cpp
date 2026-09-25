#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "cpp_engine/include/fee_config.hpp"
#include "cpp_engine/include/friction.hpp"
#include "cpp_engine/include/models.hpp"
#include "cpp_engine/include/risk.hpp"
#include "cpp_engine/include/execution.hpp"
#include "cpp_engine/include/execution_manager.hpp"
#include "cpp_engine/include/ingestion_engine.hpp"
#include "cpp_engine/include/signals.hpp"
#include "cpp_engine/include/feed_handler.hpp"
#include "cpp_engine/include/websocket_client.hpp"

namespace crossflux {

// Helper function for atomic addition to double
static void atomic_add(std::atomic<double>& atom, double val) noexcept {
    double expected = atom.load(std::memory_order_relaxed);
    while (!atom.compare_exchange_weak(expected, expected + val,
           std::memory_order_release, std::memory_order_relaxed)) {}
}

// Helper function for clamping values
static double clamp_value(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// Global variables for basis calculation and adaptive sizing
std::mutex g_basis_mutex;
std::atomic<double> g_binance_mid{0.0};
std::atomic<double> g_kraken_mid{0.0};
std::atomic<double> g_current_basis{0.0};
std::atomic<int64_t> g_last_basis_update_ms{0};
std::atomic<int64_t> g_binance_update_ms{0};
std::atomic<int64_t> g_kraken_update_ms{0};
std::atomic<uint64_t> g_basis_input_revision{0};
std::atomic<bool> g_basis_ready{false};

struct BasisStats {
    std::atomic<int> count{0};
    std::atomic<double> sum{0.0};
    std::atomic<double> sum_sq{0.0};

    void add(double value) {
        count.fetch_add(1, std::memory_order_relaxed);
        atomic_add(sum, value);
        atomic_add(sum_sq, value * value);
    }

    double mean() const {
        int c = count.load(std::memory_order_relaxed);
        return c > 0 ? sum.load(std::memory_order_relaxed) / c : 0.0;
    }

    double stdev() const {
        int c = count.load(std::memory_order_relaxed);
        if (c <= 1) return 0.0;
        double m = mean();
        double variance = (sum_sq.load(std::memory_order_relaxed) / c) - (m * m);
        return variance < 0 ? 0.0 : std::sqrt(variance);
    }
} g_basis_stats;

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    g_running.store(false);
}

// Adaptive position sizing function (moved from backtest/engine.py)
static double calculate_adaptive_position_size(
    double signal_strength,
    double volatility,
    double liquidity_score,
    double recent_performance,
    double base_size,
    double max_size
) {
    // Signal strength component (0-1)
    double signal_component = std::pow(signal_strength, 1.5);

    // Volatility adjustment (inverse relationship - higher vol = smaller size)
    // Target volatility of 0.0225 (2.25%) - adjust size inversely
    double vol_target = 0.0225;
    double vol_adjustment = vol_target / (volatility + 0.001);  // Avoid division by zero
    vol_adjustment = clamp_value(vol_adjustment, 0.5, 2.0);  // Limit adjustment range

    // Liquidity component (higher liquidity = larger size)
    double liquidity_component = liquidity_score;

    // Performance component (better recent performance = larger size)
    // Using tanh to bound the performance effect between -1 and 1
    double performance_component = 1.0 + std::tanh(recent_performance);
    performance_component = clamp_value(performance_component, 0.5, 1.5);

    // Combine all components
    double size_multiplier = signal_component * vol_adjustment * liquidity_component * performance_component;

    // Calculate final size with bounds
    double adaptive_size = base_size * size_multiplier;
    adaptive_size = clamp_value(adaptive_size, base_size * 0.1, max_size);  // 10% of base_size to max_size

    return adaptive_size;
}

class AdaptiveOrderDispatcher : public SimulatedOrderDispatcher {
public:
    AdaptiveOrderDispatcher(std::shared_ptr<CircuitBreaker> risk_mgr,
                            std::string_view exchange_a,
                            std::string_view exchange_b,
                            double max_position,
                            double cooldown_ms,
                            double min_profit_bps,
                            double slippage_model_bps)
        : SimulatedOrderDispatcher(risk_mgr, exchange_a, exchange_b,
                                   max_position, cooldown_ms,
                                   min_profit_bps, slippage_model_bps) {}

    bool execute(const DispatchSignal& ds) noexcept override {
        // Reject invalid/crossed snapshots before sizing; malformed books must
        // never become a NaN or negative order quantity.
        if (!std::isfinite(ds.bid_price_a) || !std::isfinite(ds.ask_price_a) ||
            !std::isfinite(ds.bid_price_b) || !std::isfinite(ds.ask_price_b) ||
            ds.bid_price_a <= 0.0 || ds.ask_price_a <= ds.bid_price_a ||
            ds.bid_price_b <= 0.0 || ds.ask_price_b <= ds.bid_price_b ||
            !std::isfinite(ds.signal.obi_delta)) {
            return false;
        }

        const double spread_a = (ds.ask_price_a - ds.bid_price_a) / ds.bid_price_a;
        const double spread_b = (ds.ask_price_b - ds.bid_price_b) / ds.bid_price_b;
        const double liquidity_score = clamp_value(
            1.0 - ((spread_a + spread_b) * 0.5 * 10000.0), 0.0, 1.0);
        const double signal_strength = std::min(std::abs(ds.signal.obi_delta), 1.0);

        // Volatility and recent PnL are not yet measured by this demo dispatcher.
        // Use explicit neutral assumptions; do not synthesize performance data.
        constexpr double kAssumedVolatility = 0.01;
        constexpr double kNeutralRecentPerformance = 0.0;
        const double adaptive_qty = calculate_adaptive_position_size(
            signal_strength, kAssumedVolatility, liquidity_score,
            kNeutralRecentPerformance, 0.01, 0.05);

        // The base simulator is the single owner of risk checks, modeled latency,
        // fees, accounting, and order logging. Pass the adaptive size into it.
        return SimulatedOrderDispatcher::execute(ds, adaptive_qty);
    }
};

void basis_calculation_thread() {
    uint64_t last_revision = 0;
    while (g_running.load()) {
        // Calculate basis from latest mid prices
        double binance_mid = 0.0;
        double kraken_mid = 0.0;
        {
            std::lock_guard<std::mutex> lock(g_basis_mutex);
            binance_mid = g_binance_mid.load();
            kraken_mid = g_kraken_mid.load();
        }

        const uint64_t revision = g_basis_input_revision.load(std::memory_order_acquire);
        const int64_t binance_update_ms = g_binance_update_ms.load(std::memory_order_acquire);
        const int64_t kraken_update_ms = g_kraken_update_ms.load(std::memory_order_acquire);
        const int64_t last_update_ms = std::max(binance_update_ms, kraken_update_ms);
        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const auto is_fresh = [now_ms](int64_t update_ms) {
            return update_ms > 0 && now_ms >= update_ms && now_ms - update_ms <= 2000;
        };
        const bool fresh = is_fresh(binance_update_ms) && is_fresh(kraken_update_ms);
        g_basis_ready.store(fresh && binance_mid > 0.0 && kraken_mid > 0.0,
                            std::memory_order_release);

        if (revision != last_revision && fresh && binance_mid > 0.0 && kraken_mid > 0.0) {
            double basis = kraken_mid - binance_mid; // USDT/USD basis
            {
                std::lock_guard<std::mutex> lock(g_basis_mutex);
                g_current_basis = basis;
                g_basis_stats.add(basis);
            }
            last_revision = revision;

            // Log basis periodically
            static auto last_log = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (now - last_log > std::chrono::seconds(5)) {
                std::lock_guard<std::mutex> lock(g_basis_mutex);
                std::ofstream basis_log("/tmp/live_basis.csv", std::ios::app);
                if (basis_log.tellp() == 0)
                    basis_log << "timestamp_ms,binance_mid,kraken_mid,basis,basis_mean,basis_stdev\n";
                basis_log << last_update_ms << ","
                          << binance_mid << ","
                          << kraken_mid << ","
                          << basis << ","
                          << g_basis_stats.mean() << ","
                          << g_basis_stats.stdev() << "\n";
                last_log = now;
            }
        }

        // Sleep for 100ms (10 updates per second)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


} // namespace crossflux

int main(int argc, char* argv[]) {
    signal(SIGINT, crossflux::signal_handler);
    signal(SIGTERM, crossflux::signal_handler);

    std::cout << "=== Crossflux Cross-Venue Arbitrage Engine with Basis Hedging and Adaptive Sizing ===\n";
    std::cout << "Connecting to public market-data streams; execution is simulated\n\n";

    // ─── Crypto Configuration ───────────────────────────────────────────
    const std::string exchange_a = "binance";
    const std::string exchange_b = "kraken";
    const std::vector<std::string> crypto_symbols = {"btcusdt"};

    // Use parameters aligned with backtesting for realistic performance
    const double latency_mu = 3.5;
    const double latency_sigma = 0.4;
    const double alpha_lifetime_ms = 50.0;
    const double delta_threshold = 0.65;  // Aligned with backtest (was 0.10)
    const double min_p_execute = 0.80;
    const double min_spread_pct = 0.0012;  // Aligned with backtest (was 0.0005)

    const double max_position = 5.0;
    const double cooldown_ms = 5000.0;
    const double min_profit_bps = 1.0;
    const double slippage_bps = 0.3;

    try {
        // ─── Crypto Pipeline (existing) ────────────────────────────────
        // CircuitBreaker expects a negative cumulative-loss threshold.
        auto risk_mgr = std::make_shared<crossflux::CircuitBreaker>(
            -5.0, 1000, 5);

        // Use our custom adaptive dispatcher instead of the standard one
        auto dispatcher = std::make_shared<crossflux::AdaptiveOrderDispatcher>(
            risk_mgr, exchange_a, exchange_b,
            max_position, cooldown_ms, min_profit_bps, slippage_bps);

        auto signal_aggregator = std::make_shared<crossflux::SignalAggregator>(
            exchange_a, exchange_b,
            latency_mu, latency_sigma, alpha_lifetime_ms,
            delta_threshold, min_p_execute, min_spread_pct);

        // Clear old CSVs
        std::ofstream("/tmp/live_signals.csv").close();
        std::ofstream("/tmp/live_signals_py.csv").close();
        std::ofstream("/tmp/live_signals_enhanced.csv").close();
        std::ofstream("/tmp/live_basis.csv").close();
        std::ofstream("/tmp/live_status.csv").close();
        std::ofstream("/tmp/live_prices.csv").close();
        std::ofstream("/tmp/live_depth.csv").close();

        auto ingestion_engine = std::make_shared<crossflux::IngestionEngine>(
            signal_aggregator, dispatcher);
        ingestion_engine->set_book_update_callback([](
            const crossflux::OrderBookSnapshot<>& snapshot) {
            if (snapshot.bid_depth == 0 || snapshot.ask_depth == 0) return;
            const double mid = snapshot.mid_price();
            if (!std::isfinite(mid) || mid <= 0.0) return;
            std::lock_guard<std::mutex> lock(crossflux::g_basis_mutex);
            const auto received_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (snapshot.exchange() == "binance") {
                crossflux::g_binance_mid.store(mid);
                crossflux::g_binance_update_ms.store(received_ms, std::memory_order_release);
            } else if (snapshot.exchange() == "kraken") {
                crossflux::g_kraken_mid.store(mid);
                crossflux::g_kraken_update_ms.store(received_ms, std::memory_order_release);
            } else return;
            crossflux::g_last_basis_update_ms.store(received_ms, std::memory_order_release);
            crossflux::g_basis_input_revision.fetch_add(1, std::memory_order_release);
        });

        auto binance_client = std::make_shared<crossflux::BinanceWebSocketClient>(
            ingestion_engine, crypto_symbols);
        auto kraken_client = std::make_shared<crossflux::KrakenWebSocketClient>(
            ingestion_engine, crypto_symbols);

        // ─── Basis Calculation Thread ───────────────────────────────────
        std::thread basis_thread(crossflux::basis_calculation_thread);
        std::cout << "[Main] Basis calculation thread started\n";

        // ─── Start Crypto Engine ────────────────────────────────────────
        ingestion_engine->start();
        std::cout << "[Main] Ingestion engine started\n";

        binance_client->start();
        std::cout << "[Main] Binance WebSocket client started (crypto cross-arb)\n";

        kraken_client->start();
        std::cout << "[Main] Kraken WebSocket client started (crypto cross-arb)\n";

        std::cout << "\n[Main] Market-data demo running. Press Ctrl+C to stop.\n";
        std::cout << "[Main] Execution is simulated; this binary does not submit venue orders.\n\n";

        // Main loop
        while (crossflux::g_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            {
                auto now_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                auto sim_disp = std::dynamic_pointer_cast<crossflux::SimulatedOrderDispatcher>(dispatcher);
                int tot_ord = 0, fill_ord = 0;
                double rpnl = 0.0, pos = 0.0;
                if (sim_disp) {
                    tot_ord = sim_disp->total_orders();
                    fill_ord = sim_disp->filled_orders();
                    rpnl = sim_disp->realized_pnl();
                    pos = sim_disp->current_position();
                }

                // Get current basis and mid prices for status reporting
                double current_basis = 0.0;
                bool basis_ready = false;
                double binance_mid = 0.0;
                double kraken_mid = 0.0;
                {
                    std::lock_guard<std::mutex> lock(crossflux::g_basis_mutex);
                    current_basis = crossflux::g_current_basis.load();
                    basis_ready = crossflux::g_basis_ready.load();
                    binance_mid = crossflux::g_binance_mid.load();
                    kraken_mid = crossflux::g_kraken_mid.load();
                }

                std::ofstream status("/tmp/live_status.csv");
                status << "timestamp_ms,status,p_execute,total_orders,filled_orders,realized_pnl,position,"
                          "binance_mid,kraken_mid,basis,basis_ready\n"
                       << now_ts << ",running," << signal_aggregator->p_execute() << ","
                       << tot_ord << "," << fill_ord << "," << rpnl << "," << pos << ","
                       << binance_mid << "," << kraken_mid << "," << current_basis << ","
                       << (basis_ready ? 1 : 0) << "\n";
            }

            static auto last_stats = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats > std::chrono::seconds(30)) {
                double basis_mean = 0.0, basis_stdev = 0.0, basis_count = 0;
                double current_basis = 0.0;
                {
                    std::lock_guard<std::mutex> lock(crossflux::g_basis_mutex);
                    basis_mean = crossflux::g_basis_stats.mean();
                    basis_stdev = crossflux::g_basis_stats.stdev();
                    basis_count = crossflux::g_basis_stats.count;
                    current_basis = crossflux::g_current_basis.load();
                }

                std::cout << "[Main] Running... "
                          << "p_execute: " << signal_aggregator->p_execute()
                          << " | crypto signals: " << binance_client.use_count()
                          << " | market-data clients: 2 (Binance + Kraken)"
                          << " | basis: " << current_basis
                          << " (mean: " << basis_mean
                          << ", stdev: " << basis_stdev
                          << ", samples: " << basis_count << ")"
                          << std::endl;
                last_stats = now;
            }
        }

        // Shutdown
        {
            auto now_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::ofstream status("/tmp/live_status.csv");
            status << "timestamp_ms,status,p_execute,total_orders,filled_orders,realized_pnl,position,"
                      "binance_mid,kraken_mid,basis,basis_ready\n"
                   << now_ts << ",stopped,0,0,0,0,0,0,0,0,0\n";
        }

        std::cout << "\n[Main] Shutting down market-data clients...\n";
        std::cout << "[Main] Shutting down WebSocket clients...\n";
        binance_client->stop();
        kraken_client->stop();

        std::cout << "[Main] Stopping basis calculation thread...\n";
        crossflux::g_running.store(false);
        if (basis_thread.joinable()) {
            basis_thread.join();
        }

        std::cout << "[Main] Stopping ingestion engine...\n";
        ingestion_engine->stop();

        std::cout << "[Main] Live trading application stopped successfully.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Main] Fatal error: " << e.what() << std::endl;
        crossflux::g_running.store(false);
        return 1;
    }
}

