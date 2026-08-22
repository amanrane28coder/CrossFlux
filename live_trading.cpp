#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>
#include <mutex>
#include <sstream>

#include "websocket_client.hpp"
#include "predictor.hpp"
#include "execution.hpp"
#include "risk.hpp"
#include "feed_handler.hpp"

using namespace crossflux;

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_running.store(false);
}

// Write a normalized book to the multi-asset signals CSV
std::mutex g_csv_mutex;
void log_signal(const NormalizedOrderBook& nb, AssetClass ac) {
    std::lock_guard<std::mutex> lock(g_csv_mutex);
    std::ofstream f("/tmp/live_signals_py.csv", std::ios::app);
    if (f.tellp() == 0)
        f << "timestamp_ms,symbol,asset_class,bid_price,bid_qty,ask_price,ask_qty\n";
    f << nb.timestamp_ms << ","
      << nb.symbol << ","
      << (ac == AssetClass::CRYPTO ? "crypto" : "equity") << ","
      << nb.bid_price << ","
      << nb.bid_qty << ","
      << nb.ask_price << ","
      << nb.ask_qty << "\n";
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "=== Crossflux Cross-Venue Arbitrage Engine ===\n";
    std::cout << "Connecting to exchange WebSocket streams for real-time data\n\n";

    // ─── Crypto Configuration ───────────────────────────────────────────
    const std::string exchange_a = "binance";
    const std::string exchange_b = "kraken";
    const std::vector<std::string> crypto_symbols = {"btcusdt"};

    const double latency_mu = 3.5;
    const double latency_sigma = 0.4;
    const double alpha_lifetime_ms = 50.0;
    const double delta_threshold = 0.10;
    const double min_p_execute = 0.80;
    const double min_spread_pct = 0.0005;
    const double qty = 0.01;

    const double max_position = 5.0;
    const double cooldown_ms = 5000.0;
    const double min_profit_bps = 1.0;
    const double slippage_bps = 0.3;

    // ─── Equity Configuration ───────────────────────────────────────────
    const std::vector<std::string> equity_symbols = {"AAPL"};

    try {
        // ─── Crypto Pipeline (existing) ────────────────────────────────
        auto risk_mgr = std::make_shared<CircuitBreaker>(
            5.0, 1000.0, 5.0);

        auto dispatcher = std::make_shared<SimulatedOrderDispatcher>(
            risk_mgr, exchange_a, exchange_b,
            max_position, cooldown_ms, min_profit_bps, slippage_bps);

        auto signal_aggregator = std::make_shared<SignalAggregator>(
            exchange_a, exchange_b,
            latency_mu, latency_sigma, alpha_lifetime_ms,
            delta_threshold, min_p_execute, min_spread_pct);

        auto ingestion_engine = std::make_shared<IngestionEngine>(
            signal_aggregator, dispatcher);

        auto binance_client = std::make_shared<BinanceWebSocketClient>(
            ingestion_engine, crypto_symbols);
        auto kraken_client = std::make_shared<KrakenWebSocketClient>(
            ingestion_engine, crypto_symbols);

        // ─── Multi-Asset Feeds ──────────────────────────────────────────
        auto binance_feed = create_binance_feed(crypto_symbols);
        auto alpaca_feed = create_alpaca_feed(equity_symbols);

        binance_feed->connect([](const NormalizedOrderBook& nb) {
            log_signal(nb, AssetClass::CRYPTO);
        });

        alpaca_feed->connect([](const NormalizedOrderBook& nb) {
            log_signal(nb, AssetClass::EQUITY);
        });

        // ─── Start Crypto Engine ────────────────────────────────────────
        ingestion_engine->start();
        std::cout << "[Main] Ingestion engine started\n";

        binance_client->start();
        std::cout << "[Main] Binance WebSocket client started (crypto cross-arb)\n";

        kraken_client->start();
        std::cout << "[Main] Kraken WebSocket client started (crypto cross-arb)\n";

        std::cout << "[Main] BinanceFeed started (multi-asset normalized books)\n";
        std::cout << "[Main] AlpacaFeed started (equity: ";
        for (const auto& s : equity_symbols) std::cout << s << " ";
        std::cout << ")\n";

        // Clear old CSVs
        std::ofstream("/tmp/live_signals.csv").close();
        std::ofstream("/tmp/live_signals_py.csv").close();
        std::ofstream("/tmp/live_status.csv").close();
        std::ofstream("/tmp/live_prices.csv").close();
        std::ofstream("/tmp/live_depth.csv").close();

        std::cout << "\n[Main] Live trading engine running. Press Ctrl+C to stop.\n";
        std::cout << "[Main] Streaming crypto (binance↔kraken) + equities (alpaca)\n\n";

        // Main loop
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            {
                auto now_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                auto sim_disp = std::dynamic_pointer_cast<SimulatedOrderDispatcher>(dispatcher);
                int tot_ord = 0, fill_ord = 0;
                double rpnl = 0.0, pos = 0.0;
                if (sim_disp) {
                    tot_ord = sim_disp->total_orders();
                    fill_ord = sim_disp->filled_orders();
                    rpnl = sim_disp->realized_pnl();
                    pos = sim_disp->current_position();
                }
                std::ofstream status("/tmp/live_status.csv");
                status << "timestamp_ms,status,p_execute,total_orders,filled_orders,realized_pnl,position\n"
                       << now_ts << ",running," << signal_aggregator->p_execute() << ","
                       << tot_ord << "," << fill_ord << "," << rpnl << "," << pos << "\n";
            }

            static auto last_stats = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats > std::chrono::seconds(30)) {
                std::cout << "[Main] Running... "
                          << "p_execute: " << signal_aggregator->p_execute()
                          << " | crypto signals: " << binance_client.use_count()
                          << " | equity feeds: 2 (binance_feed + alpaca_feed)"
                          << std::endl;
                last_stats = now;
            }
        }

        // Shutdown
        {
            auto now_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::ofstream status("/tmp/live_status.csv");
            status << "timestamp_ms,status,p_execute,total_orders,filled_orders,realized_pnl,position\n"
                   << now_ts << ",stopped,0,0,0,0,0\n";
        }

        std::cout << "\n[Main] Shutting down multi-asset feeds...\n";
        binance_feed->disconnect();
        alpaca_feed->disconnect();

        std::cout << "[Main] Shutting down WebSocket clients...\n";
        binance_client->stop();
        kraken_client->stop();

        std::cout << "[Main] Stopping ingestion engine...\n";
        ingestion_engine->stop();

        std::cout << "[Main] Live trading application stopped successfully.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Main] Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
