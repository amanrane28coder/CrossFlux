/**
 * @file    ingestion_engine.cpp
 * @brief   Implementation of the IngestionEngine class.
 *
 * Phase 10: C++ translation of Python ingestion.py with Blueprint enhancements.
 */

#include "ingestion_engine.hpp"
#include <iostream>
#include <chrono>
#include <cmath>        // std::isnan — weighted_obi_delta "not computed" sentinel
#include <filesystem>   // std::filesystem::rename — stale CSV rotation
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <system_error> // std::error_code

namespace crossflux {

// Column layout of /tmp/live_signals.csv, which dashboard/app.py reads.
//
// Kept as a named constant because it is used twice: once to write the header
// and once to detect a stale file from a previous build. Adding a column here
// without updating the dashboard's reader leaves the new field unread but
// harmless; removing or reordering one will misalign it.
//
// weighted_obi_delta may be empty, meaning the engine was not computing it.
// obi_profile records which weight profile produced that column, so a CSV
// remains attributable after the environment variable that selected it is gone.
static constexpr const char* kSignalCsvHeader =
    "timestamp_ms,obi_delta,weighted_obi_delta,p_execute,action,"
    "exchange_a,exchange_b,obi_profile";

IngestionEngine::IngestionEngine(
    std::shared_ptr<SignalAggregator> signal_aggregator,
    std::shared_ptr<OrderDispatcher> dispatcher
)
    : signal_aggregator_(std::move(signal_aggregator))
    , dispatcher_(std::move(dispatcher))
{
}

void IngestionEngine::start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        // Already running
        return;
    }

    // Start the evaluation loop thread
    eval_thread_ = std::thread(&IngestionEngine::evaluation_loop, this);

    // Start the dispatch thread
    dispatch_thread_ = std::thread([this]() {
        try {
            while (running_.load(std::memory_order_acquire)) {
                DispatchSignal ds;
                {
                    std::unique_lock<std::mutex> lock(signal_mutex_);
                    signal_cv_.wait(lock, [this] {
                        return !signal_queue_.empty() || !running_.load(std::memory_order_acquire);
                    });

                    if (!signal_queue_.empty()) {
                        ds = signal_queue_.front();
                        signal_queue_.pop();
                    }
                }

                if (!running_.load(std::memory_order_acquire)) {
                    break;
                }

                const auto& signal = ds.signal;

                {
                    static std::mutex csv_mutex;
                    std::lock_guard<std::mutex> csv_lock(csv_mutex);

                    // Schema guard. The header is only written to an empty file,
                    // so appending to a CSV left over from an older build would
                    // silently mix 6-column and 8-column rows -- pandas then
                    // either throws or, worse, shifts every field. Detect a
                    // stale header once and rotate the file aside rather than
                    // corrupting it.
                    static bool schema_checked = false;
                    if (!schema_checked) {
                        schema_checked = true;
                        std::ifstream probe("/tmp/live_signals.csv");
                        std::string   first_line;
                        if (probe.good() && std::getline(probe, first_line)) {
                            probe.close();
                            if (first_line != kSignalCsvHeader) {
                                const std::string backup =
                                    "/tmp/live_signals.csv.stale";
                                std::error_code ec;
                                std::filesystem::rename(
                                    "/tmp/live_signals.csv", backup, ec);
                                std::cerr
                                    << "[IngestionEngine] /tmp/live_signals.csv "
                                       "has an outdated header; moved to "
                                    << backup
                                    << " and starting a fresh file. (Appending "
                                       "would have mixed column counts.)"
                                    << std::endl;
                            }
                        }
                    }

                    std::ofstream csv("/tmp/live_signals.csv", std::ios::app);
                    if (csv.tellp() == 0) {
                        csv << kSignalCsvHeader << "\n";
                    }
                    csv << signal.timestamp_ms << ","
                        << signal.obi_delta << ","
                        // NaN when the engine is not computing the weighted
                        // reading. Written as an empty field so pandas parses it
                        // as NaN instead of the string "nan" or "-nan", which
                        // would make the whole column dtype=object.
                        << (std::isnan(signal.weighted_obi_delta)
                                ? std::string{}
                                : std::to_string(signal.weighted_obi_delta))
                        << ","
                        << signal.p_execute << ","
                        << static_cast<int>(signal.action) << ","
                        << signal_aggregator_->exchange_a() << ","
                        << signal_aggregator_->exchange_b() << ","
                        << signal_aggregator_->obi_profile() << "\n";
                }
                std::cout << "[IngestionEngine] Signal received: "
                          << "timestamp=" << signal.timestamp_ms
                          << ", obi_delta=" << signal.obi_delta
                          << ", p_execute=" << signal.p_execute
                          << ", action=" << static_cast<int>(signal.action)
                          << std::endl;

                if (dispatcher_) {
                    dispatcher_->execute(ds);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[DispatchLoop] Fatal: " << e.what() << std::endl;
        }
    });

    std::cout << "[IngestionEngine] Started" << std::endl;
}

void IngestionEngine::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        // Already stopped
        return;
    }

    // Notify the evaluation thread to wake up and exit
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }

    // Notify the dispatch thread to wake up and exit
    {
        std::lock_guard<std::mutex> lock(signal_mutex_);
        signal_cv_.notify_all();
    }

    // Wait for threads to finish
    if (eval_thread_.joinable()) {
        eval_thread_.join();
    }

    if (dispatch_thread_.joinable()) {
        dispatch_thread_.join();
    }

    std::cout << "[IngestionEngine] Stopped" << std::endl;
}

void IngestionEngine::push_tick(const MarketTick<>& tick) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tick_queue_.push(tick);
    }
    queue_cv_.notify_one();
}

void IngestionEngine::evaluation_loop() noexcept {
    try {
        std::optional<OrderBookSnapshot<10>> snap_a_opt;
        std::optional<OrderBookSnapshot<10>> snap_b_opt;
        const std::string_view ex_a = signal_aggregator_->exchange_a();
        const std::string_view ex_b = signal_aggregator_->exchange_b();

        while (running_.load(std::memory_order_acquire)) {
            MarketTick<> tick;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !tick_queue_.empty() || !running_.load(std::memory_order_acquire);
                });

                if (!running_.load(std::memory_order_acquire)) {
                    break;
                }

                if (!tick_queue_.empty()) {
                    tick = tick_queue_.front();
                    tick_queue_.pop();
                }
            }

            if (!running_.load(std::memory_order_acquire)) {
                break;
            }

            // Store this tick's snapshot under the correct exchange
            // Each tick has two identical snapshots from the same exchange
            std::string_view ex = tick.snap_a.exchange();
            if (ex == ex_a) {
                snap_a_opt = tick.snap_a;
            } else if (ex == ex_b) {
                snap_b_opt = tick.snap_b;
            }

            // If we have snapshots from both exchanges, create a cross-exchange pair
            if (snap_a_opt && snap_b_opt) {
                uint64_t ts = std::min(snap_a_opt->timestamp_ms, snap_b_opt->timestamp_ms);
                MarketTick<> cross_tick = make_market_tick(ts, *snap_a_opt, *snap_b_opt);

                std::vector<ArbitrageSignal> signals = signal_aggregator_->evaluate(
                    std::vector<MarketTick<>>{cross_tick});

                for (const auto& signal : signals) {
                    DispatchSignal ds;
                    ds.signal = signal;
                    ds.bid_price_a = snap_a_opt->bids[0].price;
                    ds.ask_price_a = snap_a_opt->asks[0].price;
                    ds.bid_price_b = snap_b_opt->bids[0].price;
                    ds.ask_price_b = snap_b_opt->asks[0].price;
                    {
                        std::lock_guard<std::mutex> lock(signal_mutex_);
                        signal_queue_.push(ds);
                    }
                    signal_cv_.notify_one();
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[EvalLoop] Fatal: " << e.what() << std::endl;
    }
}
    IngestionEngine::~IngestionEngine() {
        try {
            stop();
        } catch (...) {
            // Ensure threads are detached if stop() fails
            if (eval_thread_.joinable()) eval_thread_.detach();
            if (dispatch_thread_.joinable()) dispatch_thread_.detach();
        }
    }

} // namespace crossflux