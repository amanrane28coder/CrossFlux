#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include "models.hpp"
#include "predictor.hpp"
#include "execution.hpp"

namespace crossflux {

/**
 * @brief Ingestion engine that processes market ticks and generates trading signals.
 *
 * This component connects market data feeds (WebSocket clients) to the signal
 * generation pipeline. It receives MarketTick objects, processes them through
 * the SignalAggregator to produce ArbitrageSignal objects, and forwards those
 * signals to the OrderDispatcher for execution.
 *
 * Threading model:
 *   - Public methods (submit_market_tick) are thread-safe and can be called
 *     from any thread (typically WebSocket client threads).
 *   - Internal processing happens on a dedicated evaluation thread.
 *   - Signal dispatch happens on a dedicated dispatch thread.
 */
class IngestionEngine {
public:
    /**
     * Construct an IngestionEngine.
     *
     * @param signal_aggregator Shared pointer to the signal aggregator component
     * @param dispatcher        Shared pointer to the order dispatcher component
     */
    IngestionEngine(
        std::shared_ptr<SignalAggregator> signal_aggregator,
        std::shared_ptr<OrderDispatcher> dispatcher
    );

    /** Start the ingestion engine (launchs internal processing threads). */
    void start();

    /** Stop the ingestion engine (stops internal processing threads). */
    void stop();

    /** Submit a market tick for processing (thread-safe). */
    void push_tick(const MarketTick<>& tick);

    /** Destructor. */
    ~IngestionEngine();

private:
    // Internal implementation details
    void evaluation_loop() noexcept;
    void dispatch_loop() noexcept;

    // Components
    std::shared_ptr<SignalAggregator> signal_aggregator_;
    std::shared_ptr<OrderDispatcher> dispatcher_;

    // Thread control
    std::atomic<bool> running_{false};
    std::thread eval_thread_;
    std::thread dispatch_thread_;

    // Market tick queue (WebSocket threads -> evaluation thread)
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<MarketTick<>> tick_queue_;

    // Signal queue (evaluation thread -> dispatch thread)
    std::mutex signal_mutex_;
    std::condition_variable signal_cv_;
    std::queue<DispatchSignal> signal_queue_;
};

} // namespace crossflux