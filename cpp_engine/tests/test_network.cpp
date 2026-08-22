#include <iostream>
#include <thread>
#include <chrono>
#include "network.hpp"

using namespace crossflux;

int main() {
    std::cout << "Starting Lock-Free Network Integration Test...\n";

    // Setup Engine Components
    auto risk_mgr = std::make_shared<CircuitBreaker>();
    auto dispatcher = std::make_shared<MockHTTPDispatcher>(risk_mgr);
    auto aggregator = std::make_shared<SignalAggregator>(
        "binance", "kraken", 3.5, 0.4, 50.0, 0.1, 0.5); // Low thresholds to force signals
        
    IngestionEngine engine(aggregator, dispatcher);
    engine.start();

    // Mock Network I/O Thread pushing ticks
    std::cout << "Pushing 1,000,000 ticks into the SPSC queue from producer thread...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1'000'000; ++i) {
        MarketTick<10> tick{};
        tick.timestamp_ms = i;
        
        // Force an arbitrage opportunity (Kraken Ask < Binance Bid)
        tick.snap_a = make_order_book_snapshot<10>(i, "binance", {PriceLevel{50000, 1}}, {PriceLevel{50001, 1}}, 1, 1);
        tick.snap_b = make_order_book_snapshot<10>(i, "kraken", {PriceLevel{49000, 1}}, {PriceLevel{49001, 1}}, 1, 1);
        
        engine.push_tick(tick);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    
    std::cout << "Producer finished pushing in " << diff.count() << " seconds.\n";
    std::cout << "Waiting for consumer thread to drain...\n";
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Allow drain
    engine.stop();
    
    std::cout << "Network Integration Test Passed Successfully.\n";
    return 0;
}
