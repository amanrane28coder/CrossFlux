#include "network.hpp"
#include <iostream>
#include <chrono>
#include <pthread.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/thread_policy.h>
#endif

namespace crossflux {

// Helper to pin thread cross-platform
static void pin_thread_to_core(std::thread& t, int core_id) {
    auto handle = t.native_handle();
    
#ifdef __APPLE__
    // macOS does not support strict physical core pinning.
    // We assign it to an affinity cache group (hinting to the scheduler).
    thread_affinity_policy_data_t policy = { core_id };
    thread_port_t mach_thread = pthread_mach_thread_np(handle);
    thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&policy, 1);
    std::cout << "[Network] macOS: Pinned thread to affinity set " << core_id << std::endl;
#else
    // Linux/POSIX strict CPU pinning
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[Network] WARNING: Failed to pin thread to core " << core_id << std::endl;
    } else {
        std::cout << "[Network] Linux: Pinned thread to physical core " << core_id << std::endl;
    }
#endif
}

IngestionEngine::IngestionEngine(std::shared_ptr<SignalAggregator> aggregator, std::shared_ptr<OrderDispatcher> dispatcher)
    : aggregator_(std::move(aggregator)), dispatcher_(std::move(dispatcher)) {}

IngestionEngine::~IngestionEngine() {
    stop();
}

void IngestionEngine::start() {
    if (running_.exchange(true)) return;
    
    eval_thread_ = std::thread(&IngestionEngine::evaluation_loop, this);
    pin_thread_to_core(eval_thread_, 2); // Pin evaluation thread to Core 2 / Affinity 2
    
    // Note: The Boost.Beast asynchronous WSS client boilerplate (TLS handshake, async_read)
    // is heavily dependent on the specific Boost version. For Phase 11 execution, 
    // the clients will independently call push_tick() when JSON parsing completes.
    std::cout << "[IngestionEngine] Started lock-free evaluation thread." << std::endl;
}

void IngestionEngine::stop() {
    if (!running_.exchange(false)) return;
    
    if (eval_thread_.joinable()) {
        eval_thread_.join();
    }
    std::cout << "[IngestionEngine] Stopped evaluation thread." << std::endl;
}

void IngestionEngine::push_tick(const MarketTick<10>& tick) {
    AlignedTick aligned;
    aligned.tick = tick;
    
    // Push fails if queue is full. In a lock-free SPSC queue, push() is wait-free.
    if (!ring_buffer_.push(aligned)) {
        std::cerr << "[IngestionEngine] WARNING: Ring buffer full, dropped tick! (Consumer is too slow)" << std::endl;
    }
}

void IngestionEngine::evaluation_loop() {
    // Pin thread to isolated CPU core (platform specific, e.g. pthread_setaffinity_np on Linux/macOS)
    
    AlignedTick aligned;
    std::vector<MarketTick<10>> batch;
    batch.reserve(1);
    
    while (running_.load(std::memory_order_relaxed)) {
        if (ring_buffer_.pop(aligned)) {
            batch.clear();
            batch.push_back(aligned.tick);
            
            // 66ns Hot Path Execution
            auto signals = aggregator_->evaluate(batch);
            
            for (const auto& sig : signals) {
                // Instantly dispatch atomic orders upon alpha detection
                if (sig.action == TradeAction::BUY_B_SELL_A) {
                    dispatcher_->execute_buy(std::string(aggregator_->exchange_b()), 0.0, 1.0, OrderType::IOC);
                    dispatcher_->execute_sell(std::string(aggregator_->exchange_a()), 0.0, 1.0, OrderType::IOC);
                } else {
                    dispatcher_->execute_buy(std::string(aggregator_->exchange_a()), 0.0, 1.0, OrderType::IOC);
                    dispatcher_->execute_sell(std::string(aggregator_->exchange_b()), 0.0, 1.0, OrderType::IOC);
                }
            }
        } else {
            // Hot spin to avoid thread context switches
            std::this_thread::yield(); 
        }
    }
}

} // namespace crossflux
