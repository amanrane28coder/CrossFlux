#pragma once

#include <boost/lockfree/spsc_queue.hpp>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include "models.hpp"
#include "predictor.hpp"
#include "execution.hpp"

namespace crossflux {

/**
 * @brief Cache-line aligned MarketTick wrapper.
 * 
 * MarketTick<10> is 712 bytes. By forcing 64-byte alignment, 
 * the compiler pads the struct to exactly 768 bytes (12 full cache lines).
 * This prevents false sharing and cache tearing during the SPSC ring buffer handoff.
 */
struct alignas(64) AlignedTick {
    MarketTick<10> tick;
};

// Compile-time assertion to guarantee size and alignment
static_assert(sizeof(AlignedTick) == 768, "AlignedTick must be exactly 768 bytes");
static_assert(alignof(AlignedTick) == 64, "AlignedTick must be 64-byte aligned");

/**
 * @brief Lock-free Single-Producer Single-Consumer queue for tick handoff.
 * Capacity is fixed at compile-time (1024 elements = ~786 KB).
 */
using TickQueue = boost::lockfree::spsc_queue<AlignedTick, boost::lockfree::capacity<1024>>;

// IngestionEngine removed — duplicate with ingestion_engine.hpp.
// Use ingestion_engine.hpp instead (included via websocket_client.hpp).

} // namespace crossflux
