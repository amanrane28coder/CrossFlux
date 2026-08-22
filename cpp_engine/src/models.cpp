/**
 * @file    models.cpp
 * @brief   Compile-time layout verification for Argus data structures.
 *
 * This translation unit's sole purpose is to host the static_assert battery
 * that enforces struct layout contracts at build time.
 *
 * Rationale
 * ---------
 * C++ struct layout is determined by the compiler and can change silently if:
 *   - A member is added, removed, or reordered.
 *   - The compiler's ABI or alignment rules change.
 *   - A platform-specific pragma or attribute is inadvertently introduced.
 *
 * By asserting sizeof and alignof for every struct here, any layout regression
 * produces a build error with a human-readable message — not a silent
 * runtime bug that corrupts data or crashes the system under load.
 *
 * Layout reference (64-byte cache lines assumed)
 * -----------------------------------------------
 *   PriceLevel             :  16 bytes  (0 padding — natural struct)
 *   OrderBookSnapshot<10>  : 352 bytes  (6 bytes padding after depth fields)
 *   ArbitrageSignal        :  32 bytes  (7 bytes padding after TradeAction)
 *   TradeAction            :   1 byte   (enum class : uint8_t)
 */

#include "models.hpp"

namespace crossflux {

// ─────────────────────────────────────────────────────────────────────────────
// PriceLevel layout assertions
//
// Expected layout (N/A padding — two 8-byte doubles):
//   [price : 8B][volume : 8B]  →  16 bytes total
// ─────────────────────────────────────────────────────────────────────────────

static_assert(sizeof(PriceLevel)  == 16,
    "PriceLevel size changed from expected 16 bytes. "
    "Review member types and recheck the memory map in models.hpp.");

static_assert(alignof(PriceLevel) == 8,
    "PriceLevel alignment changed from expected 8 bytes. "
    "All members are double (8-byte aligned); alignment should be 8.");

static_assert(offsetof(PriceLevel, price)  == 0,
    "PriceLevel::price must be at offset 0.");

static_assert(offsetof(PriceLevel, volume) == 8,
    "PriceLevel::volume must be at offset 8.");


// ─────────────────────────────────────────────────────────────────────────────
// OrderBookSnapshot<10> layout assertions
//
// Expected layout:
//   [timestamp_ms : 8B]                              →  offset   0
//   [exchange_id  : 16B char[16]]                    →  offset   8
//   [bids         : 160B  (PriceLevel × 10)]         →  offset  24
//   [asks         : 160B  (PriceLevel × 10)]         →  offset 184
//   [bid_depth    : 1B  uint8_t]                     →  offset 344
//   [ask_depth    : 1B  uint8_t]                     →  offset 345
//   [padding      : 6B  — align next uint64_t]       →  offset 346
//   Total: 352 bytes
// ─────────────────────────────────────────────────────────────────────────────

static_assert(sizeof(OrderBookSnapshot<10>)  == 352,
    "OrderBookSnapshot<10> size changed from expected 352 bytes. "
    "Check member order: timestamp_ms(8) + exchange_id[16](16) + "
    "bids(160) + asks(160) + bid_depth(1) + ask_depth(1) + padding(6) = 352.");

static_assert(alignof(OrderBookSnapshot<10>) == 8,
    "OrderBookSnapshot<10> alignment changed from expected 8 bytes. "
    "The dominant member is uint64_t (8-byte aligned).");

static_assert(offsetof(OrderBookSnapshot<10>, timestamp_ms) == 0,
    "OrderBookSnapshot::timestamp_ms must be at offset 0.");

static_assert(offsetof(OrderBookSnapshot<10>, exchange_id) == 8,
    "OrderBookSnapshot::exchange_id must be at offset 8.");

static_assert(offsetof(OrderBookSnapshot<10>, bids) == 24,
    "OrderBookSnapshot::bids must be at offset 24 "
    "(after timestamp_ms[8] + exchange_id[16]).");

static_assert(offsetof(OrderBookSnapshot<10>, asks) == 184,
    "OrderBookSnapshot::asks must be at offset 184 "
    "(after bids: 24 + 160 = 184).");

static_assert(offsetof(OrderBookSnapshot<10>, bid_depth) == 344,
    "OrderBookSnapshot::bid_depth must be at offset 344 "
    "(after asks: 184 + 160 = 344).");

static_assert(offsetof(OrderBookSnapshot<10>, ask_depth) == 345,
    "OrderBookSnapshot::ask_depth must be at offset 345.");


// ─────────────────────────────────────────────────────────────────────────────
// ArbitrageSignal layout assertions
//
// Expected layout:
//   [timestamp_ms       : 8B  uint64_t]    →  offset  0
//   [obi_delta          : 8B  double]      →  offset  8
//   [weighted_obi_delta : 8B  double]      →  offset 16
//   [p_execute          : 8B  double]      →  offset 24
//   [action             : 1B  TradeAction] →  offset 32
//   [padding            : 7B]              →  offset 33
//   Total: 40 bytes
// ─────────────────────────────────────────────────────────────────────────────

static_assert(sizeof(ArbitrageSignal)  == 40,
    "ArbitrageSignal size changed from expected 40 bytes. "
    "Check member order: timestamp_ms(8) + obi_delta(8) + "
    "weighted_obi_delta(8) + p_execute(8) + action(1) + padding(7) = 40. "
    "This was 32 before weighted_obi_delta was added; if you are seeing 32, "
    "the build is picking up a stale models.hpp.");

static_assert(alignof(ArbitrageSignal) == 8,
    "ArbitrageSignal alignment changed from expected 8 bytes. "
    "The dominant member is uint64_t/double (8-byte aligned).");

static_assert(offsetof(ArbitrageSignal, timestamp_ms) == 0,
    "ArbitrageSignal::timestamp_ms must be at offset 0.");

static_assert(offsetof(ArbitrageSignal, obi_delta)    == 8,
    "ArbitrageSignal::obi_delta must be at offset 8.");

static_assert(offsetof(ArbitrageSignal, weighted_obi_delta) == 16,
    "ArbitrageSignal::weighted_obi_delta must be at offset 16 — "
    "adjacent to obi_delta, so the two deltas share a cache line.");

static_assert(offsetof(ArbitrageSignal, p_execute)    == 24,
    "ArbitrageSignal::p_execute must be at offset 24.");

static_assert(offsetof(ArbitrageSignal, action)       == 32,
    "ArbitrageSignal::action must be at offset 32.");


// ─────────────────────────────────────────────────────────────────────────────
// TradeAction size assertion
// ─────────────────────────────────────────────────────────────────────────────

static_assert(sizeof(TradeAction) == 1,
    "TradeAction must be exactly 1 byte (enum class : uint8_t). "
    "Check the underlying type declaration.");

} // namespace crossflux
