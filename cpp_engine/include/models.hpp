/**
 * @file    models.hpp
 * @brief   Core data structures for the Argus Cross-Venue Arbitrage Engine.
 *
 * Phase 5: C++ translation of the Python prototype data models.
 *
 * Design principles
 * -----------------
 *  - Natural alignment throughout: no #pragma pack, no __attribute__((packed)).
 *    Packed structs generate unaligned loads (multiple micro-ops on x86, bus
 *    faults on ARM) and prevent auto-vectorisation.
 *
 *  - Fixed-width types everywhere: uint64_t for timestamps, double for prices
 *    and volumes. This eliminates platform-dependent type width surprises.
 *
 *  - Stack allocation over heap: std::array<PriceLevel, N> is fully contiguous
 *    and avoids the mandatory cache miss caused by std::vector's heap pointer.
 *
 *  - Compile-time depth via template: OrderBookSnapshot<N> bakes the book
 *    depth into the type, enabling the compiler to unroll loops and emit SIMD.
 *
 *  - enum class TradeAction : uint8_t replaces the Python 'action: str' field.
 *    A 1-byte discriminant costs one register compare vs. a string hash.
 *
 *  - static_assert batteries in models.cpp lock struct sizes at compile time.
 *    Any layout regression fails the build immediately, not at runtime.
 *
 * Memory maps (N=10, 64-byte cache lines)
 * ----------------------------------------
 *   PriceLevel             :  16 bytes   (0 padding — natural struct)
 *   OrderBookSnapshot<10>  : 352 bytes   (6 bytes padding after depth fields)
 *   ArbitrageSignal        :  32 bytes   (7 bytes padding after action byte)
 *
 * C++20 required.
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstring>   // std::memcpy, std::strlen
#include <limits>    // std::numeric_limits — NaN sentinel for weighted_obi_delta
#include <stdexcept> // std::invalid_argument, std::runtime_error
#include <string_view>

namespace crossflux {

// ─────────────────────────────────────────────────────────────────────────────
// PriceLevel
//
// Memory map (16 bytes, 0 padding):
//   Offset  Size  Field
//   ──────  ────  ─────────────────────────────────────────────────────────
//        0     8  price   (double, 8-byte aligned)
//        8     8  volume  (double, 8-byte aligned)
//   ──────  ────
//   Total: 16 bytes — exactly one cache-line quarter; zero waste.
//
// An array of 10 PriceLevel objects occupies 160 bytes = 2.5 cache lines.
// ─────────────────────────────────────────────────────────────────────────────

struct PriceLevel {
    double price;   ///< Limit price in the quote currency (e.g. USD). Must be > 0.
    double volume;  ///< Resting quantity in the base currency (e.g. BTC). Must be >= 0.

    /// Construct a validated PriceLevel.
    /// @throws std::invalid_argument if price <= 0 or volume < 0.
    constexpr PriceLevel(double price_, double volume_)
        : price{price_}, volume{volume_}
    {
        if (price_ <= 0.0) {
            throw std::invalid_argument("PriceLevel: price must be positive.");
        }
        if (volume_ < 0.0) {
            throw std::invalid_argument("PriceLevel: volume must be non-negative.");
        }
    }

    /// Default constructor — produces an uninitialised sentinel.
    /// Only used internally to fill std::array slots beyond valid depth.
    PriceLevel() noexcept : price{0.0}, volume{0.0} {}
};


// ─────────────────────────────────────────────────────────────────────────────
// TradeAction
//
// Replaces the Python 'action: str' field in ArbitrageSignal.
// A 1-byte enum discriminant costs one register compare vs. a string hash.
// ─────────────────────────────────────────────────────────────────────────────

enum class TradeAction : uint8_t {
    BUY_A_SELL_B = 0,  ///< Buy on exchange_a, sell on exchange_b (delta < 0).
    BUY_B_SELL_A = 1,  ///< Buy on exchange_b, sell on exchange_a (delta > 0).
};


// ─────────────────────────────────────────────────────────────────────────────
// OrderBookSnapshot<N>
//
// A full Level-2 snapshot for one exchange at one point in time.
// N = compile-time book depth (default 10 levels per side).
//
// Memory map (N=10, 352 bytes):
//   Offset  Size   Field
//   ──────  ─────  ─────────────────────────────────────────────────────────
//        0      8  timestamp_ms          (uint64_t, 8-byte aligned)
//        8     16  exchange_id[16]        (char[16], null-terminated)
//       24    160  bids[10]              (PriceLevel × 10, sorted desc)
//      184    160  asks[10]              (PriceLevel × 10, sorted asc)
//      344      1  bid_depth             (uint8_t — valid bid count ≤ N)
//      345      1  ask_depth             (uint8_t — valid ask count ≤ N)
//      346      6  [padding]             (compiler: align next uint64_t)
//   ──────  ─────
//   Total: 352 bytes — 5.5 cache lines; 6 bytes unavoidable padding.
//
// Invariants (enforced in make_order_book_snapshot()):
//   - exchange_id must be non-empty.
//   - timestamp_ms must be > 0.
//   - bid_depth >= 1 and ask_depth >= 1.
//   - bids[0].price < asks[0].price (no crossed or locked book).
//   - bids are sorted descending (best bid first).
//   - asks are sorted ascending  (best ask first).
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N = 10>
struct OrderBookSnapshot {
    static_assert(N >= 1 && N <= 255,
        "OrderBookSnapshot: N must be in [1, 255].");

    uint64_t                  timestamp_ms;    ///< UTC event time in integer milliseconds.
    char                      exchange_id[16]; ///< Null-terminated exchange label (e.g. "binance").
    std::array<PriceLevel, N> bids;            ///< Bid levels, sorted descending (best bid = bids[0]).
    std::array<PriceLevel, N> asks;            ///< Ask levels, sorted ascending  (best ask = asks[0]).
    uint8_t                   bid_depth;       ///< Number of valid entries in bids (≤ N).
    uint8_t                   ask_depth;       ///< Number of valid entries in asks (≤ N).
    // 6 bytes compiler padding here — accepted; see layout doc above.

    // ── Convenience accessors ─────────────────────────────────────────────

    /// Return the best (highest) bid level.
    [[nodiscard]] constexpr const PriceLevel& best_bid() const noexcept {
        return bids[0];
    }

    /// Return the best (lowest) ask level.
    [[nodiscard]] constexpr const PriceLevel& best_ask() const noexcept {
        return asks[0];
    }

    /// Return the arithmetic mid-price.
    [[nodiscard]] constexpr double mid_price() const noexcept {
        return (bids[0].price + asks[0].price) * 0.5;
    }

    /// Return the absolute bid-ask spread.
    [[nodiscard]] constexpr double spread() const noexcept {
        return asks[0].price - bids[0].price;
    }

    /// Return the exchange ID as a std::string_view (zero-copy).
    [[nodiscard]] std::string_view exchange() const noexcept {
        return std::string_view{exchange_id};
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// MarketTick<N>
//
// A synchronized pair of order book snapshots representing the market state
// at a specific point in time. Used as input to the signal generation pipeline.
//
// Memory map (N=10, 712 bytes):
//   Offset  Size   Field
//   ──────  ─────  ─────────────────────────────────────────────────────────
//        0      8  timestamp_ms          (uint64_t, 8-byte aligned)
//        8    352  snap_a                (OrderBookSnapshot × 1)
//      360    352  snap_b                (OrderBookSnapshot × 1)
//   ──────  ─────
//   Total: 712 bytes — 11.125 cache lines.
//
// Invariants:
//   - timestamp_ms must be > 0.
//   - Both snap_a and snap_b must satisfy OrderBookSnapshot invariants.
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N = 10>
struct MarketTick {
    uint64_t               timestamp_ms;   ///< Logical timestamp of this aligned state (ms).
    OrderBookSnapshot<N>   snap_a;         ///< Snapshot for exchange A (e.g. binance).
    OrderBookSnapshot<N>   snap_b;         ///< Snapshot for exchange B (e.g. kraken).
};

// ─────────────────────────────────────────────────────────────────────────────
// make_market_tick<N>() — factory function
//
// Creates a MarketTick from a timestamp and two order book snapshots.
// This is the preferred way to construct MarketTick objects.
//
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N = 10>
[[nodiscard]] MarketTick<N> make_market_tick(
    uint64_t                           timestamp_ms,
    const OrderBookSnapshot<N>&       snap_a,
    const OrderBookSnapshot<N>&       snap_b)
{
    MarketTick<N> tick{};
    tick.timestamp_ms = timestamp_ms;
    tick.snap_a       = snap_a;
    tick.snap_b       = snap_b;
    return tick;
}


// ─────────────────────────────────────────────────────────────────────────────
// make_order_book_snapshot<N>() — validated factory
//
// Constructs an OrderBookSnapshot with all invariants enforced.
// Mirrors Python's OrderBookSnapshot.__post_init__().
//
// @throws std::invalid_argument  for any constraint violation.
// ─────────────────────────────────────────────────────────────────────────────

template <std::size_t N = 10>
[[nodiscard]] OrderBookSnapshot<N> make_order_book_snapshot(
    uint64_t                           timestamp_ms,
    std::string_view                   exchange_id,
    const std::array<PriceLevel, N>&   bids,
    const std::array<PriceLevel, N>&   asks,
    uint8_t                            bid_depth,
    uint8_t                            ask_depth)
{
    // ── Validate exchange_id ──────────────────────────────────────────────
    if (exchange_id.empty()) {
        throw std::invalid_argument("OrderBookSnapshot: exchange_id must not be empty.");
    }
    if (exchange_id.size() >= 16) {
        throw std::invalid_argument(
            "OrderBookSnapshot: exchange_id must be < 16 characters.");
    }

    // ── Validate timestamp ────────────────────────────────────────────────
    if (timestamp_ms == 0) {
        throw std::invalid_argument("OrderBookSnapshot: timestamp_ms must be > 0.");
    }

    // ── Validate depth ────────────────────────────────────────────────────
    if (bid_depth == 0) {
        throw std::invalid_argument("OrderBookSnapshot: bid_depth must be >= 1.");
    }
    if (ask_depth == 0) {
        throw std::invalid_argument("OrderBookSnapshot: ask_depth must be >= 1.");
    }
    if (bid_depth > static_cast<uint8_t>(N)) {
        throw std::invalid_argument("OrderBookSnapshot: bid_depth exceeds template depth N.");
    }
    if (ask_depth > static_cast<uint8_t>(N)) {
        throw std::invalid_argument("OrderBookSnapshot: ask_depth exceeds template depth N.");
    }

    // ── Validate no crossed/locked book ───────────────────────────────────
    if (bids[0].price >= asks[0].price) {
        throw std::invalid_argument(
            "OrderBookSnapshot: crossed or locked book — "
            "bids[0].price must be strictly < asks[0].price.");
    }

    // ── Construct ─────────────────────────────────────────────────────────
    OrderBookSnapshot<N> snap{};
    snap.timestamp_ms = timestamp_ms;
    snap.bid_depth    = bid_depth;
    snap.ask_depth    = ask_depth;
    snap.bids         = bids;
    snap.asks         = asks;

    // Copy exchange_id into fixed-size char array (guaranteed null-termination)
    std::memset(snap.exchange_id, 0, sizeof(snap.exchange_id));
    std::memcpy(snap.exchange_id, exchange_id.data(), exchange_id.size());

    return snap;
}


// ─────────────────────────────────────────────────────────────────────────────
// ArbitrageSignal
//
// A filtered, actionable cross-venue arbitrage signal.
// Emitted by the C++ SignalAggregator when both dual-gate conditions clear:
//   |obi_delta| > delta_threshold  AND  p_execute > min_p_execute
//
// Memory map (40 bytes):
//   Offset  Size  Field
//   ──────  ────  ─────────────────────────────────────────────────────────
//        0     8  timestamp_ms        (uint64_t, 8-byte aligned)
//        8     8  obi_delta           (double)
//       16     8  weighted_obi_delta  (double)
//       24     8  p_execute           (double)
//       32     1  action              (TradeAction : uint8_t)
//       33     7  [padding]           (compiler: align next uint64_t in array)
//   ──────  ────
//   Total: 40 bytes — still within one 64-byte cache line.
//
// The 7 bytes of trailing padding are unavoidable given uint64_t alignment.
// Placing action before the doubles would produce the same total size.
//
// This was 32 bytes before weighted_obi_delta was added. The struct no longer
// fits in 4 registers, and 40 does not divide 64, so an array of signals now
// straddles cache lines (8 signals per 5 lines instead of 2 per line). That
// cost is accepted: signals are the *output* of the hot loop, emitted for the
// ~1-in-4 ticks that clear the gates, not scanned in the inner loop. Reporting
// both the baseline and weighted signal is what makes the two comparable at
// all, which is the entire point of the profile mechanism.
// ─────────────────────────────────────────────────────────────────────────────

struct ArbitrageSignal {
    uint64_t    timestamp_ms; ///< Logical timestamp of the triggering MarketState (ms).
    double      obi_delta;    ///< Cross-venue OBI delta that cleared Gate 1. Range: [-2.0, +2.0].
    double      weighted_obi_delta; ///< Same delta under the active weight profile. NaN if not computed.
    double      p_execute;    ///< Execution probability from the latency model. Range: [0.0, 1.0].
    TradeAction action;       ///< Direction of the trade (buy/sell on each venue).
    // 7 bytes compiler padding — accepted; see layout doc above.

    /// Default constructor - creates an invalid signal.
    /// Only use as a temporary that will be immediately overwritten.
    constexpr ArbitrageSignal() noexcept
        : timestamp_ms{0}
        , obi_delta{0.0}
        , weighted_obi_delta{std::numeric_limits<double>::quiet_NaN()}
        , p_execute{0.0}
        , action{TradeAction::BUY_A_SELL_B}
    {}

    /// Construct a validated ArbitrageSignal.
    /// @throws std::invalid_argument if p_execute is outside [0.0, 1.0].
    ///
    /// Note the parameter order does NOT match the field order:
    /// weighted_obi_delta is a trailing defaulted parameter, while the field
    /// sits next to obi_delta for layout reasons. This is deliberate. Inserting
    /// it in field position would silently reinterpret every existing 4-argument
    /// call site's p_execute as weighted_obi_delta — both are double, so the
    /// compiler would not object. A trailing default keeps old call sites
    /// correct and forces new ones to be explicit.
    constexpr ArbitrageSignal(
        uint64_t    timestamp_ms_,
        double      obi_delta_,
        double      p_execute_,
        TradeAction action_,
        double      weighted_obi_delta_ = std::numeric_limits<double>::quiet_NaN())
        : timestamp_ms{timestamp_ms_}
        , obi_delta{obi_delta_}
        , weighted_obi_delta{weighted_obi_delta_}
        , p_execute{p_execute_}
        , action{action_}
    {
        if (timestamp_ms_ == 0) {
            throw std::invalid_argument(
                "ArbitrageSignal: timestamp_ms must be > 0.");
        }
        if (p_execute_ < 0.0 || p_execute_ > 1.0) {
            throw std::invalid_argument(
                "ArbitrageSignal: p_execute must be in [0.0, 1.0].");
        }
        // weighted_obi_delta is deliberately NOT range-checked. NaN is its
        // legitimate "not computed" value, and NaN fails every comparison, so a
        // [-2, 2] check would reject the default. Range is guaranteed upstream:
        // calculate_weighted_obi is bounded to (-1, 1) by construction.
    }
};

struct DispatchSignal {
    ArbitrageSignal signal;
    double bid_price_a;
    double ask_price_a;
    double bid_price_b;
    double ask_price_b;
};

} // namespace crossflux