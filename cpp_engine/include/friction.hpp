#ifndef CROSSFLUX_FRICTION_HPP
#define CROSSFLUX_FRICTION_HPP

// Execution friction: walking the book for size, and holding an order for a
// latency window before it fills.
//
// Mirrors src/friction.py. That file is the reference implementation; when the
// two disagree, the Python one is right and this is the bug.
//
// How much of that mirroring is actually checked, and how:
//
//   walk_book            bit-for-bit against src/friction.py's reference loop,
//                        1,370 cases x 6 fields, by
//                        cpp_engine/tests/run_parity_check.py --only walk.
//   FrictionModel        preset values and quantiles to ~1e-12 by
//                        test_friction.cpp f1-f6. Not bit-exact, and cannot be:
//                        latency_quantile goes through erfinv, whose polynomial
//                        the compiler may contract.
//   PendingOrderQueue    behaviour only (test_friction.cpp q1-q14). There is no
//                        Python equivalent to compare against — the backtester
//                        buffers with pandas, not a heap.
//
// Why this exists at all: the backtester used to gate on `margin > fee` and then
// book `margin - fee` from the same static snapshot, so every trade it took was
// profitable by construction. Decoupling the two needs exactly two mechanisms —
// a delay between signal and fill, and a fill price that depends on order size.
// Those are `PendingOrderQueue` and `walk_book` respectively.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "models.hpp"

namespace crossflux {
namespace friction {

// Quantities below this are zero. Book amounts are BTC and the smallest
// meaningful trade is ~1e-5 BTC, so 1e-12 sits below anything real while still
// absorbing accumulation error in the running total below. Same constant as
// src/friction.py:EPS_QTY.
inline constexpr double kEpsQty = 1e-12;

// Spelled out rather than using the M_PI macro, which is POSIX rather than ISO
// C++ and is absent under /std:c++20 on MSVC. Same digits as math.pi.
inline constexpr double kPi = 3.14159265358979323846;

// ─────────────────────────────────────────────────────────────────────────────
// Walking the book
// ─────────────────────────────────────────────────────────────────────────────

/// Result of consuming `requested_qty` from one side of one book.
struct BookWalk {
    double      vwap{0.0};             ///< Volume-weighted fill price; 0.0 means nothing filled.
    double      filled_qty{0.0};       ///< What the book could actually supply.
    std::size_t levels_consumed{0};    ///< Levels *examined*, including zero-padded ones.
    double      notional{0.0};         ///< vwap * filled_qty, i.e. cash exchanged.
    double      requested_qty{0.0};    ///< What was asked for, so shortfall is visible.

    /// Quantity the book could not supply. Zero on a complete fill.
    [[nodiscard]] constexpr double shortfall() const noexcept {
        const double s = requested_qty - filled_qty;
        return s > 0.0 ? s : 0.0;
    }

    [[nodiscard]] constexpr bool partial() const noexcept {
        return shortfall() > kEpsQty;
    }

    /// Cost of size in bps against the top-of-book price.
    ///
    /// Unsigned on purpose: a buy filling above the touch and a sell filling
    /// below it are both costs, and giving them opposite signs invites them to
    /// cancel out in an average.
    [[nodiscard]] double slippage_bps(double touch_price) const noexcept {
        if (touch_price <= 0.0 || filled_qty <= kEpsQty) {
            return 0.0;
        }
        return std::abs(vwap - touch_price) / touch_price * 1e4;
    }
};

/// Consume `qty` from a book side, cheapest level first.
///
/// `levels` must already be ordered outward from the touch (ascending for asks,
/// descending for bids). This does not sort, because a real order does not get
/// to reorder the book. Levels with non-positive price or volume are skipped:
/// exchange snapshots pad absent levels with zeros, and treating a zero-volume
/// level as depth would invent liquidity.
///
/// Note the outstanding-quantity update: `qty - taken` against a running total
/// rather than `remaining -= take`. Algebraically identical, different in
/// floating point — repeated subtraction rounds at every level, a running total
/// rounds the way a cumulative sum does. src/friction.py's vectorised path is
/// built on np.cumsum, and writing this the obvious way put the two ~2.8e-14
/// apart. A parity check that only *nearly* passes is not a parity check.
[[nodiscard]] constexpr BookWalk walk_book(std::span<const PriceLevel> levels,
                                           double qty) noexcept {
    const double want = qty > 0.0 ? qty : 0.0;
    if (qty <= kEpsQty || levels.empty()) {
        return BookWalk{0.0, 0.0, 0, 0.0, want};
    }

    double      taken    = 0.0;
    double      notional = 0.0;
    std::size_t consumed = 0;

    for (const PriceLevel& level : levels) {
        if (want - taken <= kEpsQty) {
            break;
        }
        ++consumed;
        if (level.price <= 0.0 || level.volume <= kEpsQty) {
            continue;
        }
        const double outstanding = want - taken;
        const double take = outstanding < level.volume ? outstanding : level.volume;
        notional += take * level.price;
        taken    += take;
    }

    if (taken <= kEpsQty) {
        return BookWalk{0.0, 0.0, consumed, 0.0, want};
    }
    return BookWalk{notional / taken, taken, consumed, notional, want};
}

/// Convenience overload for an OrderBookSnapshot side.
template <std::size_t N>
[[nodiscard]] BookWalk walk_asks(const OrderBookSnapshot<N>& snap, double qty) noexcept {
    return walk_book(std::span<const PriceLevel>{snap.asks.data(), snap.ask_depth}, qty);
}

template <std::size_t N>
[[nodiscard]] BookWalk walk_bids(const OrderBookSnapshot<N>& snap, double qty) noexcept {
    return walk_book(std::span<const PriceLevel>{snap.bids.data(), snap.bid_depth}, qty);
}

// ─────────────────────────────────────────────────────────────────────────────
// How long a fill takes, and what an unhedged leg costs to clean up
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse error function: Winitzki's approximation, Newton-refined twice.
///
/// Written out rather than pulled from a library so this header stays
/// dependency-free, and in the same shape as src/friction.py:_erfinv so the two
/// can be diffed. Accurate to ~1e-15, which is far beyond what a latency
/// quantile needs.
///
/// Not constexpr: std::erf and std::exp are not constant-evaluable.
[[nodiscard]] inline double erfinv(double y) {
    if (y <= -1.0 || y >= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (y == 0.0) {
        return 0.0;
    }
    constexpr double a = 0.147;
    const double ln1my2 = std::log(1.0 - y * y);
    const double t1     = 2.0 / (kPi * a) + ln1my2 / 2.0;
    double x = std::copysign(std::sqrt(std::sqrt(t1 * t1 - ln1my2 / a) - t1), y);
    for (int i = 0; i < 2; ++i) {
        const double err = std::erf(x) - y;
        x -= err / (2.0 / std::sqrt(kPi) * std::exp(-x * x));
    }
    return x;
}

/// Latency and legging assumptions for one execution environment.
///
/// Mirrors src/friction.py:FrictionModel. `latency_ms` is *one-way, per leg*, and
/// each leg draws independently — so the two halves of one arbitrage do not fill
/// at the same instant, and the quantity mismatch that leaves behind is charged
/// at `legging_cost_bps`. That charge is an assumption, not a measurement:
/// roughly one spread crossing on whichever venue over-filled. Set it to zero to
/// recover the old costless-legging behaviour and see how much of a result
/// depended on it.
struct FrictionModel {
    std::string_view name{};
    double latency_ms{0.0};
    double jitter_log_sigma{0.0};
    double legging_cost_bps{0.0};

    /// Negative latency would fill an order before the signal existed; negative
    /// legging cost would make unwinding an unwanted position a rebate. Both are
    /// nonsense rather than merely extreme, so they are rejected rather than
    /// clamped. Checked, not asserted, because release builds drop asserts and a
    /// preset built from config should fail visibly in either build.
    [[nodiscard]] constexpr bool valid() const noexcept {
        return latency_ms >= 0.0 && jitter_log_sigma >= 0.0 && legging_cost_bps >= 0.0;
    }

    [[nodiscard]] constexpr bool deterministic() const noexcept {
        return jitter_log_sigma == 0.0;
    }

    /// One latency draw, in ms.
    ///
    /// With `jitter_log_sigma == 0` this returns exactly `latency_ms` and never
    /// touches the generator, so a run is reproducible without a seed and two
    /// runs differ only by the thing under test. That default is deliberate:
    /// three Python tests were flaky against an unseeded normal draw, and a
    /// stochastic default makes preset comparisons incommensurable.
    ///
    /// Otherwise `latency_ms * exp(sigma * Z)` — multiplicative, so the *median*
    /// is exactly `latency_ms` and the tail is right-skewed the way network delay
    /// actually is. An additive Gaussian is symmetric and can go negative.
    template <class Rng>
    [[nodiscard]] double sample_latency(Rng& rng) const {
        if (deterministic()) {
            return latency_ms;
        }
        std::normal_distribution<double> z{0.0, 1.0};
        return latency_ms * std::exp(jitter_log_sigma * z(rng));
    }

    /// Latency at probability `p`, for stating the tail without sampling.
    ///
    /// Agrees with src/friction.py:latency_quantile to ~1e-12 rather than
    /// bit-exactly: `erfinv` contains a subtraction that the compiler may
    /// contract into an FMA, and this value is reported, never compared against a
    /// gate. The exact-parity guarantee is `walk_book`'s alone.
    [[nodiscard]] double latency_quantile(double p) const {
        if (!(p > 0.0 && p < 1.0)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (deterministic()) {
            return latency_ms;
        }
        const double z = std::sqrt(2.0) * erfinv(2.0 * p - 1.0);
        return latency_ms * std::exp(jitter_log_sigma * z);
    }

    /// Cost of flattening the quantity one leg acquired and the other did not.
    ///
    /// Grouped `(|qty| * price) * (bps / 1e4)` to match the Python left-to-right
    /// evaluation order exactly; regrouping changes the last bit.
    [[nodiscard]] constexpr double legging_cost(double residual_qty,
                                                double price) const noexcept {
        const double q = residual_qty < 0.0 ? -residual_qty : residual_qty;
        return q * price * (legging_cost_bps / 1e4);
    }
};

// Presets, byte-for-byte the same numbers as src/friction.py:PRESETS. Kept as
// functions rather than a map so they are usable in a constant expression and
// carry no static-init order risk.
//
// `stress` is the default and the one to quote: 100 ms is a harsh round number
// that lands almost exactly on Binance's median inter-quote gap in this sample,
// so the fill reads roughly one snapshot past the signal.
[[nodiscard]] constexpr FrictionModel preset_stress() noexcept {
    return FrictionModel{"stress", 100.0, 0.0, 5.0};
}

/// Best case for a co-located taker. Below Kraken's 6.4 ms median gap, so the
/// fill usually sees the signal's own book and friction comes almost entirely
/// from size — useful for isolating VWAP from latency.
[[nodiscard]] constexpr FrictionModel preset_colocated() noexcept {
    return FrictionModel{"colocated", 5.0, 0.0, 5.0};
}

/// Retail over the public internet: 250 ms median, heavy right tail (p99 about
/// 800 ms), wider unwind cost. The only preset with jitter, so results move
/// between runs unless the generator is seeded.
[[nodiscard]] constexpr FrictionModel preset_retail() noexcept {
    return FrictionModel{"retail", 250.0, 0.5, 10.0};
}

/// Frictionless: the fill reads the signal's own book. Diagnostic only — this is
/// the configuration that produces the tautological ~100% win rate, kept so the
/// old result stays reproducible and so friction can be measured against it.
/// Never report a return from it.
[[nodiscard]] constexpr FrictionModel preset_zero() noexcept {
    return FrictionModel{"zero", 0.0, 0.0, 0.0};
}

/// Preset by name. Returns `zero` for an unknown name *and* reports it through
/// `ok`, because silently substituting the frictionless model for a typo would
/// resurrect the tautology this module exists to break.
[[nodiscard]] inline FrictionModel get_preset(std::string_view name, bool* ok = nullptr) {
    if (ok != nullptr) {
        *ok = true;
    }
    if (name == "stress")    return preset_stress();
    if (name == "colocated") return preset_colocated();
    if (name == "retail")    return preset_retail();
    if (name == "zero")      return preset_zero();
    if (ok != nullptr) {
        *ok = false;
    }
    return preset_zero();
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffering an order across its latency window
//
// A signal at T does not book PnL at T. It goes in here, and each of its two
// legs comes back out at its own T + latency to be priced against whatever book
// is prevailing *then*. Nothing about the outcome is decided at submit time —
// that separation is the entire point, and it is why the queue cannot simply
// store the answer alongside the order.
// ─────────────────────────────────────────────────────────────────────────────

enum class Venue : std::uint8_t { kA = 0, kB = 1 };
enum class Leg   : std::uint8_t { kBuy = 0, kSell = 1 };

/// One half of an arbitrage, in flight.
struct PendingLeg {
    double fill_at_ms{0.0};    ///< Absolute time this leg resolves: T + its own latency draw.
    double signal_price{0.0};  ///< Touch price at T, kept only to attribute slippage afterwards.
    Venue  venue{Venue::kA};
};

/// An order between signal and fill.
///
/// The two legs carry independent `fill_at_ms`, which is what makes legging risk
/// representable at all: with a single shared fill time the quantities can only
/// differ because of depth, never because of timing.
struct PendingOrder {
    std::uint64_t seq{0};              ///< Submission order, for deterministic tie-breaks.
    double        signal_ts_ms{0.0};   ///< T, in market time.
    double        qty{0.0};            ///< Requested size per leg.
    double        signal_edge_bps{0.0};///< The edge that justified the trade, for later comparison.
    /// Signal strength at T, carried through only so the fill can be logged next
    /// to what triggered it. The queue never reads it.
    double        obi_delta{0.0};
    PendingLeg    buy{};
    PendingLeg    sell{};

    /// When the *order* is complete, i.e. its slower leg has resolved.
    [[nodiscard]] constexpr double ready_at_ms() const noexcept {
        return buy.fill_at_ms > sell.fill_at_ms ? buy.fill_at_ms : sell.fill_at_ms;
    }

    /// How long the position sits half-on. Zero under a deterministic preset,
    /// since both legs then draw the identical latency.
    [[nodiscard]] constexpr double leg_gap_ms() const noexcept {
        const double d = buy.fill_at_ms - sell.fill_at_ms;
        return d < 0.0 ? -d : d;
    }
};

/// An order whose legs have both resolved. This is what gets booked.
struct ResolvedOrder {
    PendingOrder order{};
    BookWalk     buy{};
    BookWalk     sell{};

    /// The part that is actually an arbitrage: quantity held on both sides.
    [[nodiscard]] constexpr double hedged_qty() const noexcept {
        return buy.filled_qty < sell.filled_qty ? buy.filled_qty : sell.filled_qty;
    }

    /// The part that is not: a naked position one venue gave us and the other
    /// did not, which has to be flattened at `FrictionModel::legging_cost`.
    [[nodiscard]] constexpr double residual_qty() const noexcept {
        const double d = buy.filled_qty - sell.filled_qty;
        return d < 0.0 ? -d : d;
    }

    [[nodiscard]] constexpr bool legged() const noexcept {
        return residual_qty() > kEpsQty;
    }

    /// True when both legs got everything they asked for.
    [[nodiscard]] constexpr bool complete() const noexcept {
        return !buy.partial() && !sell.partial();
    }

    /// Nothing filled on either side. Distinguished from a loss: the order
    /// arrived at a book that could not supply it at any price.
    [[nodiscard]] constexpr bool empty() const noexcept {
        return hedged_qty() <= kEpsQty && residual_qty() <= kEpsQty;
    }
};

/// Fixed-capacity buffer of orders awaiting their fills, drained in fill-time
/// order.
///
/// Legs, not orders, are the scheduling unit. Under a jittered preset the buy leg
/// of a later signal can resolve before the sell leg of an earlier one, so
/// insertion order is not fill order and a plain FIFO would price legs against
/// the wrong books. A binary min-heap on (fill time, seq, leg) gives O(log n)
/// with a total order that does not depend on how the compiler laid out the data.
///
/// Capacity is fixed at construction and `submit` returns false when full rather
/// than growing. Two reasons: the storage never reallocates, so a reference held
/// across a callback stays valid; and an execution path that can allocate under
/// load is not one to run against a live venue. Overflow is counted, never
/// silent — see `overflowed()`.
///
/// Time must not go backwards. If it does, the queue clamps to the last time it
/// saw and counts it in `rewound()`, because replaying a fill against an earlier
/// book is exactly the look-ahead this class exists to prevent.
///
/// `submit` is refused from inside `drain` (counted in `refused_reentrant()`).
/// Allowing it would let a zero-latency preset append legs that are already due
/// and spin the drain loop forever. The intended shape is sequential anyway:
/// drain what is due, then evaluate the new signal.
class PendingOrderQueue {
  public:
    explicit PendingOrderQueue(std::size_t capacity = 4096)
        : slots_(capacity == 0 ? 1 : capacity) {
        const std::size_t n = slots_.size();
        free_.reserve(n);
        heap_.reserve(2 * n);
        // Descending, so the first slot handed out is 0 and a trace reads in
        // submission order rather than backwards.
        for (std::size_t i = n; i-- > 0;) {
            free_.push_back(static_cast<std::uint32_t>(i));
        }
    }

    /// Buffer an order. `seq` is assigned here and overwrites whatever the caller
    /// put there, so tie-breaks stay under this class's control.
    ///
    /// Returns false if the queue is full or if called re-entrantly from a drain
    /// callback. A false return is a dropped trade and must be handled, not
    /// ignored — hence [[nodiscard]].
    [[nodiscard]] bool submit(PendingOrder order) {
        if (draining_) {
            ++refused_reentrant_;
            return false;
        }
        if (free_.empty()) {
            ++overflowed_;
            return false;
        }
        const std::uint32_t slot = free_.back();
        free_.pop_back();

        order.seq = next_seq_++;
        Slot& s = slots_[slot];
        s.order     = order;
        s.buy       = BookWalk{};
        s.sell      = BookWalk{};
        s.buy_done  = false;
        s.sell_done = false;

        push_leg(Key{order.buy.fill_at_ms,  order.seq, slot, Leg::kBuy});
        push_leg(Key{order.sell.fill_at_ms, order.seq, slot, Leg::kSell});
        ++submitted_;
        return true;
    }

    /// Resolve every leg due at or before `now_ms`, earliest first.
    ///
    /// `resolve(const PendingOrder&, Leg, double at_ms) -> BookWalk`
    ///     Called once per due leg. The caller looks up the book prevailing at
    ///     `at_ms` and walks it; the queue only decides *when* and in what order.
    ///     Returning a default BookWalk (nothing filled) is the correct answer
    ///     when no quote had printed by then — that leaves the order fully
    ///     legged, which is what actually happens.
    ///
    /// `book(const ResolvedOrder&)`
    ///     Called once per order whose second leg has just resolved. This is
    ///     where PnL is booked, including a negative one: an order that reaches
    ///     here has executed, and no path in this class can decline it.
    ///
    /// Returns the number of orders booked. Pass infinity to flush everything
    /// still in flight at the end of a run; those legs resolve against whatever
    /// the caller's as-of lookup gives for a time past the data.
    template <class Resolve, class Book>
    std::size_t drain(double now_ms, Resolve&& resolve, Book&& book) {
        if (now_ms < last_drain_ms_) {
            ++rewound_;
            now_ms = last_drain_ms_;
        }
        last_drain_ms_ = now_ms;

        draining_ = true;
        std::size_t n_booked = 0;
        while (!heap_.empty() && heap_.front().at_ms <= now_ms) {
            const Key key = heap_.front();
            std::pop_heap(heap_.begin(), heap_.end(), LaterFirst{});
            heap_.pop_back();

            Slot& s = slots_[key.slot];
            const BookWalk walk = resolve(std::as_const(s.order), key.leg, key.at_ms);
            if (key.leg == Leg::kBuy) {
                s.buy      = walk;
                s.buy_done = true;
            } else {
                s.sell      = walk;
                s.sell_done = true;
            }
            ++resolved_legs_;

            if (s.buy_done && s.sell_done) {
                book(ResolvedOrder{s.order, s.buy, s.sell});
                free_.push_back(key.slot);
                ++booked_;
                ++n_booked;
            }
        }
        draining_ = false;
        return n_booked;
    }

    /// Time of the next leg to resolve; infinity when nothing is in flight.
    [[nodiscard]] double next_due_ms() const noexcept {
        return heap_.empty() ? std::numeric_limits<double>::infinity()
                             : heap_.front().at_ms;
    }

    [[nodiscard]] std::size_t capacity()  const noexcept { return slots_.size(); }
    [[nodiscard]] std::size_t in_flight() const noexcept { return slots_.size() - free_.size(); }
    [[nodiscard]] std::size_t legs_pending() const noexcept { return heap_.size(); }
    [[nodiscard]] bool        empty()     const noexcept { return heap_.empty(); }

    [[nodiscard]] std::uint64_t submitted()         const noexcept { return submitted_; }
    [[nodiscard]] std::uint64_t booked()            const noexcept { return booked_; }
    [[nodiscard]] std::uint64_t resolved_legs()     const noexcept { return resolved_legs_; }
    [[nodiscard]] std::uint64_t overflowed()        const noexcept { return overflowed_; }
    [[nodiscard]] std::uint64_t refused_reentrant() const noexcept { return refused_reentrant_; }
    [[nodiscard]] std::uint64_t rewound()           const noexcept { return rewound_; }

  private:
    struct Key {
        double        at_ms{0.0};
        std::uint64_t seq{0};
        std::uint32_t slot{0};
        Leg           leg{Leg::kBuy};
    };

    /// Comparator for std::push_heap, which builds a *max*-heap: reporting "a
    /// sorts later than b" therefore puts the earliest leg on top.
    struct LaterFirst {
        [[nodiscard]] bool operator()(const Key& a, const Key& b) const noexcept {
            if (a.at_ms != b.at_ms) return a.at_ms > b.at_ms;
            if (a.seq   != b.seq)   return a.seq   > b.seq;
            return a.leg > b.leg;   // buy before sell, so ties are reproducible
        }
    };

    struct Slot {
        PendingOrder order{};
        BookWalk     buy{};
        BookWalk     sell{};
        bool         buy_done{false};
        bool         sell_done{false};
    };

    void push_leg(const Key& k) {
        heap_.push_back(k);
        std::push_heap(heap_.begin(), heap_.end(), LaterFirst{});
    }

    std::vector<Slot>          slots_;
    std::vector<std::uint32_t> free_;
    std::vector<Key>           heap_;

    std::uint64_t next_seq_{0};
    double        last_drain_ms_{-std::numeric_limits<double>::infinity()};
    bool          draining_{false};

    std::uint64_t submitted_{0};
    std::uint64_t booked_{0};
    std::uint64_t resolved_legs_{0};
    std::uint64_t overflowed_{0};
    std::uint64_t refused_reentrant_{0};
    std::uint64_t rewound_{0};
};

}  // namespace friction
}  // namespace crossflux

#endif  // CROSSFLUX_FRICTION_HPP
