/**
 * @file    bindings.cpp
 * @brief   pybind11 Python bindings for the Argus C++ arbitrage engine.
 *
 * Phase 8: Exposes the C++ decision core as the native Python module `argus_engine`.
 *
 * Template fixation strategy
 * --------------------------
 * All C++ types are templated on book depth N.  Python has no template concept,
 * so this binding layer fixes N = 10 for every exposed type.  Python callers
 * see plain `OrderBookSnapshot`, `MarketTick`, etc. — the depth is an invisible
 * implementation detail.
 *
 * Type alias shorthands used throughout:
 *   Snap10 = crossflux::OrderBookSnapshot<10>
 *   Tick10 = crossflux::MarketTick<10>
 *
 * Module contents (in declaration order)
 * ----------------------------------------
 *   1. PriceLevel             — price + volume, validated constructor
 *   2. TradeAction            — Python enum mirroring C++ enum class
 *   3. OrderBookSnapshot      — N=10 snapshot + read-only bids/asks lists
 *   4. make_order_book_snapshot() — Python-friendly factory function
 *   5. MarketTick             — N=10 pair of snapshots
 *   6. make_market_tick()     — Python-friendly factory function
 *   7. ArbitrageSignal        — signal with str `.action` property
 *   8. SignalAggregator       — constructor + evaluate() + diagnostic properties
 *
 * Module name: arbitrage_engine
 * pybind11 3.0+ required.  C++20 required.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // automatic std::vector <-> Python list conversion
#include <pybind11/operators.h>  // py::self operators

#include <cmath>     // std::isnan — weighted_obi_delta "not computed" sentinel
#include <sstream>
#include <stdexcept>
#include <string>
#include <span>
#include <vector>

#include "models.hpp"
#include "predictor.hpp"
#include "signals.hpp"
#include "obi_config.hpp"
#include "risk.hpp"

namespace py = pybind11;

// Convenience aliases — fixes the template parameter at the binding boundary
using Snap10 = crossflux::OrderBookSnapshot<10>;
using Tick10 = crossflux::MarketTick<10>;


// ─────────────────────────────────────────────────────────────────────────────
// Python-friendly factory helpers (defined before the module macro)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build an OrderBookSnapshot<10> from Python lists of PriceLevel objects.
 *
 * Accepts lists of arbitrary length ≤ 10.  Fills the std::array<PriceLevel,10>
 * with the provided levels; remaining slots hold default-constructed sentinels.
 * Sets bid_depth and ask_depth to the length of the input lists.
 *
 * @throws py::value_error  if depth > 10, or if any PriceLevel is invalid.
 * @throws py::value_error  mirrors the C++ make_order_book_snapshot invariants.
 */
[[nodiscard]] static Snap10
py_make_order_book_snapshot(
    uint64_t                               timestamp_ms,
    const std::string&                     exchange_id,
    const std::vector<crossflux::PriceLevel>&  bids_in,
    const std::vector<crossflux::PriceLevel>&  asks_in)
{
    if (bids_in.size() > 10) {
        throw py::value_error("make_order_book_snapshot: bids list length must be ≤ 10");
    }
    if (asks_in.size() > 10) {
        throw py::value_error("make_order_book_snapshot: asks list length must be ≤ 10");
    }
    if (bids_in.empty()) {
        throw py::value_error("make_order_book_snapshot: bids list must not be empty");
    }
    if (asks_in.empty()) {
        throw py::value_error("make_order_book_snapshot: asks list must not be empty");
    }

    std::array<crossflux::PriceLevel, 10> bids{};
    std::array<crossflux::PriceLevel, 10> asks{};
    for (std::size_t i = 0; i < bids_in.size(); ++i) bids[i] = bids_in[i];
    for (std::size_t i = 0; i < asks_in.size(); ++i) asks[i] = asks_in[i];

    const auto bid_depth = static_cast<uint8_t>(bids_in.size());
    const auto ask_depth = static_cast<uint8_t>(asks_in.size());

    // Delegates to the validated C++ factory — all book invariants enforced here
    return crossflux::make_order_book_snapshot<10>(
        timestamp_ms, exchange_id, bids, asks, bid_depth, ask_depth);
}


/**
 * Build a MarketTick<10> from a timestamp and two Python OrderBookSnapshot objects.
 * This is the primary entry point for building input data from Python.
 */
[[nodiscard]] static Tick10
py_make_market_tick(uint64_t ts, const Snap10& snap_a, const Snap10& snap_b)
{
    Tick10 tick{};
    tick.timestamp_ms = ts;
    tick.snap_a       = snap_a;
    tick.snap_b       = snap_b;
    return tick;
}


/**
 * Convert a TradeAction enum to its canonical string representation.
 * Matches the Python prototype's action: str field exactly.
 */
[[nodiscard]] static std::string
trade_action_to_str(crossflux::TradeAction action) noexcept
{
    return (action == crossflux::TradeAction::BUY_B_SELL_A)
        ? "BUY_B_SELL_A"
        : "BUY_A_SELL_B";
}


// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(arbitrage_engine, m)
{
    m.doc() =
        "Arbitrage C++ Cross-Venue Arbitrage Engine — Phase 8 Python Bindings.\n\n"
        "Exposes the C++ decision core (Phase 5–7) as a native Python extension.\n"
        "All types are instantiated at book depth N=10.\n\n"
        "Quickstart::\n\n"
        "    import arbitrage_engine as ae\n"
        "    snap_a = ae.make_order_book_snapshot(ts, 'binance',\n"
        "                 [ae.PriceLevel(49999, 10)], [ae.PriceLevel(50001, 1)])\n"
        "    snap_b = ae.make_order_book_snapshot(ts, 'kraken',\n"
        "                 [ae.PriceLevel(49999, 1)],  [ae.PriceLevel(50001, 1)])\n"
        "    tick   = ae.make_market_tick(ts, snap_a, snap_b)\n"
        "    agg    = ae.SignalAggregator('binance', 'kraken', 3.5, 0.4, 50.0)\n"
        "    sigs   = agg.evaluate([tick])\n";


    // ── 1. PriceLevel ─────────────────────────────────────────────────────────
    py::class_<crossflux::PriceLevel>(m, "PriceLevel",
        "A single resting order at one price level.\n\n"
        "Attributes\n----------\n"
        "price  : float — limit price in quote currency (must be > 0)\n"
        "volume : float — resting quantity in base currency (must be >= 0)")
        .def(py::init<double, double>(),
             py::arg("price"), py::arg("volume"),
             "Construct a validated PriceLevel.\n\n"
             "Raises ValueError if price <= 0 or volume < 0.")
        .def_readonly("price",  &crossflux::PriceLevel::price,
                      "Limit price in quote currency.")
        .def_readonly("volume", &crossflux::PriceLevel::volume,
                      "Resting quantity in base currency.")
        .def("__repr__", [](const crossflux::PriceLevel& p) {
            std::ostringstream oss;
            oss << "PriceLevel(price=" << p.price << ", volume=" << p.volume << ")";
            return oss.str();
        });


    // ── 2. TradeAction enum ───────────────────────────────────────────────────
    py::enum_<crossflux::TradeAction>(m, "TradeAction",
        "Direction of the arbitrage trade.\n\n"
        "BUY_A_SELL_B : buy on exchange A, sell on exchange B (delta < 0)\n"
        "BUY_B_SELL_A : buy on exchange B, sell on exchange A (delta > 0)")
        .value("BUY_A_SELL_B", crossflux::TradeAction::BUY_A_SELL_B)
        .value("BUY_B_SELL_A", crossflux::TradeAction::BUY_B_SELL_A)
        .export_values();


    // ── 3. OrderBookSnapshot (N=10 fixed) ─────────────────────────────────────
    py::class_<Snap10>(m, "OrderBookSnapshot",
        "Level-2 order book snapshot for one exchange (depth N=10).\n\n"
        "Construct via make_order_book_snapshot(), not directly.")
        .def_readonly("timestamp_ms", &Snap10::timestamp_ms,
                      "UTC event timestamp in integer milliseconds.")
        .def_readonly("bid_depth", &Snap10::bid_depth,
                      "Number of valid bid levels (≤ 10).")
        .def_readonly("ask_depth", &Snap10::ask_depth,
                      "Number of valid ask levels (≤ 10).")
        .def_property_readonly("exchange_id",
            [](const Snap10& s) { return std::string(s.exchange_id); },
            "Exchange identifier string (e.g. 'binance').")
        .def_property_readonly("bids",
            [](const Snap10& s) {
                // Return only the valid levels — sentinel slots are invisible
                std::vector<crossflux::PriceLevel> out;
                out.reserve(s.bid_depth);
                for (uint8_t i = 0; i < s.bid_depth; ++i) out.push_back(s.bids[i]);
                return out;
            },
            "Bid levels sorted descending (best bid first). Read-only list.")
        .def_property_readonly("asks",
            [](const Snap10& s) {
                std::vector<crossflux::PriceLevel> out;
                out.reserve(s.ask_depth);
                for (uint8_t i = 0; i < s.ask_depth; ++i) out.push_back(s.asks[i]);
                return out;
            },
            "Ask levels sorted ascending (best ask first). Read-only list.")
        .def_property_readonly("mid_price", &Snap10::mid_price,
                               "Arithmetic mid-price: (best_bid + best_ask) / 2.")
        .def_property_readonly("spread",    &Snap10::spread,
                               "Absolute bid-ask spread: best_ask - best_bid.")
        .def("__repr__", [](const Snap10& s) {
            std::ostringstream oss;
            oss << "OrderBookSnapshot(exchange='" << s.exchange_id
                << "', ts=" << s.timestamp_ms
                << ", bid_depth=" << static_cast<int>(s.bid_depth)
                << ", ask_depth=" << static_cast<int>(s.ask_depth)
                << ", mid=" << s.mid_price()
                << ")";
            return oss.str();
        });


    // ── 4. make_order_book_snapshot factory ───────────────────────────────────
    m.def("make_order_book_snapshot", &py_make_order_book_snapshot,
          py::arg("timestamp_ms"),
          py::arg("exchange_id"),
          py::arg("bids"),
          py::arg("asks"),
          "Build a validated OrderBookSnapshot from Python lists of PriceLevel.\n\n"
          "Parameters\n----------\n"
          "timestamp_ms : int   — UTC event time in milliseconds (must be > 0)\n"
          "exchange_id  : str   — exchange label, e.g. 'binance' (max 15 chars)\n"
          "bids         : list[PriceLevel] — bid levels, sorted descending (len ≤ 10)\n"
          "asks         : list[PriceLevel] — ask levels, sorted ascending  (len ≤ 10)\n\n"
          "Raises\n------\n"
          "ValueError on any constraint violation (crossed book, empty sides, etc.)");


    // ── 5. MarketTick (N=10 fixed) ────────────────────────────────────────────
    py::class_<Tick10>(m, "MarketTick",
        "A temporally aligned pair of exchange snapshots (depth N=10).\n\n"
        "C++ equivalent of Python's MarketState, but stack-allocated.\n"
        "Construct via make_market_tick().")
        .def_readonly("timestamp_ms", &Tick10::timestamp_ms,
                      "Logical UTC timestamp (ms) of this aligned state.")
        .def_readwrite("snap_a", &Tick10::snap_a,
                       "Snapshot for exchange A (e.g. binance).")
        .def_readwrite("snap_b", &Tick10::snap_b,
                       "Snapshot for exchange B (e.g. kraken).")
        .def("__repr__", [](const Tick10& t) {
            std::ostringstream oss;
            oss << "MarketTick(ts=" << t.timestamp_ms
                << ", a='" << t.snap_a.exchange_id
                << "', b='" << t.snap_b.exchange_id << "')";
            return oss.str();
        });


    // ── 6. make_market_tick factory ───────────────────────────────────────────
    m.def("make_market_tick", &py_make_market_tick,
          py::arg("timestamp_ms"),
          py::arg("snap_a"),
          py::arg("snap_b"),
          "Build a MarketTick from a timestamp and two OrderBookSnapshot objects.\n\n"
          "Parameters\n----------\n"
          "timestamp_ms : int               — logical UTC timestamp (ms)\n"
          "snap_a       : OrderBookSnapshot — snapshot for exchange A\n"
          "snap_b       : OrderBookSnapshot — snapshot for exchange B");


    // ── 7. ArbitrageSignal ────────────────────────────────────────────────────
    py::class_<crossflux::ArbitrageSignal>(m, "ArbitrageSignal",
        "A filtered, actionable cross-venue arbitrage signal.\n\n"
        "Emitted by SignalAggregator.evaluate() when both gates clear:\n"
        "  |obi_delta| > delta_threshold  AND  p_execute > min_p_execute\n\n"
        "Attributes\n----------\n"
        "timestamp_ms       : int   — triggering MarketTick timestamp (ms)\n"
        "obi_delta          : float — cross-venue OBI delta in [-2.0, +2.0]\n"
        "weighted_obi_delta : float — same delta under the active weight\n"
        "                             profile, or NaN if not computed\n"
        "p_execute          : float — execution probability in [0.0, 1.0]\n"
        "action             : str   — 'BUY_A_SELL_B' or 'BUY_B_SELL_A'\n"
        "action_enum        : TradeAction — type-safe enum variant")
        .def_readonly("timestamp_ms", &crossflux::ArbitrageSignal::timestamp_ms,
                      "Triggering timestamp in milliseconds.")
        .def_readonly("obi_delta",    &crossflux::ArbitrageSignal::obi_delta,
                      "Cross-venue OBI delta value. Range: [-2.0, +2.0].")
        .def_readonly("weighted_obi_delta",
                      &crossflux::ArbitrageSignal::weighted_obi_delta,
                      "Cross-venue delta recomputed with the active per-level\n"
                      "weight profile. Range: [-2.0, +2.0].\n\n"
                      "float('nan') means the engine did not compute it — the\n"
                      "profile is flat and not gating, so this would merely\n"
                      "duplicate obi_delta. Test with math.isnan(), not == 0.0:\n"
                      "0.0 is a legitimate value meaning both books are\n"
                      "balanced, which is a different statement entirely.")
        .def_readonly("p_execute",    &crossflux::ArbitrageSignal::p_execute,
                      "Execution probability from the latency model. Range: [0.0, 1.0].")
        .def_property_readonly("action",
            [](const crossflux::ArbitrageSignal& s) { return trade_action_to_str(s.action); },
            "Trade directive string: 'BUY_A_SELL_B' or 'BUY_B_SELL_A'.\n"
            "Matches the Python prototype's action: str field exactly.")
        .def_readonly("action_enum", &crossflux::ArbitrageSignal::action,
                      "Type-safe TradeAction enum variant.")
        .def("__repr__", [](const crossflux::ArbitrageSignal& s) {
            std::ostringstream oss;
            oss << "ArbitrageSignal(ts=" << s.timestamp_ms
                << ", obi_delta=" << s.obi_delta;
            // Omitted entirely when not computed, rather than printing "nan".
            // A repr that shows only the fields the engine actually populated
            // makes the profile configuration visible at a glance in a REPL.
            if (!std::isnan(s.weighted_obi_delta)) {
                oss << ", weighted_obi_delta=" << s.weighted_obi_delta;
            }
            oss << ", p_execute=" << s.p_execute
                << ", action='" << trade_action_to_str(s.action) << "')";
            return oss.str();
        });


    // ── 8. SignalAggregator ───────────────────────────────────────────────────
    py::class_<crossflux::SignalAggregator>(m, "SignalAggregator",
        "Iterates over MarketTick batches and emits ArbitrageSignal events.\n\n"
        "The execution probability (p_execute) is pre-computed once in the\n"
        "constructor via the log-normal CDF.  The evaluate() hot loop is pure\n"
        "FMA arithmetic with zero transcendental function calls.\n\n"
        "Parameters\n----------\n"
        "exchange_a        : str   — venue A identifier (e.g. 'binance')\n"
        "exchange_b        : str   — venue B identifier (e.g. 'kraken')\n"
        "latency_mu        : float — log-normal μ (mean of ln(RTT))\n"
        "latency_sigma     : float — log-normal σ, must be > 0\n"
        "alpha_lifetime_ms : float — discrepancy lifetime (ms), must be > 0\n"
        "delta_threshold   : float — Gate 1 minimum |ΔOBI| (default 0.3)\n"
        "min_p_execute     : float — Gate 2 minimum p_execute (default 0.80)\n"
        "min_spread_pct    : float — Gate 3 minimum spread pct (default 0.0012)")
        .def(py::init<
                std::string_view, std::string_view,
                double, double, double, double, double, double>(),
             py::arg("exchange_a"),
             py::arg("exchange_b"),
             py::arg("latency_mu"),
             py::arg("latency_sigma"),
             py::arg("alpha_lifetime_ms"),
             py::arg("delta_threshold") = crossflux::SignalAggregator::kDefaultDeltaThreshold,
             py::arg("min_p_execute")   = crossflux::SignalAggregator::kDefaultMinPExecute,
             py::arg("min_spread_pct")  = 0.0012,
             "Construct and configure the SignalAggregator.\n\n"
             "Pre-computes p_execute via the log-normal CDF on construction.\n"
             "Raises ValueError on any invalid parameter.")
        .def("evaluate",
            [](const crossflux::SignalAggregator& agg,
               const std::vector<Tick10>& ticks) {
                // pybind11/stl.h handles std::vector<ArbitrageSignal> -> Python list
                return agg.evaluate(ticks);
            },
            py::arg("ticks"),
            "Evaluate a list of MarketTick objects and return ArbitrageSignals.\n\n"
            "Parameters\n----------\n"
            "ticks : list[MarketTick] — batch of aligned market updates\n\n"
            "Returns\n-------\n"
            "list[ArbitrageSignal] — signals that cleared both threshold gates\n\n"
            "The hot path per tick: 2×OBI + 1×delta + 2×gate compares. Zero erf calls.")
        .def_property_readonly("p_execute",
            &crossflux::SignalAggregator::p_execute,
            "Pre-computed execution probability (log-normal CDF result).")
        .def_property_readonly("delta_threshold",
            &crossflux::SignalAggregator::delta_threshold,
            "Gate 1 minimum |ΔOBI| threshold.")
        .def_property_readonly("min_p_execute",
            &crossflux::SignalAggregator::min_p_execute,
            "Gate 2 minimum execution probability threshold.")
        .def_property_readonly("min_spread_pct",
            &crossflux::SignalAggregator::min_spread_pct,
            "Gate 3 minimum spread percentage.")
        .def_property_readonly("exchange_a",
            [](const crossflux::SignalAggregator& a) {
                return std::string(a.exchange_a());
            },
            "Venue A identifier.")
        .def_property_readonly("exchange_b",
            [](const crossflux::SignalAggregator& a) {
                return std::string(a.exchange_b());
            },
            "Venue B identifier.")
        .def("__repr__", [](const crossflux::SignalAggregator& a) {
            std::ostringstream oss;
            oss << "SignalAggregator(exchange_a='" << a.exchange_a()
                << "', exchange_b='" << a.exchange_b()
                << "', delta_threshold=" << a.delta_threshold()
                << ", min_p_execute=" << a.min_p_execute()
                << ", min_spread_pct=" << a.min_spread_pct()
                << ", p_execute=" << a.p_execute() << ")";
            return oss.str();
        });

    // ── 9. CircuitBreaker ─────────────────────────────────────────────────────
    py::class_<crossflux::CircuitBreaker>(m, "CircuitBreaker",
        "Hard risk controls evaluated on the hot execution path.\n\n"
        "Trips automatically if cumulative loss, consecutive timeouts, or\n"
        "consecutive severe slippage exceeds defined limits.")
        .def(py::init<>())
        .def_property_readonly("is_tripped", &crossflux::CircuitBreaker::is_tripped,
                               "True if trading should be halted immediately.")
        .def_property_readonly("cumulative_pnl", &crossflux::CircuitBreaker::get_cumulative_pnl,
                               "Current cumulative net PnL (USD).")
        .def("record_trade_pnl", &crossflux::CircuitBreaker::record_trade_pnl,
             py::arg("pnl_net"),
             "Record the net PnL of a trade. Trips if max daily loss is reached.")
        .def("record_timeout", &crossflux::CircuitBreaker::record_timeout,
             "Record an API timeout. Trips if consecutive timeouts limit is reached.")
        .def("reset_timeouts", &crossflux::CircuitBreaker::reset_timeouts,
             "Reset consecutive timeouts after a successful API response.")
        .def("record_slippage", &crossflux::CircuitBreaker::record_slippage,
             py::arg("exceeded"),
             "Record if slippage exceeded 1.5x spread. Trips if consecutive.")
        .def("__repr__", [](const crossflux::CircuitBreaker& c) {
            std::ostringstream oss;
            oss << "CircuitBreaker(tripped=" << (c.is_tripped() ? "True" : "False")
                << ", pnl=" << c.get_cumulative_pnl() << ")";
            return oss.str();
        });

    // ── 10. Signal functions ──────────────────────────────────────────────────
    //
    // These were previously unreachable from Python: the module exposed
    // SignalAggregator, which calls calculate_obi internally, but never the
    // functions themselves. That made the claim "the C++ and Python signals
    // agree" untestable from the test suite — the only way to compare them was
    // a throwaway C++ harness. Exposing them makes parity a real assertion.

    m.def("calculate_obi", &crossflux::calculate_obi<10>,
          py::arg("snapshot"),
          "Unweighted Order Book Imbalance over the top levels of a snapshot.\n\n"
          "Consumes min(bid_depth, ask_depth) levels -- the shallower side wins.\n"
          "Note this differs from src.signals.calculate_obi, which slices each\n"
          "side independently. The two therefore disagree on asymmetric books.\n"
          "That divergence is pre-existing and deliberately left in place:\n"
          "changing it would move already-published backtest numbers. Use\n"
          "calculate_weighted_obi with the 'flat' profile if you need a function\n"
          "that agrees across both languages.\n\n"
          "Returns a float in (-1.0, 1.0), or 0.0 for a zero-volume book.");

    m.def("calculate_weighted_obi",
          [](const Snap10& snap, const std::vector<double>& weights) {
              // pybind converts the Python sequence into a std::vector; the
              // span borrows it for the duration of the call only.
              return crossflux::calculate_weighted_obi<10>(
                  snap, std::span<const double>{weights.data(), weights.size()});
          },
          py::arg("snapshot"), py::arg("weights"),
          "Multi-level OBI with explicit per-level weights.\n\n"
          "weights[0] applies to the touch. Consumes\n"
          "min(bid_depth, ask_depth, len(weights)) levels. Weights must be\n"
          "non-negative; negative weights break the (-1, 1) bound that\n"
          "calculate_obi_delta's assert relies on.\n\n"
          "This is NOT MLOFI. MLOFI is computed from successive book deltas and\n"
          "measures order flow; this weights a static depth snapshot. See\n"
          "src/obi_weights.py.\n\n"
          "Pass src.obi_weights.active().weights to use the configured profile:\n\n"
          "    from src.obi_weights import active\n"
          "    ae.calculate_weighted_obi(snap, list(active().weights))\n\n"
          "Agrees with src.signals.calculate_weighted_obi to within one ULP.\n"
          "Returns a float in (-1.0, 1.0), or 0.0 for a degenerate book.");

    m.def("calculate_obi_delta", &crossflux::calculate_obi_delta,
          py::arg("obi_a"), py::arg("obi_b"),
          "Cross-venue OBI delta: obi_a - obi_b, in [-2.0, 2.0].\n\n"
          "Positive means exchange A is more bid-heavy than exchange B.\n\n"
          "Unlike src.signals.calculate_obi_delta, which raises ValueError, the\n"
          "C++ version is noexcept and checks its [-1.0, 1.0] contract with\n"
          "assert() -- so in a Release build (-DNDEBUG) an out-of-range input is\n"
          "silently accepted and returns a nonsense delta. Validate upstream.");

    m.def("obi_profile_weights",
          []() {
              const auto& p = crossflux::obi::active();
              const auto  s = p.span();
              return py::make_tuple(
                  std::string{p.name},
                  std::vector<double>{s.begin(), s.end()});
          },
          "Return (name, weights) of the C++ engine's active weight profile.\n\n"
          "Reads $CROSSFLUX_OBI_PROFILE, else the compiled-in default. Provided\n"
          "so a test can assert the C++ engine and src.obi_weights resolved the\n"
          "same profile from the same environment -- the generated header can go\n"
          "stale if src/obi_weights.py changed without a rebuild.\n\n"
          "Note the profile is resolved once and cached, so changing the\n"
          "environment variable after the first call has no effect.");

    // ── Module-level constants ────────────────────────────────────────────────
    m.attr("DEFAULT_DEPTH")          = 10;
    m.attr("DEFAULT_DELTA_THRESHOLD")= crossflux::SignalAggregator::kDefaultDeltaThreshold;
    m.attr("DEFAULT_MIN_P_EXECUTE")  = crossflux::SignalAggregator::kDefaultMinPExecute;
    m.attr("OBI_PROFILE")            = std::string{crossflux::obi::active().name};
}
