#!/usr/bin/env python3
"""
dashboard/seed_demo_feed.py
===========================
Populate the /tmp CSVs that dashboard/app.py reads, so the dashboard has
something to render when the C++ engine and the exchange WebSockets are not
available.

Why this exists
---------------
The dashboard is a view over seven files in /tmp written by the live engine. With
no engine running, every panel reads "--" and a shared link shows an empty shell.

Why it keeps writing instead of writing once
--------------------------------------------
app.py treats a feed as live only while its last heartbeat is under
STALE_AFTER_S = 10 seconds old. A one-shot backfill therefore renders as a dead
feed ten seconds after it is written. `--follow` keeps appending, which is the
only way a demo link still looks alive when someone opens it an hour later.

Honesty
-------
Every row is synthetic. The order book is a random walk, not recorded market
data, and the PnL is arithmetic on that walk. The status column is written as
"demo" rather than "running" so the dashboard's own header can distinguish this
from a real engine, and the seeded signals are labelled with the profile name
suffixed "-demo" so a CSV left in /tmp can never be mistaken for a real capture.

What the demo's PnL is, specifically: a modelled +4.79 bps cross-venue basis,
minus fees, minus a latency haircut. It is not the OBI signal earning anything.
The signal here picks the direction and is uncorrelated with the basis, so it
rejects about as often as it points the right way -- which is the same shape as
the real finding in TEST_REPORT.md 2.3/2.4, arrived at by construction rather
than by measurement. Fills that come back negative are booked and flagged
`adverse`, because the engine books them.

Usage
-----
    python3 dashboard/seed_demo_feed.py             # backfill ~3 min, exit
    python3 dashboard/seed_demo_feed.py --follow    # backfill, then stream
    python3 dashboard/seed_demo_feed.py --clear     # truncate to headers, exit
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import random
import signal
import sys
import time

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Schemas ──────────────────────────────────────────────────────────────────
# Each of these must match the producer it stands in for. The signals header is
# the one shared with C++ (cpp_engine/src/ingestion_engine.cpp) and Python
# (src/live_ingestion.py); it is imported rather than retyped so a change there
# cannot silently desynchronise the demo feed.
try:
    from src.live_ingestion import SIGNALS_EVAL_HEADER as _SIGNALS_HEADER
except Exception:                                   # websockets/certifi absent
    _SIGNALS_HEADER = ["timestamp_ms", "obi_delta", "weighted_obi_delta",
                       "p_execute", "action", "exchange_a", "exchange_b",
                       "obi_profile"]

TMP = pathlib.Path("/tmp")

SCHEMAS: dict[str, list[str]] = {
    "live_prices.csv": ["timestamp_ms", "exchange", "bid_price", "bid_vol",
                        "ask_price", "ask_vol"],
    "live_depth.csv": ["timestamp_ms", "exchange", "side", "level", "price",
                       "volume"],
    "live_signals.csv": list(_SIGNALS_HEADER),
    "live_signals_py.csv": ["timestamp_ms", "symbol", "asset_class",
                            "bid_price", "bid_qty", "ask_price", "ask_qty"],
    "live_status.csv": ["timestamp_ms", "status", "p_execute", "total_orders",
                        "filled_orders", "realized_pnl", "position"],
    # 24 columns, in the order SimulatedExecutor::log_order writes them
    # (cpp_engine/src/execution_manager.cpp). The last eight were appended to the
    # C++ writer by the friction work and never added here, so this file was
    # eight columns short of the producer it stands in for. The damage was not a
    # blank panel -- app.py does not read the eight yet -- it was ensure_headers
    # below: an engine-written file failed the header comparison, was judged
    # stale, and was truncated with the engine's orders in it.
    # tests/test_demo_feed_schema.py now compares this list against the header
    # literal in the .cpp and fails if either side moves alone.
    "live_orders.csv": ["timestamp_ms", "signal_timestamp_ms", "obi_delta",
                        "buy_exchange", "sell_exchange", "fill_price_buy",
                        "fill_price_sell", "fill_qty", "spread_pct",
                        "gross_pnl", "net_pnl", "fees_paid", "slippage_bps",
                        "latency_ms", "filled", "reject_reason",
                        "filled_qty_buy", "filled_qty_sell", "residual_qty",
                        "legging_cost", "signal_edge_bps", "realized_edge_bps",
                        "leg_gap_ms", "adverse"],
}

# ── Model parameters ─────────────────────────────────────────────────────────
TICK_MS = 500                    # matches the C++ price-log throttle
DELTA_THRESHOLD = 0.3            # SignalAggregator::kDefaultDeltaThreshold
P_EXECUTE = 0.9312               # representative lognorm CDF result
MID_START = 61_450.0
VOL_BASE = 2.4
QTY = 0.01

# The cross-venue basis, in bps, and the pull that holds it there.
#
# This was a flat +$4.00 offset between the two mids (0.65 bps) on top of two
# independently diffusing random walks, which had two consequences worth naming.
# 0.65 bps never covers a 3 bps round trip, so the demo filled 0 of 47 orders and
# the whole orders panel was rejection rows. And the offset was only a starting
# condition: with each mid taking its own N(0, 1.2) step, the gap between them is
# itself a random walk that wanders ~$32 over a 3-minute backfill, so whatever
# basis the demo began with was gone by the end of it.
#
# 4.79 bps is the USDT/USD quote-currency basis measured from the raw feeds in
# TEST_REPORT.md 2.3. A quote-currency basis is a persistent level, not a walk,
# so the gap is pulled back to it rather than left to diffuse.
BASIS_BPS = 4.79
BASIS_PULL = 0.25                # fraction of the gap closed per tick
MIN_EDGE_BPS = 0.5               # src.execution_simulator.MIN_PROFIT_NET_BPS

# Book-imbalance bias. In the measured data 99.7% of signals point one way (buy
# binance/USDT, sell kraken/USD) -- that one-sidedness is what a basis looks like
# through an OBI signal. With both venues' skew mean-reverting to zero the demo
# split its direction ~50/50, so half its signals pointed against the basis and
# could not clear any gate. Biasing the two means apart reproduces the character
# of that asymmetry. It does not reproduce the 99.7%, and is not offered as a
# measurement of it: it is the smallest change that lets the demo trade the basis
# it models.
SKEW_BIAS = 0.30

# Adverse-selection haircut: the fraction the touch moves against each leg per
# second of delay. Same coefficient as src/execution_simulator.simulate_latency
# and the C++ synchronous path (`latency_ms / 1000 * 0.01`), used here so the
# demo's arithmetic is the engine's arithmetic. It is a haircut conditional on
# being picked off, not a volatility estimate -- an order of magnitude above BTC's
# unconditional per-second move -- and both the C++ comment and TEST_REPORT.md 5
# carry it as an assumption rather than a measurement.
DRIFT_PER_SECOND = 0.01

# Round-trip cost in basis points. Taken from the fee registry so the demo's
# PnL is computed against the same numbers as the backtests rather than an
# invented figure.
def _round_trip_bps() -> float:
    """Round-trip cost for a binance->kraken pair under the active fee preset.

    Raises nothing: a demo feed must start even with a broken import. But the
    fallback is announced, because a silent fallback to a hardcoded 3.0 would
    make the demo's PnL disagree with backtest/fee_sensitivity_report.html while
    looking authoritative.
    """
    try:
        from src import fees
        return float(fees.active().round_trip_bps("binance", "kraken"))
    except Exception as exc:                                # noqa: BLE001
        print(f"[seed_demo_feed] could not read src.fees ({exc.__class__.__name__}"
              f": {exc}); falling back to 3.0 bps round-trip", file=sys.stderr)
        return 3.0


ROUND_TRIP_BPS = _round_trip_bps()


# The demo's own default, deliberately not src.friction.DEFAULT_PRESET.
#
# That default is "stress" (100 ms), chosen so no return is ever reported from an
# optimistic model. 100 ms of the haircut above is 10 bps against each leg, which
# buries a 4.79 bps basis twenty times over: every fill would be adverse and the
# dashboard would be uniformly red for a reason that has nothing to do with the
# dashboard. A demo of the live engine stands in for the engine on its co-located
# box, so it defaults to "colocated" (5 ms) and says so. $CROSSFLUX_FRICTION_PRESET
# overrides it, and the backfill prints the adverse-fill rate it produced, so the
# cost of the choice is visible rather than assumed.
DEMO_FRICTION_PRESET = "colocated"


def _latency_params() -> tuple[str, float, float]:
    """``(preset name, median latency ms, jitter sigma)`` for the demo's fills.

    Only the parameters are read; the draw itself is reproduced below against the
    demo's own seeded ``random.Random`` rather than calling
    ``FrictionModel.sample_latency``, which returns a numpy array. This file is
    stdlib-only by design -- it has to be able to paint a dashboard on a machine
    where the scientific stack is not installed -- and a demo whose numbers move
    between runs is harder to describe to whoever is looking at the link.
    """
    try:
        from src import friction
        name = os.environ.get(friction.ENV_VAR, DEMO_FRICTION_PRESET)
        m = friction.get_preset(name)
        return m.name, float(m.latency_ms), float(m.jitter_log_sigma)
    except Exception as exc:                                # noqa: BLE001
        print(f"[seed_demo_feed] could not read src.friction ({exc.__class__.__name__}"
              f": {exc}); falling back to 5 ms fixed latency", file=sys.stderr)
        return "colocated-fallback", 5.0, 0.0


LATENCY_PRESET, LATENCY_MS, LATENCY_SIGMA = _latency_params()


def _profile_label() -> str:
    """Active weight-profile name, suffixed so demo rows are self-identifying."""
    try:
        from src import obi_weights
        return f"{obi_weights.active().name}-demo"
    except Exception:
        return "flat-demo"


PROFILE_LABEL = _profile_label()
IS_WEIGHTED = not PROFILE_LABEL.startswith("flat")


# ── File helpers ─────────────────────────────────────────────────────────────

def _write_header(name: str) -> None:
    path = TMP / name
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(SCHEMAS[name])


def clear_all() -> None:
    """Truncate every demo-owned file back to a bare header row."""
    for name in SCHEMAS:
        _write_header(name)


def ensure_headers() -> None:
    """Create or reset each file whose first line is not the expected header.

    A file left behind by a previous build gets replaced rather than appended
    to. Appending a row with a different column count produces a single CSV with
    two schemas, which pandas either rejects outright or, worse, reads with every
    field shifted by one.
    """
    for name, header in SCHEMAS.items():
        path = TMP / name
        expected = ",".join(header)
        if path.exists() and path.stat().st_size > 0:
            with path.open() as f:
                if f.readline().strip() == expected:
                    continue
        _write_header(name)


def _append(name: str, rows: list[list]) -> None:
    """Append rows, refusing any row that is not the width of its own header.

    The eight-column gap this file carried against the C++ writer was invisible
    precisely because nothing checked. A positional writer and a named-column
    reader can disagree indefinitely without either one erroring: pandas is handed
    a rectangle, every value lands under some name, and the names are simply the
    wrong ones from the first missing field onward. Failing loudly here costs one
    comparison per row and turns that class of drift into a crash on the first
    tick instead of a mislabelled chart.
    """
    if not rows:
        return
    width = len(SCHEMAS[name])
    for row in rows:
        if len(row) != width:
            raise ValueError(
                f"{name}: row has {len(row)} fields, header has {width} "
                f"({', '.join(SCHEMAS[name])})"
            )
    with (TMP / name).open("a", newline="") as f:
        csv.writer(f).writerows(rows)


# Everything a live_orders row carries when nothing happened: a rejection has no
# fill, no cost and no edge. Only the seven fields that exist for every order --
# the timestamps, the signal, the venues and the two prices -- are required.
_ORDER_DEFAULTS: dict[str, object] = {
    "fill_qty": 0.0, "spread_pct": 0.0, "gross_pnl": 0.0, "net_pnl": 0.0,
    "fees_paid": 0.0, "slippage_bps": 0.0, "latency_ms": 0.0,
    "filled": 0, "reject_reason": "",
    "filled_qty_buy": 0.0, "filled_qty_sell": 0.0, "residual_qty": 0.0,
    "legging_cost": 0.0, "signal_edge_bps": 0.0, "realized_edge_bps": 0.0,
    "leg_gap_ms": 0.0, "adverse": 0,
}


def _order_row(**fields) -> list:
    """One live_orders row, ordered by the schema instead of by hand.

    The two row builders here used to be hand-written positional lists, which is
    how this file came to be eight columns behind the C++ writer: appending to the
    header is one edit, appending to two literal lists is two more, and nothing
    connected them. Projecting a dict through ``SCHEMAS`` makes the header the
    single ordering authority -- a column added there and left unpopulated raises
    on the next tick, and a column inserted rather than appended reorders both
    rows automatically instead of shifting every field after it.
    """
    cols = SCHEMAS["live_orders.csv"]
    row = {**_ORDER_DEFAULTS, **fields}
    missing = [c for c in cols if c not in row]
    unknown = [k for k in row if k not in cols]
    if missing or unknown:
        raise ValueError(
            f"live_orders row: missing={missing} unknown={unknown}"
        )
    return [row[c] for c in cols]


# ── Synthetic market ─────────────────────────────────────────────────────────

class DemoMarket:
    """A two-venue walk with a mean-reverting basis and a persistent book skew.

    The skew is what makes the demo interesting: a pure coin-flip imbalance
    almost never clears |Δ| > 0.3 on both venues at once, so the signals file
    would stay nearly empty. An OU-style process on each venue's imbalance
    produces stretches of one-sided pressure, which is what the strategy is
    looking for and what the charts need in order to show anything.

    Two levels, two behaviours. The *price* is one walk shared by both venues,
    because they quote the same asset. The *basis* between them is a separate,
    mean-reverting series, because a quote-currency dislocation is a level that
    persists rather than a difference that diffuses. Modelling it as two
    independent walks -- which is what this did -- makes the two venues drift
    arbitrarily far apart given enough ticks, and produces a demo whose edge
    depends on how long ago it was started.
    """

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        # A separate stream for the latency draw. Sharing self.rng would let a
        # jittered friction preset consume draws the price walk was going to use,
        # so switching preset would silently produce a *different market* -- and
        # the four-preset comparison in TEST_REPORT.md 5 would be four markets
        # rather than one market under four costs. Only "retail" has jitter today,
        # which is exactly why this was easy to miss.
        self.lat_rng = random.Random(seed ^ 0x5EED)
        self.mid_a = MID_START
        self.mid_b = MID_START * (1.0 + BASIS_BPS / 10_000.0)
        self.skew_a = -SKEW_BIAS
        self.skew_b = SKEW_BIAS
        self.total_orders = 0
        self.filled_orders = 0
        self.adverse_fills = 0
        self.realized_pnl = 0.0
        self.position = 0.0

    def _ou(self, x: float, mean: float) -> float:
        """One mean-reverting step, clamped to the range an OBI can occupy."""
        return max(-0.92, min(0.92,
                              mean + 0.86 * (x - mean) + self.rng.gauss(0.0, 0.22)))

    def step(self) -> None:
        # Level: one walk, shared, so the pair never decouples.
        self.mid_a = max(1.0, self.mid_a + self.rng.gauss(0.0, 3.0))
        # Basis: pulled back toward BASIS_BPS of the level, with noise around it.
        target = self.mid_a * BASIS_BPS / 10_000.0
        gap = self.mid_b - self.mid_a
        gap += BASIS_PULL * (target - gap) + self.rng.gauss(0.0, 1.2)
        self.mid_b = max(1.0, self.mid_a + gap)
        # Book skew: mean-reverting to opposite biases, so most signals point the
        # way the measured ones do instead of splitting evenly across the basis.
        self.skew_a = self._ou(self.skew_a, -SKEW_BIAS)
        self.skew_b = self._ou(self.skew_b, SKEW_BIAS)

    def book(self, which: str) -> tuple[float, float, list[float], list[float]]:
        """Return (mid, half_spread, bid_volumes, ask_volumes) for one venue."""
        mid = self.mid_a if which == "binance" else self.mid_b
        skew = self.skew_a if which == "binance" else self.skew_b
        half = 0.5 + abs(self.rng.gauss(0.0, 0.35))
        # Level volumes decay with depth; the skew tilts bid against ask.
        bids, asks = [], []
        for lvl in range(5):
            decay = math.exp(-0.45 * lvl)
            base = VOL_BASE * decay
            bids.append(round(base * (1.0 + skew) * (0.85 + 0.3 * self.rng.random()), 6))
            asks.append(round(base * (1.0 - skew) * (0.85 + 0.3 * self.rng.random()), 6))
        return mid, half, bids, asks

    @staticmethod
    def obi(bids: list[float], asks: list[float], weights=None) -> float:
        num = den = 0.0
        for i, (b, a) in enumerate(zip(bids, asks)):
            w = 1.0 if weights is None else weights[i]
            num += w * (b - a)
            den += w * (b + a)
        return 0.0 if den == 0.0 else num / den


DECAY_50 = [1.0, 0.5, 0.25, 0.125, 0.0625]


def emit(market: DemoMarket, ts_ms: int) -> None:
    """Write one tick's worth of rows across every file.

    The status heartbeat is written on *every* tick, not just on ticks that
    produce a trade. app.py derives liveness from this file's timestamp, so
    writing it only on fills makes a quiet stretch of the book -- which is most
    of them -- render as a tripped circuit breaker and a dead PnL panel while
    prices are visibly still arriving.
    """
    _tick(market, ts_ms)
    _write_status(market, ts_ms)


def _write_status(market: DemoMarket, ts_ms: int) -> None:
    # Rewritten, not appended: live_trading.cpp truncates this file, and two
    # producers appending to it is an existing race in this repo.
    with (TMP / "live_status.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SCHEMAS["live_status.csv"])
        w.writerow([ts_ms, "demo", P_EXECUTE, market.total_orders,
                    market.filled_orders, round(market.realized_pnl, 4),
                    market.position])


def _tick(market: DemoMarket, ts_ms: int) -> None:
    """Advance the market and write market-data, signal and order rows."""
    market.step()

    price_rows, depth_rows, py_rows = [], [], []
    books = {}
    for ex in ("binance", "kraken"):
        mid, half, bids, asks = market.book(ex)
        books[ex] = (mid, half, bids, asks)
        bid_px, ask_px = round(mid - half, 2), round(mid + half, 2)
        price_rows.append([ts_ms, ex, bid_px, bids[0], ask_px, asks[0]])
        for lvl in range(5):
            depth_rows.append([ts_ms, ex, 0, lvl, round(bid_px - lvl * 0.5, 2), bids[lvl]])
            depth_rows.append([ts_ms, ex, 1, lvl, round(ask_px + lvl * 0.5, 2), asks[lvl]])
        if ex == "binance":
            py_rows.append([ts_ms, "BTC/USD", "crypto", bid_px, bids[0], ask_px, asks[0]])

    _append("live_prices.csv", price_rows)
    _append("live_depth.csv", depth_rows)
    _append("live_signals_py.csv", py_rows)

    # ── Signal gate, mirroring SignalAggregator::evaluate ────────────────────
    _, _, ba, aa = books["binance"]
    _, _, bb, ab = books["kraken"]
    delta = market.obi(ba, aa) - market.obi(bb, ab)
    wdelta = (market.obi(ba, aa, DECAY_50) - market.obi(bb, ab, DECAY_50)
              if IS_WEIGHTED else None)
    gate = wdelta if IS_WEIGHTED else delta

    if abs(gate) <= DELTA_THRESHOLD:
        return

    action = 1 if gate > 0 else 0
    _append("live_signals.csv", [[
        ts_ms, round(delta, 6),
        "" if wdelta is None else round(wdelta, 6),
        P_EXECUTE, action, "binance", "kraken", PROFILE_LABEL,
    ]])

    # ── Execution, mirroring the spread-friction gate ─────────────────────────
    mid_a, half_a, _, _ = books["binance"]
    mid_b, half_b, _, _ = books["kraken"]
    if action == 1:                       # buy kraken, sell binance
        buy_px, sell_px = mid_b + half_b, mid_a - half_a
        buy_ex, sell_ex = "kraken", "binance"
    else:
        buy_px, sell_px = mid_a + half_a, mid_b - half_b
        buy_ex, sell_ex = "binance", "kraken"

    market.total_orders += 1
    signal_edge_bps = (sell_px - buy_px) / buy_px * 10_000.0
    row = dict(timestamp_ms=ts_ms, signal_timestamp_ms=ts_ms,
               obi_delta=round(delta, 6), buy_exchange=buy_ex,
               sell_exchange=sell_ex, fill_price_buy=round(buy_px, 2),
               fill_price_sell=round(sell_px, 2),
               signal_edge_bps=round(signal_edge_bps, 4))

    # The entry gate, and the only gate. It is checked at signal time on the
    # prices the signal was formed from, which makes it a forecast the fill is
    # then allowed to falsify. Mirrors the `raw_spread_bps < min_profit_bps_`
    # check in SimulatedExecutor::evaluate_and_execute.
    #
    # Most rejections here are direction, not size: the OBI signal picks the
    # venue pair, and when it points against the basis the edge is negative
    # before any cost is applied. That is the §2.4 finding in miniature.
    if signal_edge_bps - ROUND_TRIP_BPS < MIN_EDGE_BPS:
        _append("live_orders.csv", [_order_row(
            **row, filled=0, reject_reason="spread_below_min")])
        return

    # Latency, and this time it costs something. The order reaches the book
    # `latency_ms` later and both touches have moved against it. The previous
    # version drew a latency of `12 + 8 * random()` purely to fill the column and
    # then priced the fill at the signal-time touch, so the demo's latency was
    # decoration: no row's PnL would have changed if it had been zero.
    latency_ms = (LATENCY_MS if LATENCY_SIGMA == 0.0 else
                  LATENCY_MS * math.exp(LATENCY_SIGMA * market.lat_rng.gauss(0.0, 1.0)))
    drift = latency_ms / 1000.0 * DRIFT_PER_SECOND
    fill_buy = buy_px * (1.0 + drift)
    fill_sell = sell_px * (1.0 - drift)
    # Overwrite the signal-time touches with what the order actually paid. The
    # gate above quoted the former; the ledger has to carry the latter, and a row
    # showing the pre-drift prices next to a post-drift PnL would not reconcile.
    row["fill_price_buy"] = round(fill_buy, 2)
    row["fill_price_sell"] = round(fill_sell, 2)

    # No gate here, and none after. `if net_bps <= 0: return` used to sit at this
    # point, recomputing the edge once the cost was known and dropping whatever
    # came out badly -- which keeps exactly the winners, and is the tautology
    # TEST_REPORT.md 2.1 documents. Both engines removed it (execution_manager.cpp
    # and src/execution_simulator.py, "No gate here"); a demo that still declined
    # its losers would be advertising behaviour the engine no longer has.
    realized_edge_bps = (fill_sell - fill_buy) / fill_buy * 10_000.0
    notional = fill_buy * QTY
    gross = (fill_sell - fill_buy) * QTY
    # One notional at the round-trip rate. The two legs' notionals differ only by
    # the basis, so charging each at its own venue rate moves the fee by a
    # fraction of a cent on QTY = 0.01 and would not survive rounding.
    fees = notional * ROUND_TRIP_BPS / 10_000.0
    net = gross - fees

    adverse = net < 0.0
    if adverse:
        # Same two labels the C++ path writes, distinguishing the case the
        # requirement names -- an edge the latency window turned over on its own
        # -- from one that survived but could not cover the fees.
        reason = ("adverse_fill_spread_collapsed" if realized_edge_bps <= 0.0
                  else "adverse_fill")
        market.adverse_fills += 1
    else:
        reason = ""

    market.filled_orders += 1
    market.realized_pnl += net
    market.position = round(market.position + (QTY if action == 1 else -QTY), 6)

    _append("live_orders.csv", [_order_row(
        **row,
        fill_qty=QTY,
        spread_pct=round(realized_edge_bps / 10_000.0, 8),
        gross_pnl=round(gross, 4), net_pnl=round(net, 4), fees_paid=round(fees, 4),
        # Genuinely zero, not unmodelled: QTY = 0.01 BTC against level-0 volumes
        # of ~2.4 BTC never reaches the second level, so walking the five levels
        # this demo writes to live_depth.csv would return the touch price. The
        # figure this replaces was `1.5 * random()`, which made a demo that fills
        # inside the touch look like one that pays for depth.
        slippage_bps=0.0,
        latency_ms=round(latency_ms, 1),
        filled=1,
        reject_reason=reason,
        # Both legs, filled together, in full. The demo mirrors the synchronous
        # path, which has no per-leg latency and so cannot leg: residual and
        # leg_gap are zero here because this model cannot produce them, not
        # because legging is rare. The async path is where those columns get
        # their values; see friction::ResolvedOrder.
        filled_qty_buy=QTY, filled_qty_sell=QTY, residual_qty=0.0,
        legging_cost=0.0, leg_gap_ms=0.0,
        realized_edge_bps=round(realized_edge_bps, 4),
        adverse=int(adverse),
    )])


# ── Entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--follow", action="store_true",
                    help="keep appending after the backfill (Ctrl-C to stop)")
    ap.add_argument("--clear", action="store_true",
                    help="truncate every demo file to its header and exit")
    ap.add_argument("--backfill-seconds", type=int, default=180,
                    help="history to synthesise before now (default: 180)")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed")
    args = ap.parse_args(argv)

    if args.clear:
        clear_all()
        print("[seed_demo_feed] cleared demo rows from /tmp")
        return 0

    ensure_headers()
    market = DemoMarket(seed=args.seed)

    now_ms = int(time.time() * 1000)
    ticks = max(1, args.backfill_seconds * 1000 // TICK_MS)
    for i in range(ticks):
        # _tick, not emit: the backfill would otherwise rewrite the status file
        # once per historical tick to produce the one row that survives.
        _tick(market, now_ms - (ticks - i) * TICK_MS)
    _write_status(market, now_ms - TICK_MS)

    signals = max(0, sum(1 for _ in (TMP / "live_signals.csv").open()) - 1)
    filled = market.filled_orders
    # The adverse rate is printed because it is the number that says whether the
    # demo is showing the engine's current behaviour. A run that reports 0
    # adverse fills out of a hundred is either a very fast preset or a gate that
    # has crept back in, and both are worth noticing from the console line.
    adverse_pct = 100.0 * market.adverse_fills / filled if filled else 0.0
    print(f"[seed_demo_feed] backfilled {ticks} ticks / {args.backfill_seconds}s "
          f"-> {signals} signals, {filled}/{market.total_orders} "
          f"orders filled, {market.adverse_fills} adverse ({adverse_pct:.1f}%), "
          f"net ${market.realized_pnl:.2f}")
    print(f"[seed_demo_feed] profile={PROFILE_LABEL}, "
          f"round_trip={ROUND_TRIP_BPS:.2f} bps, friction={LATENCY_PRESET} "
          f"({LATENCY_MS:.0f} ms), basis={BASIS_BPS:.2f} bps")

    if not args.follow:
        print("[seed_demo_feed] not following; feeds go stale in ~10s "
              "(dashboard STALE_AFTER_S). Re-run with --follow for a live page.")
        return 0

    stop = {"now": False}

    def _stop(_sig, _frm):
        stop["now"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"[seed_demo_feed] following at {TICK_MS} ms/tick (pid {os.getpid()})")
    while not stop["now"]:
        emit(market, int(time.time() * 1000))
        time.sleep(TICK_MS / 1000.0)

    print("[seed_demo_feed] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
