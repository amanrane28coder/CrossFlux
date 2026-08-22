"""
src/live_ingestion.py
=====================
Phase 10: Live WebSocket Ingestion Layer

Connects to Binance and Kraken real-time L2 order book streams,
constructs the `arbitrage_engine.MarketTick` objects, and feeds them
to the C++ `SignalAggregator` for evaluation.
"""

import asyncio
import csv
import json
import logging
import math
import os
import threading
import time
from typing import Dict, Optional, List

import websockets
import ssl
import certifi

# Fix SSL certificate verification issues on macOS Python 3.14
ssl_context = ssl.create_default_context(cafile=certifi.where())

import sys
import pathlib
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arbitrage_engine as ae
from src import obi_weights
from src.risk_manager import RiskManager, OrderCheckResult
from src.execution_simulator import simulate_cross_venue_fill, ExecutionReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_ingestion")

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms"
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"
ORDERS_PATH = "/tmp/live_orders_py.csv"

# Appended, never inserted or reordered: dashboard/app.py reads this file by
# column name (normalize_orders, ORDER_FILL_FLAGS) and pandas will happily hand
# it a rectangle whose values sit under the wrong headings.
#
# `adverse` is the last column and was added with the adverse-fill logging fix. A
# fill and an adverse fill are both `approved=1` with a net PnL, so without this
# column nothing in the file distinguished a win from a booked loss except the
# sign of net_pnl -- and `reason` carried the risk manager's "approved" rather
# than the execution model's label, so the two adverse labels the simulator
# produces never reached the CSV at all.
ORDERS_HEADER = ["timestamp_ms", "action", "obi_delta", "p_execute", "qty",
                 "approved", "reason", "vwap_buy", "vwap_sell",
                 "fill_qty", "gross_pnl", "net_pnl", "fees_paid",
                 "slippage_bps", "latency_ms", "levels_buy", "levels_sell",
                 "adverse"]


def _init_orders_csv() -> None:
    """Create the file, or refuse to append rows of a different shape to it.

    The plain `if not exists: write header` this replaced would append 18-column
    rows under a 17-column header left by an earlier run, which no reader errors
    on: every value lands under some name and they are the wrong names from
    `adverse` backwards. dashboard/seed_demo_feed.py carried exactly that drift
    against the C++ writer for a whole release (tests/test_demo_feed_schema.py).

    A file whose header does not match is renamed, not truncated -- unlike the
    demo seeder's /tmp scratch files, this one is the only record that a live
    session happened, and discarding an operator's order log to fix a column
    count is not a trade worth making.
    """
    expected = ",".join(ORDERS_HEADER)
    if os.path.isfile(ORDERS_PATH) and os.path.getsize(ORDERS_PATH) > 0:
        with open(ORDERS_PATH, "r", newline="") as f:
            if f.readline().strip() == expected:
                return
        backup = f"{ORDERS_PATH}.{int(time.time())}.bak"
        os.rename(ORDERS_PATH, backup)
        logger.warning(
            "%s had a different header (an older build wrote it). Moved it to %s "
            "and started a new file; the old rows are not lost, but they are no "
            "longer in the file the dashboard reads.",
            ORDERS_PATH, backup,
        )
    with open(ORDERS_PATH, "w", newline="") as f:
        csv.writer(f).writerow(ORDERS_HEADER)


def _write_order(sig, qty: float, result: OrderCheckResult,
                 vwap_buy=0.0, vwap_sell=0.0, fill_qty=0.0,
                 gross_pnl=0.0, net_pnl=0.0, fees_paid=0.0,
                 slippage_bps=0.0, latency_ms=0.0,
                 levels_buy=0, levels_sell=0,
                 adverse: bool = False, reason: Optional[str] = None) -> None:
    """Append one order row.

    `reason` overrides `result.reason` when the execution model has something more
    specific to say than the risk manager did -- which for a booked adverse fill
    it always does, since the risk check passed and the loss came later.
    """
    ts = int(time.time() * 1000)
    row = [
        ts, sig.action, round(sig.obi_delta, 6),
        round(sig.p_execute, 6), qty,
        int(result.approved), result.reason if reason is None else reason,
        round(vwap_buy, 2), round(vwap_sell, 2),
        round(fill_qty, 6), round(gross_pnl, 4), round(net_pnl, 4),
        round(fees_paid, 4), round(slippage_bps, 2),
        round(latency_ms, 1), levels_buy, levels_sell,
        int(adverse),
    ]
    if len(row) != len(ORDERS_HEADER):
        raise ValueError(
            f"order row has {len(row)} fields, header has {len(ORDERS_HEADER)}"
        )
    with open(ORDERS_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)

class LiveArbitrageNode:
    def __init__(self):
        self.aggregator = ae.SignalAggregator(
            "binance", "kraken",
            latency_mu=3.5, latency_sigma=0.4, alpha_lifetime_ms=50.0,
            delta_threshold=0.3, min_p_execute=0.80
        )
        self.risk_mgr = RiskManager(
            max_trade_qty=0.01,
            cooldown_us=5_000_000,      # 5s
            max_drawdown_pct=10.0,
        )
        self.latest_snap_a: Optional[ae.OrderBookSnapshot] = None
        self.latest_snap_b: Optional[ae.OrderBookSnapshot] = None
        self.total_orders: int = 0
        self.filled_orders: int = 0
        self.realized_pnl: float = 0.0
        self.current_position: float = 0.0
        _init_orders_csv()

    def process_tick(self):
        if self.latest_snap_a is None or self.latest_snap_b is None:
            return
            
        ts = int(time.time() * 1000)

        try:
            # Write crypto signals to Python CSVs for dashboard
            _write_crypto_signal("BTC/USD", self.latest_snap_a, ts)

            tick = ae.make_market_tick(ts, self.latest_snap_a, self.latest_snap_b)
            signals = self.aggregator.evaluate([tick])
            p_execute = signals[0].p_execute if signals else 0.0
            _write_status(self, ts, p_execute)
        except Exception as e:
            logger.error(f"process_tick error: {e}")
            return
        for sig in signals:
            try:
                _write_signal(sig, ts)
            except Exception as e:
                logger.error(f"Failed to write signal: {e}")
        
        for sig in signals:
            logger.info(f"🚨 SIGNAL: {sig.action} | ΔOBI={sig.obi_delta:.4f} | p_exec={sig.p_execute:.4f}")
            self.total_orders += 1
            result = self.risk_mgr.check_order(0.01)
            if result.approved:
                snap_a = self.latest_snap_a
                snap_b = self.latest_snap_b
                report = simulate_cross_venue_fill(
                    snap_a.bids, snap_a.asks,
                    snap_b.bids, snap_b.asks,
                    sig.action, 0.01,
                )
                if report.rejected:
                    _write_order(sig, 0.01, OrderCheckResult(False, report.reject_reason, "ExecutionSimulator"),
                                 slippage_bps=report.total_slippage_bps,
                                 latency_ms=report.latency_ms)
                    logger.warning(f"⏭️ {report.reject_reason}: latency={report.latency_ms:.0f}ms")
                    # Nothing booked and no guard armed: see the comment below.
                else:
                    bf, sf = report.buy_fill, report.sell_fill
                    _write_order(sig, 0.01, result,
                                 vwap_buy=bf.vwap, vwap_sell=sf.vwap,
                                 fill_qty=bf.filled_qty,
                                 gross_pnl=report.gross_pnl, net_pnl=report.net_pnl,
                                 fees_paid=report.total_fees,
                                 slippage_bps=report.total_slippage_bps,
                                 latency_ms=report.latency_ms,
                                 levels_buy=bf.levels_consumed,
                                 levels_sell=sf.levels_consumed,
                                 adverse=report.adverse,
                                 reason=report.reject_reason or None)
                    # An adverse fill is a fill: it books, it counts, it arms the
                    # guards. What it must not do is read like a win. A green tick
                    # over a negative net is how a loss-booking execution model
                    # ends up looking like a profitable one in the terminal, which
                    # is the same misreading TEST_REPORT 2.1 documents for the
                    # ledger -- see the `adverse` branch of the log line.
                    if report.adverse:
                        logger.warning(
                            f"🔻 ADVERSE FILL ({report.reject_reason}): "
                            f"{bf.filled_qty:.4f} @ ${bf.vwap:.2f} → ${sf.vwap:.2f} | "
                            f"gross=${report.gross_pnl:.2f} net=${report.net_pnl:.2f} "
                            f"fees=${report.total_fees:.4f} slip={report.total_slippage_bps:.1f}bps "
                            f"lat={report.latency_ms:.0f}ms")
                    else:
                        logger.info(
                            f"✅ FILLED: {bf.filled_qty:.4f} @ ${bf.vwap:.2f} → ${sf.vwap:.2f} | "
                            f"gross=${report.gross_pnl:.2f} net=${report.net_pnl:.2f} "
                            f"fees=${report.total_fees:.4f} slip={report.total_slippage_bps:.1f}bps "
                            f"lat={report.latency_ms:.0f}ms")
                    # Inside the else, and that placement is the fix. These three
                    # lines used to sit at the `if result.approved` level, so every
                    # rejection booked itself as a fill.
                    #
                    # record_trade is the half that changed behaviour: it arms the
                    # 5 s MicrosecondCooldownGuard that check_order consults above,
                    # so a rejected order blocked the next real opportunity for
                    # five seconds. The most common rejection is the signal-time
                    # spread gate, which fires on most ticks, so this suppressed
                    # trading rather than merely mis-reporting it.
                    #
                    # filled_orders was the reporting half, and it was latent
                    # rather than visible: it reaches the dashboard only through
                    # live_status.csv, which app.py parses into filled_orders_stat
                    # and currently renders nowhere (the fill count on screen is
                    # counted from this file's own rows, where a rejection is
                    # approved=0 and was already excluded). Latent is not benign --
                    # the first consumer of that column would have inherited a fill
                    # rate near 100%.
                    #
                    # realized_pnl was harmless *by luck*: all three rejection
                    # paths in simulate_cross_venue_fill happen to carry
                    # net_pnl = 0.0. Depending on that is depending on another
                    # module's invariant to keep this one's books straight.
                    self.risk_mgr.record_trade(report.net_pnl)
                    self.filled_orders += 1
                    self.realized_pnl += report.net_pnl
            else:
                _write_order(sig, 0.01, result)
                logger.debug(f"⏭️ ORDER REJECTED: {result.guard} — {result.reason}")

async def binance_worker(node: LiveArbitrageNode):
    async for websocket in websockets.connect(BINANCE_WS_URL, ssl=ssl_context):
        logger.info("Connected to Binance WebSocket.")
        try:
            async for message in websocket:
                data = json.loads(message)
                # Binance depth payload structure
                bids = [ae.PriceLevel(float(price), float(qty)) for price, qty in data.get("bids", [])[:10]]
                asks = [ae.PriceLevel(float(price), float(qty)) for price, qty in data.get("asks", [])[:10]]
                
                if not bids or not asks:
                    continue
                    
                ts = data.get("E", int(time.time() * 1000))
                snap = ae.make_order_book_snapshot(ts, "binance", bids, asks)
                node.latest_snap_a = snap
                node.process_tick()
                
        except websockets.ConnectionClosed:
            logger.warning("Binance connection closed, reconnecting...")
        except Exception as e:
            logger.error(f"Binance stream error: {e}")

async def kraken_worker(node: LiveArbitrageNode):
    payload = {
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": ["BTC/USD"],
            "depth": 10
        }
    }
    
    async for websocket in websockets.connect(KRAKEN_WS_URL, ssl=ssl_context):
        logger.info("Connected to Kraken WebSocket.")
        await websocket.send(json.dumps(payload))
        
        try:
            async for message in websocket:
                data = json.loads(message)
                if isinstance(data, list):
                    # Kraken V1 legacy format
                    logger.debug(f"Kraken V1 data (unsupported): {str(data)[:200]}")
                    continue
                chan = data.get("channel", "?")
                dtype = data.get("type", "?")
                if chan not in ("book",) or dtype not in ("snapshot", "update"):
                    logger.debug(f"Kraken skipping chan={chan} type={dtype}")
                    continue
                if isinstance(data, dict) and data.get("channel") == "book" and data.get("type") == "snapshot":
                    # Kraken snapshot
                    book_data = data.get("data", [{}])[0]
                    bids = [ae.PriceLevel(float(lvl["price"]), float(lvl["qty"])) for lvl in book_data.get("bids", [])[:10]]
                    asks = [ae.PriceLevel(float(lvl["price"]), float(lvl["qty"])) for lvl in book_data.get("asks", [])[:10]]
                    
                    if not bids or not asks:
                        continue
                        
                    ts = int(time.time() * 1000)
                    snap = ae.make_order_book_snapshot(ts, "kraken", bids, asks)
                    node.latest_snap_b = snap
                    node.process_tick()
                
                elif isinstance(data, dict) and data.get("channel") == "book" and data.get("type") == "update":
                    # Kraken update. For simplicity in V1 blueprint, we will request full depth/snapshot 
                    # from kraken if possible or reconstruct book. 
                    # But the v2 websocket sends updates. We must reconstruct.
                    # Since this is a blueprint mock stream, we just warn.
                    logger.debug("Received Kraken book update (need to reconstruct book in V2).")
                    
        except websockets.ConnectionClosed:
            logger.warning("Kraken connection closed, reconnecting...")
        except Exception as e:
            logger.error(f"Kraken stream error: {e}")

SIGNALS_PY_PATH = "/tmp/live_signals_py.csv"
SIGNALS_EVAL_PATH = "/tmp/live_signals.csv"
STATUS_PATH = "/tmp/live_status.csv"

SIGNALS_PY_HEADER = ["timestamp_ms", "symbol", "asset_class",
                     "bid_price", "bid_qty", "ask_price", "ask_qty"]

# Must stay byte-identical to kSignalCsvHeader in
# cpp_engine/src/ingestion_engine.cpp. Both processes append to the *same*
# /tmp/live_signals.csv, so a mismatch here does not produce two schemas -- it
# produces one file with interleaved column counts, which pandas either rejects
# or silently misaligns. If you change one, change both.
SIGNALS_EVAL_HEADER = ["timestamp_ms", "obi_delta", "weighted_obi_delta",
                       "p_execute", "action", "exchange_a", "exchange_b",
                       "obi_profile"]


def _engine_obi_profile() -> str:
    """Name of the weight profile the *engine* is using.

    Read from the compiled extension rather than from src.obi_weights, even
    though both resolve $CROSSFLUX_OBI_PROFILE and normally agree. The signals
    written here are computed in C++, so the C++ answer is the true one. A stale
    .so built before profiles existed would otherwise be labelled with whatever
    the Python registry happens to select -- a CSV claiming decay_50 rows that
    were actually produced by unweighted code. "unknown" is the honest label for
    that case.

    Disagreement between the two is worth one loud warning: it means the .so on
    the path was not built from the current obi_config.hpp, so nothing else in
    this session can be trusted to mean what it says.
    """
    name = getattr(ae, "OBI_PROFILE", None)
    if name is None:
        if not _engine_obi_profile._warned:
            _engine_obi_profile._warned = True
            logger.warning(
                "arbitrage_engine has no OBI_PROFILE attribute -- the extension "
                "predates weighted OBI. Rebuild it, or the weighted_obi_delta "
                "column will stay empty. Labelling rows 'unknown'."
            )
        return "unknown"
    expected = obi_weights.active().name
    if name != expected and not _engine_obi_profile._warned:
        _engine_obi_profile._warned = True
        logger.warning(
            "Profile mismatch: the C++ engine reports '%s' but src.obi_weights "
            "resolves '%s'. The loaded .so was built from a different "
            "obi_config.hpp. Rebuild before trusting these signals.",
            name, expected,
        )
    return name


_engine_obi_profile._warned = False

def _init_signals_py_csv():
    exists = os.path.isfile(SIGNALS_PY_PATH)
    if exists and os.path.getsize(SIGNALS_PY_PATH) > 0:
        with open(SIGNALS_PY_PATH, "r", newline="") as f:
            first_line = f.readline().strip()
        if first_line == ",".join(SIGNALS_PY_HEADER):
            return
    with open(SIGNALS_PY_PATH, "w", newline="") as f:
        csv.writer(f).writerow(SIGNALS_PY_HEADER)

def _init_status_csv():
    exists = os.path.isfile(STATUS_PATH)
    with open(STATUS_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp_ms", "status", "p_execute", "total_orders", "filled_orders", "realized_pnl", "position"])

def _write_status(node: LiveArbitrageNode, ts: int, p_execute: float) -> None:
    with open(STATUS_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, "running", round(p_execute, 6),
            node.total_orders, node.filled_orders,
            round(node.realized_pnl, 4), round(node.current_position, 4),
        ])

def _init_signals_eval_csv():
    # Rotate a file left behind by an older build instead of appending to it.
    # The header is only written to a new/empty file, so appending 8-column rows
    # to a 6-column file would leave one CSV with two schemas and no marker
    # saying where the change happened. Mirrors the same guard in
    # cpp_engine/src/ingestion_engine.cpp, which writes this same path.
    if os.path.isfile(SIGNALS_EVAL_PATH) and os.path.getsize(SIGNALS_EVAL_PATH) > 0:
        with open(SIGNALS_EVAL_PATH, "r", newline="") as f:
            first_line = f.readline().strip()
        if first_line == ",".join(SIGNALS_EVAL_HEADER):
            return
        backup = SIGNALS_EVAL_PATH + ".stale"
        os.replace(SIGNALS_EVAL_PATH, backup)
        logger.warning(
            "%s had an outdated header; moved to %s and starting a fresh file.",
            SIGNALS_EVAL_PATH, backup,
        )
    with open(SIGNALS_EVAL_PATH, "w", newline="") as f:
        csv.writer(f).writerow(SIGNALS_EVAL_HEADER)

def _write_signal(sig, ts: int) -> None:
    action_val = int(sig.action_enum)

    # getattr, not sig.weighted_obi_delta: the compiled extension in the repo
    # may predate this field. A stale .so should degrade to an empty column,
    # not take down the live node with an AttributeError mid-session.
    wdelta = getattr(sig, "weighted_obi_delta", float("nan"))
    # Empty field, not the string "nan" -- pandas reads "" as float NaN and
    # keeps the column numeric, where "nan" would force dtype=object and break
    # the dashboard's arithmetic on it.
    wdelta_field = "" if wdelta is None or math.isnan(wdelta) else round(wdelta, 6)

    with open(SIGNALS_EVAL_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, round(sig.obi_delta, 6), wdelta_field, round(sig.p_execute, 6),
            action_val, "binance", "kraken", _engine_obi_profile(),
        ])

def _write_crypto_signal(symbol: str, snap, ts: int) -> None:
    if not snap or not snap.bids or not snap.asks:
        return
    bid = snap.bids[0].price
    ask = snap.asks[0].price
    bid_qty = snap.bids[0].volume
    ask_qty = snap.asks[0].volume
    with open(SIGNALS_PY_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, symbol, "crypto",
            round(bid, 2), round(bid_qty, 6),
            round(ask, 2), round(ask_qty, 6),
        ])

def _write_equity_quote(symbol: str, bid: float, ask: float,
                        bid_qty: float = 100.0, ask_qty: float = 100.0) -> None:
    ts = int(time.time() * 1000)
    with open(SIGNALS_PY_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, symbol, "equity",
            round(bid, 4), round(bid_qty, 2),
            round(ask, 4), round(ask_qty, 2),
        ])

def _poll_yfinance(symbols: List[str]):
    import yfinance as yf
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.info
            bid = info.get("bid")
            ask = info.get("ask")
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            if bid and ask:
                _write_equity_quote(sym, float(bid), float(ask))
            elif price:
                spread = float(price) * 0.0005
                _write_equity_quote(sym, float(price) - spread, float(price) + spread)
            else:
                logger.debug(f"yfinance: no quote data for {sym}")
        except Exception as e:
            logger.debug(f"yfinance poll error for {sym}: {e}")

async def yfinance_worker(symbols: Optional[List[str]] = None):
    if symbols is None:
        symbols = ["AAPL", "SPY"]
    _init_signals_py_csv()
    logger.info(f"yFinance equity worker started for {symbols}")
    while True:
        await asyncio.to_thread(_poll_yfinance, symbols)
        await asyncio.sleep(2)

async def main():
    logger.info("Starting Live Ingestion Engine...")
    _init_signals_py_csv()
    _init_signals_eval_csv()
    _init_status_csv()
    node = LiveArbitrageNode()
    
    await asyncio.gather(
        binance_worker(node),
        kraken_worker(node),
        yfinance_worker(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Live engine halted by user.")
