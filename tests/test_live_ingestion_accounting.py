"""What the live node counts as a fill, and what it tells the operator it did.

Two defects, one root cause: ``process_tick`` treated "the risk manager approved
the order" as "the order filled". The three booking lines --
``risk_mgr.record_trade``, ``filled_orders``, ``realized_pnl`` -- sat outside the
``if report.rejected`` split, so a rejection was recorded as a trade; and the fill
branch printed one ``✅ FILLED`` line whether the net PnL came back positive or
negative.

The costly half is ``record_trade``. It arms the 5 s ``MicrosecondCooldownGuard``
that ``check_order`` consults on the next signal, so an order the execution model
declined blocked the next real opportunity for five seconds. The most common
rejection is the signal-time spread gate, which fires on most ticks -- so the
effect was to suppress trading, not merely to mis-report it.

``realized_pnl`` was harmless only by luck: all three rejection paths in
``simulate_cross_venue_fill`` happen to return ``net_pnl = 0.0``. These tests use
a rejection carrying a *non-zero* net so that they fail on the accounting rather
than on that coincidence, which is what the fix is actually about -- one module's
books should not depend on another module's choice of filler value.

The logging half matters because of what the execution model now does. It books
losses instead of declining them (TEST_REPORT.md 2.1), so a green tick over a
negative net is no longer an anomaly to be read past: it is the normal appearance
of the failure mode the whole friction rework exists to surface.

These tests drive the real ``RiskManager`` and the real ``ExecutionReport``, and
observe ``record_trade`` through its effect -- whether the next order is still
allowed through -- rather than by counting calls to a double. A mock would pass
against a version that armed the guard somewhere else.
"""
from __future__ import annotations

import csv
import logging
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_live_ingestion():
    """Import the module, standing in only for imports this sandbox lacks.

    ``src/live_ingestion.py`` pulls in ``websockets`` and the compiled
    ``arbitrage_engine``; the Linux sandbox has neither (the .so is built for
    macOS). Substituting only what is *missing* keeps the real modules in play
    wherever they exist, so this test does not quietly become a test of its own
    stubs on the machine that matters.

    Nothing under test comes from either module: ``ae`` is replaced per-test
    below, and the websocket workers are never called.
    """
    for name, attrs in (("websockets", ("connect", "ConnectionClosed")),
                        ("arbitrage_engine", ("SignalAggregator", "PriceLevel",
                                              "OrderBookSnapshot",
                                              "make_market_tick",
                                              "make_order_book_snapshot"))):
        try:
            __import__(name)
            continue
        except ImportError:
            pass
        mod = types.ModuleType(name)
        for a in attrs:
            setattr(mod, a, type(a, (Exception,), {}))
        sys.modules[name] = mod

    from src import live_ingestion
    return live_ingestion


LI = _import_live_ingestion()

from src.execution_simulator import ExecutionReport, FillResult  # noqa: E402
from src.risk_manager import RiskManager  # noqa: E402


class _Sig:
    action, obi_delta, p_execute = 0, 0.42, 0.95


class _Snap:
    """Enough of an OrderBookSnapshot for process_tick to pass it along.

    The books are never walked -- simulate_cross_venue_fill is replaced by a
    canned report -- so one level each is plenty, and the numbers are irrelevant.
    """
    def __init__(self):
        self.bids = [(100.0, 1.0)]
        self.asks = [(101.0, 1.0)]


def _fill(net: float, adverse: bool, reason: str = "") -> ExecutionReport:
    leg = FillResult(vwap=100.0, filled_qty=0.01, levels_consumed=1,
                     slippage_bps=1.5, fees_paid=0.004, partial=False)
    return ExecutionReport(leg, leg, gross_pnl=net + 0.008, net_pnl=net,
                           total_slippage_bps=3.0, total_fees=0.008,
                           rejected=False, reject_reason=reason,
                           latency_ms=30.0, adverse=adverse)


def _rejection(net: float = -7.5) -> ExecutionReport:
    """A rejection carrying a non-zero net PnL, deliberately.

    The real rejection paths all carry 0.0, which would let a version that adds
    ``report.net_pnl`` on both branches pass a realized_pnl assertion. The
    accounting must be right because the branch is right, not because the other
    module chose a convenient filler.
    """
    return ExecutionReport(None, None, 0.0, net, 0.0, 0.0, rejected=True,
                           reject_reason="net_spread_0.10bps_below_min_0.5bps",
                           latency_ms=0.0)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _run(reports: list) -> tuple:
    """Drive process_tick once per report. Returns (node, log records, csv rows).

    Everything the tick writes apart from the order log is stubbed out -- the
    signal and status CSVs are not what is under test -- and the order log is
    pointed at a temp file so the real _write_order runs and can be read back.
    The risk manager is the production one, with the live parameters, because the
    cooldown it arms is the behaviour these tests assert on.
    """
    node = object.__new__(LI.LiveArbitrageNode)
    node.aggregator = types.SimpleNamespace(evaluate=lambda ticks: [_Sig()])
    node.risk_mgr = RiskManager(max_trade_qty=0.01, cooldown_us=5_000_000,
                                max_drawdown_pct=10.0)
    node.latest_snap_a = node.latest_snap_b = _Snap()
    node.total_orders = node.filled_orders = 0
    node.realized_pnl = node.current_position = 0.0

    pending = list(reports)
    saved = {n: getattr(LI, n) for n in
             ("_write_crypto_signal", "_write_status", "_write_signal",
              "simulate_cross_venue_fill", "ORDERS_PATH", "ae")}
    cap = _Capture()
    LI.logger.addHandler(cap)
    try:
        with tempfile.TemporaryDirectory() as td:
            LI.ORDERS_PATH = str(pathlib.Path(td) / "live_orders_py.csv")
            LI._write_crypto_signal = lambda *a, **k: None
            LI._write_status = lambda *a, **k: None
            LI._write_signal = lambda *a, **k: None
            LI.simulate_cross_venue_fill = lambda *a, **k: pending.pop(0)
            LI.ae = types.SimpleNamespace(make_market_tick=lambda *a: object())
            LI._init_orders_csv()
            for _ in reports:
                node.process_tick()
            with open(LI.ORDERS_PATH, newline="") as f:
                rows = list(csv.DictReader(f))
    finally:
        LI.logger.removeHandler(cap)
        for n, v in saved.items():
            setattr(LI, n, v)
    assert not pending, "a report was never consumed -- process_tick bailed early"
    return node, cap.records, rows


def test_a_rejected_order_is_not_counted_as_a_fill():
    node, _, rows = _run([_rejection(net=-7.5)])

    assert node.total_orders == 1, "the attempt should still be counted"
    assert node.filled_orders == 0, (
        "a rejection was counted as a fill -- the booking lines are outside the "
        "if report.rejected branch"
    )
    assert node.realized_pnl == 0.0, (
        f"a rejection moved realized_pnl to {node.realized_pnl}; nothing was "
        "bought or sold"
    )
    assert len(rows) == 1 and rows[0]["approved"] == "0"


def test_a_rejected_order_does_not_arm_the_cooldown():
    """The half of the defect that changed behaviour rather than reporting.

    record_trade arms a 5 s cooldown that check_order consults on the next
    signal. Called on a rejection, it blocked the next real opportunity -- and
    the commonest rejection is the signal-time spread gate, which fires on most
    ticks. Asserted through the guard rather than by counting calls to a double:
    what matters is that the next order gets through, however that is arranged.
    """
    node, _, _ = _run([_rejection(), _rejection()])

    assert node.total_orders == 2
    assert node.risk_mgr.check_order(0.01).approved, (
        "the cooldown is running after two rejections, so the next genuine "
        "opportunity would be declined for five seconds"
    )


def test_a_fill_does_arm_the_cooldown():
    """The control for the test above.

    Moving record_trade into the fill branch is only correct if it still runs
    there. Without this, deleting the call outright would pass.
    """
    node, _, _ = _run([_fill(net=+0.31, adverse=False)])

    assert node.filled_orders == 1
    assert not node.risk_mgr.check_order(0.01).approved, (
        "a real fill left the cooldown unarmed -- record_trade is not being "
        "called on the fill branch either"
    )


def test_an_adverse_fill_is_counted_and_its_loss_is_booked():
    """An adverse fill is a fill. It counts, it books, it arms the guards.

    This is the direction the fix must not overshoot in: the tautology being
    removed from this project is exactly "do not count the losers", so a change
    that excluded adverse fills from filled_orders or realized_pnl would
    reintroduce it one layer up from the execution model.
    """
    node, _, rows = _run([_fill(net=-1.25, adverse=True,
                                reason="adverse_fill_spread_collapsed")])

    assert node.filled_orders == 1, "an adverse fill was not counted as a fill"
    assert node.realized_pnl == -1.25, (
        f"the loss was not booked: realized_pnl = {node.realized_pnl}"
    )
    assert not node.risk_mgr.check_order(0.01).approved, "the cooldown was not armed"
    assert rows[0]["approved"] == "1", "a booked fill was logged as not approved"


def test_an_adverse_fill_is_not_logged_as_a_win():
    """No green tick over a negative net, and the label says what happened."""
    _, records, _ = _run([_fill(net=-1.25, adverse=True,
                                reason="adverse_fill_spread_collapsed")])

    fill_lines = [r for r in records if "0.0100" in r.getMessage()]
    assert len(fill_lines) == 1, f"expected one fill line, got {fill_lines}"
    line = fill_lines[0]
    msg = line.getMessage()
    assert "✅" not in msg, f"an adverse fill printed the success tick: {msg}"
    assert "FILLED:" not in msg, f"an adverse fill printed as a plain fill: {msg}"
    assert "ADVERSE" in msg.upper(), f"the line is not tagged adverse: {msg}"
    assert "adverse_fill_spread_collapsed" in msg, (
        f"the specific label is missing, so the two adverse causes are "
        f"indistinguishable in the terminal: {msg}"
    )
    assert line.levelno >= logging.WARNING, (
        f"an adverse fill logged at {line.levelname}; it needs to survive a "
        "log level set above INFO, which is where the noise floor sits"
    )


def test_a_profitable_fill_still_reads_as_one():
    """The control for the test above: satisfying it by deleting the tick fails."""
    _, records, _ = _run([_fill(net=+0.31, adverse=False)])

    fill_lines = [r for r in records if "0.0100" in r.getMessage()]
    assert len(fill_lines) == 1
    msg = fill_lines[0].getMessage()
    assert "✅ FILLED" in msg, f"a winning fill no longer reads as one: {msg}"
    assert "ADVERSE" not in msg.upper(), f"a winning fill was tagged adverse: {msg}"
    assert fill_lines[0].levelno == logging.INFO


def test_the_csv_carries_the_adverse_flag_and_the_execution_label():
    """The dashboard cannot infer this from the sign of net_pnl alone.

    Two rows, one winner and one booked loss, both ``approved=1`` with a non-zero
    net. Before the fix nothing in the file separated them except that sign, and
    ``reason`` carried the risk manager's verdict -- which for a booked adverse
    fill is "approved", the one thing it is not useful to record.

    The second tick would be refused by the cooldown the first one arms, so the
    two are run as separate nodes rather than as one sequence.
    """
    _, _, adverse_rows = _run([_fill(net=-1.25, adverse=True,
                                     reason="adverse_fill")])
    _, _, good_rows = _run([_fill(net=+0.31, adverse=False)])

    assert adverse_rows[0]["adverse"] == "1", "the adverse flag never reached the CSV"
    assert adverse_rows[0]["reason"] == "adverse_fill", (
        f"reason is {adverse_rows[0]['reason']!r} -- the execution model's label "
        "was overwritten by the risk manager's"
    )
    assert float(adverse_rows[0]["net_pnl"]) == -1.25

    assert good_rows[0]["adverse"] == "0"
    assert float(good_rows[0]["net_pnl"]) == 0.31
    # And the flag is not merely the sign of the PnL rebadged: a winning fill and
    # a losing one differ in `adverse`, not only in `net_pnl`.
    assert adverse_rows[0]["adverse"] != good_rows[0]["adverse"]


def test_rows_are_never_appended_under_a_header_of_a_different_width():
    """Adding the `adverse` column is the drift this project has already had once.

    dashboard/seed_demo_feed.py spent a release writing sixteen columns to a file
    whose other producer wrote twenty-four, and nothing errored, because a CSV
    reader given a wider row than its header simply mislabels every field from the
    gap onward (tests/test_demo_feed_schema.py). ``_init_orders_csv`` used to open
    in append mode and write the header only when the file was absent, so an
    order log left by yesterday's build would have collected the new eighteenth
    column under a seventeen-column heading.

    It renames rather than truncates: unlike the demo feed's scratch files, this
    is the only record that a live session happened.
    """
    old_header = [c for c in LI.ORDERS_HEADER if c != "adverse"]
    saved = LI.ORDERS_PATH
    try:
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "live_orders_py.csv"
            LI.ORDERS_PATH = str(path)
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(old_header)
                w.writerow(["1"] * len(old_header))

            LI._init_orders_csv()

            lines = path.read_text().splitlines()
            assert lines == [",".join(LI.ORDERS_HEADER)], (
                f"expected a fresh header and nothing else, got {lines}"
            )
            backups = [p for p in pathlib.Path(td).iterdir()
                       if p.name.endswith(".bak")]
            assert len(backups) == 1, (
                "the old order log was destroyed rather than set aside"
            )
            assert old_header[0] in backups[0].read_text()
    finally:
        LI.ORDERS_PATH = saved


def test_the_header_is_the_width_the_writer_writes():
    """A guard on the guard: _write_order raises rather than writing short.

    The width check inside _write_order is what makes the schema self-enforcing,
    so a future column added to ORDERS_HEADER and not to the row fails here
    instead of silently shifting the dashboard's columns.
    """
    saved_header, saved_path = LI.ORDERS_HEADER, LI.ORDERS_PATH
    try:
        with tempfile.TemporaryDirectory() as td:
            LI.ORDERS_PATH = str(pathlib.Path(td) / "orders.csv")
            LI.ORDERS_HEADER = list(saved_header) + ["a_column_nobody_writes"]
            LI._init_orders_csv()
            raised = False
            try:
                LI._write_order(_Sig(), 0.01,
                                LI.OrderCheckResult(True, "approved"))
            except ValueError as exc:
                raised = True
                assert "18 fields" in str(exc), exc
            assert raised, "a row one field short of the header was written anyway"
    finally:
        LI.ORDERS_HEADER, LI.ORDERS_PATH = saved_header, saved_path
