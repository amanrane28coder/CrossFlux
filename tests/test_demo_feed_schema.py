"""The demo seeder's order schema against the C++ writer that owns /tmp/live_orders.csv.

Two producers write this file -- ``SimulatedExecutor::log_order`` in
``cpp_engine/src/execution_manager.cpp`` when the engine runs, and
``dashboard/seed_demo_feed.py`` when it does not -- and one consumer,
``dashboard/app.py``, reads it by column name. That arrangement drifted: the
friction work appended eight columns to the C++ header (``filled_qty_buy`` ...
``adverse``) and the seeder was never updated, so the demo produced sixteen-column
rows for a file whose other producer wrote twenty-four.

The consequence was not a missing panel -- app.py does not read any of the eight
yet -- it was destructive. ``ensure_headers`` replaces any file whose first line
is not the schema it expects, which is correct behaviour against a stale build and
wrong against a *newer* producer: pointed at a 24-column file the engine had just
written, the seeder judged it stale and truncated it, discarding the engine's
orders. Measured, before the fix: one 24-column header plus one engine row in,
one 16-column header and no rows out. ``test_a_file_the_engine_wrote_is_not_treated_as_stale``
holds that shut.

Nothing failed loudly because nothing compared the two schemas. This does. It
parses the header literal out of the .cpp rather than restating it here, because a
copy of the list in a third file is a third thing to forget: if the C++ header and
the seeder schema are both read from source, the only way to pass is to change
both.

It also checks the ordering, not just the set of names. The seeder writes
positionally and app.py reads by name, so a *reordering* on either side is
silently destructive in a way a set comparison would not catch -- every value
lands under some column, and pandas is handed a rectangle it has no reason to
question.
"""
from __future__ import annotations

import csv
import io
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CPP = ROOT / "cpp_engine" / "src" / "execution_manager.cpp"


def _cpp_order_header() -> list[str]:
    """The column names ``SimulatedExecutor``'s constructor writes, in order.

    The header is a run of ``<<``-chained string literals split across lines for
    readability, with a ``\\n`` on the last one. Concatenating every literal
    between the ``std::ofstream csv(...)`` for live_orders.csv and the closing
    ``;`` reconstructs the line the compiler emits, whatever the line breaks and
    interleaved comments happen to be.
    """
    text = CPP.read_text()
    start = text.index('std::ofstream csv("/tmp/live_orders.csv")')
    stmt = text[text.index("csv <<", start):]
    stmt = stmt[:stmt.index(";")]
    # Drop // comments first: the real header sits among explanatory ones, and a
    # commented-out column name would otherwise be indistinguishable from a live
    # one. Then take every double-quoted literal that remains.
    stmt = re.sub(r"//[^\n]*", "", stmt)
    joined = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', stmt))
    assert joined.endswith("\\n"), (
        f"live_orders header literal does not end in a newline escape: {joined!r}"
    )
    return joined[: -len("\\n")].split(",")


def test_seeder_schema_matches_the_cpp_writer_exactly():
    from dashboard import seed_demo_feed as seeder

    cpp = _cpp_order_header()
    py = seeder.SCHEMAS["live_orders.csv"]
    assert cpp == py, (
        "dashboard/seed_demo_feed.py and execution_manager.cpp disagree on "
        f"live_orders.csv.\n  C++: {cpp}\n  py : {py}\n"
        f"  only in C++: {[c for c in cpp if c not in py]}\n"
        f"  only in py : {[c for c in py if c not in cpp]}"
    )


def test_the_parser_finds_the_columns_the_friction_work_appended():
    """A guard on the guard.

    If ``_cpp_order_header`` silently returned the sixteen original columns -- a
    regex that stopped at the comment between the two literal groups would -- the
    test above would pass against the very defect it exists to catch. These eight
    names are the ones that were missing, spelled out here so that failure mode
    is loud.
    """
    cpp = _cpp_order_header()
    for col in ("filled_qty_buy", "filled_qty_sell", "residual_qty",
                "legging_cost", "signal_edge_bps", "realized_edge_bps",
                "leg_gap_ms", "adverse"):
        assert col in cpp, f"{col} not found in the C++ header: {cpp}"
    assert len(cpp) == 24, f"expected 24 columns, parsed {len(cpp)}: {cpp}"


def test_every_row_the_seeder_writes_is_the_width_of_its_header():
    """Both branches, checked through the real builder rather than by eye.

    ``_order_row`` is what makes this true, and it is only true while every field
    name it is handed is in the schema -- a typo'd keyword would raise, which is
    the point. A rejection and a fill go through separate call sites, so both are
    exercised: the rejection is the one that carried defaults for seventeen
    fields and would be the easier of the two to leave short.
    """
    from dashboard import seed_demo_feed as seeder

    cols = seeder.SCHEMAS["live_orders.csv"]
    base = dict(timestamp_ms=1, signal_timestamp_ms=1, obi_delta=-0.5,
                buy_exchange="binance", sell_exchange="kraken",
                fill_price_buy=1.0, fill_price_sell=2.0)

    rejected = seeder._order_row(**base, filled=0, reject_reason="spread_below_min")
    assert len(rejected) == len(cols)
    assert rejected[cols.index("adverse")] == 0
    assert rejected[cols.index("fill_qty")] == 0.0

    filled = seeder._order_row(**base, filled=1, fill_qty=0.01, adverse=1,
                               reject_reason="adverse_fill", net_pnl=-1.0)
    assert len(filled) == len(cols)
    assert filled[cols.index("net_pnl")] == -1.0

    # And the ordering is the schema's, not the keyword order at the call site.
    assert filled[cols.index("buy_exchange")] == "binance"
    assert filled[cols.index("reject_reason")] == "adverse_fill"


def test_an_unpopulated_new_column_raises_instead_of_writing_a_short_row():
    """The failure mode this whole file exists to prevent, forced deliberately.

    Adding a column to the schema without giving the builders a value used to
    produce a row one field short, which the CSV writer accepts. Now it raises
    before anything is written.
    """
    from dashboard import seed_demo_feed as seeder

    cols = seeder.SCHEMAS["live_orders.csv"]
    seeder.SCHEMAS["live_orders.csv"] = cols + ["a_column_nobody_populated"]
    try:
        raised = False
        try:
            seeder._order_row(timestamp_ms=1, signal_timestamp_ms=1, obi_delta=0.0,
                              buy_exchange="a", sell_exchange="b",
                              fill_price_buy=1.0, fill_price_sell=2.0)
        except ValueError as exc:
            raised = True
            assert "a_column_nobody_populated" in str(exc), exc
        assert raised, "a schema column with no value produced a row anyway"
    finally:
        seeder.SCHEMAS["live_orders.csv"] = cols


def test_append_rejects_a_row_of_the_wrong_width():
    """_append is the last line of defence, so it gets its own check.

    Written against a StringIO rather than /tmp: a test that appends to the file
    the dashboard reads would leave a row behind, and this row is deliberately
    malformed.
    """
    from dashboard import seed_demo_feed as seeder

    buf = io.StringIO()
    csv.writer(buf).writerow(["only", "three", "fields"])
    raised = False
    try:
        seeder._append("live_orders.csv", [["only", "three", "fields"]])
    except ValueError as exc:
        raised = True
        assert "3 fields" in str(exc), exc
    assert raised, "_append accepted a 3-field row into a 24-column file"


def test_the_demo_books_its_losers_instead_of_declining_them():
    """The tautology, in the file most likely to reintroduce it.

    The seeder used to reject on ``net_bps <= 0`` after subtracting fees, which
    is the same "recompute the edge, drop the losers" shape that was removed from
    both engines. Under a preset harsh enough to turn every edge over, a demo
    that still declined them would report zero fills; a demo that books them
    reports fills, negative PnL and an ``adverse`` flag on every one.
    """
    from dashboard import seed_demo_feed as seeder

    market = seeder.DemoMarket(seed=7)
    rows: list[list] = []
    schema = seeder.SCHEMAS["live_orders.csv"]
    real_append, real_status = seeder._append, seeder._write_status
    # 100 ms against a 4.79 bps basis is ~20 bps of haircut: nothing survives it.
    real_lat = (seeder.LATENCY_MS, seeder.LATENCY_SIGMA)
    seeder.LATENCY_MS, seeder.LATENCY_SIGMA = 100.0, 0.0
    seeder._append = lambda name, rs: rows.extend(rs) if name == "live_orders.csv" else None
    seeder._write_status = lambda *a, **k: None
    try:
        for i in range(200):
            seeder._tick(market, 1_700_000_000_000 + i * seeder.TICK_MS)
    finally:
        seeder._append, seeder._write_status = real_append, real_status
        seeder.LATENCY_MS, seeder.LATENCY_SIGMA = real_lat

    fills = [r for r in rows if r[schema.index("filled")] == 1]
    assert fills, "no order filled at all -- the entry gate rejected everything"
    assert all(r[schema.index("adverse")] == 1 for r in fills)
    assert all(r[schema.index("net_pnl")] < 0.0 for r in fills)
    assert market.realized_pnl < 0.0, (
        f"200 ticks of guaranteed losses booked {market.realized_pnl:+.2f}"
    )
    # The label the requirement names, not the generic one: at this latency the
    # drift turns the edge over by itself rather than merely failing to cover fees.
    assert all(r[schema.index("reject_reason")] == "adverse_fill_spread_collapsed"
               for r in fills)


def test_a_file_the_engine_wrote_is_not_treated_as_stale():
    """The concrete damage the eight-column gap did, and the fix that closes it.

    ``ensure_headers`` replaces any file whose first line is not the header it
    expects. Against a file left by an older build that is right. Against a file
    left by a *newer* producer it is destructive, and the C++ engine is exactly
    that: it writes twenty-four columns and the seeder used to expect sixteen, so
    starting the demo after an engine run discarded the engine's orders.

    Both halves matter. The first is the property being pinned; the second proves
    the first is not vacuous, by showing the row does not survive a schema that
    disagrees. If ``ensure_headers`` is ever changed to migrate rather than
    truncate, the second half becomes wrong and should be deleted -- it describes
    today's mechanism, not a requirement.
    """
    from dashboard import seed_demo_feed as seeder

    engine_header = ",".join(_cpp_order_header())
    engine_row = ",".join(["1"] * len(_cpp_order_header()))

    def survives(schema: list[str]) -> tuple[bool, int]:
        """Write an engine-style file, run ensure_headers under `schema`, report."""
        import tempfile
        real_schema = seeder.SCHEMAS["live_orders.csv"]
        real_tmp = seeder.TMP
        with tempfile.TemporaryDirectory() as td:
            seeder.TMP = pathlib.Path(td)
            seeder.SCHEMAS["live_orders.csv"] = schema
            try:
                path = seeder.TMP / "live_orders.csv"
                path.write_text(engine_header + "\n" + engine_row + "\n")
                seeder.ensure_headers()
                lines = [l for l in path.read_text().splitlines() if l.strip()]
            finally:
                seeder.SCHEMAS["live_orders.csv"] = real_schema
                seeder.TMP = real_tmp
        return engine_row in lines, len(lines[0].split(","))

    kept, width = survives(list(seeder.SCHEMAS["live_orders.csv"]))
    assert kept, (
        "ensure_headers destroyed a row the C++ engine wrote -- the seeder's "
        "schema no longer matches the engine's header"
    )
    assert width == 24, f"header left {width} columns wide, expected 24"

    # And with the schema as it was: the row is gone. This is the measured
    # before-state, not a hypothetical.
    old_sixteen = list(seeder.SCHEMAS["live_orders.csv"])[:16]
    kept_old, width_old = survives(old_sixteen)
    assert not kept_old and width_old == 16, (
        "a 16-column schema no longer truncates an engine-written file, so the "
        "assertion above proves nothing; see this test's docstring"
    )


def test_the_latency_stream_is_independent_of_the_price_walk():
    """Switching friction preset must not change the market.

    A jittered preset drawing from the price walk's own RNG would shift every
    subsequent price, so two presets would be compared across two different
    markets. Same seed, different jitter, identical prices is the assertion.
    """
    from dashboard import seed_demo_feed as seeder

    def path(sigma: float) -> list[float]:
        market = seeder.DemoMarket(seed=11)
        real_append, real_status = seeder._append, seeder._write_status
        real_lat = (seeder.LATENCY_MS, seeder.LATENCY_SIGMA)
        seeder.LATENCY_MS, seeder.LATENCY_SIGMA = 5.0, sigma
        seeder._append = lambda *a, **k: None
        seeder._write_status = lambda *a, **k: None
        try:
            mids = []
            for i in range(120):
                seeder._tick(market, 1_700_000_000_000 + i * seeder.TICK_MS)
                mids.append(market.mid_a)
                mids.append(market.mid_b)
            return mids
        finally:
            seeder._append, seeder._write_status = real_append, real_status
            seeder.LATENCY_MS, seeder.LATENCY_SIGMA = real_lat

    assert path(0.0) == path(0.9)


def test_the_basis_is_a_level_and_does_not_diffuse_away():
    """The gap between the venues is pulled back to BASIS_BPS, not left to walk.

    With each mid taking its own independent step -- which is what this did -- the
    basis is a random walk whose spread grows without bound, so the demo's edge
    depended on how long ago the process started. Over 2000 ticks the mean basis
    should sit near the target and the excursions should stay bounded; a free walk
    fails both.
    """
    from dashboard import seed_demo_feed as seeder

    market = seeder.DemoMarket(seed=3)
    seen = []
    for _ in range(2000):
        market.step()
        seen.append((market.mid_b - market.mid_a) / market.mid_a * 10_000.0)

    mean = sum(seen) / len(seen)
    assert abs(mean - seeder.BASIS_BPS) < 0.5, (
        f"mean basis {mean:.2f} bps drifted off the {seeder.BASIS_BPS} bps target"
    )
    assert max(abs(b - seeder.BASIS_BPS) for b in seen) < 5.0, (
        "basis excursions are unbounded -- it is still diffusing"
    )
