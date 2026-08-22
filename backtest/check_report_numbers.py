#!/usr/bin/env python3
"""Check that every hand-typed sweep figure in the repo still matches the JSON.

Three files quote numbers produced by ``backtest/sweep_order_size.py``:

    TEST_REPORT.md        section 2.1's table and the prose around it
    src/friction.py       the module docstring, and two preset descriptions
    backtest/engine.py    the comment above the entry gate

All three are prose, so their numbers are typed by hand and rot the first time
the sweep is rerun. This re-derives them from ``backtest/sweep_results.json``
and fails loudly on any disagreement.

It exists because that has already happened twice. It caught four wrong
adverse-fill counts in TEST_REPORT.md on its very first run. Then, because it
guarded *only* TEST_REPORT.md, a retracted set of figures -- 93.3% / 59.8% /
54.96%, from a script that read a truncated slice of the day -- went on being
asserted in ``src/friction.py``'s docstring and in the ``stress`` preset's own
description long after the report had struck them. A reader of the module would
have believed them, since nothing in the module said otherwise. Hence the file
list above: a guard that covers the document but not the code is not a guard.

    python3 backtest/check_report_numbers.py

Exit 0 means the repo agrees with the data it claims to describe.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "TEST_REPORT.md"
SWEEP = ROOT / "backtest" / "sweep_results.json"

fails: list[str] = []


def ck(ok: bool, msg: str) -> None:
    if not ok:
        fails.append(msg)


# A number as these tables write one: optional sign, optional thousands commas,
# optional decimals. Deliberately not anchored, so it pulls "18.12" out of
# "$18.12M" and finds nothing in "bps".
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

# pnl_net is quoted in millions. Anything not listed is quoted as stored.
SCALE = {"pnl_net": 1e-6}


@dataclass(frozen=True)
class Table:
    """A fixed-width table of sweep figures embedded in a comment or docstring.

    Located by a pair of literal markers rather than line numbers, because line
    numbers in a 1,400-line file are a promise nobody keeps. Rows are matched on
    their *numeric tokens* rather than the whole line, so realigning the columns
    does not fail the check but changing a value does.
    """

    label: str
    path: Path
    after: str                    # literal on the line above the first row
    before: str                   # literal on the line below the last row
    fields: tuple[str, ...]       # sweep key per column; fields[0] must be qty
    fmt: tuple[str, ...]          # format spec per column
    n_rows: int                   # how many data rows to insist on
    friction: str = "stress"
    # Whether this table claims to describe the whole curve. If it does, it has
    # to quote the endpoints and the peak: a table that drops one of those
    # misdescribes the curve while every surviving row still matches, which is
    # exactly how a wrong table passes review.
    whole_curve: bool = False
    names: tuple[str, ...] = ()

    def cells(self, row: dict) -> list[str]:
        return [f"{row[f] * SCALE.get(f, 1.0):{sp}}"
                for f, sp in zip(self.fields, self.fmt)]


# Every file that quotes a sweep figure, by the name used in messages and in the
# prose-claim table below. Adding a file here is what widens the guard.
SOURCES: dict[str, Path] = {
    "TEST_REPORT.md": REPORT,
    "src/friction.py": ROOT / "src" / "friction.py",
    "backtest/engine.py": ROOT / "backtest" / "engine.py",
}

TABLES: tuple[Table, ...] = (
    Table(
        label="src/friction.py docstring",
        path=SOURCES["src/friction.py"],
        after="qty BTC   win%   adverse%   unfilled%   mean edge    net PnL",
        before="Regenerate with",
        fields=("qty", "win_rate_pct", "adverse_pct", "unfilled_pct",
                "mean_edge_bps", "pnl_net"),
        fmt=(".2f", ".1f", ".1f", ".2f", ".2f", ".2f"),
        n_rows=6,
        whole_curve=True,
        names=("qty", "win", "adverse", "unfilled", "edge", "pnl"),
    ),
    Table(
        label="backtest/engine.py entry-gate comment",
        path=SOURCES["backtest/engine.py"],
        after="qty BTC   win%    adverse%   unfilled%   mean edge",
        before="A losing trade is now reachable",
        fields=("qty", "win_rate_pct", "adverse_pct", "unfilled_pct",
                "mean_edge_bps"),
        fmt=(".2f", ".1f", ".1f", ".2f", ".2f"),
        n_rows=5,
        names=("qty", "win", "adverse", "unfilled", "edge"),
    ),
)


def block(spec: Table) -> str | None:
    """The text strictly between the two markers, or None if either is missing.

    Both markers must appear exactly once. A marker that matches twice means the
    check is reading an arbitrary one of two regions, which is worse than not
    checking, so it is reported as a failure rather than resolved by guessing.
    """
    text = spec.path.read_text()
    for mark in (spec.after, spec.before):
        n = text.count(mark)
        if n != 1:
            fails.append(f"{spec.label}: marker {mark[:40]!r} appears {n} times, "
                         f"expected exactly 1")
            return None
    return text.split(spec.after)[1].split(spec.before)[0]


def check_table(spec: Table, by: dict[str, dict[float, dict]]) -> int:
    """Compare one embedded table against the sweep. Returns rows checked."""
    body = block(spec)
    if body is None:
        return 0
    rows = by[spec.friction]

    quoted: list[float] = []
    for line in body.splitlines():
        toks = NUM.findall(line)
        if len(toks) != len(spec.fields):
            continue                      # header, units, rule, blank, prose
        try:
            qty = float(toks[0])
        except ValueError:
            continue
        if qty not in rows:
            # A plausible-looking row whose size was never swept: report it
            # rather than skipping, since skipping is how a stale row hides.
            fails.append(f"{spec.label}: quotes qty={qty:g}, not in the sweep JSON")
            continue
        quoted.append(qty)
        for got, exp, name in zip(toks, spec.cells(rows[qty]), spec.names):
            ck(got.replace(",", "") == exp,
               f"{spec.label} qty={qty:g} {name}: file {got!r} != sweep {exp!r}")

    ck(len(quoted) == spec.n_rows,
       f"{spec.label}: found {len(quoted)} data rows, expected {spec.n_rows}")
    ck(quoted == sorted(quoted), f"{spec.label}: rows are out of order: {quoted}")
    if spec.whole_curve:
        peak = max(rows, key=lambda q: rows[q]["pnl_net"])
        for must, why in ((min(rows), "smallest size swept"),
                          (max(rows), "largest size swept"),
                          (peak, "PnL peak")):
            ck(must in quoted, f"{spec.label}: omits the {why} ({must:g} BTC)")
    return len(quoted)


def prose_claims(by: dict[str, dict[float, dict]]) -> list[tuple[str, str, str]]:
    """(file, label, literal) for every number asserted in prose, not a table.

    Sizes are written with one decimal in prose and two in tables. Both
    conventions are checked as written rather than normalised, so a size that
    changes in one place and not the other still shows up.
    """
    stress, zero = by["stress"], by["zero"]
    peak = max(stress.values(), key=lambda r: r["pnl_net"])
    big = stress[max(stress)]
    small = stress[min(stress)]
    zsmall, zbig = zero[0.01], zero[3.0]
    signals = f"{next(iter(stress.values()))['signals']:,} signals"
    z_pair = (f"{zsmall['win_rate_pct']:.1f}% at 0.01 BTC and "
              f"{zbig['win_rate_pct']:.1f}% at 3.0")

    return [
        ("TEST_REPORT.md", "signal count", signals),
        ("TEST_REPORT.md", "peak size", f"peaks at {peak['qty']:.1f} BTC"),
        ("TEST_REPORT.md", "zero preset, small",
         f"{zsmall['win_rate_pct']:.1f}% at 0.01 BTC"),
        ("TEST_REPORT.md", "zero preset, large",
         f"{zbig['win_rate_pct']:.1f}% at 3.0"),
        ("TEST_REPORT.md", "largest unfilled",
         f"{big['unfilled_pct']:.2f}% at {big['qty']:.1f} BTC"),

        ("src/friction.py", "signal count", signals),
        ("src/friction.py", "peak size", f"peaks at {peak['qty']:.1f} BTC"),
        ("src/friction.py", "largest size win rate",
         f"gives {big['win_rate_pct']:.1f}%"),
        # The one figure that isolates latency from size: same 0.01 BTC order
        # under zero and stress. If either side moves, the "0.7 points" claim in
        # the stress description is no longer arithmetic anyone can redo.
        ("src/friction.py", "latency cost at the smallest size",
         f"({zsmall['win_rate_pct']:.1f}% under `zero`, "
         f"{small['win_rate_pct']:.1f}% here)"),
        ("src/friction.py", "zero preset pair", z_pair),

        ("backtest/engine.py", "zero preset pair", z_pair),
        ("backtest/engine.py", "adverse endpoints",
         f"{small['adverse_fills']:,} adverse fills at 0.01 BTC, "
         f"{stress[3.0]['adverse_fills']:,} at 3.0 BTC"),
    ]


# 93.3% / 59.8% / 54.96% came from a script that read a truncated slice of the
# day. Scanned in every file that quotes sweep numbers, not just the report:
# leaving them live in the module while the report struck them is the exact
# failure this check was widened to catch.
#
# check_report_numbers_mutations.py is deliberately not in SOURCES. It has to
# contain these literals to inject them, so scanning it would fail by design.
RETRACTED = ("93.3%", "59.8%", "54.96%")
MARKERS = ("struck", "wrong")

# A sentence boundary, or a paragraph one. The scope of "is this figure marked as
# retracted?" is the sentence containing it and nothing wider. A wider window was
# tried first -- 600 characters back, 300 forward -- and it let a re-assertion
# through: a new sentence stating 93.3% as fact, dropped two paragraphs below the
# real retraction, passed because the word "wrong" was still inside the window.
# Proximity to a retraction is not a retraction.
BOUNDARY = re.compile(r"(?<=[.!?])[ \t\n]|\n[ \t]*\n")


def sentence_at(text: str, i: int) -> str:
    """The sentence containing offset `i`, bounded also by blank lines."""
    start = 0
    for m in BOUNDARY.finditer(text[:i]):
        start = m.end()
    m = BOUNDARY.search(text, i)
    return text[start:m.start() + 1 if m else len(text)]


def check_retracted(texts: dict[str, str]) -> int:
    """Each retracted figure may appear only in a sentence that retracts it."""
    seen = 0
    for name, text in texts.items():
        for stale in RETRACTED:
            for m in re.finditer(re.escape(stale), text):
                seen += 1
                sent = sentence_at(text, m.start())
                ck(any(w in sent for w in MARKERS),
                   f"{name}: retracted figure {stale} is asserted as fact -- the "
                   f"sentence containing it says none of {MARKERS}: "
                   f"{' '.join(sent.split())[:120]!r}")
    return seen


def main() -> int:
    rows = json.loads(SWEEP.read_text())
    by: dict[str, dict[float, dict]] = {}
    for r in rows:
        by.setdefault(r["friction"], {})[r["qty"]] = r
    stress = by["stress"]
    md = REPORT.read_text()

    # ── TEST_REPORT.md section 2.1's markdown table, cell by cell ────────────
    # Kept separate from TABLES: it is pipe-delimited with a bolded cell and a
    # $M column, so it shares nothing with the fixed-width comment tables but
    # the data behind it.
    try:
        sec = md.split("Swept across order size")[1].split("Three things there")[0]
    except IndexError:
        print("FAIL: could not locate section 2.1's sweep table in TEST_REPORT.md")
        return 1
    trs = [l for l in sec.splitlines()
           if l.startswith("| ") and "---" not in l and "BTC/leg" not in l]
    ck(len(trs) >= 5, f"section 2.1's table has only {len(trs)} data rows")

    quoted: list[float] = []
    for line in trs:
        cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
        ck(len(cells) == 6, f"row {cells[:1]} has {len(cells)} cells, expected 6")
        qty = float(cells[0])
        quoted.append(qty)
        ck(qty in stress, f"section 2.1 quotes qty={qty} which is not in the sweep JSON")
        if qty not in stress or len(cells) != 6:
            continue
        r = stress[qty]
        want = [f"{qty:.2f}", f"{r['win_rate_pct']:.1f}%",
                f"{r['adverse_fills']:,}", f"{r['unfilled_pct']:.2f}%",
                f"{r['mean_edge_bps']:.2f} bps", f"${r['pnl_net']/1e6:.2f}M"]
        for got, exp, name in zip(cells, want, ("qty", "win", "adverse",
                                                "unfilled", "edge", "pnl")):
            ck(got == exp, f"section 2.1 qty={qty:g} {name}: report {got!r} != {exp!r}")

    # Coverage, not just correctness. The table is a summary and may skip sizes,
    # but dropping an endpoint or the peak would misdescribe the curve while
    # every surviving row still matched — which is exactly how a wrong table
    # passes review.
    peak_qty = max(stress, key=lambda q: stress[q]["pnl_net"])
    for must, why in ((min(stress), "smallest size swept"),
                      (max(stress), "largest size swept"),
                      (peak_qty, "PnL peak")):
        ck(must in quoted, f"section 2.1's table omits the {why} ({must:g} BTC)")
    ck(quoted == sorted(quoted), f"section 2.1's table rows are out of order: {quoted}")

    # ── the same treatment for the tables embedded in code ───────────────────
    checked = len(trs) + sum(check_table(spec, by) for spec in TABLES)

    # ── prose claims that name a number ──────────────────────────────────────
    texts = {name: p.read_text() for name, p in SOURCES.items()}
    claims = prose_claims(by)
    for name, label, needle in claims:
        text = texts.get(name)
        if text is None:
            fails.append(f"{name} is not in SOURCES (claim {label!r})")
            continue
        ck(needle in text, f"{name} prose {label}: {needle!r} not found")

    # A capacity claim needs an interior peak, or it is unsupported by this data.
    peak = max(stress.values(), key=lambda r: r["pnl_net"])
    ck(peak["qty"] != max(stress),
       "the report claims a capacity peak but PnL is still rising at the largest "
       "size swept")

    stale = check_retracted(texts)

    print(f"checked {checked} table rows, {len(claims)} prose claims and "
          f"{stale} retracted-figure mentions across {len(texts)} files "
          f"against {SWEEP.name}")
    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("TEST_REPORT.md, src/friction.py and backtest/engine.py all agree "
          "with the sweep data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
