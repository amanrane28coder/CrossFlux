#!/usr/bin/env python3
"""Mutation-check tests/test_live_ingestion_accounting.py against its own defects.

    python3 tests/run_live_ingestion_mutations.py [-k SUBSTRING]

``process_tick`` treated "the risk manager approved the order" as "the order
filled": ``record_trade``, ``filled_orders`` and ``realized_pnl`` all ran outside
the ``if report.rejected`` split, and the fill branch printed one ``✅ FILLED``
line regardless of the sign of the net PnL. Both are fixed. A suite that passes
against the fixed code says nothing about whether it would notice either coming
back, so this puts each one back, one at a time, in a throwaway copy of the tree,
and requires the suite to fail.

M1 is the defect exactly as it was -- the three lines at the ``if
result.approved`` level. M2 is the logging half. M5 is the *opposite* error, and
the more tempting one: excluding adverse fills from the counters would look like
prudence and would reinstate the tautology this project spent its last four
changes removing, one layer above the execution model.

The controls must survive. Without them a harness that reported "caught" for any
reason at all -- a bad anchor, a stale copy, an import error in the mutant tree --
would look perfect.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "src/live_ingestion.py"
SUITE = "tests/test_live_ingestion_accounting.py"

SKIP = shutil.ignore_patterns("data", "__pycache__", "*.so", ".git", "logs",
                              "backtest_results", "*.csv.gz", ".venv")

_BOOKING = (
    "                    self.risk_mgr.record_trade(report.net_pnl)\n"
    "                    self.filled_orders += 1\n"
    "                    self.realized_pnl += report.net_pnl\n"
)

_ADVERSE_LOG = (
    "                    if report.adverse:\n"
    "                        logger.warning(\n"
    '                            f"🔻 ADVERSE FILL ({report.reject_reason}): "\n'
)


@dataclass(frozen=True)
class Mutation:
    name: str
    old: str
    new: str
    control: bool = False        # a genuine no-op: must survive


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "M1 book on approval again, not on the fill (the original defect)",
        _BOOKING,
        _BOOKING.replace("                    ", "                "),
    ),
    Mutation(
        "M2 arm the cooldown on rejections too",
        _BOOKING,
        _BOOKING + "                if report.rejected:\n"
                   "                    self.risk_mgr.record_trade(report.net_pnl)\n",
    ),
    Mutation(
        "M3 count rejections as fills but leave the guards alone",
        _BOOKING,
        _BOOKING + "                if report.rejected:\n"
                   "                    self.filled_orders += 1\n",
    ),
    Mutation(
        "M4 one FILLED line for wins and losses alike",
        _ADVERSE_LOG,
        "                    if False:\n"
        "                        logger.warning(\n"
        '                            f"🔻 ADVERSE FILL ({report.reject_reason}): "\n',
    ),
    Mutation(
        "M5 exclude adverse fills from the counters (the tautology, one layer up)",
        _BOOKING,
        "                    if not report.adverse:\n" + _BOOKING.replace(
            "                    ", "                        "),
    ),
    Mutation(
        "M6 drop the adverse flag from the CSV row",
        "                                 adverse=report.adverse,\n",
        "                                 adverse=False,\n",
    ),
    Mutation(
        "M7 let the risk manager's reason win over the execution label",
        "                                 reason=report.reject_reason or None)\n",
        "                                 reason=None)\n",
    ),
    Mutation(
        "M8 append to an order log whose header is a different width",
        "        backup = f\"{ORDERS_PATH}.{int(time.time())}.bak\"\n",
        "        return\n"
        "        backup = f\"{ORDERS_PATH}.{int(time.time())}.bak\"\n",
    ),
    Mutation(
        "M9 write the row without checking it against the header",
        "    if len(row) != len(ORDERS_HEADER):\n",
        "    if False:\n",
    ),
    Mutation(
        "C1 control: whitespace around an assignment",
        '    expected = ",".join(ORDERS_HEADER)',
        '    expected  = ",".join(ORDERS_HEADER)',
        control=True,
    ),
    Mutation(
        "C2 control: reword the comment above the booking lines",
        "                    # Inside the else, and that placement is the fix.",
        "                    # Inside the else. That placement is the whole fix.",
        control=True,
    ),
)


def run_suite(cwd: Path) -> tuple[bool, str, list[str]]:
    """Run the suite in `cwd`. Returns (failed, summary line, failing test names).

    Falls back to the /tmp shim when pytest is unavailable, which is the case in
    this sandbox -- see the note in TEST_REPORT 1 about the test environment.
    """
    env = {**os.environ}
    shim = Path("/tmp/pytest_shim")
    if shim.is_dir():
        env["PYTHONPATH"] = f"/tmp/stubs:{shim}:."
        cmd = [sys.executable, str(shim / "runtests.py"), SUITE]
    else:
        env["PYTHONPATH"] = "."
        cmd = [sys.executable, "-m", "pytest", "-q", SUITE]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    summary = next((l for l in reversed(out.splitlines())
                    if "passed" in l or "error" in l.lower()), out.strip()[-160:])
    failed = r.returncode != 0
    names = sorted({tok for l in out.splitlines() if l.strip().startswith("FAIL")
                    for tok in [l.split("::")[-1].split()[0]]
                    if tok.startswith("test_")})
    return failed, summary.strip(), names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--only", default="",
                    help="run only mutations whose name contains this")
    args = ap.parse_args()

    picked = [m for m in MUTATIONS if args.only.lower() in m.name.lower()]
    if not picked:
        print(f"no mutation matches {args.only!r}")
        return 1

    print(f"baseline: {run_suite(ROOT)[1]}\n")

    bad = 0
    for m in picked:
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "repo"
            shutil.copytree(ROOT, tree, ignore=SKIP)
            path = tree / TARGET
            text = path.read_text()
            # An anchor that matches twice mutates an arbitrary one of two sites,
            # and the verdict then describes neither. Fail instead of guessing.
            n = text.count(m.old)
            if n != 1:
                print(f"  !! ANCHOR   {m.name}: appears {n}x in {TARGET}")
                bad += 1
                continue
            path.write_text(text.replace(m.old, m.new))
            failed, summary, names = run_suite(tree)

        verdict = "CAUGHT  " if failed else "SURVIVED"
        ok = failed != m.control
        print(f"  {'ok' if ok else '!!'} {verdict} {m.name}")
        print(f"          {summary}")
        if names:
            print(f"          by: {', '.join(names[:5])}")
        bad += not ok

    real = [m for m in picked if not m.control]
    ctrl = [m for m in picked if m.control]
    print(f"\n{len(real) - bad if bad <= len(real) else 0}/{len(real)} mutations "
          f"caught, {len(ctrl)} controls checked")
    if bad:
        print(f"FAIL: {bad} unexpected outcome(s)")
        return 1
    print("PASS: every reinstated defect fails the suite, every no-op does not")
    return 0


if __name__ == "__main__":
    sys.exit(main())
