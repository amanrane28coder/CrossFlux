#!/usr/bin/env python3
"""Mutation-check tests/test_execution_simulator.py against the defects it exists to stop.

    python3 tests/run_execution_mutations.py [-k SUBSTRING]

``simulate_cross_venue_fill`` used to reject three times *after* friction had been
applied -- on a collapsed spread, on a spread below the minimum, and on a negative
net PnL -- and each rejection booked ``pnl_net = 0.0``. That is the tautology:
recompute the edge once the costs are known, drop whatever came out badly, and the
ledger keeps only winners. The gates are gone and the losses are booked.

A test suite that passes against the fixed code proves nothing about whether it
would notice the defect coming back. This puts each defect back, one at a time, in
a throwaway copy of the tree, and requires the suite to fail. The negative control
must survive: without it, a harness that reported "caught" for any reason at all
(wrong path, stale copy, import error) would look perfect.

Two of the mutations below are not hypothetical:

  M3  is how this function actually behaved before the fix. The latency drift was
      computed, used to decide whether to reject, and then discarded -- the fill
      walked the signal-time book. So latency gated trades without ever costing
      one anything, and the old ``test_latency_squeeze_scales_with_latency``
      passed on the rejection alone while all three of its outcomes booked an
      identical PnL. It is the reason that test now asserts an ordering.
  M4a is a hole this harness found in a test written moments earlier. The fixture
      made the sell leg the short one, so ``min(buy, sell)`` and
      ``sell_filled_qty`` agreed and the mutation passed all 20 tests. The test
      now runs both orientations.

M2 is the Python spelling of C++ mutation F4 in
``cpp_engine/tests/run_mutation_check.py`` (``o.filled = true`` ->
``o.filled = (o.net_pnl >= 0.0)``). Any future edit that makes booking
conditional on profitability is this defect, whichever language it lands in.
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
TARGET = "src/execution_simulator.py"
SUITE = "tests/test_execution_simulator.py"

# Copying the tree is cheap only if the data directory is left behind.
SKIP = shutil.ignore_patterns("data", "__pycache__", "*.so", ".git", "logs",
                              "backtest_results", "*.csv.gz", ".venv")


@dataclass(frozen=True)
class Mutation:
    name: str
    old: str
    new: str
    control: bool = False        # a genuine no-op: must survive


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "M1 reinstate the spread-collapse rejection",
        "    adverse = net_pnl < 0.0\n",
        "    if moved_sell - moved_buy <= 0.0:\n"
        "        return ExecutionReport(None, None, 0.0, 0.0, 0.0, 0.0,\n"
        "            rejected=True,\n"
        "            reject_reason='spread_collapsed_during_latency',\n"
        "            latency_ms=latency_ms)\n"
        "    adverse = net_pnl < 0.0\n",
    ),
    Mutation(
        "M2 reinstate the fees_exceed_profit gate (C++ F4, in Python)",
        "    adverse = net_pnl < 0.0\n",
        "    if net_pnl <= 0.0:\n"
        "        return ExecutionReport(buy_fill, sell_fill, gross_pnl, net_pnl,\n"
        "            total_slippage, total_fees, rejected=True,\n"
        "            reject_reason='fees_exceed_profit', latency_ms=latency_ms)\n"
        "    adverse = net_pnl < 0.0\n",
    ),
    Mutation(
        "M3 make the latency drift inert again (walk the signal-time book)",
        "    fill_buy_book = drift_book(buy_book, 1.0 + drift)\n"
        "    fill_sell_book = drift_book(sell_book, 1.0 - drift)\n",
        "    fill_buy_book = drift_book(buy_book, 1.0)\n"
        "    fill_sell_book = drift_book(sell_book, 1.0)\n",
    ),
    Mutation(
        "M4a book gross on the sell leg instead of the hedged min",
        "    hedged = min(buy_fill.filled_qty, sell_fill.filled_qty)\n",
        "    hedged = sell_fill.filled_qty\n",
    ),
    Mutation(
        "M4b book gross on the buy leg instead of the hedged min",
        "    hedged = min(buy_fill.filled_qty, sell_fill.filled_qty)\n",
        "    hedged = buy_fill.filled_qty\n",
    ),
    Mutation(
        "M5 collapse the two adverse labels into one",
        "                  if moved_sell - moved_buy <= 0.0 < sell_price - buy_price\n",
        "                  if True\n",
    ),
    Mutation(
        "M6 shift only the touch, not the whole book (non-parallel drift)",
        "    return [_Level(l.price * factor, l.volume) for l in levels]\n",
        "    return [_Level(l.price * (factor if i == 0 else 1.0), l.volume)\n"
        "            for i, l in enumerate(levels)]\n",
    ),
    Mutation(
        "M7 drift the caller's book in place instead of copying",
        "    return [_Level(l.price * factor, l.volume) for l in levels]\n",
        "    for l in levels:\n        l.price *= factor\n    return levels\n",
    ),
    Mutation(
        "M8 drift the volumes as well as the prices",
        "    return [_Level(l.price * factor, l.volume) for l in levels]\n",
        "    return [_Level(l.price * factor, l.volume * factor) for l in levels]\n",
    ),
    Mutation(
        "M9 relabel the no-fill guard as an adverse fill",
        '            rejected=True, reject_reason="no_liquidity_at_fill",\n',
        '            rejected=False, reject_reason="adverse_fill",\n',
    ),
    Mutation(
        "C1 control: whitespace around an assignment",
        "    hedged = min(",
        "    hedged  = min(",
        control=True,
    ),
    Mutation(
        "C2 control: reword the comment above the removed gates",
        "    # No gate here, and none after the fill either.",
        "    # Nothing is re-gated here, and nothing after the fill either.",
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
