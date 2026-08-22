#!/usr/bin/env python3
"""Mutation check for check_report_numbers.py -- does the guard actually guard?

check_report_numbers.py exists to catch hand-typed figures that have drifted from
backtest/sweep_results.json. That makes it a test, and an unusual one: it passes
by finding *nothing*, which is also how it would behave if it were broken, or
pointed at the wrong text, or looking for a string that no longer appears
anywhere. Silence is the pass condition and the failure mode at once.

So it is checked the same way the C++ suites are (see
cpp_engine/tests/run_mutation_check.py): perturb one figure at a time in a
throwaway copy of the tree, and insist the checker notices. Controls perturb
something the checker must *not* care about -- re-padded columns, reworded prose
-- because a guard that fires on any edit whatsoever is a guard nobody keeps.

    python3 backtest/check_report_numbers_mutations.py

Exit 0 means every perturbation was caught and every control survived.

One mutation here earned its place by finding a real hole. M7 re-asserts 93.3% as
fact two paragraphs below the sentence that retracts it. The original scan looked
600 characters back and 300 forward for the word "wrong", so the re-assertion sat
inside the window of the genuine retraction and passed. The scan is now scoped to
the sentence containing the figure, and M7 fails as it should.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
CHECKER = "backtest/check_report_numbers.py"

# Everything the checker reads. Copied, never touched in place.
FILES = ("TEST_REPORT.md", "src/friction.py", "backtest/engine.py",
         "backtest/sweep_results.json", CHECKER)


class Tree:
    """A disposable copy of the files the checker reads."""

    def __init__(self, root: Path) -> None:
        self.root = root
        for rel in FILES:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)
        # __init__.py so `backtest` stays a package if the checker ever imports.
        (root / "backtest" / "__init__.py").touch()
        self.pristine = {rel: (root / rel).read_text() for rel in FILES}

    def restore(self) -> None:
        for rel, text in self.pristine.items():
            (self.root / rel).write_text(text)

    def sub(self, rel: str, old: str, new: str) -> None:
        """Replace a unique anchor. A non-unique anchor is a bug in the mutation
        itself, not a finding, so it raises rather than being reported."""
        p = self.root / rel
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            raise AssertionError(f"{rel}: anchor {old[:50]!r} appears {n}x, want 1")
        p.write_text(text.replace(old, new, 1))

    def run(self) -> tuple[int, str]:
        r = subprocess.run([sys.executable, CHECKER], cwd=self.root,
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


# ── the mutations ────────────────────────────────────────────────────────────
# Each is (id, what it corrupts, edit). Every one leaves a file that reads
# plausibly -- a wrong figure that looks like a right one is the only kind that
# survives review, so it is the only kind worth testing against.

FRICTION_ROW_001 = "       0.01   98.6        1.4        0.01    4.25 bps     $0.08M\n"
FRICTION_ROW_010 = "       0.10   97.7        2.3        0.08    4.19 bps     $0.80M\n"
FRICTION_ROW_300 = "       3.00   85.3       14.7        7.25    3.40 bps    $18.12M\n"

MUTATIONS: list[tuple[str, str, Callable[[Tree], None]]] = [
    ("M1", "friction.py table: win rate at 0.01 BTC, 98.6 -> 98.7",
     lambda t: t.sub("src/friction.py", "0.01   98.6", "0.01   98.7")),

    ("M2", "friction.py table: net PnL at the peak, $18.12M -> $18.21M",
     lambda t: t.sub("src/friction.py", "$18.12M", "$18.21M")),

    ("M3", "friction.py table: the PnL-peak row deleted (curve still monotone)",
     lambda t: t.sub("src/friction.py", FRICTION_ROW_300, "")),

    ("M4", "friction.py table: two rows swapped",
     lambda t: t.sub("src/friction.py", FRICTION_ROW_001 + FRICTION_ROW_010,
                     FRICTION_ROW_010 + FRICTION_ROW_001)),

    ("M5", "friction.py table: an extra row for a size never swept",
     lambda t: t.sub("src/friction.py", FRICTION_ROW_010,
                     "       0.02   98.4        1.6        0.02    4.23 bps     $0.16M\n"
                     + FRICTION_ROW_010)),

    ("M6", "friction.py prose: the zero-vs-stress pair that isolates latency",
     lambda t: t.sub("src/friction.py", "99.3% under `zero`", "99.9% under `zero`")),

    ("M7", "friction.py: 93.3% re-asserted as fact below its own retraction",
     lambda t: t.sub("src/friction.py", "Three things that table does say",
                     "At 0.01 BTC the win rate is 93.3%.\n\n"
                     "Three things that table does say")),

    ("M8", "engine.py table: mean edge at 0.25 BTC, 4.14 -> 4.15",
     lambda t: t.sub("backtest/engine.py", "0.22      4.14 bps", "0.22      4.15 bps")),

    ("M9", "engine.py prose: an adverse-fill endpoint, 45,269 -> 45,268",
     lambda t: t.sub("backtest/engine.py", "45,269 at 3.0 BTC", "45,268 at 3.0 BTC")),

    ("M10", "engine.py: the table's locating marker renamed",
     lambda t: t.sub("backtest/engine.py", "qty BTC   win%    adverse%",
                     "size      win%    adverse%")),

    ("M11", "engine.py: the locating marker duplicated (block is ambiguous)",
     lambda t: t.sub("backtest/engine.py", "        # What that changes, measured on",
                     "        #   qty BTC   win%    adverse%   unfilled%   mean edge\n"
                     "        # What that changes, measured on")),

    ("M12", "TEST_REPORT.md table: adverse fills at 3.0 BTC, 45,269 -> 45,270",
     lambda t: t.sub("TEST_REPORT.md", "| 45,269 |", "| 45,270 |")),

    ("M13", "TEST_REPORT.md: 54.96% re-asserted in a new sentence",
     lambda t: t.sub("TEST_REPORT.md", "Three things there could not have happened",
                     "Unfilled reaches 54.96% at the top size.\n\n"
                     "Three things there could not have happened")),

    ("M14", "sweep_results.json rerun: win rate at 0.01 BTC moves (all 3 files stale)",
     lambda t: t.sub("backtest/sweep_results.json",
                     '"win_rate_pct": 98.59393726797042',
                     '"win_rate_pct": 97.49393726797042')),
]

# ── the controls ─────────────────────────────────────────────────────────────
# Edits the checker must ignore. Without these, "every mutation caught" would be
# satisfiable by a script that fails on any diff at all.

CONTROLS: list[tuple[str, str, Callable[[Tree], None]]] = [
    ("C1", "friction.py table columns re-padded, every value unchanged",
     lambda t: t.sub("src/friction.py",
                     "       0.25   97.1        2.9        0.22    4.14 bps     $1.98M",
                     "     0.25    97.1     2.9      0.22   4.14 bps   $1.98M")),

    ("C2", "engine.py: a comment beside the table reworded",
     lambda t: t.sub("backtest/engine.py", "# What that changes, measured on",
                     "# What this changes, measured on")),

    ("C3", "TEST_REPORT.md: the retraction reworded, still marking the figures wrong",
     lambda t: t.sub("TEST_REPORT.md",
                     "from a throwaway `measure_tautology_break.py` — **is wrong and has been struck.**",
                     "from a throwaway `measure_tautology_break.py` — **is wrong.**")),
]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ckrn-mut-") as tmp:
        tree = Tree(Path(tmp))

        code, out = tree.run()
        if code != 0:
            print("baseline FAILS before any mutation -- fix that first:\n")
            print(out)
            return 2
        print(f"baseline: {out.splitlines()[0]}\n")

        bad = 0
        for mid, what, mutate in MUTATIONS:
            tree.restore()
            mutate(tree)
            code, out = tree.run()
            lines = [l[4:] for l in out.splitlines() if l.startswith("  - ")]
            verdict = "CAUGHT" if code else "MISSED"
            print(f"{mid:4} {verdict:7} {len(lines)} failure(s)  {what}")
            if code == 0:
                bad += 1
                print("          the checker passed on a corrupted tree")
            else:
                print(f"          {lines[0][:150] if lines else out.strip()[:150]}")

        print()
        for mid, what, mutate in CONTROLS:
            tree.restore()
            mutate(tree)
            code, out = tree.run()
            verdict = "SURVIVED" if code == 0 else "FALSE ALARM"
            print(f"{mid:4} {verdict:11}  {what}")
            if code:
                bad += 1
                lines = [l[4:] for l in out.splitlines() if l.startswith("  - ")]
                for l in lines:
                    print(f"          {l[:150]}")

    print()
    if bad:
        print(f"{bad} unexpected outcome(s): the guard is not doing what it claims.")
        return 1
    print(f"All {len(MUTATIONS)} mutations caught, all {len(CONTROLS)} controls "
          f"survived. check_report_numbers.py is not vacuous.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
