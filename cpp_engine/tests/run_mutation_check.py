#!/usr/bin/env python3
"""
cpp_engine/tests/run_mutation_check.py
======================================
Asks whether the C++ test suites actually *test* the engine, or merely execute it.

A green suite proves nothing on its own: a test that asserts a value it copied
from the implementation passes no matter what the implementation does. So this
script breaks the implementation on purpose -- one small, plausible edit at a
time -- and checks that the suite notices. A mutation the suite fails to notice
marks a property nothing is pinning down.

Two groups, one per thing worth doubting:

  M*  calculate_weighted_obi in include/signals.hpp, checked by test_signals.cpp
  F*  walk_book, the pending-order queue and the fill accounting, in
      include/friction.hpp and src/execution_manager.cpp, checked by
      test_friction.cpp

The F group exists because asserting a loss is easy to do vacuously. Every
mutation there makes the engine wrong in a direction that *still* shows friction
biting -- filling at the touch, hedging on the larger leg, resolving legs in the
wrong order -- so a suite that only checked "PnL came out negative" would pass
all of them. F4 is the one to watch: it reinstates the profitability gate at fill
time, which is the exact tautology this work removed.

Each mutation is a single textual edit to a *copy* of one file under
cpp_engine/; the repo's own sources are never touched. Headers and sources are
both staged, so a mutation can target a .cpp as well as a .hpp.

Usage
-----
    python3 cpp_engine/tests/run_mutation_check.py
    python3 cpp_engine/tests/run_mutation_check.py --keep        # leave build dirs
    python3 cpp_engine/tests/run_mutation_check.py --only F1,F4  # controls still run

Exit status is 0 only if every mutation is caught.

Why the classification is more careful than "did it print FAILED"
----------------------------------------------------------------
A mutant can be caught several ways, and they are not equally good news. Two
earlier versions of this check each reported a false negative by looking only
for the first:

  assertion  the suite runs and reports failures -- the ideal case, because the
             failing test names tell you *which* property caught it
  crash      the mutant reads out of bounds and dies on a signal. Still caught
             (the suite cannot report green), but it stops the run, so every
             assertion after the crash point goes unevaluated -- and stdout is
             block-buffered when piped, so the PASS lines before it are lost
             too. We run under `stdbuf -o0` where available to recover them and
             name where it died.
  build      the mutant does not compile. Counted, but weakest: the compiler
             caught it, not the tests.

Only exit 0 with an "All N tests PASSED" banner counts as SURVIVED.

Each mutant is built twice, because the implementation can catch a mutation
before the tests get to speak:

  release (-DNDEBUG)  asserts compiled out, so the *tests* are the sole judge.
                      This is the verdict that answers "do the tests check it?"
  debug               asserts live, as the normal test build has them.
                      calculate_obi_delta asserts its inputs are in [-1, 1], so
                      a mutation that unbounds the signal aborts on SIGABRT
                      there instead of failing a test -- the implementation's
                      own contract check masking the suite's answer.

The primary verdict is the release one. A debug/release disagreement is
reported, since it tells you the property is guarded in two independent places.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP_ENGINE = HERE.parent
PROJECT_ROOT = CPP_ENGINE.parent

# Must match the shipped flags. -ffp-contract=off in particular: see the parity
# note in signals.hpp.
FLAGS = ["-std=c++20", "-O2", "-ffp-contract=off", "-Wall", "-Wextra"]


@dataclass(frozen=True)
class Suite:
    """A test binary, and the translation units it needs besides its own."""

    name: str
    test: Path                       # the test .cpp, never mutated
    sources: tuple[str, ...] = ()    # extra .cpp files, named under cpp_engine/src


SUITES: dict[str, Suite] = {
    "signals": Suite("signals", HERE / "test_signals.cpp"),
    # The friction suite exercises the asynchronous execution path, so it links
    # the executor rather than being header-only like the signals one.
    "friction": Suite("friction", HERE / "test_friction.cpp",
                      ("execution_manager.cpp",)),
}


@dataclass(frozen=True)
class Mutation:
    """A single-edit corruption of the implementation, and what should notice."""

    mid: str
    what: str          # the property being removed, in words
    old: str           # exact text to find in `target` (must be unique)
    new: str           # replacement
    expect: str        # which test(s) should catch it, for the record
    target: str = "include/signals.hpp"  # path under cpp_engine/ to corrupt
    suite: str = "signals"               # which suite should notice
    rescue: str = ""   # test-name filter to rerun alone if a crash hides it
    control: bool = False  # a NO-OP edit that must survive; see M0


MUTATIONS: list[Mutation] = [
    # ── Negative control ────────────────────────────────────────────────────
    # Every other entry here is meant to be caught, so a harness that reported
    # "caught" unconditionally -- wrong flags, a stale binary, a crash on
    # startup -- would look perfect. M0 is a genuine no-op: i starts at 0 and
    # increments by 1, so `!=` and `<` are the same loop. It must SURVIVE. If it
    # does not, the harness is reacting to something other than the mutation and
    # none of the verdicts below can be trusted.
    Mutation(
        "M0",
        "nothing -- semantically identical loop condition (control)",
        "for (std::size_t i = 0; i < depth; ++i) {\n        const double w = weights[i];",
        "for (std::size_t i = 0; i != depth; ++i) {\n        const double w = weights[i];",
        "nothing: this edit changes no behaviour",
        control=True,
    ),
    Mutation(
        "M1",
        "sign of the imbalance (bid-heavy should be positive)",
        "num += w * (b - a);",
        "num += w * (a - b);",
        "every signed-value test",
    ),
    Mutation(
        "M2",
        "loop bound takes the SHALLOWER book side",
        "const std::size_t depth = std::min(\n        std::min(static_cast<std::size_t>(snap.bid_depth),\n                 static_cast<std::size_t>(snap.ask_depth)),",
        "const std::size_t depth = std::min(\n        std::max(static_cast<std::size_t>(snap.bid_depth),\n                 static_cast<std::size_t>(snap.ask_depth)),",
        "t14_weighted_sparse_book",
    ),
    Mutation(
        "M3",
        "weights are applied at all (not all levels equal)",
        "const double w = weights[i];",
        "const double w = 1.0;",
        "t11_decay_favours_touch, t15_profiles_distinct",
    ),
    Mutation(
        "M4",
        "loop bound is also capped by the WEIGHT VECTOR length",
        "weights.size()\n    );",
        "static_cast<std::size_t>(snap.bid_depth)\n    );",
        "t16_weight_span_length_bounds_depth (value), t13 empty span (crash)",
        rescue="t16",
    ),
    Mutation(
        "M5",
        "normalization (returning the raw numerator would be unbounded)",
        "return num / den;",
        "return num;",
        "t9_flat_reproduces_obi, t10_weighted_bounded",
    ),

    # ── friction.hpp / the asynchronous execution path ──────────────────────
    # Same idea, second suite. These matter more than the signals ones: the
    # whole point of the async path is that it can lose money, and a suite that
    # asserts losses is unusually easy to write vacuously -- a mutation that
    # makes the engine *more* pessimistic still shows a loss, so the tests have
    # to pin the number, not the sign.
    Mutation(
        "F0",
        "nothing -- reordered operands of a side-effect-free || (control)",
        "if (level.price <= 0.0 || level.volume <= kEpsQty) {",
        "if (level.volume <= kEpsQty || level.price <= 0.0) {",
        "nothing: both operands are pure, so short-circuit order cannot matter",
        target="include/friction.hpp",
        suite="friction",
        control=True,
    ),
    Mutation(
        "F1",
        "walking the book at all -- takes the full size from level 1",
        "const double outstanding = want - taken;\n"
        "        const double take = outstanding < level.volume ? outstanding : level.volume;",
        "const double outstanding = want - taken;\n"
        "        const double take = outstanding;",
        "w2_vwap_blended, w4 partial fill, e2, e5 (25 checks)",
        target="include/friction.hpp",
        suite="friction",
    ),
    Mutation(
        "F2",
        "fill-time ordering of the drain (leaves submission order)",
        "if (a.at_ms != b.at_ms) return a.at_ms > b.at_ms;\n"
        "            if (a.seq   != b.seq)   return a.seq   > b.seq;",
        "if (a.seq   != b.seq)   return a.seq   > b.seq;",
        "q4_legs_resolve_in_time_order (and nothing else -- the booking "
        "order is unchanged, only the pricing order)",
        target="include/friction.hpp",
        suite="friction",
    ),
    Mutation(
        "F3",
        "hedged quantity is the SMALLER leg (max would book gross on size "
        "one venue never supplied)",
        "return buy.filled_qty < sell.filled_qty ? buy.filled_qty : sell.filled_qty;",
        "return buy.filled_qty > sell.filled_qty ? buy.filled_qty : sell.filled_qty;",
        "q13_hedged_is_the_minimum, e6_pnl_decomposition",
        target="include/friction.hpp",
        suite="friction",
    ),
    Mutation(
        "F4",
        "booking losses -- reinstates the entry gate at fill time, which is "
        "the original tautology",
        "o.net_pnl    = o.gross_pnl - o.fees_paid - o.legging_cost;\n"
        "    o.filled     = true;",
        "o.net_pnl    = o.gross_pnl - o.fees_paid - o.legging_cost;\n"
        "    o.filled     = (o.net_pnl >= 0.0);",
        "e3_booked_a_loss, e3_logged_as_adverse_selection, "
        "e3_counted_as_a_fill and the e5 equivalents (8 checks)",
        target="src/execution_manager.cpp",
        suite="friction",
    ),
]


def find_compiler() -> str:
    for name in ("g++", "clang++", "c++"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("no C++ compiler found (tried g++, clang++, c++)")


def build_and_run(
    tree: Path, out_dir: Path, cxx: str, suite: Suite, ndebug: bool, filt: str = ""
) -> tuple[int, str, str]:
    """Compile and run `suite` against the headers and sources under `tree`.

    `tree` is a directory laid out like cpp_engine/ -- an include/ and a src/.
    For the baseline that is cpp_engine itself; for a mutant it is the staged
    copy. Nothing is ever compiled from a mix of the two: -I points at one
    include dir only, so a mutated header cannot be silently shadowed by the
    real one. Returns (rc, output, phase).
    """
    tag = "release" if ndebug else "debug"
    binary = out_dir / f"test_{suite.name}_{tag}"
    if not binary.exists():
        compile_cmd = [cxx, *FLAGS, *(["-DNDEBUG"] if ndebug else []),
                       "-I", str(tree / "include"), str(suite.test),
                       *[str(tree / "src" / s) for s in suite.sources],
                       "-o", str(binary)]
        proc = subprocess.run(compile_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return proc.returncode, proc.stdout + proc.stderr, "build"

    # Unbuffered so a crash does not swallow the PASS lines that preceded it.
    run_cmd = [str(binary)] + ([filt] if filt else [])
    if shutil.which("stdbuf"):
        run_cmd = ["stdbuf", "-o0", "-e0", *run_cmd]
    run = subprocess.run(run_cmd, capture_output=True, text=True)
    return run.returncode, run.stdout + run.stderr, "run"


def stage_tree(dst: Path) -> None:
    """Copy the compilable half of cpp_engine/ so one file can be corrupted.

    Sources as well as headers: the tautology mutation (F4) lives in
    execution_manager.cpp, and mutating a file the build does not read would
    look like a survivor.
    """
    for sub, pattern in (("include", "*.hpp"), ("src", "*.cpp")):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for f in (CPP_ENGINE / sub).glob(pattern):
            shutil.copy2(f, dst / sub / f.name)


def failing_tests(output: str) -> list[str]:
    names = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAIL"):
            rest = stripped[4:].strip()
            names.append(rest.split()[0] if rest else "<unnamed>")
    return names


def passed_count(output: str) -> int:
    return sum(1 for ln in output.splitlines() if ln.strip().startswith("PASS"))


def last_test_started(output: str) -> str:
    """The test after the last PASS -- i.e. where a crash most likely happened."""
    passes = [ln.strip() for ln in output.splitlines() if ln.strip().startswith("PASS")]
    if not passes:
        return "before the first assertion"
    return f"after {passes[-1][4:].strip().split()[0]}"


def classify(rc: int, out: str, phase: str, total: int) -> tuple[str, list[str]]:
    """Returns (verdict, detail_lines). verdict is one of
    build / survived / crash / assertion."""
    if phase == "build":
        return "build", ["did not compile -- the compiler noticed, not the tests"]

    if rc == 0 and "PASSED" in out and "FAILED" not in out:
        return "survived", [f"suite still reports {total} passing -- "
                            f"nothing pins this property down"]

    if rc < 0 or rc >= 128:
        sig = -rc if rc < 0 else rc - 128
        name = {6: "SIGABRT", 11: "SIGSEGV", 8: "SIGFPE"}.get(sig, f"signal {sig}")
        ran = passed_count(out)
        detail = [f"died on {name}, {last_test_started(out)}"]
        if ran < total:
            detail.append(f"{total - ran} of {total} assertions never ran, so "
                          f"their verdict on this mutation is unknown")
        return "crash", detail

    names = failing_tests(out)
    shown = ", ".join(names[:4]) + (f", +{len(names) - 4} more" if len(names) > 4 else "")
    return "assertion", [f"{len(names)} of {total} assertions failed", shown]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    ap.add_argument("--keep", action="store_true", help="keep the mutant build dirs")
    ap.add_argument("--only", default="",
                    help="comma-separated mutation ids to run (controls always run)")
    args = ap.parse_args()

    cxx = find_compiler()

    selected = MUTATIONS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = want - {m.mid for m in MUTATIONS}
        if unknown:
            sys.exit(f"unknown mutation id(s): {', '.join(sorted(unknown))}")
        # Controls stay in: without them a filtered run has nothing telling it
        # whether the harness can distinguish caught from survived at all.
        selected = [m for m in MUTATIONS if m.mid in want or m.control]

    # Read each target once. Mutations are applied to this text, never in place.
    source: dict[str, str] = {}
    for m in selected:
        if m.target not in source:
            source[m.target] = (CPP_ENGINE / m.target).read_text()

    suites = [SUITES[name] for name in
              dict.fromkeys(m.suite for m in selected)]  # order-preserving

    print("=" * 74)
    print("  Mutation check -- weighted OBI, book walking, and the async fill path")
    print("=" * 74)
    print(f"compiler : {cxx}")
    print(f"flags    : {' '.join(FLAGS)}")
    for s in suites:
        extra = "".join(f" + src/{x}" for x in s.sources)
        print(f"suite    : {s.test.relative_to(PROJECT_ROOT)}{extra}")
    print(f"targets  : {', '.join(sorted(source))}")
    print()

    workdir = Path(tempfile.mkdtemp(prefix="crossflux_mut_"))

    # ── Baselines. If these are not green, nothing below means anything. ──────
    baseline: dict[str, int] = {}
    for s in suites:
        base_dir = workdir / "baseline" / s.name
        base_dir.mkdir(parents=True)
        for ndebug in (True, False):
            rc, out, phase = build_and_run(CPP_ENGINE, base_dir, cxx, s, ndebug)
            if rc != 0 or "PASSED" not in out:
                print(f"BASELINE ({s.name}) IS NOT GREEN -- mutation results "
                      f"would be meaningless.")
                print(f"  build={'release' if ndebug else 'debug'} "
                      f"phase={phase} rc={rc}")
                print(out[-2000:])
                return 2
            for line in out.splitlines():
                if line.strip().startswith("All ") and "PASSED" in line:
                    baseline[s.name] = int(line.split()[1])
        print(f"baseline : {s.name} -- {baseline[s.name]} assertions, all passing "
              f"(both debug and release)")
    print()

    # ── Mutants ─────────────────────────────────────────────────────────────
    survived: list[Mutation] = []
    weak: list[Mutation] = []
    control_broken: list[Mutation] = []
    LABEL = {
        "build": "CAUGHT (build)",
        "survived": "*** SURVIVED ***",
        "crash": "CAUGHT (crash)",
        "assertion": "CAUGHT (assertion)",
    }

    for m in selected:
        suite = SUITES[m.suite]
        total = baseline[suite.name]
        text = source[m.target]

        occurrences = text.count(m.old)
        if occurrences != 1:
            reason = ("the anchor is not unique -- make it longer"
                      if occurrences > 1
                      else f"the anchor is gone -- {m.target} changed")
            print(f"{m.mid}: SKIPPED -- anchor text found {occurrences} times, "
                  f"expected exactly 1:")
            print(f"      {reason}. A mutation that is not applied proves nothing,")
            print(f"      so it counts against the run rather than being ignored.")
            (control_broken if m.control else survived).append(m)
            continue

        mdir = workdir / m.mid
        stage_tree(mdir)
        (mdir / m.target).write_text(text.replace(m.old, m.new, 1))

        results = {}
        for tag, ndebug in (("release", True), ("debug", False)):
            rc, out, phase = build_and_run(mdir, mdir, cxx, suite, ndebug)
            results[tag] = classify(rc, out, phase, total)

        verdict, detail = results["release"]      # tests are the sole judge here
        print(f"{m.mid}  removes: {m.what}")
        print(f"      in {m.target}, checked by test_{suite.name}.cpp")

        if m.control:
            # Inverted expectation: surviving is the pass condition.
            if verdict == "survived":
                print(f"      CONTROL OK         suite still green, as it must be "
                      f"-- the harness is not crying wolf")
            else:
                print(f"      *** CONTROL FAILED ***  {LABEL[verdict]} on a no-op "
                      f"edit: {detail[0]}")
                print(f"                         Every verdict below is suspect.")
                control_broken.append(m)
            print(f"                         expected: {m.expect}")
            print()
            continue

        print(f"      {LABEL[verdict]:<18} {detail[0]}")
        for line in detail[1:]:
            print(f"                         {line}")

        if verdict == "survived":
            survived.append(m)
        elif verdict == "build":
            weak.append(m)

        # A crash stops the run, so the tests after it were never asked. If we
        # know which one was written for this mutation, rerun it on its own.
        if verdict == "crash" and m.rescue:
            r_rc, r_out, r_phase = build_and_run(
                mdir, mdir, cxx, suite, ndebug=True, filt=m.rescue)
            r_verdict, r_detail = classify(r_rc, r_out, r_phase,
                                           passed_count(r_out) + len(failing_tests(r_out)))
            mark = "and still catches it" if r_verdict != "survived" else "DOES NOT catch it"
            print(f"                         rerun \"{m.rescue}\" alone: "
                  f"{mark} -- {r_detail[0]}")
            if r_verdict == "survived":
                survived.append(m)

        # Where the two builds disagree, the implementation's own assert is
        # doing work the tests would otherwise have to do alone.
        dbg_verdict, dbg_detail = results["debug"]
        if dbg_verdict != verdict:
            print(f"                         debug build differs: "
                  f"{LABEL[dbg_verdict].strip('* ')} -- {dbg_detail[0]}")
            if dbg_verdict == "crash" and verdict == "assertion":
                print(f"                         (an assert() in {m.target} "
                      f"fires first, masking the suite's own verdict)")

        print(f"                         expected: {m.expect}")
        print()

    # ── Verdict ─────────────────────────────────────────────────────────────
    real = [m for m in selected if not m.control]
    print("=" * 74)
    if control_broken:
        print("CONTROL FAILED -- the harness reacted to an edit that changes")
        print("nothing, so it cannot be trusted to tell caught from survived.")
        print("Fix that before reading anything above as a result.")
    elif survived:
        print(f"{len(survived)} of {len(real)} mutations SURVIVED:")
        for m in survived:
            print(f"  {m.mid} -- {m.what}")
        print("\nEach survivor is a property the suite does not actually check.")
    else:
        print(f"All {len(real)} mutations caught, and "
              f"{'both controls' if len([m for m in selected if m.control]) > 1 else 'the control'} "
              f"survived.")
        print("The suites are not vacuous.")
    if weak:
        print(f"\nNote: {len(weak)} caught only by the compiler "
              f"({', '.join(m.mid for m in weak)}) -- a type error, not a "
              f"behavioural check.")

    if args.keep:
        print(f"\nbuild dirs kept: {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    return 1 if (survived or control_broken) else 0


if __name__ == "__main__":
    sys.exit(main())
