#!/usr/bin/env python3
"""Bit-for-bit parity checks between the C++ engine and its Python reference.

Run from the repository root:

    python3 cpp_engine/tests/run_parity_check.py

Two things are checked, each with its own harness binary:

    calculate_weighted_obi  vs  src/obi_weights.py     (the signal)
    friction::walk_book     vs  src/friction.py        (the fill price)

Both are built with the flags the project actually ships (see
cpp_engine/CMakeLists.txt), fed inputs as hex doubles, and asserted to return the
*same double* as Python -- not merely a close one. Exit code 0 means bit-exact
on both, 1 means a real divergence, 3 means the harness could not run (no
compiler, missing source).

Why bit-exact and not a tolerance
---------------------------------
Each pair is a transcription of the other, so any disagreement is a rounding
difference, and a tolerance loose enough to pass would also hide a mis-indexed
loop. Both harnesses were written to catch a divergence around 1 ULP:

  weighted OBI  GCC's default -ffp-contract=fast fuses ``num += w * (b - a)``
                into an FMA, rounding once where Python rounds twice. 339 of 3756
                values differed, all on the decay_75 profile -- the only preset
                whose weights are not powers of two, hence the only one where
                that multiply rounds at all.
  walk_book     writing the outstanding-quantity update as ``remaining -= take``
                rather than ``want - taken`` against a running total put the two
                sides ~2.8e-14 apart: inside any tolerance anyone would choose,
                and the exact drift that makes a vectorised fast path
                unauditable.

The default run also rebuilds each harness with -ffp-contract=fast and reports
the contrast, so the measurement recorded in the CMake comments stays
reproducible instead of becoming folklore. Those second builds are
informational; only the project-flags runs decide the exit code.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.friction import EPS_QTY, walk_book  # noqa: E402
from src.obi_weights import PROFILES, get_profile, weighted_obi_from_volumes  # noqa: E402

N = 5
#: Must match the profile array in parity_weighted_obi.cpp, in order. Asserted
#: against the labels the binary prints, so a reorder fails loudly.
ORDER = ["flat", "decay_50", "decay_75", "l1_only"]

SRC = REPO_ROOT / "cpp_engine" / "tests" / "parity_weighted_obi.cpp"
SRC_WALK = REPO_ROOT / "cpp_engine" / "tests" / "parity_walk_book.cpp"
INCLUDE = REPO_ROOT / "cpp_engine" / "include"

#: The project's own optimisation flags, minus -march=native: a parity failure
#: caused by a host-specific instruction selection would be untriageable, and
#: -march=native is documented in cpp_engine/CMakeLists.txt as accepted risk for
#: cross-machine reproducibility. Add it with --march-native to check a host.
PROJECT_FLAGS = ["-O3", "-ffp-contract=off"]
CONTRAST_FLAGS = ["-O3", "-ffp-contract=fast"]


def find_compiler() -> str | None:
    for name in ("g++", "clang++", "c++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build(compiler: str, flags: list[str], out: pathlib.Path,
          src: pathlib.Path = SRC) -> None:
    cmd = [compiler, "-std=c++20", "-Wall", "-Wextra", *flags,
           "-I", str(INCLUDE), str(src), "-o", str(out)]
    subprocess.run(cmd, check=True)


def books() -> list[tuple[list[float], list[float], int, int]]:
    """Books spanning the cases where the two sides could plausibly diverge.

    Wildly different magnitudes (where summation order shows up), asymmetric
    depths (where the three-way min matters), zero volumes, and near-identical
    sides (where the numerator is pure cancellation).
    """
    rng = random.Random(20260821)
    out: list[tuple[list[float], list[float], int, int]] = []
    # 1. plain uniform volumes
    for _ in range(300):
        out.append(([rng.uniform(0.1, 100) for _ in range(N)],
                    [rng.uniform(0.1, 100) for _ in range(N)], N, N))
    # 2. extreme magnitude spread -- summation order is visible here
    for _ in range(300):
        out.append(([10.0 ** rng.uniform(-8, 8) for _ in range(N)],
                    [10.0 ** rng.uniform(-8, 8) for _ in range(N)], N, N))
    # 3. asymmetric depths, including 0
    for bd in range(N + 1):
        for ad in range(N + 1):
            out.append(([rng.uniform(0.1, 10) for _ in range(N)],
                        [rng.uniform(0.1, 10) for _ in range(N)], bd, ad))
    # 4. degenerate: all zero volumes
    out.append(([0.0] * N, [0.0] * N, N, N))
    # 5. one side zero -- the closed end of the bound
    out.append(([0.0] * N, [1.0] * N, N, N))
    out.append(([1.0] * N, [0.0] * N, N, N))
    # 6. near-identical sides (numerator is pure cancellation)
    for _ in range(200):
        b = [rng.uniform(1, 100) for _ in range(N)]
        out.append((b, [x * (1 + rng.uniform(-1e-12, 1e-12)) for x in b], N, N))
    # 7. huge level 1 swamping the rest
    for _ in range(100):
        out.append(([1e16] + [rng.uniform(0.1, 10) for _ in range(N - 1)],
                    [1e16] + [rng.uniform(0.1, 10) for _ in range(N - 1)], N, N))
    return out


def run_binary(binary: pathlib.Path,
               all_books: list[tuple[list[float], list[float], int, int]],
               ) -> list[str]:
    lines = []
    for bids, asks, bd, ad in all_books:
        vals = " ".join(x.hex() for x in bids + asks)
        lines.append(f"{bd} {ad} {vals}")
    proc = subprocess.run([str(binary)], input="\n".join(lines) + "\n",
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"parity binary exited {proc.returncode}: {proc.stderr.strip()}")
    return [ln for ln in proc.stdout.split("\n") if ln.strip()]


def compare(all_books, got: list[str]):
    """Return the list of (index, profile, python, cpp) that disagree."""
    expected = len(all_books) * len(ORDER)
    assert len(got) == expected, f"expected {expected} result lines, got {len(got)}"

    mismatches = []
    for i, (bids, asks, bd, ad) in enumerate(all_books):
        for j, prof_name in enumerate(ORDER):
            name, hexval = got[i * len(ORDER) + j].split()
            assert name == prof_name, f"profile order drift: {name} != {prof_name}"
            cpp = float.fromhex(hexval)
            # Python's depth comes from the sequence lengths, so slice to the
            # depth the C++ side was told to use.
            py = weighted_obi_from_volumes(bids[:bd], asks[:ad],
                                           get_profile(prof_name))
            if cpp.hex() != py.hex():
                mismatches.append((i, prof_name, py, cpp))
    return mismatches


# ─────────────────────────────────────────────────────────────────────────────
# walk_book
# ─────────────────────────────────────────────────────────────────────────────

#: prices, volumes, depth, qty, touch. Prices and volumes are always N long; the
#: walk only sees the first `depth` of them, on both sides of the language line.
WalkCase = tuple[list[float], list[float], int, float, float]

#: The fields compared, in the order parity_walk_book.cpp prints them. All are
#: doubles except the last, which is a level count.
WALK_FIELDS = ("vwap", "filled_qty", "notional", "requested_qty", "slippage_bps")


def walk_cases() -> list[WalkCase]:
    """Cases spanning every branch of the walk, and the boundaries between them.

    The interesting ones are not the ordinary fills -- those would agree even if
    the loop were written badly. They are the boundaries: a size that lands
    exactly on a cumulative-volume total (where the ``want - taken <= EPS_QTY``
    break decides whether one more level is touched), a book padded with zeros,
    a size larger than the book holds, and volumes at the epsilon itself, which
    is the one place where a mismatch between EPS_QTY and kEpsQty would show.
    """
    rng = random.Random(20260822)
    cases: list[WalkCase] = []

    def add(prices, volumes, depth, qty, touch=None) -> None:
        prices = list(prices) + [0.0] * (N - len(prices))
        volumes = list(volumes) + [0.0] * (N - len(volumes))
        cases.append((prices, volumes, depth, float(qty),
                      float(prices[0] if touch is None else touch)))

    def ladder(up: bool) -> tuple[list[float], list[float]]:
        p = rng.uniform(1.0, 5e4)
        prices = [p]
        for _ in range(N - 1):
            step = 1.0 + rng.uniform(1e-6, 1e-3)
            prices.append(prices[-1] * step if up else prices[-1] / step)
        return prices, [rng.uniform(0.01, 3.0) for _ in range(N)]

    # 1/2. Ordinary asks (ascending) and bids (descending), sizes from inside
    #      level 1 to half again more than the book holds.
    for up in (True, False):
        for _ in range(250):
            prices, volumes = ladder(up)
            add(prices, volumes, N, rng.uniform(0.001, sum(volumes) * 1.5))

    # 3. Sizes landing exactly on a cumulative total. The sum is computed the
    #    way the walk accumulates it, so `want - taken` should hit zero rather
    #    than a residue -- and one level fewer gets touched than for qty+1ULP.
    for _ in range(150):
        prices, volumes = ladder(True)
        k = rng.randrange(1, N + 1)
        total = 0.0
        for v in volumes[:k]:
            total += v
        add(prices, volumes, N, total)
        add(prices, volumes, N, total * (1.0 + 2 ** -52))

    # 4. Zero-padded tails: full depth declared, only k levels real. Exchange
    #    snapshots look like this and treating a pad as depth invents liquidity.
    for k in range(1, N):
        for _ in range(40):
            prices, volumes = ladder(True)
            for i in range(k, N):
                prices[i] = 0.0
                volumes[i] = 0.0
            add(prices, volumes, N, rng.uniform(0.01, sum(volumes[:k]) * 1.4))

    # 5. Non-positive and zero-volume levels interleaved among good ones.
    for _ in range(120):
        prices, volumes = ladder(True)
        bad = rng.randrange(N)
        if rng.random() < 0.5:
            prices[bad] = 0.0 if rng.random() < 0.5 else -prices[bad]
        else:
            volumes[bad] = 0.0
        add(prices, volumes, N, rng.uniform(0.01, sum(volumes)))

    # 6. Extreme magnitudes, where the order of summation becomes visible.
    for _ in range(200):
        prices = sorted(10.0 ** rng.uniform(-6, 8) for _ in range(N))
        volumes = [10.0 ** rng.uniform(-9, 6) for _ in range(N)]
        add(prices, volumes, N, 10.0 ** rng.uniform(-9, 6))

    # 7. One enormous level 1 swamping the rest.
    for _ in range(60):
        prices, volumes = ladder(True)
        volumes[0] = 1e15
        add(prices, volumes, N, rng.uniform(0.01, 10.0))

    # 8. Degenerate sizes, including the epsilon itself and its neighbours: the
    #    only cases that would separate EPS_QTY from kEpsQty if they drifted.
    flat_p = [100.0, 100.5, 101.0, 101.5, 102.0]
    flat_v = [0.5, 0.5, 0.5, 0.5, 0.5]
    for qty in (0.0, -0.0, -1.0, -1e300, EPS_QTY, EPS_QTY * (1 + 2 ** -52),
                EPS_QTY * (1 - 2 ** -52), 1e-15, 1e300, float(N) * 0.5):
        add(flat_p, flat_v, N, qty)
    #    ...and volumes at the epsilon, which the walk must treat as no level.
    add(flat_p, [EPS_QTY] * N, N, 1.0)
    add(flat_p, [EPS_QTY * 2] * N, N, 1.0)

    # 9. Every depth, including 0 (an empty span, not a thin book).
    for depth in range(N + 1):
        add(flat_p, flat_v, depth, 1.2)
        add(flat_p, flat_v, depth, 1e9)

    # 10. Touch prices that are not level 1: zero and negative must disable the
    #     slippage calculation rather than divide by them, and a far-away touch
    #     exercises the branch where the walk beat the reference price.
    for touch in (0.0, -0.0, -100.0, 1e-300, 50.0, 1e6):
        add(flat_p, flat_v, N, 1.2, touch)

    return cases


def run_walk_binary(binary: pathlib.Path, cases: list[WalkCase]) -> list[str]:
    lines = []
    for prices, volumes, depth, qty, touch in cases:
        vals = " ".join(x.hex() for x in prices + volumes)
        lines.append(f"{depth} {qty.hex()} {touch.hex()} {vals}")
    proc = subprocess.run([str(binary)], input="\n".join(lines) + "\n",
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"walk parity binary exited {proc.returncode}: {proc.stderr.strip()}")
    return [ln for ln in proc.stdout.split("\n") if ln.strip()]


def compare_walk(cases: list[WalkCase], got: list[str]):
    """Return (mismatches, python_results). Mismatches are (index, field, py, cpp)."""
    assert len(got) == len(cases), \
        f"expected {len(cases)} result lines, got {len(got)}"

    mismatches = []
    results = []
    for i, (prices, volumes, depth, qty, touch) in enumerate(cases):
        py = walk_book(prices[:depth], volumes[:depth], qty)
        results.append(py)
        parts = got[i].split()
        assert len(parts) == len(WALK_FIELDS) + 1, f"malformed line: {got[i]!r}"

        cpp_vals = [float.fromhex(p) for p in parts[:-1]]
        py_vals = [py.vwap, py.filled_qty, py.notional, py.requested_qty,
                   py.slippage_bps(touch)]
        for field, pv, cv in zip(WALK_FIELDS, py_vals, cpp_vals):
            if pv.hex() != cv.hex():
                mismatches.append((i, field, pv, cv))
        if int(parts[-1]) != py.levels_consumed:
            mismatches.append((i, "levels_consumed",
                               float(py.levels_consumed), float(parts[-1])))
    return mismatches, results


def check_walk(compiler: str, flags: list[str], contrast: list[str],
               tmpdir: pathlib.Path, do_contrast: bool) -> bool:
    """Build, run and compare the walk_book harness. True if bit-exact."""
    cases = walk_cases()
    primary = tmpdir / "walk_off"
    build(compiler, flags, primary, SRC_WALK)
    mismatches, results = compare_walk(cases, run_walk_binary(primary, cases))
    values = len(cases) * (len(WALK_FIELDS) + 1)

    print("\nwalk_book vs src/friction.py")
    print(f"cases:    {len(cases)}")
    print(f"compared: {values} values ({len(WALK_FIELDS) + 1} per case), bit-for-bit")

    if mismatches:
        print(f"\nMISMATCHES: {len(mismatches)}")
        for i, field, py, cpp in mismatches[:10]:
            prices, volumes, depth, qty, touch = cases[i]
            print(f"  case {i} field {field}  (depth={depth} qty={qty!r})")
            print(f"    prices  = {prices[:depth]}")
            print(f"    volumes = {volumes[:depth]}")
            print(f"    python  = {py!r}  {py.hex()}")
            print(f"    c++     = {cpp!r}  {cpp.hex()}")
            print(f"    diff    = {cpp - py!r}")
        return False

    print("OK -- C++ and Python agree on every field, exactly")

    # Agreement over easy cases would be worth little, so state what the cases
    # actually reached. Each of these counts must be non-zero for the check
    # above to mean what it says.
    deep = sum(1 for r in results if r.levels_consumed >= 3)
    partial = sum(1 for r in results if r.partial)
    empty = sum(1 for r in results if r.filled_qty == 0.0)
    worse = sum(1 for i, r in enumerate(results)
                if r.filled_qty > 0.0 and r.slippage_bps(cases[i][4]) > 0.0)
    for label, count in (("walks of 3+ levels", deep), ("partial fills", partial),
                         ("zero fills", empty), ("fills worse than touch", worse)):
        assert count > 0, (
            f"no case produced {label}, so parity over this set says nothing "
            f"about that branch")
    print(f"   (coverage: {deep} walks of 3+ levels, {partial} partial fills, "
          f"{empty} zero fills, {worse} filled worse than the touch)")

    if do_contrast:
        print(f"\ncontrast build: {' '.join(contrast)}")
        fast = tmpdir / "walk_fast"
        build(compiler, contrast, fast, SRC_WALK)
        fast_bad, _ = compare_walk(cases, run_walk_binary(fast, cases))
        if fast_bad:
            # Per field, because the fields are not equally affected and a single
            # worst-case number would misrepresent both of them: the walk's own
            # arithmetic moves by ~1 ULP, and slippage_bps then amplifies that by
            # five orders of magnitude. Absolute differences are useless here --
            # notional spans some 23 decades across these cases.
            per_field: dict[str, list] = {}
            for i, field, py, cpp in fast_bad:
                rel = abs(cpp - py) / abs(py) if py != 0.0 else float("inf")
                slot = per_field.setdefault(field, [0, 0.0, i])
                slot[0] += 1
                if rel > slot[1]:
                    slot[1], slot[2] = rel, i
            print(f"   {len(fast_bad)} / {values} values diverge")
            for field, (count, rel, idx) in sorted(per_field.items()):
                print(f"     {field:<14} {count:>4} values, worst "
                      f"{rel / 2 ** -52:>9,.0f} ULP ({rel:.2e} relative)")
            print("   vwap and notional: `notional += take * price` is a "
                  "multiply-accumulate, so FMA")
            print("   fuses it and rounds once where Python rounds twice. Same "
                  "mechanism as the OBI case.")
            if "slippage_bps" in per_field:
                idx = per_field["slippage_bps"][2]
                r = results[idx]
                touch = cases[idx][4]
                gap = abs(r.vwap - touch)
                print(f"   slippage_bps is that error amplified, not a second one: "
                      f"it subtracts two")
                print(f"   nearly equal numbers. On case {idx}, |vwap - touch| is "
                      f"{gap / abs(r.vwap):.1e} of vwap,")
                print(f"   so one ULP of vwap arrives ~{abs(r.vwap) / gap:,.0f}x "
                      f"larger in the bps figure.")
                print(f"   Worth knowing independently of parity: a tolerance on "
                      f"slippage_bps has to be")
                print(f"   scaled by vwap, not by the bps value.")
        else:
            print(f"   0 / {values} diverge -- this compiler does not contract the "
                  "walk's multiply-accumulate.")

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-contrast", action="store_true",
                    help="skip the informational -ffp-contract=fast rebuild")
    ap.add_argument("--march-native", action="store_true",
                    help="add -march=native, matching the shipped CMake flags")
    ap.add_argument("--only", choices=("obi", "walk"), default="",
                    help="run just one of the two parity checks")
    args = ap.parse_args()

    for src in (SRC, SRC_WALK):
        if not src.exists():
            print(f"missing {src}", file=sys.stderr)
            return 3
    compiler = find_compiler()
    if compiler is None:
        print("no C++ compiler found (tried g++, clang++, c++)", file=sys.stderr)
        return 3

    flags = list(PROJECT_FLAGS)
    contrast = list(CONTRAST_FLAGS)
    if args.march_native:
        flags.append("-march=native")
        contrast.append("-march=native")

    all_books = books()
    assert set(ORDER) == set(PROFILES), (
        f"ORDER is stale: {sorted(ORDER)} != {sorted(PROFILES)}. A profile was "
        "added to src/obi_weights.py without being added to the C++ harness, so "
        "it is going unchecked."
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        print(f"compiler: {compiler}")
        print(f"flags:    {' '.join(flags)}")

        ok = True
        if args.only != "walk":
            ok = check_obi(compiler, flags, contrast, tmpdir, all_books,
                           not args.no_contrast) and ok
        if args.only != "obi":
            ok = check_walk(compiler, flags, contrast, tmpdir,
                            not args.no_contrast) and ok

    return 0 if ok else 1


def check_obi(compiler: str, flags: list[str], contrast: list[str],
              tmpdir: pathlib.Path, all_books, do_contrast: bool) -> bool:
    """Build, run and compare the weighted-OBI harness. True if bit-exact."""
    print("\nweighted OBI vs src/obi_weights.py")
    primary = tmpdir / "parity_off"
    build(compiler, flags, primary)
    mismatches = compare(all_books, run_binary(primary, all_books))

    print(f"books:    {len(all_books)}")
    print(f"profiles: {len(ORDER)} ({', '.join(ORDER)})")
    print(f"compared: {len(all_books) * len(ORDER)} values, bit-for-bit")

    if mismatches:
        print(f"\nMISMATCHES: {len(mismatches)}")
        for i, prof, py, cpp in mismatches[:10]:
            print(f"  book {i} profile {prof}")
            print(f"    python = {py!r}  {py.hex()}")
            print(f"    c++    = {cpp!r}  {cpp.hex()}")
            print(f"    diff   = {cpp - py!r}")
        return False

    print("OK -- C++ and Python agree on every value, exactly")

    # The comparison must be capable of failing: if the four profiles
    # produced the same number, "agreement" would be vacuous.
    probe_b = [100.0, 1.0, 1.0, 1.0, 1.0]
    probe_a = [1.0, 80.0, 80.0, 80.0, 80.0]
    vals = {p: weighted_obi_from_volumes(probe_b, probe_a, get_profile(p))
            for p in ORDER}
    assert len(set(vals.values())) == len(ORDER), (
        f"profiles are not distinguishable on the probe book: {vals}")
    print("   (profiles verified distinguishable: "
          + ", ".join(f"{k}={v:+.4f}" for k, v in vals.items()) + ")")

    if do_contrast:
        print(f"\ncontrast build: {' '.join(contrast)}")
        fast = tmpdir / "parity_fast"
        build(compiler, contrast, fast)
        fast_bad = compare(all_books, run_binary(fast, all_books))
        total = len(all_books) * len(ORDER)
        if fast_bad:
            by_profile: dict[str, int] = {}
            worst = 0.0
            for _, prof, py, cpp in fast_bad:
                by_profile[prof] = by_profile.get(prof, 0) + 1
                worst = max(worst, abs(cpp - py))
            summary = ", ".join(f"{k}={v}" for k, v in sorted(by_profile.items()))
            print(f"   {len(fast_bad)} / {total} values diverge ({summary})")
            print(f"   largest absolute difference: {worst:.3e}")
            print("   This is the FMA contraction documented in "
                  "cpp_engine/CMakeLists.txt -- not a bug in either side.")
        else:
            print(f"   0 / {total} diverge -- this compiler does not contract "
                  "the expression, so -ffp-contract=off is a no-op here.")

    return True


if __name__ == "__main__":
    sys.exit(main())
