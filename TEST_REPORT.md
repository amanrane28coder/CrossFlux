# HFT-trader — Test Report

**Date:** 2026-08-21
**Scope:** Python test suite, C++ engine tests, C++ benchmark, and the backtester that produces the README's headline numbers.

> **Status update (same day, after the fee / weighted-OBI / friction work).**
> §1 has been rewritten to the current state: the suite is green and has grown
> from 74 to 134 tests. §2.1 has been **rewritten** — the entry-gate tautology is
> broken and the section now carries the post-fix numbers, including a capacity
> peak. §2.2 is resolved: `src/fees.py` is the single fee source of truth, with
> named presets. §2.3 (the USDT/USD basis) and §2.4 (the OBI signal contributing
> nothing) are **still open**, and §2.3 is now the highest-value item on the list
> precisely because §2.1 no longer hides it. §4 is left as written.

---

## 1. Test results

| Suite | Result |
|---|---|
| `tests/` (11 test files, Python) | **158 passed, 0 failed, 0 skipped** (includes the 9 in `test_bindings.py` below) |
| — of which `tests/test_friction.py` | 32 (new — latency presets, VWAP book walk) |
| — of which `tests/test_signals.py` | 26 (weighted-OBI profiles) |
| — of which `tests/test_execution_simulator.py` | 21 (was 7) |
| — of which `tests/test_engine_friction.py` | 19 (new — pending buffer, adverse fills, legging) |
| — of which `tests/test_demo_feed_schema.py` | 9 (new — the demo seeder's schema against the C++ writer's, parsed from the .cpp) |
| — of which `tests/test_live_ingestion_accounting.py` | 9 (new — rejections are not fills, adverse fills are not wins) |
| `tests/test_bindings.py` | **9 passed** — but against a stub that mirrors `bindings.cpp`, not the compiled `.so`, which is macOS-only (see the appendix). It skips cleanly when the module is absent instead of erroring at collection. |
| `cpp_engine/tests/test_signals.cpp` | **56 / 56 passed** (was 12; weighted-OBI cases t9–t16 added) |
| `cpp_engine/tests/test_friction.cpp` | **251 / 251 checks passed** (47 tests — new: VWAP walk, latency presets, pending queue, the async executor end to end) |
| `cpp_engine/tests/run_parity_check.py` | **3,756 / 3,756 values bit-identical** to `src/obi_weights.py`; **8,220 / 8,220** bit-identical to `src/friction.py`'s `walk_book` |
| `cpp_engine/tests/run_mutation_check.py` | **9 / 9 mutations caught**, both negative controls survived |
| `tests/run_execution_mutations.py` | **10 / 10 mutations caught**, 2 controls survived — each reinstated rejection in `simulate_cross_venue_fill` fails the suite |
| `tests/run_live_ingestion_mutations.py` | **9 / 9 mutations caught**, 2 controls survived — including the inverse error of *excluding* adverse fills from the counters |
| `backtest/friction_report.py` generator | **12 / 12 mutations caught**, control passed |
| `backtest/check_report_numbers.py` | every sweep figure in this report, `src/friction.py` and `backtest/engine.py` re-derived from the JSON — 17 table rows, 12 prose claims, 9 retracted-figure mentions; **14 / 14 mutations caught**, 3 controls survived |
| `cpp_engine/tests/bench_predictor.cpp` | runs; **still measures the wrong code path** (see §3) |
| `cpp_engine/tests/test_network.cpp` | not runnable — requires Boost.Beast |

### What changed since the audit

**The 3 stale failures are fixed.** `test_simulate_cross_venue_action_0/1` and
`test_simulate_multilevel_slippage` failed with `TypeError: cannot unpack
non-iterable ExecutionReport object`. They were stale in two independent ways:
the return type had changed to a single `ExecutionReport`, *and* their 4–6 bps
fixtures could not survive the round-trip fee plus the new latency squeeze. Both
are addressed — `test_execution_simulator.py` went from 7 tests to 15, the old
4 bps fixture is kept deliberately as an explicit rejection case, and a
`fixed_latency()` context manager sets `LATENCY_STD_MS = 0.0` so the tests no
longer depend on an unseeded `random.gauss`.

**A real build break was found and fixed.** All three `CMakeLists.txt` specified
C++17 while `signals.hpp`, `obi_config.hpp` and `predictor.hpp` require C++20
(`std::span`, `[[likely]]`). The build failed with *"'std::span' has not been
declared"*. Raised to 20 in all three.

**A reproducibility defect was found and fixed.** Under GCC's default
`-ffp-contract=fast`, `num += w * (b - a)` in `calculate_weighted_obi` is fused
into an FMA — one rounding where Python does two. 339 of 3,756 values differed by
~1 ULP, all on `decay_75`, the only preset whose weights are not powers of two.
`-ffp-contract=off` is now set on every target, and
`cpp_engine/tests/run_parity_check.py` reproduces both halves of the measurement
on demand. The gap could never change a trade decision (~1e-16 against a 0.3
threshold), but it meant the backtest and the live engine computed different
numbers from identical input.

**A documented bound was wrong.** The weighted OBI was documented as living in
the *open* interval (-1, 1). It is closed: an empty book side gives exactly ±1,
and so does a side below the other's ULP (`1e9 + 1e-9 == 1e9`). Found by a C++
test written to match the docstring, which failed. Docs corrected in
`signals.hpp` and `src/obi_weights.py`; both suites now assert the closed bound
explicitly.

**The new tests were checked for vacuity, and one blind spot turned up.**
`cpp_engine/tests/run_mutation_check.py` corrupts `calculate_weighted_obi` five
ways — sign flip, drop the weights, unbound the result, and both halves of the
three-way loop bound — and confirms the suite notices each. Four were caught
immediately. The fifth was not: replacing the `weights.size()` term of the loop
bound changed nothing observable, because **every shipped `Profile` zero-pads its
weight array to `kMaxLevels`**, so a read past the intended end lands on 0.0. No
preset-driven test could ever falsify that bound. `t16` now covers it with a
deliberately short `span` over a densely populated array, and the mutation is
caught by value.

Two things about that check are worth stating, because both had me draw the
wrong conclusion first:

- **A crash is a catch, but it hides the rest of the run.** With the loop bound
  removed, `t13`'s empty-`span` case dereferences a null pointer and the process
  dies on SIGSEGV before reaching `t16`. An earlier version of the harness
  classified only "printed FAILED" as caught and reported this mutation as
  surviving. The suite now takes a test-name filter (`./test_signals t16`) so the
  harness can rerun a single hidden test and get its verdict.
- **The implementation's own asserts can mask the tests.** Unbounding the signal
  aborts on `calculate_obi_delta`'s `assert(obi >= -1.0 && obi <= 1.0)` in a
  Debug build, before any test reports. Each mutant is therefore built twice, and
  the `-DNDEBUG` run is the one that answers "do the *tests* check this?"

The harness carries a negative control — a genuinely no-op edit (`i < depth` →
`i != depth`) that must survive — so a run that reported "caught" for everything
would be recognisable as broken rather than perfect.

**The same treatment was applied to the C++ friction path, and the mutation that
matters most is the one that puts the bug back.** `run_mutation_check.py` now
targets two suites and can corrupt sources as well as headers. Four edits attack
`friction.hpp` and `execution_manager.cpp`: fill the whole order at the touch
instead of walking the book (`F1`, 25 checks fail), drop the drain's fill-time
ordering so legs price in submission order (`F2`, 2 fail), hedge on the larger leg
instead of the smaller (`F3`, 2 fail), and **reinstate the profitability gate at
fill time** (`F4`, 8 fail). `F4` is a two-token edit — `o.filled = true` becomes
`o.filled = (o.net_pnl >= 0.0)` — and it is the entire original defect: keep the
winners, drop the losers, report a perfect win rate. `e3` and `e5` fail loudly on
it. A second no-op control (`F0`, reordered operands of a side-effect-free `||`)
guards the new suite the way `M0` guards the old one.

**`walk_book` is now checked bit-for-bit against Python too, and that turned up a
conditioning result worth recording.** `parity_walk_book.cpp` feeds 1,370 cases
through both implementations as hex doubles and compares six fields each — vwap,
filled quantity, notional, requested quantity, slippage and levels consumed —
including the boundaries that ordinary fills never reach: a size landing exactly
on a cumulative volume total, zero-padded books, negative-price levels, and
volumes at `EPS_QTY` itself, which is the one place a drift between `EPS_QTY` and
`kEpsQty` would show. All 8,220 values agree exactly under the shipped flags.
Rebuilt with `-ffp-contract=fast`, 647 diverge — and unlike the OBI case the
spread is not uniform: **vwap and notional move by 1–2 ULP, but `slippage_bps`
moves by up to 145,159 ULP.** That is not a second bug. `slippage_bps` subtracts
two nearly equal numbers (`vwap - touch`), and on the worst case the gap is
6.5e-6 of the vwap, so a 1-ULP error arrives ~150,000× larger. The practical
consequence outlives the parity question: any tolerance on a slippage figure has
to be scaled by the *price*, not by the bps value, or it is meaningless.

**The entry-gate tautology is broken, and the numbers below are the new ones.**
This is the largest change since the audit and it rewrites §2.1. Signal and fill
no longer read the same snapshot: `src/friction.py` holds latency presets and a
shared VWAP book walk, `backtest/engine.py` buffers each signal and fills each
leg against the prevailing quote on its own venue at `T + latency`, and a trade
that turns against us is booked as a loss rather than rejected. Test count went
from 115 to **134** (19 new in `tests/test_engine_friction.py`). The result is a
capacity curve with an interior maximum, which is the first thing this backtester
has produced that it could not have produced by construction.

Two claims of mine were disproved by measurement in the course of that work, and
both are corrected in §2.1 rather than quietly dropped. I predicted
`friction="zero"` would restore a tautological 100% win rate; it gives 99.3% at
0.01 BTC and 91.4% at 3.0, because removing latency does not remove the book
walk. And an earlier throwaway script's curve (93.3% / 59.8% win rates, 54.96%
unfilled) is **wrong** — it read only the first 400,000 rows of each file, which
truncates the two venues to different lengths and leaves a frozen book across the
final 45% of its sample. `backtest/check_report_numbers.py` now re-derives every
figure in §2.1 from `backtest/sweep_results.json` and fails if the prose drifts;
it caught four wrong adverse-fill counts the first time it ran, in text I had
just written.

**That guard was too narrow, and the narrowness cost something.** It read this
report and nothing else, so while §2.1 struck the truncated-script curve, the
same numbers went on being asserted in `src/friction.py` — in the module
docstring, in the `stress` preset's own `description`, and in a `zero` preset
description claiming it "reproduces the tautological 100% win rate" when the
measurement says 99.3%. Anyone reading the module rather than the report would
have believed all of it. The checker now covers `src/friction.py` and
`backtest/engine.py` as well: 17 table rows and 12 prose claims across three
files, located by literal markers rather than line numbers, and compared on
numeric tokens so re-padding a column is allowed but changing a value is not.

The checker is itself checked, by `backtest/check_report_numbers_mutations.py`,
and that was not ceremony — it found a hole. A guard that passes by finding
nothing looks identical, from the outside, to a guard that is broken. Fourteen
perturbations of a throwaway copy of the tree (a wrong cell, a deleted peak row,
a row for a size never swept, a renamed marker, a rerun of the JSON) must each be
caught, and three controls must survive. The thirteenth attempt re-asserted one
of the struck win rates as fact, two paragraphs below the sentence that retracts
it, and **the checker passed** — its scan looked 600 characters either side for
the word "wrong", so the re-assertion was sheltered by the retraction it
contradicted. The scan is now scoped to the sentence containing the figure.
Proximity to a retraction is not a retraction.

---

## 2. The headline results do not hold up

This is the most important finding. The README advertises a
"mathematically validated **+6.77% net return** and a **100.0% win-rate**" over
31,528 trades. Four independent problems, each verified:

### 2.1 The win rate was a tautology — now fixed

The entry gate (`backtest/engine.py:915-917`):

```python
margin_b_a = aligned["best_bid_a"] - aligned["best_ask_b"]
fee_b_a    = (aligned["best_bid_a"] + aligned["best_ask_b"]) * TAKER_FEE_RATE
cond_b_a   = (aligned["obi_delta"] > delta_threshold) & (margin_b_a > fee_b_a)
```

The PnL (`:1028-1029`):

```python
fee     = TAKER_FEE_RATE * self.qty * (buy_price + sell_price)
pnl_net = (sell_price - buy_price) * self.qty - fee
```

`pnl_net > 0` reduces to `(sell − buy) > T·(buy + sell)` — the same inequality as
the entry filter. **A trade is admitted if and only if it is profitable.** The
win rate therefore measures the gate arithmetic, not predictive skill. Measured
actual rate is 99.0% (not 100.0%): the ±5 bps slippage tolerance at `:202-203`
admits a small band of near-zero losers, worst fill −$0.25.

That tolerance deserves its own sentence, because it was the tautology's real
protector rather than a stray rounding allowance. It rejected the *whole order*
whenever any level came in more than 5 bps worse than the signal price, and
booked `pnl_net = 0.0`. Favourable fills were kept and unfavourable ones
discarded, so the ledger held only the good tail.

#### Fixed — and the fix produces a capacity limit

Signal and execution are now decoupled by two mechanisms. A signal at `T` is
buffered and each leg fills against the *prevailing* quote on its own venue at
`T + latency` (last update at or before that time — `searchsorted(..., "right")
− 1`; dropping the `− 1` would read the next quote, which is look-ahead). The
fill price is the VWAP of the levels the order actually consumes, not the touch.
A trade that turns against us inside the window is **booked as a loss**, counted
in `adverse_selection_fills`. The only rejection left in the path is a venue with
no quote at all.

Swept across order size on the same day, institutional fees, `stress` preset
(100 ms/leg), 307,881 signals, both directions:

| BTC/leg | win rate | adverse fills | unfilled | mean edge | net PnL |
|---|---|---|---|---|---|
| 0.01 | 98.6% | 4,329 | 0.01% | 4.25 bps | $0.08M |
| 0.10 | 97.7% | 6,974 | 0.08% | 4.19 bps | $0.80M |
| 0.25 | 97.1% | 8,930 | 0.22% | 4.14 bps | $1.98M |
| 1.00 | 95.0% | 15,547 | 1.14% | 4.00 bps | $7.57M |
| 3.00 | 85.3% | 45,269 | 7.25% | 3.40 bps | **$18.12M** |
| 5.00 | 68.8% | 95,937 | 20.14% | 2.22 bps | $16.96M |

Three things there could not have happened before. Losing trades exist at every
size, so the strategy is falsifiable. Mean edge decays with size, which is
slippage being paid rather than assumed. And **net PnL peaks at 3.0 BTC** and
falls at 5.0 — a capacity limit. The old engine had none; doubling size doubled
PnL exactly.

`friction="zero"` does *not* restore 100% (99.3% at 0.01 BTC, 91.4% at 3.0). I
predicted it would and was wrong: removing latency removes the delay but not the
book walk, and size alone loses money because the gate quotes the touch while
the fill pays the VWAP of five levels. Recovering a true 100% would need zero
latency *and* an infinitely deep top level — i.e. it would need the tautology
back.

Latency on its own barely bites at small size, and that is the feed, not a bug:
`binance_book_snapshot_5` updates on a ~100 ms cadence, so a 100 ms delay
advances the book about one row and the touch price changes on only 14.0% of
binance rows and 8.4% of kraken rows (mean |Δask| ≈ $0.45). Size is what bites.

Three caveats bound the result. Above ~0.25 BTC the unfilled fraction measures
the *depth of the file* rather than illiquidity — these are five-level snapshots,
so 7.25% at 3.0 BTC (and 20.14% at 5.0 BTC) is an upper bound on the real
shortfall and the PnL peak is a lower bound on real capacity. Total PnL stays
positive because of the §2.3 basis, not because the tautology survived: 99.7% of
signals point one way and the ~4 bps mean edge sits on the +4.79 bps basis. And
the gate selects deep books — on signal rows the buy leg absorbs a full 3.0 BTC
87.4% of the time against ~55% of rows unconditionally — so these numbers are
specific to this gate.

Verification: the fill path is bit-exact against an independent scalar
reimplementation (separate book walk, separate bisect), worst relative deviation
**2.2 × 10⁻¹⁵** across buy price, sell price, matched quantity, fee, legging cost,
net PnL and fill timestamp over 10,000 trades at two sizes. Wiring is covered by
19 tests in `tests/test_engine_friction.py`; the page at
`backtest/friction_report.html` is generated from the sweep JSON so it cannot
drift from the run, and its generator is checked by 12 mutants that must each be
caught.

One earlier version of this curve — 93.3% and 59.8% win rates, 54.96% unfilled,
from a throwaway `measure_tautology_break.py` — **is wrong and has been struck.**
It read only the first 400,000 rows of each file, which truncates binance at
13.86 h but kraken at 7.58 h, and its `ffill` then held a single frozen kraken
snapshot across the final 45% of its sample. Every fill in that tail priced
against a stale book.

### 2.2 A single fee constant flips the sign of the return

`engine.py:69` uses a flat 5 bps/leg (10 bps round-trip). But
`src/execution_simulator.py:7-9` **and** `cpp_engine/src/execution_manager.cpp:21-26`
both use Binance 4 bps + Kraken 10 bps = 14 bps round-trip. Mean gross edge on
the winning trades is **13.51 bps**. Re-pricing the identical 31,501 fills:

| fee model | fees | net PnL | return | win rate |
|---|---|---|---|---|
| `engine.py` — 10 bps RT | $19,697 | **+$6,900** | +6.90% | 99.0% |
| `execution_simulator.py` — 14 bps RT | $27,585 | **−$987** | **−0.99%** | 41.4% |

Three mutually inconsistent fee models coexist in the repo, and the only one
that produces the advertised number is the one used nowhere else. The strategy
sits almost exactly on top of its true cost basis.

### 2.3 This is a USDT/USD basis trade, not cross-venue arbitrage

The data is Binance **BTCUSDT** vs Kraken **XBT/USD** — different quote assets.
Computed directly from the raw CSVs (2,249,379 aligned ticks):

- Kraken's mid sits **+4.79 bps above** Binance's on average (median +4.10 bps)
- Kraken is the richer venue on **87.2%** of ticks

That is a persistent quote-currency basis, not a transient dislocation.
Consequences visible in the output: **all 31,528 trades are `BUY_A_SELL_B`**, and
the reverse-direction condition `cond_b_a` (`:917`) fires **0 times in 1.99M
ticks**. The realized PnL is an unhedged short-USDT / long-USD FX exposure
booked as risk-free arbitrage.

### 2.4 The OBI signal contributes nothing

Dropping the OBI gate entirely and trading every fee-clearing tick gives
**288,999 trades at 3.14 bps mean net edge**, versus **3.44 bps for the
OBI-gated 31,528**. Naive total PnL $56,710 vs $6,770. `delta_threshold` is
acting as a throughput throttle, not a predictor — which undercuts README:40
("visually confirming the alpha signal's predictive validity"): the scatter plot
is confirming a correlation the harness imposes.

---

## 3. The benchmark measures the wrong path

`cpp_engine/tests/bench_predictor.cpp:11-12` promises "~50% Gate 1 pass rate".
As committed it reports:

```
Signals emitted   : 0  (0.0% pass rate)
Per-tick latency  : 12.0 ns
```

`make_snap` (`:62-63`) hardcodes `49999.0 / 50001.0` for **both** venues, so
Gate 3 — the spread filter, default `min_spread_pct = 0.0012` at
`predictor.hpp:94`, applied at `:196-199` — computes
`spread_pct = (49999 − 50001)/50001 = −4.0e-5` and rejects every tick that
cleared Gates 1 and 2. Nothing is ever emitted.

Giving Kraken a real price gap restores the documented behaviour and raises the
cost by ~47%:

| fixture | signals | per-tick |
|---|---|---|
| as committed | 0 (0.0%) | 12.0 ns |
| with a real cross-venue gap | 50,000 (50.0%) | 17.6 ns |

So the README's "66ns Evaluation Core" is derived from a benchmark that never
emits a signal. (Nuance: both OBI computations and Gates 1–2 *do* execute; what
never runs is signal construction and vector growth. The conclusion stands —
it isn't timing full evaluation.) Fix is one line in the fixture.

---

## 4. Other real bugs

**Failed trades are free, and invisible.** The rejection path (`:993-1021`)
appends the trade with `fee = 0.0`, `pnl_net = 0.0`, `status = "rejected"`.
`win_rate` (`:251-257`) only counts `filled_trades`, so latency-destroyed trades
leave the denominator entirely; `_compute_equity_curve` (`:620-632`) sums all
trades but they contribute exactly 0.0. A trade where one leg filled and the
other did not leaves a real unhedged position — **legging risk is modeled as
costless**.

**The advertised out-of-sample split never executes.**
`_run_real_data_vectorized` defaults `train_frac = 1.0` (`:832`) and `run()`
calls it with no argument (`:1099`). No caller anywhere passes `train_frac`, so
the label at `:974` is always `"train"` and `split_summary()` always prints
`TEST : (no trades)`. The docstring at `:838-840` defers OOS validation to
"the caller's responsibility — see `run()`", which never does it.

**Adverse-selection accounting is biased the wrong way.** `:202-205` rejects a
fill only when price moved *against* us past tolerance; favorable moves are
always accepted. Conditioning on `status == "filled"` truncates only the adverse
tail, so the reported cost is **−$142.51 (−53.9 bps)** — i.e. the market
supposedly moved *in our favour* on average across 31,501 arbitrage fills. This
contradicts the module's own docstring at `:99-100` ("should expect
adverse_selection_cost > 0 in aggregate") yet `summary()` prints it as validated.

**`GlobalKillSwitch` trips on the first loss.** `src/risk_manager.py:87-106`
measures drawdown against peak *PnL*, not capital. Verified: a single −$0.01
first trade gives `dd = 100.0%, tripped = True`; `+$1.00` then `−$0.10` gives
`dd = 10.0%, tripped = True` while cumulative PnL is still **+$0.90**. A 5% kill
switch halts the system almost immediately.

**Look-ahead in the data alignment.** `src/ingestion.py:124` keys off the Tardis
`timestamp` (exchange clock) rather than `local_timestamp` (receipt). In row 1 of
the Kraken file these differ by 63 ms, so signals can be acted on before the
data was receivable. Compounding it, the `ffill` as-of join (`engine.py:900`)
leaves quotes stale — Kraken staleness p90 = 251 ms, p99 = 1,387 ms, max
9,988 ms — so many "cross-venue margins" never existed simultaneously.

**Two different quantities both reported as "Sharpe."** `_annualised_sharpe`
(`:642-670`) does proper time-based scaling → 30.69. But `split_summary`
(`:320`) and `regime_decomposition` (`:288`) compute `mean/std*sqrt(n_trades)`,
a t-statistic → prints **+285.45** in the same report.

**Incorrect docstring values in `src/latency_model.py`.** The worked examples
claim `(50.0, 3.5, 0.3) → 0.9297` and `(50.0, 3.5, 2.0) → 0.5791`; actual values
are **0.91519** and **0.58161**. (The function itself is correct — cross-checked
against the independent C++ implementation, which agrees to 4e-9.)

**Minor.** `engine.py:351` labels real-CSV runs "24.0h *synthetic*".
`predictor.hpp:22-30, 100-105` document only Gates 1–2, omitting Gate 3.
`engine.py:337` computes `realized_total` and never uses it. `live_trading.cpp:59`
uses `delta_threshold = 0.10` and `min_spread_pct = 0.0005` against the
backtest's 0.65 / 0.0012 — the live path is a different, unbacktested strategy.
README:48 claims "Python >= 3.7" but the only compiled artifact is
`cpython-314-darwin`. `data/raw/` contains a 91 MB `.unconfirmed` partial
download that never completed.

---

## 5. Suggested order of work

1. ~~**Settle on one fee model**~~ — **done.** `src/fees.py` holds named presets
   (institutional 3.0 bps round-trip, retail 14.0, zero, legacy_flat 10.0) and
   both the backtester and `execution_simulator.py` read from it.
2. ~~**Break the entry-gate tautology (§2.1)**~~ — **done** for the Python
   backtester. Latency buffer, VWAP book walk and booked adverse fills are in
   `src/friction.py` + `backtest/engine.py`, covered by 19 tests and verified
   bit-exact against an independent reimplementation. The result is a real
   capacity curve peaking at 3.0 BTC. Still open: the same two mechanisms in the
   C++ path (`cpp_engine/include/friction.hpp` has the book walk; the pending
   queue and the wiring into `execution_manager.hpp` are not written), plus the
   C++/Python parity check its docstring already promises.
3. **Fix the quote mismatch** — use BTCUSDT on both venues, or explicitly hedge
   and report the USDT/USD basis as a separate P&L line. **This is now the
   highest-value open item**, because §2.1 no longer hides it: with the tautology
   gone, ~4 bps of the surviving edge is visibly the +4.79 bps basis, and 99.7%
   of signals point one way. Until the basis is hedged or reported separately,
   the strategy's headline return is a currency position wearing an arbitrage's
   clothes.
4. ~~**Charge rejected legs a real cost**~~ — **done.** Adverse fills are booked
   at their loss instead of `pnl_net = 0.0`, and a partially filled pair charges
   the unhedged residual at the preset's legging cost while paying fees on what
   each leg actually filled.
5. **Make `run()` actually pass `train_frac < 1.0`** so the OOS split runs.
6. ~~**Repair the 3 stale tests**~~ — **done**, see §1.
7. **Fix the benchmark fixture** so it times the path it claims to.
8. **Fix `GlobalKillSwitch`** to measure drawdown against capital.
9. **Get the five-level ceiling out of the way** — the 91 MB L2 incremental feed
   is the shared blocker for both a real slippage curve above 0.25 BTC and true
   multi-level OFI.

The tautology is gone, so the backtest can now be wrong, which is the
prerequisite for it being informative. What it currently measures honestly is how
much of a *known basis* a given order size can collect after latency and
slippage. Item 3 is what stands between that and a claim about arbitrage.

### Known limitations of the execution model

These are simplifications, not open bugs — each is a place where the model
declines to invent a number it has no way to know. They are listed here because
four comments across three files point at this section rather than papering over
the gap, and a limitation nobody can find is indistinguishable from a defect.

**The `max_position` guard is inert on the simulated path.**
`SimulatedExecutor::check_risk_guards` rejects a signal when
`|position_| >= max_position_`, but `position_` is initialised to `0.0` and never
stored to — only `load`ed, at `execution_manager.cpp:204` and through
`current_position()`. So the comparison is `0.0 >= max_position_`, which is false
for any positive limit: the guard cannot fire, `current_position()` always reports
zero, and the `max_position` constructor argument has no effect on behaviour.

This follows from `legging_cost` rather than from an oversight. Charging the
residual at the preset's legging cost models flattening it immediately, so no
exposure carries to the next tick and a running position would be double-counting
a risk already paid for. The honest options were to track a position the model
has already flattened, or to leave the field at zero and say so; a third option —
incrementing `position_` by the fill quantity — would produce a number that looks
like risk tracking while contradicting the cost the same fill just paid. The
guard is therefore dead code *for as long as residuals are flattened on the spot*.
Holding a residual across ticks is a real modelling choice someone may want, and
it is the change that makes this guard live again, so it should not be deleted.

**A one-legged fill is treated as no trade in Python and as a loss in C++.**
`simulate_cross_venue_fill` returns `no_liquidity_at_fill` when *either* leg
filled nothing, booking `net_pnl = 0.0` while `total_fees` still carries the
filled leg's fees. `price_resolved` takes its matching early return only when
hedged *and* residual are both zero, so in C++ the same shape falls through and is
booked as an adverse fill: fees on the leg that filled, the residual charged at
`legging_cost`. The C++ behaviour is the correct one — a one-legged fill leaves a
real position, which is exactly the "legging risk is modeled as costless" defect
in §4.

The divergence is unreachable rather than benign, and the reason is worth
recording because "unreachable" decays. The Python branch needs positive volume
at a non-positive price: an all-zero-volume book takes the earlier
`no_liquidity` branch, and the Tardis feeds contain no such level.
`crossflux::PriceLevel`'s validating two-argument constructor will not construct
one either, so no parity harness can submit the input that would expose the
disagreement (`cpp_engine/tests/parity_walk_book.cpp:54`). Independently, the
synchronous path sizes both legs to the same `used_qty` and walks them together,
so it cannot leg on its own; genuine legging comes from the async path's
independent per-leg latency, and that path charges it. Parity is claimed for
well-formed books only.

**The latency haircut is an assumption, and a deliberately harsh one.** Both
scalar paths move each touch against the order by `latency_ms / 1000 × 0.01` —
1 bp per 10 ms per side. As an unconditional drift that is roughly two orders of
magnitude above BTC's per-second move; it is meant as an adverse-selection
haircut, the move *conditional on* being picked off, and it is always adverse and
never favourable. That is the opposite bias to the one it replaced but a bias all
the same. The async path does not use it: it re-reads the actual prevailing book
at `T + latency` out of the feed, which is why the capacity curve in §2.1 comes
from that path and not from these. Any number quoted from a scalar path inherits
this coefficient — `dashboard/seed_demo_feed.py` shows the size of the effect
directly, booking +$25 at `zero` and −$272 at `stress` over the same 360 ticks of
the same synthetic market.

**The demo feed's edge is a modelled basis, not a measurement.**
`dashboard/seed_demo_feed.py` exists to paint the dashboard when no engine is
running, and it now holds a +4.79 bps mean-reverting basis between its two venues
because that is the figure §2.3 measures. Its skew biases are chosen so most
signals point the way the measured ones do; it does not reproduce the 99.7%
one-sidedness and does not claim to. Every row it writes is synthetic and labelled
as such — status `demo`, profile suffixed `-demo`.

---

## Appendix — how these were run

The Linux sandbox has no pytest, scipy, cmake, Boost, OpenSSL or PyPI access,
and the committed `arbitrage_engine.cpython-314-darwin.so` is a macOS Mach-O
arm64 binary that cannot be imported. The original audit modified no repo files.
Instead:

- A `pytest` shim (the suite uses `pytest.raises`, `parametrize`, `skipif`, one
  `tmp_path` fixture and a module-level `skip`) plus a small discovery runner.
- A `scipy.stats.lognorm.cdf` shim using the exact closed form
  `0.5·(1+erf(ln(x/scale)/(σ√2)))`. Validated against known standard-normal
  values *and* against the C++ engine's independent implementation — both give
  0.84850850 for `(50.0, 3.5, 0.4)`, agreeing to 4e-9.
- C++ tests compiled directly with
  `g++ -std=c++20 -Wall -Wextra -O2 -ffp-contract=off -I cpp_engine/include`,
  no cmake or Boost needed. **C++17 does not work** — that was the build break
  fixed in §1, and the original audit's `-std=c++17` invocation predates the
  headers that need `std::span`.
- The suite was proven non-vacuous with `cpp_engine/tests/run_mutation_check.py`
  (see §1): five mutations, each caught, plus a no-op control that survives to
  show the harness can tell the difference. A shim that silently swallowed
  assertions, or a test that copied its expected value out of the
  implementation, would otherwise report all-green. The shim's `raises` was
  separately verified able to fail.
- The backtester was run by stubbing the `.so`; the real-data path
  (`_run_real_data_vectorized`) is pure pandas and never calls the C++ core.
- `tests/test_bindings.py`'s 9 passes are against that stub, whose signatures and
  defaults were made to match `cpp_engine/src/bindings.cpp` line by line
  (`DEFAULT_DEPTH`, `DEFAULT_DELTA_THRESHOLD`, `DEFAULT_MIN_P_EXECUTE`, the
  `SignalAggregator` keyword defaults, `OrderBookSnapshot.bid_depth`/`ask_depth`,
  and `predictor.hpp:281`'s strict `<` on the spread gate). Six failures during
  that work were stub shortfalls, not repo defects. Every attribute the stub does
  *not* deliberately provide raises instead of answering, which is what caught an
  earlier silent fallthrough to synthetic data. **This is not a substitute for
  running the suite against the real `.so` on macOS.**
- The friction sweep was run with explicit `binance_path=`/`kraken_path=`. A bare
  `Backtester()` silently generates synthetic GBM data, because the default
  filenames do not exist in this repo — and until this session `summary()`
  printed the word "synthetic" unconditionally, so a real run was mislabelled as
  a synthetic one. Both are fixed (`BacktestResult.data_source`), and that
  mislabelling is why the sweep script passes paths explicitly.
- The fill path was cross-checked element-wise by
  `crosscheck_engine_fills.py`, which reimplements the book walk and the as-of
  lookup from the specification rather than from the engine's code.
- The cross-venue basis in §2.3 was computed straight from the raw gzipped CSVs
  with pandas, independent of the project's own code.

Findings in §2 and §4 were produced twice, independently, and cross-checked;
§2.3's basis figures were reproduced from the raw data by a second method.
