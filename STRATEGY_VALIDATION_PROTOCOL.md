# CrossFlux strategy validation protocol

**Status:** proposed protocol; not yet executed.  
**Purpose:** define what evidence is required before describing the cross-venue signal as robust or profitable.

## Why the current result is not enough

The checked-in real-data run covers one UTC day, one BTC market, and two quote currencies (USDT and USD). The existing holdout is chronological, but it is drawn from that same day. A high in-sample or one-day holdout Sharpe cannot establish robustness across market regimes, and the mixed-quote PnL is not a common-currency return series.

Sharpe estimates depend on their sampling uncertainty and serial dependence; square-root annualization is not generally valid when returns are serially correlated ([Lo, *The Statistics of Sharpe Ratios*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=377260)). Selecting the best result after trying many configurations creates another source of optimism. The Probability of Backtest Overfitting paper proposes CSCV as a way to estimate how often in-sample selection ranks poorly out of sample ([Bailey et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)); the Deflated Sharpe Ratio adjusts Sharpe inference for selection bias and non-normality ([Bailey and López de Prado](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)). These measures complement, but do not replace, a final untouched chronological holdout.

Execution quality also depends on arrival-time conditions. A recent empirical study of crypto order execution finds fill failures increase with volatility and latency and decrease with liquidity ([Cucuringu et al., *The good, the bad, and latency*](https://doi.org/10.1080/14697688.2025.2515933)). High-frequency execution research likewise treats fill probability and position constraints as part of strategy PnL, not as a detail to add after evaluating the signal ([Dixon, *A high-frequency trade execution model for supervised learning*](https://doi.org/10.1002/hf2.10016)).

## Data admission rules

1. Use synchronized L2 snapshots and, where available, incremental updates from both venues. Preserve original exchange timestamps, local receive timestamps, sequence numbers/checksums, and source checksums.
2. Obtain a timestamped USDT/USD conversion series for the same period. Convert all cash flows using a predeclared point-in-time rule and report stablecoin basis separately. Missing/stale conversion means the run is labeled mixed-quote sensitivity, not USD performance.
3. Record gaps, clock skew, stale books, crossed/invalid books, venue downtime, and symbol changes. Do not forward-fill a missing book into a fill opportunity.
4. Start with at least 90 calendar days covering materially different volatility and liquidity conditions. This is a project screening threshold, not a universal statistical guarantee; expand the history if the results are regime-sensitive.
5. Save a manifest containing URLs/provider, licenses, filenames, SHA-256 hashes, symbol mappings, time zone, and parser version.

## Freeze before evaluation

Before touching the final holdout, write down and hash:

- signal formula, depth and weighting profile;
- all thresholds and sizing rules;
- fees, rebates, latency distributions, order types, partial-fill behavior, and quote-age rules;
- the complete list of configurations already tried, including failed experiments;
- the data windows assigned to development, validation, and final test.

Use chronological development and validation windows. Purge/embargo observations whose signal, order-arrival, or fill horizon overlaps a split boundary. Use the validation window for model selection only. Open the final chronological holdout once after the configuration and report code are frozen. Estimate CSCV/PBO across the recorded configuration family as a separate selection-risk diagnostic; do not use CSCV to justify repeatedly revising the final holdout.

## Required experiment set

### 1. Execution replay

- Evaluate fills using the book at modeled order-arrival time, not the signal snapshot.
- Use empirical or explicitly stressed latency distributions per venue; report latency percentiles and outcomes by latency bucket.
- Charge each venue's applicable maker/taker fee, book-walking impact, partial-fill residual, and hedging cost. Report unfilled quantity and adverse fills.
- Include a conservative scenario for queue priority and non-fill risk. If historical queue position cannot be reconstructed, disclose that limitation and avoid assuming every displayed quantity is executable.
- Stress quote staleness, one-leg delay, order size, fees, and a temporary venue outage. Include zero-fill and rejected-hedge cases.

### 2. Negative controls

- Reverse the trade direction while keeping the same signal events.
- Shift signals beyond the predeclared alpha lifetime while preserving time-of-day and event clustering.
- Compare against a no-signal baseline and a block-permuted signal baseline that preserves short-range serial structure.

These controls should remove the measured edge. If they retain similar net performance, treat the result as evidence of a data, accounting, or market-basis artifact and investigate before any score increase.

### 3. Statistical reporting

Report per chronological split, venue pair, volatility/liquidity regime, and stress scenario:

- common-currency net PnL and return, with full fee/impact/FX decomposition;
- number of candidate configurations tried;
- HAC-adjusted Sharpe with sampling uncertainty, plus Deflated Sharpe Ratio using the full recorded trial count;
- CSCV/PBO on the development/validation configuration family;
- daily-block-bootstrap confidence interval for mean net return and maximum drawdown;
- win rate, payoff ratio, fill rate, adverse-fill rate, unhedged exposure time, and capacity by order size.

Do not report only the best parameter set or aggregate train-plus-test results. A single aggregate Sharpe hides whether a small number of periods or a quote-currency basis generated most of the PnL.

## Proposed internal acceptance gate

These are project policy thresholds, not universal scientific constants. Before describing the strategy evidence as **8/10 or higher**, require all of the following on a frozen run:

1. At least 90 days of admitted synchronized data with point-in-time currency conversion and a separately untouched final period.
2. Positive final-period net returns in common currency after all modeled costs, with the 95% daily block-bootstrap confidence interval lower bound above zero.
3. Deflated Sharpe probability of skill above 0.95 after accounting for every recorded configuration trial, and PBO below 0.10 on the development configuration family.
4. Positive results in at least three predeclared volatility/liquidity regimes, with no one day contributing more than 25% of final-period net PnL.
5. Negative controls do not retain a comparable edge; fee, latency, quote-age, and size stress results are reported even when they fail.
6. An independent rerun from the manifest reproduces the report and checksums.

Passing this gate would justify a strong **historical research-evidence** score only. It would not establish live readiness, guarantee future returns, or substitute for venue adapters and a monitored paper-trading record.

## Current score

**Strategy evidence remains 2/10.** The protocol is ready to execute, but the data and experiment gates above have not been met. The research-methodology plan can be rated separately from empirical evidence.
