# Exchange execution readiness research

**Reviewed:** 2026-10-01  
**Scope:** repository implementation and public Binance Spot / Kraken Spot API documentation. This is a requirements audit, not evidence of exchange connectivity or trading performance.

## Finding

The current C++ program consumes public market data and dispatches to `SimulatedOrderDispatcher`; `AdaptiveOrderDispatcher` inherits from that simulator. It prints that execution is simulated. There is no authenticated order submission, cancel path, private order/fill stream, or exchange reconciliation in the reviewed execution path. The capital and drawdown environment settings protect the simulator startup path; they do not protect an exchange account.

**Live operational readiness: 3/10.** The project has a dispatcher abstraction, simulated sizing/risk controls, and fail-closed risk configuration. It lacks the venue integration and recovery controls below, so this score is a readiness assessment, not a prediction of profitability.

## Exchange requirements verified from primary documentation

### Binance Spot

- An HTTP `5XX` response leaves execution status unknown; clients must reconcile order state instead of interpreting the response as a definite rejection. A retry without reconciliation can duplicate exposure. The official REST documentation describes order querying and user-data events as ways to determine state: [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api).
- Symbol filters constrain prices and quantities. The adapter must load and enforce tick size, quantity step, minimum/maximum quantity, and minimum notional for the selected symbol before sending an order: [Binance symbol filters](https://developers.binance.com/en/docs/products/spot/filters).
- HTTP 429 indicates rate limiting; continuing to send requests can lead to an HTTP 418 IP ban. The client needs a shared rate-limit budget, backoff, and a halt policy for repeated throttling: [Binance Spot REST API](https://developers.binance.com/en/docs/products/spot/rest-api).

### Kraken Spot

- WebSocket v2 `add_order` supports `validate=true`, which validates an order without placing it, and supports a client order identifier. This can be used for adapter validation and correlation, but validation-only success is not a fill or a live execution test: [Kraken WebSocket v2 Add Order](https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/add_order).
- The authenticated `executions` channel delivers order status and execution events, including snapshots and partial-fill states. The client needs to subscribe, persist/reconcile the initial snapshot, and process subsequent events as authoritative account state: [Kraken WebSocket v2 Executions](https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/executions).

## Historical data availability audit

This checkout contains one synchronized date of Binance BTCUSDT and Kraken XBT/USD depth snapshots (2024-03-01). I checked the exchanges' public historical-data descriptions for ways to extend that sample:

- Binance's official public-data catalog documents downloadable spot aggregate trades, trades, and klines. Those are useful for trade-flow and lower-frequency studies, but they are not the historical multi-level order-book snapshots this backtest needs to replay depth-aware fills: [Binance Public Data catalog](https://github.com/binance/binance-public-data#data-information).
- Kraken's downloadable archive contains time-and-sales rows (timestamp, price, volume, side/type, order type, and trade ID), not historical L2 snapshots and deltas: [Kraken downloadable historical market data](https://support.kraken.com/articles/360047543791-downloadable-historical-market-data-time-and-sales-).
- Therefore, adding those trade archives as if they were book data would change the experiment and overstate execution realism. The next valid expansion is synchronized L2 snapshots/deltas from both venues plus a timestamped USDT/USD series, with source checksums and capture/availability timestamps. If that data is not publicly available for the same window, label the run as a different trade-only study or obtain a licensed market-data source; do not splice unlike data into this backtest.

## Required implementation gates

1. **Venue adapter and symbol metadata.** Implement authenticated adapter(s), secrets from a secret store/environment, startup account and symbol checks, and exact tick/step/notional validation. Reject unknown symbols and stale metadata.
2. **Durable order lifecycle.** Persist an intent and client correlation ID before submission. Model `pending`, `open`, `partially filled`, `filled`, `cancel pending`, `canceled`, `rejected`, and `unknown` states. On timeout, disconnect, or ambiguous server response, reconcile by client ID/order ID before retrying. Make retries idempotent.
3. **Private stream and recovery.** Consume authenticated executions/order events, deduplicate by venue execution ID, handle out-of-order or repeated events, and reconcile open orders, fills, and balances after reconnect and process restart. A REST/WebSocket acknowledgement alone is not a fill.
4. **Cross-venue exposure control.** Track base and quote inventory separately per venue and currency. Define a maximum unhedged quantity, maximum age of the first leg, and deterministic recovery for partial fill, rejected hedge, cancel race, and venue outage. Stop new entries when state is uncertain.
5. **Kill and restart semantics.** A kill must block new orders, cancel outstanding orders where possible, reconcile resulting fills, and retain the halted state across restart until an operator-reviewed reconciliation clears it. Validate the capital and drawdown limits against normalized account-currency equity; do not treat USDT as USD without an FX source and timestamp.
6. **Operational controls.** Add clock-drift and feed-staleness gates, bounded reconnect/backoff, rate-limit handling, structured audit logs, alerts for unknown order state and inventory breaches, and a manual emergency procedure. Never log credentials or signed request material.

## Evidence-based rollout gates

1. **Adapter contract tests:** deterministic mocks for rejection, timeout-before-accept, timeout-after-accept, duplicate/out-of-order execution events, partial fills, cancel/fill races, rate limits, and restart reconciliation. Assert no duplicate order and bounded unhedged exposure.
2. **Venue validation:** Binance symbol-filter validation plus test-environment order lifecycle coverage; Kraken `validate=true` plus authenticated private-stream lifecycle checks. Test-environment behavior must be documented separately from production behavior.
3. **Paper soak:** at least 30 calendar days across process restarts and injected disconnects; zero unexplained balance/order mismatches, all unknown outcomes reconciled, and every kill-switch scenario recorded. Preserve run logs and configuration hashes.
4. **Reviewed limited pilot:** only after the prior gates pass, with an explicitly capped account allocation, human monitoring, predeclared stop criteria, and a tested kill/reconciliation procedure. This audit does not authorize or claim such a pilot has occurred.

## Project rating under separate rubrics

| Dimension | Rating | What the number measures |
|---|---:|---|
| Research audit and protocol quality | **8.5/10** | The record now cites primary venue documentation and peer-reviewed/preprint methodology, separates assumptions from results, and defines a predeclared validation protocol with falsification and acceptance gates. The protocol is not yet executed. |
| Strategy evidence | **2/10** | One day of mixed BTC/USDT and XBT/USD quote-unit results, without observed historical USDT/USD conversion, is insufficient evidence for profitability or robustness. HAC-adjusted Sharpe does not cure limited sample size or selection bias. |
| Live operational readiness | **3/10** | Public market data and simulated dispatch exist; authenticated execution and order-state recovery do not. |

The **8.5/10 applies only to the quality of the research audit and proposed protocol**. It does not raise the strategy-evidence or live-readiness score. Those scores can rise only when the missing data and implementation gates produce reviewable results. See [STRATEGY_VALIDATION_PROTOCOL.md](STRATEGY_VALIDATION_PROTOCOL.md) for the experiment design.

## What would justify scores above 8

- **Strategy evidence above 8:** at least six months of synchronized, point-in-time L2 data from both venues; timestamped conversion for every quote currency; a frozen strategy specification; chronological train/validation/test windows spanning distinct market regimes; locked holdout results reported once; cost and latency stress tests; and uncertainty intervals that account for serial dependence and repeated parameter search. The current one-day sample cannot meet this bar, and trade-only archives cannot substitute for L2 execution replay.
- **Live readiness above 8:** working authenticated adapters for each venue; exchange filter and balance validation; durable idempotent order intents; private-stream plus REST reconciliation; safe recovery for ambiguous submissions, partial fills, cancel races, restarts, rate limits, and venue outages; demonstrated kill-and-reconcile behavior; and at least 30 days of monitored paper operation with zero unexplained order/balance mismatches. None of these should be inferred from simulated fills or a successful build.

These are evidence gates, not a promise that passing them produces a profitable strategy. Until they are met, the honest current ratings remain strategy evidence **2/10** and live readiness **3/10**.
