<div align="center">
  <h1>⚡ CrossFlux</h1>
  <p><strong>Cross-venue market-data research and execution simulation</strong></p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg" alt="C++20" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/Architecture-SPSC%20Lock--Free-orange.svg" alt="Lock-Free Architecture" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  </p>
</div>

---

## 📖 Project Status
CrossFlux is a research prototype for cross-venue order-book signals, market-data ingestion, backtesting experiments, and a dashboard. The C++ executable currently uses simulated execution: it does **not** authenticate to Binance or Kraken or submit, cancel, or reconcile exchange orders. Do not use it to trade real funds.

The backtester has a configurable latency and VWAP book-walking model, with separate fees and legging costs. It remains a simulation: it does not model queue priority, exchange acknowledgements, exchange-confirmed fills, balances, or venue outages. Performance figures in older reports are not endorsed until reproduced from the exact source revision and data files.

## 🏗️ Architecture
The repository contains a Python signal/backtest/dashboard stack and a C++ market-data and signal-processing stack. WebSocket market data and simulated order dispatch are useful for research and local demonstrations. The README previously described lock-free ingestion, core pinning, and sub-microsecond end-to-end evaluation as production characteristics; those performance claims need reproducible benchmark evidence and should not be read as verified end-to-end trading latency.

## 🔬 Backtesting Limits
The Python backtester can delay each simulated leg, fill against the prevailing book, walk available levels for VWAP, charge venue-specific fees on filled notional, and price residual legging risk. These are explicit assumptions, not exchange-confirmed fills. It does not model queue priority, order acknowledgements, inventory constraints, funding, or venue outages. Real-data runs require both venue files; synthetic data is an explicit demo mode. `scripts/run_backtest.py` refuses to silently replace missing historical files with generated data; use `--synthetic` only for a demonstration.

Do not interpret synthetic-feed output as evidence of strategy profitability. For historical experiments, record the source files, checksums, time range, fee schedule, latency distribution, order size, and command used. Publish performance claims only with that reproducible run and the matching output artifact.

## 📊 Analytics Dashboard
The repository features an interactive web visualization suite built on `Streamlit` and `Plotly` for deep introspection of execution logs:

*   **Zoomable Equity & Drawdown Curves**: Interactive deep-dives into high-volatility micro-batch windows.
*   **Alpha Signature Verification**: A custom scatter plot mapping `obi_delta` against Execution Edge (`pnl_net`). (Note: Analysis reveals `delta_threshold` acts primarily as a throughput throttle rather than a pure directional predictor).
*   **Metrics Grid**: Real-time KPI introspection including Annualized Sharpe Ratio, Max Drawdown, and cumulative fee bleed.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/amanrane28coder/CrossFlux.git
cd CrossFlux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the true-black theme
mkdir -p .streamlit
echo -e "[theme]\nbase = \"dark\"\nfont = \"monospace\"" > .streamlit/config.toml

# 4. Start the local simulated demo feed
python3 dashboard/seed_demo_feed.py &

# Optional: generated-data simulation (demo only)
python3 scripts/run_backtest.py --synthetic

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

---
## Live-Trading Readiness
This repository is **not live-trading ready**. Before any live adapter is considered, implement venue-specific order submission, exchange-rule validation (quantity steps, minimum quantity/notional, and price ticks), authenticated fill/order-state reconciliation, explicit partial-fill and legging recovery, balance/inventory checks, rate-limit and reconnect handling, and a hard operator kill switch. Begin with exchange validation/dry-run modes and paper trading. Binance documents symbol filters and order-state APIs in its [official Spot API docs](https://developers.binance.com/en/docs/products/spot/rest-api); Kraken documents [WebSocket v2 order submission](https://docs-legacy.kraken.com/api/docs/websocket-v2/add_order/) and [book integrity guidance](https://docs.kraken.com/api/docs/guides/spot-ws-book-v2). These venue features are prerequisites to implement and verify, not capabilities this project currently provides.

*For research and education only. No profitability or execution-quality claim is guaranteed.*
