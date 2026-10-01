<div align="center">
  <h1>⚡ CrossFlux</h1>
  <p><strong>Cross-venue market-data research and execution simulation</strong></p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/amanrane28coder/CrossFlux/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/amanrane28coder/CrossFlux/ci.yml?branch=main&label=CI" alt="CI status" /></a>
    <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg" alt="C++20" />
    <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+" />
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
The Python backtester can delay each simulated leg, fill against the prevailing book, walk available levels for VWAP, charge venue-specific fees on filled notional, and price residual legging risk. Real-data runs label an 80/20 chronological train/holdout split and reject rows where either quote is older than 250 ms by default; tune that threshold with `--max-quote-age-ms`. Sharpe is computed from one-minute portfolio returns with a 60-lag Newey–West serial-correlation adjustment; it remains an estimate and is not decision-grade on this one-day sample. Finance research shows conventional square-root annualization assumes independent returns, and a single holdout does not protect against repeated strategy selection ([Lo, “The Statistics of Sharpe Ratios”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=377260); [Bailey et al., “The Probability of Backtest Overfitting”](https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2326253)). The combined summary includes both training and holdout periods, so use split details to evaluate holdout separately. Current Binance BTC/USDT and Kraken XBT/USD runs lack historical USDT/USD conversion; their dollar-looking PnL and equity metrics are mixed-quote diagnostics and must not be read as common-currency performance. Run `python3 -m backtest.currency_sensitivity` for a post-hoc constant-rate sensitivity analysis; it does not replace observed FX or re-run signal selection. These are explicit simulation assumptions, not exchange-confirmed fills. The model does not cover queue priority, order acknowledgements, inventory constraints, funding, or venue outages. Real-data runs require both venue files; synthetic data is an explicit demo mode. `scripts/run_backtest.py` refuses to silently replace missing historical files with generated data; use `--synthetic` only for a demonstration.

Do not interpret synthetic-feed output as evidence of strategy profitability. For historical experiments, record the source files, checksums, time range, fee schedule, latency distribution, order size, and command used. Publish performance claims only with that reproducible run and the matching output artifact.

See [STRATEGY_VALIDATION_PROTOCOL.md](STRATEGY_VALIDATION_PROTOCOL.md) for the research-backed data, holdout, execution-replay, negative-control, and acceptance criteria.

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

Python direct dependencies are exact-pinned in `requirements.txt`; `pyproject.toml` records project metadata and optional history-download dependencies. The transitive lockfile could not be generated in this environment because package-index DNS/network access is unavailable, so full dependency-graph reproducibility is still pending. Copy `.env.example` to `.env` for local settings and load it in your shell; the application does not read `.env` automatically. These example values configure research/simulation only.

To build the native engine and run the explicit synthetic demo in Docker:

```bash
docker build -t crossflux-research .
docker run --rm crossflux-research
```

## 🧰 Native Build and CI

The CI workflow runs on Ubuntu 24.04 for pushes and pull requests. It installs the native toolchain and pinned Python dependencies, builds the C++ engine and simulated trading executable, then runs the Python suite against the built extension. Reproduce those steps locally on Ubuntu with:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-venv \
  libboost-all-dev libssl-dev nlohmann-json3-dev

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

PYBIND11_CMAKE_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake -S . -B build/ci \
  -DCROSSFLUX_OUTPUT_DIRECTORY="$PWD/build/artifacts" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"
cmake --build build/ci --parallel 2

PYTHONPATH="$PWD/build/artifacts:$PWD" .venv/bin/python -m pytest -q
```

The CI workflow builds the C++ targets and runs the Python suite. It does not exercise authenticated exchange orders; the C++ dispatcher remains simulated.

The Python market-data simulator can optionally expose Prometheus text metrics at `http://127.0.0.1:9108/metrics` by setting `CROSSFLUX_METRICS_ENABLED=true`. It binds to loopback by default. Metric names distinguish simulated orders/fills and unconverted account-unit PnL from real venue executions or USD PnL.

---
## Live-Trading Readiness
This repository is **not live-trading ready**. Before any live adapter is considered, implement venue-specific order submission, exchange-rule validation (quantity steps, minimum quantity/notional, and price ticks), authenticated fill/order-state reconciliation, explicit partial-fill and legging recovery, balance/inventory checks, rate-limit and reconnect handling, and a hard operator kill switch. Begin with exchange validation/dry-run modes and paper trading. Binance documents symbol filters and order-state APIs in its [official Spot API docs](https://developers.binance.com/en/docs/products/spot/rest-api); Kraken documents [WebSocket v2 order submission](https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/add_order) and [authenticated execution events](https://docs.kraken.com/exchange/api-reference/spot-websocket-v2/executions). These venue features are prerequisites to implement and verify, not capabilities this project currently provides. See the [exchange execution readiness research](LIVE_TRADING_RESEARCH.md) for source-backed failure cases, rollout gates, and separate project ratings.

The Python risk manager requires `CROSSFLUX_INITIAL_CAPITAL` (in the common quote currency used for realized P&L) and fails closed when it is absent. The C++ market-data demo also requires `CROSSFLUX_INITIAL_CAPITAL` and `CROSSFLUX_MAX_DRAWDOWN_PCT`; it derives the simulated loss trip from those settings and fails closed if either is missing or invalid. Set capital only after quote-currency P&L has been normalized; BTCUSDT versus XBT/USD still requires explicit USDT/USD basis handling. These controls do not turn the simulator into a real order execution system.

*For research and education only. No profitability or execution-quality claim is guaranteed.*
