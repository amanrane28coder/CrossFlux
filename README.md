<div align="center">
  <h1>⚡ Cross-Venue Arbitrage Predictor</h1>
  <p><strong>Nanosecond-Optimized HFT Architecture for Structural OBI Dislocations</strong></p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg" alt="C++20" />
    <img src="https://img.shields.io/badge/Python-3.14-blue.svg" alt="Python 3.14" />
    <img src="https://img.shields.io/badge/Architecture-SPSC%20Lock--Free-orange.svg" alt="Lock-Free Architecture" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  </p>
</div>

---

## 📖 Executive Summary
The **Cross-Venue Arbitrage Predictor** is a high-frequency trading (HFT) infrastructure engine designed to identify and execute on structural Order Book Imbalance (OBI) dislocations across fragmented liquidity venues (e.g., Binance vs. Kraken). 

Engineered for strict sub-microsecond latency boundaries, this system pushes real-time WebSocket tick evaluation entirely to a custom C++ hardware-aligned core, eliminating Python's GIL overhead from the hot execution path. 

*Note: Following rigorous internal audits, the backtested strategy focuses on realistic capacity modeling, slippage, and latency rather than theoretical "perfect" execution. The system demonstrates genuine capacity limits and realistic win rates under strict VWAP book-walking constraints.*

## 🏗️ Architecture Deep Dive
This engine is built on absolute performance and hardware-level determinism.

*   **Sub-Microsecond Evaluation Core**: The central `SignalAggregator` evaluation loop leverages purely FMA arithmetic, eliminating transcendental function branches on the hot path. (Note: Benchmark times like 66ns reflect the early OBI computations and gating logic prior to full signal vector construction).
*   **Zero-Copy Memory Layout**: Custom `PriceLevel` and `OrderBookSnapshot` structs are strictly mapped for contiguous L2 cache alignment, maximizing hardware prefetcher efficiency and eliminating cache misses.
*   **Lock-Free SPSC Ring Buffer**: Real-time network ingestion is entirely decoupled from the execution loop. An asynchronous `Boost.Beast` thread pushes live market updates into a single-producer, single-consumer (SPSC) lock-free ring buffer, allowing the execution spin-loop to pop and evaluate without locking, blocking, or yielding the CPU.
*   **Thread Affinity & Pinning**: Core pinning protocols lock the network receiver and execution loops strictly to isolated CPU cores, circumventing OS context switches and thread migrations.

## 🔬 Backtesting Rigor
True alpha validation requires merciless execution constraints. The included event-driven backtester natively parses over **2 million rows** of real Level 2 Tardis.dev snapshot data without relying on generalized abstractions.

*   **Taker Friction & Latency**: Dual-leg execution costs (0.10% total taker fee friction) and network latency are rigorously modeled. A trade that turns against the model during the latency window is correctly booked as a loss.
*   **VWAP Order Book Walking**: The engine evaluates fills against the *prevailing* quote at `T + latency`. The fill price is the Volume-Weighted Average Price (VWAP) of the levels the order actually consumes, completely eliminating "touch price" tautologies.
*   **Realistic Results & Capacity Limits**: Swept across order sizes, the strategy exhibits a realistic capacity curve. At 0.01 BTC, it maintains a ~98.6% win rate, which naturally decays as slippage increases with size. Net PnL is shown to peak at an optimal size (e.g., 3.0 BTC per leg at 85.3% win rate), proving the strategy is falsifiable and respects real-world liquidity bounds.

## 📊 Analytics Dashboard
The repository features an interactive web visualization suite built on `Streamlit` and `Plotly` for deep introspection of execution logs:

*   **Zoomable Equity & Drawdown Curves**: Interactive deep-dives into high-volatility micro-batch windows.
*   **Alpha Signature Verification**: A custom scatter plot mapping `obi_delta` against Execution Edge (`pnl_net`). (Note: Analysis reveals `delta_threshold` acts primarily as a throughput throttle rather than a pure directional predictor).
*   **Metrics Grid**: Real-time KPI introspection including Annualized Sharpe Ratio, Max Drawdown, and cumulative fee bleed.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/CrossFlux.git
cd CrossFlux

# 2. Configure the true-black theme
mkdir -p .streamlit
echo -e "[theme]\nbase = \"dark\"\nfont = \"monospace\"" > .streamlit/config.toml

# 3. Launch the synthetic demo feed (or start the C++ backend)
python dashboard/seed_demo_feed.py &

# 4. Spin up the terminal UI
streamlit run dashboard/app.py
```

---
*Disclaimer: This repository is intended for research and educational purposes. Always simulate strategies extensively before deploying live capital.*
