"""
src/predictor.py
================
Signal Aggregator for the Cross-Venue Arbitrage Predictor — Phase 4.

This module wires together every prior phase into a single cohesive loop:

    Phase 2  →  calculate_obi()           per venue snapshot
    Phase 3  →  calculate_obi_delta()     cross-venue pressure divergence
    Phase 3  →  calculate_execution_probability()  stochastic latency gate
    Phase 4  →  ArbitrageSignal           emitted when both gates pass

Pipeline (per MarketState)
--------------------------
1. calculate_obi(snap_a) and calculate_obi(snap_b)
2. delta = calculate_obi_delta(obi_a, obi_b)
3. Gate 1: abs(delta) > delta_threshold  — if not, skip immediately
4. p     = calculate_execution_probability(alpha, mu, sigma)
5. Gate 2: p > min_p_execute             — if not, skip
6. Resolve action label from sign of delta
7. Emit ArbitrageSignal

Gate Ordering
-------------
Gate 1 (OBI delta check) is evaluated BEFORE the latency CDF call (Gate 2).
This is a deliberate performance optimisation: in live tick data the vast
majority of states will have |delta| ≤ threshold, so we avoid invoking
scipy.stats.lognorm.cdf on every tick.

Action Label Convention
-----------------------
A positive delta (exchange A bid-heavy, B ask-heavy) means price pressure
is concentrated on A → the arbitrage is to buy on B (cheaper) and sell on A
(where demand is high).  Labels are constructed from exchange_id strings so
the aggregator is venue-agnostic:

    delta > 0  →  "BUY_{exchange_b}_SELL_{exchange_a}"
    delta < 0  →  "BUY_{exchange_a}_SELL_{exchange_b}"

Python 3.10+ required.
"""

from __future__ import annotations

import logging
from typing import List

from src.models import ArbitrageSignal, MarketState
from src.signals import calculate_obi, calculate_obi_delta
from src.latency_model import calculate_execution_probability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (overridable at construction time)
# ---------------------------------------------------------------------------

DEFAULT_DELTA_THRESHOLD: float = 0.3   # minimum |Δ_OBI| to pass Gate 1
DEFAULT_MIN_P_EXECUTE:   float = 0.80  # minimum execution probability for Gate 2
DEFAULT_OBI_DEPTH:       int   = 5     # number of book levels per side


class SignalAggregator:
    """Iterates over aligned ``MarketState`` objects and emits ``ArbitrageSignal`` events.

    The aggregator applies a strict two-gate filter:

      - **Gate 1 (volume pressure):** ``|obi_delta| > delta_threshold``
      - **Gate 2 (latency risk):**    ``p_execute   > min_p_execute``

    Only states that clear **both** gates produce an ``ArbitrageSignal``.

    Parameters
    ----------
    exchange_a:
        Canonical ID of the first venue (e.g. ``"binance"``).
    exchange_b:
        Canonical ID of the second venue (e.g. ``"kraken"``).
    latency_mu:
        Log-normal μ parameter for network round-trip latency.
        ``exp(latency_mu)`` is the median latency in milliseconds.
    latency_sigma:
        Log-normal σ parameter (shape) for network jitter.
        Must be strictly positive.
    alpha_lifetime_ms:
        Expected lifetime of a price discrepancy in milliseconds.
        Used as the CDF evaluation point in the latency model.
    obi_depth:
        Number of order book levels to consume per side when computing OBI.
        Default: ``5``.
    delta_threshold:
        Minimum absolute OBI delta to pass Gate 1.  Default: ``0.3``.
    min_p_execute:
        Minimum execution probability to pass Gate 2.  Default: ``0.80``.

    Examples
    --------
    >>> aggregator = SignalAggregator(
    ...     exchange_a="binance",
    ...     exchange_b="kraken",
    ...     latency_mu=3.5,
    ...     latency_sigma=0.4,
    ...     alpha_lifetime_ms=50.0,
    ... )
    >>> signals = aggregator.evaluate(market_states)
    >>> print(f"Fired {len(signals)} signals from {len(market_states)} states")
    """

    def __init__(
        self,
        exchange_a: str,
        exchange_b: str,
        latency_mu: float,
        latency_sigma: float,
        alpha_lifetime_ms: float,
        obi_depth: int = DEFAULT_OBI_DEPTH,
        delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
        min_p_execute: float = DEFAULT_MIN_P_EXECUTE,
    ) -> None:
        if not exchange_a:
            raise ValueError("SignalAggregator: exchange_a must not be empty.")
        if not exchange_b:
            raise ValueError("SignalAggregator: exchange_b must not be empty.")
        if exchange_a == exchange_b:
            raise ValueError(
                f"SignalAggregator: exchange_a and exchange_b must be distinct, "
                f"got '{exchange_a}' for both."
            )
        if obi_depth <= 0:
            raise ValueError(
                f"SignalAggregator: obi_depth must be positive, got {obi_depth}."
            )
        if delta_threshold < 0:
            raise ValueError(
                f"SignalAggregator: delta_threshold must be non-negative, "
                f"got {delta_threshold}."
            )
        if not (0.0 <= min_p_execute <= 1.0):
            raise ValueError(
                f"SignalAggregator: min_p_execute must be in [0.0, 1.0], "
                f"got {min_p_execute}."
            )

        self.exchange_a = exchange_a
        self.exchange_b = exchange_b
        self.latency_mu = latency_mu
        self.latency_sigma = latency_sigma
        self.alpha_lifetime_ms = alpha_lifetime_ms
        self.obi_depth = obi_depth
        self.delta_threshold = delta_threshold
        self.min_p_execute = min_p_execute

        # Pre-compute the execution probability — it is constant per aggregator
        # instance because latency parameters and alpha are fixed at construction.
        # This avoids redundant CDF evaluations on the hot path when Gate 1 passes.
        self._p_execute: float = calculate_execution_probability(
            alpha_lifetime_expected=self.alpha_lifetime_ms,
            latency_mu=self.latency_mu,
            latency_sigma=self.latency_sigma,
        )

        logger.info(
            "SignalAggregator initialised: %s vs %s | "
            "delta_threshold=%.3f | min_p_execute=%.3f | "
            "p_execute=%.6f (alpha=%.1f ms, mu=%.3f, sigma=%.3f)",
            exchange_a, exchange_b,
            delta_threshold, min_p_execute,
            self._p_execute,
            alpha_lifetime_ms, latency_mu, latency_sigma,
        )

    # ------------------------------------------------------------------
    # Action label resolution
    # ------------------------------------------------------------------

    def _resolve_action(self, delta: float) -> str:
        """Return the trade directive string based on the sign of ``delta``.

        Positive delta → A is bid-heavy; buy the cheaper B, sell into A's demand.
        Negative delta → B is bid-heavy; buy the cheaper A, sell into B's demand.
        """
        if delta > 0:
            return f"BUY_{self.exchange_b.upper()}_SELL_{self.exchange_a.upper()}"
        return f"BUY_{self.exchange_a.upper()}_SELL_{self.exchange_b.upper()}"

    # ------------------------------------------------------------------
    # Core evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self, states: List[MarketState]) -> List[ArbitrageSignal]:
        """Evaluate a list of aligned market states and emit arbitrage signals.

        Parameters
        ----------
        states:
            List of ``MarketState`` objects as produced by
            ``src.ingestion.align_venues()`` or ``load_market_states()``.
            May be empty — returns an empty list without error.

        Returns
        -------
        List[ArbitrageSignal]
            Filtered list of actionable signals.  May be empty if no state
            cleared both threshold gates.
        """
        signals: List[ArbitrageSignal] = []
        gate1_passed = 0
        gate2_passed = 0

        for state in states:
            # ── Step 1: retrieve per-venue snapshots ──────────────────────
            snap_a = state.venues.get(self.exchange_a)
            snap_b = state.venues.get(self.exchange_b)

            if snap_a is None or snap_b is None:
                logger.warning(
                    "MarketState at %d ms missing venue '%s' or '%s' — skipping.",
                    state.timestamp_ms,
                    self.exchange_a,
                    self.exchange_b,
                )
                continue

            # ── Step 2: compute per-venue OBI ────────────────────────────
            obi_a = calculate_obi(snap_a, depth=self.obi_depth)
            obi_b = calculate_obi(snap_b, depth=self.obi_depth)

            # ── Step 3: compute cross-venue delta ────────────────────────
            delta = calculate_obi_delta(obi_a, obi_b)

            # ── Gate 1: volume-pressure filter (fast path) ───────────────
            if abs(delta) <= self.delta_threshold:
                logger.debug(
                    "Gate 1 FAIL at %d ms: |delta|=%.4f ≤ %.3f",
                    state.timestamp_ms, abs(delta), self.delta_threshold,
                )
                continue

            gate1_passed += 1

            # ── Gate 2: latency-risk filter ──────────────────────────────
            # p_execute is pre-computed at construction; reuse it here.
            if self._p_execute <= self.min_p_execute:
                logger.debug(
                    "Gate 2 FAIL at %d ms: p_execute=%.4f ≤ %.3f",
                    state.timestamp_ms, self._p_execute, self.min_p_execute,
                )
                continue

            gate2_passed += 1

            # ── Step 4: resolve action and emit signal ───────────────────
            action = self._resolve_action(delta)
            signal = ArbitrageSignal(
                timestamp_ms=state.timestamp_ms,
                obi_delta=delta,
                p_execute=self._p_execute,
                action=action,
            )
            signals.append(signal)

            logger.info(
                "SIGNAL at %d ms: Δ=%+.4f  p=%.4f  → %s",
                state.timestamp_ms, delta, self._p_execute, action,
            )

        logger.info(
            "evaluate() complete: %d states | gate1=%d | gate2=%d | signals=%d",
            len(states), gate1_passed, gate2_passed, len(signals),
        )
        return signals
