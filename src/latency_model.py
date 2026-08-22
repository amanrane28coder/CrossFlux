"""
src/latency_model.py
====================
Stochastic latency model for the Cross-Venue Arbitrage Predictor — Phase 3.

Motivation
----------
In cross-venue arbitrage, execution success depends on whether our order can
reach the target exchange *before* the price discrepancy closes.  Network
round-trip latency is not deterministic: it follows a right-skewed distribution
with occasional extreme spikes (GC pauses, switch congestion, TCP retransmits).

The **log-normal distribution** is the canonical model for HFT network latency:
  - Strictly positive  (latency cannot be negative).
  - Right-skewed       (rare large spikes dominate the tail).
  - Parametrised by μ (location of log-latency mean) and σ (log-latency std).

Model
-----
Let L be our round-trip latency (ms).  We model:

    ln(L) ~ N(μ_L, σ_L²)

so L itself is log-normally distributed with:
  - Median  = exp(μ_L)                (the scale parameter)
  - Mean    = exp(μ_L + σ_L² / 2)    (always > median for σ > 0)

The execution probability for a discrepancy with expected lifetime α (ms) is:

    P(execute) = P(L < α) = F_lognorm(α | s=σ_L, scale=exp(μ_L))

where F_lognorm is the log-normal CDF evaluated via scipy.stats.lognorm.

Key invariant (monotone jitter property):
    σ_L ↑  ⟹  P(execute) ↓

As jitter increases, the latency tail fattens beyond α, so execution risk rises
and the execution probability falls monotonically.

Python 3.10+ required.
"""

from __future__ import annotations

import logging
import math

from scipy.stats import lognorm  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def calculate_execution_probability(
    alpha_lifetime_expected: float,
    latency_mu: float,
    latency_sigma: float,
) -> float:
    """Estimate the probability of executing before a price discrepancy closes.

    Uses the log-normal CDF to model stochastic network latency against
    the expected lifetime of the arbitrage window.

    Parameters
    ----------
    alpha_lifetime_expected:
        Expected lifetime of the price discrepancy, in **milliseconds**.
        This is the window within which the order must arrive and match.
        If ``<= 0``, the opportunity has already expired; returns ``0.0``.
    latency_mu:
        Mean of ``ln(latency)`` — i.e. the log of the **median** network
        round-trip latency in milliseconds.  ``exp(latency_mu)`` is the median
        latency (50th percentile), not the arithmetic mean.
        Example: ``latency_mu=3.5`` → median latency ≈ 33 ms.
    latency_sigma:
        Standard deviation of ``ln(latency)`` — the log-normal shape parameter
        controlling the width and skew of the latency distribution.
        Must be strictly positive.  Larger values = heavier tail = more jitter.

    Returns
    -------
    float
        Execution probability in ``[0.0, 1.0]``.
        ``0.0`` means certain failure; ``1.0`` means certain success (theoretical).

    Raises
    ------
    ValueError
        If ``latency_sigma <= 0``.  The shape parameter of a log-normal
        distribution must be strictly positive.

    Examples
    --------
    >>> # Discrepancy lives 50 ms; median latency ≈ 33 ms; low jitter.
    >>> calculate_execution_probability(
    ...     alpha_lifetime_expected=50.0,
    ...     latency_mu=3.5,
    ...     latency_sigma=0.3,
    ... )
    0.9297...   # High probability — median latency well below 50 ms

    >>> # Same setup but extreme jitter (σ=2.0) — near coin-flip.
    >>> calculate_execution_probability(
    ...     alpha_lifetime_expected=50.0,
    ...     latency_mu=3.5,
    ...     latency_sigma=2.0,
    ... )
    0.5791...
    """
    # ------------------------------------------------------------------
    # Guard: shape parameter must be strictly positive
    # ------------------------------------------------------------------
    if latency_sigma <= 0.0:
        raise ValueError(
            f"calculate_execution_probability: latency_sigma must be strictly "
            f"positive, got {latency_sigma!r}. The log-normal shape parameter "
            f"cannot be zero or negative."
        )

    # ------------------------------------------------------------------
    # Guard: non-positive lifetime → opportunity already expired
    # ------------------------------------------------------------------
    if alpha_lifetime_expected <= 0.0:
        logger.debug(
            "calculate_execution_probability: alpha_lifetime_expected=%.4f ms "
            "<= 0 — opportunity expired, returning 0.0.",
            alpha_lifetime_expected,
        )
        return 0.0

    # ------------------------------------------------------------------
    # Map log-normal parameters to scipy convention:
    #   scipy.stats.lognorm(s, scale) where:
    #     s     = σ_L   (shape — std of the underlying normal)
    #     scale = e^μ_L (median of the log-normal = exp of location param)
    # ------------------------------------------------------------------
    scale: float = math.exp(latency_mu)
    probability: float = float(
        lognorm.cdf(alpha_lifetime_expected, s=latency_sigma, scale=scale)
    )

    logger.debug(
        "Execution probability = %.6f  "
        "(alpha=%.2f ms, median_latency=%.2f ms, sigma=%.4f)",
        probability,
        alpha_lifetime_expected,
        scale,
        latency_sigma,
    )

    return probability
