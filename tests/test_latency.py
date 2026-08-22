"""
tests/test_latency.py
=====================
Validation suite for the Phase 3 OBI Delta and Latency Model.

Test inventory (7 tests)
------------------------
  1. test_obi_delta_positive          – A bid-heavy, B ask-heavy → delta = +1.4
  2. test_obi_delta_negative          – A ask-heavy, B bid-heavy → delta = -1.4
  3. test_obi_delta_zero              – Identical OBIs → delta = 0.0
  4. test_obi_delta_invalid_input     – OBI outside [-1,1] raises ValueError
  5. test_execution_probability_range – Valid inputs → result ∈ [0.0, 1.0]
  6. test_execution_probability_jitter_monotone – σ↑ ⟹ P↓ (core physical invariant)
  7. test_execution_probability_edge_cases – alpha≤0 → 0.0; sigma≤0 → ValueError

All tests use synthetic fixture data; no I/O, no network calls.

Jitter monotone reference values (alpha=50ms, mu=3.5 → median≈33ms):
    σ=0.1  →  P ≈ 0.9997  (near-certain: very tight latency distribution)
    σ=0.5  →  P ≈ 0.9772  (likely: moderate jitter)
    σ=1.0  →  P ≈ 0.8643  (uncertain: high jitter)
    σ=2.0  →  P ≈ 0.5791  (risky: severe jitter, near coin-flip)

Python 3.10+ required.
"""

from __future__ import annotations

import math

import pytest

from src.signals import calculate_obi_delta
from src.latency_model import calculate_execution_probability


# ===========================================================================
# Test 1 — OBI delta: positive divergence (A bid-heavy, B ask-heavy)
# ===========================================================================


def test_obi_delta_positive() -> None:
    """A bid-heavy exchange A vs. ask-heavy exchange B must yield a positive delta.

    Construction:
        obi_a = +0.8  (heavily bid-side)
        obi_b = -0.6  (heavily ask-side)
        delta = 0.8 - (-0.6) = +1.4
    """
    obi_a = 0.8
    obi_b = -0.6
    result = calculate_obi_delta(obi_a, obi_b)

    expected = 1.4
    assert result > 0.0, f"Expected positive delta, got {result}"
    assert abs(result - expected) < 1e-9, (
        f"OBI delta mismatch: expected {expected}, got {result}"
    )


# ===========================================================================
# Test 2 — OBI delta: negative divergence (A ask-heavy, B bid-heavy)
# ===========================================================================


def test_obi_delta_negative() -> None:
    """An ask-heavy exchange A vs. bid-heavy exchange B must yield a negative delta.

    Construction:
        obi_a = -0.6  (heavily ask-side)
        obi_b = +0.8  (heavily bid-side)
        delta = -0.6 - 0.8 = -1.4
    """
    obi_a = -0.6
    obi_b = 0.8
    result = calculate_obi_delta(obi_a, obi_b)

    expected = -1.4
    assert result < 0.0, f"Expected negative delta, got {result}"
    assert abs(result - expected) < 1e-9, (
        f"OBI delta mismatch: expected {expected}, got {result}"
    )


# ===========================================================================
# Test 3 — OBI delta: zero divergence (identical venues)
# ===========================================================================


def test_obi_delta_zero() -> None:
    """Identical OBI values on both venues must yield delta = 0.0."""
    for obi_value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        result = calculate_obi_delta(obi_value, obi_value)
        assert result == 0.0, (
            f"Expected delta=0.0 for equal OBIs ({obi_value}), got {result}"
        )


# ===========================================================================
# Test 4 — OBI delta: invalid input raises ValueError
# ===========================================================================


def test_obi_delta_invalid_input() -> None:
    """An OBI value outside [-1.0, 1.0] must raise ValueError immediately."""
    invalid_cases = [
        (1.1,  0.0,   "obi_exchange_a above +1"),
        (-1.1, 0.0,   "obi_exchange_a below -1"),
        (0.0,  1.001, "obi_exchange_b above +1"),
        (0.0, -2.5,   "obi_exchange_b below -1"),
    ]
    for obi_a, obi_b, description in invalid_cases:
        with pytest.raises(ValueError, match="outside the valid OBI range"):
            calculate_obi_delta(obi_a, obi_b), f"Expected ValueError for: {description}"


# ===========================================================================
# Test 5 — Execution probability: output always in [0.0, 1.0]
# ===========================================================================


def test_execution_probability_range() -> None:
    """calculate_execution_probability must always return a value in [0.0, 1.0]."""
    test_cases = [
        # (alpha_ms, mu, sigma, description)
        (50.0,   3.5,  0.3,  "typical HFT scenario"),
        (10.0,   4.0,  0.5,  "short-lived opportunity"),
        (500.0,  2.0,  1.0,  "long-lived opportunity, high jitter"),
        (1.0,    5.0,  0.1,  "very short window, high median latency"),
        (100.0,  1.0,  2.0,  "extremely high jitter"),
    ]
    for alpha, mu, sigma, description in test_cases:
        result = calculate_execution_probability(alpha, mu, sigma)
        assert 0.0 <= result <= 1.0, (
            f"[{description}] Expected probability in [0,1], got {result}"
        )


# ===========================================================================
# Test 6 — Execution probability: strictly decreases as jitter increases
# ===========================================================================


def test_execution_probability_jitter_monotone() -> None:
    """As network jitter (latency_sigma) increases, execution probability must
    strictly decrease — the core physical invariant of the latency model.

    Setup:
        alpha_lifetime_expected = 50.0 ms  (fixed discrepancy window)
        latency_mu              = 3.5      (median latency ≈ 33 ms, fixed)
        latency_sigma           = 0.1, 0.5, 1.0, 2.0  (increasing jitter)

    Expected ordering:
        P(σ=0.1) > P(σ=0.5) > P(σ=1.0) > P(σ=2.0)

    Reference values (approximate):
        σ=0.1  →  P ≈ 0.9997
        σ=0.5  →  P ≈ 0.9772
        σ=1.0  →  P ≈ 0.8643
        σ=2.0  →  P ≈ 0.5791
    """
    alpha = 50.0        # ms — fixed discrepancy lifetime
    mu    = 3.5         # fixed — median latency ≈ exp(3.5) ≈ 33 ms
    sigma_values = [0.1, 0.5, 1.0, 2.0]

    probabilities = [
        calculate_execution_probability(alpha, mu, sigma)
        for sigma in sigma_values
    ]

    # Verify strict monotone decrease at every step
    for i in range(len(probabilities) - 1):
        sigma_lo = sigma_values[i]
        sigma_hi = sigma_values[i + 1]
        p_lo = probabilities[i]
        p_hi = probabilities[i + 1]
        assert p_lo > p_hi, (
            f"Monotone decrease violated: "
            f"P(σ={sigma_lo})={p_lo:.6f} is NOT > P(σ={sigma_hi})={p_hi:.6f}. "
            f"Higher jitter must reduce execution probability."
        )

    # Sanity: all results in valid range
    for p, sigma in zip(probabilities, sigma_values):
        assert 0.0 <= p <= 1.0, (
            f"Probability {p} out of [0,1] for sigma={sigma}"
        )

    # Sanity: low jitter case should be near-certain (>0.99)
    assert probabilities[0] > 0.99, (
        f"Expected P(σ=0.1) > 0.99, got {probabilities[0]:.6f}"
    )

    # Sanity: high jitter case should be genuinely uncertain (<0.7)
    assert probabilities[-1] < 0.70, (
        f"Expected P(σ=2.0) < 0.70, got {probabilities[-1]:.6f}"
    )


# ===========================================================================
# Test 7 — Execution probability: edge cases
# ===========================================================================


def test_execution_probability_edge_cases() -> None:
    """Verify all documented edge case behaviours.

    Cases:
        1. alpha <= 0  →  return 0.0 (opportunity expired; data condition)
        2. alpha = 0   →  return 0.0
        3. sigma <= 0  →  raise ValueError (programming error)
        4. sigma = 0   →  raise ValueError
    """
    # Case 1: negative alpha → 0.0 sentinel
    result_neg = calculate_execution_probability(
        alpha_lifetime_expected=-1.0,
        latency_mu=3.5,
        latency_sigma=0.5,
    )
    assert result_neg == 0.0, (
        f"Expected 0.0 for negative alpha, got {result_neg}"
    )

    # Case 2: zero alpha → 0.0 sentinel
    result_zero = calculate_execution_probability(
        alpha_lifetime_expected=0.0,
        latency_mu=3.5,
        latency_sigma=0.5,
    )
    assert result_zero == 0.0, (
        f"Expected 0.0 for alpha=0, got {result_zero}"
    )

    # Case 3: negative sigma → ValueError
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_execution_probability(
            alpha_lifetime_expected=50.0,
            latency_mu=3.5,
            latency_sigma=-0.1,
        )

    # Case 4: zero sigma → ValueError
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_execution_probability(
            alpha_lifetime_expected=50.0,
            latency_mu=3.5,
            latency_sigma=0.0,
        )
