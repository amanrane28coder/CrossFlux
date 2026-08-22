"""
src/models.py
=============
Strict, immutable data structures for the Cross-Venue Arbitrage Predictor.

Design constraints:
  - All dataclasses are frozen=True (runtime immutability enforcement).
  - All dataclasses use slots=True (≈40% lower per-instance memory overhead —
    critical when holding millions of tick-level snapshots in memory).
  - Timestamps are stored as integer *milliseconds* (truncated from Tardis µs).
  - Bids are always ordered *descending* by price  (bids[0] = best bid).
  - Asks are always ordered *ascending*  by price  (asks[0] = best ask).

Python 3.10+ required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Primitive level
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """A single resting order quantity at one price tick on the order book.

    Attributes
    ----------
    price:
        Limit price in the quote currency (e.g. USD).
    volume:
        Resting quantity in the base currency (e.g. BTC).
    """

    price: float
    volume: float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"PriceLevel.price must be positive, got {self.price}")
        if self.volume < 0:
            raise ValueError(f"PriceLevel.volume must be non-negative, got {self.volume}")


# ---------------------------------------------------------------------------
# Single-venue snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """A full Level-2 snapshot for one exchange at one point in time.

    Attributes
    ----------
    exchange_id:
        Canonical exchange identifier (e.g. ``"binance"``, ``"kraken"``).
    timestamp_ms:
        UTC event timestamp in **integer milliseconds**.
    bids:
        List of bid ``PriceLevel`` objects, sorted *descending* (best bid first).
    asks:
        List of ask ``PriceLevel`` objects, sorted *ascending* (best ask first).

    Invariants (enforced at construction):
        - ``bids`` and ``asks`` must both be non-empty.
        - ``bids[0].price < asks[0].price``  (no crossed or locked book).
        - ``timestamp_ms`` must be a positive integer.
    """

    exchange_id: str
    timestamp_ms: int
    bids: List[PriceLevel]
    asks: List[PriceLevel]

    def __post_init__(self) -> None:
        if not self.exchange_id:
            raise ValueError("OrderBookSnapshot.exchange_id must not be empty.")
        if self.timestamp_ms <= 0:
            raise ValueError(
                f"OrderBookSnapshot.timestamp_ms must be positive, got {self.timestamp_ms}"
            )
        if not self.bids:
            raise ValueError(
                f"[{self.exchange_id}] OrderBookSnapshot.bids must not be empty."
            )
        if not self.asks:
            raise ValueError(
                f"[{self.exchange_id}] OrderBookSnapshot.asks must not be empty."
            )
        best_bid = self.bids[0].price
        best_ask = self.asks[0].price
        if best_bid >= best_ask:
            raise ValueError(
                f"[{self.exchange_id}] Crossed/locked book at {self.timestamp_ms} ms: "
                f"best_bid={best_bid} >= best_ask={best_ask}"
            )

    @property
    def best_bid(self) -> PriceLevel:
        """Return the highest bid level."""
        return self.bids[0]

    @property
    def best_ask(self) -> PriceLevel:
        """Return the lowest ask level."""
        return self.asks[0]

    @property
    def mid_price(self) -> float:
        """Return the arithmetic mid-price."""
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread(self) -> float:
        """Return the absolute bid-ask spread."""
        return self.asks[0].price - self.bids[0].price


# ---------------------------------------------------------------------------
# Multi-venue aligned state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketState:
    """A temporally aligned, multi-venue order book state.

    After As-Of (forward-fill) alignment, every ``MarketState`` holds the
    *last known* snapshot from each venue as of ``timestamp_ms``.

    Attributes
    ----------
    timestamp_ms:
        The logical UTC timestamp of this aligned state, in integer milliseconds.
        This is drawn from the union of all venue event timestamps.
    venues:
        Mapping from ``exchange_id`` → ``OrderBookSnapshot``.
        After alignment, every configured exchange should have an entry here.
    """

    timestamp_ms: int
    venues: Dict[str, OrderBookSnapshot]

    def __post_init__(self) -> None:
        if self.timestamp_ms <= 0:
            raise ValueError(
                f"MarketState.timestamp_ms must be positive, got {self.timestamp_ms}"
            )
        if not self.venues:
            raise ValueError("MarketState.venues must not be empty.")

    def spread_between(self, exchange_a: str, exchange_b: str) -> float:
        """Return the cross-venue spread: best ask on A minus best bid on B.

        A negative value indicates a potential arbitrage opportunity
        (you can buy cheaper on A than you can sell on B).
        """
        snap_a = self.venues[exchange_a]
        snap_b = self.venues[exchange_b]
        return snap_a.best_ask.price - snap_b.best_bid.price


# ---------------------------------------------------------------------------
# Actionable arbitrage signal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArbitrageSignal:
    """A filtered, actionable cross-venue arbitrage signal.

    Emitted by ``SignalAggregator.evaluate()`` only when **both** threshold
    gates are cleared:

      - ``|obi_delta| > delta_threshold``   (significant volume-pressure divergence)
      - ``p_execute   > min_p_execute``      (sufficient probability of on-time execution)

    Attributes
    ----------
    timestamp_ms:
        Logical UTC timestamp of the triggering ``MarketState``, in integer
        milliseconds.
    obi_delta:
        Cross-venue OBI delta value that cleared the threshold.
        Positive means exchange A is bid-heavy relative to B;
        negative means the reverse.
    p_execute:
        Execution probability returned by the log-normal latency model at
        the time of signal generation.  Always in ``[0.0, 1.0]``.
    action:
        Human-readable trade directive, e.g.
        ``"BUY_KRAKEN_SELL_BINANCE"`` or ``"BUY_BINANCE_SELL_KRAKEN"``.
        Constructed dynamically from the configured exchange IDs.
    """

    timestamp_ms: int
    obi_delta: float
    p_execute: float
    action: str

    def __post_init__(self) -> None:
        if self.timestamp_ms <= 0:
            raise ValueError(
                f"ArbitrageSignal.timestamp_ms must be positive, got {self.timestamp_ms}"
            )
        if not (0.0 <= self.p_execute <= 1.0):
            raise ValueError(
                f"ArbitrageSignal.p_execute must be in [0.0, 1.0], got {self.p_execute}"
            )
        if not self.action:
            raise ValueError("ArbitrageSignal.action must not be empty.")

