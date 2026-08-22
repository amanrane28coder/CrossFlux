"""
tests/test_ingestion.py
=======================
Validation suite for the Cross-Venue Arbitrage Predictor — Phase 1.

Test inventory (9 tests)
------------------------
  1. test_price_level_creation            – PriceLevel fields and immutability
  2. test_order_book_snapshot_structure   – Bid/ask ordering invariants
  3. test_validate_snapshot_valid         – Clean book passes validation
  4. test_validate_snapshot_crossed       – Crossed book raises ValueError
  5. test_validate_snapshot_locked        – Locked book raises ValueError
  6. test_validate_snapshot_empty_bids    – Empty bids raises ValueError
  7. test_parse_tardis_csv_columns        – CSV parser produces correct schema
  8. test_ffill_alignment                 – As-Of merge fills gaps correctly
  9. test_market_state_venues             – MarketState keys contain both venues

All tests use synthetic fixture data; no network calls, no raw data files needed.

Python 3.10+ required.
"""

from __future__ import annotations

import io
import textwrap
from typing import List

import pandas as pd
import pytest

from src.models import MarketState, OrderBookSnapshot, PriceLevel
from src.ingestion import (
    align_venues,
    build_snapshots,
    parse_tardis_csv,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================


def _make_snapshot(
    exchange_id: str = "binance",
    timestamp_ms: int = 1_700_000_000_000,
    best_bid: float = 49_999.0,
    best_ask: float = 50_001.0,
    n_levels: int = 3,
) -> OrderBookSnapshot:
    """Factory: build a synthetic, valid OrderBookSnapshot."""
    bids = [
        PriceLevel(price=best_bid - i * 1.0, volume=1.0 + i * 0.1)
        for i in range(n_levels)
    ]
    asks = [
        PriceLevel(price=best_ask + i * 1.0, volume=1.0 + i * 0.1)
        for i in range(n_levels)
    ]
    return OrderBookSnapshot(
        exchange_id=exchange_id,
        timestamp_ms=timestamp_ms,
        bids=bids,
        asks=asks,
    )


def _make_tardis_csv(n_rows: int = 5, depth: int = 3) -> str:
    """Generate a minimal Tardis.dev book_snapshot CSV string in-memory.

    Column schema mirrors the real Tardis format.  Timestamps are in
    microseconds (µs); the parser is expected to convert to ms by // 1000.
    """
    header_parts = ["timestamp", "local_timestamp"]
    for i in range(depth):
        header_parts += [f"asks[{i}].price", f"asks[{i}].amount"]
    for i in range(depth):
        header_parts += [f"bids[{i}].price", f"bids[{i}].amount"]

    rows = [",".join(header_parts)]
    base_ts_us = 1_700_000_000_000_000  # µs
    base_bid = 49_999.0
    base_ask = 50_001.0

    for row_idx in range(n_rows):
        ts_us = base_ts_us + row_idx * 1_000_000  # +1 second per row
        local_ts_us = ts_us + 500
        values = [str(ts_us), str(local_ts_us)]
        for i in range(depth):
            values += [str(base_ask + i * 1.0 + row_idx * 0.1), "1.0"]
        for i in range(depth):
            values += [str(base_bid - i * 1.0 - row_idx * 0.1), "1.0"]
        rows.append(",".join(values))

    return "\n".join(rows)


# ===========================================================================
# validate_snapshot() — standalone validator used throughout the test suite
# ===========================================================================


def validate_snapshot(snapshot: OrderBookSnapshot) -> bool:
    """Ensure a snapshot has a valid, non-crossed order book.

    Parameters
    ----------
    snapshot:
        An ``OrderBookSnapshot`` to validate.

    Returns
    -------
    bool
        ``True`` if the book is valid (best bid strictly < best ask).

    Raises
    ------
    ValueError
        If ``bids`` or ``asks`` is empty.
        If the book is crossed (best bid ≥ best ask).
        If the book is locked (best bid == best ask).
    """
    if not snapshot.bids:
        raise ValueError(
            f"[{snapshot.exchange_id}] Snapshot at {snapshot.timestamp_ms} ms "
            f"has an empty bids list."
        )
    if not snapshot.asks:
        raise ValueError(
            f"[{snapshot.exchange_id}] Snapshot at {snapshot.timestamp_ms} ms "
            f"has an empty asks list."
        )

    best_bid_price = snapshot.bids[0].price
    best_ask_price = snapshot.asks[0].price

    if best_bid_price >= best_ask_price:
        raise ValueError(
            f"[{snapshot.exchange_id}] Crossed/locked book at {snapshot.timestamp_ms} ms: "
            f"best_bid={best_bid_price} >= best_ask={best_ask_price}"
        )

    return True


# ===========================================================================
# Test 1 — PriceLevel creation and immutability
# ===========================================================================


def test_price_level_creation() -> None:
    """PriceLevel fields must be accessible and the object must be immutable."""
    lvl = PriceLevel(price=50_000.0, volume=2.5)

    assert lvl.price == 50_000.0
    assert lvl.volume == 2.5

    # frozen=True means attribute assignment must raise FrozenInstanceError
    with pytest.raises((AttributeError, TypeError)):
        lvl.price = 99_999.0  # type: ignore[misc]


# ===========================================================================
# Test 2 — OrderBookSnapshot ordering invariants
# ===========================================================================


def test_order_book_snapshot_structure() -> None:
    """Bids must be descending; asks must be ascending; best bid < best ask."""
    snap = _make_snapshot(n_levels=5)

    # Bid ordering: highest price first
    for i in range(len(snap.bids) - 1):
        assert snap.bids[i].price > snap.bids[i + 1].price, (
            f"Bids not descending at index {i}: "
            f"{snap.bids[i].price} vs {snap.bids[i+1].price}"
        )

    # Ask ordering: lowest price first
    for i in range(len(snap.asks) - 1):
        assert snap.asks[i].price < snap.asks[i + 1].price, (
            f"Asks not ascending at index {i}: "
            f"{snap.asks[i].price} vs {snap.asks[i+1].price}"
        )

    # Spread must be positive
    assert snap.best_bid.price < snap.best_ask.price


# ===========================================================================
# Test 3 — validate_snapshot: valid book
# ===========================================================================


def test_validate_snapshot_valid() -> None:
    """A clean, non-crossed book must return True."""
    snap = _make_snapshot(best_bid=49_999.0, best_ask=50_001.0)
    result = validate_snapshot(snap)
    assert result is True


# ===========================================================================
# Test 4 — validate_snapshot: crossed book
# ===========================================================================


def test_validate_snapshot_crossed() -> None:
    """A crossed book (best bid > best ask) must raise ValueError."""
    # We bypass OrderBookSnapshot.__post_init__ to inject the bad state
    # by using object.__setattr__ on a mock-like object.
    # Instead, build the crossed condition directly as a raw dict and
    # test our standalone validate_snapshot function by creating snapshots
    # with manually set attributes using a subclass workaround.

    # Strategy: build a valid snapshot, then create a wrapper struct
    # that mimics the interface but holds a crossed book.
    class _FakeCrossedBook:
        exchange_id = "test"
        timestamp_ms = 1_700_000_000_000

        bids = [PriceLevel(price=50_002.0, volume=1.0)]
        asks = [PriceLevel(price=50_001.0, volume=1.0)]

    with pytest.raises(ValueError, match="Crossed/locked"):
        validate_snapshot(_FakeCrossedBook())  # type: ignore[arg-type]


# ===========================================================================
# Test 5 — validate_snapshot: locked book
# ===========================================================================


def test_validate_snapshot_locked() -> None:
    """A locked book (best bid == best ask) must raise ValueError."""

    class _FakeLockedBook:
        exchange_id = "test"
        timestamp_ms = 1_700_000_000_000
        bids = [PriceLevel(price=50_000.0, volume=1.0)]
        asks = [PriceLevel(price=50_000.0, volume=1.0)]

    with pytest.raises(ValueError, match="Crossed/locked"):
        validate_snapshot(_FakeLockedBook())  # type: ignore[arg-type]


# ===========================================================================
# Test 6 — validate_snapshot: empty bids
# ===========================================================================


def test_validate_snapshot_empty_bids() -> None:
    """A snapshot with an empty bids list must raise ValueError."""

    class _FakeEmptyBids:
        exchange_id = "test"
        timestamp_ms = 1_700_000_000_000
        bids: list = []
        asks = [PriceLevel(price=50_001.0, volume=1.0)]

    with pytest.raises(ValueError, match="empty bids"):
        validate_snapshot(_FakeEmptyBids())  # type: ignore[arg-type]


# ===========================================================================
# Test 7 — parse_tardis_csv: correct schema output
# ===========================================================================


def test_parse_tardis_csv_columns(tmp_path: pytest.TempPathFactory) -> None:
    """parse_tardis_csv must return a DataFrame with timestamp_ms index and
    correct bid/ask column names for the requested depth."""
    depth = 3
    csv_content = _make_tardis_csv(n_rows=5, depth=depth)

    # Write synthetic CSV to a temp file
    csv_file = tmp_path / "binance_test.csv"
    csv_file.write_text(csv_content)

    df = parse_tardis_csv(csv_file, exchange_id="binance", depth=depth)

    # Index must be named timestamp_ms
    assert df.index.name == "timestamp_ms", f"Expected 'timestamp_ms', got '{df.index.name}'"

    # Timestamps must be in milliseconds (µs // 1000)
    base_ts_us = 1_700_000_000_000_000
    expected_first_ts_ms = base_ts_us // 1000
    assert df.index[0] == expected_first_ts_ms, (
        f"Expected first ts_ms={expected_first_ts_ms}, got {df.index[0]}"
    )

    # All bid/ask columns must be present
    for i in range(depth):
        assert f"bids[{i}].price" in df.columns
        assert f"bids[{i}].amount" in df.columns
        assert f"asks[{i}].price" in df.columns
        assert f"asks[{i}].amount" in df.columns

    # exchange_id column must be present
    assert "exchange_id" in df.columns
    assert (df["exchange_id"] == "binance").all()

    # Row count must match
    assert len(df) == 5


# ===========================================================================
# Test 8 — align_venues: forward-fill gaps correctly
# ===========================================================================


def test_ffill_alignment() -> None:
    """After As-Of alignment, no MarketState should have a None venue entry.

    Scenario:
        Binance fires at t=1000, t=3000, t=5000 ms.
        Kraken  fires at t=2000, t=4000          ms.

    Expected aligned states (after dropping t=1000 where Kraken has no prior):
        t=2000: Binance=snap@1000 (ffill), Kraken=snap@2000
        t=3000: Binance=snap@3000,          Kraken=snap@2000 (ffill)
        t=4000: Binance=snap@3000 (ffill),  Kraken=snap@4000
        t=5000: Binance=snap@5000,          Kraken=snap@4000 (ffill)
    """
    binance_snaps = [
        _make_snapshot("binance", timestamp_ms=1_000),
        _make_snapshot("binance", timestamp_ms=3_000),
        _make_snapshot("binance", timestamp_ms=5_000),
    ]
    kraken_snaps = [
        _make_snapshot("kraken", timestamp_ms=2_000),
        _make_snapshot("kraken", timestamp_ms=4_000),
    ]

    states = align_venues(binance_snaps, kraken_snaps)

    # After dropping leading NaN row (t=1000), we expect 4 aligned states
    assert len(states) == 4, f"Expected 4 aligned states, got {len(states)}"

    # No state should have None for either venue
    for state in states:
        assert state.venues.get("binance") is not None, (
            f"Missing binance snapshot at ts={state.timestamp_ms}"
        )
        assert state.venues.get("kraken") is not None, (
            f"Missing kraken snapshot at ts={state.timestamp_ms}"
        )

    # Verify specific ffill behaviour at t=2000: Binance should carry snap@1000
    state_t2000 = states[0]
    assert state_t2000.timestamp_ms == 2_000
    assert state_t2000.venues["binance"].timestamp_ms == 1_000, (
        "Expected Binance to be forward-filled from t=1000 at the t=2000 state"
    )
    assert state_t2000.venues["kraken"].timestamp_ms == 2_000

    # Verify ffill at t=3000: Kraken should carry snap@2000
    state_t3000 = states[1]
    assert state_t3000.timestamp_ms == 3_000
    assert state_t3000.venues["kraken"].timestamp_ms == 2_000, (
        "Expected Kraken to be forward-filled from t=2000 at the t=3000 state"
    )


# ===========================================================================
# Test 9 — MarketState: venue keys contain both exchanges
# ===========================================================================


def test_market_state_venues() -> None:
    """A MarketState produced by align_venues must contain both venue keys."""
    binance_snaps = [_make_snapshot("binance", timestamp_ms=1_000)]
    kraken_snaps = [_make_snapshot("kraken", timestamp_ms=1_000)]

    states = align_venues(binance_snaps, kraken_snaps)

    assert len(states) >= 1
    for state in states:
        venue_keys = set(state.venues.keys())
        assert "binance" in venue_keys, f"'binance' key missing; got {venue_keys}"
        assert "kraken" in venue_keys, f"'kraken' key missing; got {venue_keys}"
        # Each venue value must be a typed OrderBookSnapshot
        assert isinstance(state.venues["binance"], OrderBookSnapshot)
        assert isinstance(state.venues["kraken"], OrderBookSnapshot)
