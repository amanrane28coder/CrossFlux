"""
src/ingestion.py
================
Data ingestion pipeline for the Cross-Venue Arbitrage Predictor — Phase 1.

Responsibilities
----------------
1. Parse historical Level-2 tick data from Tardis.dev CSV/GZIP files.
2. Convert raw rows into typed ``OrderBookSnapshot`` objects.
3. Temporally align asynchronous venue streams via As-Of (forward-fill) merge.
4. Expose a single top-level orchestrator: ``load_market_states()``.

Alignment strategy
------------------
The two exchanges (Binance, Kraken) emit snapshots at independent, asynchronous
timestamps.  The limit order book is a *step function*: the book state does not
change between events, so interpolation is explicitly forbidden.

We apply an **As-Of merge** using the union-timestamp + ffill approach:

    1. Build a DataFrame per venue, indexed by ``timestamp_ms``.
    2. Outer-join both DataFrames on the union of all timestamps.
    3. Forward-fill (``ffill()``) each column independently — this propagates
       the last known book state forward in time until the next real event.
    4. Drop leading rows where either venue has NaN (no prior state available).
    5. Construct ``MarketState`` objects from the aligned rows.

Python 3.10+ required.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from src.models import MarketState, OrderBookSnapshot, PriceLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default book depth (number of price levels per side).
DEFAULT_DEPTH: int = 10

#: Tardis.dev timestamp column (microseconds since Unix epoch).
TARDIS_TIMESTAMP_COL: str = "timestamp"

#: Tardis.dev local timestamp column (arrival time on their infrastructure).
TARDIS_LOCAL_TIMESTAMP_COL: str = "local_timestamp"


# ---------------------------------------------------------------------------
# Step 1 — Parse raw Tardis CSV into a tidy DataFrame
# ---------------------------------------------------------------------------


def parse_tardis_csv(
    filepath: Path | str,
    exchange_id: str,
    depth: int = DEFAULT_DEPTH,
) -> pd.DataFrame:
    """Parse a Tardis.dev ``book_snapshot`` CSV or GZIP file into a tidy DataFrame.

    The Tardis ``book_snapshot_<N>`` schema contains columns of the form::

        timestamp, local_timestamp,
        asks[0].price, asks[0].amount, ..., asks[N-1].price, asks[N-1].amount,
        bids[0].price, bids[0].amount, ..., bids[N-1].price, bids[N-1].amount

    Parameters
    ----------
    filepath:
        Absolute or relative path to a ``.csv`` or ``.csv.gz`` file.
    exchange_id:
        Canonical exchange label (e.g. ``"binance"``, ``"kraken"``).
    depth:
        Number of price levels to retain per side.  Must be ≤ N in the file.

    Returns
    -------
    pd.DataFrame
        Index: ``timestamp_ms`` (integer milliseconds, UTC).
        Columns: ``exchange_id``,
                 ``bids[i].price``, ``bids[i].amount`` for i in range(depth),
                 ``asks[i].price``, ``asks[i].amount`` for i in range(depth).

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    KeyError
        If the required Tardis columns are missing from the file.
    ValueError
        If ``depth`` exceeds the number of levels present in the file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    logger.info("Parsing %s  (exchange=%s, depth=%d)", filepath.name, exchange_id, depth)

    # Determine compression from extension
    compression: str | None = "gzip" if filepath.suffix == ".gz" else None

    raw: pd.DataFrame = pd.read_csv(filepath, compression=compression, low_memory=False)

    # -----------------------------------------------------------------------
    # Timestamp conversion: Tardis stores µs; we need integer ms.
    # Integer division by 1000 truncates (no rounding) — correct for step fn.
    # -----------------------------------------------------------------------
    if TARDIS_TIMESTAMP_COL not in raw.columns:
        raise KeyError(
            f"Expected column '{TARDIS_TIMESTAMP_COL}' not found in {filepath.name}. "
            f"Available columns: {list(raw.columns)}"
        )

    timestamp_ms: pd.Series = (raw[TARDIS_TIMESTAMP_COL] // 1000).astype(np.int64)

    # -----------------------------------------------------------------------
    # Select bid/ask columns up to requested depth
    # -----------------------------------------------------------------------
    bid_price_cols = [f"bids[{i}].price" for i in range(depth)]
    bid_vol_cols = [f"bids[{i}].amount" for i in range(depth)]
    ask_price_cols = [f"asks[{i}].price" for i in range(depth)]
    ask_vol_cols = [f"asks[{i}].amount" for i in range(depth)]

    required_cols = bid_price_cols + bid_vol_cols + ask_price_cols + ask_vol_cols
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise KeyError(
            f"Required book-depth columns missing from {filepath.name}: {missing[:4]}{'...' if len(missing) > 4 else ''}\n"
            f"Available: {[c for c in raw.columns if 'bids' in c or 'asks' in c][:8]}"
        )

    df = raw[required_cols].copy()
    df.insert(0, "exchange_id", exchange_id)
    df.index = timestamp_ms
    df.index.name = "timestamp_ms"

    # Coerce all numeric columns to float64 for consistent downstream handling
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Parsed %d snapshots from %s (timestamp range: %d – %d ms)",
        len(df),
        filepath.name,
        int(df.index.min()),
        int(df.index.max()),
    )
    return df


# ---------------------------------------------------------------------------
# Step 2 — Build typed OrderBookSnapshot objects from a tidy DataFrame
# ---------------------------------------------------------------------------


def build_snapshots(
    df: pd.DataFrame,
    exchange_id: str,
    depth: int = DEFAULT_DEPTH,
) -> List[OrderBookSnapshot]:
    """Convert a tidy parsed DataFrame into a list of typed ``OrderBookSnapshot`` objects.

    Parameters
    ----------
    df:
        Output of ``parse_tardis_csv()``.
    exchange_id:
        Canonical exchange label.
    depth:
        Number of price levels to include per side.

    Returns
    -------
    List[OrderBookSnapshot]
        One snapshot per row. Rows with NaN bid/ask data are silently skipped
        and logged as warnings.
    """
    snapshots: List[OrderBookSnapshot] = []
    skipped = 0

    bid_price_cols = [f"bids[{i}].price" for i in range(depth)]
    bid_vol_cols = [f"bids[{i}].amount" for i in range(depth)]
    ask_price_cols = [f"asks[{i}].price" for i in range(depth)]
    ask_vol_cols = [f"asks[{i}].amount" for i in range(depth)]

    for ts_ms, row in df.iterrows():
        try:
            bids = [
                PriceLevel(price=float(row[bp]), volume=float(row[bv]))
                for bp, bv in zip(bid_price_cols, bid_vol_cols)
                if not (pd.isna(row[bp]) or pd.isna(row[bv]))
            ]
            asks = [
                PriceLevel(price=float(row[ap]), volume=float(row[av]))
                for ap, av in zip(ask_price_cols, ask_vol_cols)
                if not (pd.isna(row[ap]) or pd.isna(row[av]))
            ]

            # Defensive sort: Tardis guarantees ordering but we enforce it
            bids.sort(key=lambda lvl: lvl.price, reverse=True)   # best bid first
            asks.sort(key=lambda lvl: lvl.price, reverse=False)  # best ask first

            snapshot = OrderBookSnapshot(
                exchange_id=exchange_id,
                timestamp_ms=int(ts_ms),
                bids=bids,
                asks=asks,
            )
            snapshots.append(snapshot)

        except (ValueError, KeyError) as exc:
            skipped += 1
            logger.debug("Skipping row at ts_ms=%d: %s", ts_ms, exc)

    if skipped:
        logger.warning(
            "[%s] Skipped %d malformed rows out of %d total.",
            exchange_id,
            skipped,
            len(df),
        )

    logger.info("[%s] Built %d OrderBookSnapshot objects.", exchange_id, len(snapshots))
    return snapshots


# ---------------------------------------------------------------------------
# Step 3 — As-Of alignment (union timestamp + ffill)
# ---------------------------------------------------------------------------


def _snapshots_to_dataframe(snapshots: Sequence[OrderBookSnapshot]) -> pd.DataFrame:
    """Serialise a list of ``OrderBookSnapshot`` objects into a DataFrame.

    Each snapshot is stored as a single opaque Python object in one column
    (``"snapshot"``).  This avoids re-flattening price levels during alignment
    and lets us carry the fully typed object through the merge.

    Index: ``timestamp_ms`` (int64).
    """
    records = {snap.timestamp_ms: snap for snap in snapshots}
    series = pd.Series(records, name="snapshot", dtype=object)
    series.index = series.index.astype(np.int64)
    series.index.name = "timestamp_ms"
    return series.to_frame()


def align_venues(
    snapshots_a: Sequence[OrderBookSnapshot],
    snapshots_b: Sequence[OrderBookSnapshot],
) -> List[MarketState]:
    """Temporally align two asynchronous snapshot streams via As-Of (ffill) merge.

    The limit order book is a *step function*: between two real events, the
    book state is unchanged.  We therefore:

      1. Build a DataFrame per venue, indexed by ``timestamp_ms``.
      2. Outer-join on the union of all timestamps.
      3. Forward-fill (``ffill()``) — NO interpolation, NO backward fill.
      4. Drop leading rows where either venue has NaN (insufficient history).
      5. Construct ``MarketState`` objects from the aligned rows.

    Parameters
    ----------
    snapshots_a:
        Snapshot list from the first venue (e.g. Binance).
    snapshots_b:
        Snapshot list from the second venue (e.g. Kraken).

    Returns
    -------
    List[MarketState]
        Aligned states, one per unique timestamp across both venues.
        The first N states (where one venue has no prior data yet) are dropped.

    Raises
    ------
    ValueError
        If either input list is empty.
    """
    if not snapshots_a:
        raise ValueError("align_venues: snapshots_a is empty.")
    if not snapshots_b:
        raise ValueError("align_venues: snapshots_b is empty.")

    exchange_a = snapshots_a[0].exchange_id
    exchange_b = snapshots_b[0].exchange_id

    df_a = _snapshots_to_dataframe(snapshots_a).rename(columns={"snapshot": exchange_a})
    df_b = _snapshots_to_dataframe(snapshots_b).rename(columns={"snapshot": exchange_b})

    # --- Outer join on the union of all timestamps ---
    aligned: pd.DataFrame = df_a.join(df_b, how="outer", sort=True)

    # --- Forward-fill only (step function; no interpolation) ---
    # ffill propagates the last known OrderBookSnapshot forward in time.
    aligned[exchange_a] = aligned[exchange_a].ffill()
    aligned[exchange_b] = aligned[exchange_b].ffill()

    # --- Drop leading rows where either venue has no prior state ---
    aligned.dropna(subset=[exchange_a, exchange_b], inplace=True)

    logger.info(
        "Aligned %d rows across %s and %s (union timestamps before drop: %d)",
        len(aligned),
        exchange_a,
        exchange_b,
        len(df_a) + len(df_b),
    )

    # --- Construct MarketState objects ---
    market_states: List[MarketState] = []
    for ts_ms, row in aligned.iterrows():
        state = MarketState(
            timestamp_ms=int(ts_ms),
            venues={
                exchange_a: row[exchange_a],
                exchange_b: row[exchange_b],
            },
        )
        market_states.append(state)

    logger.info("Produced %d MarketState objects.", len(market_states))
    return market_states


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def load_market_states(
    binance_path: Path | str,
    kraken_path: Path | str,
    depth: int = DEFAULT_DEPTH,
) -> List[MarketState]:
    """End-to-end pipeline: parse → build → align → return aligned ``MarketState`` list.

    Parameters
    ----------
    binance_path:
        Path to the Binance Tardis CSV/GZIP file.
    kraken_path:
        Path to the Kraken Tardis CSV/GZIP file.
    depth:
        Number of price levels to retain per side (default: 10).

    Returns
    -------
    List[MarketState]
        Temporally aligned market states for all timestamps across both venues.

    Example
    -------
    >>> from pathlib import Path
    >>> from src.ingestion import load_market_states
    >>> states = load_market_states(
    ...     Path("data/raw/binance_btcusdt.csv.gz"),
    ...     Path("data/raw/kraken_xbtusd.csv.gz"),
    ... )
    >>> print(f"Loaded {len(states)} aligned MarketState objects")
    """
    binance_path = Path(binance_path)
    kraken_path = Path(kraken_path)

    logger.info("=== Phase 1 Ingestion Pipeline Starting ===")

    # 1. Parse raw CSVs
    df_binance = parse_tardis_csv(binance_path, exchange_id="binance", depth=depth)
    df_kraken = parse_tardis_csv(kraken_path, exchange_id="kraken", depth=depth)

    # 2. Build typed snapshot objects
    snaps_binance = build_snapshots(df_binance, exchange_id="binance", depth=depth)
    snaps_kraken = build_snapshots(df_kraken, exchange_id="kraken", depth=depth)

    # 3. As-Of align both streams
    states = align_venues(snaps_binance, snaps_kraken)

    logger.info("=== Ingestion complete: %d MarketState objects ===", len(states))
    return states
