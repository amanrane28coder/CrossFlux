"""
scripts/download_data.py
========================
Downloads historical Level-2 book_snapshot_10 data from Tardis.dev for
Binance (BTCUSDT) and Kraken (XBT/USD) into data/raw/.

Data type
---------
book_snapshot_10: full top-10 depth snapshots at each tick.
Columns: timestamp, local_timestamp,
         asks[0].price, asks[0].amount, ..., asks[9].price, asks[9].amount,
         bids[0].price, bids[0].amount, ..., bids[9].price, bids[9].amount

This matches the column schema expected by src/ingestion.py exactly.

Free-tier note
--------------
With an empty API key, Tardis serves SAMPLE DATA only for the 1st day of
each month.  To download any arbitrary date, set TARDIS_API_KEY in your
environment or pass it explicitly.

Usage
-----
    # Sample data (free, 1st of month only):
    python3 scripts/download_data.py

    # Full data with API key:
    TARDIS_API_KEY=your_key python3 scripts/download_data.py --date 2024-03-05
"""

import argparse
import logging
import os
from pathlib import Path

# ── macOS Python 3.14 SSL fix ─────────────────────────────────────────────────
# Python.org macOS builds do not link the system CA bundle.  aiohttp (used
# internally by tardis-dev) inherits SSL_CERT_FILE from the environment.
# Injecting certifi's bundle here makes HTTPS work without modifying system state.
import certifi, os as _os
_os.environ.setdefault("SSL_CERT_FILE", certifi.where())
_os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
# ─────────────────────────────────────────────────────────────────────────────

from tardis_dev import download_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_data")

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
DOWNLOAD_DIR = _ROOT / "data" / "raw"

# ── Exchange → correct symbol mapping ────────────────────────────────────────
# Free-tier available data types: book_snapshot_5, book_snapshot_25,
# trades, incremental_book_L2, quotes, book_ticker
DATA_TYPE = "book_snapshot_5"   # top-5 depth; matches src/ingestion.py with depth=5
EXCHANGE_CONFIG = {
    "binance": {
        "symbol":    "BTCUSDT",
        "filename":  "binance_btcusdt",
    },
    "kraken": {
        "symbol":    "XBT/USD",
        "filename":  "kraken_xbtusd",
    },
}


def download(date: str, api_key: str) -> None:
    """
    Download book_snapshot_10 for Binance and Kraken on a given date.

    Parameters
    ----------
    date    : ISO date string, e.g. "2024-03-01"
    api_key : Tardis API key (empty string = free sample tier, 1st of month only)
    """
    from_date = date
    # Tardis requires to_date > from_date (exclusive end — the day after)
    from datetime import datetime, timedelta
    to_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for exchange, cfg in EXCHANGE_CONFIG.items():
        symbol   = cfg["symbol"]
        logger.info("Downloading %s  %s  %s ...", exchange, symbol, from_date)

        try:
            # download_datasets is SYNCHRONOUS in tardis-dev v4 — no await needed
            download_datasets(
                exchange     = exchange,
                data_types   = [DATA_TYPE],            # top-5 depth snapshots
                from_date    = from_date,
                to_date      = to_date,
                symbols      = [symbol],               # one symbol per exchange
                api_key      = api_key,
                download_dir = str(DOWNLOAD_DIR),
            )
            logger.info("✓  %s  data staged in %s/", exchange, DOWNLOAD_DIR)

        except Exception as exc:
            logger.error("✗  %s download failed: %s", exchange, exc)
            raise

    # ── List downloaded files ─────────────────────────────────────────────────
    files = sorted(DOWNLOAD_DIR.glob("*.gz")) + sorted(DOWNLOAD_DIR.glob("*.csv"))
    if files:
        logger.info("\nDownloaded files:")
        for f in files:
            size_mb = f.stat().st_size / 1_048_576
            logger.info("  %-60s  %.1f MB", f.name, size_mb)
    else:
        logger.warning("No files found in %s — check API key and date.", DOWNLOAD_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Tardis.dev book_snapshot_10 data.")
    parser.add_argument(
        "--date",
        default="2024-03-01",
        help="ISO date to download (default: 2024-03-01). "
             "Free tier: only the 1st of each month is available.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TARDIS_API_KEY", ""),
        help="Tardis API key (default: $TARDIS_API_KEY env var, or empty for sample data).",
    )
    args = parser.parse_args()

    if not args.api_key:
        logger.warning(
            "No API key provided — downloading FREE SAMPLE DATA only.\n"
            "  • Sample data is available for the 1st of each month.\n"
            "  • Date overridden to 2024-03-01 to guarantee availability.\n"
            "  Set TARDIS_API_KEY env var or use --api-key for full access."
        )
        args.date = "2024-03-01"

    logger.info("=" * 60)
    logger.info("  Tardis.dev Data Downloader")
    logger.info("  Date     : %s", args.date)
    logger.info("  API key  : %s", "SET" if args.api_key else "EMPTY (sample mode)")
    logger.info("  Dest     : %s", DOWNLOAD_DIR)
    logger.info("=" * 60)

    download(date=args.date, api_key=args.api_key)


if __name__ == "__main__":
    main()
