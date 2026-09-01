"""
AFML Quant Pipeline - Data Download Script
======================================
Downloads daily OHLCV price history for the core watchlist tickers
to today's date and saves them as CSV files in data/raw/.

Run manually whenever you want fresh data:
    python scripts/00_data_download.py

Output files:
    data/raw/SPY.csv
    data/raw/AAPL.csv
    data/raw/MSFT.csv
"""

import sys
import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / ".venv_packages"))

from datetime import datetime  # noqa: E402 (must come after sys.path setup)

import pandas as pd  # noqa: E402

from utils.obb_client import get_obb_client, OBBClientError  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE = '2021-01-01'
END_DATE   = datetime.today().strftime('%Y-%m-%d')   # auto-detects today

TICKERS = ['SPY', 'AAPL', 'MSFT']

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_ticker(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV from OBBClient and return as DataFrame."""
    logger.info(f"  Fetching {symbol}  ({start} → {end}) …")
    client = get_obb_client()
    df = client.get_price_history(symbol=symbol, start_date=start, end_date=end)
    logger.info(f"  {symbol}: {len(df)} rows  ({df.index[0].date()} – {df.index[-1].date()})")
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Download window: {START_DATE}  →  {END_DATE}")
    logger.info(f"Output directory: {RAW_DIR}")
    logger.info("")

    errors = []
    for ticker in TICKERS:
        try:
            df = download_ticker(ticker, START_DATE, END_DATE)
            out_path = RAW_DIR / f"{ticker}.csv"
            df.to_csv(out_path)
            logger.info(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")
        except OBBClientError as exc:
            logger.error(f"  FAILED {ticker}: {exc}")
            errors.append(ticker)
        logger.info("")

    if errors:
        logger.warning(f"Completed with errors for: {errors}")
        sys.exit(1)
    else:
        logger.info("All tickers downloaded successfully.")


if __name__ == "__main__":
    main()
