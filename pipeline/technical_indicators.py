"""
AFML Quant Pipeline - Technical Indicators (pandas-ta)
Wraps pandas-ta to add a standard set of indicators in one call.

Usage:
    ti = TechnicalIndicators()
    df = ti.add_all(df)   # df must have OHLCV columns
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# pandas-ta attaches itself to pd.DataFrame via its Strategy API.
try:
    import pandas_ta as ta  # noqa: F401 — side-effect: registers df.ta accessor
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("pandas-ta not installed; TechnicalIndicators will be a no-op.")


class TechnicalIndicators:
    """
    Adds a standard battery of technical indicators to an OHLCV DataFrame
    using pandas-ta.

    All indicators are appended in-place to the DataFrame and the enriched
    DataFrame is returned.  Missing columns for a given indicator are silently
    skipped so the class degrades gracefully with partial OHLCV data.

    Indicators added by add_all():
        RSI(14), MACD, Bollinger Bands, ATR, ADX,
        Stochastic %K/%D, OBV, EMA(20), EMA(50)
    """

    def __init__(self):
        self._available = _TA_AVAILABLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all standard indicators to *df* and return the enriched DataFrame.

        Args:
            df: DataFrame with at minimum a 'Close' column.
                'High', 'Low', 'Volume' are used by some indicators and skipped
                if absent.

        Returns:
            The same DataFrame (modified in-place) with indicator columns added.
        """
        if not self._available:
            logger.warning("pandas-ta unavailable; returning df unchanged.")
            return df

        df = df.copy()

        self._add_rsi(df)
        self._add_macd(df)
        self._add_bbands(df)
        self._add_atr(df)
        self._add_adx(df)
        self._add_stoch(df)
        self._add_obv(df)
        self._add_ema(df, length=20)
        self._add_ema(df, length=50)

        return df

    # ------------------------------------------------------------------
    # Individual indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_rsi(df: pd.DataFrame, length: int = 14) -> None:
        try:
            df.ta.rsi(length=length, append=True)
        except Exception as exc:
            logger.debug(f"RSI skipped: {exc}")

    @staticmethod
    def _add_macd(df: pd.DataFrame) -> None:
        try:
            df.ta.macd(append=True)
        except Exception as exc:
            logger.debug(f"MACD skipped: {exc}")

    @staticmethod
    def _add_bbands(df: pd.DataFrame) -> None:
        try:
            df.ta.bbands(append=True)
        except Exception as exc:
            logger.debug(f"BBands skipped: {exc}")

    @staticmethod
    def _add_atr(df: pd.DataFrame) -> None:
        if not all(c in df.columns for c in ["High", "Low", "Close"]):
            return
        try:
            df.ta.atr(append=True)
        except Exception as exc:
            logger.debug(f"ATR skipped: {exc}")

    @staticmethod
    def _add_adx(df: pd.DataFrame) -> None:
        if not all(c in df.columns for c in ["High", "Low", "Close"]):
            return
        try:
            df.ta.adx(append=True)
        except Exception as exc:
            logger.debug(f"ADX skipped: {exc}")

    @staticmethod
    def _add_stoch(df: pd.DataFrame) -> None:
        if not all(c in df.columns for c in ["High", "Low", "Close"]):
            return
        try:
            df.ta.stoch(append=True)
        except Exception as exc:
            logger.debug(f"Stochastic skipped: {exc}")

    @staticmethod
    def _add_obv(df: pd.DataFrame) -> None:
        if "Volume" not in df.columns:
            return
        try:
            df.ta.obv(append=True)
        except Exception as exc:
            logger.debug(f"OBV skipped: {exc}")

    @staticmethod
    def _add_ema(df: pd.DataFrame, length: int) -> None:
        try:
            df.ta.ema(length=length, append=True)
        except Exception as exc:
            logger.debug(f"EMA({length}) skipped: {exc}")
