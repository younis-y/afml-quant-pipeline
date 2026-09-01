"""
AFML Quant Pipeline - OpenBB Data Provider (compatibility shim)
===========================================================
Keeps the same public method signatures used across the codebase, but
delegates all implementation to OBBClient.

OpenBB SDK is the single data path — no yfinance fallbacks.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from utils.obb_client import get_obb_client, OBBClientError

logger = logging.getLogger(__name__)


class OpenBBDataProvider:
    """
    Compatibility wrapper that preserves the public API used in 6 files
    while delegating all calls to OBBClient.
    """

    def __init__(self, default_provider: str = "yfinance"):
        self.default_provider = default_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            ticker:   Stock symbol (e.g. "SPY")
            start:    ISO date string "YYYY-MM-DD"  (mutually exclusive with period)
            end:      ISO date string "YYYY-MM-DD"
            period:   shorthand period string ("1y", "2y", "5y", …)
            interval: Bar interval — only "1d" supported via OBB free tier
            provider: OpenBB provider name; defaults to self.default_provider

        Returns:
            DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
        """
        if start is None and period is not None:
            start, end = get_obb_client()._period_to_dates(period)

        if start is None:
            start, end = get_obb_client()._period_to_dates("1y")

        _provider = provider or self.default_provider

        return get_obb_client().get_price_history(
            symbol=ticker,
            start_date=start,
            end_date=end,
            provider=_provider,
        )

    def fetch_fundamentals(
        self,
        ticker: str,
        provider: Optional[str] = None,
    ) -> dict:
        """
        Fetch key fundamental data.

        Returns:
            Dict with keys: pe_ratio, eps, market_cap, revenue,
            forward_pe, beta, and any additional OBB fields.
            Missing values are None.
        """
        _provider = provider or self.default_provider

        try:
            df = get_obb_client().get_fundamentals_metrics(
                symbol=ticker,
                period="annual",
                limit=1,
                provider=_provider,
            )
            row = df.iloc[0]

            def _f(key: str) -> Optional[float]:
                try:
                    val = row.get(key)
                    return float(val) if val is not None else None
                except (TypeError, ValueError):
                    return None

            return {
                "pe_ratio":   _f("pe_ratio"),
                "eps":        _f("eps"),
                "market_cap": _f("market_cap"),
                "revenue":    _f("revenue"),
                "forward_pe": _f("forward_pe"),
                "beta":       _f("beta"),
                # Additional OBB fields
                "pb_ratio":          _f("price_to_book"),
                "roe":               _f("return_on_equity"),
                "gross_margin":      _f("gross_profit_margin"),
                "operating_margin":  _f("operating_profit_margin"),
                "net_margin":        _f("net_profit_margin"),
                "debt_to_equity":    _f("debt_equity_ratio"),
                "current_ratio":     _f("current_ratio"),
                "revenue_growth":    _f("revenue_growth"),
                "earnings_growth":   _f("earnings_growth"),
            }
        except OBBClientError as exc:
            logger.warning(f"fetch_fundamentals({ticker}): {exc}; returning empty dict.")
            return {
                "pe_ratio": None, "eps": None, "market_cap": None,
                "revenue": None, "forward_pe": None, "beta": None,
            }

    def fetch_macro(
        self,
        series_id: str = "GDP",
        start: Optional[str] = None,
        end: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        FRED / macro series are not available on the free OBB tier.
        Returns an empty DataFrame and logs a warning.
        """
        logger.warning(
            "fetch_macro: obb.economy.* endpoints are not available on the "
            "free OpenBB tier. Returning empty DataFrame."
        )
        return pd.DataFrame()

    def fetch_news(
        self,
        ticker: str,
        limit: int = 10,
        provider: Optional[str] = None,
    ) -> list:
        """
        Fetch recent news articles for a ticker.

        Returns:
            List of dicts with keys: title, date, url, summary.
            Empty list if data unavailable.
        """
        _provider = provider or self.default_provider
        return get_obb_client().get_company_news(
            symbol=ticker, limit=limit, provider=_provider
        )

    # ------------------------------------------------------------------
    # Helpers (kept for any code that calls them directly)
    # ------------------------------------------------------------------

    @staticmethod
    def _period_to_dates(period: str) -> tuple[str, str]:
        return get_obb_client()._period_to_dates(period)


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_default_provider: Optional[OpenBBDataProvider] = None


def get_data_provider() -> OpenBBDataProvider:
    """Return a module-level singleton OpenBBDataProvider instance."""
    global _default_provider
    if _default_provider is None:
        _default_provider = OpenBBDataProvider()
    return _default_provider
