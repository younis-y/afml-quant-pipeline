"""
AFML Quant Pipeline — OBB Client
============================
Single place in the codebase where ``from openbb import obb`` is called.
Modules that need OpenBB import ``get_obb_client()`` from here.

Error contract
--------------
- Price / fundamentals empty  → raise OBBClientError
- Consensus / news / profile empty → return {} / [] (optional enrichment)
- Any obb.* exception  → re-raise as OBBClientError(...) from exc
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class OBBClientError(RuntimeError):
    """Raised when an OBB call fails or returns empty mandatory data."""


class OBBClient:
    """
    Thin, stateless wrapper around the OpenBB SDK v4.x.

    All methods are *instance* methods so the singleton pattern works,
    but none carry mutable state — safe to call from multiple threads.
    """

    DEFAULT_PROVIDER = "yfinance"

    # ------------------------------------------------------------------
    # Price / OHLCV
    # ------------------------------------------------------------------

    def get_price_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars.

        Returns:
            DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].

        Raises:
            OBBClientError: on empty result or any API exception.
        """
        try:
            from openbb import obb

            result = obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_price_history({symbol}): {exc}"
            ) from exc

        df = self._normalise_ohlcv(df)

        if df.empty:
            raise OBBClientError(
                f"get_price_history({symbol!r}): empty result. "
                "Check the symbol — common mistakes: APPL→AAPL, GOOG→GOOGL, FB→META."
            )
        return df

    def get_quote(
        self,
        symbol: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> dict:
        """
        Fetch the latest quote (38-field flat dict).

        Raises:
            OBBClientError: on empty result or API exception.
        """
        try:
            from openbb import obb

            result = obb.equity.price.quote(symbol=symbol, provider=provider)
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(f"get_quote({symbol}): {exc}") from exc

        if df.empty:
            raise OBBClientError(f"get_quote({symbol!r}): empty result.")

        return df.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """
        Fetch financial metrics (41 fields, newest-first).

        Raises:
            OBBClientError: on empty result or API exception.
        """
        try:
            from openbb import obb

            result = obb.equity.fundamental.metrics(
                symbol=symbol,
                period=period,
                limit=limit,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_fundamentals_metrics({symbol}): {exc}"
            ) from exc

        if df.empty:
            raise OBBClientError(
                f"get_fundamentals_metrics({symbol!r}): empty result."
            )
        return df

    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """
        Fetch income statements.

        Raises:
            OBBClientError: on empty result or API exception.
        """
        try:
            from openbb import obb

            result = obb.equity.fundamental.income(
                symbol=symbol,
                period=period,
                limit=limit,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_income_statement({symbol}): {exc}"
            ) from exc

        if df.empty:
            raise OBBClientError(
                f"get_income_statement({symbol!r}): empty result."
            )
        return df

    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """Fetch balance sheets."""
        try:
            from openbb import obb

            result = obb.equity.fundamental.balance(
                symbol=symbol,
                period=period,
                limit=limit,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_balance_sheet({symbol}): {exc}"
            ) from exc

        if df.empty:
            raise OBBClientError(
                f"get_balance_sheet({symbol!r}): empty result."
            )
        return df

    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """Fetch cash flow statements."""
        try:
            from openbb import obb

            result = obb.equity.fundamental.cash(
                symbol=symbol,
                period=period,
                limit=limit,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_cash_flow({symbol}): {exc}"
            ) from exc

        if df.empty:
            raise OBBClientError(
                f"get_cash_flow({symbol!r}): empty result."
            )
        return df

    def get_dividends(
        self,
        symbol: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """Fetch dividend history."""
        try:
            from openbb import obb

            result = obb.equity.fundamental.dividends(
                symbol=symbol,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_dividends({symbol}): {exc}"
            ) from exc

        if df.empty:
            raise OBBClientError(
                f"get_dividends({symbol!r}): empty result."
            )
        return df

    # ------------------------------------------------------------------
    # Analyst / profile / news (optional — return {} / [] on failure)
    # ------------------------------------------------------------------

    def get_analyst_consensus(
        self,
        symbol: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> dict:
        """
        Fetch analyst price-target consensus.

        Returns:
            Flat dict (target_consensus, target_high, target_low,
            number_of_analysts, recommendation, …) or {} on any failure.
        """
        try:
            from openbb import obb

            result = obb.equity.estimates.consensus(
                symbol=symbol, provider=provider
            )
            df = self._to_df(result)
            if df.empty:
                return {}
            return df.iloc[0].to_dict()
        except Exception as exc:
            logger.debug(f"get_analyst_consensus({symbol}): {exc}")
            return {}

    def get_equity_profile(
        self,
        symbol: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> dict:
        """
        Fetch company profile (48 fields).

        Returns:
            Flat dict or {} on any failure.
        """
        try:
            from openbb import obb

            result = obb.equity.profile(symbol=symbol, provider=provider)
            df = self._to_df(result)
            if df.empty:
                return {}
            return df.iloc[0].to_dict()
        except Exception as exc:
            logger.debug(f"get_equity_profile({symbol}): {exc}")
            return {}

    def get_company_news(
        self,
        symbol: str,
        limit: int = 10,
        provider: str = DEFAULT_PROVIDER,
    ) -> list[dict]:
        """
        Fetch recent news articles.

        Returns:
            List of dicts with keys title / date / url / summary,
            or [] on any failure.
        """
        try:
            from openbb import obb

            result = obb.news.company(
                symbol=symbol, limit=limit, provider=provider
            )
            df = self._to_df(result)
            if df.empty:
                return []

            news = []
            for _, row in df.iterrows():
                news.append(
                    {
                        "title": str(row.get("title", "")),
                        "date": str(row.get("date", "")),
                        "url": str(row.get("url", "")),
                        "summary": str(
                            row.get("text", row.get("summary", ""))
                        ),
                    }
                )
            return news
        except Exception as exc:
            logger.debug(f"get_company_news({symbol}): {exc}")
            return []

    # ------------------------------------------------------------------
    # Market movers / screeners
    # ------------------------------------------------------------------

    def get_market_movers(self, screen: str) -> pd.DataFrame:
        """
        Fetch equity screener results.

        Args:
            screen: One of gainers / losers / active /
                    undervalued_growth / growth_tech.

        Returns:
            DataFrame of results.

        Raises:
            OBBClientError: on unknown screen name or API failure.
        """
        _VALID = {
            "gainers",
            "losers",
            "active",
            "undervalued_growth",
            "growth_tech",
        }
        if screen not in _VALID:
            raise OBBClientError(
                f"get_market_movers: unknown screen '{screen}'. "
                f"Valid: {sorted(_VALID)}"
            )

        try:
            from openbb import obb

            endpoint = getattr(obb.equity.discovery, screen, None)
            if endpoint is None:
                raise OBBClientError(
                    f"obb.equity.discovery.{screen} not available."
                )
            result = endpoint(provider="yfinance")
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_market_movers({screen}): {exc}"
            ) from exc

        if df.empty:
            raise OBBClientError(
                f"get_market_movers({screen!r}): empty result."
            )
        return df

    # ------------------------------------------------------------------
    # ETF / crypto / index
    # ------------------------------------------------------------------

    def get_etf_info(
        self,
        symbol: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> dict:
        """Fetch ETF info dict."""
        try:
            from openbb import obb

            result = obb.etf.info(symbol=symbol, provider=provider)
            df = self._to_df(result)
            if df.empty:
                raise OBBClientError(
                    f"get_etf_info({symbol!r}): empty result."
                )
            return df.iloc[0].to_dict()
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(f"get_etf_info({symbol}): {exc}") from exc

    def get_crypto_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """Fetch crypto OHLCV history."""
        try:
            from openbb import obb

            result = obb.crypto.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_crypto_history({symbol}): {exc}"
            ) from exc

        df = self._normalise_ohlcv(df)
        if df.empty:
            raise OBBClientError(
                f"get_crypto_history({symbol!r}): empty result."
            )
        return df

    def get_index_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        provider: str = DEFAULT_PROVIDER,
    ) -> pd.DataFrame:
        """Fetch index OHLCV history."""
        try:
            from openbb import obb

            result = obb.index.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
            )
            df = self._to_df(result)
        except OBBClientError:
            raise
        except Exception as exc:
            raise OBBClientError(
                f"get_index_history({symbol}): {exc}"
            ) from exc

        df = self._normalise_ohlcv(df)
        if df.empty:
            raise OBBClientError(
                f"get_index_history({symbol!r}): empty result."
            )
        return df

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename lowercase OBB columns → Title-case OHLCV, ensure DatetimeIndex.
        """
        rename = {}
        for col in df.columns:
            low = col.lower()
            if low == "open":
                rename[col] = "Open"
            elif low == "high":
                rename[col] = "High"
            elif low == "low":
                rename[col] = "Low"
            elif low in ("close", "adj_close", "adjusted_close"):
                rename[col] = "Close"
            elif low == "volume":
                rename[col] = "Volume"
        df = df.rename(columns=rename)

        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[keep].copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df

    @staticmethod
    def _to_df(result) -> pd.DataFrame:
        """
        Convert an OBB result object to a DataFrame.
        Tries result.to_df() first; falls back to model_dump iteration.
        """
        try:
            return result.to_df()
        except Exception:
            pass
        try:
            return pd.DataFrame([r.model_dump() for r in result.results])
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _period_to_dates(period: str) -> tuple[str, str]:
        """Convert shorthand period string to (start_iso, end_iso)."""
        end = datetime.today()
        mapping: dict[str, timedelta] = {
            "1d":  timedelta(days=1),
            "5d":  timedelta(days=5),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=182),
            "1y":  timedelta(days=365),
            "2y":  timedelta(days=730),
            "3y":  timedelta(days=365 * 3),
            "5y":  timedelta(days=365 * 5),
            "10y": timedelta(days=365 * 10),
            "ytd": timedelta(days=(end - end.replace(month=1, day=1)).days),
            "max": timedelta(days=365 * 30),
        }
        delta = mapping.get(period, timedelta(days=365))
        start = end - delta
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_obb_client: Optional[OBBClient] = None


def get_obb_client() -> OBBClient:
    """Return the module-level OBBClient singleton."""
    global _obb_client
    if _obb_client is None:
        _obb_client = OBBClient()
    return _obb_client
