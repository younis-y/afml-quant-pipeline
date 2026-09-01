"""
AFML Quant Pipeline - Strategy Optimiser (VectorBT)
Grid-search optimisation over strategy parameters using vectorised backtesting.

Usage:
    opt = StrategyOptimiser()
    result = opt.optimize_ma_crossover("SPY")
    print(result.best_params)
    print(result.best_sharpe)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional VectorBT import — gracefully degrade if not installed.
try:
    import vectorbt as vbt
    _VBT_AVAILABLE = True
except Exception:
    _VBT_AVAILABLE = False
    logger.warning("vectorbt not installed; StrategyOptimiser will use a fallback implementation.")


# =============================================================================
# RESULT DATA CLASS
# =============================================================================

@dataclass
class OptimizationResult:
    """Results returned by a StrategyOptimiser run."""
    best_params: dict
    sharpe_grid: pd.DataFrame          # 2-D Sharpe-ratio heatmap data
    best_sharpe: float
    best_returns: pd.Series = field(default_factory=pd.Series)   # for QuantStats


# =============================================================================
# OPTIMISER
# =============================================================================

class StrategyOptimiser:
    """
    Grid-search parameter optimiser backed by VectorBT.

    Each ``optimize_*`` method:
    1. Downloads / receives price data via OBBClient.
    2. Computes the indicator over the full parameter grid (vectorised).
    3. Runs ``vbt.Portfolio.from_signals`` for every combination at once.
    4. Extracts Sharpe ratios, finds the argmax, returns an OptimizationResult.

    The ``OptimizationResult.best_params`` dict can be fed directly into the
    MetaLabelingPipeline so the pipeline uses empirically-optimised parameters.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_ma_crossover(
        self,
        ticker: str,
        fast_range=range(5, 50, 5),
        slow_range=range(20, 200, 10),
        period: str = "2y",
    ) -> OptimizationResult:
        """
        Grid-search over fast/slow MA-crossover windows.

        Args:
            ticker:     Ticker symbol.
            fast_range: Iterable of fast-window values.
            slow_range: Iterable of slow-window values.
            period:     Historical data period for optimisation.

        Returns:
            OptimizationResult with best (fast, slow) params.
        """
        close = self._get_close(ticker, period)

        fast_windows = list(fast_range)
        slow_windows = [s for s in slow_range if s > max(fast_windows)]

        if _VBT_AVAILABLE:
            return self._vbt_ma_crossover(close, fast_windows, slow_windows)
        else:
            return self._fallback_ma_crossover(close, fast_windows, slow_windows)

    def optimize_rsi(
        self,
        ticker: str,
        rsi_levels=range(20, 80, 5),
        window_range=range(7, 30, 1),
        period: str = "2y",
    ) -> OptimizationResult:
        """
        Grid-search over RSI window and oversold/overbought levels.

        Returns:
            OptimizationResult with best (rsi_window, os_level, ob_level) params.
        """
        close = self._get_close(ticker, period)

        windows = list(window_range)
        levels = list(rsi_levels)

        if _VBT_AVAILABLE:
            return self._vbt_rsi(close, windows, levels)
        else:
            return self._fallback_rsi(close, windows, levels)

    def optimize_bollinger(
        self,
        ticker: str,
        window_range=range(10, 50, 5),
        std_range=None,
        period: str = "2y",
    ) -> OptimizationResult:
        """
        Grid-search over Bollinger Band window and standard-deviation multiplier.

        Returns:
            OptimizationResult with best (bb_window, bb_std) params.
        """
        if std_range is None:
            std_range = [1.5, 2.0, 2.5, 3.0]

        close = self._get_close(ticker, period)

        windows = list(window_range)
        stds = list(std_range)

        if _VBT_AVAILABLE:
            return self._vbt_bollinger(close, windows, stds)
        else:
            return self._fallback_bollinger(close, windows, stds)

    # ------------------------------------------------------------------
    # VectorBT implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _vbt_ma_crossover(
        close: pd.Series,
        fast_windows: list,
        slow_windows: list,
    ) -> OptimizationResult:
        # Build all fast/slow MAs at once
        fast_ma = vbt.MA.run(close, window=fast_windows, short_name="fast")
        slow_ma = vbt.MA.run(close, window=slow_windows, short_name="slow")

        # Broadcast: entries when fast crosses above slow, exits the opposite
        # vbt.MA.run returns a (n_bars, n_params) MA object
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)

        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            freq="D",
            init_cash=100_000,
        )

        sharpe = pf.sharpe_ratio()

        # sharpe is a Series with MultiIndex (fast_window, slow_window)
        sharpe_grid = sharpe.unstack()
        sharpe_grid.index.name = "fast_window"
        sharpe_grid.columns.name = "slow_window"

        best_idx = sharpe.idxmax()
        best_fast, best_slow = best_idx if isinstance(best_idx, tuple) else (best_idx, slow_windows[0])
        best_sharpe = float(sharpe.max())

        # Best portfolio returns
        best_pf = vbt.Portfolio.from_signals(
            close,
            fast_ma.ma_crossed_above(slow_ma)[[best_idx]],
            fast_ma.ma_crossed_below(slow_ma)[[best_idx]],
            freq="D",
            init_cash=100_000,
        )
        best_returns = best_pf.returns()
        if hasattr(best_returns, "iloc"):
            best_returns = best_returns.iloc[:, 0] if best_returns.ndim > 1 else best_returns

        return OptimizationResult(
            best_params={"fast_window": int(best_fast), "slow_window": int(best_slow)},
            sharpe_grid=sharpe_grid.fillna(0),
            best_sharpe=best_sharpe,
            best_returns=best_returns,
        )

    @staticmethod
    def _vbt_rsi(
        close: pd.Series,
        windows: list,
        levels: list,
    ) -> OptimizationResult:
        # We treat levels as oversold thresholds; overbought = 100 - level
        rsi = vbt.RSI.run(close, window=windows, short_name="rsi")

        sharpes: dict = {}
        for w_idx, w in enumerate(windows):
            for lvl in levels:
                ob = 100 - lvl
                rsi_col = rsi.rsi.iloc[:, w_idx]
                entries = (rsi_col < lvl).shift(1).fillna(False)
                exits = (rsi_col > ob).shift(1).fillna(False)
                pf = vbt.Portfolio.from_signals(close, entries, exits, freq="D", init_cash=100_000)
                sharpes[(w, lvl)] = float(pf.sharpe_ratio())

        sharpe_series = pd.Series(sharpes)
        sharpe_grid = sharpe_series.unstack()
        sharpe_grid.index.name = "rsi_window"
        sharpe_grid.columns.name = "os_level"

        best_idx = sharpe_series.idxmax()
        best_window, best_level = best_idx
        best_sharpe = float(sharpe_series.max())

        best_rsi = rsi.rsi.iloc[:, windows.index(best_window)]
        best_entries = (best_rsi < best_level).shift(1).fillna(False)
        best_exits = (best_rsi > (100 - best_level)).shift(1).fillna(False)
        best_pf = vbt.Portfolio.from_signals(close, best_entries, best_exits, freq="D", init_cash=100_000)

        return OptimizationResult(
            best_params={
                "rsi_window": int(best_window),
                "os_level": int(best_level),
                "ob_level": int(100 - best_level),
            },
            sharpe_grid=sharpe_grid.fillna(0),
            best_sharpe=best_sharpe,
            best_returns=best_pf.returns(),
        )

    @staticmethod
    def _vbt_bollinger(
        close: pd.Series,
        windows: list,
        stds: list,
    ) -> OptimizationResult:
        sharpes: dict = {}
        for w in windows:
            for s in stds:
                bb = vbt.BBANDS.run(close, window=w, alpha=s)
                entries = close < bb.lower   # price below lower band → buy
                exits = close > bb.upper     # price above upper band → sell
                pf = vbt.Portfolio.from_signals(close, entries, exits, freq="D", init_cash=100_000)
                sharpes[(w, s)] = float(pf.sharpe_ratio())

        sharpe_series = pd.Series(sharpes)
        sharpe_grid = sharpe_series.unstack()
        sharpe_grid.index.name = "bb_window"
        sharpe_grid.columns.name = "bb_std"

        best_idx = sharpe_series.idxmax()
        best_window, best_std = best_idx
        best_sharpe = float(sharpe_series.max())

        bb_best = vbt.BBANDS.run(close, window=best_window, alpha=best_std)
        best_entries = close < bb_best.lower
        best_exits = close > bb_best.upper
        best_pf = vbt.Portfolio.from_signals(close, best_entries, best_exits, freq="D", init_cash=100_000)

        return OptimizationResult(
            best_params={"bb_window": int(best_window), "bb_std": float(best_std)},
            sharpe_grid=sharpe_grid.fillna(0),
            best_sharpe=best_sharpe,
            best_returns=best_pf.returns(),
        )

    # ------------------------------------------------------------------
    # Pure-pandas fallbacks (no VectorBT dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_ma_crossover(
        close: pd.Series,
        fast_windows: list,
        slow_windows: list,
    ) -> OptimizationResult:
        """Simplified fallback using pandas-only rolling MA crossover."""
        sharpes: dict = {}
        for f in fast_windows:
            for s in slow_windows:
                if s <= f:
                    continue
                fast = close.rolling(f).mean()
                slow = close.rolling(s).mean()
                signal = (fast > slow).astype(int)
                rets = close.pct_change() * signal.shift(1)
                sharpes[(f, s)] = _annualised_sharpe(rets)

        sharpe_series = pd.Series(sharpes)
        sharpe_grid = sharpe_series.unstack()
        sharpe_grid.index.name = "fast_window"
        sharpe_grid.columns.name = "slow_window"

        best_idx = sharpe_series.idxmax()
        best_fast, best_slow = best_idx
        best_sharpe = float(sharpe_series.max())

        f_ma = close.rolling(best_fast).mean()
        s_ma = close.rolling(best_slow).mean()
        best_signal = (f_ma > s_ma).astype(int)
        best_returns = close.pct_change() * best_signal.shift(1)

        return OptimizationResult(
            best_params={"fast_window": int(best_fast), "slow_window": int(best_slow)},
            sharpe_grid=sharpe_grid.fillna(0),
            best_sharpe=best_sharpe,
            best_returns=best_returns.dropna(),
        )

    @staticmethod
    def _fallback_rsi(
        close: pd.Series,
        windows: list,
        levels: list,
    ) -> OptimizationResult:
        sharpes: dict = {}
        for w in windows:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
            rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
            for lvl in levels:
                signal = (rsi < lvl).astype(int)
                rets = close.pct_change() * signal.shift(1)
                sharpes[(w, lvl)] = _annualised_sharpe(rets)

        sharpe_series = pd.Series(sharpes)
        sharpe_grid = sharpe_series.unstack()
        sharpe_grid.index.name = "rsi_window"
        sharpe_grid.columns.name = "os_level"

        best_idx = sharpe_series.idxmax()
        best_window, best_level = best_idx
        best_sharpe = float(sharpe_series.max())

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(best_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(best_window).mean()
        best_rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        best_signal = (best_rsi < best_level).astype(int)
        best_returns = close.pct_change() * best_signal.shift(1)

        return OptimizationResult(
            best_params={
                "rsi_window": int(best_window),
                "os_level": int(best_level),
                "ob_level": int(100 - best_level),
            },
            sharpe_grid=sharpe_grid.fillna(0),
            best_sharpe=best_sharpe,
            best_returns=best_returns.dropna(),
        )

    @staticmethod
    def _fallback_bollinger(
        close: pd.Series,
        windows: list,
        stds: list,
    ) -> OptimizationResult:
        sharpes: dict = {}
        for w in windows:
            ma = close.rolling(w).mean()
            std = close.rolling(w).std()
            for s in stds:
                lower = ma - s * std
                upper = ma + s * std
                long_signal = (close < lower).astype(int)
                rets = close.pct_change() * long_signal.shift(1)
                sharpes[(w, s)] = _annualised_sharpe(rets)

        sharpe_series = pd.Series(sharpes)
        sharpe_grid = sharpe_series.unstack()
        sharpe_grid.index.name = "bb_window"
        sharpe_grid.columns.name = "bb_std"

        best_idx = sharpe_series.idxmax()
        best_window, best_std = best_idx
        best_sharpe = float(sharpe_series.max())

        ma = close.rolling(best_window).mean()
        std = close.rolling(best_window).std()
        best_lower = ma - best_std * std
        best_signal = (close < best_lower).astype(int)
        best_returns = close.pct_change() * best_signal.shift(1)

        return OptimizationResult(
            best_params={"bb_window": int(best_window), "bb_std": float(best_std)},
            sharpe_grid=sharpe_grid.fillna(0),
            best_sharpe=best_sharpe,
            best_returns=best_returns.dropna(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_close(self, ticker: str, period: str) -> pd.Series:
        from utils.obb_client import get_obb_client

        start, end = get_obb_client()._period_to_dates(period)
        df = get_obb_client().get_price_history(
            symbol=ticker, start_date=start, end_date=end
        )

        if "Close" not in df.columns:
            raise ValueError(f"No 'Close' column in data for {ticker}")

        return df["Close"].dropna()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _annualised_sharpe(returns: pd.Series, periods: int = 252) -> float:
    """Simple annualised Sharpe ratio."""
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods))
