"""
AFML Quant Pipeline - Pluggable Primary Models
ABC-based signal generators that slot into MetaLabelingPipeline.

Each model:
  - Inherits from PrimaryModel (ABC)
  - Implements generate_signals(close) → DataFrame with 'signal' column
  - Can be swapped into any pipeline expecting generate_signals()

Models included:
  1. SMACrossover      (already in meta_model.py — re-exported here)
  2. RSIMeanReversion  (RSI oversold/overbought)
  3. BollingerBreakout (Bollinger Band breakout)
  4. MACDCrossover     (MACD line / signal crossover)
  5. MomentumRotation  (volume-weighted momentum)

Usage:
    from pipeline.primary_models import get_model, list_models
    model = get_model("macd", fast=12, slow=26, signal=9)
    signals = model.generate_signals(close)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class PrimaryModel(ABC):
    """
    Abstract base class for all primary signal generators.

    Contract:
      - fit(close) → self              (optional training step)
      - generate_signals(close) → df   (must return DataFrame with 'signal' col)
      - get_params() → dict
    """

    @abstractmethod
    def generate_signals(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals from a close-price series.

        Returns:
            DataFrame with at least a 'signal' column (1, -1, 0).
        """
        ...

    def fit(self, close: pd.Series) -> "PrimaryModel":
        """Optional training step. No-op by default."""
        return self

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# =============================================================================
# RSI MEAN REVERSION
# =============================================================================

class RSIMeanReversion(PrimaryModel):
    """
    Relative Strength Index mean-reversion strategy.

    Buy when RSI drops below ``oversold``, sell when RSI rises above
    ``overbought``.  Hold until the opposite threshold is breached.

    Args:
        period: RSI look-back window (default 14)
        oversold: Buy threshold (default 30)
        overbought: Sell threshold (default 70)
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, close: pd.Series) -> pd.DataFrame:
        result = pd.DataFrame(index=close.index)

        # Calculate RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(span=self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        result["rsi"] = 100 - (100 / (1 + rs))

        # Signal logic
        result["signal"] = 0
        result.loc[result["rsi"] < self.oversold, "signal"] = 1   # buy
        result.loc[result["rsi"] > self.overbought, "signal"] = -1  # sell

        # Forward-fill signals to hold position
        result["signal"] = result["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

        return result.dropna(subset=["rsi"])

    def get_params(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }


# =============================================================================
# BOLLINGER BAND BREAKOUT
# =============================================================================

class BollingerBreakout(PrimaryModel):
    """
    Bollinger Band breakout strategy.

    Buy when price breaks above upper band, sell when price breaks below
    lower band.  Exit (flat) when price reverts to the middle band.

    Args:
        period: Band look-back window (default 20)
        num_std: Number of standard deviations (default 2.0)
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    def generate_signals(self, close: pd.Series) -> pd.DataFrame:
        result = pd.DataFrame(index=close.index)

        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        result["bb_mid"] = sma
        result["bb_upper"] = sma + self.num_std * std
        result["bb_lower"] = sma - self.num_std * std
        result["bb_pct"] = (close - result["bb_lower"]) / (result["bb_upper"] - result["bb_lower"])

        # Signal logic
        result["signal"] = 0
        result.loc[close > result["bb_upper"], "signal"] = 1
        result.loc[close < result["bb_lower"], "signal"] = -1

        # Hold until mean reversion (cross middle band)
        result["signal"] = result["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

        return result.dropna(subset=["bb_mid"])

    def get_params(self) -> Dict[str, Any]:
        return {"period": self.period, "num_std": self.num_std}


# =============================================================================
# MACD CROSSOVER
# =============================================================================

class MACDCrossover(PrimaryModel):
    """
    MACD line / signal line crossover strategy.

    Buy on bullish crossover (MACD > signal), sell on bearish crossover.

    Args:
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal_period: Signal EMA period (default 9)
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def generate_signals(self, close: pd.Series) -> pd.DataFrame:
        result = pd.DataFrame(index=close.index)

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        result["macd"] = ema_fast - ema_slow
        result["macd_signal"] = result["macd"].ewm(span=self.signal_period, adjust=False).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]

        # Signal: long when MACD > signal, short when MACD < signal
        result["signal"] = np.where(
            result["macd"] > result["macd_signal"], 1,
            np.where(result["macd"] < result["macd_signal"], -1, 0),
        )

        # Crossover detection
        prev = result["signal"].shift(1)
        result["crossover"] = np.where(
            (result["signal"] == 1) & (prev == -1), 1,
            np.where((result["signal"] == -1) & (prev == 1), -1, 0),
        )

        return result.dropna()

    def get_params(self) -> Dict[str, Any]:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "signal_period": self.signal_period,
        }


# =============================================================================
# MOMENTUM ROTATION
# =============================================================================

class MomentumRotation(PrimaryModel):
    """
    Volume-weighted momentum strategy.

    Computes a composite momentum score combining price momentum and
    volume trend.  Goes long when momentum is positive, short when
    negative.

    Args:
        fast_period: Short momentum look-back (default 10)
        slow_period: Long momentum look-back (default 50)
        volume_period: Volume trend look-back (default 20)
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        volume_period: int = 20,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_period = volume_period

    def generate_signals(
        self,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=close.index)

        # Price momentum
        result["mom_fast"] = close.pct_change(self.fast_period)
        result["mom_slow"] = close.pct_change(self.slow_period)

        # Composite momentum score (normalised)
        result["mom_score"] = (
            0.6 * result["mom_fast"] / result["mom_fast"].rolling(self.slow_period).std().replace(0, 1e-10)
            + 0.4 * result["mom_slow"] / result["mom_slow"].rolling(self.slow_period).std().replace(0, 1e-10)
        )

        # Volume trend (if volume available)
        if volume is not None:
            vol_ma = volume.rolling(self.volume_period).mean()
            result["vol_trend"] = (volume / vol_ma.replace(0, 1e-10)) - 1
            # Amplify momentum by volume trend
            result["mom_score"] = result["mom_score"] * (1 + result["vol_trend"].clip(-0.5, 0.5))

        # Signal
        result["signal"] = np.where(
            result["mom_score"] > 0, 1,
            np.where(result["mom_score"] < 0, -1, 0),
        )

        return result.dropna()

    def get_params(self) -> Dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "volume_period": self.volume_period,
        }


# =============================================================================
# REGISTRY AND FACTORY
# =============================================================================

_REGISTRY: Dict[str, type] = {
    "rsi": RSIMeanReversion,
    "bollinger": BollingerBreakout,
    "macd": MACDCrossover,
    "momentum": MomentumRotation,
}

# Lazy-register the existing SMACrossover from meta_model

def _register_sma():
    """Register SMACrossover if available."""
    try:
        from .meta_model import SMACrossover
        _REGISTRY["sma"] = SMACrossover
    except ImportError:
        pass

_register_sma()


def list_models() -> List[str]:
    """Return the names of all registered primary models."""
    return sorted(_REGISTRY.keys())


def get_model(name: str, **kwargs) -> PrimaryModel:
    """
    Factory: instantiate a model by name.

    Args:
        name: One of list_models()
        **kwargs: Forwarded to the model constructor

    Returns:
        PrimaryModel instance

    Raises:
        ValueError: Unknown model name
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list_models()}"
        )
    return _REGISTRY[key](**kwargs)


def register_model(name: str, cls: type) -> None:
    """
    Register a custom primary model.

    Args:
        name: Short name for the registry
        cls: Class that implements generate_signals(close)
    """
    _REGISTRY[name.lower()] = cls


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("Testing Primary Models...")
    print("=" * 60)

    # Generate synthetic prices
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.02 + 0.0002
    prices = pd.Series(
        100 * np.exp(np.cumsum(returns)),
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
        name="Close",
    )

    print(f"\nAvailable models: {list_models()}\n")

    for model_name in list_models():
        model = get_model(model_name)
        sigs = model.generate_signals(prices)
        longs = (sigs["signal"] == 1).sum()
        shorts = (sigs["signal"] == -1).sum()
        print(f"  {model_name:12s} | signals: {len(sigs):4d} | long: {longs:4d} | short: {shorts:4d}")
