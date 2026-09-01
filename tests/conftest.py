"""
AFML Quant Pipeline - Shared Test Fixtures
Provides synthetic data and pre-configured objects used across all test modules.

Also patches pathlib.Path.stat to handle macOS SIP PermissionErrors during collection.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# macOS SIP workaround — silently skip stat() on protected dotfiles
# ---------------------------------------------------------------------------
_orig_stat = pathlib.Path.stat


def _safe_stat(self, *args, **kwargs):
    try:
        return _orig_stat(self, *args, **kwargs)
    except PermissionError:
        import os
        return os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))


pathlib.Path.stat = _safe_stat


# ---------------------------------------------------------------------------
# Synthetic Price Data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    """Generate a realistic synthetic price series (500 bars)."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.015 + 0.0002
    prices = 100 * np.exp(np.cumsum(returns))
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture
def sample_ohlcv():
    """Generate OHLCV DataFrame (500 bars)."""
    np.random.seed(42)
    n = 500
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015 + 0.0002))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    opn = close * (1 + np.random.randn(n) * 0.003)
    volume = np.random.randint(10_000, 1_000_000, size=n).astype(float)
    return pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=index)


@pytest.fixture
def sample_volume(sample_ohlcv):
    """Volume series extracted from OHLCV."""
    return sample_ohlcv["Volume"]


@pytest.fixture
def long_prices():
    """Longer price series (2000 bars) for pipeline tests."""
    np.random.seed(123)
    n = 2000
    returns = np.random.randn(n) * 0.015 + 0.0002
    prices = 100 * np.exp(np.cumsum(returns))
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture
def long_volume():
    """Volume series matching long_prices."""
    np.random.seed(456)
    n = 2000
    vol = np.random.randint(10_000, 1_000_000, size=n).astype(float)
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(vol, index=index, name="Volume")


@pytest.fixture
def sample_returns(sample_prices):
    """Simple returns from sample prices."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_binary_labels():
    """Binary labels for classification tests (500 samples)."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], size=500, p=[0.4, 0.6]),
                     index=pd.date_range("2023-01-01", periods=500, freq="h"),
                     name="label")


@pytest.fixture
def sample_features():
    """Feature DataFrame for model tests (500 x 5)."""
    np.random.seed(42)
    n = 500
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "feat_3": np.random.randn(n),
        "feat_4": np.random.randn(n),
        "feat_5": np.random.randn(n),
    }, index=index)


# ---------------------------------------------------------------------------
# Pair Trading Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cointegrated_pair():
    """Two cointegrated price series (beta=0.7, n=500)."""
    np.random.seed(99)
    n = 500
    # Common stochastic trend
    trend = np.cumsum(np.random.randn(n) * 0.5)
    noise1 = np.random.randn(n) * 0.3
    noise2 = np.random.randn(n) * 0.3
    s1 = 50 + trend + noise1
    s2 = 30 + 0.7 * trend + noise2
    index = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(s1, index=index, name="Asset1"), pd.Series(s2, index=index, name="Asset2")


@pytest.fixture
def sample_independent_pair():
    """Two independent random walks (n=500)."""
    np.random.seed(77)
    n = 500
    s1 = 100 + np.cumsum(np.random.randn(n) * 0.5)
    s2 = 50 + np.cumsum(np.random.randn(n) * 0.8)
    index = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(s1, index=index, name="Indep1"), pd.Series(s2, index=index, name="Indep2")


# ---------------------------------------------------------------------------
# GARCH Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_garch_returns():
    """1000-observation return series suitable for GARCH modelling."""
    np.random.seed(2024)
    n = 1000
    returns = np.random.randn(n) * 0.015 + 0.0001
    # Inject volatility clustering
    for i in range(1, n):
        if abs(returns[i - 1]) > 0.02:
            returns[i] *= 1.5
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(returns, index=index, name="returns")
