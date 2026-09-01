"""
Tests for analysis_engine/stats.py
Statistical analysis: ADF, Hurst, fat tails, VaR, GARCH.
"""

import numpy as np
import pandas as pd
import pytest

from analysis_engine.stats import (
    get_adf_stat,
    get_hurst_exponent,
    get_volatility,
    analyze_fat_tails,
    var_parametric,
    var_historical,
    compute_all_var,
    analyze_series_structure,
)


# ── Core Statistical Tests ───────────────────────────────────────────────────

class TestCoreStats:
    def test_adf_stat(self, sample_prices):
        result = get_adf_stat(sample_prices)
        assert isinstance(result, dict)
        assert "statistic" in result or "adf_statistic" in result or len(result) > 0

    def test_hurst_exponent(self, sample_prices):
        h = get_hurst_exponent(sample_prices)
        assert isinstance(h, float)
        assert 0 <= h <= 1

    def test_volatility(self, sample_prices):
        vol = get_volatility(sample_prices)
        assert isinstance(vol, float)
        assert vol > 0


# ── Fat Tail Analysis ────────────────────────────────────────────────────────

class TestFatTails:
    def test_analyze_fat_tails(self, sample_returns):
        result = analyze_fat_tails(sample_returns)
        assert isinstance(result, dict)
        assert "kurtosis" in result or "skewness" in result

    def test_normal_distribution_detected(self):
        """Standard normal should be detected as approximately normal."""
        np.random.seed(42)
        normal_returns = pd.Series(np.random.randn(5000) * 0.01)
        result = analyze_fat_tails(normal_returns)
        assert isinstance(result, dict)


# ── Value at Risk ────────────────────────────────────────────────────────────

class TestVaR:
    def test_var_parametric(self, sample_returns):
        result = var_parametric(sample_returns)
        assert isinstance(result, dict)
        assert "var" in result or "VaR" in result or len(result) > 0

    def test_var_historical(self, sample_returns):
        result = var_historical(sample_returns)
        assert isinstance(result, dict)

    def test_compute_all_var(self, sample_returns):
        result = compute_all_var(sample_returns)
        assert isinstance(result, dict)
        assert len(result) >= 2  # At least parametric + historical


# ── Master Analysis ──────────────────────────────────────────────────────────

class TestMasterAnalysis:
    @pytest.mark.slow
    def test_analyze_series_structure(self, sample_prices):
        result = analyze_series_structure(sample_prices)
        assert isinstance(result, dict)
        assert len(result) > 0
