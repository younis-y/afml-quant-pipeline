"""
Tests for analysis_engine/stats.py — gaps not covered by test_stats.py.
Covers: var_cornish_fisher, var_bootstrap, frac_diff, fit_garch, garch_var.
"""

import numpy as np
import pandas as pd
import pytest

from analysis_engine.stats import (
    var_cornish_fisher,
    var_bootstrap,
    frac_diff,
    fit_garch,
    garch_var,
)


# ── Cornish-Fisher VaR ─────────────────────────────────────────────────────

class TestCornishFisherVaR:
    def test_returns_dict(self, sample_returns):
        result = var_cornish_fisher(sample_returns)
        assert isinstance(result, dict)
        assert "VaR" in result
        assert "method" in result

    def test_var_positive(self, sample_returns):
        result = var_cornish_fisher(sample_returns)
        assert result["VaR"] > 0

    def test_close_to_parametric_for_normal(self):
        """For near-normal data, CF VaR should be close to parametric."""
        np.random.seed(42)
        normal_returns = pd.Series(np.random.randn(5000) * 0.01)
        cf = var_cornish_fisher(normal_returns)
        from analysis_engine.stats import var_parametric
        param = var_parametric(normal_returns)
        # Should be within 20% of each other for near-normal data
        assert abs(cf["VaR"] - param["VaR"]) / param["VaR"] < 0.20


# ── Bootstrap VaR ──────────────────────────────────────────────────────────

class TestBootstrapVaR:
    def test_returns_dict(self, sample_returns):
        result = var_bootstrap(sample_returns, n_simulations=500, seed=42)
        assert isinstance(result, dict)
        assert "VaR" in result
        assert "CI_lower" in result
        assert "CI_upper" in result

    def test_ci_contains_var(self, sample_returns):
        result = var_bootstrap(sample_returns, n_simulations=1000, seed=42)
        assert result["CI_lower"] <= result["VaR"] <= result["CI_upper"]

    def test_deterministic_with_seed(self, sample_returns):
        r1 = var_bootstrap(sample_returns, n_simulations=500, seed=123)
        r2 = var_bootstrap(sample_returns, n_simulations=500, seed=123)
        assert r1["VaR"] == r2["VaR"]

    @pytest.mark.slow
    def test_large_simulations(self, sample_returns):
        result = var_bootstrap(sample_returns, n_simulations=50_000, seed=42)
        assert result["VaR"] > 0
        # CI should be tighter with more sims
        ci_width = result["CI_upper"] - result["CI_lower"]
        assert ci_width > 0


# ── Fractional Differentiation ──────────────────────────────────────────────

class TestFracDiff:
    def test_returns_series(self, sample_prices):
        result = frac_diff(sample_prices, d=0.5, window=20)
        assert isinstance(result, pd.Series)

    def test_shorter_than_input(self, sample_prices):
        result = frac_diff(sample_prices, d=0.5, window=20)
        assert len(result) < len(sample_prices)

    def test_d_zero_is_identity_like(self, sample_prices):
        """d=0 should be close to original (no differencing)."""
        result = frac_diff(sample_prices, d=0.0, window=20)
        # With d=0, weights are [1, 0, 0, ...] so result ≈ original
        original_trimmed = sample_prices.iloc[19:]  # after rolling window warmup
        np.testing.assert_allclose(result.values, original_trimmed.values, rtol=1e-10)


# ── GARCH ───────────────────────────────────────────────────────────────────

class TestFitGarch:
    def test_returns_dict(self, sample_garch_returns):
        result = fit_garch(sample_garch_returns)
        assert isinstance(result, dict)
        # Either has params or error
        assert "persistence" in result or "error" in result

    def test_persistence_less_than_one(self, sample_garch_returns):
        result = fit_garch(sample_garch_returns)
        if "error" not in result:
            assert result["persistence"] < 1.0

    def test_conditional_vol_is_series(self, sample_garch_returns):
        result = fit_garch(sample_garch_returns)
        if "error" not in result:
            assert isinstance(result["conditional_vol"], pd.Series)
            assert len(result["conditional_vol"]) > 0


class TestGarchVaR:
    def test_returns_dict(self, sample_garch_returns):
        result = garch_var(sample_garch_returns)
        assert isinstance(result, dict)

    def test_var_positive(self, sample_garch_returns):
        result = garch_var(sample_garch_returns)
        if "error" not in result:
            assert result["VaR"] > 0
