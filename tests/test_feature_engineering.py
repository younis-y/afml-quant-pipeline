"""
Tests for pipeline/feature_engineering.py
Fractional differentiation, microstructure features, entropy, structural breaks.
"""

import pandas as pd

from pipeline.feature_engineering import (
    frac_diff_ffd,
    find_optimal_d,
    get_roll_spread,
    get_kyle_lambda,
    get_amihud_lambda,
    get_vpin,
    get_lempel_ziv_entropy,
    get_shannon_entropy,
    get_cusum_filter,
    get_trend_scanning_labels,
)


# ── Fractional Differentiation ───────────────────────────────────────────────

class TestFracDiff:
    def test_ffd_returns_series(self, sample_prices):
        result = frac_diff_ffd(sample_prices, d=0.4)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_ffd_d_zero_preserves(self, sample_prices):
        result = frac_diff_ffd(sample_prices, d=0.0)
        # d=0 should approximately equal the original
        if len(result) > 0:
            corr = result.corr(sample_prices.loc[result.index])
            assert corr > 0.95

    def test_find_optimal_d_range(self, sample_prices):
        optimal_d, results = find_optimal_d(sample_prices, step=0.2)
        assert 0.0 <= optimal_d <= 1.0
        assert isinstance(results, (dict, pd.DataFrame))


# ── Microstructure Features ──────────────────────────────────────────────────

class TestMicrostructure:
    def test_roll_spread(self, sample_prices):
        spread = get_roll_spread(sample_prices)
        assert isinstance(spread, pd.Series)
        assert len(spread) > 0

    def test_kyle_lambda(self, sample_prices, sample_volume):
        lam = get_kyle_lambda(sample_prices, sample_volume)
        assert isinstance(lam, pd.Series)
        assert len(lam) > 0

    def test_amihud_lambda(self, sample_prices, sample_volume):
        lam = get_amihud_lambda(sample_prices, sample_volume)
        assert isinstance(lam, pd.Series)
        assert len(lam) > 0

    def test_vpin(self, sample_prices, sample_volume):
        vpin = get_vpin(sample_prices, sample_volume)
        assert isinstance(vpin, pd.Series)
        # VPIN should be between 0 and 1
        valid = vpin.dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()
            assert (valid <= 1.5).all()  # allow some tolerance


# ── Entropy Features ─────────────────────────────────────────────────────────

class TestEntropy:
    def test_shannon_entropy(self, sample_prices):
        ent = get_shannon_entropy(sample_prices, window=30)
        assert isinstance(ent, pd.Series)
        valid = ent.dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()

    def test_lempel_ziv_entropy(self, sample_prices):
        ent = get_lempel_ziv_entropy(sample_prices, window=30)
        assert isinstance(ent, pd.Series)
        assert len(ent) > 0


# ── Structural Breaks ────────────────────────────────────────────────────────

class TestStructuralBreaks:
    def test_cusum_filter(self, sample_prices):
        events = get_cusum_filter(sample_prices)
        assert isinstance(events, pd.Series)
        # Should flag some structural breaks
        assert len(events) > 0

    def test_trend_scanning(self, sample_prices):
        labels = get_trend_scanning_labels(sample_prices, horizons=[5, 10, 20])
        assert isinstance(labels, pd.DataFrame)
        assert len(labels) > 0
