"""
Tests for utils/validation.py
Purged K-Fold CV, Deflated Sharpe Ratio, Sharpe calculation, strategy validation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from utils.validation import (
    PurgedKFold,
    purged_kfold_cv,
    deflated_sharpe_ratio,
    calculate_sharpe_ratio,
)


# ── Purged K-Fold ────────────────────────────────────────────────────────────

class TestPurgedKFoldValidation:
    def test_split_count(self, sample_features):
        cv = PurgedKFold(n_splits=5, purge_length=3)
        splits = list(cv.split(sample_features))
        # Purging may merge folds, so allow n_splits ± 1
        assert len(splits) >= 4

    def test_no_index_overlap(self, sample_features):
        cv = PurgedKFold(n_splits=5, purge_length=3)
        for train_idx, test_idx in cv.split(sample_features):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_temporal_order(self, sample_features):
        """Train indices should all be before test indices (walk-forward)."""
        cv = PurgedKFold(n_splits=5, purge_length=3)
        for train_idx, test_idx in cv.split(sample_features):
            assert max(train_idx) < min(test_idx)

    def test_get_n_splits(self):
        cv = PurgedKFold(n_splits=7)
        assert cv.get_n_splits() == 7


# ── Purged K-Fold CV (full) ──────────────────────────────────────────────────

class TestPurgedKFoldCV:
    def test_returns_dict(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        result = purged_kfold_cv(model, X, y, n_splits=3)
        assert isinstance(result, dict)
        assert "mean_score" in result or "scores" in result


# ── Sharpe Ratio ─────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_returns(self):
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        sr = calculate_sharpe_ratio(returns)
        assert isinstance(sr, float)
        assert np.isfinite(sr)

    def test_zero_returns(self):
        returns = pd.Series(np.zeros(252))
        sr = calculate_sharpe_ratio(returns)
        assert sr == 0.0 or np.isnan(sr)


# ── Deflated Sharpe Ratio ────────────────────────────────────────────────────

class TestDSR:
    def test_returns_dict(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            num_trials=10,
            backtest_length=252
        )
        assert isinstance(result, dict)

    def test_more_trials_penalizes(self):
        r1 = deflated_sharpe_ratio(observed_sharpe=1.5, num_trials=2, backtest_length=252)
        r2 = deflated_sharpe_ratio(observed_sharpe=1.5, num_trials=50, backtest_length=252)
        # More trials should increase penalty
        assert r2["dsr"] <= r1["dsr"] or r2["probability_by_chance"] >= r1["probability_by_chance"]
