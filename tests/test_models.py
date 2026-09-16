"""
Tests for pipeline/models.py
Financial models, ensemble, feature importance, Sharpe ratio statistics.
"""

import numpy as np
import pandas as pd

from pipeline.models import (
    FinancialRandomForest,
    compute_sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    mean_decrease_impurity,
)


# ── Financial Random Forest ──────────────────────────────────────────────────

class TestFinancialRandomForest:
    def test_fit_predict(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        rf = FinancialRandomForest(n_estimators=10, max_depth=3, min_samples_leaf=5)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == n
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        rf = FinancialRandomForest(n_estimators=10, max_depth=3)
        rf.fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (n, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importance(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        rf = FinancialRandomForest(n_estimators=10, max_depth=3)
        rf.fit(X, y)
        imp = rf.get_feature_importance(list(X.columns))
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]

    def test_oob_score(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        rf = FinancialRandomForest(n_estimators=50, max_depth=3)
        rf.fit(X, y)
        oob = rf.oob_score
        assert isinstance(oob, (float, property))


# ── MDI Feature Importance ───────────────────────────────────────────────────

class TestMDI:
    def test_mdi_returns_series(self, sample_features, sample_binary_labels):
        from sklearn.ensemble import RandomForestClassifier
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)
        imp = mean_decrease_impurity(clf, list(X.columns))
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]


# ── Sharpe Ratio Statistics ──────────────────────────────────────────────────

class TestSharpeStatistics:
    def test_compute_sharpe(self, sample_returns):
        sr = compute_sharpe_ratio(sample_returns)
        assert isinstance(sr, float)
        assert np.isfinite(sr)

    def test_psr(self):
        psr = probabilistic_sharpe_ratio(
            observed_sr=1.5,
            benchmark_sr=0.0,
            n_obs=252,
            skew=0,
            kurtosis=3
        )
        assert 0 <= psr <= 1

    def test_psr_low_sr_low_probability(self):
        psr = probabilistic_sharpe_ratio(
            observed_sr=0.1,
            benchmark_sr=1.0,
            n_obs=252,
        )
        assert psr < 0.5

    def test_dsr(self):
        dsr = deflated_sharpe_ratio(
            observed_sr=1.5,
            sr_std=0.5,
            n_trials=10,
            n_obs=252,
        )
        assert isinstance(dsr, float)
        assert 0 <= dsr <= 1

    def test_dsr_many_trials_lower(self):
        """More trials should reduce the DSR (multiple testing penalty)."""
        dsr_few = deflated_sharpe_ratio(
            observed_sr=1.5, sr_std=0.5, n_trials=2, n_obs=252
        )
        dsr_many = deflated_sharpe_ratio(
            observed_sr=1.5, sr_std=0.5, n_trials=100, n_obs=252
        )
        assert dsr_many <= dsr_few
