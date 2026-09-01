"""
Tests for pipeline/meta_model.py
SMA Crossover primary model and MetaModel secondary model.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.meta_model import SMACrossover, MetaModel, MetaLabelingPipeline


# ── SMA Crossover ────────────────────────────────────────────────────────────

class TestSMACrossover:
    def test_generate_signals(self, long_prices):
        sma = SMACrossover(fast_period=20, slow_period=50)
        signals = sma.generate_signals(long_prices)
        assert isinstance(signals, pd.DataFrame)
        assert "signal" in signals.columns
        assert "sma_fast" in signals.columns
        assert "sma_slow" in signals.columns

    def test_signal_values(self, long_prices):
        sma = SMACrossover(fast_period=20, slow_period=50)
        signals = sma.generate_signals(long_prices)
        valid_vals = {-1, 0, 1}
        assert set(signals["signal"].unique()).issubset(valid_vals)

    def test_get_params(self):
        sma = SMACrossover(fast_period=10, slow_period=30)
        params = sma.get_params()
        assert params["fast_period"] == 10
        assert params["slow_period"] == 30


# ── MetaModel ────────────────────────────────────────────────────────────────

class TestMetaModel:
    def test_prepare_features(self, long_prices):
        from pipeline.data_processor import frac_diff_fixed
        frac = frac_diff_fixed(long_prices, d=0.4)
        mm = MetaModel(n_estimators=10, max_depth=3)
        features = mm.prepare_features(long_prices, frac)
        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] > 0

    def test_fit_predict(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        mm = MetaModel(n_estimators=10, max_depth=3)
        mm.fit(X, y)
        proba = mm.predict_probability(X)
        assert isinstance(proba, pd.Series)
        assert len(proba) == n
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        mm = MetaModel(n_estimators=10, max_depth=3)
        mm.fit(X, y)
        preds = mm.predict(X, threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_feature_importance(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        mm = MetaModel(n_estimators=10, max_depth=3)
        mm.fit(X, y)
        imp = mm.get_feature_importance()
        assert isinstance(imp, pd.Series)

    def test_evaluate(self, sample_features, sample_binary_labels):
        n = min(len(sample_features), len(sample_binary_labels))
        X = sample_features.iloc[:n]
        y = sample_binary_labels.iloc[:n]

        mm = MetaModel(n_estimators=10, max_depth=3)
        mm.fit(X, y)
        metrics = mm.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert "precision" in metrics or "accuracy" in metrics or len(metrics) > 0
