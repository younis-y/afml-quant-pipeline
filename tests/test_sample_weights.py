"""
Tests for pipeline/sample_weights.py
Concurrent labels, sequential bootstrap, sample weights, and PurgedKFold CV.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.sample_weights import (
    get_indicator_matrix,
    get_num_concurrent_labels,
    get_average_uniqueness,
    sequential_bootstrap,
    get_sample_weights_by_time_decay,
    PurgedKFold,
)
from pipeline.labeling import triple_barrier_labels


# ── Helper ───────────────────────────────────────────────────────────────────

@pytest.fixture
def labeled_data(sample_prices):
    """Generate labels with t1 (end times) for sample_weights tests."""
    labels = triple_barrier_labels(sample_prices, num_bars=20)
    t1 = labels.index + pd.Timedelta(hours=20)
    t1 = pd.Series(t1, index=labels.index, name="t1")
    return labels, t1


# ── Concurrent Labels ────────────────────────────────────────────────────────

class TestConcurrentLabels:
    def test_indicator_matrix_shape(self, sample_prices, labeled_data):
        labels, t1 = labeled_data
        ind = get_indicator_matrix(sample_prices.index, t1)
        assert ind.shape[0] == len(sample_prices)
        assert ind.shape[1] == len(t1)

    def test_concurrent_labels_positive(self, sample_prices, labeled_data):
        _, t1 = labeled_data
        concurrent = get_num_concurrent_labels(sample_prices, t1)
        assert isinstance(concurrent, pd.Series)
        assert (concurrent >= 0).all()


# ── Average Uniqueness ───────────────────────────────────────────────────────

class TestAverageUniqueness:
    def test_uniqueness_between_zero_and_one(self, sample_prices, labeled_data):
        _, t1 = labeled_data
        uniqueness = get_average_uniqueness(t1, sample_prices)
        valid = uniqueness.dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()
            assert (valid <= 1).all()


# ── Sequential Bootstrap ────────────────────────────────────────────────────

class TestSequentialBootstrap:
    def test_returns_array(self, sample_prices, labeled_data):
        _, t1 = labeled_data
        ind = get_indicator_matrix(sample_prices.index, t1)
        bootstrap_idx = sequential_bootstrap(ind)
        assert isinstance(bootstrap_idx, (list, np.ndarray))
        assert len(bootstrap_idx) > 0

    def test_sample_length(self, sample_prices, labeled_data):
        _, t1 = labeled_data
        ind = get_indicator_matrix(sample_prices.index, t1)
        sample_len = min(50, ind.shape[1])
        bootstrap_idx = sequential_bootstrap(ind, sample_length=sample_len)
        assert len(bootstrap_idx) == sample_len


# ── Sample Weights ───────────────────────────────────────────────────────────

class TestSampleWeights:
    def test_time_decay_weights(self, labeled_data):
        _, t1 = labeled_data
        weights = get_sample_weights_by_time_decay(t1)
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(t1)
        assert (weights > 0).all()


# ── PurgedKFold ──────────────────────────────────────────────────────────────

class TestPurgedKFold:
    def test_n_splits(self, sample_features, sample_prices, labeled_data):
        _, t1 = labeled_data
        n = min(len(sample_features), len(t1))
        X = sample_features.iloc[:n]
        t1_aligned = t1.iloc[:n]
        cv = PurgedKFold(n_splits=3, t1=t1_aligned, pct_embargo=0.01)
        splits = list(cv.split(X))
        assert len(splits) == 3

    def test_no_overlap_train_test(self, sample_features, labeled_data):
        _, t1 = labeled_data
        n = min(len(sample_features), len(t1))
        X = sample_features.iloc[:n]
        t1_aligned = t1.iloc[:n]
        cv = PurgedKFold(n_splits=3, t1=t1_aligned, pct_embargo=0.01)
        for train_idx, test_idx in cv.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test indices must not overlap"
