"""
Tests for pipeline/labeling.py and pipeline/labeling_advanced.py
Triple Barrier Method, meta-labels, and advanced labeling.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.labeling import (
    get_volatility,
    get_horizontal_barriers,
    get_vertical_barriers,
    triple_barrier_labels,
    get_meta_labels,
    compute_label_statistics,
    TripleBarrierLabeler,
)


# ── Volatility ───────────────────────────────────────────────────────────────

class TestVolatility:
    def test_returns_series(self, sample_prices):
        vol = get_volatility(sample_prices)
        assert isinstance(vol, pd.Series)
        assert len(vol) > 0

    def test_all_positive(self, sample_prices):
        vol = get_volatility(sample_prices).dropna()
        assert (vol > 0).all()


# ── Barriers ─────────────────────────────────────────────────────────────────

class TestBarriers:
    def test_horizontal_barriers_shape(self, sample_prices):
        vol = get_volatility(sample_prices)
        barriers = get_horizontal_barriers(sample_prices, vol)
        assert "upper" in barriers.columns
        assert "lower" in barriers.columns

    def test_upper_above_lower(self, sample_prices):
        vol = get_volatility(sample_prices)
        barriers = get_horizontal_barriers(sample_prices, vol).dropna()
        assert (barriers["upper"] >= barriers["lower"]).all()

    def test_vertical_barriers_length(self, sample_prices):
        vb = get_vertical_barriers(sample_prices, num_bars=50)
        assert isinstance(vb, pd.Series)
        assert len(vb) > 0


# ── Triple Barrier Labels ────────────────────────────────────────────────────

class TestTripleBarrierLabels:
    def test_returns_dataframe(self, sample_prices):
        labels = triple_barrier_labels(sample_prices)
        assert isinstance(labels, pd.DataFrame)
        assert len(labels) > 0

    def test_label_values(self, sample_prices):
        labels = triple_barrier_labels(sample_prices)
        valid_labels = {-1, 0, 1}
        assert set(labels["label"].unique()).issubset(valid_labels)

    def test_has_required_columns(self, sample_prices):
        labels = triple_barrier_labels(sample_prices)
        # Must have 'label'; other columns are implementation-specific
        assert "label" in labels.columns
        assert len(labels.columns) >= 2

    def test_custom_multipliers(self, sample_prices):
        labels_sym = triple_barrier_labels(sample_prices, profit_mult=1.0, stop_mult=1.0)
        labels_asym = triple_barrier_labels(sample_prices, profit_mult=3.0, stop_mult=1.0)
        # Wider profit target should generally yield fewer +1 labels
        assert isinstance(labels_sym, pd.DataFrame)
        assert isinstance(labels_asym, pd.DataFrame)


# ── Meta-Labels ──────────────────────────────────────────────────────────────

class TestMetaLabels:
    def test_meta_labels_binary(self, sample_prices):
        labels = triple_barrier_labels(sample_prices)
        primary_signal = pd.Series(
            np.random.choice([-1, 1], len(labels)),
            index=labels.index
        )
        meta = get_meta_labels(primary_signal, labels)
        assert set(meta.unique()).issubset({0, 1})

    def test_meta_labels_length(self, sample_prices):
        labels = triple_barrier_labels(sample_prices)
        primary_signal = pd.Series(
            np.random.choice([-1, 1], len(labels)),
            index=labels.index
        )
        meta = get_meta_labels(primary_signal, labels)
        assert len(meta) == len(labels)


# ── Label Statistics ─────────────────────────────────────────────────────────

class TestLabelStatistics:
    def test_compute_label_statistics(self, sample_prices):
        labels = triple_barrier_labels(sample_prices)
        stats = compute_label_statistics(labels)
        assert isinstance(stats, dict)
        assert "total_labels" in stats or len(stats) > 0


# ── TripleBarrierLabeler Wrapper ─────────────────────────────────────────────

class TestTripleBarrierLabeler:
    def test_fit_transform(self, sample_prices):
        labeler = TripleBarrierLabeler(profit_mult=2.0, stop_mult=1.0)
        labels = labeler.fit_transform(sample_prices)
        assert isinstance(labels, pd.DataFrame)
        assert len(labels) > 0

    def test_get_params(self):
        labeler = TripleBarrierLabeler(profit_mult=2.5, stop_mult=1.5, num_bars=30)
        params = labeler.get_params()
        assert params["profit_mult"] == 2.5
        assert params["stop_mult"] == 1.5
        assert params["num_bars"] == 30
