"""
Tests for pipeline/pair_trading.py — CointegrationTester & MeanReversionStrategy.
Pure math, no network/API calls.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.pair_trading import CointegrationTester, MeanReversionStrategy


# ── CointegrationTester ─────────────────────────────────────────────────────

class TestCointegrationTester:
    def test_cointegrated_pair_detected(self, sample_cointegrated_pair):
        s1, s2 = sample_cointegrated_pair
        result = CointegrationTester.engage_engle_granger(s1, s2)
        assert isinstance(result, dict)
        assert result["is_cointegrated"] is True

    def test_independent_pair_not_cointegrated(self, sample_independent_pair):
        s1, s2 = sample_independent_pair
        result = CointegrationTester.engage_engle_granger(s1, s2)
        assert isinstance(result, dict)
        assert result["is_cointegrated"] == False

    def test_hedge_ratio_returned(self, sample_cointegrated_pair):
        s1, s2 = sample_cointegrated_pair
        result = CointegrationTester.engage_engle_granger(s1, s2)
        assert "hedge_ratio" in result
        assert isinstance(result["hedge_ratio"], float)

    def test_result_keys(self, sample_cointegrated_pair):
        s1, s2 = sample_cointegrated_pair
        result = CointegrationTester.engage_engle_granger(s1, s2)
        expected_keys = {"hedge_ratio", "intercept", "correlation", "spread_adf_stat",
                         "spread_p_value", "is_cointegrated"}
        assert expected_keys.issubset(result.keys())


# ── MeanReversionStrategy ──────────────────────────────────────────────────

class TestMeanReversionStrategy:
    def test_backtest_returns_dict(self, sample_cointegrated_pair):
        s1, s2 = sample_cointegrated_pair
        strat = MeanReversionStrategy(lookback=20)
        result = strat.backtest(s1, s2)
        assert isinstance(result, dict)
        assert "ratio" in result
        assert "z_score" in result
        assert "mean" in result

    def test_z_score_length(self, sample_cointegrated_pair):
        s1, s2 = sample_cointegrated_pair
        strat = MeanReversionStrategy(lookback=20)
        result = strat.backtest(s1, s2)
        assert len(result["z_score"]) == len(s1)

    def test_ratio_calculation(self, sample_cointegrated_pair):
        s1, s2 = sample_cointegrated_pair
        strat = MeanReversionStrategy()
        result = strat.backtest(s1, s2)
        expected_ratio = s1 / s2
        pd.testing.assert_series_equal(result["ratio"], expected_ratio, check_names=False)

    def test_custom_lookback(self, sample_cointegrated_pair):
        s1, s2 = sample_cointegrated_pair
        strat = MeanReversionStrategy(lookback=50)
        result = strat.backtest(s1, s2)
        # First 49 z_scores should be NaN (rolling window not full)
        assert result["z_score"].iloc[:49].isna().all()
