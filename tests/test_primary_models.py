"""
Tests for pipeline/primary_models.py — Pluggable Primary Models
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.primary_models import (
    PrimaryModel,
    RSIMeanReversion,
    BollingerBreakout,
    MACDCrossover,
    MomentumRotation,
    list_models,
    get_model,
    register_model,
)


# ── Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def prices():
    """500-bar synthetic price series."""
    np.random.seed(42)
    n = 500
    rets = np.random.randn(n) * 0.015 + 0.0002
    return pd.Series(
        100 * np.exp(np.cumsum(rets)),
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
        name="Close",
    )


# ── Common contract checks ───────────────────────────────────────────────

def _check_signal_contract(model, prices):
    """Every model must return a DataFrame with 'signal' in {-1, 0, 1}."""
    result = model.generate_signals(prices)
    assert isinstance(result, pd.DataFrame)
    assert "signal" in result.columns
    assert set(result["signal"].unique()).issubset({-1, 0, 1})
    assert len(result) > 0
    return result


# ── RSI ───────────────────────────────────────────────────────────────────

class TestRSI:
    def test_signal_contract(self, prices):
        _check_signal_contract(RSIMeanReversion(), prices)

    def test_has_rsi_column(self, prices):
        df = RSIMeanReversion().generate_signals(prices)
        assert "rsi" in df.columns
        assert df["rsi"].between(0, 100).all()

    def test_custom_params(self, prices):
        model = RSIMeanReversion(period=7, oversold=25, overbought=75)
        assert model.get_params()["period"] == 7
        _check_signal_contract(model, prices)


# ── Bollinger ─────────────────────────────────────────────────────────────

class TestBollinger:
    def test_signal_contract(self, prices):
        _check_signal_contract(BollingerBreakout(), prices)

    def test_band_columns(self, prices):
        df = BollingerBreakout().generate_signals(prices)
        for col in ("bb_mid", "bb_upper", "bb_lower", "bb_pct"):
            assert col in df.columns

    def test_upper_above_lower(self, prices):
        df = BollingerBreakout().generate_signals(prices)
        assert (df["bb_upper"] >= df["bb_lower"]).all()


# ── MACD ──────────────────────────────────────────────────────────────────

class TestMACD:
    def test_signal_contract(self, prices):
        _check_signal_contract(MACDCrossover(), prices)

    def test_macd_columns(self, prices):
        df = MACDCrossover().generate_signals(prices)
        for col in ("macd", "macd_signal", "macd_hist", "crossover"):
            assert col in df.columns

    def test_crossover_values(self, prices):
        df = MACDCrossover().generate_signals(prices)
        assert set(df["crossover"].unique()).issubset({-1, 0, 1})


# ── Momentum ──────────────────────────────────────────────────────────────

class TestMomentum:
    def test_signal_contract(self, prices):
        _check_signal_contract(MomentumRotation(), prices)

    def test_with_volume(self, prices):
        vol = pd.Series(
            np.random.lognormal(10, 1, len(prices)),
            index=prices.index,
        )
        df = MomentumRotation().generate_signals(prices, volume=vol)
        assert "vol_trend" in df.columns

    def test_mom_score_column(self, prices):
        df = MomentumRotation().generate_signals(prices)
        assert "mom_score" in df.columns


# ── Registry / Factory ────────────────────────────────────────────────────

class TestRegistry:
    def test_list_models(self):
        models = list_models()
        assert isinstance(models, list)
        assert "rsi" in models
        assert "macd" in models
        assert "bollinger" in models
        assert "momentum" in models

    def test_get_model(self, prices):
        for name in list_models():
            model = get_model(name)
            _check_signal_contract(model, prices)

    def test_get_model_with_kwargs(self, prices):
        model = get_model("rsi", period=7, oversold=20, overbought=80)
        assert model.get_params()["period"] == 7

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model")

    def test_register_custom(self, prices):
        class MyModel(PrimaryModel):
            def generate_signals(self, close):
                df = pd.DataFrame(index=close.index)
                df["signal"] = 1
                return df
            def get_params(self):
                return {}

        register_model("custom", MyModel)
        assert "custom" in list_models()
        _check_signal_contract(get_model("custom"), prices)
