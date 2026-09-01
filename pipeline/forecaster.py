"""
AFML Quant Pipeline - Multi-Model Ensemble Stock Forecaster
=======================================================
Runs ARIMA, ETS, GARCH, Random Forest, XGBoost, LightGBM, Ridge, Prophet,
and Monte Carlo GBM in parallel (all CPU cores) then combines them into a
weighted ensemble forecast.

Usage:
    fc = StockForecaster(horizon=30, n_simulations=50_000)
    result = fc.forecast("AAPL")
    print(result.summary())
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Optional model libraries ──────────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _STATSMODELS = True
except Exception:
    _STATSMODELS = False

try:
    from arch import arch_model
    _ARCH = True
except Exception:
    _ARCH = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    _SKLEARN = True
except Exception:
    _SKLEARN = False

try:
    import xgboost as xgb
    _XGB = True
except Exception:
    _XGB = False

try:
    import lightgbm as lgb
    _LGB = True
except Exception:
    _LGB = False

try:
    from prophet import Prophet
    _PROPHET = True
except Exception:
    _PROPHET = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelForecast:
    """Single-model forecast output."""
    name: str
    predicted_prices: pd.Series      # index = future DatetimeIndex, values = price
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None
    validation_rmse: float = float("nan")
    weight: float = 1.0              # set by ensemble after all models run


@dataclass
class ForecastResult:
    """Full ensemble forecast result returned to the dashboard."""
    ticker: str
    current_price: float
    horizon: int
    forecast_dates: pd.DatetimeIndex

    # Per-model results
    model_forecasts: Dict[str, ModelForecast] = field(default_factory=dict)

    # Ensemble
    ensemble_mean: pd.Series = field(default_factory=pd.Series)
    ensemble_lower: pd.Series = field(default_factory=pd.Series)  # 5th pct
    ensemble_upper: pd.Series = field(default_factory=pd.Series)  # 95th pct

    # Monte Carlo
    mc_paths: Optional[np.ndarray] = None          # (n_sims, horizon)
    mc_final_prices: Optional[np.ndarray] = None   # (n_sims,)
    mc_percentiles: Optional[pd.DataFrame] = None  # index=dates, cols=p5/p25/p50/p75/p95

    # Summary statistics
    predicted_return_pct: float = 0.0
    probability_of_gain: float = 0.0
    probability_of_loss_5pct: float = 0.0
    probability_of_loss_10pct: float = 0.0
    volatility_annual: float = 0.0
    models_used: List[str] = field(default_factory=list)

    def predicted_price(self) -> float:
        if len(self.ensemble_mean) > 0:
            return float(self.ensemble_mean.iloc[-1])
        return self.current_price

    def summary(self) -> str:
        lines = [
            f"Ticker:            {self.ticker}",
            f"Current price:     ${self.current_price:.2f}",
            f"Predicted price:   ${self.predicted_price():.2f}  ({self.predicted_return_pct:+.1f}%)",
            f"Horizon:           {self.horizon} days",
            f"P(gain):           {self.probability_of_gain:.1%}",
            f"P(loss > 5%):      {self.probability_of_loss_5pct:.1%}",
            f"Annual vol:        {self.volatility_annual:.1%}",
            f"Models used:       {', '.join(self.models_used)}",
        ]
        return "\n".join(lines)


# =============================================================================
# FEATURE ENGINEERING (lightweight, for ML models)
# =============================================================================

def _fetch_analyst_consensus(ticker: str) -> Optional[dict]:
    """
    Fetch analyst price-target consensus via OBBClient.
    Returns a flat dict with target_price / target_high / target_low /
    n_analysts / recommendation, or None if unavailable.

    This is genuinely useful data that no other model in the ensemble has access to —
    it encodes forward-looking analyst estimates, not just historical price patterns.
    """
    try:
        from utils.obb_client import get_obb_client
        data = get_obb_client().get_analyst_consensus(ticker)
        if not data:
            return None

        target_price = float(
            data.get("target_consensus") or data.get("target_median") or 0
        )
        if target_price <= 0:
            return None

        return {
            "target_price":  target_price,
            "target_high":   float(data.get("target_high") or target_price),
            "target_low":    float(data.get("target_low") or target_price),
            "n_analysts":    int(data.get("number_of_analysts") or 0),
            "recommendation": str(data.get("recommendation") or ""),
        }
    except Exception:
        return None


def _build_features(close: pd.Series, ohlcv: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build a feature matrix suitable for supervised ML forecasting.
    All features derived from price/return history (pandas only — no broken obb.technical calls).
    """
    df = pd.DataFrame(index=close.index)
    returns = close.pct_change()
    log_ret = np.log(close / close.shift(1))

    # Lag returns
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_lag_{lag}"] = returns.shift(lag)

    # Rolling volatility
    for w in [5, 10, 20, 60]:
        df[f"vol_{w}"] = returns.rolling(w).std()

    # Rolling mean return
    for w in [5, 10, 20]:
        df[f"mean_ret_{w}"] = returns.rolling(w).mean()

    # Distance from moving averages
    for w in [20, 50]:
        ma = close.rolling(w).mean()
        df[f"dist_ma_{w}"] = (close - ma) / ma

    # Momentum
    for w in [20, 60]:
        df[f"momentum_{w}"] = close / close.shift(w) - 1

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Z-score (20-day)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["zscore_20"] = (close - ma20) / std20.replace(0, np.nan)

    # Log return (target proxy)
    df["log_ret"] = log_ret

    return df.dropna()


def _train_test_split_ts(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2):
    """Time-series safe train/test split (no shuffle)."""
    n = len(X)
    split = int(n * (1 - test_frac))
    return X[:split], X[split:], y[:split], y[split:]


# =============================================================================
# INDIVIDUAL MODEL RUNNERS (free functions for joblib pickling)
# =============================================================================

def _run_arima(close: pd.Series, horizon: int) -> Optional[ModelForecast]:
    if not _STATSMODELS:
        return None
    try:
        returns = close.pct_change().dropna()
        # Fit on returns (stationary), forecast returns, then cumulate to prices
        train_size = int(len(returns) * 0.8)
        train, test = returns.iloc[:train_size], returns.iloc[train_size:]

        model = ARIMA(train, order=(2, 0, 2))
        fit = model.fit()
        val_pred = fit.predict(start=0, end=len(train) - 1)
        rmse = float(np.sqrt(mean_squared_error(train.values, val_pred.values))) if _SKLEARN else float(np.std(train - val_pred))

        # Re-fit on full series
        fit_full = ARIMA(returns, order=(2, 0, 2)).fit()
        fc = fit_full.get_forecast(steps=horizon)
        fc_mean = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.1)  # 90% CI

        future_dates = _future_dates(close, horizon)
        last_price = float(close.iloc[-1])

        # Cumulative return → price
        cum_ret = np.cumprod(1 + fc_mean) - 1
        prices = last_price * (1 + cum_ret)

        lower = last_price * (1 + np.cumprod(1 + ci.iloc[:, 0].values) - 1)
        upper = last_price * (1 + np.cumprod(1 + ci.iloc[:, 1].values) - 1)

        return ModelForecast(
            name="ARIMA",
            predicted_prices=pd.Series(prices, index=future_dates),
            lower=pd.Series(lower, index=future_dates),
            upper=pd.Series(upper, index=future_dates),
            validation_rmse=rmse,
        )
    except Exception as exc:
        logger.debug(f"ARIMA failed: {exc}")
        return None


def _run_ets(close: pd.Series, horizon: int) -> Optional[ModelForecast]:
    if not _STATSMODELS:
        return None
    try:
        train_size = int(len(close) * 0.8)
        train, test = close.iloc[:train_size], close.iloc[train_size:]

        model = ExponentialSmoothing(train, trend="add", damped_trend=True, seasonal=None)
        fit = model.fit(optimized=True)
        val_pred = fit.fittedvalues
        rmse = float(np.sqrt(np.mean((test.values - fit.forecast(len(test)).values) ** 2)))

        fit_full = ExponentialSmoothing(close, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
        fc_mean = fit_full.forecast(horizon).values

        future_dates = _future_dates(close, horizon)
        std_err = float(close.pct_change().std() * close.iloc[-1])
        z = 1.645  # 90% CI
        prices = fc_mean
        lower = prices - z * std_err * np.sqrt(np.arange(1, horizon + 1))
        upper = prices + z * std_err * np.sqrt(np.arange(1, horizon + 1))

        return ModelForecast(
            name="ETS (Holt-Winters)",
            predicted_prices=pd.Series(prices, index=future_dates),
            lower=pd.Series(lower, index=future_dates),
            upper=pd.Series(upper, index=future_dates),
            validation_rmse=rmse,
        )
    except Exception as exc:
        logger.debug(f"ETS failed: {exc}")
        return None


def _run_garch(close: pd.Series, horizon: int) -> Optional[ModelForecast]:
    if not _ARCH:
        return None
    try:
        returns = close.pct_change().dropna() * 100  # percent returns for ARCH

        model = arch_model(returns, vol="Garch", p=1, q=1, mean="AR", lags=1, dist="Normal")
        fit = model.fit(disp="off", show_warning=False)

        fc = fit.forecast(horizon=horizon, reindex=False)
        mean_fc = fc.mean.iloc[-1].values / 100     # back to decimal
        var_fc = fc.variance.iloc[-1].values / 10000  # back to decimal

        last_price = float(close.iloc[-1])
        future_dates = _future_dates(close, horizon)

        # Price path from mean return
        cum_ret = np.cumprod(1 + mean_fc) - 1
        prices = last_price * (1 + cum_ret)

        # CI from conditional variance
        vol_fc = np.sqrt(var_fc)
        z = 1.645
        cum_vol = np.sqrt(np.cumsum(var_fc))
        lower = last_price * np.exp(-z * cum_vol)
        upper = last_price * np.exp(z * cum_vol)

        # Validation RMSE (in-sample)
        resid = fit.resid.dropna()
        rmse = float(resid.std())

        return ModelForecast(
            name="GARCH(1,1)",
            predicted_prices=pd.Series(prices, index=future_dates),
            lower=pd.Series(lower, index=future_dates),
            upper=pd.Series(upper, index=future_dates),
            validation_rmse=rmse,
        )
    except Exception as exc:
        logger.debug(f"GARCH failed: {exc}")
        return None


def _run_ml_model(
    close: pd.Series,
    horizon: int,
    model_name: str,
) -> Optional[ModelForecast]:
    """Generic runner for sklearn / XGBoost / LightGBM models."""
    if not _SKLEARN:
        return None

    try:
        feat_df = _build_features(close)
        target = close.pct_change().shift(-1).reindex(feat_df.index).dropna()
        feat_df = feat_df.reindex(target.index)

        X = feat_df.values
        y = target.values

        X_tr, X_te, y_tr, y_te = _train_test_split_ts(X, y)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if model_name == "RandomForest":
            # n_jobs=1: already parallelised at the outer Parallel level
            mdl = RandomForestRegressor(n_estimators=150, n_jobs=1, random_state=42)
        elif model_name == "XGBoost":
            if not _XGB:
                return None
            mdl = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, n_jobs=1,
                verbosity=0, random_state=42,
            )
        elif model_name == "LightGBM":
            if not _LGB:
                return None
            mdl = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                n_jobs=1, verbose=-1, random_state=42,
            )
        elif model_name == "Ridge":
            mdl = Ridge(alpha=1.0)
        else:
            return None

        mdl.fit(X_tr_s, y_tr)
        val_pred = mdl.predict(X_te_s)
        rmse = float(np.sqrt(mean_squared_error(y_te, val_pred)))

        # Recursive multi-step forecast
        # Start from the last known feature row, predict 1 step, roll features
        last_close = close.copy()
        predicted_returns = []

        for _ in range(horizon):
            f = _build_features(last_close)
            if f.empty:
                predicted_returns.append(0.0)
                continue
            x = scaler.transform(f.values[[-1]])
            ret = float(mdl.predict(x)[0])
            predicted_returns.append(ret)
            # Append synthetic next price
            next_price = float(last_close.iloc[-1]) * (1 + ret)
            next_idx = last_close.index[-1] + pd.Timedelta(days=1)
            last_close = pd.concat([last_close, pd.Series([next_price], index=[next_idx])])

        last_price = float(close.iloc[-1])
        cum_ret = np.cumprod(1 + np.array(predicted_returns)) - 1
        prices = last_price * (1 + cum_ret)
        future_dates = _future_dates(close, horizon)

        # Simple CI based on model's val RMSE spread
        z = 1.645
        spread = last_price * rmse * np.sqrt(np.arange(1, horizon + 1))
        lower = prices - z * spread
        upper = prices + z * spread

        return ModelForecast(
            name=model_name,
            predicted_prices=pd.Series(prices, index=future_dates),
            lower=pd.Series(lower, index=future_dates),
            upper=pd.Series(upper, index=future_dates),
            validation_rmse=rmse,
        )
    except Exception as exc:
        logger.debug(f"{model_name} failed: {exc}")
        return None


def _run_prophet(close: pd.Series, horizon: int) -> Optional[ModelForecast]:
    if not _PROPHET:
        return None
    try:
        df_p = pd.DataFrame({"ds": close.index, "y": close.values})
        df_p["ds"] = pd.to_datetime(df_p["ds"])

        train_size = int(len(df_p) * 0.8)
        train_df = df_p.iloc[:train_size]
        test_df = df_p.iloc[train_size:]

        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.90,
            uncertainty_samples=200,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(train_df)

        val_future = m.make_future_dataframe(periods=len(test_df), freq="B")
        val_fc = m.predict(val_future)
        val_pred = val_fc["yhat"].iloc[-len(test_df):].values
        rmse = float(np.sqrt(np.mean((test_df["y"].values - val_pred) ** 2)))

        # Re-fit on full data
        m2 = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.90,
            uncertainty_samples=200,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2.fit(df_p)

        future = m2.make_future_dataframe(periods=horizon, freq="B")
        fc = m2.predict(future)
        fc_tail = fc.tail(horizon)
        future_dates = pd.DatetimeIndex(fc_tail["ds"].values)

        return ModelForecast(
            name="Prophet",
            predicted_prices=pd.Series(fc_tail["yhat"].values, index=future_dates),
            lower=pd.Series(fc_tail["yhat_lower"].values, index=future_dates),
            upper=pd.Series(fc_tail["yhat_upper"].values, index=future_dates),
            validation_rmse=rmse,
        )
    except Exception as exc:
        logger.debug(f"Prophet failed: {exc}")
        return None


def _run_monte_carlo(
    close: pd.Series,
    horizon: int,
    n_simulations: int,
    garch_vol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Vectorised Geometric Brownian Motion Monte Carlo simulation.

    Returns:
        paths:          (n_sims, horizon) array of price paths
        final_prices:   (n_sims,) final prices
        percentiles:    DataFrame with p5/p25/p50/p75/p95 per day
    """
    returns = close.pct_change().dropna()
    mu = float(returns.mean())
    sigma = garch_vol if garch_vol else float(returns.std())
    last_price = float(close.iloc[-1])

    dt = 1.0
    Z = np.random.standard_normal((n_simulations, horizon))
    daily_returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    paths = last_price * np.cumprod(daily_returns, axis=1)   # (n_sims, horizon)
    final_prices = paths[:, -1]

    future_dates = _future_dates(close, horizon)
    pct_df = pd.DataFrame(
        np.percentile(paths, [5, 25, 50, 75, 95], axis=0).T,
        index=future_dates,
        columns=["p5", "p25", "p50", "p75", "p95"],
    )

    return paths, final_prices, pct_df


# =============================================================================
# ENSEMBLE WEIGHTING
# =============================================================================

def _compute_ensemble(
    model_forecasts: Dict[str, ModelForecast],
    mc_percentiles: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Weighted-average ensemble of all model point forecasts.
    Weight = 1 / RMSE, normalised.  MC provides the CI.
    """
    valid = {k: v for k, v in model_forecasts.items() if len(v.predicted_prices) > 0}
    if not valid:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    # Assign weights
    rmse_vals = np.array([v.validation_rmse if not np.isnan(v.validation_rmse) and v.validation_rmse > 0
                          else 1e-3 for v in valid.values()])
    raw_weights = 1.0 / rmse_vals
    weights = raw_weights / raw_weights.sum()

    for (k, v), w in zip(valid.items(), weights):
        v.weight = float(w)

    # Align on common index
    common_idx = list(valid.values())[0].predicted_prices.index
    matrix = np.stack([v.predicted_prices.reindex(common_idx).ffill().values
                       for v in valid.values()], axis=0)  # (n_models, horizon)
    ensemble_mean = pd.Series(np.average(matrix, weights=weights, axis=0), index=common_idx)

    if mc_percentiles is not None and len(mc_percentiles) > 0:
        lower = mc_percentiles["p5"].reindex(common_idx).ffill()
        upper = mc_percentiles["p95"].reindex(common_idx).ffill()
    else:
        std_spread = ensemble_mean.std()
        lower = ensemble_mean - 1.645 * std_spread
        upper = ensemble_mean + 1.645 * std_spread

    return ensemble_mean, lower, upper


# =============================================================================
# MAIN FORECASTER CLASS
# =============================================================================

class StockForecaster:
    """
    Parallel multi-model ensemble stock price forecaster.

    All models are trained and run concurrently using all available CPU cores.
    A Monte Carlo GBM simulation (50,000 paths by default) provides the
    probability distribution of outcomes.

    Args:
        horizon:        Days ahead to forecast (default 30).
        n_simulations:  Monte Carlo paths (default 50,000).
        n_jobs:         CPU cores to use (-1 = all, default).
        history_days:   Days of historical data to fetch (default 365*3).
    """

    def __init__(
        self,
        horizon: int = 30,
        n_simulations: int = 20_000,   # reduced from 50k — vectorised GBM is fast enough
        n_jobs: int = 4,               # fixed cap; inner models use n_jobs=1
        history_days: int = 365 * 3,
    ):
        self.horizon = horizon
        self.n_simulations = n_simulations
        self.n_jobs = n_jobs
        self.history_days = history_days

    def forecast(self, ticker: str) -> ForecastResult:
        """
        Run the full forecasting pipeline for `ticker`.

        Steps:
          1. Fetch historical close prices.
          2. Run all models in parallel.
          3. Run Monte Carlo GBM.
          4. Compute weighted ensemble.
          5. Return ForecastResult.
        """
        ticker = ticker.upper().strip()
        logger.info(f"Forecasting {ticker} | horizon={self.horizon}d | sims={self.n_simulations:,}")

        # ── 1. Data (via OpenBB) ──────────────────────────────────────────────
        ohlcv = self._fetch_ohlcv(ticker)
        close = ohlcv["Close"].dropna() if "Close" in ohlcv.columns else self._fetch_close(ticker)
        current_price = float(close.iloc[-1])
        returns = close.pct_change().dropna()
        annual_vol = float(returns.std() * np.sqrt(252))
        future_dates = _future_dates(close, self.horizon)

        # ── 2. Parallel model runs ────────────────────────────────────────────
        model_tasks = [
            delayed(_run_arima)(close, self.horizon),
            delayed(_run_ets)(close, self.horizon),
            delayed(_run_garch)(close, self.horizon),
            delayed(_run_ml_model)(close, self.horizon, "RandomForest"),
            delayed(_run_ml_model)(close, self.horizon, "Ridge"),
            delayed(_run_ml_model)(close, self.horizon, "XGBoost"),
            delayed(_run_ml_model)(close, self.horizon, "LightGBM"),
            delayed(_run_prophet)(close, self.horizon),
        ]

        # prefer="threads": sklearn/xgb/lgb release the GIL for C extensions, so
        # threads are safe here and avoid the oversubscription that crashed the system
        # when 8 processes each spawned n_jobs=-1 inner threads.
        # n_jobs capped at 4 to leave headroom for the OS and Streamlit.
        safe_jobs = min(4, len(model_tasks))
        raw_results = Parallel(n_jobs=safe_jobs, prefer="threads", verbose=0)(model_tasks)

        model_forecasts: Dict[str, ModelForecast] = {}
        for r in raw_results:
            if r is not None:
                model_forecasts[r.name] = r

        # ── 2b. Analyst consensus (OpenBB) ────────────────────────────────────
        # This is the one genuinely unique signal OpenBB provides without API keys:
        # forward-looking analyst price targets unavailable from price history alone.
        analyst_data = _fetch_analyst_consensus(ticker)
        if analyst_data is not None:
            target = analyst_data["target_price"]
            # Flat forecast: consensus target held constant over the horizon
            # (analysts don't give day-by-day paths, just a 12-month target)
            analyst_prices = pd.Series(
                np.linspace(current_price, target, self.horizon),
                index=future_dates,
            )
            # Approximate RMSE: use spread between high/low targets as uncertainty proxy
            spread = analyst_data["target_high"] - analyst_data["target_low"]
            approx_rmse = (spread / 4) / current_price if spread > 0 else 0.05
            model_forecasts["Analyst Consensus"] = ModelForecast(
                name="Analyst Consensus",
                predicted_prices=analyst_prices,
                lower=pd.Series(
                    np.linspace(current_price, analyst_data["target_low"], self.horizon),
                    index=future_dates,
                ),
                upper=pd.Series(
                    np.linspace(current_price, analyst_data["target_high"], self.horizon),
                    index=future_dates,
                ),
                validation_rmse=float(approx_rmse),
            )
            logger.info(
                f"Analyst consensus: target=${target:.2f}, "
                f"n={analyst_data['n_analysts']} analysts, "
                f"rec={analyst_data['recommendation']}"
            )

        # ── 2c. Fundamental bias ──────────────────────────────────────────────
        self._apply_fundamental_bias(model_forecasts, ticker)

        # ── 3. Monte Carlo ────────────────────────────────────────────────────
        garch_vol = None
        if "GARCH(1,1)" in model_forecasts:
            # extract terminal vol from GARCH forecast for better MC vol
            try:
                garch_prices = model_forecasts["GARCH(1,1)"].predicted_prices
                if len(garch_prices) > 1:
                    garch_vol = float(
                        (garch_prices / current_price).pct_change().dropna().std()
                        or annual_vol / np.sqrt(252)
                    )
            except Exception:
                pass

        mc_paths, mc_final_prices, mc_percentiles = _run_monte_carlo(
            close, self.horizon, self.n_simulations, garch_vol
        )

        # ── 4. Ensemble ───────────────────────────────────────────────────────
        ensemble_mean, ensemble_lower, ensemble_upper = _compute_ensemble(
            model_forecasts, mc_percentiles
        )

        # ── 5. Summary stats ──────────────────────────────────────────────────
        if len(ensemble_mean) > 0:
            pred_price = float(ensemble_mean.iloc[-1])
        else:
            pred_price = float(mc_percentiles["p50"].iloc[-1])

        predicted_return_pct = (pred_price / current_price - 1) * 100

        p_gain = float(np.mean(mc_final_prices > current_price))
        p_loss_5 = float(np.mean(mc_final_prices < current_price * 0.95))
        p_loss_10 = float(np.mean(mc_final_prices < current_price * 0.90))

        return ForecastResult(
            ticker=ticker,
            current_price=current_price,
            horizon=self.horizon,
            forecast_dates=future_dates,
            model_forecasts=model_forecasts,
            ensemble_mean=ensemble_mean,
            ensemble_lower=ensemble_lower,
            ensemble_upper=ensemble_upper,
            mc_paths=mc_paths,
            mc_final_prices=mc_final_prices,
            mc_percentiles=mc_percentiles,
            predicted_return_pct=predicted_return_pct,
            probability_of_gain=p_gain,
            probability_of_loss_5pct=p_loss_5,
            probability_of_loss_10pct=p_loss_10,
            volatility_annual=annual_vol,
            models_used=list(model_forecasts.keys()),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_close(self, ticker: str) -> pd.Series:
        """Fetch Close price series via OBBClient."""
        return self._fetch_ohlcv(ticker)["Close"].dropna()

    def _fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Fetch full OHLCV DataFrame via OBBClient."""
        from utils.obb_client import get_obb_client
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (
            pd.Timestamp.today() - pd.Timedelta(days=self.history_days)
        ).strftime("%Y-%m-%d")
        return get_obb_client().get_price_history(
            symbol=ticker, start_date=start, end_date=end
        )

    @staticmethod
    def _apply_fundamental_bias(
        model_forecasts: Dict[str, ModelForecast],
        ticker: str,
    ) -> None:
        """
        Adjust per-model RMSE weights using fundamental signals.

        Modifies model_forecasts[*].validation_rmse in-place so that
        _compute_ensemble weights reflect fundamental data.

        Rules:
        - High beta (>1.5)             → GARCH / XGBoost RMSE × 0.85  (upweight)
        - Low beta (<0.7)              → ARIMA / ETS RMSE × 0.85       (upweight)
        - fwd_pe / trailing_pe < 0.85  → Analyst Consensus RMSE × 0.80 (upweight)
        - Gross margin improving YoY   → Analyst Consensus RMSE × 0.90 (upweight)
        """
        try:
            from utils.obb_client import get_obb_client

            df = get_obb_client().get_fundamentals_metrics(ticker, limit=2)
            if df.empty:
                return

            row = df.iloc[0]

            def _f(key: str) -> Optional[float]:
                try:
                    val = row.get(key)
                    return float(val) if val is not None else None
                except (TypeError, ValueError):
                    return None

            beta = _f("beta")
            fwd_pe = _f("forward_pe") or _f("price_to_earnings_ttm")
            trailing_pe = _f("pe_ratio") or _f("price_to_earnings")
            gross_margin = _f("gross_profit_margin")

            # Prior year gross margin
            prior_gm: Optional[float] = None
            if len(df) > 1:
                row2 = df.iloc[1]
                try:
                    v = row2.get("gross_profit_margin")
                    prior_gm = float(v) if v is not None else None
                except (TypeError, ValueError):
                    pass

            # Apply bias adjustments
            if beta is not None:
                if beta > 1.5:
                    for name in ("GARCH(1,1)", "XGBoost"):
                        if name in model_forecasts:
                            model_forecasts[name].validation_rmse *= 0.85
                elif beta < 0.7:
                    for name in ("ARIMA", "ETS (Holt-Winters)"):
                        if name in model_forecasts:
                            model_forecasts[name].validation_rmse *= 0.85

            if (
                fwd_pe is not None
                and trailing_pe is not None
                and trailing_pe > 0
                and fwd_pe / trailing_pe < 0.85
                and "Analyst Consensus" in model_forecasts
            ):
                model_forecasts["Analyst Consensus"].validation_rmse *= 0.80

            if (
                gross_margin is not None
                and prior_gm is not None
                and gross_margin > prior_gm
                and "Analyst Consensus" in model_forecasts
            ):
                model_forecasts["Analyst Consensus"].validation_rmse *= 0.90

        except Exception as exc:
            logger.debug(f"_apply_fundamental_bias({ticker}): {exc}")


# =============================================================================
# UTILITIES
# =============================================================================

def _future_dates(close: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Generate business-day future dates starting the day after the last known date."""
    last = pd.Timestamp(close.index[-1])
    return pd.bdate_range(start=last + pd.Timedelta(days=1), periods=horizon)


def resolve_ticker(query: str) -> Tuple[str, str]:
    """
    Resolve a company name or symbol to (ticker, display_name).

    Tries a hardcoded map first for speed, then falls back to the
    Yahoo Finance autocomplete API via symbol_resolver.search_ticker().
    """
    _MAP = {
        "apple": ("AAPL", "Apple Inc."),
        "microsoft": ("MSFT", "Microsoft Corp."),
        "google": ("GOOGL", "Alphabet Inc."),
        "alphabet": ("GOOGL", "Alphabet Inc."),
        "amazon": ("AMZN", "Amazon.com Inc."),
        "tesla": ("TSLA", "Tesla Inc."),
        "nvidia": ("NVDA", "NVIDIA Corp."),
        "meta": ("META", "Meta Platforms Inc."),
        "facebook": ("META", "Meta Platforms Inc."),
        "netflix": ("NFLX", "Netflix Inc."),
        "disney": ("DIS", "The Walt Disney Co."),
        "jpmorgan": ("JPM", "JPMorgan Chase & Co."),
        "jp morgan": ("JPM", "JPMorgan Chase & Co."),
        "goldman sachs": ("GS", "Goldman Sachs Group"),
        "berkshire": ("BRK-B", "Berkshire Hathaway"),
        "walmart": ("WMT", "Walmart Inc."),
        "exxon": ("XOM", "Exxon Mobil Corp."),
        "chevron": ("CVX", "Chevron Corp."),
        "johnson": ("JNJ", "Johnson & Johnson"),
        "bank of america": ("BAC", "Bank of America"),
        "wells fargo": ("WFC", "Wells Fargo & Co."),
        "coinbase": ("COIN", "Coinbase Global"),
        "palantir": ("PLTR", "Palantir Technologies"),
        "amd": ("AMD", "Advanced Micro Devices"),
        "intel": ("INTC", "Intel Corp."),
        "salesforce": ("CRM", "Salesforce Inc."),
        "adobe": ("ADBE", "Adobe Inc."),
        "paypal": ("PYPL", "PayPal Holdings"),
        "uber": ("UBER", "Uber Technologies"),
        "airbnb": ("ABNB", "Airbnb Inc."),
        "spotify": ("SPOT", "Spotify Technology"),
        "bitcoin": ("BTC-USD", "Bitcoin"),
        "ethereum": ("ETH-USD", "Ethereum"),
        "gold": ("GC=F", "Gold Futures"),
        "oil": ("CL=F", "Crude Oil Futures"),
        "spy": ("SPY", "SPDR S&P 500 ETF"),
        "qqq": ("QQQ", "Invesco QQQ Trust"),
        "iwm": ("IWM", "iShares Russell 2000 ETF"),
        "tlt": ("TLT", "iShares 20+ Year Treasury ETF"),
    }

    key = query.lower().strip()
    if key in _MAP:
        return _MAP[key]

    # Looks like a raw ticker already
    if len(query) <= 6 and query.replace("-", "").replace(".", "").isalnum():
        return query.upper(), query.upper()

    # Fuzzy search via Yahoo Finance
    try:
        from pipeline.symbol_resolver import search_ticker
        results = search_ticker(query)
        if results:
            r = results[0]
            return r["symbol"], r.get("shortname", r["symbol"])
    except Exception:
        pass

    return query.upper(), query.upper()
