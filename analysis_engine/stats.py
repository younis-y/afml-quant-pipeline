"""
AFML Quant Pipeline - Statistical Tools Library
Common statistical tests and transformations for financial time series.

Addresses Chris Brooks (Introductory Econometrics) critique:
- Fat tail detection (kurtosis, Jarque-Bera, KS-test)
- Value at Risk: Parametric (Gaussian), Historical, Cornish-Fisher, Bootstrapped
- GARCH(1,1) conditional volatility modelling
- Bootstrapped confidence intervals for risk metrics
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Union, Optional, List
import warnings

# ---------------------------------------------------------------------------
# 1. CORE STATISTICAL TESTS
# ---------------------------------------------------------------------------

def get_adf_stat(series: pd.Series) -> Dict[str, Union[float, bool]]:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    series = series.dropna()
    if len(series) < 10:
        return {"is_stationary": False, "p_value": 1.0, "stat": 0.0}

    result = adfuller(series)
    return {
        "stat": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
    }


def get_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate the Hurst Exponent via variance of lagged differences.
    H < 0.5 : Mean-Reverting
    H ≈ 0.5 : Geometric Brownian Motion / Random Walk
    H > 0.5 : Trending / Momentum
    """
    series = series.dropna().values
    if len(series) < max_lag + 10:
        return 0.5

    lags = list(range(2, max_lag))
    variances = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        variances.append(np.var(diff))

    # log(Var) = 2H * log(lag) + c
    poly = np.polyfit(np.log(lags), np.log(variances), 1)
    return poly[0] / 2.0


def get_volatility(series: pd.Series, annualize: bool = True) -> float:
    """
    Calculate (annualized) volatility of simple returns.
    """
    returns = series.pct_change().dropna()
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)
    return vol


# ---------------------------------------------------------------------------
# 2. FAT TAIL / DISTRIBUTIONAL ANALYSIS  (Brooks Ch. 5 & 13)
# ---------------------------------------------------------------------------

def analyze_fat_tails(returns: pd.Series) -> Dict[str, Union[float, bool, str]]:
    """
    Comprehensive distributional analysis of return series.
    
    Tests whether the Gaussian assumption holds — critical for any
    VaR or risk model (Brooks, 'Introductory Econometrics for Finance').

    Returns dict with:
        skewness, kurtosis (excess), jarque_bera_stat, jb_p_value,
        is_normal (JB), ks_stat, ks_p_value, tail_classification
    """
    r = returns.dropna().values

    skew = sp_stats.skew(r)
    excess_kurt = sp_stats.kurtosis(r)  # excess kurtosis (Fisher)

    # Jarque-Bera test  (H0: returns are normally distributed)
    jb_stat, jb_p = sp_stats.jarque_bera(r)

    # Kolmogorov-Smirnov against fitted normal
    ks_stat, ks_p = sp_stats.kstest(r, "norm", args=(np.mean(r), np.std(r)))

    # Classification
    if excess_kurt > 3:
        tail_class = "Severely Leptokurtic"
    elif excess_kurt > 1:
        tail_class = "Leptokurtic"
    elif excess_kurt > 0:
        tail_class = "Mildly Fat-Tailed"
    else:
        tail_class = "Platykurtic / Thin-Tailed"

    is_normal = jb_p > 0.05

    return {
        "skewness": float(skew),
        "excess_kurtosis": float(excess_kurt),
        "jarque_bera_stat": float(jb_stat),
        "jb_p_value": float(jb_p),
        "is_normal_jb": is_normal,
        "ks_stat": float(ks_stat),
        "ks_p_value": float(ks_p),
        "tail_classification": tail_class,
    }


# ---------------------------------------------------------------------------
# 3. VALUE AT RISK  (Brooks Ch. 13)
# ---------------------------------------------------------------------------

def var_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> Dict[str, float]:
    """
    Parametric (Gaussian) VaR & CVaR.
    
    WARNING: Assumes normally distributed returns — unreliable when
    fat tails are present (see Brooks Ch. 13).  Use as a baseline only.
    """
    r = returns.dropna()
    mu = r.mean()
    sigma = r.std()
    z = sp_stats.norm.ppf(1 - confidence)

    var = -(mu + z * sigma) * np.sqrt(holding_period)
    # Conditional VaR (Expected Shortfall)
    cvar = -(mu - sigma * sp_stats.norm.pdf(z) / (1 - confidence)) * np.sqrt(holding_period)

    return {"VaR": float(var), "CVaR": float(cvar), "method": "Parametric-Gaussian"}


def var_cornish_fisher(
    returns: pd.Series,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> Dict[str, float]:
    """
    Cornish-Fisher VaR — adjusts the Gaussian quantile for skewness
    and excess kurtosis.  A quick analytical fix for fat tails without
    full simulation (Brooks, Ch. 13).
    """
    r = returns.dropna().values
    mu = np.mean(r)
    sigma = np.std(r)
    s = sp_stats.skew(r)
    k = sp_stats.kurtosis(r)  # excess
    z = sp_stats.norm.ppf(1 - confidence)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )

    var = -(mu + z_cf * sigma) * np.sqrt(holding_period)
    return {"VaR": float(var), "method": "Cornish-Fisher"}


def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    holding_period: int = 1,
) -> Dict[str, float]:
    """
    Historical simulation VaR & CVaR.
    Non-parametric — no distributional assumption.
    """
    r = returns.dropna()
    cutoff = r.quantile(1 - confidence)

    var = -cutoff * np.sqrt(holding_period)
    cvar = -r[r <= cutoff].mean() * np.sqrt(holding_period)

    return {"VaR": float(var), "CVaR": float(cvar), "method": "Historical"}


def var_bootstrap(
    returns: pd.Series,
    confidence: float = 0.95,
    holding_period: int = 1,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Dict[str, Union[float, List[float]]]:
    """
    Bootstrapped VaR with confidence interval.
    
    Resamples return distribution (with replacement) to build an
    empirical distribution of VaR estimates — directly addresses
    Brooks' recommendation for non-normal return series.
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    n = len(r)

    var_estimates = []
    for _ in range(n_simulations):
        sample = rng.choice(r, size=n, replace=True)
        q = np.percentile(sample, (1 - confidence) * 100)
        var_estimates.append(-q * np.sqrt(holding_period))

    var_estimates = np.array(var_estimates)

    return {
        "VaR": float(np.mean(var_estimates)),
        "VaR_median": float(np.median(var_estimates)),
        "CI_lower": float(np.percentile(var_estimates, 2.5)),
        "CI_upper": float(np.percentile(var_estimates, 97.5)),
        "method": "Bootstrap",
        "n_simulations": n_simulations,
    }


def compute_all_var(
    returns: pd.Series,
    confidence: float = 0.95,
    holding_period: int = 1,
    n_bootstrap: int = 10_000,
) -> Dict[str, Dict]:
    """
    Run every VaR methodology and return a comparison dict.
    """
    return {
        "parametric": var_parametric(returns, confidence, holding_period),
        "cornish_fisher": var_cornish_fisher(returns, confidence, holding_period),
        "historical": var_historical(returns, confidence, holding_period),
        "bootstrap": var_bootstrap(returns, confidence, holding_period, n_bootstrap),
    }


# ---------------------------------------------------------------------------
# 4. GARCH(1,1) CONDITIONAL VOLATILITY  (Brooks Ch. 9)
# ---------------------------------------------------------------------------

def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
) -> Dict[str, Union[float, pd.Series, str]]:
    """
    Fit a GARCH(p,q) model to a return series.

    Uses Student-t innovations by default — appropriate for fat-tailed
    financial data (Brooks, Ch. 9).  Falls back to Normal if t-dist
    estimation fails.

    Returns:
        omega, alpha, beta, persistence, conditional_vol (series),
        distribution, aic, bic, annualized_long_run_vol
    """
    try:
        from arch import arch_model
    except ImportError:
        return {
            "error": "arch package not installed. Run: pip install arch",
            "conditional_vol": pd.Series(dtype=float),
        }

    r = returns.dropna() * 100  # arch expects percentage returns

    # Try Student-t first, fall back to Normal
    for d in ([dist, "normal"] if dist != "normal" else ["normal"]):
        try:
            model = arch_model(r, vol="Garch", p=p, q=q, dist=d, rescale=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(disp="off", show_warning=False)
            break
        except Exception:
            result = None

    if result is None:
        return {
            "error": "GARCH fitting failed for all distributions.",
            "conditional_vol": pd.Series(dtype=float),
        }

    params = result.params
    omega = float(params.get("omega", 0))
    alpha = float(params.get("alpha[1]", 0))
    beta = float(params.get("beta[1]", 0))
    persistence = alpha + beta

    # Annualized long-run variance (if stationary)
    if persistence < 1:
        long_run_var = omega / (1 - persistence)
        ann_long_run_vol = np.sqrt(long_run_var * 252) / 100  # back to decimal
    else:
        ann_long_run_vol = np.nan

    cond_vol = result.conditional_volatility / 100  # back to decimal
    cond_vol.index = r.index

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "is_stationary": persistence < 1,
        "annualized_long_run_vol": float(ann_long_run_vol) if not np.isnan(ann_long_run_vol) else None,
        "conditional_vol": cond_vol,
        "current_cond_vol_annualized": float(cond_vol.iloc[-1] * np.sqrt(252)),
        "distribution": d,
        "aic": float(result.aic),
        "bic": float(result.bic),
    }


def garch_var(
    returns: pd.Series,
    confidence: float = 0.95,
    garch_result: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    1-day VaR using GARCH conditional volatility with Student-t quantile
    if available — the recommended approach for assets with time-varying
    volatility and fat tails (Brooks, Ch. 13).
    """
    if garch_result is None:
        garch_result = fit_garch(returns)

    if "error" in garch_result:
        return {"error": garch_result["error"]}

    cond_vol = garch_result["conditional_vol"]
    if cond_vol.empty:
        return {"error": "No conditional volatility series available."}

    current_vol = cond_vol.iloc[-1]
    mu = returns.dropna().mean()

    # Use t-distribution quantile if GARCH was fit with t-dist
    if garch_result.get("distribution") == "t":
        # Approximate df from excess kurtosis
        ek = sp_stats.kurtosis(returns.dropna().values)
        df = max(4, 6 / ek + 4) if ek > 0 else 30
        z = sp_stats.t.ppf(1 - confidence, df)
    else:
        z = sp_stats.norm.ppf(1 - confidence)

    var = -(mu + z * current_vol)

    return {
        "VaR": float(var),
        "current_cond_vol": float(current_vol),
        "method": f"GARCH(1,1)-{garch_result.get('distribution', 'normal')}",
    }


# ---------------------------------------------------------------------------
# 5. FRACTIONAL DIFFERENTIATION  (De Prado)
# ---------------------------------------------------------------------------

def frac_diff(series: pd.Series, d: float, window: int = 20) -> pd.Series:
    """
    Apply fractional differentiation (simplified fixed window).
    For a robust implementation, usage of FFT or full weights is recommended (De Prado).
    This is a quick approximation using the standard expansion of (1-L)^d
    """
    weights = [1.0]
    for k in range(1, window):
        w_prev = weights[-1]
        w_k = -w_prev * (d - k + 1) / k
        weights.append(w_k)

    weights = np.array(weights)[::-1]  # Reverse for convolution

    res = series.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
    return res.dropna()


# ---------------------------------------------------------------------------
# 6. MASTER ANALYSIS FUNCTION
# ---------------------------------------------------------------------------

def analyze_series_structure(series: pd.Series) -> Dict:
    """
    Run full suite of structural & risk analysis on a price series.

    Returns a comprehensive dict covering:
        - Stationarity (ADF)
        - Memory / trend (Hurst)
        - Unconditional volatility
        - Fat-tail diagnostics
        - VaR comparison (Parametric, Cornish-Fisher, Historical, Bootstrap)
        - GARCH conditional volatility & GARCH-VaR
    """
    log_prices = np.log(series.dropna())
    returns = series.pct_change().dropna()

    # --- Structural ---
    adf_res = get_adf_stat(log_prices)
    hurst = get_hurst_exponent(log_prices)
    vol = get_volatility(series)

    structure = "Random Walk"
    if hurst < 0.45:
        structure = "Mean Reverting"
    if hurst > 0.55:
        structure = "Trending"

    # --- Fat-tail diagnostics ---
    tail_info = analyze_fat_tails(returns)

    # --- VaR suite (95 % 1-day) ---
    var_suite = compute_all_var(returns, confidence=0.95, holding_period=1)

    # --- GARCH ---
    garch_res = fit_garch(returns)
    garch_var_res = garch_var(returns, confidence=0.95, garch_result=garch_res)

    # --- Risk model recommendation ---
    if tail_info["is_normal_jb"]:
        risk_model_rec = "Parametric Gaussian VaR is acceptable."
    elif tail_info["excess_kurtosis"] > 3:
        risk_model_rec = (
            "Severely fat-tailed returns — use Bootstrap or GARCH-t VaR. "
            "Parametric Gaussian VaR will UNDERESTIMATE tail risk."
        )
    else:
        risk_model_rec = (
            "Non-normal returns — prefer Cornish-Fisher, Historical, or GARCH VaR "
            "over naive Parametric Gaussian."
        )

    return {
        # Structure
        "adf_p_value": adf_res["p_value"],
        "hurst": hurst,
        "volatility_annual": vol,
        "structure_verdict": structure,
        # Fat tails
        "fat_tails": tail_info,
        # VaR comparison
        "var_95": var_suite,
        # GARCH
        "garch": {
            k: v
            for k, v in garch_res.items()
            if k != "conditional_vol"  # exclude the full series from dict repr
        },
        "garch_var_95": garch_var_res,
        # Recommendation
        "risk_model_recommendation": risk_model_rec,
    }
