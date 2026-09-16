"""
AFML Quant Pipeline - Advanced Feature Engineering
Implements state-of-the-art financial feature engineering from:
- Advances in Financial Machine Learning (López de Prado)
- Machine Learning for Asset Managers (López de Prado)
- Machine Learning for Algorithmic Trading (Jansen)
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy as scipy_entropy
from numba import jit

from pipeline.technical_indicators import TechnicalIndicators


# =============================================================================
# FUNDAMENTAL FEATURES
# =============================================================================

def get_fundamental_features(
    symbol: str,
    client=None,
) -> pd.DataFrame:
    """
    Fetch a single-row DataFrame of fundamental metrics prefixed with ``fund_``.

    Args:
        symbol: Equity ticker.
        client: OBBClient instance (uses singleton if None).

    Returns:
        Single-row DataFrame with columns:
            fund_pe_ratio, fund_forward_pe, fund_pb_ratio, fund_roe,
            fund_gross_margin, fund_operating_margin, fund_net_margin,
            fund_debt_to_equity, fund_current_ratio, fund_beta,
            fund_revenue_growth, fund_earnings_growth.
        Returns an empty DataFrame on any failure (fundamental features
        are optional enrichment).
    """
    try:
        if client is None:
            from utils.obb_client import get_obb_client
            client = get_obb_client()

        df = client.get_fundamentals_metrics(symbol, limit=1)
        if df.empty:
            return pd.DataFrame()

        row = df.iloc[0]

        def _f(key: str) -> Optional[float]:
            try:
                val = row.get(key)
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        data = {
            "fund_pe_ratio":         _f("pe_ratio"),
            "fund_forward_pe":       _f("forward_pe"),
            "fund_pb_ratio":         _f("price_to_book"),
            "fund_roe":              _f("return_on_equity"),
            "fund_gross_margin":     _f("gross_profit_margin"),
            "fund_operating_margin": _f("operating_profit_margin"),
            "fund_net_margin":       _f("net_profit_margin"),
            "fund_debt_to_equity":   _f("debt_equity_ratio"),
            "fund_current_ratio":    _f("current_ratio"),
            "fund_beta":             _f("beta"),
            "fund_revenue_growth":   _f("revenue_growth"),
            "fund_earnings_growth":  _f("earnings_growth"),
        }
        return pd.DataFrame([data])

    except Exception:
        return pd.DataFrame()


# =============================================================================
# FRACTIONAL DIFFERENTIATION - Chapter 5 AFML
# =============================================================================

def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    """
    Calculate weights for Fixed-Width Window Fractional Differentiation.
    
    The key insight: integer differentiation (d=1) makes series stationary
    but destroys memory. Fractional differentiation (0 < d < 1) achieves
    stationarity while preserving memory.
    
    Mathematical formula: w_k = -w_{k-1} * (d - k + 1) / k
    
    Reference: AFML Chapter 5
    """
    weights = [1.0]
    k = 1
    while abs(weights[-1]) >= threshold and k < max_size:
        w_k = -weights[-1] * (d - k + 1) / k
        weights.append(w_k)
        k += 1
    
    weights = np.array(weights[:-1] if len(weights) > 1 else weights)
    return weights


def frac_diff_ffd(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differentiation.
    
    This is the PREFERRED method over expanding window because:
    1. Fixed lookback window (more practical)
    2. Weights don't change over time
    3. Can be applied to real-time data
    
    Args:
        series: Log prices recommended (or prices)
        d: Differentiation order (0 < d < 1)
        threshold: Minimum weight cutoff
        
    Returns:
        Fractionally differentiated series
        
    Reference: AFML Chapter 5, Snippet 5.3
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    
    # Use log prices for better behavior
    if series.min() > 0:
        log_series = np.log(series)
    else:
        log_series = series
    
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(width - 1, len(series)):
        window = log_series.iloc[i - width + 1:i + 1].values
        result.iloc[i] = np.dot(weights[::-1], window)
    
    return result


def find_optimal_d(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    significance: float = 0.05,
    min_correlation: float = 0.5
) -> Tuple[float, pd.DataFrame]:
    """
    Find minimum d that achieves stationarity while maximizing correlation
    with original series (preserving memory).
    
    The goal: stationary series with maximum memory preservation.
    
    Reference: AFML Chapter 5
    """
    from statsmodels.tsa.stattools import adfuller
    
    results = []
    d_values = np.arange(d_range[0], d_range[1] + step, step)
    
    log_series = np.log(series) if series.min() > 0 else series
    
    for d in d_values:
        if d == 0:
            frac_series = log_series.copy()
        else:
            frac_series = frac_diff_ffd(series, d=d)
        
        valid_series = frac_series.dropna()
        if len(valid_series) < 50:
            continue
            
        try:
            adf_stat, p_value, *_ = adfuller(valid_series, maxlag=12, autolag='AIC')
        except Exception:
            continue
        
        # Correlation with original (memory preservation)
        common_idx = log_series.index.intersection(valid_series.index)
        if len(common_idx) > 10:
            corr = log_series.loc[common_idx].corr(valid_series.loc[common_idx])
        else:
            corr = np.nan
        
        results.append({
            'd': d,
            'adf_stat': adf_stat,
            'p_value': p_value,
            'is_stationary': p_value < significance,
            'correlation': corr,
            'n_samples': len(valid_series)
        })
    
    df = pd.DataFrame(results)
    
    # Find minimum d that is stationary and has acceptable correlation
    stationary = df[(df['is_stationary']) & (df['correlation'] >= min_correlation)]
    
    if len(stationary) > 0:
        optimal_d = stationary['d'].min()
    else:
        # Fall back to first stationary
        stationary_any = df[df['is_stationary']]
        optimal_d = stationary_any['d'].min() if len(stationary_any) > 0 else 0.5
    
    return optimal_d, df


# =============================================================================
# MICROSTRUCTURE FEATURES - Chapter 18-19 AFML
# =============================================================================

def get_roll_spread(close: pd.Series) -> pd.Series:
    """
    Roll Model: Estimate effective spread from price changes.
    
    Based on the negative serial covariance of price changes.
    Spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))
    
    Reference: AFML Chapter 19
    """
    diff = close.diff()
    cov = diff.rolling(20).apply(lambda x: np.cov(x[:-1], x[1:])[0, 1], raw=False)
    spread = 2 * np.sqrt(np.abs(cov.clip(upper=0) * -1))
    return spread


def get_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Kyle's Lambda: Price impact coefficient.
    
    Measures how much prices move per unit of signed volume.
    Higher lambda = lower liquidity.
    
    Reference: AFML Chapter 19
    """
    returns = close.pct_change()
    signed_volume = volume * np.sign(returns)
    
    def compute_lambda(ret, vol):
        if len(ret) < 5 or vol.std() == 0:
            return np.nan
        try:
            slope, _, _, _, _ = stats.linregress(vol, ret)
            return abs(slope)
        except Exception:
            return np.nan
    
    lambda_series = pd.Series(index=close.index, dtype=float)
    for i in range(window, len(close)):
        ret_window = returns.iloc[i-window:i].values
        vol_window = signed_volume.iloc[i-window:i].values
        mask = ~(np.isnan(ret_window) | np.isnan(vol_window))
        if mask.sum() >= 5:
            lambda_series.iloc[i] = compute_lambda(ret_window[mask], vol_window[mask])
    
    return lambda_series


def get_amihud_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Amihud's Lambda: Price impact measure (illiquidity).
    
    Illiquidity = |return| / dollar_volume
    
    Reference: AFML Chapter 19
    """
    returns = close.pct_change().abs()
    dollar_volume = close * volume
    
    # Avoid division by zero
    dollar_volume = dollar_volume.replace(0, np.nan)
    
    illiq = returns / dollar_volume
    return illiq.rolling(window).mean()


def get_vpin(
    close: pd.Series,
    volume: pd.Series,
    bucket_size: float = 50000,
    n_buckets: int = 50
) -> pd.Series:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).
    
    Measures order flow toxicity - probability of informed trading.
    Higher VPIN = more toxic flow = potential adverse selection.
    
    Reference: AFML Chapter 18-19
    """
    # Classify volume using tick rule
    price_diff = close.diff()
    buy_volume = volume.where(price_diff > 0, 0)
    sell_volume = volume.where(price_diff < 0, 0)
    
    # Create volume buckets
    cum_volume = volume.cumsum()
    bucket_idx = (cum_volume / bucket_size).astype(int)
    
    # Aggregate by bucket
    buy_agg = buy_volume.groupby(bucket_idx).sum()
    sell_agg = sell_volume.groupby(bucket_idx).sum()
    total_agg = volume.groupby(bucket_idx).sum()
    
    # VPIN = sum(|Buy - Sell|) / sum(Total) over n_buckets
    imbalance = (buy_agg - sell_agg).abs()
    
    vpin = imbalance.rolling(n_buckets).sum() / total_agg.rolling(n_buckets).sum()
    
    # Map back to original index
    result = pd.Series(index=close.index, dtype=float)
    for i, idx in enumerate(close.index):
        bucket = bucket_idx.get(idx, np.nan)
        if pd.notna(bucket) and bucket in vpin.index:
            result.iloc[i] = vpin.loc[bucket]
    
    return result


# =============================================================================
# ENTROPY FEATURES - Chapter 18 AFML
# =============================================================================

@jit(nopython=True)
def _lempel_ziv_complexity(binary_str: np.ndarray) -> float:
    """JIT-compiled Lempel-Ziv complexity calculation."""
    n = len(binary_str)
    if n == 0:
        return 0.0
    
    complexity = 1
    i = 0
    k = 1
    k_max = 1
    
    while i + k <= n:
        # Check if substring exists in seen portion
        substr = binary_str[i:i+k]
        found = False
        
        for j in range(i):
            if j + k <= i:
                match = True
                for m in range(k):
                    if binary_str[j + m] != substr[m]:
                        match = False
                        break
                if match:
                    found = True
                    break
        
        if found:
            k += 1
            if k > k_max:
                k_max = k
        else:
            complexity += 1
            i = i + k
            k = 1
    
    # Normalize by theoretical maximum
    if n > 0:
        return complexity / (n / np.log2(n + 1))
    return 0.0


def get_lempel_ziv_entropy(series: pd.Series, n_bins: int = 10, window: int = 50) -> pd.Series:
    """
    Lempel-Ziv Complexity as entropy measure.
    
    Measures the randomness/complexity of a time series.
    Higher complexity = more random = harder to predict.
    
    Reference: AFML Chapter 18
    """
    # Discretize returns
    returns = series.pct_change()
    
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(window, len(series)):
        window_returns = returns.iloc[i-window:i].dropna()
        if len(window_returns) < window // 2:
            continue
        
        # Binning
        try:
            bins = pd.qcut(window_returns, n_bins, labels=False, duplicates='drop')
            binary = (bins > bins.median()).astype(int).values
            result.iloc[i] = _lempel_ziv_complexity(binary)
        except Exception:
            continue
    
    return result


def get_shannon_entropy(series: pd.Series, n_bins: int = 10, window: int = 50) -> pd.Series:
    """
    Shannon Entropy of return distribution.
    
    Measures information content / uncertainty in returns.
    
    Reference: AFML Chapter 18
    """
    returns = series.pct_change()
    
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(window, len(series)):
        window_returns = returns.iloc[i-window:i].dropna()
        if len(window_returns) < window // 2:
            continue
        
        try:
            hist, _ = np.histogram(window_returns, bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zeros
            result.iloc[i] = scipy_entropy(hist, base=2)
        except Exception:
            continue
    
    return result


# =============================================================================
# STRUCTURAL BREAK FEATURES - Chapter 17 AFML
# =============================================================================

def get_cusum_filter(close: pd.Series, threshold: float = None) -> pd.Series:
    """
    CUSUM Filter for detecting structural breaks.
    
    Returns a series where 1 indicates a structural break event.
    These can be used to sample only meaningful observations.
    
    Reference: AFML Chapter 17, Snippet 17.1
    """
    log_prices = np.log(close)
    diff = log_prices.diff().dropna()
    
    if threshold is None:
        threshold = diff.std()
    
    events = pd.Series(0, index=close.index)
    s_pos = 0
    s_neg = 0
    
    for i, (idx, ret) in enumerate(diff.items()):
        s_pos = max(0, s_pos + ret)
        s_neg = min(0, s_neg + ret)
        
        if s_pos > threshold:
            events.loc[idx] = 1
            s_pos = 0
        elif s_neg < -threshold:
            events.loc[idx] = -1
            s_neg = 0
    
    return events


def get_sadf(
    log_prices: pd.Series,
    min_window: int = 50,
    lags: int = 1
) -> pd.Series:
    """
    Supremum Augmented Dickey-Fuller test statistic.
    
    Detects explosive behavior (bubbles) in prices.
    High SADF values indicate bubble-like conditions.
    
    Reference: AFML Chapter 17
    """
    from statsmodels.tsa.stattools import adfuller
    
    n = len(log_prices)
    result = pd.Series(index=log_prices.index, dtype=float)
    
    for end in range(min_window * 2, n):
        adf_stats = []
        
        for start in range(0, end - min_window):
            window = log_prices.iloc[start:end]
            try:
                adf_stat, _, _, _, _, _ = adfuller(window, maxlag=lags, regression='c')
                adf_stats.append(adf_stat)
            except Exception:
                continue
        
        if adf_stats:
            result.iloc[end] = max(adf_stats)
    
    return result


# =============================================================================
# ADVANCED TECHNICAL FEATURES
# =============================================================================

def get_trend_scanning_labels(
    close: pd.Series,
    horizons: List[int] = [5, 10, 20, 40, 60],
    min_r2: float = 0.5
) -> pd.DataFrame:
    """
    Trend-Scanning Labels: Find the most significant trend.
    
    For each observation, finds the horizon that yields the highest
    t-statistic for a linear trend, then labels based on sign.
    
    Reference: Machine Learning for Asset Managers, Chapter 5
    """
    from scipy.stats import linregress
    
    result = pd.DataFrame(index=close.index)
    result['t1'] = pd.NaT
    result['tval'] = np.nan
    result['label'] = np.nan
    result['horizon'] = np.nan
    
    for i in range(max(horizons), len(close)):
        best_tval = 0
        best_horizon = horizons[0]
        
        for h in horizons:
            if i + h >= len(close):
                continue
            
            y = close.iloc[i:i+h].values
            x = np.arange(h)
            
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                
                if std_err > 0:
                    tval = slope / std_err
                else:
                    tval = 0
                
                # Only consider if R² is acceptable
                if r_value ** 2 >= min_r2 and abs(tval) > abs(best_tval):
                    best_tval = tval
                    best_horizon = h
            except Exception:
                continue
        
        result.iloc[i, result.columns.get_loc('tval')] = best_tval
        result.iloc[i, result.columns.get_loc('label')] = np.sign(best_tval)
        result.iloc[i, result.columns.get_loc('horizon')] = best_horizon
        result.iloc[i, result.columns.get_loc('t1')] = close.index[min(i + best_horizon, len(close) - 1)]
    
    return result


def get_orthogonal_features(features: pd.DataFrame, variance_threshold: float = 0.95) -> pd.DataFrame:
    """
    Create orthogonal features using PCA to reduce multicollinearity.
    
    Critical for financial data which is highly correlated.
    
    Reference: AFML Chapter 8, Machine Learning for Asset Managers Chapter 6
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.fillna(0))
    
    # PCA
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    orthogonal = pca.fit_transform(scaled)
    
    # Create DataFrame
    n_components = orthogonal.shape[1]
    columns = [f'PC_{i+1}' for i in range(n_components)]
    
    result = pd.DataFrame(orthogonal, index=features.index, columns=columns)
    result['explained_variance'] = pca.explained_variance_ratio_.sum()
    
    return result


# =============================================================================
# COMPREHENSIVE FEATURE PIPELINE
# =============================================================================

class AdvancedFeatureEngine:
    """
    Complete feature engineering pipeline implementing López de Prado methods.
    """
    
    def __init__(
        self,
        frac_d: float = None,  # Auto-detect if None
        use_orthogonal: bool = True,
        include_entropy: bool = True,
        include_microstructure: bool = True
    ):
        self.frac_d = frac_d
        self.use_orthogonal = use_orthogonal
        self.include_entropy = include_entropy
        self.include_microstructure = include_microstructure
        self.optimal_d = None
        self.feature_names_ = None
    
    def fit_transform(
        self,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature set.

        Args:
            close:   Close price series.
            volume:  Volume series (optional).
            high:    High price series (optional).
            low:     Low price series (optional).
            symbol:  Equity ticker — if provided, fundamental features are
                     fetched and broadcast as scalar columns.
        """
        features = pd.DataFrame(index=close.index)

        # 0. pandas-ta standard indicators (prepended before custom features)
        _ohlcv = pd.DataFrame({"Close": close})
        if high is not None:
            _ohlcv["High"] = high
        if low is not None:
            _ohlcv["Low"] = low
        if volume is not None:
            _ohlcv["Volume"] = volume
        _ohlcv = TechnicalIndicators().add_all(_ohlcv)
        # Bring in only the newly added indicator columns (not OHLCV itself)
        _ta_cols = [c for c in _ohlcv.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
        for col in _ta_cols:
            features[f"ta_{col}"] = _ohlcv[col]

        # 0b. Fundamental features (optional scalar broadcast)
        if symbol is not None:
            fund_df = get_fundamental_features(symbol)
            if not fund_df.empty:
                for col in fund_df.columns:
                    features[col] = fund_df[col].iloc[0]  # scalar broadcast

        # 1. Fractional Differentiation
        if self.frac_d is None:
            self.optimal_d, _ = find_optimal_d(close)
            print(f"Optimal d found: {self.optimal_d:.3f}")
        else:
            self.optimal_d = self.frac_d
        
        features['frac_diff'] = frac_diff_ffd(close, d=self.optimal_d)
        
        # 2. Return-based features
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        features['returns_5'] = close.pct_change(5)
        features['returns_20'] = close.pct_change(20)
        
        # 3. Volatility features
        features['volatility'] = features['returns'].ewm(span=20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        features['realized_vol'] = features['returns'].rolling(20).std() * np.sqrt(252)
        
        # Parkinson volatility (if high/low available)
        if high is not None and low is not None:
            features['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * np.log(high / low) ** 2
            ).rolling(20).mean()
        
        # 4. Momentum features
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = close / close.shift(period) - 1
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # 5. Mean reversion features
        for period in [20, 50]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'zscore_{period}'] = (close - ma) / std
            features[f'distance_ma_{period}'] = (close - ma) / close
        
        # 6. Microstructure features
        if self.include_microstructure and volume is not None:
            features['roll_spread'] = get_roll_spread(close)
            features['kyle_lambda'] = get_kyle_lambda(close, volume)
            features['amihud_lambda'] = get_amihud_lambda(close, volume)
            features['dollar_volume'] = close * volume
            features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        
        # 7. Entropy features
        if self.include_entropy:
            features['shannon_entropy'] = get_shannon_entropy(close)
            # Lempel-Ziv is expensive, use sparingly
            # features['lz_complexity'] = get_lempel_ziv_entropy(close)
        
        # 8. Structural break features
        features['cusum'] = get_cusum_filter(close)
        
        # 9. Autocorrelation features
        for lag in [1, 5, 10]:
            features[f'autocorr_{lag}'] = features['returns'].rolling(50).apply(
                lambda x: x.autocorr(lag=lag), raw=False
            )
        
        # Store feature names before orthogonalization
        self.feature_names_ = list(features.columns)
        
        # 10. Orthogonalize if requested
        if self.use_orthogonal:
            numeric_features = features.select_dtypes(include=[np.number])
            valid_cols = [c for c in numeric_features.columns if numeric_features[c].notna().sum() > 100]
            if len(valid_cols) > 5:
                orth_features = get_orthogonal_features(numeric_features[valid_cols].dropna())
                # Keep both raw and orthogonal
                features = pd.concat([features, orth_features], axis=1)
        
        return features.dropna(how='all', axis=1)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        return self.feature_names_ or []


if __name__ == "__main__":
    print("Testing Advanced Feature Engineering...")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    # Simulate trending random walk
    returns = np.random.randn(n) * 0.02 + 0.0001
    prices = 100 * np.exp(np.cumsum(returns))
    volume = np.random.lognormal(10, 0.5, n)
    
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    close = pd.Series(prices, index=dates, name='Close')
    vol = pd.Series(volume, index=dates, name='Volume')
    
    # Test feature engine
    engine = AdvancedFeatureEngine(
        use_orthogonal=True,
        include_entropy=True,
        include_microstructure=True
    )
    
    features = engine.fit_transform(close, volume=vol)
    
    print(f"\nGenerated {len(features.columns)} features:")
    print(features.columns.tolist()[:20], "...")
    print(f"\nFeature matrix shape: {features.shape}")
    print("\nSample features (last 5 rows):")
    print(features.iloc[-5:, :5])
