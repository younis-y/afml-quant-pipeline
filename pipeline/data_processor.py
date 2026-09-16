"""
AFML Quant Pipeline - Data Processor
Converts raw market data to Dollar Bars and applies Fractional Differentiation
Reference: Advances in Financial Machine Learning, Chapters 2-5
"""

from typing import Tuple
import numpy as np
import pandas as pd

from utils.data_provider import get_data_provider


def fetch_data(
    ticker: str = "SPY",
    period: str = "1mo",
    interval: str = "1m"
) -> pd.DataFrame:
    """
    Fetch market data via OpenBBDataProvider. There is no fallback:
    a failed fetch raises rather than returning an empty frame.

    Args:
        ticker: Stock symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with OHLCV data

    Note:
        1-minute data is only available for the last 7 days
    """
    provider = get_data_provider()
    data = provider.fetch_ohlcv(ticker, period=period, interval=interval)

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure we have the dollar value column
    if 'Close' in data.columns and 'Volume' in data.columns:
        data['DollarVolume'] = data['Close'] * data['Volume']

    return data


def dollar_bars(
    df: pd.DataFrame,
    dollar_threshold: float = 1_000_000,
    price_col: str = 'Close',
    volume_col: str = 'Volume'
) -> pd.DataFrame:
    """
    Convert time-based bars to dollar bars.
    
    Dollar bars sample the data every time a fixed dollar amount is traded,
    making the bars more uniform in terms of information content.
    
    Args:
        df: DataFrame with price and volume columns
        dollar_threshold: Dollar amount to trigger new bar (default: $1M)
        price_col: Column name for price
        volume_col: Column name for volume
    
    Returns:
        DataFrame with dollar bars (OHLCV)
    
    Reference:
        Advances in Financial Machine Learning, Chapter 2
        "Dollar bars are the most important type of information-driven bars"
    """
    # Calculate dollar volume for each tick
    df = df.copy()
    df['DollarVol'] = df[price_col] * df[volume_col]
    
    # Accumulate dollar volume
    df['CumDollarVol'] = df['DollarVol'].cumsum()
    
    # Determine bar boundaries
    df['BarIndex'] = (df['CumDollarVol'] // dollar_threshold).astype(int)
    
    # Aggregate into bars
    bars = df.groupby('BarIndex').agg({
        price_col: ['first', 'max', 'min', 'last'],
        volume_col: 'sum',
        'DollarVol': 'sum'
    })
    
    # Flatten column names
    bars.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'DollarVolume']
    
    # Add timestamp (use the last timestamp in each bar)
    bar_times = df.groupby('BarIndex').apply(lambda x: x.index[-1])
    bars.index = bar_times.values
    bars.index.name = 'Timestamp'
    
    return bars


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calculate weights for Fixed-Width Window Fractional Differentiation.
    
    The weights follow the formula:
    w_k = -w_{k-1} * (d - k + 1) / k
    
    Args:
        d: Fractional differentiation order (0 < d < 1)
        threshold: Minimum weight value to include
    
    Returns:
        Array of weights
    
    Reference:
        Advances in Financial Machine Learning, Chapter 5
    """
    weights = [1.0]
    k = 1
    
    while abs(weights[-1]) >= threshold:
        w_k = -weights[-1] * (d - k + 1) / k
        weights.append(w_k)
        k += 1
    
    # Remove the last weight that's below threshold
    weights = np.array(weights[:-1])
    
    return weights


def frac_diff_fixed(
    series: pd.Series,
    d: float = 0.4,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differentiation.
    
    This method makes the time series stationary while preserving memory,
    unlike standard differencing (d=1) which destroys valuable information.
    
    Args:
        series: Price series to differentiate
        d: Differentiation order (0 < d < 1, recommended: 0.3-0.5)
        threshold: Minimum weight to include
    
    Returns:
        Fractionally differentiated series
    
    Mathematical intuition:
        - d=0: Original series (non-stationary but full memory)
        - d=1: First difference (stationary but no memory)
        - 0<d<1: Balance between stationarity and memory
    
    Reference:
        Advances in Financial Machine Learning, Chapter 5
        "The goal is to make the series stationary without erasing all memory"
    """
    # Get weights
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    
    # Apply convolution
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values
        result.iloc[i] = np.dot(weights[::-1], window)
    
    return result.dropna()


def find_min_d_for_stationarity(
    series: pd.Series,
    max_d: float = 1.0,
    step: float = 0.05,
    significance: float = 0.05
) -> Tuple[float, pd.DataFrame]:
    """
    Find the minimum d value that makes the series stationary.
    
    Uses the Augmented Dickey-Fuller test to check for stationarity.
    
    Args:
        series: Price series
        max_d: Maximum d value to test
        step: Step size for d values
        significance: ADF test significance level
    
    Returns:
        Tuple of (optimal_d, results_dataframe)
    """
    from statsmodels.tsa.stattools import adfuller
    
    results = []
    d_values = np.arange(0, max_d + step, step)
    
    for d in d_values:
        if d == 0:
            frac_series = series
        else:
            frac_series = frac_diff_fixed(series, d=d)
        
        if len(frac_series.dropna()) < 20:
            continue
        
        adf_result = adfuller(frac_series.dropna())
        
        results.append({
            'd': d,
            'adf_stat': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < significance
        })
    
    results_df = pd.DataFrame(results)
    
    # Find minimum d that achieves stationarity
    stationary = results_df[results_df['is_stationary']]
    optimal_d = stationary['d'].min() if len(stationary) > 0 else max_d
    
    return optimal_d, results_df


def compute_correlation_with_original(
    series: pd.Series,
    d: float
) -> float:
    """
    Compute correlation between original and fractionally differentiated series.
    
    Higher correlation means more memory is preserved.
    
    Args:
        series: Original price series
        d: Fractional differentiation order
    
    Returns:
        Correlation coefficient
    """
    frac_series = frac_diff_fixed(series, d=d)
    
    # Align series
    common_idx = series.index.intersection(frac_series.index)
    
    return series.loc[common_idx].corr(frac_series.loc[common_idx])


def plot_fracdiff_analysis(
    series: pd.Series,
    d_values: list = None,
    figsize: tuple = (14, 10)
):
    """
    Plot analysis of fractional differentiation at various d values.
    
    Args:
        series: Price series
        d_values: List of d values to test
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    if d_values is None:
        d_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig, axes = plt.subplots(len(d_values), 1, figsize=figsize)
    
    for i, d in enumerate(d_values):
        if d == 0:
            frac_series = series
            title = "Original Series (d=0)"
        elif d == 1:
            frac_series = series.diff().dropna()
            title = "First Difference (d=1)"
        else:
            frac_series = frac_diff_fixed(series, d=d)
            title = f"FracDiff (d={d})"
        
        corr = series.loc[frac_series.index].corr(frac_series) if d > 0 else 1.0
        
        axes[i].plot(frac_series)
        axes[i].set_title(f"{title} - Corr with original: {corr:.3f}")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fracdiff_analysis.png', dpi=150)
    plt.close()
    
    print("Saved plot to fracdiff_analysis.png")


class DataProcessor:
    """
    Complete data processing pipeline for quantitative trading.
    
    Converts raw market data → Dollar Bars → FracDiff (Stationary)
    """
    
    def __init__(
        self,
        ticker: str = "SPY",
        dollar_threshold: float = 1_000_000,
        frac_d: float = 0.4
    ):
        self.ticker = ticker
        self.dollar_threshold = dollar_threshold
        self.frac_d = frac_d
        self._raw_data = None
        self._dollar_bars = None
        self._frac_diff = None
    
    def fetch(self, period: str = "7d", interval: str = "1m") -> pd.DataFrame:
        """Fetch raw market data."""
        self._raw_data = fetch_data(self.ticker, period, interval)
        return self._raw_data
    
    def process(self) -> pd.DataFrame:
        """Run full processing pipeline."""
        if self._raw_data is None:
            self.fetch()
        
        # Step 1: Convert to dollar bars
        self._dollar_bars = dollar_bars(
            self._raw_data,
            dollar_threshold=self.dollar_threshold
        )
        
        # Step 2: Apply fractional differentiation
        self._frac_diff = frac_diff_fixed(
            self._dollar_bars['Close'],
            d=self.frac_d
        )
        
        # Combine results
        result = self._dollar_bars.copy()
        result['FracDiff'] = self._frac_diff
        
        return result.dropna()
    
    def get_features(self) -> pd.DataFrame:
        """Get processed features for ML model."""
        if self._frac_diff is None:
            self.process()
        
        features = pd.DataFrame(index=self._dollar_bars.index)
        features['FracDiff'] = self._frac_diff
        features['Volatility'] = self._dollar_bars['Close'].rolling(20).std()
        features['Volume'] = self._dollar_bars['Volume']
        
        return features.dropna()


if __name__ == "__main__":
    print("Testing Data Processor...")
    print("=" * 50)
    
    # Test with sample data
    processor = DataProcessor(ticker="SPY", dollar_threshold=500_000)
    
    print("\n1. Fetching data...")
    raw = processor.fetch(period="7d", interval="1m")
    print(f"   Raw data shape: {raw.shape}")
    
    print("\n2. Processing to dollar bars...")
    processed = processor.process()
    print(f"   Dollar bars shape: {processed.shape}")
    
    print("\n3. Getting features...")
    features = processor.get_features()
    print(f"   Feature shape: {features.shape}")
    print(f"   Features: {list(features.columns)}")
    
    print("\n4. Sample output:")
    print(features.head())
