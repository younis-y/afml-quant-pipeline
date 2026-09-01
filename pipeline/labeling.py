"""
AFML Quant Pipeline - Triple Barrier Labeling
Labels trades based on profit target, stop loss, and time barriers
Reference: Advances in Financial Machine Learning, Chapter 3
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit


def get_volatility(
    close: pd.Series,
    span: int = 20,
    min_periods: int = 5
) -> pd.Series:
    """
    Compute volatility as exponentially weighted moving standard deviation.
    
    Args:
        close: Close price series
        span: EWM span parameter
        min_periods: Minimum periods for valid output
    
    Returns:
        Volatility series
    """
    # Daily returns
    returns = close.pct_change()
    
    # EWM standard deviation
    volatility = returns.ewm(span=span, min_periods=min_periods).std()
    
    return volatility


def get_horizontal_barriers(
    close: pd.Series,
    volatility: pd.Series,
    profit_mult: float = 2.0,
    stop_mult: float = 1.0
) -> pd.DataFrame:
    """
    Calculate horizontal barriers (profit take and stop loss levels).
    
    Args:
        close: Close price series
        volatility: Volatility series
        profit_mult: Multiplier for profit target (volatility * mult)
        stop_mult: Multiplier for stop loss (volatility * mult)
    
    Returns:
        DataFrame with 'upper' and 'lower' barrier columns
    """
    barriers = pd.DataFrame(index=close.index)
    
    barriers['upper'] = close + (volatility * profit_mult)
    barriers['lower'] = close - (volatility * stop_mult)
    
    return barriers


def get_vertical_barriers(
    close: pd.Series,
    num_bars: int = 50
) -> pd.Series:
    """
    Calculate vertical barriers (time-based exit).
    
    Args:
        close: Close price series
        num_bars: Number of bars until time exit
    
    Returns:
        Series mapping each index to its time barrier index
    """
    # Get the exit timestamp for each entry
    timestamps = close.index
    
    vertical = pd.Series(index=timestamps, dtype='datetime64[ns]')
    
    for i, ts in enumerate(timestamps):
        exit_idx = min(i + num_bars, len(timestamps) - 1)
        vertical.iloc[i] = timestamps[exit_idx]
    
    return vertical


@jit(nopython=True)
def _find_first_barrier_hit(
    prices: np.ndarray,
    upper: float,
    lower: float,
    max_idx: int
) -> Tuple[int, int]:
    """
    JIT-compiled function to find which barrier is hit first.
    
    Returns:
        Tuple of (hit_index, label)
        label: 1 = upper hit, -1 = lower hit, 0 = timeout
    """
    for i in range(max_idx):
        if prices[i] >= upper:
            return i, 1
        elif prices[i] <= lower:
            return i, -1
    
    return max_idx - 1, 0


def triple_barrier_labels(
    close: pd.Series,
    volatility: Optional[pd.Series] = None,
    profit_mult: float = 2.0,
    stop_mult: float = 1.0,
    num_bars: int = 50,
    volatility_span: int = 20
) -> pd.DataFrame:
    """
    Apply the Triple Barrier Method to label trades.
    
    The method sets three barriers:
    1. Upper (Profit Take): Price + (Volatility × profit_mult)
    2. Lower (Stop Loss): Price - (Volatility × stop_mult)
    3. Vertical (Time): After num_bars periods
    
    Labels:
    - +1: Upper barrier hit first (profitable trade)
    - -1: Lower barrier hit first (losing trade)
    -  0: Vertical barrier hit first (time exit, uncertain)
    
    Args:
        close: Close price series
        volatility: Optional pre-computed volatility (if None, computed internally)
        profit_mult: Profit target multiplier
        stop_mult: Stop loss multiplier
        num_bars: Time barrier in number of bars
        volatility_span: EWM span for volatility calculation
    
    Returns:
        DataFrame with columns:
        - 'label': Trade outcome (+1, -1, 0)
        - 'barrier_hit': Which barrier was hit ('upper', 'lower', 'vertical')
        - 'bars_until_hit': Number of bars until the barrier was hit
        - 'upper_barrier': Upper barrier price
        - 'lower_barrier': Lower barrier price
    
    Reference:
        Advances in Financial Machine Learning, Chapter 3
        "The idea is to label an observation according to the first barrier touched"
    """
    # Ensure we have volatility
    if volatility is None:
        volatility = get_volatility(close, span=volatility_span)
    
    # Calculate barriers
    horizontal = get_horizontal_barriers(close, volatility, profit_mult, stop_mult)
    
    # Prepare result DataFrame
    result = pd.DataFrame(index=close.index)
    result['label'] = np.nan
    result['barrier_hit'] = ''
    result['bars_until_hit'] = np.nan
    result['upper_barrier'] = horizontal['upper']
    result['lower_barrier'] = horizontal['lower']
    result['entry_price'] = close
    
    prices = close.values
    n = len(prices)
    
    # Process each bar
    for i in range(n - num_bars):
        upper = horizontal['upper'].iloc[i]
        lower = horizontal['lower'].iloc[i]
        
        # Look ahead up to num_bars
        future_prices = prices[i:i + num_bars + 1]
        
        hit_idx, label = _find_first_barrier_hit(
            future_prices,
            upper,
            lower,
            num_bars + 1
        )
        
        result.iloc[i, result.columns.get_loc('label')] = label
        result.iloc[i, result.columns.get_loc('bars_until_hit')] = hit_idx
        
        if label == 1:
            result.iloc[i, result.columns.get_loc('barrier_hit')] = 'upper'
        elif label == -1:
            result.iloc[i, result.columns.get_loc('barrier_hit')] = 'lower'
        else:
            result.iloc[i, result.columns.get_loc('barrier_hit')] = 'vertical'
    
    return result.dropna(subset=['label'])


def get_meta_labels(
    primary_signal: pd.Series,
    triple_barrier_result: pd.DataFrame
) -> pd.Series:
    """
    Generate meta-labels for secondary model training.
    
    Meta-labels indicate whether the primary model's signal was correct.
    
    Args:
        primary_signal: Binary signal from primary model (1 = long, -1 = short, 0 = no position)
        triple_barrier_result: Result from triple_barrier_labels()
    
    Returns:
        Series with meta-labels:
        - 1: Primary signal was correct (profitable)
        - 0: Primary signal was wrong (losing)
    
    Reference:
        Advances in Financial Machine Learning, Chapter 3
        "Meta-labeling is about learning to size positions"
    """
    # Align indices
    common_idx = primary_signal.index.intersection(triple_barrier_result.index)
    
    signal = primary_signal.loc[common_idx]
    labels = triple_barrier_result.loc[common_idx, 'label']
    
    # Meta-label: 1 if signal direction matches outcome
    meta_labels = pd.Series(index=common_idx, dtype=int)
    
    # Long signals (signal == 1) are correct if label == 1
    # Short signals (signal == -1) are correct if label == -1
    meta_labels = ((signal == 1) & (labels == 1)) | ((signal == -1) & (labels == -1))
    meta_labels = meta_labels.astype(int)
    
    return meta_labels


def compute_label_statistics(labels: pd.DataFrame) -> dict:
    """
    Compute statistics about the labeling.
    
    Args:
        labels: Result from triple_barrier_labels()
    
    Returns:
        Dict with label statistics
    """
    label_col = labels['label']
    
    stats = {
        'total_samples': len(label_col),
        'positive_labels': (label_col == 1).sum(),
        'negative_labels': (label_col == -1).sum(),
        'neutral_labels': (label_col == 0).sum(),
        'positive_ratio': (label_col == 1).mean(),
        'negative_ratio': (label_col == -1).mean(),
        'avg_bars_until_hit': labels['bars_until_hit'].mean(),
        'median_bars_until_hit': labels['bars_until_hit'].median()
    }
    
    return stats


class TripleBarrierLabeler:
    """
    Wrapper class for Triple Barrier labeling with configuration.
    """
    
    def __init__(
        self,
        profit_mult: float = 2.0,
        stop_mult: float = 1.0,
        num_bars: int = 50,
        volatility_span: int = 20
    ):
        self.profit_mult = profit_mult
        self.stop_mult = stop_mult
        self.num_bars = num_bars
        self.volatility_span = volatility_span
    
    def fit_transform(self, close: pd.Series) -> pd.DataFrame:
        """Apply labeling to price series."""
        return triple_barrier_labels(
            close,
            profit_mult=self.profit_mult,
            stop_mult=self.stop_mult,
            num_bars=self.num_bars,
            volatility_span=self.volatility_span
        )
    
    def get_params(self) -> dict:
        """Get labeling parameters."""
        return {
            'profit_mult': self.profit_mult,
            'stop_mult': self.stop_mult,
            'num_bars': self.num_bars,
            'volatility_span': self.volatility_span
        }


if __name__ == "__main__":
    print("Testing Triple Barrier Labeling...")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 500
    
    # Simulate a random walk with drift
    returns = np.random.randn(n) * 0.02 + 0.0001
    prices = 100 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    close = pd.Series(prices, index=dates, name='Close')
    
    print("\n1. Computing volatility...")
    vol = get_volatility(close)
    print(f"   Mean volatility: {vol.mean():.4f}")
    
    print("\n2. Applying Triple Barrier Method...")
    labeler = TripleBarrierLabeler(profit_mult=2.0, stop_mult=1.0, num_bars=50)
    labels = labeler.fit_transform(close)
    
    print("\n3. Label Statistics:")
    stats = compute_label_statistics(labels)
    for k, v in stats.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    
    print("\n4. Sample labels:")
    print(labels.head(10))
