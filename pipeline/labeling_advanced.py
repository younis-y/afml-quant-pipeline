"""
AFML Quant Pipeline - Advanced Labeling Methods
Implements state-of-the-art financial labeling from AFML:
- Triple Barrier Method with dynamic thresholds
- Trend-Scanning Labels  
- Meta-Labels generation
- Event-driven sampling (CUSUM filter)

Reference: AFML Chapters 3, 17
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from numba import jit


# =============================================================================
# VOLATILITY ESTIMATION
# =============================================================================

def get_daily_volatility(
    close: pd.Series,
    span: int = 20,
    method: str = 'ewm'
) -> pd.Series:
    """Compute daily volatility."""
    returns = close.pct_change().dropna()
    
    if method == 'ewm':
        volatility = returns.ewm(span=span).std()
    elif method == 'rolling':
        volatility = returns.rolling(span).std()
    else:
        volatility = returns.ewm(span=span).std()
    
    return volatility


def get_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    span: int = 20
) -> pd.Series:
    """Parkinson volatility estimator using high-low range."""
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2))
    parkinson = np.sqrt(factor * (log_hl ** 2).rolling(span).mean())
    return parkinson


# =============================================================================
# TRIPLE BARRIER METHOD
# =============================================================================

@jit(nopython=True)
def _find_barrier_touch(
    prices: np.ndarray,
    upper_barrier: float,
    lower_barrier: float,
    max_bars: int
) -> Tuple[int, int]:
    """JIT-optimized barrier touch detection."""
    for i in range(min(max_bars, len(prices))):
        if prices[i] >= upper_barrier:
            return i, 1
        if prices[i] <= lower_barrier:
            return i, -1
    return min(max_bars - 1, len(prices) - 1), 0


def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: int = 50
) -> pd.Series:
    """Add vertical (time) barrier."""
    t1 = pd.Series(index=t_events, dtype='datetime64[ns]')
    
    for t0 in t_events:
        try:
            loc = close.index.get_loc(t0)
            end_loc = min(loc + num_bars, len(close.index) - 1)
            t1.loc[t0] = close.index[end_loc]
        except KeyError:
            t1.loc[t0] = pd.NaT
    
    return t1


def triple_barrier_labels(
    close: pd.Series,
    volatility: Optional[pd.Series] = None,
    profit_mult: float = 2.0,
    stop_mult: float = 1.0,
    num_bars: int = 50,
    volatility_span: int = 20,
    min_ret: float = 0.001,
    side: pd.Series = None
) -> pd.DataFrame:
    """
    Complete Triple Barrier labeling pipeline.
    
    Labels:
    +1: Upper barrier hit first (profitable for long)
    -1: Lower barrier hit first (stop loss for long)
     0: Vertical barrier hit (time exit)
    """
    # Compute volatility if not provided
    if volatility is None:
        volatility = get_daily_volatility(close, span=volatility_span)
    
    # Get events (use all timestamps minus buffer at end)
    # Also need to account for volatility warmup period
    start_idx = max(volatility_span, 1)
    end_idx = len(close) - num_bars if num_bars > 0 else len(close)
    t_events = close.index[start_idx:end_idx]
    
    # Add vertical barriers
    t1 = add_vertical_barrier(t_events, close, num_bars)
    
    # Build result DataFrame
    result = pd.DataFrame(index=t_events)
    result['t1'] = t1
    
    # Reindex volatility to match t_events
    result['trgt'] = volatility.reindex(t_events)
    result['label'] = np.nan
    result['touch_time'] = pd.NaT
    result['ret'] = np.nan
    result['entry_price'] = close.loc[t_events]
    
    if side is not None:
        result['side'] = side.loc[t_events]
    else:
        result['side'] = 1
    
    # Filter by minimum return
    result = result[result['trgt'] > min_ret]
    
    if len(result) == 0:
        return result
    
    # Apply barriers
    prices = close.values
    price_index = close.index
    
    for t0, row in result.iterrows():
        t1_val = row['t1']
        if pd.isna(t1_val):
            continue
        
        try:
            start_loc = price_index.get_loc(t0)
            end_loc = price_index.get_loc(t1_val)
        except:
            continue
        
        price_path = prices[start_loc:end_loc + 1]
        if len(price_path) < 2:
            continue
        
        entry_price = price_path[0]
        vol = row['trgt']
        trade_side = row['side']
        
        # Set barriers
        if trade_side == 1:  # Long
            upper = entry_price * (1 + profit_mult * vol)
            lower = entry_price * (1 - stop_mult * vol)
        else:  # Short
            upper = entry_price * (1 + stop_mult * vol)
            lower = entry_price * (1 - profit_mult * vol)
        
        # Find barrier touch
        touch_idx, touch_type = _find_barrier_touch(
            price_path, upper, lower, len(price_path)
        )
        
        result.loc[t0, 'touch_time'] = price_index[start_loc + touch_idx]
        result.loc[t0, 'ret'] = (price_path[touch_idx] / entry_price - 1) * trade_side
        result.loc[t0, 'label'] = touch_type if trade_side == 1 else -touch_type
    
    return result.dropna(subset=['label'])


# =============================================================================
# META-LABELING
# =============================================================================

def get_meta_labels(
    primary_signal: pd.Series,
    triple_barrier_result: pd.DataFrame
) -> pd.Series:
    """Generate meta-labels for secondary model training."""
    common_idx = primary_signal.index.intersection(triple_barrier_result.index)
    
    if len(common_idx) == 0:
        return pd.Series(dtype=int)
    
    signals = primary_signal.loc[common_idx]
    labels = triple_barrier_result.loc[common_idx, 'label']
    
    meta_labels = ((signals == 1) & (labels == 1)) | ((signals == -1) & (labels == -1))
    
    return meta_labels.astype(int)


# =============================================================================
# EVENT-DRIVEN SAMPLING
# =============================================================================

def cusum_filter(
    close: pd.Series,
    threshold: float = None
) -> pd.DatetimeIndex:
    """CUSUM Filter for sampling based on cumulative sums."""
    log_close = np.log(close)
    diff = log_close.diff().dropna()
    
    if threshold is None:
        threshold = diff.std()
    
    events = []
    s_pos = 0
    s_neg = 0
    
    for idx, val in diff.items():
        s_pos = max(0, s_pos + val)
        s_neg = min(0, s_neg + val)
        
        if s_pos > threshold:
            events.append(idx)
            s_pos = 0
        
        if s_neg < -threshold:
            events.append(idx)
            s_neg = 0
    
    return pd.DatetimeIndex(events)


# =============================================================================
# TREND-SCANNING LABELS
# =============================================================================

def trend_scanning_labels(
    close: pd.Series,
    horizons: List[int] = None,
    min_window: int = 10
) -> pd.DataFrame:
    """Trend-Scanning Labels: Find the most significant trend direction."""
    from scipy.stats import linregress
    
    if horizons is None:
        horizons = [5, 10, 20, 40, 60]
    
    result = pd.DataFrame(index=close.index)
    result['t1'] = pd.NaT
    result['tval'] = np.nan
    result['label'] = np.nan
    result['horizon'] = np.nan
    
    log_prices = np.log(close)
    
    for i in range(min_window, len(close) - max(horizons)):
        best_tval = 0
        best_horizon = horizons[0]
        
        for h in horizons:
            if i + h > len(close):
                continue
            
            y = log_prices.iloc[i:i+h].values
            x = np.arange(len(y))
            
            if len(y) < 3:
                continue
            
            try:
                slope, _, r_value, _, std_err = linregress(x, y)
                
                if std_err > 0:
                    tval = slope / std_err
                else:
                    continue
                
                if abs(tval) > abs(best_tval):
                    best_tval = tval
                    best_horizon = h
                    
            except:
                continue
        
        result.iloc[i, result.columns.get_loc('tval')] = best_tval
        result.iloc[i, result.columns.get_loc('label')] = np.sign(best_tval)
        result.iloc[i, result.columns.get_loc('horizon')] = best_horizon
        
        try:
            end_idx = min(i + int(best_horizon), len(close) - 1)
            result.iloc[i, result.columns.get_loc('t1')] = close.index[end_idx]
        except:
            pass
    
    return result.dropna(subset=['label'])


# =============================================================================
# LABEL STATISTICS
# =============================================================================

def compute_label_statistics(labels: pd.DataFrame) -> Dict:
    """Compute comprehensive statistics about labels."""
    label_col = labels.get('label', labels.get('bin'))
    
    if label_col is None:
        return {}
    
    stats = {
        'total_samples': len(label_col),
        'positive_labels': int((label_col == 1).sum()),
        'negative_labels': int((label_col == -1).sum()),
        'neutral_labels': int((label_col == 0).sum()),
        'positive_ratio': float((label_col == 1).mean()) if len(label_col) > 0 else 0,
        'negative_ratio': float((label_col == -1).mean()) if len(label_col) > 0 else 0,
    }
    
    if 'ret' in labels.columns:
        stats['avg_return'] = float(labels['ret'].mean())
        stats['return_std'] = float(labels['ret'].std())
    
    return stats


# =============================================================================
# MAIN LABELING PIPELINE
# =============================================================================

class AdvancedLabeler:
    """Complete labeling pipeline with multiple methods."""
    
    def __init__(
        self,
        method: str = 'triple_barrier',
        profit_mult: float = 2.0,
        stop_mult: float = 1.0,
        num_bars: int = 50,
        volatility_span: int = 20,
        event_filter: str = 'all',
        min_ret: float = 0.001
    ):
        self.method = method
        self.profit_mult = profit_mult
        self.stop_mult = stop_mult
        self.num_bars = num_bars
        self.volatility_span = volatility_span
        self.event_filter = event_filter
        self.min_ret = min_ret
        
        self.volatility_ = None
        self.labels_ = None
    
    def fit_transform(
        self,
        close: pd.Series,
        side: pd.Series = None,
        high: pd.Series = None,
        low: pd.Series = None
    ) -> pd.DataFrame:
        """Generate labels using configured method."""
        if high is not None and low is not None:
            self.volatility_ = get_parkinson_volatility(high, low, self.volatility_span)
        else:
            self.volatility_ = get_daily_volatility(close, self.volatility_span)
        
        if self.method == 'triple_barrier':
            self.labels_ = triple_barrier_labels(
                close,
                volatility=self.volatility_,
                profit_mult=self.profit_mult,
                stop_mult=self.stop_mult,
                num_bars=self.num_bars,
                min_ret=self.min_ret,
                side=side
            )
        elif self.method == 'trend_scanning':
            self.labels_ = trend_scanning_labels(close)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.labels_
    
    def get_statistics(self) -> Dict:
        """Get label statistics."""
        if self.labels_ is None:
            return {}
        return compute_label_statistics(self.labels_)


if __name__ == "__main__":
    print("Testing Advanced Labeling Methods...")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    
    returns = np.random.randn(n) * 0.02 + 0.0001
    prices = 100 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    close = pd.Series(prices, index=dates)
    
    print("\n1. Testing volatility estimation...")
    vol = get_daily_volatility(close)
    print(f"   Mean volatility: {vol.mean():.4f}")
    
    print("\n2. Testing CUSUM filter...")
    events = cusum_filter(close)
    print(f"   Found {len(events)} events out of {n} bars")
    
    print("\n3. Testing Triple Barrier labels...")
    labeler = AdvancedLabeler(
        method='triple_barrier',
        profit_mult=2.0,
        stop_mult=1.0,
        num_bars=50
    )
    labels = labeler.fit_transform(close)
    
    stats = labeler.get_statistics()
    print("\n   Label Statistics:")
    for k, v in stats.items():
        print(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")
    
    print("\n4. Testing Meta-labels...")
    primary_signal = pd.Series(np.random.choice([-1, 1], len(labels)), index=labels.index)
    meta_labels = get_meta_labels(primary_signal, labels)
    print(f"   Meta-label distribution: {meta_labels.value_counts().to_dict()}")
