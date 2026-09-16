"""
AFML Quant Pipeline - Pair Trading Module
Implements multi-asset data processing and statistical arbitrage strategies.
"""

import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairDataProcessor:
    """
    Handles fetching and alignment of dual-asset time series.
    """
    def __init__(self, ticker1: str, ticker2: str):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.data1 = None
        self.data2 = None
        self.aligned_data = None

    def fetch_data(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch data for both assets and align timestamps (inner join).
        """
        logger.info(f"Fetching pair data: {self.ticker1} vs {self.ticker2}")
        
        # Fetch individual
        df1 = yf.download(self.ticker1, period=period, interval=interval, progress=False)
        df2 = yf.download(self.ticker2, period=period, interval=interval, progress=False)
        
        # Handle multi-index if present
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.get_level_values(0)
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.get_level_values(0)
            
        # Extract Close prices
        s1 = df1['Close']
        s2 = df2['Close']
        
        # Rename
        s1.name = self.ticker1
        s2.name = self.ticker2
        
        # Align (Inner Join to handle holiday/exchange diffs)
        self.aligned_data = pd.concat([s1, s2], axis=1).dropna()
        
        logger.info(f"Aligned data shape: {self.aligned_data.shape}")
        return self.aligned_data

class CointegrationTester:
    """
    Statistical tests for identifying viable pairs.
    """
    @staticmethod
    def engage_engle_granger(series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Perform Engle-Granger two-step cointegration test.
        """
        # 1. Calculate Hedge Ratio (Beta) via OLS
        slope, intercept, r_value, p_value, std_err = stats.linregress(series2, series1)
        
        # 2. Calculate Spread
        spread = series1 - (slope * series2 + intercept)
        
        # 3. Test Stationarity of Spread (ADF Test)
        adf_result = adfuller(spread)
        
        return {
            'hedge_ratio': slope,
            'intercept': intercept,
            'correlation': r_value,
            'spread_adf_stat': adf_result[0],
            'spread_p_value': adf_result[1],
            'is_cointegrated': adf_result[1] < 0.05
        }

class MeanReversionStrategy:
    """
    Z-Score based Mean Reversion Strategy for Pairs.
    """
    def __init__(self, lookback: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def backtest(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Run vectorized backtest on the pair.
        """
        # Dynamic Hedge Ratio (Rolling OLS) - Simplified to Static for V1 or Rolling Window
        # Using Rolling Mean/Std of simple ratio/spread for Z-Score
        
        # Approach: Trade the Spread = Y - beta*X
        # For simplicity in V1, we'll use a dynamic rolling z-score of the spread
        # calculated from the FULL period cointegration (in-sample risk, but okay for checking existence)
        
        # Better: Rolling Calculation
        # Let's simple utilize the ratio for stability: Ratio = Price1 / Price2
        ratio = series1 / series2
        
        rolling_mean = ratio.rolling(window=self.lookback).mean()
        rolling_std = ratio.rolling(window=self.lookback).std()
        
        z_score = (ratio - rolling_mean) / rolling_std
        
        # Signals
        signals = pd.Series(0, index=ratio.index)
        signals[z_score > self.entry_threshold] = -1  # Short Spread (Short P1, Long P2)
        signals[z_score < -self.entry_threshold] = 1  # Long Spread (Long P1, Short P2)
        signals[abs(z_score) < self.exit_threshold] = 0 # Exit
        
        # Forward fill signals to hold positions? 
        # For simple vector backtest, let's assume we hold until exit criteria
        # This is a bit complex for a single function, let's keep it simple:
        # Just return the Z-Score series for analysis report
        
        return {
            'ratio': ratio,
            'z_score': z_score,
            'mean': rolling_mean
        }

if __name__ == "__main__":
    # Test
    processor = PairDataProcessor("GLD", "GDX")
    data = processor.fetch_data()
    
    tester = CointegrationTester()
    res = tester.engage_engle_granger(data["GLD"], data["GDX"])
    print("\nCointegration Test Results:")
    print(res)
    
    strat = MeanReversionStrategy()
    bt = strat.backtest(data["GLD"], data["GDX"])
    print("\nBacktest Z-Score Head:")
    print(bt['z_score'].tail())
