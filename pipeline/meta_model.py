"""
AFML Quant Pipeline - Meta-Labeling Model
Primary model (SMA Crossover) + Secondary model (Random Forest)
Reference: Advances in Financial Machine Learning, Chapter 3
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

from .labeling import triple_barrier_labels, get_meta_labels, get_volatility


@dataclass
class PrimarySignal:
    """Container for primary model signal."""
    signal: int  # 1 = long, -1 = short, 0 = neutral
    strength: float  # 0-1 confidence
    timestamp: pd.Timestamp


class SMACrossover:
    """
    Simple Moving Average Crossover - Primary Model.
    
    Generates binary BUY/SELL signals based on SMA crossover.
    This is intentionally "dumb" - the meta-model filters it.
    """
    
    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def fit(self, close: pd.Series) -> 'SMACrossover':
        """No fitting needed for SMA crossover."""
        return self
    
    def generate_signals(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals from SMA crossover.
        
        Args:
            close: Close price series
        
        Returns:
            DataFrame with:
            - 'sma_fast': Fast SMA
            - 'sma_slow': Slow SMA
            - 'signal': 1 (bullish), -1 (bearish), 0 (neutral)
            - 'crossover': 1 (golden cross), -1 (death cross), 0 (none)
        """
        result = pd.DataFrame(index=close.index)
        
        # Calculate SMAs
        result['sma_fast'] = close.rolling(self.fast_period).mean()
        result['sma_slow'] = close.rolling(self.slow_period).mean()
        
        # Signal: 1 when fast > slow (bullish), -1 when fast < slow (bearish)
        result['signal'] = np.where(
            result['sma_fast'] > result['sma_slow'], 1,
            np.where(result['sma_fast'] < result['sma_slow'], -1, 0)
        )
        
        # Crossover detection
        result['prev_signal'] = result['signal'].shift(1)
        result['crossover'] = np.where(
            (result['signal'] == 1) & (result['prev_signal'] == -1), 1,  # Golden cross
            np.where(
                (result['signal'] == -1) & (result['prev_signal'] == 1), -1,  # Death cross
                0
            )
        )
        
        result.drop('prev_signal', axis=1, inplace=True)
        
        return result.dropna()
    
    def get_params(self) -> Dict[str, int]:
        """Get model parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }


class MetaModel:
    """
    Meta-Labeling Model - The "Brain" that filters primary signals.
    
    Uses a Random Forest to predict the probability that the primary
    model's signal will be profitable.
    
    Reference:
        Advances in Financial Machine Learning, Chapter 3
        "The secondary model predicts the probability of the signal being correct"
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.is_fitted = False
    
    def prepare_features(
        self,
        close: pd.Series,
        frac_diff: pd.Series,
        volatility: Optional[pd.Series] = None,
        additional_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare feature matrix for the meta-model.
        
        Args:
            close: Close price series
            frac_diff: Fractionally differentiated prices
            volatility: Optional volatility series
            additional_features: Optional additional features
        
        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame(index=close.index)
        
        # Core features
        if frac_diff is not None:
            features['frac_diff'] = frac_diff
        
        if volatility is not None:
            features['volatility'] = volatility
        else:
            features['volatility'] = get_volatility(close)
        
        # Additional derived features
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        # Volume features (if available)
        if additional_features is not None:
            for col in additional_features.columns:
                features[col] = additional_features[col]
        
        self.feature_names = list(features.columns)
        
        return features.dropna()
    
    def fit(
        self,
        features: pd.DataFrame,
        meta_labels: pd.Series
    ) -> 'MetaModel':
        """
        Train the meta-model.
        
        Args:
            features: Feature matrix
            meta_labels: Meta-labels (1 = primary signal correct, 0 = wrong)
        
        Returns:
            self
        """
        # Align
        common_idx = features.index.intersection(meta_labels.index)
        X = features.loc[common_idx]
        y = meta_labels.loc[common_idx]
        
        # Remove any remaining NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        
        # Validate we have samples to train on
        if len(X) == 0:
            raise ValueError(
                f"No valid training samples after alignment. "
                f"This usually means insufficient data. Requirements:\n"
                f"  - Minimum bars needed: slow_sma_period + num_bars + 50 (for features)\n"
                f"  - Typical minimum: 300+ bars\n"
                f"  - Common causes: data too short, misaligned indices, all NaN values\n"
                f"  - Features shape: {features.shape if 'features' in dir() else 'N/A'}\n"
                f"  - Meta labels length: {len(meta_labels) if 'meta_labels' in dir() else 'N/A'}"
            )
        
        if len(X) < 50:
            print(f"Warning: only {len(X)} training samples. Results may be unreliable.")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict_probability(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict probability that primary signal is correct.
        
        Args:
            features: Feature matrix
        
        Returns:
            Series of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle NaN
        valid_mask = ~features.isna().any(axis=1)
        valid_features = features.loc[valid_mask]
        
        probs = pd.Series(index=features.index, dtype=float)
        probs.loc[valid_mask] = self.model.predict_proba(valid_features)[:, 1]
        
        return probs
    
    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Predict whether to take the trade.
        
        Args:
            features: Feature matrix
            threshold: Probability threshold for taking trade
        
        Returns:
            Series of predictions (1 = take trade, 0 = pass)
        """
        probs = self.predict_probability(features)
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
    
    def evaluate(
        self,
        features: pd.DataFrame,
        meta_labels: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            features: Feature matrix
            meta_labels: True meta-labels
        
        Returns:
            Dict with evaluation metrics
        """
        common_idx = features.index.intersection(meta_labels.index)
        X = features.loc[common_idx]
        y = meta_labels.loc[common_idx]
        
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        
        predictions = self.predict(X)
        valid_preds = predictions.loc[X.index]
        
        return {
            'precision': precision_score(y, valid_preds, zero_division=0),
            'recall': recall_score(y, valid_preds, zero_division=0),
            'f1': f1_score(y, valid_preds, zero_division=0)
        }
    
    def save(self, path: str):
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'MetaModel':
        """Load model from file."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.is_fitted = data['is_fitted']
        return instance


class MetaLabelingPipeline:
    """
    Complete Meta-Labeling pipeline.
    
    Combines:
    1. Primary Model (SMA Crossover)
    2. Triple Barrier Labeling
    3. Meta-Model (Random Forest)
    """
    
    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        profit_mult: float = 2.0,
        stop_mult: float = 1.0,
        num_bars: int = 50,
        confidence_threshold: float = 0.75
    ):
        self.primary = SMACrossover(sma_fast, sma_slow)
        self.meta = MetaModel()
        self.profit_mult = profit_mult
        self.stop_mult = stop_mult
        self.num_bars = num_bars
        self.confidence_threshold = confidence_threshold
    
    def train(
        self,
        close: pd.Series,
        frac_diff: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train the complete pipeline.
        
        Args:
            close: Historical close prices
            frac_diff: Optional fractionally differentiated prices
        
        Returns:
            Training results and metrics
        """
        # Validate input data length
        min_required = self.primary.slow_period + self.num_bars + 100
        if len(close) < min_required:
            raise ValueError(
                f"Insufficient data: got {len(close)} bars, need at least {min_required}.\n"
                f"  - slow_sma_period: {self.primary.slow_period}\n"
                f"  - num_bars (triple barrier): {self.num_bars}\n"
                f"  - feature buffer: 100"
            )
        
        # Step 1: Generate primary signals
        primary_signals = self.primary.generate_signals(close)
        if len(primary_signals) == 0:
            raise ValueError(
                f"Primary model generated 0 signals. Data length: {len(close)}, "
                f"SMA slow period: {self.primary.slow_period}"
            )
        
        # Step 2: Apply triple barrier labeling
        labels = triple_barrier_labels(
            close,
            profit_mult=self.profit_mult,
            stop_mult=self.stop_mult,
            num_bars=self.num_bars
        )
        if len(labels) == 0:
            raise ValueError(
                f"Triple barrier labeling produced 0 labels. "
                f"Check that data length ({len(close)}) > num_bars ({self.num_bars})"
            )
        
        # Step 3: Generate meta-labels
        meta_labels = get_meta_labels(primary_signals['signal'], labels)
        if len(meta_labels) == 0:
            raise ValueError(
                f"Meta-labels generation produced 0 samples.\n"
                f"  - Primary signals: {len(primary_signals)} (index: {primary_signals.index[0]} to {primary_signals.index[-1]})\n"
                f"  - Labels: {len(labels)} (index: {labels.index[0]} to {labels.index[-1]})\n"
                f"  - Overlap: {len(primary_signals.index.intersection(labels.index))} indices"
            )
        
        # Step 4: Prepare features
        volatility = get_volatility(close)
        features = self.meta.prepare_features(close, frac_diff, volatility)
        if len(features) == 0:
            raise ValueError(
                "Feature preparation produced 0 samples. Check for NaN values in input data."
            )
        
        # Step 5: Align features and meta_labels
        common_idx = features.index.intersection(meta_labels.index)
        if len(common_idx) == 0:
            raise ValueError(
                f"No overlapping indices between features and meta_labels.\n"
                f"  - Features: {len(features)} (index: {features.index[0]} to {features.index[-1]})\n"
                f"  - Meta-labels: {len(meta_labels)} (index: {meta_labels.index[0]} to {meta_labels.index[-1]})"
            )
        
        # Debug info
        print(f"Training with {len(common_idx)} samples")
        print(f"   - Primary signals: {len(primary_signals)}")
        print(f"   - Triple barrier labels: {len(labels)}")
        print(f"   - Meta-labels: {len(meta_labels)}")
        print(f"   - Features: {len(features)}")
        
        # Step 6: Train meta-model
        self.meta.fit(features, meta_labels)
        
        # Step 7: Evaluate
        metrics = self.meta.evaluate(features, meta_labels)
        
        return {
            'metrics': metrics,
            'feature_importance': self.meta.get_feature_importance(),
            'primary_signals_count': len(primary_signals),
            'labels_count': len(labels),
            'meta_labels_positive_rate': meta_labels.mean(),
            'training_samples': len(common_idx)
        }
    
    def predict(
        self,
        close: pd.Series,
        frac_diff: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate filtered trading signals.
        
        Args:
            close: Price series
            frac_diff: Optional fractionally differentiated prices
        
        Returns:
            DataFrame with:
            - 'primary_signal': Raw signal from SMA
            - 'confidence': Meta-model probability
            - 'filtered_signal': Signal after meta-model filter
            - 'action': 'BUY', 'SELL', or 'PASS'
        """
        # Get primary signals
        signals = self.primary.generate_signals(close)
        
        # Prepare features
        volatility = get_volatility(close)
        features = self.meta.prepare_features(close, frac_diff, volatility)
        
        # Get meta-model confidence
        confidence = self.meta.predict_probability(features)
        
        # Combine results
        result = pd.DataFrame(index=close.index)
        result['primary_signal'] = signals['signal']
        result['confidence'] = confidence
        
        # Filter: only take signal if confidence >= threshold
        result['filtered_signal'] = np.where(
            result['confidence'] >= self.confidence_threshold,
            result['primary_signal'],
            0
        )
        
        # Human-readable action
        result['action'] = np.where(
            result['filtered_signal'] == 1, 'BUY',
            np.where(result['filtered_signal'] == -1, 'SELL', 'PASS')
        )
        
        return result.dropna()
    
    def save(self, path: str):
        """Save pipeline to file."""
        joblib.dump({
            'primary': self.primary,
            'meta': self.meta,
            'profit_mult': self.profit_mult,
            'stop_mult': self.stop_mult,
            'num_bars': self.num_bars,
            'confidence_threshold': self.confidence_threshold
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'MetaLabelingPipeline':
        """Load pipeline from file."""
        data = joblib.load(path)
        instance = cls(
            sma_fast=data['primary'].fast_period,
            sma_slow=data['primary'].slow_period,
            profit_mult=data['profit_mult'],
            stop_mult=data['stop_mult'],
            num_bars=data['num_bars'],
            confidence_threshold=data['confidence_threshold']
        )
        instance.primary = data['primary']
        instance.meta = data['meta']
        return instance


if __name__ == "__main__":
    print("Testing Meta-Labeling Model...")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    # Simulate a random walk with some trend
    returns = np.random.randn(n) * 0.02 + 0.0002
    prices = 100 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    close = pd.Series(prices, index=dates, name='Close')
    
    print("\n1. Testing Primary Model (SMA Crossover)...")
    sma = SMACrossover(fast_period=20, slow_period=50)  # Shorter for test
    signals = sma.generate_signals(close)
    
    print(f"   Total signals: {len(signals)}")
    print(f"   Bullish periods: {(signals['signal'] == 1).sum()}")
    print(f"   Bearish periods: {(signals['signal'] == -1).sum()}")
    print(f"   Golden crosses: {(signals['crossover'] == 1).sum()}")
    print(f"   Death crosses: {(signals['crossover'] == -1).sum()}")
    
    print("\n2. Testing Complete Pipeline...")
    pipeline = MetaLabelingPipeline(
        sma_fast=20,
        sma_slow=50,
        num_bars=30,
        confidence_threshold=0.6
    )
    
    results = pipeline.train(close)
    
    print("\n   Training Metrics:")
    for k, v in results['metrics'].items():
        print(f"   - {k}: {v:.4f}")
    
    print("\n   Feature Importance:")
    for feat, imp in results['feature_importance'].head(5).items():
        print(f"   - {feat}: {imp:.4f}")
    
    print("\n3. Generating Predictions...")
    predictions = pipeline.predict(close)
    
    print("\n   Action Distribution:")
    print(predictions['action'].value_counts())
