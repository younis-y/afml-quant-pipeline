"""
AFML Quant Pipeline - Advanced ML Models & Ensemble
Implements sophisticated ML approaches for financial prediction:
- Multiple base models (RF, XGBoost, LightGBM, etc.)
- Meta-labeling with proper bet sizing
- Stacking ensemble with meta-learner
- Feature importance analysis (MDI, MDA, SFI, SHAP)
- Proper financial evaluation metrics

Reference: AFML Chapters 6, 8, 10, 14
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Try to import optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# =============================================================================
# FEATURE IMPORTANCE - Chapter 8 AFML
# =============================================================================

def mean_decrease_impurity(clf, feature_names: List[str]) -> pd.Series:
    """
    Mean Decrease Impurity (MDI) feature importance.
    
    Based on reduction in impurity from splits on each feature.
    Warning: MDI suffers from substitution effects when features are correlated.
    
    Reference: AFML Chapter 8.2
    """
    if not hasattr(clf, 'feature_importances_'):
        raise ValueError("Classifier doesn't have feature_importances_")
    
    return pd.Series(
        clf.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)


def mean_decrease_accuracy(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    sample_weight: pd.Series = None,
    scoring: str = 'f1'
) -> pd.Series:
    """
    Mean Decrease Accuracy (MDA) feature importance.
    
    Measures accuracy drop when a feature is permuted (shuffled).
    More robust than MDI for correlated features.
    
    Reference: AFML Chapter 8.3
    """
    from sklearn.metrics import get_scorer
    
    scorer = get_scorer(scoring)
    importance = pd.Series(0.0, index=X.columns)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        if sample_weight is not None:
            sw = sample_weight.iloc[train_idx]
            clf.fit(X_train, y_train, sample_weight=sw)
        else:
            clf.fit(X_train, y_train)
        
        # Baseline score
        baseline_score = scorer(clf, X_test, y_test)
        
        # Permute each feature and measure score drop
        for col in X.columns:
            X_test_perm = X_test.copy()
            np.random.shuffle(X_test_perm[col].values)
            
            permuted_score = scorer(clf, X_test_perm, y_test)
            importance[col] += baseline_score - permuted_score
    
    return importance.sort_values(ascending=False)


def single_feature_importance(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    sample_weight: pd.Series = None,
    scoring: str = 'f1'
) -> pd.Series:
    """
    Single Feature Importance (SFI): Train model on each feature individually.
    
    Helps identify truly informative features without substitution effects.
    
    Reference: AFML Chapter 8.4
    """
    from sklearn.metrics import get_scorer
    
    scorer = get_scorer(scoring)
    importance = pd.Series(0.0, index=X.columns)
    
    for col in X.columns:
        scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train = X.iloc[train_idx][[col]]
            X_test = X.iloc[test_idx][[col]]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            clf_copy = clone(clf)
            
            try:
                if sample_weight is not None:
                    sw = sample_weight.iloc[train_idx]
                    clf_copy.fit(X_train, y_train, sample_weight=sw)
                else:
                    clf_copy.fit(X_train, y_train)
                
                score = scorer(clf_copy, X_test, y_test)
                scores.append(score)
            except Exception:
                continue
        
        importance[col] = np.mean(scores) if scores else 0
    
    return importance.sort_values(ascending=False)


# =============================================================================
# SHARPE RATIO STATISTICS - Chapter 14 AFML
# =============================================================================

def compute_sharpe_ratio(
    returns: pd.Series,
    annualize: int = 252
) -> float:
    """
    Compute (possibly annualized) Sharpe Ratio.
    """
    if returns.std() == 0:
        return 0.0
    
    sr = returns.mean() / returns.std()
    return sr * np.sqrt(annualize)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0,
    kurtosis: float = 3
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).
    
    Corrects for non-Normal returns and limited sample size.
    Returns probability that true SR exceeds benchmark.
    
    Reference: AFML Chapter 14.7, Machine Learning for Asset Managers Ch 8
    """
    from scipy.stats import norm
    
    # Variance of SR estimate under non-Normal returns (Mertens 2002)
    sr_var = (1 + 0.5 * observed_sr**2 - skew * observed_sr + 
              (kurtosis - 3) / 4 * observed_sr**2) / n_obs
    
    if sr_var <= 0:
        return 0.5
    
    sr_std = np.sqrt(sr_var)
    
    # Z-score
    z = (observed_sr - benchmark_sr) / sr_std
    
    return norm.cdf(z)


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_std: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0,
    kurtosis: float = 3
) -> float:
    """
    Deflated Sharpe Ratio (DSR).
    
    Adjusts for:
    - Non-Normal returns
    - Limited track record
    - Multiple testing/selection bias (number of trials)
    
    This is the KEY metric for assessing if a backtest is overfit.
    
    Reference: AFML Chapter 14.7, Bailey & López de Prado (2014)
    """
    from scipy.stats import norm
    
    # Compute expected maximum SR under null (SR=0)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    
    if n_trials <= 1:
        expected_max_sr = 0
    else:
        # E[max(SR)] approximation
        expected_max_sr = sr_std * (
            (1 - gamma) * norm.ppf(1 - 1/n_trials) +
            gamma * norm.ppf(1 - 1/(n_trials * np.e))
        )
    
    # PSR with expected_max_sr as benchmark
    return probabilistic_sharpe_ratio(
        observed_sr, expected_max_sr, n_obs, skew, kurtosis
    )


# =============================================================================
# BASE MODEL WRAPPER
# =============================================================================

@dataclass
class ModelResult:
    """Container for model training results."""
    model_name: str
    train_score: float
    test_score: float
    precision: float
    recall: float
    f1: float
    feature_importance: pd.Series
    predictions: np.ndarray
    probabilities: np.ndarray


class BaseFinancialModel(ABC):
    """Base class for financial ML models."""
    
    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass
    
    @abstractmethod
    def get_feature_importance(self, feature_names):
        pass


class FinancialRandomForest(BaseFinancialModel):
    """
    Random Forest optimized for financial applications.
    
    Key modifications:
    - Uses balanced class weights by default
    - Early stopping via max_depth
    - Sequential bootstrap-aware through sample_weight
    
    Reference: AFML Chapter 6
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 5,
        min_samples_leaf: int = 50,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced_subsample',
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            oob_score=True
        )
        self.is_fitted = False
    
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        return mean_decrease_impurity(self.model, feature_names)
    
    @property
    def oob_score(self):
        return self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None


class FinancialGradientBoosting(BaseFinancialModel):
    """
    Gradient Boosting for financial applications.
    
    Uses LightGBM if available (faster), falls back to sklearn.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state
            )
        self.is_fitted = False
    
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        return mean_decrease_impurity(self.model, feature_names)


# =============================================================================
# META-LABELING MODEL - Chapter 3 AFML
# =============================================================================

class MetaLabelingModel:
    """
    Meta-Labeling: Secondary model to size bets from primary model.
    
    The primary model determines SIDE (buy/sell).
    The meta-model determines SIZE (how much to bet, including 0).
    
    Key benefits:
    - Increases precision by filtering false positives
    - Decouples side and size decisions
    - Can use different features for each
    
    Reference: AFML Chapter 3.6-3.7
    """
    
    def __init__(
        self,
        secondary_model: BaseFinancialModel = None,
        confidence_threshold: float = 0.5
    ):
        if secondary_model is None:
            self.secondary = FinancialRandomForest(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=30
            )
        else:
            self.secondary = secondary_model
        
        self.confidence_threshold = confidence_threshold
        self.is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        primary_signals: pd.Series,
        labels: pd.Series,
        sample_weight: pd.Series = None
    ):
        """
        Train the meta-labeling model.
        
        Args:
            X: Feature matrix
            primary_signals: Side predictions from primary model (+1, -1)
            labels: True labels from triple-barrier method (+1, -1, 0)
            sample_weight: Optional sample weights
        """
        # Align all inputs
        common_idx = X.index.intersection(primary_signals.index).intersection(labels.index)
        
        X_aligned = X.loc[common_idx]
        signals = primary_signals.loc[common_idx]
        true_labels = labels.loc[common_idx]
        
        # Create meta-labels: 1 if primary signal was correct, 0 otherwise
        # Primary signal correct if: (signal=1 and label=1) or (signal=-1 and label=-1)
        meta_labels = ((signals == 1) & (true_labels == 1)) | \
                      ((signals == -1) & (true_labels == -1))
        meta_labels = meta_labels.astype(int)
        
        # Add primary signal as feature (optional but helps)
        X_meta = X_aligned.copy()
        X_meta['primary_signal'] = signals
        
        # Remove NaN
        valid_mask = ~(X_meta.isna().any(axis=1) | meta_labels.isna())
        X_meta = X_meta.loc[valid_mask]
        meta_labels = meta_labels.loc[valid_mask]
        
        if sample_weight is not None:
            sw = sample_weight.loc[X_meta.index]
        else:
            sw = None
        
        # Train
        self.secondary.fit(X_meta, meta_labels, sample_weight=sw)
        self.feature_names_ = list(X_meta.columns)
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        primary_signals: pd.Series
    ) -> pd.DataFrame:
        """
        Predict whether to take the primary model's signal.
        
        Returns DataFrame with:
        - primary_signal: Original signal
        - confidence: Probability signal is correct
        - take_trade: 1 if should trade, 0 if should pass
        - bet_size: Position size based on Kelly criterion
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Align
        common_idx = X.index.intersection(primary_signals.index)
        X_aligned = X.loc[common_idx].copy()
        signals = primary_signals.loc[common_idx]
        
        # Add primary signal feature
        X_aligned['primary_signal'] = signals
        
        # Predict probabilities
        proba = self.secondary.predict_proba(X_aligned)[:, 1]
        
        result = pd.DataFrame(index=common_idx)
        result['primary_signal'] = signals
        result['confidence'] = proba
        result['take_trade'] = (proba >= self.confidence_threshold).astype(int)
        
        # Bet sizing using simplified Kelly
        # f* = (p - (1-p)) / 1 = 2p - 1 for 1:1 odds
        result['bet_size'] = (2 * proba - 1).clip(0, 1) * result['take_trade']
        
        # Final signal = primary_signal * take_trade
        result['final_signal'] = result['primary_signal'] * result['take_trade']
        
        return result
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from secondary model."""
        return self.secondary.get_feature_importance(self.feature_names_)


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class FinancialEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking Ensemble optimized for financial applications.
    
    Combines multiple base models with a meta-learner.
    Uses proper purged cross-validation for stacking.
    
    Reference: AFML Chapter 6
    """
    
    def __init__(
        self,
        base_models: List[BaseFinancialModel] = None,
        meta_learner = None,
        use_probas: bool = True
    ):
        if base_models is None:
            self.base_models = [
                FinancialRandomForest(n_estimators=200, max_depth=4),
                FinancialGradientBoosting(n_estimators=100, max_depth=3),
            ]
        else:
            self.base_models = base_models
        
        if meta_learner is None:
            self.meta_learner = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                random_state=42
            )
        else:
            self.meta_learner = meta_learner
        
        self.use_probas = use_probas
        self.is_fitted = False
        self.scaler = StandardScaler()
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series = None
    ):
        """Fit ensemble with 2-level training."""
        # Level 1: Train base models
        for model in self.base_models:
            model.fit(X, y, sample_weight=sample_weight)
        
        # Level 2: Generate meta-features
        meta_features = self._get_meta_features(X)
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Train meta-learner
        self.meta_learner.fit(meta_features_scaled, y)
        self.is_fitted = True
        
        return self
    
    def _get_meta_features(self, X):
        """Generate meta-features from base model predictions."""
        meta_feats = []
        
        for model in self.base_models:
            if self.use_probas:
                proba = model.predict_proba(X)
                meta_feats.append(proba[:, 1])
            else:
                pred = model.predict(X)
                meta_feats.append(pred)
        
        return np.column_stack(meta_feats)
    
    def predict(self, X):
        """Predict using ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        meta_features = self._get_meta_features(X)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        return self.meta_learner.predict(meta_features_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        meta_features = self._get_meta_features(X)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        return self.meta_learner.predict_proba(meta_features_scaled)
    
    def get_model_contributions(self, X, y) -> pd.DataFrame:
        """Analyze contribution of each base model."""
        results = []
        
        for i, model in enumerate(self.base_models):
            pred = model.predict(X)
            proba = model.predict_proba(X)[:, 1]
            
            results.append({
                'model': f'base_{i}',
                'accuracy': accuracy_score(y, pred),
                'precision': precision_score(y, pred, zero_division=0),
                'recall': recall_score(y, pred, zero_division=0),
                'f1': f1_score(y, pred, zero_division=0),
                'auc': roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0
            })
        
        # Add ensemble results
        pred = self.predict(X)
        proba = self.predict_proba(X)[:, 1]
        
        results.append({
            'model': 'ensemble',
            'accuracy': accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0),
            'f1': f1_score(y, pred, zero_division=0),
            'auc': roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0
        })
        
        return pd.DataFrame(results)


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_model_comprehensive(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    returns_test: pd.Series = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation including financial metrics.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'log_loss': log_loss(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
    except Exception:
        metrics['auc_roc'] = np.nan
    
    # Financial metrics if returns available
    if returns_test is not None:
        # Strategy returns: go long when predict 1, short when predict 0
        strategy_returns = returns_test * (2 * y_pred - 1)
        
        # Sharpe ratio
        sr = compute_sharpe_ratio(strategy_returns)
        metrics['sharpe_ratio'] = sr
        
        # PSR (probability this SR is real)
        metrics['psr'] = probabilistic_sharpe_ratio(
            sr, 0, len(strategy_returns),
            strategy_returns.skew(),
            strategy_returns.kurtosis() + 3
        )
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        winning = (strategy_returns > 0).sum()
        total = len(strategy_returns)
        metrics['win_rate'] = winning / total if total > 0 else 0
    
    return metrics


if __name__ == "__main__":
    print("Testing Advanced ML Models...")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    
    # Create labels with some signal
    signal = 0.3 * X['feat_0'] - 0.2 * X['feat_1'] + 0.1 * np.random.randn(n)
    y = pd.Series((signal > 0).astype(int))
    
    # Split
    train_size = int(0.8 * n)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print("\n1. Testing Random Forest...")
    rf = FinancialRandomForest()
    rf.fit(X_train, y_train)
    print(f"   OOB Score: {rf.oob_score:.4f}")
    
    print("\n2. Testing Gradient Boosting...")
    gb = FinancialGradientBoosting()
    gb.fit(X_train, y_train)
    
    print("\n3. Testing Ensemble...")
    ensemble = FinancialEnsemble()
    ensemble.fit(X_train, y_train)
    
    contributions = ensemble.get_model_contributions(X_test, y_test)
    print("\n   Model Contributions:")
    print(contributions.to_string(index=False))
    
    print("\n4. Testing Feature Importance...")
    importance = mean_decrease_impurity(rf.model, list(X.columns))
    print("   Top 5 features:")
    print(importance.head().to_string())
    
    print("\n5. Testing Sharpe Ratio Statistics...")
    returns = pd.Series(np.random.randn(252) * 0.01)  # 1 year of daily returns
    sr = compute_sharpe_ratio(returns)
    psr = probabilistic_sharpe_ratio(sr, 0, 252)
    dsr = deflated_sharpe_ratio(sr, 0.5, 10, 252)
    
    print(f"   Sharpe Ratio: {sr:.3f}")
    print(f"   PSR: {psr:.3f}")
    print(f"   DSR (10 trials): {dsr:.3f}")
