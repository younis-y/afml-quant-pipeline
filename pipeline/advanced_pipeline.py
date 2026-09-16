"""
AFML Quant Pipeline - Advanced ML Pipeline
Complete quantitative trading ML system implementing:
- Advanced feature engineering (fractional differentiation, microstructure)
- Proper financial labeling (triple barrier, meta-labeling)
- Non-IID handling (sample weights, sequential bootstrap)
- Robust validation (purged K-fold CV, CPCV)
- Comprehensive backtesting statistics (DSR, PSR)

Reference: Advances in Financial Machine Learning (López de Prado)
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import joblib

# Local imports
from .feature_engineering import (
    AdvancedFeatureEngine
)
from .labeling_advanced import (
    AdvancedLabeler
)
from .sample_weights import (
    get_average_uniqueness,
    get_combined_sample_weights,
    PurgedKFold,
    CombinatorialPurgedKFold,
    cv_score_with_purging
)
from .models import (
    FinancialRandomForest,
    FinancialGradientBoosting,
    FinancialEnsemble,
    MetaLabelingModel,
    mean_decrease_impurity,
    deflated_sharpe_ratio,
    evaluate_model_comprehensive
)


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""
    
    # Feature Engineering
    frac_d: Optional[float] = None  # Auto-detect if None
    use_orthogonal_features: bool = True
    include_microstructure: bool = True
    include_entropy: bool = True
    
    # Labeling
    labeling_method: str = 'triple_barrier'
    profit_mult: float = 2.0
    stop_mult: float = 1.0
    num_bars: int = 50
    volatility_span: int = 20
    min_ret: float = 0.001
    
    # Sample Weights
    use_sample_weights: bool = True
    time_decay: float = 1.0
    
    # Validation
    n_cv_splits: int = 5
    pct_embargo: float = 0.01
    use_cpcv: bool = False
    cpcv_n_groups: int = 10
    cpcv_k_test: int = 2
    
    # Model
    model_type: str = 'ensemble'  # 'rf', 'gbm', 'ensemble'
    use_meta_labeling: bool = True
    confidence_threshold: float = 0.6
    
    # Random state
    random_state: int = 42


@dataclass
class PipelineResult:
    """Results from pipeline training."""
    
    # Model info
    model_type: str
    train_samples: int
    test_samples: int
    n_features: int
    
    # Metrics
    cv_scores: Dict[str, Dict[str, float]]
    test_metrics: Dict[str, float]
    
    # Feature importance
    feature_importance: pd.Series
    
    # Financial metrics
    sharpe_ratio: float = 0.0
    psr: float = 0.0
    dsr: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # Label statistics
    label_stats: Dict = field(default_factory=dict)
    
    # Metadata
    training_time: float = 0.0
    config: PipelineConfig = None


class AdvancedMLPipeline:
    """
    Complete ML pipeline for quantitative trading.
    
    Implements all techniques from Advances in Financial Machine Learning:
    1. Dollar bars / Information-driven bars
    2. Fractional differentiation for stationarity
    3. Triple barrier labeling
    4. Meta-labeling for bet sizing
    5. Sample weights for non-IID data
    6. Purged cross-validation
    7. Feature importance analysis
    8. Deflated Sharpe ratio for backtest validation
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Components (initialized in fit)
        self.feature_engine: Optional[AdvancedFeatureEngine] = None
        self.labeler: Optional[AdvancedLabeler] = None
        self.model = None
        self.meta_model: Optional[MetaLabelingModel] = None
        
        # Fitted data
        self.features_: Optional[pd.DataFrame] = None
        self.labels_: Optional[pd.DataFrame] = None
        self.sample_weights_: Optional[pd.Series] = None
        self.t1_: Optional[pd.Series] = None
        
        # Results
        self.result_: Optional[PipelineResult] = None
        self.is_fitted: bool = False
    
    def _create_feature_engine(self) -> AdvancedFeatureEngine:
        """Create feature engineering component."""
        return AdvancedFeatureEngine(
            frac_d=self.config.frac_d,
            use_orthogonal=self.config.use_orthogonal_features,
            include_entropy=self.config.include_entropy,
            include_microstructure=self.config.include_microstructure
        )
    
    def _create_labeler(self) -> AdvancedLabeler:
        """Create labeling component."""
        return AdvancedLabeler(
            method=self.config.labeling_method,
            profit_mult=self.config.profit_mult,
            stop_mult=self.config.stop_mult,
            num_bars=self.config.num_bars,
            volatility_span=self.config.volatility_span,
            min_ret=self.config.min_ret
        )
    
    def _create_model(self):
        """Create ML model based on config."""
        if self.config.model_type == 'rf':
            return FinancialRandomForest(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=50,
                random_state=self.config.random_state
            )
        elif self.config.model_type == 'gbm':
            return FinancialGradientBoosting(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=self.config.random_state
            )
        else:  # ensemble
            return FinancialEnsemble()
    
    def fit(
        self,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        primary_signals: Optional[pd.Series] = None
    ) -> 'AdvancedMLPipeline':
        """
        Fit the complete pipeline.
        
        Args:
            close: Close price series
            volume: Volume series (optional, improves microstructure features)
            high: High price series (optional, improves volatility)
            low: Low price series (optional, improves volatility)
            primary_signals: Primary model signals for meta-labeling (optional)
        
        Returns:
            self
        """
        import time
        start_time = time.time()
        
        print("=" * 70)
        print("ADVANCED ML PIPELINE - TRAINING")
        print("=" * 70)
        
        # Validate inputs
        if len(close) < 500:
            raise ValueError(f"Need at least 500 bars, got {len(close)}")
        
        # Step 1: Feature Engineering
        print("\n[1/6] Feature Engineering...")
        self.feature_engine = self._create_feature_engine()
        self.features_ = self.feature_engine.fit_transform(
            close, volume=volume, high=high, low=low
        )
        print(f"      Generated {len(self.features_.columns)} features")
        print(f"      Optimal d: {self.feature_engine.optimal_d:.3f}")
        
        # Step 2: Labeling
        print("\n[2/6] Generating Labels...")
        self.labeler = self._create_labeler()
        self.labels_ = self.labeler.fit_transform(
            close, high=high, low=low
        )
        
        label_stats = self.labeler.get_statistics()
        print(f"      Total labels: {label_stats.get('total_samples', 0)}")
        print(f"      Positive ratio: {label_stats.get('positive_ratio', 0):.3f}")
        
        # Get t1 for sample weights and CV
        self.t1_ = self.labels_['t1']
        
        # Step 3: Align features and labels
        print("\n[3/6] Aligning Data...")
        common_idx = self.features_.index.intersection(self.labels_.index)
        
        X = self.features_.loc[common_idx]
        y = self.labels_.loc[common_idx, 'label'].astype(int)
        t1_aligned = self.t1_.loc[common_idx]
        
        # Remove NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        t1_aligned = t1_aligned.loc[valid_mask]
        
        print(f"      Aligned samples: {len(X)}")
        
        # Step 4: Sample Weights
        print("\n[4/6] Computing Sample Weights...")
        if self.config.use_sample_weights:
            self.sample_weights_ = get_combined_sample_weights(
                close.loc[X.index], t1_aligned, self.config.time_decay
            )
            avg_uniqueness = get_average_uniqueness(t1_aligned, close.loc[X.index])
            print(f"      Mean uniqueness: {avg_uniqueness.mean():.3f}")
        else:
            self.sample_weights_ = pd.Series(1.0, index=X.index)
        
        # Step 5: Train/Test Split (temporal)
        print("\n[5/6] Training Model...")
        split_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        sw_train = self.sample_weights_.iloc[:split_idx]
        t1_train = t1_aligned.iloc[:split_idx]
        
        # Create model
        self.model = self._create_model()
        
        # Cross-validation with purging.
        #
        # CombinatorialPurgedKFold is built for inspection only, not for
        # scoring: its _purge() works on the [min, max] span of the test set,
        # so a combination whose test groups are not adjacent purges every
        # training observation lying between them. At the defaults
        # (n_groups=10, k_test_groups=2) that leaves one of the 45 splits with
        # an empty training set and cuts the worst of the rest to about an
        # eighth of the rows outside the test folds. Scoring therefore runs on
        # PurgedKFold. See README Scope.
        if self.config.use_cpcv:
            self.cpcv_ = CombinatorialPurgedKFold(
                n_groups=self.config.cpcv_n_groups,
                k_test_groups=self.config.cpcv_k_test,
                t1=t1_train,
                pct_embargo=self.config.pct_embargo
            )
            print(f"      CPCV built for inspection: {self.cpcv_.get_n_splits()} splits, "
                  f"{self.cpcv_.get_n_paths()} paths (not used for scoring)")

        cv = PurgedKFold(
            n_splits=self.config.n_cv_splits,
            t1=t1_train,
            pct_embargo=self.config.pct_embargo
        )
        print(f"      Scoring with Purged K-Fold: {self.config.n_cv_splits} splits")
        
        # Cross-validate
        try:
            cv_scores = cv_score_with_purging(
                self.model.model if hasattr(self.model, 'model') else self.model,
                X_train, y_train, t1_train,
                sample_weight=sw_train,
                cv=cv
            )
            print(f"      CV F1: {cv_scores['f1']['mean']:.3f} ± {cv_scores['f1']['std']:.3f}")
        except Exception as e:
            print(f"      CV failed: {e}")
            cv_scores = {'f1': {'mean': 0, 'std': 0}}
        
        # Final training
        self.model.fit(X_train, y_train, sample_weight=sw_train)
        
        # Meta-labeling (if enabled and primary signals provided)
        if self.config.use_meta_labeling and primary_signals is not None:
            print("\n      Training meta-labeling model...")
            self.meta_model = MetaLabelingModel(
                confidence_threshold=self.config.confidence_threshold
            )
            self.meta_model.fit(X_train, primary_signals.loc[X_train.index], y_train, sw_train)
        
        # Step 6: Evaluation
        print("\n[6/6] Evaluating Model...")
        
        # Get returns for financial metrics
        returns_test = close.pct_change().loc[X_test.index]
        
        test_metrics = evaluate_model_comprehensive(
            self.model, X_test, y_test, returns_test
        )
        
        print(f"      Test Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"      Test F1: {test_metrics['f1']:.3f}")
        
        # Feature importance
        try:
            feature_importance = mean_decrease_impurity(
                self.model.model if hasattr(self.model, 'model') else self.model,
                list(X.columns)
            )
        except Exception:
            feature_importance = pd.Series()
        
        # Financial metrics
        sharpe = test_metrics.get('sharpe_ratio', 0)
        psr = test_metrics.get('psr', 0)
        
        # Compute DSR (assuming some number of trials)
        n_trials = 10  # Placeholder - in practice track actual trials
        dsr = deflated_sharpe_ratio(
            sharpe, 0.5, n_trials, len(y_test),
            returns_test.skew(), returns_test.kurtosis() + 3
        ) if sharpe != 0 else 0
        
        training_time = time.time() - start_time
        
        # Store results
        self.result_ = PipelineResult(
            model_type=self.config.model_type,
            train_samples=len(X_train),
            test_samples=len(X_test),
            n_features=len(X.columns),
            cv_scores=cv_scores,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            sharpe_ratio=sharpe,
            psr=psr,
            dsr=dsr,
            max_drawdown=test_metrics.get('max_drawdown', 0),
            win_rate=test_metrics.get('win_rate', 0),
            label_stats=label_stats,
            training_time=training_time,
            config=self.config
        )
        
        self.is_fitted = True
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Time: {training_time:.1f}s")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"PSR: {psr:.3f}")
        print(f"DSR: {dsr:.3f}")
        print("=" * 70)
        
        return self
    
    def predict(
        self,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        primary_signals: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for new data.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        # Generate features
        features = self.feature_engine.fit_transform(
            close, volume=volume, high=high, low=low
        )
        
        # Drop NaN
        features = features.dropna()
        
        # Predict
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        result = pd.DataFrame(index=features.index)
        result['prediction'] = predictions
        result['probability'] = probabilities
        
        # Apply meta-labeling if available
        if self.meta_model is not None and primary_signals is not None:
            meta_result = self.meta_model.predict(features, primary_signals.loc[features.index])
            result = pd.concat([result, meta_result], axis=1)
        else:
            result['final_signal'] = np.where(
                probabilities >= self.config.confidence_threshold,
                2 * predictions - 1,  # Convert 0/1 to -1/1
                0
            )
            result['action'] = np.where(
                result['final_signal'] == 1, 'BUY',
                np.where(result['final_signal'] == -1, 'SELL', 'PASS')
            )
        
        return result
    
    def get_feature_importance(self, method: str = 'mdi') -> pd.Series:
        """Get feature importance using specified method."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        if method == 'mdi':
            return self.result_.feature_importance
        else:
            return self.result_.feature_importance
    
    def summary(self) -> str:
        """Generate a summary report."""
        if self.result_ is None:
            return "Pipeline not fitted yet."
        
        r = self.result_
        
        lines = [
            "=" * 60,
            "ADVANCED ML PIPELINE - SUMMARY REPORT",
            "=" * 60,
            "",
            "MODEL CONFIGURATION",
            "-" * 40,
            f"  Model Type: {r.model_type}",
            f"  Training Samples: {r.train_samples}",
            f"  Test Samples: {r.test_samples}",
            f"  Features: {r.n_features}",
            f"  Labeling: {self.config.labeling_method}",
            f"  Meta-labeling: {self.config.use_meta_labeling}",
            "",
            "CROSS-VALIDATION SCORES",
            "-" * 40,
        ]
        
        for metric, values in r.cv_scores.items():
            lines.append(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        lines.extend([
            "",
            "TEST METRICS",
            "-" * 40,
        ])
        
        for metric, value in r.test_metrics.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
            else:
                lines.append(f"  {metric}: {value}")
        
        lines.extend([
            "",
            "FINANCIAL METRICS",
            "-" * 40,
            f"  Sharpe Ratio: {r.sharpe_ratio:.4f}",
            f"  PSR (vs 0): {r.psr:.4f}",
            f"  DSR (10 trials): {r.dsr:.4f}",
            f"  Max Drawdown: {r.max_drawdown:.4f}",
            f"  Win Rate: {r.win_rate:.4f}",
            "",
            "TOP 10 FEATURES (MDI)",
            "-" * 40,
        ])
        
        for feat, imp in r.feature_importance.head(10).items():
            lines.append(f"  {feat}: {imp:.4f}")
        
        lines.extend([
            "",
            "LABEL STATISTICS",
            "-" * 40,
        ])
        
        for stat, value in r.label_stats.items():
            if isinstance(value, float):
                lines.append(f"  {stat}: {value:.4f}")
            else:
                lines.append(f"  {stat}: {value}")
        
        lines.extend([
            "",
            f"Training Time: {r.training_time:.1f}s",
            "=" * 60
        ])
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """Save the fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        save_dict = {
            'config': self.config,
            'feature_engine': self.feature_engine,
            'labeler': self.labeler,
            'model': self.model,
            'meta_model': self.meta_model,
            'result': self.result_,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, path)
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AdvancedMLPipeline':
        """Load a fitted pipeline."""
        save_dict = joblib.load(path)
        
        instance = cls(config=save_dict['config'])
        instance.feature_engine = save_dict['feature_engine']
        instance.labeler = save_dict['labeler']
        instance.model = save_dict['model']
        instance.meta_model = save_dict['meta_model']
        instance.result_ = save_dict['result']
        instance.is_fitted = save_dict['is_fitted']
        
        print(f"Pipeline loaded from {path}")
        return instance


# =============================================================================
# QUICK START FUNCTION
# =============================================================================

def create_pipeline(
    model_type: str = 'ensemble',
    use_meta_labeling: bool = True,
    **kwargs
) -> AdvancedMLPipeline:
    """
    Factory function to create a configured pipeline.
    
    Args:
        model_type: 'rf', 'gbm', or 'ensemble'
        use_meta_labeling: Whether to use meta-labeling
        **kwargs: Additional config parameters
    
    Returns:
        Configured AdvancedMLPipeline
    """
    config = PipelineConfig(
        model_type=model_type,
        use_meta_labeling=use_meta_labeling,
        **kwargs
    )
    
    return AdvancedMLPipeline(config)


if __name__ == "__main__":
    print("Testing Advanced ML Pipeline...")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n = 2000
    
    # Simulate realistic price data
    returns = np.random.randn(n) * 0.015 + 0.0002
    prices = 100 * np.exp(np.cumsum(returns))
    volume = np.random.lognormal(12, 0.5, n)
    
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    close = pd.Series(prices, index=dates, name='Close')
    vol = pd.Series(volume, index=dates, name='Volume')
    
    # Create pipeline
    pipeline = create_pipeline(
        model_type='rf',
        use_meta_labeling=False,
        profit_mult=2.0,
        stop_mult=1.0,
        num_bars=40,
        n_cv_splits=5
    )
    
    # Fit
    pipeline.fit(close, volume=vol)
    
    # Print summary
    print(pipeline.summary())
    
    # Predict
    print("\n\nGenerating predictions...")
    predictions = pipeline.predict(close[-200:], volume=vol[-200:])
    print(f"Predictions shape: {predictions.shape}")
    print(predictions.tail())
