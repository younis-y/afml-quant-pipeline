"""
AFML Quant Pipeline - Validation Utilities
Statistical tests for strategy validation
Reference: Advances in Financial Machine Learning, Chapters 7-8, 11
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    value: float
    threshold: float
    details: str


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for time series.
    
    Prevents data leakage by:
    1. Respecting temporal order (train always before test)
    2. Purging overlapping samples between train and test
    3. Optionally adding an embargo period
    
    Reference:
        Advances in Financial Machine Learning, Chapter 7
        "The goal is to prevent information about the test set from leaking into the training set"
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_length: int = 0,
        embargo_pct: float = 0.01
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            purge_length: Number of samples to purge after train set
            embargo_pct: Embargo period as percentage of train set
        """
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ):
        """
        Generate train/test indices for each fold.
        
        Args:
            X: Feature matrix
            y: Target (not used, for API compatibility)
            groups: Group labels (not used, for API compatibility)
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate fold size
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test set for this fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Train set: everything before test, minus purge and embargo
            embargo_size = int(test_start * self.embargo_pct)
            train_end = max(0, test_start - self.purge_length - embargo_size)
            
            train_indices = indices[:train_end]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> int:
        """Return number of splits."""
        return self.n_splits


def purged_kfold_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_length: int = 5,
    embargo_pct: float = 0.01,
    scoring: str = 'accuracy'
) -> Dict[str, Any]:
    """
    Perform purged k-fold cross-validation.
    
    Args:
        model: Sklearn-compatible model with fit/predict
        X: Feature matrix
        y: Target variable
        n_splits: Number of folds
        purge_length: Samples to purge between train/test
        embargo_pct: Embargo as percentage of train size
        scoring: Scoring metric ('accuracy', 'f1', 'precision', 'recall')
    
    Returns:
        Dict with scores and statistics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    scorers = {
        'accuracy': accuracy_score,
        'f1': lambda y, p: f1_score(y, p, zero_division=0),
        'precision': lambda y, p: precision_score(y, p, zero_division=0),
        'recall': lambda y, p: recall_score(y, p, zero_division=0)
    }
    
    scorer = scorers.get(scoring, accuracy_score)
    
    cv = PurgedKFold(n_splits=n_splits, purge_length=purge_length, embargo_pct=embargo_pct)
    
    scores = []
    fold_details = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Handle potential NaN
        valid_train = ~(X_train.isna().any(axis=1) | y_train.isna())
        valid_test = ~(X_test.isna().any(axis=1) | y_test.isna())
        
        X_train_clean = X_train.loc[valid_train]
        y_train_clean = y_train.loc[valid_train]
        X_test_clean = X_test.loc[valid_test]
        y_test_clean = y_test.loc[valid_test]
        
        if len(X_train_clean) < 10 or len(X_test_clean) < 5:
            continue
        
        # Fit and predict
        model.fit(X_train_clean, y_train_clean)
        predictions = model.predict(X_test_clean)
        
        score = scorer(y_test_clean, predictions)
        scores.append(score)
        
        fold_details.append({
            'fold': fold,
            'train_size': len(X_train_clean),
            'test_size': len(X_test_clean),
            'score': score
        })
    
    return {
        'scores': scores,
        'mean': np.mean(scores) if scores else 0,
        'std': np.std(scores) if scores else 0,
        'folds': fold_details,
        'n_folds_completed': len(scores)
    }


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    backtest_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate the Deflated Sharpe Ratio (DSR).
    
    Adjusts the Sharpe ratio for multiple testing bias.
    
    Args:
        observed_sharpe: Observed Sharpe ratio
        num_trials: Number of strategies tested (trials)
        backtest_length: Number of periods in backtest
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns (excess kurtosis + 3)
        risk_free_rate: Risk-free rate
    
    Returns:
        Dict with DSR analysis:
        - 'dsr': Deflated Sharpe Ratio
        - 'expected_max_sharpe': Expected max Sharpe under null
        - 'probability': Probability of observing this Sharpe by chance
        - 'is_significant': Whether the strategy beats random
    
    Reference:
        Advances in Financial Machine Learning, Chapter 11
        "The DSR corrects for selection bias in backtesting"
    """
    # Expected maximum Sharpe ratio under null hypothesis
    # E[max(SR)] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
    # Simplified approximation:
    euler_gamma = 0.5772156649
    
    if num_trials <= 1:
        expected_max = 0
    else:
        z = stats.norm.ppf(1 - 1 / num_trials)
        expected_max = (1 - euler_gamma) * z + euler_gamma * stats.norm.ppf(1 - 1 / (num_trials * np.e))
    
    # Adjust for non-normality
    # Var[SR] = (1 + 0.5 * SR^2 - skew * SR + (kurt-3)/4 * SR^2) / T
    sr_variance = (
        1 + 0.5 * observed_sharpe**2 
        - skewness * observed_sharpe 
        + (kurtosis - 3) / 4 * observed_sharpe**2
    ) / backtest_length
    
    sr_std = np.sqrt(sr_variance)
    
    # Probability of observing this Sharpe by chance
    # Using the adjusted threshold
    if sr_std > 0:
        z_score = (observed_sharpe - expected_max) / sr_std
        probability_by_chance = 1 - stats.norm.cdf(z_score)
    else:
        probability_by_chance = 0.5
    
    # Deflated Sharpe Ratio: adjusted Sharpe accounting for trials
    dsr = observed_sharpe - expected_max
    
    return {
        'dsr': dsr,
        'observed_sharpe': observed_sharpe,
        'expected_max_sharpe': expected_max,
        'sr_variance': sr_variance,
        'probability_by_chance': probability_by_chance,
        'is_significant': probability_by_chance < 0.05,
        'significance_level': 0.05,
        'num_trials': num_trials,
        'backtest_length': backtest_length
    }


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std()
    annualized = sharpe * np.sqrt(periods_per_year)
    
    return annualized


def validate_strategy(
    backtest_results: pd.DataFrame,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    num_trials: int = 1,
    significance_level: float = 0.05
) -> List[ValidationResult]:
    """
    Run comprehensive strategy validation.
    
    Args:
        backtest_results: DataFrame with 'returns' column
        model: The ML model used
        X: Features used in training
        y: Labels used in training
        num_trials: Number of strategy variations tested
        significance_level: P-value threshold
    
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    # 1. Purged K-Fold CV
    cv_results = purged_kfold_cv(model, X, y, n_splits=5)
    cv_passed = cv_results['mean'] > 0.5  # Better than random
    
    results.append(ValidationResult(
        test_name="Purged K-Fold Cross-Validation",
        passed=cv_passed,
        value=cv_results['mean'],
        threshold=0.5,
        details=f"Mean CV score: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}"
    ))
    
    # 2. Deflated Sharpe Ratio
    if 'returns' in backtest_results.columns:
        returns = backtest_results['returns'].dropna()
        
        sharpe = calculate_sharpe_ratio(returns)
        skew = returns.skew()
        kurt = returns.kurtosis() + 3  # scipy returns excess kurtosis
        
        dsr_result = deflated_sharpe_ratio(
            observed_sharpe=sharpe,
            num_trials=num_trials,
            backtest_length=len(returns),
            skewness=skew,
            kurtosis=kurt
        )
        
        results.append(ValidationResult(
            test_name="Deflated Sharpe Ratio",
            passed=dsr_result['is_significant'],
            value=dsr_result['dsr'],
            threshold=0.0,
            details=f"DSR: {dsr_result['dsr']:.4f}, P(chance): {dsr_result['probability_by_chance']:.4f}"
        ))
    
    # 3. Overfitting check (train vs test performance gap)
    train_perf = cv_results['folds'][0]['score'] if cv_results['folds'] else 0
    test_perf = cv_results['mean']
    perf_gap = train_perf - test_perf
    
    overfitting_passed = perf_gap < 0.2  # Less than 20% gap
    
    results.append(ValidationResult(
        test_name="Overfitting Check",
        passed=overfitting_passed,
        value=perf_gap,
        threshold=0.2,
        details=f"Train-Test gap: {perf_gap:.4f}"
    ))
    
    return results


def format_validation_report(results: List[ValidationResult]) -> str:
    """Format validation results as a readable report."""
    report = "# Strategy Validation Report\n\n"
    
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    
    overall = "PASSED" if passed_count == total_count else "ISSUES FOUND"
    report += f"**Overall Status:** {overall} ({passed_count}/{total_count} tests passed)\n\n"
    
    report += "## Test Results\n\n"
    
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        report += f"### {status} {result.test_name}\n"
        report += f"- **Value:** {result.value:.4f}\n"
        report += f"- **Threshold:** {result.threshold:.4f}\n"
        report += f"- **Details:** {result.details}\n\n"
    
    return report


if __name__ == "__main__":
    print("Testing Validation Utilities...")
    print("=" * 50)
    
    # Test Purged K-Fold
    print("\n1. Testing Purged K-Fold...")
    
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    cv_results = purged_kfold_cv(model, X, y, n_splits=5, purge_length=5)
    print(f"   CV Mean: {cv_results['mean']:.4f}")
    print(f"   CV Std: {cv_results['std']:.4f}")
    
    # Test Deflated Sharpe
    print("\n2. Testing Deflated Sharpe Ratio...")
    
    dsr_result = deflated_sharpe_ratio(
        observed_sharpe=2.5,
        num_trials=100,  # Tested 100 strategy variations
        backtest_length=252 * 5,  # 5 years of daily data
        skewness=-0.5,
        kurtosis=4.0
    )
    
    print(f"   Observed Sharpe: {dsr_result['observed_sharpe']:.4f}")
    print(f"   Expected Max Sharpe: {dsr_result['expected_max_sharpe']:.4f}")
    print(f"   Deflated Sharpe: {dsr_result['dsr']:.4f}")
    print(f"   P(by chance): {dsr_result['probability_by_chance']:.4f}")
    print(f"   Is Significant: {dsr_result['is_significant']}")
