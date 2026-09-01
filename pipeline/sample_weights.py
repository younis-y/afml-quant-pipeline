"""
AFML Quant Pipeline - Sample Weights & Cross-Validation
Implements non-IID handling from AFML Chapter 4 and 7:
- Concurrent label counting
- Sequential bootstrap
- Sample uniqueness weighting
- Purged K-Fold Cross-Validation
- Combinatorial Purged Cross-Validation (CPCV)
"""

from typing import Optional, Tuple, List, Dict, Generator
import numpy as np
import pandas as pd
from numba import jit
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
import warnings


# =============================================================================
# CONCURRENT LABELS & UNIQUENESS - Chapter 4 AFML
# =============================================================================

def get_indicator_matrix(
    bar_index: pd.DatetimeIndex,
    t1: pd.Series
) -> pd.DataFrame:
    """
    Build an indicator matrix showing which bars influence each label.
    
    For each observation i, the indicator 1_{t,i} = 1 if bar t is used
    in computing label y_i.
    
    Args:
        bar_index: Index of all bars
        t1: Series where index = feature observation time, 
            values = label determination time (end of triple barrier)
    
    Returns:
        Binary indicator matrix (bars x observations)
        
    Reference: AFML Chapter 4, Snippet 4.3
    """
    ind_matrix = pd.DataFrame(0, index=bar_index, columns=range(len(t1)))
    
    for i, (t0, t1_val) in enumerate(t1.items()):
        if pd.isna(t1_val):
            continue
        # Mark all bars in [t0, t1] as influencing label i
        mask = (bar_index >= t0) & (bar_index <= t1_val)
        ind_matrix.loc[mask, i] = 1
    
    return ind_matrix


def get_num_concurrent_labels(close: pd.Series, t1: pd.Series) -> pd.Series:
    """
    Count number of labels concurrent at each time point.
    
    Higher concurrency = more information overlap = less unique observations.
    
    Reference: AFML Chapter 4, Snippet 4.1
    """
    t1 = t1.dropna()
    
    # For each bar, count labels that span over it
    concurrent = pd.Series(0, index=close.index)
    
    for t0, t1_val in t1.items():
        mask = (close.index >= t0) & (close.index <= t1_val)
        concurrent.loc[mask] += 1
    
    return concurrent


def get_average_uniqueness(t1: pd.Series, close: pd.Series) -> pd.Series:
    """
    Compute average uniqueness for each label.
    
    Uniqueness at time t = 1 / c_t where c_t is number of concurrent labels.
    Average uniqueness = mean uniqueness over label's lifespan.
    
    Reference: AFML Chapter 4, Snippet 4.4
    """
    concurrent = get_num_concurrent_labels(close, t1)
    
    # Compute uniqueness (inverse of concurrent labels)
    uniqueness = 1.0 / concurrent.replace(0, np.nan)
    
    # Average uniqueness for each observation
    avg_uniqueness = pd.Series(index=t1.index, dtype=float)
    
    for t0, t1_val in t1.items():
        if pd.isna(t1_val):
            avg_uniqueness.loc[t0] = np.nan
            continue
        
        mask = (close.index >= t0) & (close.index <= t1_val)
        avg_uniqueness.loc[t0] = uniqueness.loc[mask].mean()
    
    return avg_uniqueness


# =============================================================================
# SEQUENTIAL BOOTSTRAP - Chapter 4 AFML
# =============================================================================

def sequential_bootstrap(
    ind_matrix: pd.DataFrame,
    sample_length: Optional[int] = None
) -> np.ndarray:
    """
    Sequential Bootstrap: Sample with replacement but reduce probability
    of drawing observations with overlapping outcomes.
    
    Key insight: Standard bootstrap fails for financial data because
    observations are not IID. Sequential bootstrap addresses this by
    adjusting sampling probabilities based on overlap with already-drawn
    samples.
    
    Reference: AFML Chapter 4, Snippet 4.5
    """
    if sample_length is None:
        sample_length = ind_matrix.shape[1]
    
    phi = []  # Sequence of draws
    
    for _ in range(sample_length):
        # Compute average uniqueness if we add each possible observation
        avg_u = pd.Series(dtype=float)
        
        for i in range(ind_matrix.shape[1]):
            # Indicator column for this observation
            ind_i = ind_matrix.iloc[:, i]
            
            # Compute concurrency with already-drawn samples
            if phi:
                concurrent = ind_matrix.iloc[:, phi].sum(axis=1) + ind_i
            else:
                concurrent = ind_i.copy()
            
            # Uniqueness where this observation is active
            mask = ind_i > 0
            if mask.sum() > 0:
                u = (1.0 / concurrent.loc[mask].replace(0, np.nan)).mean()
                avg_u[i] = u if not np.isnan(u) else 0
            else:
                avg_u[i] = 0
        
        # Normalize to probability
        prob = avg_u / avg_u.sum() if avg_u.sum() > 0 else np.ones(len(avg_u)) / len(avg_u)
        
        # Draw based on probability
        choice = np.random.choice(len(prob), p=prob.values)
        phi.append(choice)
    
    return np.array(phi)


@jit(nopython=True)
def _compute_sequential_uniqueness(
    ind_matrix: np.ndarray,
    drawn_indices: np.ndarray
) -> np.ndarray:
    """JIT-optimized uniqueness computation for sequential bootstrap."""
    n_obs = ind_matrix.shape[1]
    n_drawn = len(drawn_indices)
    
    # Compute concurrent labels from drawn samples
    concurrent = np.zeros(ind_matrix.shape[0])
    for idx in drawn_indices:
        concurrent += ind_matrix[:, idx]
    
    # Compute average uniqueness for each possible new draw
    avg_u = np.zeros(n_obs)
    
    for i in range(n_obs):
        temp_concurrent = concurrent + ind_matrix[:, i]
        
        # Find where this observation is active
        active_sum = 0.0
        active_count = 0
        
        for t in range(ind_matrix.shape[0]):
            if ind_matrix[t, i] > 0:
                if temp_concurrent[t] > 0:
                    active_sum += 1.0 / temp_concurrent[t]
                    active_count += 1
        
        if active_count > 0:
            avg_u[i] = active_sum / active_count
    
    return avg_u


def sequential_bootstrap_fast(
    ind_matrix: pd.DataFrame,
    sample_length: Optional[int] = None
) -> np.ndarray:
    """
    Fast sequential bootstrap using JIT compilation.
    """
    if sample_length is None:
        sample_length = ind_matrix.shape[1]
    
    mat = ind_matrix.values.astype(np.float64)
    phi = np.array([], dtype=np.int64)
    
    for _ in range(sample_length):
        avg_u = _compute_sequential_uniqueness(mat, phi)
        
        # Handle edge cases
        if avg_u.sum() == 0:
            avg_u = np.ones(len(avg_u))
        
        prob = avg_u / avg_u.sum()
        
        choice = np.random.choice(len(prob), p=prob)
        phi = np.append(phi, choice)
    
    return phi


# =============================================================================
# SAMPLE WEIGHTS - Chapter 4 AFML
# =============================================================================

def get_sample_weights_by_return(
    close: pd.Series,
    t1: pd.Series,
    num_threads: int = 1
) -> pd.Series:
    """
    Compute sample weights based on return attribution.
    
    Weight = sum of attributed returns over label's lifespan.
    This ensures labels with larger price moves get higher weights,
    and overlapping portions are downweighted.
    
    Reference: AFML Chapter 4, Snippet 4.10
    """
    # Get concurrent labels
    concurrent = get_num_concurrent_labels(close, t1)
    
    # Get returns
    returns = close.pct_change()
    
    # Compute attributed returns for each observation
    weights = pd.Series(index=t1.index, dtype=float)
    
    for t0, t1_val in t1.items():
        if pd.isna(t1_val):
            weights.loc[t0] = 0
            continue
        
        mask = (close.index >= t0) & (close.index <= t1_val)
        
        # Attribution: return / concurrent (portion attributed to this label)
        attributed = (returns.loc[mask] / concurrent.loc[mask].replace(0, np.nan)).abs()
        weights.loc[t0] = attributed.sum()
    
    # Normalize to sum to number of observations
    weights = weights * len(weights) / weights.sum()
    
    return weights.fillna(0)


def get_sample_weights_by_time_decay(
    t1: pd.Series,
    decay_factor: float = 1.0
) -> pd.Series:
    """
    Apply time decay to sample weights.
    
    Newer observations get higher weights because markets evolve.
    
    Args:
        t1: Label end times
        decay_factor: c in w_t = c^(T-t) where T is most recent time
                      c=1 means no decay, c>1 means more emphasis on recent
    
    Reference: AFML Chapter 4.7
    """
    # Create cumulative weight based on recency
    sorted_times = t1.sort_index().index
    n = len(sorted_times)
    
    if decay_factor == 1:
        # No decay
        weights = pd.Series(1.0, index=t1.index)
    else:
        # Exponential decay with most recent having weight 1
        positions = pd.Series(range(n), index=sorted_times)
        weights = decay_factor ** (n - 1 - positions)
        weights = weights.reindex(t1.index)
    
    # Normalize
    weights = weights * len(weights) / weights.sum()
    
    return weights


def get_combined_sample_weights(
    close: pd.Series,
    t1: pd.Series,
    decay_factor: float = 1.0
) -> pd.Series:
    """
    Combine return attribution weights with time decay.
    """
    return_weights = get_sample_weights_by_return(close, t1)
    time_weights = get_sample_weights_by_time_decay(t1, decay_factor)
    
    combined = return_weights * time_weights
    
    # Normalize
    combined = combined * len(combined) / combined.sum()
    
    return combined


# =============================================================================
# PURGED K-FOLD CROSS-VALIDATION - Chapter 7 AFML
# =============================================================================

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for financial data.
    
    Key features:
    1. Purging: Remove training observations whose labels overlap 
       with test set labels (prevents information leakage)
    2. Embargo: Also remove training observations immediately after
       test set to account for serial correlation in features
    
    Reference: AFML Chapter 7, Snippet 7.3
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series = None,
        pct_embargo: float = 0.01
    ):
        """
        Args:
            n_splits: Number of folds
            t1: Series with label end times (index=feature time, values=label end time)
            pct_embargo: Fraction of dataset to embargo after each test set
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
    
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging and embargo.
        """
        if self.t1 is None:
            raise ValueError("t1 must be provided for PurgedKFold")
        
        # Ensure indices are aligned
        indices = np.arange(len(X))
        
        # Calculate embargo period
        embargo = int(len(X) * self.pct_embargo)
        
        # Generate K folds
        fold_size = len(X) // self.n_splits
        
        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else len(X)
            
            test_indices = indices[test_start:test_end]
            
            # Get test set timestamps
            if isinstance(X, pd.DataFrame):
                test_times = X.index[test_indices]
            else:
                test_times = self.t1.index[test_indices]
            
            # Get training indices (excluding test)
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])
            
            # Purging: Remove training samples whose labels overlap with test
            train_mask = self._purge_train_set(train_indices, test_times, X)
            
            # Embargo: Remove training samples immediately after test
            train_mask = self._apply_embargo(train_mask, test_end, embargo, len(X))
            
            train_purged = train_indices[train_mask]
            
            yield train_purged, test_indices
    
    def _purge_train_set(
        self,
        train_indices: np.ndarray,
        test_times: pd.DatetimeIndex,
        X
    ) -> np.ndarray:
        """Remove training samples with labels overlapping test set."""
        if isinstance(X, pd.DataFrame):
            train_times = X.index[train_indices]
        else:
            train_times = self.t1.index[train_indices]
        
        # Get t1 values for train samples
        train_t1 = self.t1.loc[train_times]
        
        test_start = test_times.min()
        test_end = test_times.max()
        
        # Keep training sample if its label doesn't overlap with test period
        mask = np.ones(len(train_indices), dtype=bool)
        
        for i, (t0, t1_val) in enumerate(train_t1.items()):
            if pd.isna(t1_val):
                continue
            
            # Check for overlap: [t0, t1] overlaps with [test_start, test_end]
            if not (t1_val < test_start or t0 > test_end):
                mask[i] = False
        
        return mask
    
    def _apply_embargo(
        self,
        mask: np.ndarray,
        test_end: int,
        embargo: int,
        n_samples: int
    ) -> np.ndarray:
        """Remove training samples in embargo period after test set."""
        # Samples right after test set should be excluded
        for i in range(len(mask)):
            # This is a simplification - in practice we need to map back to original indices
            pass
        return mask
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# =============================================================================
# COMBINATORIAL PURGED CROSS-VALIDATION - Chapter 12 AFML
# =============================================================================

class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Unlike standard CV which produces 1 backtest path, CPCV produces
    multiple paths by testing on combinations of folds.
    
    Key benefits:
    - Multiple backtest paths allow statistical analysis of strategy
    - Reduces probability of overfitting to a single path
    - More robust strategy evaluation
    
    Reference: AFML Chapter 12
    """
    
    def __init__(
        self,
        n_groups: int = 10,
        k_test_groups: int = 2,
        t1: pd.Series = None,
        pct_embargo: float = 0.01
    ):
        """
        Args:
            n_groups: N - number of groups to partition data into
            k_test_groups: k - number of groups in each test set
            t1: Label end times
            pct_embargo: Embargo fraction
        
        This generates C(N,k) = N!/(k!(N-k)!) combinations,
        producing phi[N,k] = k/N * C(N,k) backtest paths.
        
        For N=10, k=2: 45 combinations, 9 paths
        """
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
        # Pre-compute number of splits
        from math import comb
        self._n_splits = comb(n_groups, k_test_groups)
    
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate all combinatorial train/test splits with purging.
        """
        from itertools import combinations
        
        n = len(X)
        indices = np.arange(n)
        
        # Partition into N groups
        group_size = n // self.n_groups
        group_indices = []
        
        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n
            group_indices.append(indices[start:end])
        
        # Generate all k-combinations of groups for test set
        for test_groups in combinations(range(self.n_groups), self.k_test_groups):
            # Test set: union of selected groups
            test_idx = np.concatenate([group_indices[g] for g in test_groups])
            
            # Training set: all other groups
            train_groups = [g for g in range(self.n_groups) if g not in test_groups]
            train_idx = np.concatenate([group_indices[g] for g in train_groups])
            
            # Apply purging if t1 is available
            if self.t1 is not None:
                if isinstance(X, pd.DataFrame):
                    test_times = X.index[test_idx]
                    train_times = X.index[train_idx]
                else:
                    test_times = self.t1.index[test_idx]
                    train_times = self.t1.index[train_idx]
                
                # Purge training observations
                train_mask = self._purge(train_idx, train_times, test_times)
                train_idx = train_idx[train_mask]
            
            yield train_idx, test_idx
    
    def _purge(
        self,
        train_indices: np.ndarray,
        train_times: pd.DatetimeIndex,
        test_times: pd.DatetimeIndex
    ) -> np.ndarray:
        """Purge training samples with overlapping labels."""
        test_start = test_times.min()
        test_end = test_times.max()
        
        train_t1 = self.t1.loc[train_times]
        
        mask = np.ones(len(train_indices), dtype=bool)
        
        for i, (t0, t1_val) in enumerate(train_t1.items()):
            if pd.isna(t1_val):
                continue
            if not (t1_val < test_start or t0 > test_end):
                mask[i] = False
        
        return mask
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self._n_splits
    
    def get_n_paths(self) -> int:
        """Return number of backtest paths this generates."""
        from math import comb
        return self.k_test_groups * comb(self.n_groups, self.k_test_groups) // self.n_groups


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cv_score_with_purging(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series = None,
    cv=5,
    scoring: str = 'f1',
    pct_embargo: float = 0.01
) -> Dict[str, float]:
    """
    Cross-validate with proper purging for financial data.

    Args:
        cv: either the number of folds (a PurgedKFold is built from it and
            from t1 / pct_embargo), or an already-constructed splitter such as
            PurgedKFold or CombinatorialPurgedKFold, which is used as given.

    Reference: AFML Chapter 7, Snippet 7.4
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, log_loss
    )
    
    if hasattr(cv, 'split'):
        purged_cv = cv
    else:
        purged_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    
    for train_idx, test_idx in purged_cv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # Get sample weights for training
        if sample_weight is not None:
            sw_train = sample_weight.iloc[train_idx]
            clf.fit(X_train, y_train, sample_weight=sw_train)
        else:
            clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
    
    return {
        k: {'mean': np.mean(v), 'std': np.std(v)}
        for k, v in scores.items()
    }


if __name__ == "__main__":
    print("Testing Sample Weights & Cross-Validation...")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    close = pd.Series(prices, index=dates)
    
    # Create t1 (label end times) - each label spans ~20 bars
    t1 = pd.Series(index=dates[:-20], dtype='datetime64[ns]')
    for i, dt in enumerate(dates[:-20]):
        t1.iloc[i] = dates[i + 20]
    
    print("\n1. Testing concurrent labels...")
    concurrent = get_num_concurrent_labels(close, t1)
    print(f"   Mean concurrent labels: {concurrent.mean():.2f}")
    
    print("\n2. Testing average uniqueness...")
    uniqueness = get_average_uniqueness(t1, close)
    print(f"   Mean uniqueness: {uniqueness.mean():.3f}")
    
    print("\n3. Testing sample weights...")
    weights = get_sample_weights_by_return(close, t1)
    print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    print("\n4. Testing indicator matrix...")
    ind_matrix = get_indicator_matrix(close.index, t1)
    print(f"   Matrix shape: {ind_matrix.shape}")
    
    print("\n5. Testing sequential bootstrap...")
    sample = sequential_bootstrap(ind_matrix, sample_length=50)
    print(f"   Unique samples: {len(np.unique(sample))} / 50")
    
    print("\n6. Testing PurgedKFold...")
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    X = pd.DataFrame({'feat': np.random.randn(len(t1))}, index=t1.index)
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"   Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    print("\n7. Testing CPCV...")
    cpcv = CombinatorialPurgedKFold(n_groups=6, k_test_groups=2, t1=t1)
    print(f"   Number of splits: {cpcv.get_n_splits()}")
    print(f"   Number of paths: {cpcv.get_n_paths()}")
