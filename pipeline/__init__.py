"""
AFML Quant Pipeline - Quantitative Pipeline
Data processing, labeling, and meta-modeling for trading
"""

from .data_processor import (
    fetch_data,
    dollar_bars,
    frac_diff_fixed,
    get_weights_ffd,
    find_min_d_for_stationarity,
    compute_correlation_with_original,
    DataProcessor
)

from .labeling import (
    triple_barrier_labels,
    get_volatility,
    get_horizontal_barriers,
    get_vertical_barriers,
    get_meta_labels,
    compute_label_statistics,
    TripleBarrierLabeler
)

from .meta_model import (
    SMACrossover,
    MetaModel,
    MetaLabelingPipeline,
    PrimarySignal
)

# Live Engine (may fail if agents/rag are not configured)
try:
    from .live_engine import (
        LiveEngine,
        TradeSignal,
        EngineState,
        run_paper_trading
    )
except (ImportError, OSError):
    LiveEngine = None
    TradeSignal = None
    EngineState = None
    run_paper_trading = None

# Advanced ML Pipeline (from ImprovementPackage)
from .advanced_pipeline import (
    AdvancedMLPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline
)

from .feature_engineering import (
    AdvancedFeatureEngine,
    frac_diff_ffd,
    find_optimal_d,
    get_roll_spread,
    get_kyle_lambda,
    get_amihud_lambda,
    get_vpin,
    get_lempel_ziv_entropy,
    get_shannon_entropy,
    get_cusum_filter
)

from .labeling_advanced import (
    AdvancedLabeler,
    triple_barrier_labels as advanced_triple_barrier_labels,
    get_meta_labels as advanced_get_meta_labels,
    get_daily_volatility,
    cusum_filter,
    trend_scanning_labels
)

from .sample_weights import (
    get_average_uniqueness,
    get_combined_sample_weights,
    sequential_bootstrap,
    PurgedKFold,
    CombinatorialPurgedKFold,
    cv_score_with_purging
)

from .models import (
    FinancialRandomForest,
    FinancialGradientBoosting,
    FinancialEnsemble,
    MetaLabelingModel,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    compute_sharpe_ratio
)

__all__ = [
    # Data Processing
    'fetch_data',
    'dollar_bars',
    'frac_diff_fixed',
    'get_weights_ffd',
    'find_min_d_for_stationarity',
    'compute_correlation_with_original',
    'DataProcessor',
    # Labeling
    'triple_barrier_labels',
    'get_volatility',
    'get_horizontal_barriers',
    'get_vertical_barriers',
    'get_meta_labels',
    'compute_label_statistics',
    'TripleBarrierLabeler',
    # Meta Model
    'SMACrossover',
    'MetaModel',
    'MetaLabelingPipeline',
    'PrimarySignal',
    # Live Engine
    'LiveEngine',
    'TradeSignal',
    'EngineState',
    'run_paper_trading',
    # Advanced Pipeline
    'AdvancedMLPipeline',
    'PipelineConfig',
    'PipelineResult',
    'create_pipeline',
    # Advanced Feature Engineering
    'AdvancedFeatureEngine',
    'frac_diff_ffd',
    'find_optimal_d',
    # Advanced Labeling
    'AdvancedLabeler',
    'cusum_filter',
    'trend_scanning_labels',
    # Sample Weights & CV
    'get_average_uniqueness',
    'sequential_bootstrap',
    'PurgedKFold',
    'CombinatorialPurgedKFold',
    # Advanced Models
    'FinancialRandomForest',
    'FinancialEnsemble',
    'MetaLabelingModel',
    'deflated_sharpe_ratio',
    'probabilistic_sharpe_ratio',
]

