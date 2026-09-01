from .core import fetch_market_data
from .stats import (
    analyze_series_structure,
    analyze_fat_tails,
    compute_all_var,
    var_parametric,
    var_cornish_fisher,
    var_historical,
    var_bootstrap,
    fit_garch,
    garch_var,
    get_hurst_exponent,
    get_adf_stat,
    get_volatility,
    frac_diff,
)

__all__ = [
    'fetch_market_data',
    'analyze_series_structure',
    'analyze_fat_tails',
    'compute_all_var',
    'var_parametric', 'var_cornish_fisher', 'var_historical', 'var_bootstrap',
    'fit_garch', 'garch_var',
    'get_hurst_exponent', 'get_adf_stat', 'get_volatility',
    'frac_diff',
]

# The RAG knowledge base depends on chromadb and the LangChain/Google stack,
# all optional extras. knowledge.py imports cleanly without them, but keep the
# re-export defensive so the core path never depends on the extras resolving.
try:
    from .knowledge import KnowledgeBase

    __all__ += ['KnowledgeBase']
except ImportError:  # pragma: no cover - exercised only without the extras
    pass
