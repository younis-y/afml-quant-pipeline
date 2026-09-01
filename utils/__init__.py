"""
AFML Quant Pipeline - Utilities
Validation tools, caching, and the optional alpha-decay sanity check.
"""

from .validation import (
    PurgedKFold,
    purged_kfold_cv,
    deflated_sharpe_ratio,
    calculate_sharpe_ratio,
    validate_strategy,
    format_validation_report,
    ValidationResult
)

__all__ = [
    'PurgedKFold',
    'purged_kfold_cv',
    'deflated_sharpe_ratio',
    'calculate_sharpe_ratio',
    'validate_strategy',
    'format_validation_report',
    'ValidationResult',
]

# The sanity check depends on google-generativeai, an optional extra.
try:
    from .sanity_check import (
        sanity_check_strategy,
        check_multiple_strategies,
        format_sanity_report,
    )

    __all__ += [
        'sanity_check_strategy',
        'check_multiple_strategies',
        'format_sanity_report',
    ]
except ImportError:  # pragma: no cover - exercised only without the extra
    pass
