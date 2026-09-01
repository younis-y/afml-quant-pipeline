"""
Tests for utils/validation.py — gaps not covered by test_validation.py.
Covers: format_validation_report, validate_strategy (integration), DSR edge case.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from utils.validation import (
    ValidationResult,
    format_validation_report,
    validate_strategy,
    deflated_sharpe_ratio,
)


# ── format_validation_report ────────────────────────────────────────────────

class TestFormatValidationReport:
    def test_all_passed(self):
        results = [
            ValidationResult("Test A", True, 0.75, 0.5, "Good performance"),
            ValidationResult("Test B", True, 0.10, 0.2, "Within threshold"),
        ]
        report = format_validation_report(results)
        assert "PASSED" in report
        assert "2/2" in report
        assert "Test A" in report
        assert "Test B" in report

    def test_mixed_results(self):
        results = [
            ValidationResult("CV Score", True, 0.65, 0.5, "OK"),
            ValidationResult("DSR", False, -0.5, 0.0, "Not significant"),
        ]
        report = format_validation_report(results)
        assert "ISSUES" in report
        assert "1/2" in report

    def test_empty_list(self):
        report = format_validation_report([])
        assert "0/0" in report


# ── deflated_sharpe_ratio edge case ─────────────────────────────────────────

class TestDSREdgeCase:
    def test_single_trial(self):
        """With num_trials=1, expected max Sharpe should be 0."""
        result = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            num_trials=1,
            backtest_length=252,
        )
        assert result["expected_max_sharpe"] == 0
        assert result["dsr"] == 1.0  # observed - 0


# ── validate_strategy (integration) ────────────────────────────────────────

class TestValidateStrategy:
    @pytest.mark.integration
    def test_end_to_end(self):
        """Full validate_strategy pipeline with a simple model."""
        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        })
        y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))

        backtest = pd.DataFrame({
            "returns": np.random.randn(n) * 0.01 + 0.0005,
        })

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        results = validate_strategy(backtest, model, X, y, num_trials=5)
        assert isinstance(results, list)
        assert len(results) >= 2
        for r in results:
            assert isinstance(r, ValidationResult)
