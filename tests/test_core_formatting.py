"""
Tests for analysis_engine/core.py — formatting helpers.
_format_var_block, _format_fat_tail_block, _format_garch_block.
"""


from analysis_engine.core import _format_var_block, _format_fat_tail_block, _format_garch_block


# ── _format_var_block ───────────────────────────────────────────────────────

class TestFormatVarBlock:
    def test_full_var_suite(self):
        stats = {
            "var_95": {
                "parametric": {"method": "Parametric-Gaussian", "VaR": 0.025, "CVaR": 0.032},
                "cornish_fisher": {"method": "Cornish-Fisher", "VaR": 0.028},
                "historical": {"method": "Historical", "VaR": 0.030, "CVaR": 0.040},
                "bootstrap": {
                    "method": "Bootstrap", "VaR": 0.029,
                    "CI_lower": 0.022, "CI_upper": 0.036,
                },
            },
            "garch_var_95": {"method": "GARCH(1,1)-t", "VaR": 0.031},
        }
        output = _format_var_block(stats)
        assert "Parametric-Gaussian" in output
        assert "Cornish-Fisher" in output
        assert "Bootstrap" in output
        assert "GARCH" in output

    def test_empty_var_suite(self):
        output = _format_var_block({})
        assert "unavailable" in output.lower() or output.strip() != ""


# ── _format_fat_tail_block ──────────────────────────────────────────────────

class TestFormatFatTailBlock:
    def test_full_fat_tail_dict(self):
        stats = {
            "fat_tails": {
                "skewness": -0.45,
                "excess_kurtosis": 3.2,
                "tail_classification": "Severely Leptokurtic",
                "jb_p_value": 1.2e-10,
                "is_normal_jb": False,
                "ks_p_value": 0.001,
            }
        }
        output = _format_fat_tail_block(stats)
        assert "Skewness" in output
        assert "Kurtosis" in output
        assert "Severely Leptokurtic" in output

    def test_missing_fat_tails(self):
        output = _format_fat_tail_block({})
        assert "unavailable" in output.lower()


# ── _format_garch_block ────────────────────────────────────────────────────

class TestFormatGarchBlock:
    def test_full_garch_dict(self):
        stats = {
            "garch": {
                "alpha": 0.08,
                "beta": 0.88,
                "persistence": 0.96,
                "distribution": "t",
                "current_cond_vol_annualized": 0.22,
                "annualized_long_run_vol": 0.20,
                "aic": 3200.5,
                "bic": 3220.3,
            }
        }
        output = _format_garch_block(stats)
        assert "persistence" in output.lower() or "0.96" in output
        assert "AIC" in output

    def test_garch_error(self):
        stats = {"garch": {"error": "arch package not installed"}}
        output = _format_garch_block(stats)
        assert "failed" in output.lower() or "not installed" in output.lower()

    def test_empty_garch(self):
        output = _format_garch_block({})
        assert "unavailable" in output.lower()
