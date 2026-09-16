"""
Tests for utils/sanity_check.py — _parse_sanity_check_response & format_sanity_report.
Must set GOOGLE_API_KEY env var before import (module-level genai.configure).
"""

import os

# Module calls genai.configure(api_key=...) at import time.
# Ensure env var exists so the import doesn't crash.
os.environ.setdefault("GOOGLE_API_KEY", "dummy-key-for-testing")


from utils.sanity_check import _parse_sanity_check_response, format_sanity_report


# ── Helpers ─────────────────────────────────────────────────────────────────

FULL_RESPONSE = """\
ALPHA_DECAY: YES
RISK_LEVEL: HIGH

CONSENSUS:
The strategy has shown significant alpha decay over the past 18 months.
Multiple academic papers now discuss crowding effects.

KEY FINDINGS:
- Alpha has decayed by approximately 40%
- Crowding detected in momentum space
- Regime shift to low-volatility favours mean-reversion

RECOMMENDATIONS:
- Consider adding a volatility filter
- Reduce position sizes during crowded periods

CITATIONS:
- Lopez de Prado (2018) Advances in Financial ML
- Chan (2021) Machine Trading
"""

NO_DECAY_RESPONSE = """\
ALPHA_DECAY: NO
RISK_LEVEL: LOW

CONSENSUS:
The strategy remains viable with no significant alpha erosion.

KEY FINDINGS:
- Consistent returns over the last 24 months

RECOMMENDATIONS:
- Continue monitoring

CITATIONS:
- AQR Research Paper 2024
"""


# ── _parse_sanity_check_response ────────────────────────────────────────────

class TestParseSanityCheckResponse:
    def test_alpha_decay_yes(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "TestStrat")
        assert result["alpha_decay_detected"] is True

    def test_alpha_decay_no(self):
        result = _parse_sanity_check_response(NO_DECAY_RESPONSE, "SafeStrat")
        assert result["alpha_decay_detected"] is False

    def test_risk_level_parsed(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "TestStrat")
        assert result["risk_level"] == "high"

    def test_risk_level_low(self):
        result = _parse_sanity_check_response(NO_DECAY_RESPONSE, "SafeStrat")
        assert result["risk_level"] == "low"

    def test_consensus_populated(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "TestStrat")
        assert len(result["consensus"]) > 0
        assert "alpha decay" in result["consensus"].lower()

    def test_key_findings_list(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "TestStrat")
        assert isinstance(result["key_findings"], list)
        assert len(result["key_findings"]) == 3

    def test_recommendations_list(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "TestStrat")
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 2

    def test_citations_list(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "TestStrat")
        assert isinstance(result["citations"], list)
        assert len(result["citations"]) >= 1

    def test_strategy_name_preserved(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "MyStrategy")
        assert result["strategy"] == "MyStrategy"


# ── format_sanity_report ────────────────────────────────────────────────────

class TestFormatSanityReport:
    def test_report_contains_strategy_name(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "MomentumStrat")
        report = format_sanity_report(result)
        assert "MomentumStrat" in report

    def test_positive_decay_report(self):
        result = _parse_sanity_check_response(FULL_RESPONSE, "Strat")
        report = format_sanity_report(result)
        assert "YES" in report  # alpha decay marker

    def test_negative_decay_report(self):
        result = _parse_sanity_check_response(NO_DECAY_RESPONSE, "Strat")
        report = format_sanity_report(result)
        assert "NO" in report
