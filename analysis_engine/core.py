"""
AFML Quant Pipeline - Analysis Engine Core
======================================
Pure data-fetch and formatting helpers.
The Gemini/Vertex AI agent system has been removed.
"""

import logging
from typing import Optional

import pandas as pd

from utils.data_provider import get_data_provider

logger = logging.getLogger(__name__)


def fetch_market_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch market data via OpenBBDataProvider."""
    logger.info(f"Fetching data for {ticker} ({period})...")
    provider = get_data_provider()
    df = provider.fetch_ohlcv(ticker, period=period)

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    return df


def _format_var_block(stats: dict) -> str:
    """Build a human-readable VaR comparison string."""
    lines = []

    var_suite = stats.get("var_95", {})
    for method_key in ("parametric", "cornish_fisher", "historical", "bootstrap"):
        entry = var_suite.get(method_key, {})
        if not entry:
            continue
        method_name = entry.get("method", method_key)
        var_val = entry.get("VaR")
        cvar_val = entry.get("CVaR")
        ci_lo = entry.get("CI_lower")
        ci_hi = entry.get("CI_upper")

        line = (
            f"  - {method_name}: VaR = {var_val:.4%}"
            if var_val
            else f"  - {method_name}: N/A"
        )
        if cvar_val is not None:
            line += f",  CVaR = {cvar_val:.4%}"
        if ci_lo is not None and ci_hi is not None:
            line += f"  [95% CI: {ci_lo:.4%} - {ci_hi:.4%}]"
        lines.append(line)

    gvar = stats.get("garch_var_95", {})
    if "VaR" in gvar:
        lines.append(f"  - {gvar['method']}: VaR = {gvar['VaR']:.4%}")

    return "\n".join(lines) if lines else "  VaR computation unavailable."


def _format_fat_tail_block(stats: dict) -> str:
    """Build a human-readable fat-tail diagnostics string."""
    ft = stats.get("fat_tails", {})
    if not ft:
        return "  Fat-tail analysis unavailable."

    return (
        f"  - Skewness: {ft['skewness']:.4f}\n"
        f"  - Excess Kurtosis: {ft['excess_kurtosis']:.4f}  ({ft['tail_classification']})\n"
        f"  - Jarque-Bera p-value: {ft['jb_p_value']:.4e}  "
        f"({'Normal' if ft['is_normal_jb'] else 'NON-Normal'})\n"
        f"  - KS-test p-value: {ft['ks_p_value']:.4e}"
    )


def _format_garch_block(stats: dict) -> str:
    """Build a human-readable GARCH summary string."""
    g = stats.get("garch", {})
    if "error" in g:
        return f"  GARCH fitting failed: {g['error']}"
    if not g:
        return "  GARCH analysis unavailable."

    lines = [
        f"  - alpha = {g.get('alpha', 0):.4f},  beta = {g.get('beta', 0):.4f},  "
        f"persistence (alpha+beta) = {g.get('persistence', 0):.4f}",
        f"  - Distribution: {g.get('distribution', 'N/A')}",
        f"  - Current Conditional Vol (ann.): {g.get('current_cond_vol_annualized', 0):.2%}",
    ]
    lr = g.get("annualized_long_run_vol")
    if lr is not None:
        lines.append(f"  - Long-run Vol (ann.): {lr:.2%}")
    lines.append(f"  - AIC: {g.get('aic', 'N/A'):.1f}   BIC: {g.get('bic', 'N/A'):.1f}")
    return "\n".join(lines)
