"""
AFML Quant Pipeline — Quantitative Research Platform
================================================
4-tab Streamlit dashboard backed by OpenBB v4.6.

Tabs:
  1. Forecast      — ensemble model + Monte Carlo
  2. Fundamentals  — financial statements + charts
  3. Backtest      — strategy backtesting + VBT optimiser
  4. Scanner       — market movers screener
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AFML Quant Pipeline — Quantitative Research Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  /* Base */
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stHeader"]           { background: transparent; }
  .block-container                   { padding-top: 1.2rem; max-width: 1400px; }

  /* Cards / surfaces */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.7rem 1rem;
  }

  /* Metric typography */
  .stMetric label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .stMetric [data-testid="stMetricValue"] {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e6edf3;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .stMetric [data-testid="stMetricDelta"] {
    font-size: 0.85rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px 4px 0 0;
    color: #8b949e;
    font-size: 0.82rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    padding: 0.4rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
    background: #1f2937 !important;
    color: #58a6ff !important;
    border-bottom-color: #1f2937 !important;
  }

  /* Buttons */
  .stButton > button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #58a6ff;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
    border-radius: 4px;
  }
  .stButton > button:hover { background: #30363d; }

  /* Inputs */
  .stTextInput input, .stSelectbox select {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.88rem;
  }

  /* DataFrames */
  [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 4px; }

  /* Caption / muted text */
  .stCaption { color: #8b949e; font-size: 0.75rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  h1, h2, h3 { color: #e6edf3; }
  p { color: #c9d1d9; }

  /* Divider */
  hr { border-color: #30363d; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Plotly base theme ──────────────────────────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(
        family="ui-monospace, SFMono-Regular, Menlo, monospace",
        color="#e6edf3",
        size=11,
    ),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    margin=dict(l=48, r=24, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1, font=dict(size=10)),
)

# Pastel model palette
_MODEL_COLORS = [
    "#58a6ff", "#3fb950", "#f0883e", "#d2a8ff",
    "#79c0ff", "#56d364", "#ff7b72", "#e3b341",
]


def _apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**_PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="margin-bottom:0.5rem;">
  <span style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
               font-size:1.9rem; font-weight:800; color:#58a6ff;
               letter-spacing:0.12em;">AFML PIPELINE</span>
  <span style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
               font-size:0.82rem; color:#8b949e; margin-left:1.2rem;">
    Quantitative Research Platform &mdash; OpenBB v4.6
  </span>
</div>
""",
    unsafe_allow_html=True,
)

# ── Cached data fetchers ───────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=600)
def _run_forecast(ticker: str, horizon: int, n_sims: int, history_days: int):
    from pipeline.forecaster import StockForecaster

    fc = StockForecaster(
        horizon=horizon, n_simulations=n_sims, history_days=history_days
    )
    return fc.forecast(ticker)


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_price_history(ticker, start, end)


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_fundamentals_metrics(ticker: str, period: str) -> pd.DataFrame:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_fundamentals_metrics(ticker, period=period, limit=10)


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_income(ticker: str, period: str) -> pd.DataFrame:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_income_statement(ticker, period=period, limit=10)


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_balance(ticker: str, period: str) -> pd.DataFrame:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_balance_sheet(ticker, period=period, limit=10)


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_cashflow(ticker: str, period: str) -> pd.DataFrame:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_cash_flow(ticker, period=period, limit=10)


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_profile(ticker: str) -> dict:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_equity_profile(ticker)


@st.cache_data(show_spinner=False, ttl=600)
def _fetch_consensus(ticker: str) -> dict:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_analyst_consensus(ticker)


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_news(ticker: str) -> list:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_company_news(ticker, limit=8)


@st.cache_data(show_spinner=False, ttl=120)
def _fetch_movers(screen: str) -> pd.DataFrame:
    from utils.obb_client import get_obb_client

    return get_obb_client().get_market_movers(screen)


@st.cache_data(show_spinner=False, ttl=600)
def _run_backtest(
    ticker: str,
    strategy: str,
    history: str,
    capital: float,
    allow_shorts: bool,
    params: dict,
) -> dict:
    from pipeline.backtester import Backtester

    start, end = _period_to_dates(history)
    bt = Backtester(
        ticker=ticker,
        strategy=strategy,
        start_date=start,
        end_date=end,
        initial_capital=capital,
        allow_shorts=allow_shorts,
        strategy_params=params,
    )
    return bt.run()


def _period_to_dates(period: str):
    from utils.obb_client import get_obb_client

    return get_obb_client()._period_to_dates(period)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _metric(label: str, value, delta=None, help: Optional[str] = None):
    st.metric(label=label, value=value, delta=delta, help=help)


def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.2%}"
    except Exception:
        return "N/A"


def _fmt_money(v, suffix: str = "") -> str:
    try:
        v = float(v)
        if abs(v) >= 1e12:
            return f"${v/1e12:.2f}T{suffix}"
        if abs(v) >= 1e9:
            return f"${v/1e9:.2f}B{suffix}"
        if abs(v) >= 1e6:
            return f"${v/1e6:.2f}M{suffix}"
        return f"${v:,.2f}{suffix}"
    except Exception:
        return "N/A"


def _fmt_num(v, decimals: int = 2) -> str:
    try:
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return "N/A"


# ── TABS ───────────────────────────────────────────────────────────────────────
tab_forecast, tab_fund, tab_back, tab_scan = st.tabs(
    ["Forecast", "Fundamentals", "Backtest", "Market Scanner"]
)


# ==============================================================================
# TAB 1 — FORECAST
# ==============================================================================
with tab_forecast:
    # Controls
    col_t, col_h, col_s, col_hist, col_btn = st.columns([3, 2, 2, 2, 1.5])
    with col_t:
        fc_ticker = st.text_input(
            "Ticker", value="AAPL", key="fc_ticker", label_visibility="collapsed",
            placeholder="Ticker (e.g. AAPL)"
        ).upper().strip()
    with col_h:
        fc_horizon = st.selectbox(
            "Horizon", [7, 14, 30, 60, 90], index=2, key="fc_horizon",
            format_func=lambda x: f"{x}d horizon"
        )
    with col_s:
        fc_sims = st.selectbox(
            "MC Paths",
            [5_000, 10_000, 20_000, 50_000, 100_000],
            index=2,
            key="fc_sims",
            format_func=lambda x: f"{x:,} paths",
        )
    with col_hist:
        fc_hist = st.selectbox(
            "History",
            ["1y", "2y", "3y", "5y"],
            index=2,
            key="fc_hist",
            format_func=lambda x: f"{x} history",
        )
    with col_btn:
        fc_run = st.button("RUN FORECAST", key="fc_run", use_container_width=True)

    if fc_run and fc_ticker:
        from utils.obb_client import get_obb_client

        hist_days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}[fc_hist]
        with st.spinner(f"Forecasting {fc_ticker}..."):
            try:
                result = _run_forecast(fc_ticker, fc_horizon, fc_sims, hist_days)
                st.session_state["fc_result"] = result
            except Exception as e:
                st.error(f"Forecast failed: {e}")

    result = st.session_state.get("fc_result")

    if result is not None:
        r = result
        pred_p = r.predicted_price()
        delta_pct = r.predicted_return_pct

        # ── 8-metric summary row ──────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        with c1:
            _metric("Current Price", f"${r.current_price:.2f}")
        with c2:
            _metric(
                "Predicted Price",
                f"${pred_p:.2f}",
                delta=f"{delta_pct:+.1f}%",
            )
        with c3:
            _metric("P(Gain)", _fmt_pct(r.probability_of_gain))
        with c4:
            _metric("P(Loss >5%)", _fmt_pct(r.probability_of_loss_5pct))
        with c5:
            _metric("P(Loss >10%)", _fmt_pct(r.probability_of_loss_10pct))
        with c6:
            _metric("Annual Vol", _fmt_pct(r.volatility_annual))
        with c7:
            _metric("Models", str(len(r.models_used)))
        with c8:
            rec = ""
            if r.probability_of_gain >= 0.60:
                rec = "BULLISH"
            elif r.probability_of_gain <= 0.40:
                rec = "BEARISH"
            else:
                rec = "NEUTRAL"
            _metric("Signal", rec)

        # ── Analyst strip (if available) ──────────────────────────────────────
        consensus = _fetch_consensus(r.ticker)
        if consensus:
            st.divider()
            ac1, ac2, ac3, ac4, ac5 = st.columns(5)
            target = consensus.get("target_consensus") or consensus.get("target_median")
            with ac1:
                _metric("Consensus Target", f"${float(target):.2f}" if target else "N/A")
            with ac2:
                hi = consensus.get("target_high")
                _metric("Target High", f"${float(hi):.2f}" if hi else "N/A")
            with ac3:
                lo = consensus.get("target_low")
                _metric("Target Low", f"${float(lo):.2f}" if lo else "N/A")
            with ac4:
                n = consensus.get("number_of_analysts")
                _metric("Analysts", str(int(n)) if n else "N/A")
            with ac5:
                rec_str = consensus.get("recommendation", "")
                _metric("Recommendation", rec_str or "N/A")

        st.divider()

        # ── Price chart ───────────────────────────────────────────────────────
        start_hist, end_hist = _period_to_dates("6mo")
        try:
            hist_df = _fetch_price_history(r.ticker, start_hist, end_hist)
        except Exception:
            hist_df = pd.DataFrame()

        fig_price = go.Figure()

        if not hist_df.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=hist_df.index,
                    y=hist_df["Close"],
                    name="Historical",
                    line=dict(color="#4d5566", width=1.5),
                    showlegend=True,
                )
            )
            fig_price.add_hline(
                y=r.current_price,
                line_dash="dash",
                line_color="#4d5566",
                annotation_text=f"  ${r.current_price:.2f}",
                annotation_font_color="#8b949e",
            )

        # MC fan
        if r.mc_percentiles is not None:
            pct = r.mc_percentiles
            fan_colors = [
                "rgba(88,166,255,0.08)",
                "rgba(88,166,255,0.12)",
                "rgba(88,166,255,0.16)",
                "rgba(88,166,255,0.12)",
                "rgba(88,166,255,0.08)",
            ]
            bands = [("p5", "p95"), ("p25", "p75")]
            for lo_col, hi_col in bands:
                fig_price.add_trace(
                    go.Scatter(
                        x=list(pct.index) + list(pct.index[::-1]),
                        y=list(pct[hi_col]) + list(pct[lo_col][::-1]),
                        fill="toself",
                        fillcolor="rgba(88,166,255,0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name=f"MC {lo_col}-{hi_col}",
                        showlegend=True,
                    )
                )
            fig_price.add_trace(
                go.Scatter(
                    x=pct.index,
                    y=pct["p50"],
                    name="MC Median",
                    line=dict(color="#58a6ff", width=1, dash="dot"),
                )
            )

        # Individual model lines
        for i, (name, mf) in enumerate(r.model_forecasts.items()):
            if name == "Analyst Consensus":
                continue
            col = _MODEL_COLORS[i % len(_MODEL_COLORS)]
            fig_price.add_trace(
                go.Scatter(
                    x=mf.predicted_prices.index,
                    y=mf.predicted_prices.values,
                    name=f"{name} (w={mf.weight:.2f})",
                    line=dict(color=col, width=1),
                    opacity=0.6,
                )
            )

        # Ensemble
        if len(r.ensemble_mean) > 0:
            fig_price.add_trace(
                go.Scatter(
                    x=r.ensemble_mean.index,
                    y=r.ensemble_mean.values,
                    name="Ensemble",
                    line=dict(color="#e6edf3", width=2.5),
                )
            )

        _apply_layout(fig_price, height=500, title=f"{r.ticker} Price Forecast — {r.horizon}d")
        st.plotly_chart(fig_price, use_container_width=True)

        # ── Monte Carlo section ───────────────────────────────────────────────
        if r.mc_final_prices is not None:
            col_fan, col_hist_mc = st.columns([6, 4])

            with col_fan:
                if r.mc_percentiles is not None:
                    pct = r.mc_percentiles
                    fig_fan = go.Figure()
                    fig_fan.add_trace(
                        go.Scatter(
                            x=list(pct.index) + list(pct.index[::-1]),
                            y=list(pct["p95"]) + list(pct["p5"][::-1]),
                            fill="toself",
                            fillcolor="rgba(88,166,255,0.10)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="p5-p95",
                        )
                    )
                    fig_fan.add_trace(
                        go.Scatter(
                            x=list(pct.index) + list(pct.index[::-1]),
                            y=list(pct["p75"]) + list(pct["p25"][::-1]),
                            fill="toself",
                            fillcolor="rgba(88,166,255,0.18)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="p25-p75",
                        )
                    )
                    fig_fan.add_trace(
                        go.Scatter(
                            x=pct.index,
                            y=pct["p50"],
                            name="Median",
                            line=dict(color="#58a6ff", width=2),
                        )
                    )
                    _apply_layout(fig_fan, height=340, title="MC Fan Chart")
                    st.plotly_chart(fig_fan, use_container_width=True)

            with col_hist_mc:
                fig_hist_mc = go.Figure()
                fig_hist_mc.add_trace(
                    go.Histogram(
                        x=r.mc_final_prices,
                        nbinsx=60,
                        marker_color="#58a6ff",
                        opacity=0.7,
                        name="Final Prices",
                    )
                )
                fig_hist_mc.add_vline(
                    x=r.current_price,
                    line_dash="dash",
                    line_color="#f85149",
                    annotation_text="  Current",
                    annotation_font_color="#f85149",
                )
                _apply_layout(
                    fig_hist_mc,
                    height=220,
                    title=f"Final Price Distribution (n={len(r.mc_final_prices):,})",
                )
                st.plotly_chart(fig_hist_mc, use_container_width=True)

                # Probability table
                thresholds = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
                rows = []
                for thr in thresholds:
                    target_p = r.current_price * (1 + thr)
                    prob = float(np.mean(r.mc_final_prices >= target_p))
                    rows.append({"Return Threshold": f"{thr:+.0%}", "P(price ≥ target)": f"{prob:.1%}"})
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                )

        # ── Model comparison table ────────────────────────────────────────────
        st.subheader("Model Comparison")
        model_rows = []
        for name, mf in r.model_forecasts.items():
            pp = float(mf.predicted_prices.iloc[-1]) if len(mf.predicted_prices) > 0 else np.nan
            model_rows.append(
                {
                    "Model": name,
                    "Predicted Price": f"${pp:.2f}" if not np.isnan(pp) else "N/A",
                    "Return": f"{(pp / r.current_price - 1):+.2%}" if not np.isnan(pp) else "N/A",
                    "RMSE": f"{mf.validation_rmse:.4f}" if not np.isnan(mf.validation_rmse) else "N/A",
                    "Weight": f"{mf.weight:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)

        # ── Risk panel ────────────────────────────────────────────────────────
        st.subheader("Risk Analysis")
        col_dist, col_ann = st.columns(2)

        if r.mc_final_prices is not None:
            fp = r.mc_final_prices
            ret = (fp / r.current_price - 1)
            with col_dist:
                st.markdown("**Distribution Statistics**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Statistic": ["Mean", "Std Dev", "Skewness", "Kurtosis", "Min", "Max"],
                            "Value": [
                                f"{np.mean(ret):+.2%}",
                                f"{np.std(ret):.2%}",
                                f"{float(pd.Series(ret).skew()):.3f}",
                                f"{float(pd.Series(ret).kurtosis()):.3f}",
                                f"{np.min(ret):+.2%}",
                                f"{np.max(ret):+.2%}",
                            ],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            with col_ann:
                st.markdown("**Annualised Risk Metrics**")
                ann_factor = np.sqrt(252 / max(r.horizon, 1))
                daily_ret_mean = np.mean(ret) / r.horizon
                daily_ret_std = r.volatility_annual / np.sqrt(252)
                sharpe = (daily_ret_mean / daily_ret_std * np.sqrt(252)) if daily_ret_std > 0 else 0
                var_95 = float(np.percentile(ret, 5))
                max_dd = float(np.min(ret))
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Metric": ["Annual Vol", "Sharpe (est.)", "VaR 95%", "Max Drawdown", "P(Gain)"],
                            "Value": [
                                _fmt_pct(r.volatility_annual),
                                f"{sharpe:.2f}",
                                _fmt_pct(var_95),
                                _fmt_pct(max_dd),
                                _fmt_pct(r.probability_of_gain),
                            ],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        st.caption(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==============================================================================
# TAB 2 — FUNDAMENTALS
# ==============================================================================
with tab_fund:
    col_ft, col_per, col_fbtn = st.columns([3, 2, 1.5])
    with col_ft:
        fund_ticker = st.text_input(
            "Ticker", value="AAPL", key="fund_ticker", label_visibility="collapsed",
            placeholder="Ticker (e.g. AAPL)"
        ).upper().strip()
    with col_per:
        fund_period = st.radio(
            "Period", ["annual", "quarterly"], horizontal=True, key="fund_period"
        )
    with col_fbtn:
        fund_load = st.button("LOAD", key="fund_load", use_container_width=True)

    if fund_load and fund_ticker:
        with st.spinner(f"Loading fundamentals for {fund_ticker}..."):
            try:
                _fetch_fundamentals_metrics(fund_ticker, fund_period)
                st.session_state["fund_loaded"] = fund_ticker
                st.session_state["fund_period_loaded"] = fund_period
            except Exception as e:
                st.error(f"Failed to load fundamentals: {e}")

    loaded_ticker = st.session_state.get("fund_loaded")
    loaded_period = st.session_state.get("fund_period_loaded", "annual")

    if loaded_ticker:
        tk = loaded_ticker
        per = loaded_period

        # Profile header
        profile = _fetch_profile(tk)
        if profile:
            name = profile.get("name") or profile.get("long_name") or tk
            sector = profile.get("sector", "")
            industry = profile.get("industry", "")
            desc = str(profile.get("description") or profile.get("long_description") or "")
            employees = profile.get("full_time_employees") or profile.get("employees")
            website = profile.get("website", "")

            st.markdown(
                f"<h2 style='color:#e6edf3;margin-bottom:0'>{name}</h2>"
                f"<p style='color:#8b949e;font-size:0.82rem;margin-top:0'>"
                f"{sector}{' · ' + industry if industry else ''}"
                + (f" · {int(employees):,} employees" if employees else "")
                + (f" · <a href='{website}' target='_blank' style='color:#58a6ff'>{website}</a>" if website else "")
                + "</p>",
                unsafe_allow_html=True,
            )
            if desc:
                with st.expander("Company Description"):
                    st.write(desc[:2000] + ("..." if len(desc) > 2000 else ""))
            st.divider()

        # Load metrics
        try:
            metrics_df = _fetch_fundamentals_metrics(tk, per)
            row = metrics_df.iloc[0] if not metrics_df.empty else pd.Series(dtype=float)

            def _r(key: str) -> Optional[float]:
                try:
                    v = row.get(key)
                    return float(v) if v is not None else None
                except Exception:
                    return None

            # Row 1: Valuation (8 cols)
            st.markdown("**Valuation**")
            v1, v2, v3, v4, v5, v6, v7, v8 = st.columns(8)
            with v1:
                _metric("P/E", _fmt_num(_r("pe_ratio")))
            with v2:
                _metric("Fwd P/E", _fmt_num(_r("forward_pe")))
            with v3:
                _metric("P/B", _fmt_num(_r("price_to_book")))
            with v4:
                _metric("EV/EBITDA", _fmt_num(_r("ev_to_ebitda")))
            with v5:
                _metric("Mkt Cap", _fmt_money(_r("market_cap")))
            with v6:
                _metric("EV", _fmt_money(_r("enterprise_value")))
            with v7:
                _metric("Beta", _fmt_num(_r("beta")))
            with v8:
                _metric("Div Yield", _fmt_pct(_r("dividend_yield")))

            # Row 2: Quality (8 cols)
            st.markdown("**Quality**")
            q1, q2, q3, q4, q5, q6, q7, q8 = st.columns(8)
            with q1:
                _metric("ROE", _fmt_pct(_r("return_on_equity")))
            with q2:
                _metric("ROA", _fmt_pct(_r("return_on_assets")))
            with q3:
                _metric("Gross Margin", _fmt_pct(_r("gross_profit_margin")))
            with q4:
                _metric("Op Margin", _fmt_pct(_r("operating_profit_margin")))
            with q5:
                _metric("Net Margin", _fmt_pct(_r("net_profit_margin")))
            with q6:
                _metric("D/E", _fmt_num(_r("debt_equity_ratio")))
            with q7:
                _metric("Current Ratio", _fmt_num(_r("current_ratio")))
            with q8:
                _metric("Rev Growth", _fmt_pct(_r("revenue_growth")))

        except Exception as e:
            st.warning(f"Metrics unavailable: {e}")
            row = pd.Series(dtype=float)

        st.divider()

        # Financial charts
        col_inc, col_bal = st.columns(2)

        with col_inc:
            try:
                inc_df = _fetch_income(tk, per)
                if not inc_df.empty:
                    rev_col = next((c for c in inc_df.columns if "revenue" in c.lower()), None)
                    gp_col = next((c for c in inc_df.columns if "gross_profit" in c.lower()), None)
                    ni_col = next((c for c in inc_df.columns if "net_income" in c.lower()), None)
                    nm_col = next((c for c in inc_df.columns if "net_profit_margin" in c.lower() or "net_margin" in c.lower()), None)

                    fig_inc = make_subplots(specs=[[{"secondary_y": True}]])
                    x_vals = inc_df.index.tolist() if inc_df.index.dtype != object else list(range(len(inc_df)))

                    for col_name, label, color in [
                        (rev_col, "Revenue", "#58a6ff"),
                        (gp_col, "Gross Profit", "#3fb950"),
                        (ni_col, "Net Income", "#f0883e"),
                    ]:
                        if col_name and col_name in inc_df.columns:
                            fig_inc.add_trace(
                                go.Bar(
                                    x=x_vals,
                                    y=inc_df[col_name],
                                    name=label,
                                    marker_color=color,
                                    opacity=0.8,
                                ),
                                secondary_y=False,
                            )

                    if nm_col and nm_col in inc_df.columns:
                        fig_inc.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=inc_df[nm_col],
                                name="Net Margin",
                                line=dict(color="#d2a8ff", width=2),
                                mode="lines+markers",
                            ),
                            secondary_y=True,
                        )

                    _apply_layout(fig_inc, height=380, title="Income Statement")
                    fig_inc.update_yaxes(
                        title_text="$", secondary_y=False,
                        tickfont=dict(color="#8b949e"),
                    )
                    fig_inc.update_yaxes(
                        title_text="Margin %", secondary_y=True,
                        tickfont=dict(color="#d2a8ff"),
                        tickformat=".0%",
                    )
                    st.plotly_chart(fig_inc, use_container_width=True)
            except Exception as e:
                st.caption(f"Income chart unavailable: {e}")

        with col_bal:
            try:
                bal_df = _fetch_balance(tk, per)
                if not bal_df.empty:
                    asset_col = next((c for c in bal_df.columns if "total_asset" in c.lower()), None)
                    liab_col = next((c for c in bal_df.columns if "total_liabilit" in c.lower()), None)
                    eq_col = next((c for c in bal_df.columns if "total_equity" in c.lower() or "stockholder" in c.lower()), None)
                    de_col = next((c for c in bal_df.columns if "debt_equity" in c.lower()), None)

                    fig_bal = make_subplots(specs=[[{"secondary_y": True}]])
                    x_vals = bal_df.index.tolist() if bal_df.index.dtype != object else list(range(len(bal_df)))

                    for col_name, label, color in [
                        (asset_col, "Assets", "#58a6ff"),
                        (liab_col, "Liabilities", "#f85149"),
                        (eq_col, "Equity", "#3fb950"),
                    ]:
                        if col_name and col_name in bal_df.columns:
                            fig_bal.add_trace(
                                go.Bar(x=x_vals, y=bal_df[col_name], name=label, marker_color=color, opacity=0.8),
                                secondary_y=False,
                            )

                    if de_col and de_col in bal_df.columns:
                        fig_bal.add_trace(
                            go.Scatter(
                                x=x_vals, y=bal_df[de_col], name="D/E Ratio",
                                line=dict(color="#e3b341", width=2), mode="lines+markers"
                            ),
                            secondary_y=True,
                        )

                    _apply_layout(fig_bal, height=380, title="Balance Sheet")
                    fig_bal.update_yaxes(title_text="$", secondary_y=False, tickfont=dict(color="#8b949e"))
                    fig_bal.update_yaxes(title_text="D/E", secondary_y=True, tickfont=dict(color="#e3b341"))
                    st.plotly_chart(fig_bal, use_container_width=True)
            except Exception as e:
                st.caption(f"Balance chart unavailable: {e}")

        # Cash flow chart
        try:
            cf_df = _fetch_cashflow(tk, per)
            if not cf_df.empty:
                op_col = next((c for c in cf_df.columns if "operating" in c.lower() and "cash" in c.lower()), None)
                cap_col = next((c for c in cf_df.columns if "capex" in c.lower() or "capital_expenditure" in c.lower()), None)
                fcf_col = next((c for c in cf_df.columns if "free_cash" in c.lower()), None)

                fig_cf = go.Figure()
                x_vals = cf_df.index.tolist() if cf_df.index.dtype != object else list(range(len(cf_df)))

                for col_name, label, color in [
                    (op_col, "Operating CF", "#3fb950"),
                    (cap_col, "Capex", "#f85149"),
                    (fcf_col, "Free CF", "#58a6ff"),
                ]:
                    if col_name and col_name in cf_df.columns:
                        fig_cf.add_trace(
                            go.Bar(x=x_vals, y=cf_df[col_name], name=label, marker_color=color, opacity=0.8)
                        )

                _apply_layout(fig_cf, height=380, title="Cash Flow Statement")
                st.plotly_chart(fig_cf, use_container_width=True)
        except Exception as e:
            st.caption(f"Cash flow chart unavailable: {e}")

        st.divider()

        # Analyst + news
        col_consensus, col_news = st.columns(2)

        with col_consensus:
            st.subheader("Analyst Consensus")
            cons = _fetch_consensus(tk)
            if cons:
                target = float(cons.get("target_consensus") or cons.get("target_median") or 0)
                target_lo = float(cons.get("target_low") or target)
                target_hi = float(cons.get("target_high") or target)
                rec = str(cons.get("recommendation") or "")

                try:
                    cur_price = float(
                        _fetch_price_history(
                            tk,
                            *_period_to_dates("5d"),
                        )["Close"].iloc[-1]
                    )
                except Exception:
                    cur_price = target

                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=target,
                        delta={"reference": cur_price, "valueformat": ".2f"},
                        title={"text": f"12-month Target  ({rec})", "font": {"size": 13, "color": "#8b949e"}},
                        gauge={
                            "axis": {"range": [target_lo * 0.85, target_hi * 1.10], "tickcolor": "#8b949e"},
                            "bar": {"color": "#58a6ff"},
                            "bgcolor": "#161b22",
                            "borderwidth": 1,
                            "bordercolor": "#30363d",
                            "steps": [
                                {"range": [target_lo * 0.85, target_lo], "color": "#f85149"},
                                {"range": [target_lo, target_hi], "color": "#21262d"},
                                {"range": [target_hi, target_hi * 1.10], "color": "#3fb950"},
                            ],
                            "threshold": {
                                "line": {"color": "#f85149", "width": 2},
                                "value": cur_price,
                            },
                        },
                        number={"prefix": "$", "font": {"color": "#e6edf3"}},
                    )
                )
                _apply_layout(fig_gauge, height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.caption("Analyst consensus unavailable.")

        with col_news:
            st.subheader("Company News")
            news = _fetch_news(tk)
            if news:
                for item in news[:8]:
                    title = item.get("title", "Untitled")
                    url = item.get("url", "")
                    date = item.get("date", "")[:10]
                    if url:
                        st.markdown(
                            f"<p style='margin:0.2rem 0;font-size:0.82rem;'>"
                            f"<span style='color:#8b949e;font-size:0.72rem;'>{date}</span>  "
                            f"<a href='{url}' target='_blank' style='color:#58a6ff;'>{title}</a>"
                            f"</p>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<p style='margin:0.2rem 0;font-size:0.82rem;color:#c9d1d9;'>"
                            f"<span style='color:#8b949e;font-size:0.72rem;'>{date}</span>  {title}</p>",
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("News unavailable.")

        # Raw data expander
        with st.expander("Raw Metrics Data"):
            try:
                st.dataframe(_fetch_fundamentals_metrics(tk, per), use_container_width=True)
            except Exception:
                st.caption("No data.")


# ==============================================================================
# TAB 3 — BACKTEST
# ==============================================================================
with tab_back:
    bc1, bc2, bc3, bc4, bc5, bc6 = st.columns([2.5, 2, 2, 2, 1.5, 1.5])
    with bc1:
        bt_ticker = st.text_input(
            "Ticker", value="SPY", key="bt_ticker", label_visibility="collapsed",
            placeholder="Ticker (e.g. SPY)"
        ).upper().strip()
    with bc2:
        bt_strategy = st.selectbox(
            "Strategy",
            ["MA Crossover", "RSI", "Bollinger", "MACD", "Momentum"],
            key="bt_strategy",
            label_visibility="collapsed",
        )
    with bc3:
        bt_history = st.selectbox(
            "History", ["1y", "2y", "3y", "5y"], index=2, key="bt_history",
            format_func=lambda x: f"{x} history", label_visibility="collapsed"
        )
    with bc4:
        bt_capital = st.number_input(
            "Capital ($)", min_value=1000, max_value=10_000_000,
            value=100_000, step=10_000, key="bt_capital", label_visibility="collapsed"
        )
    with bc5:
        bt_shorts = st.checkbox("Allow Shorts", key="bt_shorts", value=False)
    with bc6:
        bt_run = st.button("RUN", key="bt_run", use_container_width=True)

    # Dynamic params per strategy
    st.markdown("**Strategy Parameters**")
    bt_params = {}
    if bt_strategy == "MA Crossover":
        p1, p2 = st.columns(2)
        bt_params["fast"] = p1.slider("Fast Window", 5, 50, 20, key="bt_fast")
        bt_params["slow"] = p2.slider("Slow Window", 20, 200, 50, key="bt_slow")
    elif bt_strategy == "RSI":
        p1, p2, p3 = st.columns(3)
        bt_params["window"] = p1.slider("RSI Window", 7, 30, 14, key="bt_rsi_w")
        bt_params["oversold"] = p2.slider("Oversold", 20, 40, 30, key="bt_rsi_os")
        bt_params["overbought"] = p3.slider("Overbought", 60, 80, 70, key="bt_rsi_ob")
    elif bt_strategy == "Bollinger":
        p1, p2 = st.columns(2)
        bt_params["window"] = p1.slider("Window", 10, 50, 20, key="bt_bb_w")
        bt_params["std"] = p2.slider("Std Devs", 1.0, 3.0, 2.0, 0.5, key="bt_bb_std")
    elif bt_strategy == "MACD":
        p1, p2, p3 = st.columns(3)
        bt_params["fast"] = p1.slider("Fast", 8, 20, 12, key="bt_macd_f")
        bt_params["slow"] = p2.slider("Slow", 20, 40, 26, key="bt_macd_s")
        bt_params["signal"] = p3.slider("Signal", 5, 15, 9, key="bt_macd_sig")
    elif bt_strategy == "Momentum":
        bt_params["window"] = st.slider("Lookback", 5, 60, 20, key="bt_mom_w")

    if bt_run and bt_ticker:
        with st.spinner(f"Backtesting {bt_strategy} on {bt_ticker}..."):
            try:
                bt_result = _run_backtest(
                    bt_ticker,
                    bt_strategy,
                    bt_history,
                    float(bt_capital),
                    bt_shorts,
                    bt_params,
                )
                st.session_state["bt_result"] = bt_result
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    bt_res = st.session_state.get("bt_result")

    if bt_res is not None:
        metrics = bt_res.get("metrics", {})
        equity = bt_res.get("equity_curve", pd.Series(dtype=float))
        trades = bt_res.get("trades", pd.DataFrame())
        drawdown = bt_res.get("drawdown", pd.Series(dtype=float))

        # Summary metrics
        m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
        with m1:
            _metric("Total Return", _fmt_pct(metrics.get("total_return")))
        with m2:
            _metric("CAGR", _fmt_pct(metrics.get("cagr")))
        with m3:
            _metric("Sharpe", _fmt_num(metrics.get("sharpe")))
        with m4:
            _metric("Sortino", _fmt_num(metrics.get("sortino")))
        with m5:
            _metric("Max DD", _fmt_pct(metrics.get("max_drawdown")))
        with m6:
            _metric("Win Rate", _fmt_pct(metrics.get("win_rate")))
        with m7:
            _metric("Profit Factor", _fmt_num(metrics.get("profit_factor")))
        with m8:
            _metric("Trades", str(int(metrics.get("n_trades", 0))))

        # Equity + drawdown chart
        if len(equity) > 0:
            fig_eq = make_subplots(
                rows=2, cols=1,
                row_heights=[0.70, 0.30],
                shared_xaxes=True,
                vertical_spacing=0.04,
            )

            fig_eq.add_trace(
                go.Scatter(
                    x=equity.index, y=equity.values,
                    name="Equity", line=dict(color="#58a6ff", width=2),
                ),
                row=1, col=1,
            )

            # Buy/sell markers
            if not trades.empty:
                for side, symbol_m, color in [("BUY", "triangle-up", "#3fb950"), ("SELL", "triangle-down", "#f85149")]:
                    t_sub = trades[trades.get("side", pd.Series(dtype=str)) == side] if "side" in trades.columns else pd.DataFrame()
                    if not t_sub.empty and "date" in t_sub.columns and "price" in t_sub.columns:
                        fig_eq.add_trace(
                            go.Scatter(
                                x=t_sub["date"], y=t_sub["price"],
                                mode="markers",
                                marker=dict(symbol=symbol_m, color=color, size=8),
                                name=side,
                            ),
                            row=1, col=1,
                        )

            if len(drawdown) > 0:
                fig_eq.add_trace(
                    go.Scatter(
                        x=drawdown.index, y=drawdown.values,
                        name="Drawdown",
                        fill="tozeroy",
                        fillcolor="rgba(248,81,73,0.15)",
                        line=dict(color="#f85149", width=1),
                    ),
                    row=2, col=1,
                )

            _apply_layout(fig_eq, height=520, title=f"{bt_ticker} — {bt_strategy} Strategy")
            fig_eq.update_yaxes(title_text="Portfolio Value ($)", row=1, tickfont=dict(color="#8b949e"))
            fig_eq.update_yaxes(title_text="Drawdown", row=2, tickformat=".0%", tickfont=dict(color="#8b949e"))
            st.plotly_chart(fig_eq, use_container_width=True)

        # Trade log
        if not trades.empty:
            st.subheader("Trade Log (last 50)")
            st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)

        # VBT optimiser
        with st.expander("Parameter Optimisation (VectorBT)"):
            if st.button("RUN OPTIMISATION", key="vbt_opt"):
                with st.spinner("Running VectorBT grid search..."):
                    try:
                        from pipeline.optimiser import StrategyOptimiser

                        opt = StrategyOptimiser()
                        strat_lower = bt_strategy.lower().replace(" ", "_")
                        opt_map = {
                            "ma_crossover": opt.optimize_ma_crossover,
                            "rsi": opt.optimize_rsi,
                            "bollinger": opt.optimize_bollinger,
                        }
                        opt_fn = opt_map.get(strat_lower)
                        if opt_fn:
                            opt_res = opt_fn(bt_ticker, period=bt_history)
                            st.success(
                                f"Best params: {opt_res.best_params}  "
                                f"| Best Sharpe: {opt_res.best_sharpe:.3f}"
                            )
                            if not opt_res.sharpe_grid.empty:
                                fig_heat = go.Figure(
                                    go.Heatmap(
                                        z=opt_res.sharpe_grid.values,
                                        x=opt_res.sharpe_grid.columns.tolist(),
                                        y=opt_res.sharpe_grid.index.tolist(),
                                        colorscale="Blues",
                                        colorbar=dict(title="Sharpe"),
                                    )
                                )
                                _apply_layout(fig_heat, height=380, title="Sharpe Ratio Heatmap")
                                st.plotly_chart(fig_heat, use_container_width=True)
                        else:
                            st.info("Optimisation not available for this strategy.")
                    except Exception as e:
                        st.error(f"Optimisation failed: {e}")

        # QuantStats tearsheet download
        tearsheet_path = bt_res.get("tearsheet_path")
        if tearsheet_path:
            try:
                with open(tearsheet_path, "rb") as f:
                    st.download_button(
                        "Download QuantStats Tearsheet",
                        data=f,
                        file_name=f"tearsheet_{bt_ticker}.html",
                        mime="text/html",
                    )
            except Exception:
                pass


# ==============================================================================
# TAB 4 — MARKET SCANNER
# ==============================================================================
with tab_scan:
    sc1, sc2, sc3 = st.columns([3, 2, 2])
    with sc1:
        scan_screen = st.selectbox(
            "Screen",
            ["gainers", "losers", "active", "undervalued_growth", "growth_tech"],
            key="scan_screen",
            format_func=lambda x: {
                "gainers": "Top Gainers",
                "losers": "Top Losers",
                "active": "Most Active",
                "undervalued_growth": "Undervalued Growth",
                "growth_tech": "Growth Tech",
            }[x],
            label_visibility="collapsed",
        )
    with sc2:
        scan_btn = st.button("REFRESH", key="scan_btn", use_container_width=True)
    with sc3:
        scan_ts = st.empty()

    if scan_btn:
        with st.spinner(f"Loading {scan_screen}..."):
            try:
                scan_df = _fetch_movers(scan_screen)
                st.session_state["scan_df"] = scan_df
                st.session_state["scan_ts"] = datetime.now().strftime("%H:%M:%S")
                st.session_state["scan_screen_loaded"] = scan_screen
            except Exception as e:
                st.error(f"Scanner failed: {e}")

    scan_df = st.session_state.get("scan_df")
    scan_time = st.session_state.get("scan_ts", "")
    if scan_time:
        scan_ts.caption(f"Last updated: {scan_time}")

    if scan_df is not None and not scan_df.empty:
        # Filter to available columns
        desired_cols = [
            "symbol", "name", "price", "change_percent", "volume",
            "market_cap", "pe_ratio", "fifty_two_week_high", "fifty_two_week_low",
        ]
        available_cols = [c for c in desired_cols if c in scan_df.columns]
        if not available_cols:
            available_cols = list(scan_df.columns[:9])

        display_df = scan_df[available_cols].copy()

        col_config = {}
        if "price" in available_cols:
            col_config["price"] = st.column_config.NumberColumn("Price", format="$%.2f")
        if "change_percent" in available_cols:
            col_config["change_percent"] = st.column_config.NumberColumn("Change %", format="%.2f%%")
        if "volume" in available_cols:
            col_config["volume"] = st.column_config.NumberColumn("Volume", format="%d")
        if "market_cap" in available_cols:
            col_config["market_cap"] = st.column_config.NumberColumn("Market Cap", format="$%.0f")
        if "pe_ratio" in available_cols:
            col_config["pe_ratio"] = st.column_config.NumberColumn("P/E", format="%.1f")
        if "fifty_two_week_high" in available_cols:
            col_config["fifty_two_week_high"] = st.column_config.NumberColumn("52W High", format="$%.2f")
        if "fifty_two_week_low" in available_cols:
            col_config["fifty_two_week_low"] = st.column_config.NumberColumn("52W Low", format="$%.2f")

        selected = st.dataframe(
            display_df,
            column_config=col_config,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Mini sparkline on row selection
        if selected and selected.selection and selected.selection.get("rows"):
            sel_idx = selected.selection["rows"][0]
            sel_row = display_df.iloc[sel_idx]
            sel_sym = sel_row.get("symbol", "")
            if sel_sym:
                with st.spinner(f"Loading sparkline for {sel_sym}..."):
                    try:
                        start_90, end_90 = _period_to_dates("3mo")
                        spark_df = _fetch_price_history(str(sel_sym), start_90, end_90)
                        if not spark_df.empty:
                            fig_spark = go.Figure(
                                go.Scatter(
                                    x=spark_df.index,
                                    y=spark_df["Close"],
                                    fill="tozeroy",
                                    fillcolor="rgba(88,166,255,0.08)",
                                    line=dict(color="#58a6ff", width=2),
                                    name=str(sel_sym),
                                )
                            )
                            _apply_layout(
                                fig_spark,
                                height=200,
                                title=f"{sel_sym} — 90-day Price",
                            )
                            st.plotly_chart(fig_spark, use_container_width=True)
                    except Exception:
                        pass
