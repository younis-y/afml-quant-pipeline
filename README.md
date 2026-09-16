# AFML Quant Pipeline

[![ci](https://github.com/younis-y/afml-quant-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/younis-y/afml-quant-pipeline/actions/workflows/ci.yml)

A working implementation of the López de Prado *Advances in Financial Machine Learning* toolchain — dollar bars, fractional differentiation with a minimum-*d* stationarity search, triple-barrier labelling and meta-labelling, sequential-bootstrap sample weights, purged and combinatorial-purged cross-validation with embargo, probabilistic and deflated Sharpe ratios, and MDI/MDA/SFI feature importance — wired to an OpenBB v4.6 data layer, a walk-forward backtester and a Streamlit front end.

## The question

*Advances in Financial Machine Learning* is mostly read, rarely built. Its methods are described as code snippets scattered across the book with no common interface: the sample-weight scheme in Chapter 4 assumes the label structure of Chapter 3, which assumes the bar structure of Chapter 2, which assumes tick or volume data that most retail sources do not supply. The question this repository answers is a narrow engineering one — **can the chain be assembled end to end, against a free data source, with each link tested in isolation?**

It can. What follows is the assembled chain, the numbers that were actually measured while assembling it, and a Scope section stating what it does not do.

## Why the restraint is the point

This is a **library of the book's methods with a test suite around them**, and it
deliberately publishes no headline Sharpe ratio, no equity curve, and no backtest
offered as evidence of edge. Every number below is a property of the code,
measured by running it.

That is not a gap — it is the argument of the chapters being implemented.
Chapters 11–15 are a sustained case that a backtest is a research tool almost
always misread as a discovery, and that a Sharpe ratio quoted without a trial
count and a track-record length is not a number at all. The repository ships
`deflated_sharpe` and `probabilistic_sharpe` precisely so that claim can be made
rigorously; leading with an unadjusted backtest result would refute the very
machinery on display.

The backtester exists so the labelling and validation chain has somewhere to
terminate. What is demonstrated here is that the chain runs end to end, against a
free data source, with each link independently tested.

## Measured properties

Run on Python 3.13.5, pandas 2.3.3, NumPy 2.2.6, scikit-learn 1.6.1, pytest 8.3.4.

| | |
|---|---|
| Test modules | 18 |
| Tests collected | 183 |
| Tests passing, default run | **181** |
| Tests deselected by default | 2, marked `network` |
| Wall time, default suite | 136 s on the machine above |
| Same suite with the optional extras absent | 183 collected, 177 passing |
| Python modules (excluding tests) | 39 |

```bash
python -m pytest -q
# 181 passed, 2 deselected
```

Two tests reach the network, and both are marked `network` and deselected by default, so a
clean checkout runs green without connectivity: `tests/test_api.py::TestDashboardEndpoint::test_dashboard_returns_200`
and `::TestSymbolEndpoint::test_resolve_symbol`, which hit OpenBB and Yahoo's symbol search.
Run them with `python -m pytest -m network` if you want them; they are the only two tests here
that can fail for reasons outside this code.

Three further tests are marked `slow` for cost rather than connectivity: a 50,000-path bootstrap
in `test_stats_extended.py::TestBootstrapVaR::test_large_simulations`, plus
`test_stats.py::TestMasterAnalysis::test_analyze_series_structure` and
`test_validation_extended.py::TestValidateStrategy::test_end_to_end`, which fit a RandomForest on
seeded synthetic data. They pass offline in about 22 s and run by default. `tests/test_api.py` is
marked `integration` at module level because the module needs `fastapi` and `httpx` from the
optional `api` extra; without it the whole module skips, which is where the older figure of 177
passing comes from.

## Reproducing

```bash
git clone <this repo> && cd afml-quant-pipeline
pip install -e .
python -m pytest -q
```

`pip install -e .` covers the pipeline, the data layer, the backtester, the Streamlit dashboard
and the default test suite, `pytest` included, so the three lines above run end to end on a clean
machine. The REST and knowledge layers are optional extras, `pip install -e ".[api]"` and
`pip install -e ".[knowledge]"`, and nothing in the core path needs them, which was checked rather
than assumed. With `chromadb`, `langchain-google-genai`, `langchain-community`, `vertexai`,
`google-generativeai`, `ebooklib`, `PyPDF2`, `fastapi` and `uvicorn` all blocked at import time,
`import analysis_engine` still succeeds and the suite still reports 177 passing.
`analysis_engine/knowledge.py` imports cleanly without any of them; the classes that need one
raise `ImportError` on construction, naming the package to install.

No API key is required for anything above. Copy `.env.example` to `.env` only if you want the optional layers — see [Configuration](#configuration).

### A worked example

The chain from raw prices to purged folds, on a synthetic driftless random walk so that it runs offline and deterministically:

```python
import numpy as np, pandas as pd
from pipeline.data_processor import find_min_d_for_stationarity, frac_diff_fixed
from pipeline.labeling_advanced import triple_barrier_labels
from pipeline.sample_weights import get_average_uniqueness, PurgedKFold

rng = np.random.default_rng(0)
idx = pd.date_range("2015-01-01", periods=2000, freq="B")
close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)

d, table = find_min_d_for_stationarity(close, max_d=1.0, step=0.05)
print("minimum d passing ADF at 5%:", d)                       # 0.55

labels = triple_barrier_labels(close, profit_mult=2.0, stop_mult=1.0, num_bars=20)
print(len(labels), labels["label"].value_counts().to_dict())
# 1960 {-1.0: 1248, 1.0: 707, 0.0: 5}

u = get_average_uniqueness(labels["t1"], close)
print("mean average uniqueness:", round(float(u.mean()), 3))    # 0.048

X = pd.DataFrame({"fd": frac_diff_fixed(close, d=d)}).reindex(labels.index).ffill().fillna(0.0)
cv = PurgedKFold(n_splits=5, t1=labels["t1"], pct_embargo=0.01)
print("purged folds:", sum(1 for _ in cv.split(X)))             # 5
```

Two things in that output are worth reading, and neither is a finding about markets:

- The label distribution is skewed towards −1 because the barriers are asymmetric (2σ profit, 1σ stop) on a series with no drift. Set `profit_mult=1.0` and it moves to 1023/937. The skew is arithmetic, not signal.
- Mean average uniqueness of 0.048 on 20-bar horizons is the concurrency problem the book's Chapter 4 exists to address: with overlapping labels, the 1960 observations carry roughly the independent information of 94 (1960 x 0.048 — a rule of thumb read off the two measured numbers above, not an output of any function here). This is why the sample weights and the sequential bootstrap are there.

### Running the dashboard

```bash
python main.py dashboard        # Streamlit, four tabs: Forecast / Fundamentals / Backtest / Market Scanner
```

This one does hit the network, through OpenBB.

## Method map

Chapter numbers refer to *Advances in Financial Machine Learning* (Wiley, 2018) unless marked otherwise.

| Method | Chapter | Location |
|---|---|---|
| Dollar bars | 2 | `pipeline/data_processor.py::dollar_bars` |
| Symmetric CUSUM filter | 2 | `pipeline/labeling_advanced.py::cusum_filter`, `pipeline/feature_engineering.py::get_cusum_filter` |
| Triple-barrier labelling | 3 | `pipeline/labeling.py`, `pipeline/labeling_advanced.py` |
| Vertical barrier, target volatility | 3 | `add_vertical_barrier`, `get_daily_volatility`, `get_parkinson_volatility` |
| Meta-labelling | 3 | `pipeline/labeling.py::get_meta_labels`, `pipeline/meta_model.py`, `pipeline/models.py::MetaLabelingModel` |
| Indicator matrix, concurrency | 4 | `pipeline/sample_weights.py::get_indicator_matrix`, `get_num_concurrent_labels` |
| Average uniqueness | 4, snippet 4.4 | `pipeline/sample_weights.py::get_average_uniqueness` |
| Sequential bootstrap | 4 | `pipeline/sample_weights.py::sequential_bootstrap`, `sequential_bootstrap_fast` |
| Return-attributed and time-decay weights | 4 | `get_sample_weights_by_return`, `get_sample_weights_by_time_decay`, `get_combined_sample_weights` |
| Fractional differentiation, fixed-width window | 5 | `pipeline/data_processor.py::get_weights_ffd`, `frac_diff_fixed`; `pipeline/feature_engineering.py::frac_diff_ffd` |
| Minimum-*d* stationarity search | 5 | `pipeline/data_processor.py::find_min_d_for_stationarity`, `compute_correlation_with_original`; `pipeline/feature_engineering.py::find_optimal_d` |
| Bagged / ensemble classifiers | 6 | `pipeline/models.py::FinancialRandomForest`, `FinancialGradientBoosting`, `FinancialEnsemble` |
| Purged K-fold with embargo | 7, snippet 7.3 | `pipeline/sample_weights.py::PurgedKFold` |
| MDI, MDA, single-feature importance | 8 | `pipeline/models.py::mean_decrease_impurity`, `mean_decrease_accuracy`, `single_feature_importance` |
| Orthogonal features (PCA) | 8 | `pipeline/feature_engineering.py::get_orthogonal_features` |
| Combinatorial purged CV | 12 | `pipeline/sample_weights.py::CombinatorialPurgedKFold` |
| Probabilistic Sharpe ratio | 14 | `pipeline/models.py::probabilistic_sharpe_ratio` |
| Deflated Sharpe ratio | 14 | `pipeline/models.py::deflated_sharpe_ratio`, `utils/validation.py::deflated_sharpe_ratio` |
| SADF structural-break test | 17 | `pipeline/feature_engineering.py::get_sadf` |
| Shannon and Lempel-Ziv entropy features | 18 | `get_shannon_entropy`, `get_lempel_ziv_entropy` |
| Roll spread, Kyle λ, Amihud λ, VPIN | 19 | `pipeline/feature_engineering.py` |
| Trend-scanning labels | *MLAM* 5 | `pipeline/labeling_advanced.py::trend_scanning_labels`, `pipeline/feature_engineering.py::get_trend_scanning_labels` |

*MLAM* = *Machine Learning for Asset Managers* (Cambridge, 2020).

Supporting statistics outside the book — ADF, Hurst exponent, parametric / Cornish-Fisher / historical / bootstrap VaR and CVaR, GARCH fitting and GARCH VaR, fat-tail diagnostics — are in `analysis_engine/stats.py`.

## Data layer

`utils/obb_client.py` is a singleton wrapper over OpenBB v4.6 and the only place in the repository that calls `from openbb import obb`. `utils/data_provider.py` is a thin compatibility shim over it, and the paths this README documents — `pipeline/data_processor.py`, `pipeline/feature_engineering.py`, `pipeline/forecaster.py`, `pipeline/optimiser.py`, `analysis_engine/core.py`, `ui/dashboard.py`, `scripts/00_data_download.py` — all go through one or the other.

Four files predate the client and still call a vendor directly, which is a gap rather than a design: `pipeline/pair_trading.py`, `scripts/scan_market.py` and `scripts/research_trade.py` call `yfinance.download`, and `pipeline/symbol_resolver.py` — reachable from `api/server.py` — hits Yahoo's search endpoint with `requests`. They have not been migrated.

Fourteen methods, all against endpoints available on the free tier:

| Area | Methods |
|---|---|
| Prices | `get_price_history`, `get_quote` |
| Fundamentals | `get_fundamentals_metrics`, `get_income_statement`, `get_balance_sheet`, `get_cash_flow`, `get_dividends` |
| Estimates and profile | `get_analyst_consensus`, `get_equity_profile` |
| News and discovery | `get_company_news`, `get_market_movers` |
| Other instruments | `get_etf_info`, `get_crypto_history`, `get_index_history` |

Every response passes through `_to_df`, and the three price-series endpoints (`get_price_history`, `get_crypto_history`, `get_index_history`) additionally through `_normalise_ohlcv`, so column names and index types are consistent regardless of which provider OpenBB routes to. Failures raise `OBBClientError` rather than returning an empty frame — a silent empty frame downstream of a labelling step is very hard to notice.

The important limitation: the free tier serves **daily OHLCV, not ticks**. Dollar bars in `pipeline/data_processor.py` are therefore constructed from daily bars using close × volume as the dollar-value proxy. That is a legitimate approximation of the aggregation but it is not the tick-level construction the book describes, and the information-driven bars of Chapter 2 (imbalance and run bars) are not implemented at all, because daily data cannot support them.

## Layout

```
pipeline/            The AFML toolchain
  data_processor.py      dollar bars, FFD weights, minimum-d search
  labeling.py            triple barrier, meta-labels
  labeling_advanced.py   + CUSUM, trend scanning, Parkinson volatility
  sample_weights.py      uniqueness, sequential bootstrap, PurgedKFold, CPCV
  feature_engineering.py microstructure, entropy, SADF, orthogonal features
  models.py              classifiers, MDI/MDA/SFI, PSR, DSR
  meta_model.py          primary signal -> meta-label pipeline
  primary_models.py      RSI, Bollinger, MACD, momentum, SMA (pluggable)
  backtester.py          event-driven backtest + walk-forward
  optimiser.py           VectorBT parameter sweeps
  advanced_pipeline.py   orchestration of the above
  forecaster.py          ARIMA / ETS / GARCH / Prophet / ML ensemble
  pair_trading.py        cointegration, mean reversion
  live_engine.py         paper-trading loop
  technical_indicators.py, symbol_resolver.py
analysis_engine/     ADF, Hurst, VaR suite, GARCH, fat tails; RAG knowledge base
utils/               OpenBB client, validation, TTL cache, sanity check
ui/dashboard.py      Streamlit front end
api/server.py        FastAPI REST layer (optional)
scripts/             download, training, scanning, explanation helpers
tests/               18 modules, 183 tests
```

## Configuration

Copy `.env.example` to `.env`. Every variable is optional and none is needed for the pipeline, the data layer, the backtester, the dashboard or the tests.

| Variable | Used by | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini embeddings in `analysis_engine/knowledge.py`, `utils/sanity_check.py` | none |
| `GCP_PROJECT_ID` | Vertex AI embedding path, `scripts/check_llm.py`, `trade.sh` | none. `scripts/check_llm.py` raises `SystemExit` and `trade.sh` exits 1 when it is unset; `analysis_engine/knowledge.py` does not — it skips the Vertex branch silently and falls back to ChromaDB's default ONNX embeddings |
| `GCP_LOCATION` | as above | `us-central1` |

## Scope

What this repository does **not** do, stated plainly so that nobody has to discover it by reading the source:

- **No strategy, no edge, no result.** Nothing here has been validated as profitable and nothing is presented as such. Generated tearsheets and training reports are git-ignored; the backtester writes a QuantStats tearsheet only when `BacktestConfig(tearsheet=True)` is set explicitly.
- **No tick data, so no information-driven bars.** Dollar bars are approximated from daily OHLCV as described above. Tick, volume-imbalance and run bars (Ch. 2) are absent.
- **No live execution.** `pipeline/live_engine.py` is a paper-trading loop. There is no broker integration, no order-management system and no position reconciliation.
- **Duplicate implementations exist.** `PurgedKFold` appears twice — the `pipeline/sample_weights.py` version purges on label end times `t1` as the book specifies and is sklearn-compatible; the `utils/validation.py` version purges on index distance, trains only on data preceding the test fold, and is standalone. Fractional differentiation, triple-barrier labelling and the deflated Sharpe ratio are likewise implemented in two places. These grew from separate work sessions and have not been consolidated. The `pipeline/` versions are the ones to read.
- **The data layer is not universal.** `utils/obb_client.py` is the OpenBB entry point, but four modules bypass it and call yfinance or Yahoo's search endpoint directly, as listed under [Data layer](#data-layer). Nothing routes their requests through the client's error contract or its `_normalise_ohlcv` step.
- **Backtest costs are simplistic.** Fixed percentage commission and slippage. No market-impact model, no borrow cost, no partial fills.
- **`CombinatorialPurgedKFold` is implemented but not integrated, and its purging is too coarse.** It generates the C(N,k) splits and the corresponding backtest paths, but nothing in the pipeline consumes the path distribution to compute a probability of backtest overfitting (Ch. 11-12), which is the point of having it. Worse, its `_purge()` works on the `[min, max]` span of the test set rather than on each contiguous test block, so a combination whose test groups are not adjacent purges every training observation lying between them. At the pipeline defaults (`n_groups=10`, `k_test_groups=2`) that is 45 splits, of which one is left with an empty training set and the worst non-empty ones keep only about an eighth of the rows lying outside the test folds. `AdvancedPipeline` therefore builds the object for inspection and prints its split and path counts, but scores with `PurgedKFold`; the printed line says so. Fixing the purge properly is a change to the CPCV semantics, not a typo, and has not been made.
- **Statistical tests are not benchmarked against a reference implementation.** The AFML routines are tested for shape, sign, monotonicity and known edge cases, not against published numerical fixtures from `mlfinlab` or the book's own examples. They are unit-tested, not validated.
- **The RAG and REST layers are peripheral.** `analysis_engine/knowledge.py` and `api/server.py` are optional extras from an earlier design and are not part of the AFML argument. Two rough edges survive in the knowledge base and are called out here rather than left to be discovered: `KnowledgeBase.__init__` `chdir`s to `/` and back around ChromaDB's `Settings()` construction, to stop pydantic reading a `.env` file in the working directory; and `use_local_cache_dirs()` rewrites `HOME` and `XDG_CACHE_HOME` to point at `./.cache`, for machines where the system cache directory is locked. That function is opt-in and is never called on import — importing this package changes nothing in your environment. `api/server.py` serves JSON only; the repository ships no frontend, so its CORS origins are simply the usual localhost dev ports.

## Licence

MIT. See [LICENSE](LICENSE).
