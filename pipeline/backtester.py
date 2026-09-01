"""
AFML Quant Pipeline - Backtesting Engine
Walk-forward backtester with position tracking, P&L, and performance metrics.

Features:
  - Walk-forward and expanding window modes
  - Configurable transaction costs and slippage
  - Full trade log with entry/exit prices and reasons
  - Comprehensive performance metrics (Sharpe, Sortino, Calmar, Max DD, etc.)
  - Integration with the existing pipeline signal generators

Reference: Advances in Financial Machine Learning, Chapters 10-11
"""

from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Optional QuantStats import — gracefully degrade if not installed.
try:
    import quantstats as qs
    _QS_AVAILABLE = True
except Exception:
    _QS_AVAILABLE = False

# Default location for QuantStats tearsheets. Nothing is created at import
# time; the directory is made only when a tearsheet is actually requested.
_TEARSHEET_DIR = Path(__file__).parent.parent / "reports" / "tearsheets"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Record of a single completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float              # absolute P&L after costs
    pnl_pct: float          # percentage return
    cost: float             # total transaction costs
    bars_held: int
    exit_reason: str        # 'signal', 'stop_loss', 'take_profit', 'timeout'

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class Position:
    """Open position state."""
    direction: int          # 1 = long, -1 = short
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    bars_held: int = 0

    def mark_to_market(self, current_price: float) -> float:
        """Calculate unrealised P&L."""
        return self.direction * (current_price - self.entry_price) * self.quantity


@dataclass
class BacktestConfig:
    """Configuration for the backtester."""
    # Capital
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10     # fraction of equity per trade

    # Costs
    commission_pct: float = 0.001       # 10 bps per side
    slippage_pct: float = 0.0005        # 5 bps

    # Risk
    stop_loss_pct: Optional[float] = 0.02   # 2 % stop loss
    take_profit_pct: Optional[float] = 0.04  # 4 % take profit
    max_bars_held: Optional[int] = None      # time-based exit

    # Walk-forward
    train_window: int = 500             # bars for training window
    test_window: int = 100              # bars for testing window
    expanding: bool = False             # True = expanding, False = rolling

    # Misc
    allow_short: bool = True

    # Reporting. Off by default: writing a tearsheet creates files under
    # reports/ and, if a benchmark is named, fetches it over the network.
    tearsheet: bool = False
    tearsheet_benchmark: Optional[str] = None


@dataclass
class BacktestResult:
    """Full results of a backtest run."""
    # Equity curve
    equity_curve: pd.Series
    returns: pd.Series
    drawdown: pd.Series

    # Trade log
    trades: List[Trade]

    # Summary metrics
    metrics: Dict[str, float]

    # Configuration used
    config: BacktestConfig

    # Per-period signals
    signals: pd.DataFrame

    # QuantStats extras (populated after run if qs is available)
    qs_metrics: Dict[str, float] = field(default_factory=dict)
    tearsheet_path: Optional[str] = None

    def get_quantstats_metrics(self) -> Dict[str, float]:
        """
        Return QuantStats key metrics computed from the returns series.

        Keys: sharpe, sortino, calmar, max_drawdown, win_rate, cagr, var.
        Falls back to metrics already stored in self.qs_metrics or
        the internal metrics dict if QuantStats is unavailable.
        """
        if self.qs_metrics:
            return self.qs_metrics

        if not _QS_AVAILABLE or len(self.returns) < 2:
            # Derive what we can from the existing metrics dict
            m = self.metrics
            return {
                "sharpe": m.get("sharpe_ratio", 0.0),
                "sortino": m.get("sortino_ratio", 0.0),
                "calmar": m.get("calmar_ratio", 0.0),
                "max_drawdown": m.get("max_drawdown_pct", 0.0) / 100,
                "win_rate": m.get("win_rate", 0.0) / 100,
                "cagr": m.get("annual_return_pct", 0.0) / 100,
                "var": m.get("var_95", 0.0),
            }

        returns = self.returns.dropna()
        result: Dict[str, float] = {}

        def _safe(func, *args, **kwargs):
            try:
                val = func(*args, **kwargs)
                return float(val) if val is not None and not (isinstance(val, float) and (val != val)) else 0.0
            except Exception:
                return 0.0

        result["sharpe"] = _safe(qs.stats.sharpe, returns)
        result["sortino"] = _safe(qs.stats.sortino, returns)
        result["calmar"] = _safe(qs.stats.calmar, returns)
        result["max_drawdown"] = _safe(qs.stats.max_drawdown, returns)
        result["win_rate"] = _safe(qs.stats.win_rate, returns)
        result["cagr"] = _safe(qs.stats.cagr, returns)
        result["var"] = _safe(qs.stats.var, returns)

        self.qs_metrics = result
        return result

    def summary(self) -> str:
        """Human-readable summary report."""
        m = self.metrics
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            "RETURNS",
            "-" * 40,
            f"  Total Return:        {m.get('total_return_pct', 0):>10.2f} %",
            f"  Annual Return:       {m.get('annual_return_pct', 0):>10.2f} %",
            f"  Sharpe Ratio:        {m.get('sharpe_ratio', 0):>10.3f}",
            f"  Sortino Ratio:       {m.get('sortino_ratio', 0):>10.3f}",
            f"  Calmar Ratio:        {m.get('calmar_ratio', 0):>10.3f}",
            "",
            "RISK",
            "-" * 40,
            f"  Max Drawdown:        {m.get('max_drawdown_pct', 0):>10.2f} %",
            f"  Max DD Duration:     {m.get('max_dd_duration', 0):>10.0f} bars",
            f"  Annual Volatility:   {m.get('annual_volatility', 0):>10.2f} %",
            f"  VaR (95%):           {m.get('var_95', 0):>10.4f}",
            "",
            "TRADES",
            "-" * 40,
            f"  Total Trades:        {m.get('total_trades', 0):>10.0f}",
            f"  Win Rate:            {m.get('win_rate', 0):>10.2f} %",
            f"  Profit Factor:       {m.get('profit_factor', 0):>10.2f}",
            f"  Avg Win / Avg Loss:  {m.get('avg_win_loss_ratio', 0):>10.2f}",
            f"  Avg Bars Held:       {m.get('avg_bars_held', 0):>10.1f}",
            f"  Total Costs:         ${m.get('total_costs', 0):>9.2f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_backtest_metrics(
    equity: pd.Series,
    trades: List[Trade],
    config: BacktestConfig,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics from an equity curve and trade log.

    Args:
        equity: Equity curve (index = timestamps, values = portfolio value)
        trades: List of completed Trade objects
        config: Backtest configuration
        periods_per_year: Annualisation factor

    Returns:
        Dict of metric name → value
    """
    returns = equity.pct_change().dropna()

    # ── Return metrics ───────────────────────────────────────────────────
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_periods = len(returns)
    annual_factor = periods_per_year / max(n_periods, 1)
    annual_return = (1 + total_return) ** annual_factor - 1
    annual_vol = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0

    # ── Sharpe ───────────────────────────────────────────────────────────
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0

    # ── Sortino ──────────────────────────────────────────────────────────
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-10
    sortino = (mean_ret / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    # ── Drawdown ─────────────────────────────────────────────────────────
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # Max drawdown duration (bars)
    in_dd = (drawdown < 0).astype(int)
    dd_groups = in_dd.ne(in_dd.shift()).cumsum()
    dd_groups = dd_groups[in_dd == 1]
    max_dd_duration = dd_groups.value_counts().max() if len(dd_groups) > 0 else 0

    # ── Calmar ───────────────────────────────────────────────────────────
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # ── VaR ──────────────────────────────────────────────────────────────
    var_95 = returns.quantile(0.05) if len(returns) > 0 else 0.0

    # ── Trade metrics ────────────────────────────────────────────────────
    n_trades = len(trades)
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0.0
    avg_loss = abs(np.mean([t.pnl_pct for t in losers])) if losers else 1e-10
    avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    avg_bars = np.mean([t.bars_held for t in trades]) if trades else 0.0
    total_costs = sum(t.cost for t in trades)

    return {
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_dd * 100,
        "max_dd_duration": max_dd_duration,
        "annual_volatility": annual_vol * 100,
        "var_95": var_95,
        "total_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "avg_bars_held": avg_bars,
        "total_costs": total_costs,
        "total_winners": len(winners),
        "total_losers": len(losers),
    }


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """
    Walk-forward backtester for signal-based strategies.

    Usage:
        bt = Backtester(config)
        result = bt.run(prices, signal_func)

    The ``signal_func`` receives a price slice and must return a pandas
    Series with values in {-1, 0, 1} indicating SHORT, FLAT, LONG.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    # ── Public API ───────────────────────────────────────────────────────

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        *,
        periods_per_year: int = 252,
    ) -> BacktestResult:
        """
        Run a backtest with pre-computed signals.

        Args:
            prices: Close price series (index = DatetimeIndex)
            signals: Signal series aligned to prices: 1=long, -1=short, 0=flat
            periods_per_year: Annualisation factor for metrics

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        cfg = self.config
        common = prices.index.intersection(signals.index)
        prices = prices.loc[common]
        signals = signals.loc[common]

        # State
        equity = cfg.initial_capital
        position: Optional[Position] = None
        equity_series = []
        trade_log: List[Trade] = []

        for i, ts in enumerate(prices.index):
            price = prices.iloc[i]
            signal = int(signals.iloc[i])

            # ── Check exit conditions on open position ───────────────
            if position is not None:
                position.bars_held += 1
                ret = position.direction * (price - position.entry_price) / position.entry_price

                exit_reason = None

                # Stop-loss
                if cfg.stop_loss_pct and ret <= -cfg.stop_loss_pct:
                    exit_reason = "stop_loss"
                # Take-profit
                elif cfg.take_profit_pct and ret >= cfg.take_profit_pct:
                    exit_reason = "take_profit"
                # Time-out
                elif cfg.max_bars_held and position.bars_held >= cfg.max_bars_held:
                    exit_reason = "timeout"
                # Signal reversal
                elif signal != 0 and signal != position.direction:
                    exit_reason = "signal"
                # Flat signal
                elif signal == 0 and position.direction != 0:
                    exit_reason = "signal"

                if exit_reason is not None:
                    trade = self._close_position(position, price, ts, exit_reason, cfg)
                    trade_log.append(trade)
                    equity += trade.pnl
                    position = None

            # ── Open new position ────────────────────────────────────
            if position is None and signal != 0:
                if signal == -1 and not cfg.allow_short:
                    pass  # skip shorts if disallowed
                else:
                    position = self._open_position(signal, price, ts, equity, cfg)

            equity_series.append({"timestamp": ts, "equity": equity + (
                position.mark_to_market(price) if position else 0.0
            )})

        # Close any remaining position at last price
        if position is not None:
            trade = self._close_position(
                position, prices.iloc[-1], prices.index[-1], "end_of_data", cfg
            )
            trade_log.append(trade)
            equity += trade.pnl

        # Build result objects
        eq_df = pd.DataFrame(equity_series).set_index("timestamp")["equity"]
        rets = eq_df.pct_change().dropna()
        cummax = eq_df.cummax()
        dd = (eq_df - cummax) / cummax

        metrics = compute_backtest_metrics(eq_df, trade_log, cfg, periods_per_year)

        signals_df = pd.DataFrame({"price": prices, "signal": signals})

        result = BacktestResult(
            equity_curve=eq_df,
            returns=rets,
            drawdown=dd,
            trades=trade_log,
            metrics=metrics,
            config=cfg,
            signals=signals_df,
        )

        # QuantStats tearsheet + metrics
        if _QS_AVAILABLE and cfg.tearsheet and len(rets) > 10:
            try:
                ticker = "portfolio"
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                _TEARSHEET_DIR.mkdir(parents=True, exist_ok=True)
                html_path = _TEARSHEET_DIR / f"{ticker}_{ts_str}.html"
                qs.reports.html(
                    rets,
                    benchmark=cfg.tearsheet_benchmark,
                    output=str(html_path),
                    title=f"AFML Quant Pipeline — {ticker}",
                    download_filename=str(html_path),
                )
                result.tearsheet_path = str(html_path)
            except Exception as exc:
                warnings.warn(f"QuantStats HTML tearsheet failed: {exc}")

            result.qs_metrics = result.get_quantstats_metrics()

        return result

    def walk_forward(
        self,
        prices: pd.Series,
        signal_func: Callable[[pd.Series], pd.Series],
        *,
        periods_per_year: int = 252,
    ) -> BacktestResult:
        """
        Walk-forward backtest: retrain the signal function on each window.

        Args:
            prices: Full price series
            signal_func: Callable(train_prices) -> signals for the NEXT
                         test_window bars.  The function receives the
                         training slice and must return a Series of signals
                         indexed on the SUBSEQUENT test window dates.
            periods_per_year: Annualisation factor

        Returns:
            BacktestResult aggregated across all walk-forward windows
        """
        cfg = self.config
        n = len(prices)
        all_signals = pd.Series(dtype=int)

        start = 0
        while True:
            if cfg.expanding:
                train_start = 0
            else:
                train_start = start

            train_end = train_start + cfg.train_window + (start if cfg.expanding else 0)
            test_end = train_end + cfg.test_window

            if train_end >= n:
                break
            if test_end > n:
                test_end = n

            train_slice = prices.iloc[train_start:train_end]
            test_slice = prices.iloc[train_end:test_end]

            if len(test_slice) == 0:
                break

            # Get signals for the test window
            try:
                window_signals = signal_func(train_slice)
                # Ensure signals are aligned to the test window
                if not window_signals.index.equals(test_slice.index):
                    window_signals = window_signals.reindex(test_slice.index, fill_value=0)
            except Exception:
                window_signals = pd.Series(0, index=test_slice.index)

            all_signals = pd.concat([all_signals, window_signals])

            start += cfg.test_window

        # Remove duplicates (in case of overlapping windows)
        all_signals = all_signals[~all_signals.index.duplicated(keep="first")]

        return self.run(prices, all_signals, periods_per_year=periods_per_year)

    # ── Private Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _open_position(
        direction: int,
        price: float,
        timestamp: pd.Timestamp,
        equity: float,
        cfg: BacktestConfig,
    ) -> Position:
        notional = equity * cfg.position_size_pct
        quantity = notional / price
        return Position(
            direction=direction,
            entry_time=timestamp,
            entry_price=price,
            quantity=quantity,
        )

    @staticmethod
    def _close_position(
        pos: Position,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str,
        cfg: BacktestConfig,
    ) -> Trade:
        gross_pnl = pos.direction * (exit_price - pos.entry_price) * pos.quantity
        cost = (
            pos.entry_price * pos.quantity * cfg.commission_pct
            + exit_price * pos.quantity * cfg.commission_pct
            + pos.entry_price * pos.quantity * cfg.slippage_pct
            + exit_price * pos.quantity * cfg.slippage_pct
        )
        net_pnl = gross_pnl - cost
        pnl_pct = pos.direction * (exit_price / pos.entry_price - 1) - (cost / (pos.entry_price * pos.quantity))

        return Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            cost=cost,
            bars_held=pos.bars_held,
            exit_reason=reason,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    commission_pct: float = 0.001,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> BacktestResult:
    """
    One-liner backtest for quick prototyping.

    Args:
        prices: Close price series
        signals: Signal series (1=long, -1=short, 0=flat)
        initial_capital: Starting capital
        commission_pct: Round-trip commission
        stop_loss_pct: Stop-loss percentage
        take_profit_pct: Take-profit percentage

    Returns:
        BacktestResult
    """
    cfg = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    bt = Backtester(cfg)
    return bt.run(prices, signals)


if __name__ == "__main__":
    print("Testing Backtesting Engine...")
    print("=" * 60)

    # Synthesise price data
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n) * 0.015 + 0.0003
    prices = pd.Series(
        100 * np.exp(np.cumsum(returns)),
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
        name="Close",
    )

    # Simple momentum signal: buy when 20-bar return > 0
    mom = prices.pct_change(20)
    signals = pd.Series(
        np.where(mom > 0, 1, np.where(mom < 0, -1, 0)),
        index=prices.index,
    )

    # Run backtest
    result = quick_backtest(prices, signals)

    print(result.summary())
    print(f"\nTrade log ({len(result.trades)} trades):")
    for t in result.trades[:5]:
        print(
            f"  {t.direction:+d}  {t.entry_time} → {t.exit_time}  "
            f"PnL: ${t.pnl:+.2f}  ({t.exit_reason})"
        )
