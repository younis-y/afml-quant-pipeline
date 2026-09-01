"""
AFML Quant Pipeline - Main Entry Point
==================================
Quantitative Research Platform backed by OpenBB v4.6.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


def main():
    """Print the help menu."""
    print("""
+---------------------------------------------------------------+
|                    AFML QUANT PIPELINE                        |
|          Quantitative Research Platform                       |
|               Powered by OpenBB v4.6                         |
+---------------------------------------------------------------+

Available Commands:
-------------------
  dashboard   Launch the 4-tab Streamlit dashboard
  trade       Start paper trading engine
  ingest      Ingest books into the RAG knowledge base
  sanity      Run strategy sanity check
  optimise    Run VectorBT parameter optimisation

Examples:
  python main.py dashboard
  python main.py ingest --books-dir references/
  python main.py optimise --ticker SPY --strategy ma
  python main.py sanity --strategy "Triple Barrier Method"
""")


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess

    dashboard_path = PROJECT_ROOT / "ui" / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def run_paper_trading():
    """Start the paper trading engine."""
    from pipeline import run_paper_trading

    run_paper_trading(ticker="SPY", duration_minutes=60, confidence_threshold=0.75)


def ingest_books(books_dir: str = None):
    """Ingest books into the RAG knowledge base."""
    from analysis_engine.knowledge import create_knowledge_base

    if books_dir is None:
        books_dir = str(PROJECT_ROOT / "references")

    try:
        kb = create_knowledge_base(books_dir)
    except ImportError as exc:
        raise SystemExit(
            "The RAG ingest path needs the optional extras: "
            f"pip install -r requirements-extras.txt ({exc})"
        )
    print(f"Knowledge base created with books from {books_dir}")
    return kb


def run_optimiser(ticker: str = "SPY", strategy: str = "ma") -> None:
    """Run VectorBT parameter optimisation and print results."""
    from pipeline.optimiser import StrategyOptimiser
    from utils.obb_client import get_obb_client  # noqa: F401 — verify client available

    opt = StrategyOptimiser()
    strat_map = {
        "ma":        opt.optimize_ma_crossover,
        "rsi":       opt.optimize_rsi,
        "bollinger": opt.optimize_bollinger,
    }

    func = strat_map.get(strategy.lower())
    if func is None:
        print(f"Unknown strategy '{strategy}'. Choose from: {list(strat_map.keys())}")
        return

    print(f"\nOptimising {strategy.upper()} strategy for {ticker}...")
    result = func(ticker)

    print("\n" + "=" * 50)
    print("OPTIMISATION RESULTS")
    print("=" * 50)
    print(f"Best parameters : {result.best_params}")
    print(f"Best Sharpe     : {result.best_sharpe:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AFML Quant Pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="menu",
        choices=["menu", "dashboard", "trade", "ingest", "sanity", "optimise"],
        help="Command to run",
    )
    parser.add_argument("--books-dir", type=str, help="Directory containing books for RAG")
    parser.add_argument("--strategy", type=str, help="Strategy name for sanity check or optimise")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker for optimise command")

    args = parser.parse_args()

    if args.command in ("menu", None):
        main()
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "trade":
        run_paper_trading()
    elif args.command == "ingest":
        books_dir = args.books_dir or str(PROJECT_ROOT / "references")
        ingest_books(books_dir)
    elif args.command == "sanity":
        try:
            from utils import sanity_check_strategy, format_sanity_report
        except ImportError:
            raise SystemExit(
                "The sanity check needs the optional extras: "
                "pip install -r requirements-extras.txt"
            )

        strategy = args.strategy or "Triple Barrier Method"
        result = sanity_check_strategy(strategy)
        print(format_sanity_report(result))
    elif args.command == "optimise":
        run_optimiser(ticker=args.ticker, strategy=args.strategy or "ma")
