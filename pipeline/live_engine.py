"""
AFML Quant Pipeline - Live Trading Engine
Real-time paper trading loop with signal filtering
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import threading
import logging

import pandas as pd
import numpy as np

from .data_processor import fetch_data, dollar_bars, frac_diff_fixed
from .meta_model import MetaLabelingPipeline, SMACrossover

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LiveEngine')


@dataclass
class TradeSignal:
    """Represents a trading signal from the engine."""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'PASS'
    confidence: float
    price: float
    volatility: float
    primary_signal: int
    reason: str


@dataclass
class EngineState:
    """Current state of the trading engine."""
    is_running: bool
    last_update: datetime
    current_price: float
    current_volatility: float
    current_confidence: float
    signals_generated: int
    trades_executed: int
    position: int  # 1 = long, -1 = short, 0 = flat


class LiveEngine:
    """
    Real-time paper trading engine.
    
    Fetches data at regular intervals, processes through the pipeline,
    and generates filtered trading signals.
    """
    
    def __init__(
        self,
        ticker: str = "SPY",
        interval_seconds: int = 60,
        dollar_threshold: float = 500_000,
        frac_d: float = 0.4,
        confidence_threshold: float = 0.75,
        lookback_period: str = "7d",
        pipeline: Optional[MetaLabelingPipeline] = None
    ):
        self.ticker = ticker
        self.interval_seconds = interval_seconds
        self.dollar_threshold = dollar_threshold
        self.frac_d = frac_d
        self.confidence_threshold = confidence_threshold
        self.lookback_period = lookback_period
        
        # Pipeline (train on first run if not provided)
        self.pipeline = pipeline
        
        # Engine state
        self.state = EngineState(
            is_running=False,
            last_update=datetime.now(),
            current_price=0.0,
            current_volatility=0.0,
            current_confidence=0.0,
            signals_generated=0,
            trades_executed=0,
            position=0
        )
        
        # Data buffers
        self._raw_data = None
        self._dollar_bars = None
        self._features = None
        
        # Signal history
        self.signal_history = []
        
        # Callbacks
        self._on_signal: Optional[Callable[[TradeSignal], None]] = None
        self._on_update: Optional[Callable[[EngineState], None]] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def on_signal(self, callback: Callable[[TradeSignal], None]):
        """Register callback for new signals."""
        self._on_signal = callback
    
    def on_update(self, callback: Callable[[EngineState], None]):
        """Register callback for state updates."""
        self._on_update = callback
    
    def _fetch_and_process(self) -> pd.DataFrame:
        """Fetch latest data and process through pipeline."""
        # Fetch raw data
        self._raw_data = fetch_data(
            self.ticker,
            period=self.lookback_period,
            interval="1m"
        )
        
        if len(self._raw_data) < 50:
            logger.warning(f"Insufficient data: {len(self._raw_data)} bars")
            return pd.DataFrame()
        
        # Convert to dollar bars
        self._dollar_bars = dollar_bars(
            self._raw_data,
            dollar_threshold=self.dollar_threshold
        )
        
        if len(self._dollar_bars) < 20:
            logger.warning(f"Insufficient dollar bars: {len(self._dollar_bars)}")
            return pd.DataFrame()
        
        # Apply fractional differentiation
        frac_diff = frac_diff_fixed(
            self._dollar_bars['Close'],
            d=self.frac_d
        )
        
        # Combine into features
        features = pd.DataFrame(index=self._dollar_bars.index)
        features['close'] = self._dollar_bars['Close']
        features['frac_diff'] = frac_diff
        features['volume'] = self._dollar_bars['Volume']
        
        self._features = features.dropna()
        
        return self._features
    
    def _train_pipeline(self):
        """Train the pipeline on historical data."""
        logger.info("Training meta-labeling pipeline...")
        
        self.pipeline = MetaLabelingPipeline(
            sma_fast=50,
            sma_slow=200,
            profit_mult=2.0,
            stop_mult=1.0,
            num_bars=50,
            confidence_threshold=self.confidence_threshold
        )
        
        close = self._dollar_bars['Close']
        frac_diff = frac_diff_fixed(close, d=self.frac_d)
        
        results = self.pipeline.train(close, frac_diff)
        
        logger.info(f"Pipeline trained - F1: {results['metrics']['f1']:.4f}")
    
    def _generate_signal(self) -> Optional[TradeSignal]:
        """Generate trading signal from current data."""
        if self._features is None or len(self._features) < 2:
            return None
        
        # Get latest data point
        latest = self._features.iloc[-1]
        close = self._dollar_bars['Close']
        
        # Get prediction
        predictions = self.pipeline.predict(
            close,
            self._features['frac_diff'] if 'frac_diff' in self._features else None
        )
        
        if len(predictions) == 0:
            return None
        
        latest_pred = predictions.iloc[-1]
        
        # Calculate volatility
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        
        # Update state
        self.state.current_price = float(latest['close'])
        self.state.current_volatility = float(volatility) if not np.isnan(volatility) else 0.0
        self.state.current_confidence = float(latest_pred['confidence']) if not np.isnan(latest_pred['confidence']) else 0.0
        self.state.last_update = datetime.now()
        
        # Create signal
        action = latest_pred['action']
        
        reason = ""
        if action == 'PASS':
            reason = f"Confidence {self.state.current_confidence:.2f} < threshold {self.confidence_threshold}"
        else:
            reason = f"Confidence {self.state.current_confidence:.2f} >= threshold {self.confidence_threshold}"
        
        signal = TradeSignal(
            timestamp=datetime.now(),
            action=action,
            confidence=self.state.current_confidence,
            price=self.state.current_price,
            volatility=self.state.current_volatility,
            primary_signal=int(latest_pred['primary_signal']),
            reason=reason
        )
        
        return signal
    
    def _process_signal(self, signal: TradeSignal):
        """Process and log a signal."""
        self.signal_history.append(signal)
        self.state.signals_generated += 1
        
        if signal.action in ['BUY', 'SELL']:
            self.state.trades_executed += 1
            self.state.position = 1 if signal.action == 'BUY' else -1
            
            logger.info(
                f"{signal.action} @ ${signal.price:.2f} "
                f"(confidence: {signal.confidence:.2f})"
            )
        else:
            logger.debug(f"PASS @ ${signal.price:.2f} - {signal.reason}")
        
        # Trigger callback
        if self._on_signal:
            self._on_signal(signal)
    
    def _run_loop(self):
        """Main trading loop."""
        logger.info(f"Starting live engine for {self.ticker}...")
        
        # Initial data fetch and training
        self._fetch_and_process()
        if self.pipeline is None:
            self._train_pipeline()
        
        self.state.is_running = True
        
        while not self._stop_event.is_set():
            try:
                # Fetch and process data
                self._fetch_and_process()
                
                # Generate signal
                signal = self._generate_signal()
                
                if signal:
                    self._process_signal(signal)
                
                # Trigger update callback
                if self._on_update:
                    self._on_update(self.state)
                
                # Wait for next interval
                self._stop_event.wait(timeout=self.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                time.sleep(5)  # Brief pause on error
        
        self.state.is_running = False
        logger.info("Engine stopped.")
    
    def start(self, blocking: bool = False):
        """
        Start the trading engine.
        
        Args:
            blocking: If True, run in main thread (blocking).
                      If False, run in background thread.
        """
        if self.state.is_running:
            logger.warning("Engine already running")
            return
        
        self._stop_event.clear()
        
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Engine started in background")
    
    def stop(self):
        """Stop the trading engine."""
        if not self.state.is_running:
            return
        
        logger.info("Stopping engine...")
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5)
    
    def get_latest_signal(self) -> Optional[TradeSignal]:
        """Get the most recent signal."""
        if self.signal_history:
            return self.signal_history[-1]
        return None
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of signals generated."""
        if not self.signal_history:
            return {'total': 0}
        
        actions = [s.action for s in self.signal_history]
        
        return {
            'total': len(self.signal_history),
            'buys': actions.count('BUY'),
            'sells': actions.count('SELL'),
            'passes': actions.count('PASS'),
            'avg_confidence': np.mean([s.confidence for s in self.signal_history]),
            'latest_action': actions[-1],
            'latest_price': self.signal_history[-1].price
        }


def run_paper_trading(
    ticker: str = "SPY",
    duration_minutes: int = 60,
    confidence_threshold: float = 0.75
):
    """
    Convenience function to run paper trading.
    
    Args:
        ticker: Stock symbol
        duration_minutes: How long to run
        confidence_threshold: Minimum confidence for trades
    """
    engine = LiveEngine(
        ticker=ticker,
        confidence_threshold=confidence_threshold
    )
    
    # Define callbacks
    def on_signal(signal: TradeSignal):
        if signal.action != 'PASS':
            print(f"\n{'='*50}")
            print(f"SIGNAL: {signal.action}")
            print(f"   Price: ${signal.price:.2f}")
            print(f"   Confidence: {signal.confidence:.2%}")
            print(f"   Volatility: {signal.volatility:.4f}")
            print(f"   Reason: {signal.reason}")
            print(f"{'='*50}\n")
    
    def on_update(state: EngineState):
        print(
            f"[{state.last_update.strftime('%H:%M:%S')}] "
            f"Price: ${state.current_price:.2f} | "
            f"Vol: {state.current_volatility:.4f} | "
            f"Signals: {state.signals_generated} | "
            f"Trades: {state.trades_executed}"
        )
    
    engine.on_signal(on_signal)
    engine.on_update(on_update)
    
    print(f"Starting paper trading for {ticker}...")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Confidence threshold: {confidence_threshold:.0%}")
    print("-" * 50)
    
    engine.start()
    
    try:
        time.sleep(duration_minutes * 60)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    engine.stop()
    
    # Print summary
    summary = engine.get_signal_summary()
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    # Quick test run
    print("Testing Live Engine (5 minute demo)...")
    run_paper_trading(ticker="SPY", duration_minutes=5, confidence_threshold=0.6)
