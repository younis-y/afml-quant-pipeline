"""
AFML Quant Pipeline - Market Scanner
Scans a list of stocks to find the highest confidence signals.

Usage:
    python scripts/scan_market.py [--tickers AAPL,MSFT,NVDA...] [--universe mag7|tech|sector_etfs]
"""

import sys
import os
import argparse
import logging
import traceback
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Add project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import Pipeline (skip agents for speed in scanner)
try:
    from pipeline.advanced_pipeline import create_pipeline
except ImportError:
    # Fallback if imports fail (shouldn't happen if env is right)
    sys.exit(1)

logging.basicConfig(level=logging.ERROR) # Only show errors to keep output clean

UNIVERSES = {
    'mag7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    'tech': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'TXN'],
    'semis': ['NVDA', 'TSM', 'AVGO', 'AMD', 'QCOM', 'INTC', 'MU', 'AMAT', 'LRCX'],
    'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP']
}

def analyze_ticker(ticker):
    try:
        # 1. Fetch Data (1 year is enough for quick scan)
        df = yf.download(ticker, period="2y", progress=False)
        if df.empty or len(df) < 100:
            return None
            
        # Standardize
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        col_map = {
            'Close': 'Close', 'close': 'Close',
            'Volume': 'Volume', 'volume': 'Volume',
            'High': 'High', 'high': 'High',
            'Low': 'Low', 'low': 'Low'
        }
        df = df.rename(columns=col_map)
        
        # 2. Run Pipeline (Fast Mode)
        # We use a preset logic unless specific research overrides it
        pipeline = create_pipeline(
            model_type='ensemble',
            use_meta_labeling=True,
            profit_mult=2.0,
            stop_mult=1.0,
            num_bars=50
        )
        
        pipeline.fit(df['Close'], volume=df.get('Volume'), high=df.get('High'), low=df.get('Low'))
        
        # 3. Predict
        pred = pipeline.predict(df['Close'], volume=df.get('Volume'), high=df.get('High'), low=df.get('Low'))
        if pred.empty:
            return None
            
        last = pred.iloc[-1]
        
        # Get metrics safely
        sharpe = 0.0
        if pipeline.result_:
            sharpe = pipeline.result_.sharpe_ratio
        
        return {
            'Ticker': ticker,
            'Action': last.get('action', 'PASS'),
            'Confidence': float(last.get('probability', 0.0)),
            'Signal': int(last.get('final_signal', 0)),
            'Sharpe': float(sharpe)
        }
    except Exception:
        return {'Ticker': ticker, 'Error': traceback.format_exc()}

def main():
    parser = argparse.ArgumentParser(description='Market Scanner')
    parser.add_argument('--tickers', help='Comma-separated list of tickers')
    parser.add_argument('--universe', default='mag7', choices=UNIVERSES.keys(), help='Predefined universe')
    
    args = parser.parse_args()
    
    # Determine list
    if args.tickers:
        targets = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        targets = UNIVERSES[args.universe]
        
    print(f"Scanning {len(targets)} assets...")
    results = []
    
    # Sequential execution to avoid yfinance/pandas thread safety issues
    for ticker in tqdm(targets):
        res = analyze_ticker(ticker)
        if res:
            if 'Error' in res:
                print(f"{res['Ticker']}: {res['Error']}")
            else:
                results.append(res)
        else:
            print("No result for a ticker (empty data or prediction)")
    
    # Sort by Confidence
    results.sort(key=lambda x: x['Confidence'], reverse=True)
    
    # Display
    print("\n" + "="*70)
    print(f"{'TICKER':<8} | {'ACTION':<6} | {'CONF %':<8} | {'SIGNAL':<6} | {'SHARPE':<6}")
    print("-" * 70)
    
    for r in results:
        # Highlights
        # (Since this is terminal, we skip ANSI colors to avoid compatibility issues, 
        # or we could use them if we knew the shell supports it. Keeping plain text for safety.)
        
        print(f"{r['Ticker']:<8} | {r['Action']:<6} | {r['Confidence']*100:5.1f}%   | {r['Signal']:<6} | {r['Sharpe']:5.2f}")
        
    print("="*70)
    
    # Best Pick
    if results:
        best = results[0]
        if best['Confidence'] > 0.6:
            print(f"\nTop pick: {best['Ticker']} ({best['Action']})")
        else:
            print("\nNo high-confidence setup found (>60%).")

if __name__ == "__main__":
    main()
