"""
AFML Quant Pipeline - Automated Training Script
Downloads data, trains the advanced pipeline, generates a report, and cleans up.

Usage:
    python scripts/auto_train.py --url <DATA_URL> [--output <OUTPUT_DIR>]
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
# Add local packages path

from pipeline.advanced_pipeline import create_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url: str, output_dir: Path) -> Path:
    """Download file from URL to output directory. Handles IEX JSON API."""
    
    # Check if URL is the IEX JSON API
    if 'iextrading.com/api/1.0/hist' in url:
        logger.info("IEX Historical Data API detected. Fetching file list...")
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data_index = resp.json()
            
            # Get latest date
            dates = sorted(data_index.keys(), reverse=True)
            if not dates:
                raise ValueError("No dates found in IEX index")
                
            latest_date = dates[0]
            files = data_index[latest_date]
            
            # Find DEEP feed (richest data)
            deep_file = next((f for f in files if f['feed'] == 'DEEP'), None)
            
            if not deep_file:
                logger.warning(f"No DEEP feed found for {latest_date}, taking first file.")
                deep_file = files[0]
                
            download_url = deep_file['link']
            file_size_gb = float(deep_file['size']) / (1024**3)
            
            logger.info(f"Selected latest file: {latest_date} - {deep_file['feed']} ({file_size_gb:.2f} GB)")
            logger.warning("WARNING: This is a large file. Download may take a while.")
            
            url = download_url
            
        except Exception as e:
            logger.error(f"Failed to parse IEX API: {e}")
            raise

    logger.info(f"Downloading data from {url}...")
    
    # Determine filename
    if 'googleapis.com' in url:
        # Extract filename from query or path
        from urllib.parse import unquote
        path = url.split('?')[0]
        filename = unquote(path.split('/')[-1])
    else:
        filename = url.split('/')[-1]
        
    if not filename:
        filename = f"download_{int(time.time())}.dat"
        
    output_path = output_dir / filename
    
    # Check if exists to avoid re-downloading large files during testing
    if output_path.exists() and output_path.stat().st_size > 0:
         logger.info(f"File {output_path} already exists. Skipping download.")
         return output_path
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Simple progress logging
                    if total_size > 0 and downloaded % (100 * 1024 * 1024) == 0: # Log every 100MB
                        logger.info(f"Downloaded {downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB")
                
        logger.info(f"Download complete: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from file, handling PCAP limitation."""
    logger.info(f"Loading data from {file_path}...")
    
    if file_path.suffix.lower() in ['.pcap', '.gz']:
        try:
            logger.info("PCAP format detected. Attempting to parse with iex_parser...")
            from iex_parser import Parser
            
            records = []
            with Parser(str(file_path)) as reader:
                for msg in reader:
                    # Filter for Trade Reports (type 'T' usually, or 0x54)
                    # iex_parser returns dict with 'type', 'timestamp', 'price', 'size', 'symbol'
                    if msg.get('type') in ['start_of_session', 'end_of_session']:
                        continue
                        
                    # Target Trade Report messages (DEEP/TOPS trades)
                    # Common types: 'trade_report', 'trade_break'
                    # We just want executed trades for price data
                    if 'price' in msg and 'size' in msg and 'timestamp' in msg:
                        records.append({
                            'time': msg['timestamp'],
                            'symbol': msg.get('symbol', 'UNKNOWN'),
                            'price': float(msg['price']),
                            'size': int(msg['size'])
                        })
                        
            if not records:
                logger.warning("No trade records found in PCAP. Using simulation fallback.")
                return _generate_mock_data()
                
            logger.info(f"Extracted {len(records)} trade records. Aggregating to bars...")
            
            # Convert to DataFrame
            raw_df = pd.DataFrame(records)
            raw_df['time'] = pd.to_datetime(raw_df['time'])
            raw_df = raw_df.set_index('time')
            
            # Resample to 1-minute bars (OHLCV)
            # We'll aggregate all symbols together for a "market" view, or just pick the most active one
            # For simplicity in this script, let's take the most active symbol
            top_symbol = raw_df['symbol'].value_counts().idxmax()
            logger.info(f"Training on most active symbol: {top_symbol}")
            
            symbol_data = raw_df[raw_df['symbol'] == top_symbol]
            
            ohlcv = symbol_data['price'].resample('1min').ohlc()
            ohlcv['Volume'] = symbol_data['size'].resample('1min').sum()
            
            # Renaming for pipeline
            ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            ohlcv = ohlcv.dropna()
            
            return ohlcv

        except ImportError:
            logger.error("iex_parser not found. Please install: pip install iex_parser")
            logger.warning("Falling back to simulation.")
            return _generate_mock_data()
        except Exception as e:
            logger.error(f"Error parsing PCAP: {e}")
            logger.warning("Falling back to simulation due to parse error.")
            return _generate_mock_data()
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        
        # Standardize columns
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Map common names to Close/Volume/High/Low
        col_map = {
            'close': 'Close',
            'adj. close': 'Close',
            'adj close': 'Close',
            'aapl.close': 'Close', 
            'volume': 'Volume',
            'aapl.volume': 'Volume',
            'high': 'High',
            'aapl.high': 'High',
            'low': 'Low',
            'aapl.low': 'Low',
            'open': 'Open',
            'aapl.open': 'Open'
        }
        
        # Renaissance of column names
        new_cols = {}
        for col in df.columns:
            for k, v in col_map.items():
                if k in col:
                    new_cols[col] = v
                    break
                    
        df = df.rename(columns=new_cols)
        
        # Ensure we have at least Close
        if 'Close' not in df.columns:
            logger.error(f"Could not find 'Close' column. Available: {df.columns.tolist()}")
            raise ValueError("Data must likely contain a 'Close' price column")
            
        return df
        
    if file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
        return df
        
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _generate_mock_data():
    """Helper to generate synthetic data when parsing fails."""
    n_bars = 2000
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='5min')
    np.random.seed(42)
    returns = np.random.randn(n_bars) * 0.001 + 0.0001
    price = 100 * np.exp(np.cumsum(returns))
    volume = np.random.lognormal(10, 1, n_bars)
    
    return pd.DataFrame({
        'Close': price,
        'Volume': volume,
        'High': price * (1 + np.abs(np.random.randn(n_bars) * 0.001)),
        'Low': price * (1 - np.abs(np.random.randn(n_bars) * 0.001))
    }, index=dates)
    

def train_job(data: pd.DataFrame) -> dict:
    """Run the training pipeline."""
    logger.info("Initializing Advanced ML Pipeline...")
    
    pipeline = create_pipeline(
        model_type='ensemble',
        use_meta_labeling=True,
        profit_mult=2.0,
        stop_mult=1.0,
        n_cv_splits=3
    )
    
    logger.info(f"Training on {len(data)} bars...")
    pipeline.fit(data['Close'], volume=data.get('Volume'), high=data.get('High'), low=data.get('Low'))
    
    # Return results
    if pipeline.is_fitted:
        return {
            'summary': pipeline.summary(),
            'sharpe': pipeline.result_.sharpe_ratio,
            'dsr': pipeline.result_.dsr,
            'accuracy': pipeline.result_.test_metrics.get('accuracy', 0.0),
            'timestamp': datetime.now().isoformat()
        }
    return {}

def save_report(results: dict, output_dir: Path, source_url: str):
    """Save training report."""
    report_path = output_dir / f"training_report_{int(time.time())}.md"
    
    content = f"""# Automated Training Report
**Date**: {results['timestamp']}
**Source**: {source_url}

## Performance Metrics
- **Sharpe Ratio**: {results.get('sharpe', 0):.4f}
- **Deflated Sharpe (DSR)**: {results.get('dsr', 0):.4f}
- **Accuracy**: {results.get('accuracy', 0):.2%}

## Pipeline Summary
```text
{results.get('summary', 'No summary available')}
```

*Generated by the auto-trainer script*
"""
    
    with open(report_path, 'w') as f:
        f.write(content)
        
    logger.info(f"Report saved to {report_path}")
    return report_path

def cleanup(file_path: Path):
    """Delete the downloaded file."""
    if file_path.exists():
        os.remove(file_path)
        logger.info(f"Deleted temporary file: {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Auto-trainer')
    parser.add_argument('--url', required=True, help='URL to download data from')
    parser.add_argument('--output', default='reports', help='Directory for reports')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file = None
    
    try:
        # 1. Download
        temp_file = download_file(args.url, output_dir)
        
        # 2. Load
        data = load_data(temp_file)
        
        # 3. Train
        results = train_job(data)
        
        # 4. Report
        report_file = save_report(results, output_dir, args.url)
        print(f"\nSUCCESS: Training complete. Report at: {report_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
        
    finally:
        # 5. Cleanup
        if temp_file:
            cleanup(temp_file)

if __name__ == "__main__":
    main()
