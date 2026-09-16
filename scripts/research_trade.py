"""
AFML Quant Pipeline - Headless Research & Trade Tool
Searches the internet for strategy details and runs a backtest.

Supported Auth Modes:
1. Vertex AI (ADC): Requires --project and --location (Recommended)
2. AI Studio (API Key): Requires --api_key or .env

Usage:
    python scripts/research_trade.py --stock <SYMBOL> --strategy <QUERY> [--project <ID> --location <LOC>]
"""

import sys
import os
import argparse
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root path
sys.path.append(str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

try:
    from pipeline.advanced_pipeline import create_pipeline
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Auth Logic
def get_api_key(args_key):
    if args_key:
        return args_key
    
    # Try scripts/.env
    script_env = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(script_env):
        load_dotenv(script_env)
        if os.getenv("GOOGLE_API_KEY"):
            return os.getenv("GOOGLE_API_KEY")
            
    # Try root .env (might fail due to permissions)
    try:
        load_dotenv()
        return os.getenv("GOOGLE_API_KEY")
    except Exception:
        pass
        
    return None

class ResearchAgent:
    """Agent that uses Gemini with Google Search (Vertex or AI Studio)."""
    
    def __init__(self, api_key=None, project=None, location=None):
        self.mode = "unknown"
        
        # Priority 1: Vertex AI (ADC)
        if project and location:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel, Tool, GoogleSearchRetrieval
                
                logger.info(f"Initializing Vertex AI (Project: {project}, Location: {location})...")
                vertexai.init(project=project, location=location)
                
                # Configure Search Tool
                search_tool = Tool.from_google_search_retrieval(GoogleSearchRetrieval())
                self.model = GenerativeModel("gemini-1.5-pro", tools=[search_tool])
                self.mode = "vertex"
                return
            except ImportError:
                logger.warning("google-cloud-aiplatform not installed. Falling back to API Key.")
            except Exception as e:
                logger.error(f"Vertex AI Init failed: {e}")

        # Priority 2: AI Studio (API Key)
        if api_key:
            try:
                import google.generativeai as genai
                logger.info("Initializing AI Studio with API Key...")
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp', tools='google_search_retrieval')
                self.mode = "studio"
                return
            except ImportError:
                logger.error("google-generativeai not installed.")
                
        if self.mode == "unknown":
            raise ValueError("No valid authentication method found. Provide --project/--location (Vertex) or --api_key (Studio).")

    def research_strategy(self, stock: str, strategy_hint: str) -> dict:
        """
        Research specifically for the stock and strategy.
        Returns parameters for the pipeline.
        """
        logger.info(f"Researching online ({self.mode}): {stock} - {strategy_hint}...")
        
        prompt = f"""
        Research the stock {stock} and the strategy '{strategy_hint}' using Google Search.
        
        Find:
        1. Recent news or sentiment affecting this strategy for {stock}.
        2. Typical technical parameters used for this strategy (e.g., lookback periods, thresholds).
        3. Any specific risks currently associated with {stock}.
        
        Output a structured summary and suggest parameters:
        - Lookback Window (int)
        - Profit Target Multiplier (float)
        - Stop Loss Multiplier (float)
        - Key Risks (list)
        """
        
        try:
            if self.mode == "vertex":
                response = self.model.generate_content(prompt)
            else:
                response = self.model.generate_content(prompt)
                
            return {
                "summary": response.text,
                "params": {
                    "lookback": 50,
                    "profit": 2.0,
                    "stop": 1.0
                }
            }
        except Exception as e:
            logger.warning(f"Research failed ({e}). Using defaults.")
            return {"summary": "Search failed.", "params": {"lookback": 50, "profit": 2.0, "stop": 1.0}}

def fetch_data(symbol: str, days: int = 365*2) -> pd.DataFrame:
    """Fetch data from yfinance."""
    logger.info(f"Fetching {days} days of data for {symbol} via yfinance...")
    start_date = datetime.now() - timedelta(days=days)
    df = yf.download(symbol, start=start_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
        
    # Standardize columns (multi-index handling for new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Rename common columns to standard Title Case
    col_map = {
        'Close': 'Close', 'close': 'Close',
        'Open': 'Open', 'open': 'Open',
        'High': 'High', 'high': 'High',
        'Low': 'Low', 'low': 'Low',
        'Volume': 'Volume', 'volume': 'Volume'
    }
    df = df.rename(columns=col_map)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Headless Research & Trade')
    parser.add_argument('--stock', required=True, help='Stock symbol')
    parser.add_argument('--strategy', required=True, help='Strategy description')
    parser.add_argument('--api_key', help='Gemini API Key (AI Studio)')
    parser.add_argument('--project', help='GCP Project ID (Vertex AI)')
    parser.add_argument('--location', default='us-central1', help='GCP Region (Vertex AI)')
    
    args = parser.parse_args()
    
    # Resolve Auth
    api_key = get_api_key(args.api_key)

    # Detect Pair Trading Mode
    if "," in args.stock:
        tickers = [t.strip() for t in args.stock.split(",")]
        if len(tickers) == 2:
            print(f"Analysing pair {tickers[0]} vs {tickers[1]}...")
            
            try:
                from pipeline.pair_trading import PairDataProcessor, CointegrationTester, MeanReversionStrategy
            except ImportError:
                print("Error: Could not import Pair Trading Module.")
                sys.exit(1)
            
            # 1. Fetch
            try:
                processor = PairDataProcessor(tickers[0], tickers[1])
                data = processor.fetch_data()
            except Exception as e:
                print(f"Error fetching pair data: {e}")
                sys.exit(1)
            
            # 2. Cointegration
            tester = CointegrationTester()
            coint_res = tester.engage_engle_granger(data[tickers[0]], data[tickers[1]])
            
            print("\n" + "="*60)
            print("STATISTICAL ARBITRAGE METRICS")
            print("="*60)
            print(f"Correlation: {coint_res['correlation']:.4f}")
            print(f"Cointegrated: {coint_res['is_cointegrated']} (p={coint_res['spread_p_value']:.4f})")
            print(f"Hedge Ratio: {coint_res['hedge_ratio']:.4f}")
            
            # 3. Strategy
            strat = MeanReversionStrategy()
            backtest = strat.backtest(data[tickers[0]], data[tickers[1]])
            
            z_score = backtest['z_score'].iloc[-1]
            print(f"\nCurrent Z-Score: {z_score:.4f}")
            
            print("\n" + "="*60)
            print(f"LATEST SIGNAL (Data Time: {data.index[-1]})")
            print("="*60)
            
            if z_score > 2.0:
                print(f"Action: SHORT SPREAD (Short {tickers[0]} / Long {tickers[1]})")
                print("VERDICT: SELL SPREAD")
            elif z_score < -2.0:
                print(f"Action: LONG SPREAD (Long {tickers[0]} / Short {tickers[1]})")
                print("VERDICT: BUY SPREAD")
            else:
                print("Action: WAIT (Spread within normal range)")
                print("VERDICT: PASS")
                
            print("="*60 + "\n")
            return
    
    try:
        # 1. Research
        researcher = ResearchAgent(api_key=api_key, project=args.project, location=args.location)
        insights = researcher.research_strategy(args.stock, args.strategy)
        
        print("\n" + "="*60)
        print(f"RESEARCH REPORT: {args.stock}")
        print("="*60)
        print(insights['summary'])
        print("-"*60)
        
    except ValueError as e:
        logger.warning(f"Research Agent unavailable ({e}). Using default parameters.")
        insights = {
            "summary": "Research unavailable due to missing credentials. Using default technical parameters.",
            "params": {
                "lookback": 50,
                "profit": 2.0,
                "stop": 1.0
            }
        }
        
    # 2. Data
    try:
        data = fetch_data(args.stock)
    except Exception as e:
        logger.error(e)
        sys.exit(1)
    
    # 3. Simulate/Train
    logger.info("Initializing ML Pipeline with researched parameters...")
    params = insights['params']
    
    pipeline = create_pipeline(
        model_type='ensemble',
        use_meta_labeling=True,
        profit_mult=params['profit'],
        stop_mult=params['stop'],
        num_bars=params['lookback']
    )
    
    logger.info(f"Training on {len(data)} bars...")
    pipeline.fit(data['Close'], volume=data.get('Volume'), high=data.get('High'), low=data.get('Low'))
    
    # 4. Report
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(pipeline.summary())
    
    # Prediction (pass full data to ensure feature consistency)
    pred = pipeline.predict(data['Close'], volume=data.get('Volume'), high=data.get('High'), low=data.get('Low'))
    
    if not pred.empty:
        last_signal = pred.iloc[-1]
        
        print("\n" + "="*60)
        last_timestamp = pred.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        print(f"LATEST SIGNAL (Data Time: {last_timestamp})")
        print("="*60)
        print(f"Action: {last_signal.get('action', 'UNKNOWN')}")
        print(f"Confidence: {last_signal.get('probability', 0):.2f}")
        print(f"Signal: {last_signal.get('final_signal', 0)}")
        
        # Explicit Verdict
        if last_signal.get('final_signal') == 1:
            print("VERDICT: BUY")
        elif last_signal.get('final_signal') == -1:
            print("VERDICT: SELL")
        else:
            print("VERDICT: WAIT/PASS")
            
        print("="*60 + "\n")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()
