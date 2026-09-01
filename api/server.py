"""
AFML Quant Pipeline - FastAPI Backend
Optional REST layer over the pipeline. Serves JSON only; the repository
ships no frontend, so the CORS origins below are the localhost ports a
local single-page client would typically use.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.data_processor import fetch_data, dollar_bars
from pipeline.meta_model import MetaLabelingPipeline
from pipeline.labeling import get_volatility
from pipeline.symbol_resolver import search_ticker
from utils.cache import cache

app = FastAPI(
    title="AFML Quant Pipeline API",
    description="AI-Powered Quantitative Trading Terminal",
    version="1.0.0"
)

# CORS for a local browser client on the usual dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
_pipeline: Optional[MetaLabelingPipeline] = None

def get_pipeline() -> MetaLabelingPipeline:
    """Get or create the meta-labeling pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MetaLabelingPipeline(
            sma_fast=50,
            sma_slow=200,
            confidence_threshold=0.75
        )
    return _pipeline


# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str


class PriceDataPoint(BaseModel):
    time: str
    price: float
    open: float = 0
    high: float = 0
    low: float = 0
    close: float = 0
    volume: float = 0
    sma20: Optional[float] = None
    sma50: Optional[float] = None


class SignalResponse(BaseModel):
    action: str
    confidence: float
    threshold: float
    above_threshold: bool
    timestamp: str


class RecentSignal(BaseModel):
    time: str
    action: str
    confidence: float
    price: float


class MetricsResponse(BaseModel):
    current_price: float
    price_change: float
    price_change_pct: float
    volatility: float
    dollar_bars_count: int
    dollar_threshold: int


class DashboardDataResponse(BaseModel):
    metrics: MetricsResponse
    signal: SignalResponse
    price_data: List[PriceDataPoint]
    recent_signals: List[RecentSignal]


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/data", response_model=DashboardDataResponse)
async def get_dashboard_data(
    background_tasks: BackgroundTasks,
    ticker: str = "SPY",
    period: str = "7d",
    dollar_threshold: int = 500_000
):
    """
    Get all dashboard data in a single request.
    Returns metrics, current signal, price data, and recent signals.
    Optimized with caching and background training.
    """
    try:
        # 1. Try to get from cache first
        cache_key = f"dashboard_{ticker}_{period}_{dollar_threshold}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data

        # 2. Fetch market data (block here as we need it for display)
        # Check if we have raw data cached separately
        raw_key = f"raw_{ticker}_{period}"
        raw_data = cache.get(raw_key)
        
        if raw_data is None:
            raw_data = fetch_data(ticker, period=period, interval="1m")
            cache.set(raw_key, raw_data, ttl=60) # Cache raw data for 1 min
            
        dbar = dollar_bars(raw_data, dollar_threshold=dollar_threshold)
        
        if len(dbar) < 250:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data: {len(dbar)} bars. Need at least 250."
            )
        
        # 3. Get pipeline
        pipeline = get_pipeline()
        close = dbar['Close']
        
        # 4. Handle Training (Background Task)
        if not pipeline.meta.is_fitted:
            # If not fitted, trigger background training and return partial data
            background_tasks.add_task(pipeline.train, close)
            action = "WAIT" # Special status for training
            confidence = 0.0
            above_threshold = False
        else:
            # Generate signals if ready
            try:
                signal_df, proba = pipeline.predict(close)
                latest_signal = signal_df.iloc[-1] if len(signal_df) > 0 else None
                
                if latest_signal is not None:
                    action = "BUY" if latest_signal > 0 else "SELL" if latest_signal < 0 else "PASS"
                    confidence = float(proba.iloc[-1]) if len(proba) > 0 else 0.5
                else:
                    action = "PASS"
                    confidence = 0.5
            except Exception as e:
                # If prediction fails (e.g. data mismatch), re-train in background
                background_tasks.add_task(pipeline.train, close)
                action = "WAIT"
                confidence = 0.0
        
        above_threshold = confidence >= 0.75
        if not above_threshold and action != "WAIT":
             action = "PASS"
        
        # 5. Calculate Metrics
        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2]) if len(close) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
        
        volatility = get_volatility(close, span=20)
        current_vol = float(volatility.iloc[-1]) if len(volatility) > 0 else 0.02
        
        # 6. Prepare Response — include OHLC + SMA overlays
        price_data = []
        chart_bars = dbar.tail(80)
        closes = chart_bars['Close']
        sma20 = closes.rolling(20).mean()
        sma50 = closes.rolling(50).mean()

        for i, (idx, row) in enumerate(chart_bars.iterrows()):
            point = {
                "time": idx.strftime("%H:%M") if hasattr(idx, 'strftime') else str(idx),
                "price": float(row['Close']),
                "open": float(row['Open']) if 'Open' in row else float(row['Close']),
                "high": float(row['High']) if 'High' in row else float(row['Close']),
                "low": float(row['Low']) if 'Low' in row else float(row['Close']),
                "close": float(row['Close']),
                "volume": float(row['Volume']) if 'Volume' in row else 0,
            }
            s20 = sma20.iloc[i]
            s50 = sma50.iloc[i]
            if not pd.isna(s20):
                point["sma20"] = round(float(s20), 2)
            if not pd.isna(s50):
                point["sma50"] = round(float(s50), 2)
            price_data.append(point)
        
        # Mock recent signals for now
        recent_signals = [
            {"time": "14:23", "action": action if action != "WAIT" else "PASS", "confidence": confidence, "price": current_price},
            {"time": "13:45", "action": "PASS", "confidence": 0.62, "price": prev_price},
            {"time": "12:30", "action": "SELL", "confidence": 0.88, "price": current_price * 1.005},
            {"time": "11:15", "action": "BUY", "confidence": 0.79, "price": current_price * 0.998},
            {"time": "10:00", "action": "PASS", "confidence": 0.55, "price": current_price * 0.996},
        ]
        
        response_data = {
            "metrics": {
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "volatility": round(current_vol, 4),
                "dollar_bars_count": len(dbar),
                "dollar_threshold": dollar_threshold
            },
            "signal": {
                "action": action,
                "confidence": round(confidence, 2),
                "threshold": 0.75,
                "above_threshold": above_threshold,
                "timestamp": datetime.now().isoformat()
            },
            "price_data": price_data,
            "recent_signals": recent_signals
        }
        
        # Cache the full response for 5 seconds (short TTL for dashboard)
        cache.set(cache_key, response_data, ttl=5)
        
        return response_data
    
    except Exception as e:
        print(f"Error in dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signal")
async def get_signal(ticker: str = "SPY"):
    """Get current trading signal only."""
    data = await get_dashboard_data(ticker=ticker)
    return data["signal"]


@app.get("/api/metrics")
async def get_metrics(ticker: str = "SPY"):
    """Get current metrics only."""
    data = await get_dashboard_data(ticker=ticker)
    return data["metrics"]


@app.get("/api/resolve-symbol")
async def resolve_symbol(query: str):
    """
    Search for tickers using Yahoo Finance Autocomplete.
    Returns list of matching equities/ETFs with exchange info.
    """
    return search_ticker(query)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
