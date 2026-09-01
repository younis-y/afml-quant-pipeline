"""
Symbol Resolver Module
Resolves company names to tickers using Yahoo Finance Autocomplete API.
"""

import requests
from fake_useragent import UserAgent
from typing import List, Dict, Any

def search_ticker(query: str) -> List[Dict[str, Any]]:
    """
    Search for tickers using Yahoo Finance Autocomplete API.
    
    Args:
        query: Company name or symbol to search for.
        
    Returns:
        List of dictionaries containing symbol details.
    """
    if not query or len(query.strip()) < 1:
        return []
        
    url = f"https://query2.finance.yahoo.com/v1/finance/search"
    
    # Generate random user agent to avoid 403 blocks
    try:
        ua = UserAgent()
        user_agent = ua.chrome
    except:
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
    headers = {
        "User-Agent": user_agent
    }
    
    params = {
        "q": query,
        "quotesCount": 10,
        "newsCount": 0,
        "enableFuzzyQuery": "true",
        "quotesQueryId": "tss_match_phrase_query"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "quotes" in data:
            for quote in data["quotes"]:
                # Filter for tradeable asset types
                quote_type = quote.get("quoteType", "").upper()
                if quote_type not in ["EQUITY", "ETF", "COMMODITY", "FUTURE", "INDEX", "MUTUALFUND", "CRYPTOCURRENCY"]:
                    continue
                    
                # Extract relevant fields
                symbol = quote.get("symbol")
                shortname = quote.get("shortname", quote.get("longname", symbol))
                exchange = quote.get("exchange", "Unknown")
                
                # Identify if delayed
                is_delayed = False
                delayed_suffix = ['.L', '.NS', '.BO', '.PA', '.DE', '.TO', '.SS', '.SZ']
                if any(symbol.endswith(suffix) for suffix in delayed_suffix):
                    is_delayed = True
                
                results.append({
                    "symbol": symbol,
                    "shortname": shortname,
                    "exchange": exchange,
                    "type": quote_type.title(),
                    "is_delayed": is_delayed
                })
                
        return results
        
    except Exception as e:
        print(f"Error searching ticker '{query}': {e}")
        return []
