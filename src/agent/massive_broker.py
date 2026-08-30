"""
Massive.com API integration for stock quotes.

This module provides a thin wrapper around Massive.com's aggregates API
to retrieve previous day's stock market data. It replaces Alpaca's data API
for quote fetching while maintaining the same interface.

API endpoint: GET /v2/aggs/ticker/{stocksTicker}/prev
Returns previous day's aggregate (open, high, low, close, volume) for a stock.
"""

import base64
import os
import requests
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()

_SECRET_SCOPE = os.environ.get("MASSIVE_SECRET_SCOPE", "massive")
_API_KEY_SECRET_KEY = os.environ.get("MASSIVE_SECRET_KEY", "api-key")

_api_key: str | None = None


def _secret(key: str) -> str:
    """Fetch and base64-decode a value from the Databricks secret scope."""
    secret = _w.secrets.get_secret(scope=_SECRET_SCOPE, key=key)
    return base64.b64decode(secret.value).decode("utf-8")


def _get_api_key() -> str:
    global _api_key
    if _api_key is None:
        _api_key = _secret(_API_KEY_SECRET_KEY)
    return _api_key


def get_quote(symbol: str) -> dict:
    """
    Get previous day's aggregate data for a stock ticker symbol from Massive.com.
    
    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
    
    Returns:
        A dict with symbol, price (close), as_of (ISO timestamp), volume, change,
        change_percent, open, high, and low.
    """
    symbol = symbol.strip().upper()
    
    # Massive.com API endpoint for previous day aggregates
    url = f"https://api.massive.com/v2/aggs/ticker/{symbol}/prev"
    
    headers = {
        "Authorization": f"Bearer {_get_api_key()}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # The response typically contains a 'results' array with aggregate data
        # Example structure: {"ticker": "AAPL", "results": [{"c": close, "h": high, "l": low, "o": open, "v": volume, "t": timestamp}]}
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            close_price = float(result.get("c", 0))  # Close price
            open_price = float(result.get("o", 0))   # Open price
            volume = int(result.get("v", 0))         # Volume
            timestamp = result.get("t", 0)           # Unix timestamp in milliseconds
            
            # Calculate change and change percent
            change = close_price - open_price
            change_percent = (change / open_price * 100) if open_price > 0 else 0
            
            # Convert timestamp to ISO format
            from datetime import datetime
            as_of = datetime.fromtimestamp(timestamp / 1000).isoformat() if timestamp else ""
            
            return {
                "symbol": data.get("ticker", symbol),
                "price": close_price,
                "as_of": as_of,
                "volume": volume,
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "open": open_price,
                "high": float(result.get("h", 0)),
                "low": float(result.get("l", 0)),
            }
        else:
            raise RuntimeError(f"No results returned for {symbol}")
            
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch quote for {symbol} from Massive.com: {e}")


def get_bars(
    symbol: str,
    multiplier: int = 1,
    timespan: str = "day",
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 120,
) -> list[dict]:
    """
    Get historical OHLCV bars for a stock ticker from Massive.com.
    
    Args:
        symbol: Stock ticker symbol, e.g. "AAPL"
        multiplier: Size of the timespan multiplier (e.g., 1)
        timespan: Size of the time window (minute, hour, day, week, month)
        from_date: Start date (YYYY-MM-DD), defaults to 30 days ago
        to_date: End date (YYYY-MM-DD), defaults to today
        limit: Max number of bars (default 120)
    
    Returns:
        List of dicts with date, open, high, low, close, volume, vwap, transactions
    """
    from datetime import datetime, timedelta
    
    symbol = symbol.strip().upper()
    
    # Default date range: last 30 days
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")
    
    endpoint = f"https://api.massive.com/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": limit}
    headers = {"Authorization": f"Bearer {_get_api_key()}"}
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "results" not in data or len(data["results"]) == 0:
            return []
        
        results = data["results"]
        return [
            {
                "date": datetime.fromtimestamp(bar["t"] / 1000).date().isoformat(),
                "timestamp": datetime.fromtimestamp(bar["t"] / 1000).isoformat(),
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": int(bar["v"]),
                "vwap": float(bar.get("vw", 0)),
                "transactions": int(bar.get("n", 0)),
            }
            for bar in results
        ]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch bars for {symbol} from Massive.com: {e}")


def get_stock_performance(symbol: str, days: int = 30) -> dict:
    """
    Get stock performance over the last N calendar days.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of calendar days to look back
    
    Returns:
        Dict with ticker, period dates, bars, and performance summary
    """
    from datetime import datetime, timedelta
    
    symbol = symbol.strip().upper()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    bars = get_bars(
        symbol,
        from_date=start_date.strftime("%Y-%m-%d"),
        to_date=end_date.strftime("%Y-%m-%d"),
    )
    
    if not bars or len(bars) == 0:
        return {
            "error": f"No historical data found for {symbol}",
            "ticker": symbol,
            "requested_days": days,
            "trading_days": 0,
        }
    
    # Calculate performance summary
    start_price = bars[0]["open"]
    end_price = bars[-1]["close"]
    price_change = round(end_price - start_price, 2)
    percent_change = (
        round((price_change / start_price) * 100, 2) if start_price != 0 else 0.0
    )
    
    # Find period high and low
    all_highs = [bar["high"] for bar in bars]
    all_lows = [bar["low"] for bar in bars]
    period_high = round(max(all_highs), 2)
    period_low = round(min(all_lows), 2)
    
    return {
        "ticker": symbol,
        "requested_days": days,
        "trading_days": len(bars),
        "start_date": bars[0]["date"],
        "end_date": bars[-1]["date"],
        "bars": bars,
        "summary": {
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "price_change": price_change,
            "percent_change": percent_change,
            "period_high": period_high,
            "period_low": period_low,
        },
    }


def compare_stocks(symbols: list[str], days: int = 30) -> dict:
    """
    Compare performance of multiple stocks over the same period.
    
    Args:
        symbols: List of stock ticker symbols
        days: Number of calendar days to look back
    
    Returns:
        Dict with stocks list and comparison summary
    """
    if not symbols or len(symbols) == 0:
        return {"error": "At least one ticker is required"}
    
    # Fetch performance for each stock
    results = []
    for symbol in symbols:
        perf = get_stock_performance(symbol, days)
        results.append(perf)
    
    # Calculate comparison summary
    valid_results = [r for r in results if "error" not in r and "summary" in r]
    
    if len(valid_results) == 0:
        return {
            "requested_days": days,
            "stocks": results,
            "error": "No valid performance data for any of the requested symbols",
        }
    
    # Find best and worst performers
    percent_changes = [
        (r["ticker"], r["summary"]["percent_change"]) for r in valid_results
    ]
    percent_changes.sort(key=lambda x: x[1], reverse=True)
    
    best_performer = percent_changes[0]
    worst_performer = percent_changes[-1]
    avg_return = round(
        sum(pc[1] for pc in percent_changes) / len(percent_changes), 2
    )
    
    return {
        "requested_days": days,
        "stocks": results,
        "comparison_summary": {
            "best_performer": {
                "ticker": best_performer[0],
                "percent_change": best_performer[1],
            },
            "worst_performer": {
                "ticker": worst_performer[0],
                "percent_change": worst_performer[1],
            },
            "average_return": avg_return,
            "stocks_compared": len(valid_results),
        },
    }
