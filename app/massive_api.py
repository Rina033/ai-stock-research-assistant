"""
Massive Stocks API client.

Provides real-time and historical tick data, trades, quotes, and
fundamentals across all major US exchanges via REST.

Configure with env vars:
  MASSIVE_API_KEY  — API key (free tier available)
  MASSIVE_BASE_URL — Base URL (default: https://api.massivedata.io/v1)
"""

import os
from datetime import datetime, timedelta

import requests

_BASE_URL = os.environ.get("MASSIVE_API_BASE_URL", "https://api.massive.com")
_TIMEOUT = 15


def _get_api_key() -> str:
    """Fetch the Massive API key from Databricks secrets or env var."""
    key = os.environ.get("MASSIVE_API_KEY", "")
    if key:
        return key
    scope = os.environ.get("MASSIVE_SECRET_SCOPE", "massive")
    secret_key = os.environ.get("MASSIVE_SECRET_KEY", "api-key")
    try:
        import base64
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        secret = w.secrets.get_secret(scope=scope, key=secret_key)
        return base64.b64decode(secret.value).decode("utf-8")
    except Exception:
        pass
    return ""


def _headers() -> dict:
    h = {"Accept": "application/json"}
    key = _get_api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _get(path: str, params: dict = None) -> dict | list:
    url = f"{_BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

def get_quote(symbol: str) -> dict:
    """Get previous day's aggregate (close, open, high, low, volume) for a ticker."""
    try:
        data = _get(f"/v2/aggs/ticker/{symbol.upper()}/prev")
        results = data.get("results", [])
        if not results:
            return {"ticker": symbol.upper(), "error": "No results"}
        r = results[0]
        close = float(r.get("c", 0))
        open_p = float(r.get("o", 0))
        change = close - open_p
        change_pct = (change / open_p * 100) if open_p > 0 else 0
        ts = r.get("t", 0)
        as_of = datetime.fromtimestamp(ts / 1000).isoformat() if ts else ""
        return {
            "ticker": data.get("ticker", symbol.upper()),
            "price": close,
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "volume": int(r.get("v", 0)),
            "high": float(r.get("h", 0)),
            "low": float(r.get("l", 0)),
            "open": open_p,
            "prev_close": close,
            "timestamp": as_of,
        }
    except Exception as e:
        return {"ticker": symbol.upper(), "error": str(e)}


def get_quotes(symbols: list[str]) -> list[dict]:
    """Get real-time quotes for multiple tickers."""
    return [get_quote(s) for s in symbols]


# ---------------------------------------------------------------------------
# Historical bars (OHLCV)
# ---------------------------------------------------------------------------

def get_historical_prices(symbol: str, interval: str = "1d",
                          period: str = "3mo") -> list[dict]:
    """Get historical OHLCV bars from Massive.com aggregates API."""
    end = datetime.utcnow()
    period_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825}
    days = period_map.get(period, 90)
    start = end - timedelta(days=days)

    try:
        data = _get(f"/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}", {
            "adjusted": "true", "sort": "asc", "limit": 120,
        })
        results = data.get("results", [])
        return [
            {
                "timestamp": datetime.fromtimestamp(b["t"] / 1000).isoformat(),
                "open": float(b["o"]),
                "high": float(b["h"]),
                "low": float(b["l"]),
                "close": float(b["c"]),
                "volume": int(b.get("v", 0)),
            }
            for b in results
        ]
    except Exception as e:
        return [{"ticker": symbol.upper(), "error": str(e)}]


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def get_company_fundamentals(symbol: str) -> dict:
    """Get company fundamentals: sector, industry, market cap, financials."""
    try:
        data = _get(f"/fundamentals/{symbol.upper()}")
        if isinstance(data, list) and data:
            data = data[0]
        return {
            "ticker": symbol.upper(),
            "name": data.get("name", data.get("companyName", "")),
            "sector": data.get("sector", ""),
            "industry": data.get("industry", ""),
            "description": data.get("description", data.get("profile", "")),
            "market_cap": int(data.get("market_cap", data.get("marketCap", 0)) or 0),
            "pe_ratio": float(data.get("pe_ratio", data.get("peRatio", 0)) or 0),
            "eps": float(data.get("eps", data.get("earningsPerShare", 0)) or 0),
            "dividend_yield": float(data.get("dividend_yield", data.get("dividendYield", 0)) or 0),
            "beta": float(data.get("beta", 0) or 0),
            "revenue": float(data.get("revenue", data.get("totalRevenue", 0)) or 0),
            "net_income": float(data.get("net_income", data.get("netIncome", 0)) or 0),
            "total_debt": float(data.get("total_debt", data.get("totalDebt", 0)) or 0),
            "total_cash": float(data.get("total_cash", data.get("totalCash", 0)) or 0),
        }
    except Exception as e:
        return {"ticker": symbol.upper(), "error": str(e)}


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def get_news(symbol: str, limit: int = 10) -> list[dict]:
    """Get recent news articles for a ticker."""
    try:
        data = _get(f"/news/{symbol.upper()}", {"limit": limit})
        articles = data if isinstance(data, list) else data.get("news", data.get("data", data.get("results", [])))
        return [
            {
                "title": a.get("title", a.get("headline", "")),
                "summary": a.get("summary", a.get("description", a.get("snippet", ""))),
                "url": a.get("url", a.get("link", "")),
                "source": a.get("source", a.get("publisher", "")),
                "published_at": a.get("published_at", a.get("publishedAt", a.get("date", a.get("datetime", "")))),
            }
            for a in articles[:limit]
        ]
    except Exception as e:
        return [{"ticker": symbol.upper(), "error": str(e)}]


# ---------------------------------------------------------------------------
# Trades (recent trades)
# ---------------------------------------------------------------------------

def get_recent_trades(symbol: str, limit: int = 20) -> list[dict]:
    """Get recent trades for a ticker."""
    try:
        data = _get(f"/trades/{symbol.upper()}", {"limit": limit})
        trades = data if isinstance(data, list) else data.get("trades", data.get("data", []))
        return [
            {
                "price": float(t.get("price", 0)),
                "size": int(t.get("size", t.get("volume", 0)) or 0),
                "timestamp": t.get("timestamp", t.get("time", "")),
                "exchange": t.get("exchange", t.get("venue", "")),
            }
            for t in trades[:limit]
        ]
    except Exception as e:
        return [{"ticker": symbol.upper(), "error": str(e)}]
