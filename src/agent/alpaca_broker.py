"""
Alpaca Markets paper-trading engine backing the Alpaca paper-trading MCP
server.

Unlike paper_broker.py (which simulated everything in Lakebase), this
module is a thin wrapper around Alpaca's real, hosted paper-trading
account via alpaca-py (https://alpaca.markets/sdks/python/). Quotes, fills,
positions, cash, and order history are all real Alpaca paper-trading data
- there is no local simulation or Lakebase bookkeeping involved.

Alpaca's paper trading is one account per API key pair (not multi-tenant),
so `account_id` is accepted on every function below purely to keep the
same signatures alpaca_mcp_server.py already calls - it is not used to select
between multiple accounts. Every call operates against the single Alpaca
paper account configured via the ALPACA_* secrets below.

Swap-in note: this module implements the trading functions
(get_quote, place_order, get_positions, get_account_summary, 
get_order_history, get_balance) that mirror paper_broker.py, 
so alpaca_mcp_server.py only needs to import this module instead.
"""

import base64
import os

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()

_SECRET_SCOPE = os.environ.get("ALPACA_SECRET_SCOPE", "database")
_KEY_ID_SECRET_KEY = os.environ.get("ALPACA_KEY_ID_SECRET_KEY", "alpaca-key-id")
_SECRET_KEY_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY_SECRET_KEY", "alpaca-secret-key")

_trading_client: TradingClient | None = None
_data_client: StockHistoricalDataClient | None = None


def _secret(key: str) -> str:
    """Fetch and base64-decode a value from the Databricks secret scope."""
    secret = _w.secrets.get_secret(scope=_SECRET_SCOPE, key=key)
    return base64.b64decode(secret.value).decode("utf-8")


def _get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        _trading_client = TradingClient(
            _secret(_KEY_ID_SECRET_KEY),
            _secret(_SECRET_KEY_SECRET_KEY),
            paper=True,
        )
    return _trading_client


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            _secret(_KEY_ID_SECRET_KEY),
            _secret(_SECRET_KEY_SECRET_KEY),
        )
    return _data_client


def get_quote(symbol: str) -> dict:
    """
    Get the latest real quote for a stock ticker symbol from Alpaca's
    market data API. Returns the bid/ask midpoint as `price`.
    """
    symbol = symbol.strip().upper()
    quotes = _get_data_client().get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=symbol)
    )
    quote = quotes[symbol]
    price = round((float(quote.bid_price) + float(quote.ask_price)) / 2, 2)
    return {
        "symbol": symbol,
        "price": price,
        "as_of": quote.timestamp.isoformat(),
    }


def place_order(account_id: str, symbol: str, side: str, quantity: float) -> dict:
    """
    Place a real market order against the single configured Alpaca paper
    trading account. `account_id` is accepted for signature compatibility
    with the mock engine but is not used to select an account - Alpaca
    paper trading is one account per API key pair.
    """
    symbol = symbol.strip().upper()
    side = side.strip().upper()
    quantity = float(quantity)

    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity!r}")

    # Crypto orders require GTC (Good-Til-Canceled) or IOC (Immediate-Or-Cancel)
    # Stock orders can use DAY
    # Crypto symbols typically end with USD (e.g., BTCUSD, ETHUSD)
    is_crypto = symbol.endswith('USD') and len(symbol) > 3
    time_in_force = TimeInForce.GTC if is_crypto else TimeInForce.DAY

    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
        time_in_force=time_in_force,
    )
    order = _get_trading_client().submit_order(order_data=order_data)
    return _order_to_dict(order)


def get_positions(account_id: str) -> list[dict]:
    """Get all open positions in the Alpaca paper trading account."""
    positions = _get_trading_client().get_all_positions()
    return [
        {
            "symbol": pos.symbol,
            "quantity": float(pos.qty),
            "avg_cost": float(pos.avg_entry_price),
            "updated_at": None,
        }
        for pos in positions
    ]


def get_account_summary(account_id: str) -> dict:
    """Cash + mark-to-market value of every open position + total equity."""
    account = _get_trading_client().get_account()
    positions = _get_trading_client().get_all_positions()

    positions_out = []
    market_value = 0.0
    for pos in positions:
        qty = float(pos.qty)
        last_price = float(pos.current_price)
        value = round(last_price * qty, 2)
        market_value += value
        positions_out.append(
            {
                "symbol": pos.symbol,
                "quantity": qty,
                "avg_cost": float(pos.avg_entry_price),
                "last_price": last_price,
                "market_value": value,
                "unrealized_pl": round(float(pos.unrealized_pl), 2),
            }
        )

    cash_balance = float(account.cash)
    return {
        "account_id": account_id,
        "cash_balance": cash_balance,
        "positions": positions_out,
        "market_value": round(market_value, 2),
        "total_equity": round(float(account.equity), 2),
    }


def get_order_history(account_id: str, limit: int = 50) -> list[dict]:
    """Get recent orders for the Alpaca paper trading account, most recent first."""
    request_params = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
    orders = _get_trading_client().get_orders(filter=request_params)
    return [_order_to_dict(order) for order in orders]


def _order_to_dict(order) -> dict:
    """Convert an alpaca-py Order object into the dict shape alpaca_mcp_server expects."""
    quantity = float(order.qty) if order.qty is not None else None
    price = float(order.filled_avg_price) if order.filled_avg_price is not None else None
    notional = round(price * quantity, 2) if price is not None and quantity is not None else None
    return {
        "id": str(order.id),
        "symbol": order.symbol,
        "side": order.side.value if order.side else None,
        "quantity": quantity,
        "price": price,
        "notional": notional,
        "status": order.status.value if order.status else None,
        "created_at": order.created_at.isoformat() if order.created_at else None,
    }


def get_stock_performance(symbol: str, days: int = 30) -> dict:
    """
    Get historical price performance for a stock ticker over the specified
    number of calendar days. Returns OHLCV bars, performance summary, and
    distinguishes between trading days (bars returned) and calendar days
    (the requested window).
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        days: Number of calendar days to look back (default 30)
    
    Returns:
        dict with:
          - ticker: The symbol
          - requested_days: Calendar days requested
          - trading_days: Actual number of trading days with data
          - start_date: First bar date (ISO format)
          - end_date: Last bar date (ISO format)
          - bars: List of OHLCV dicts [{date, open, high, low, close, volume}]
          - summary: {start_price, end_price, price_change, percent_change, high, low}
    """
    from datetime import datetime, timedelta
    
    symbol = symbol.strip().upper()
    
    # Calculate date range (calendar days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Request daily bars from Alpaca
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    try:
        bars_data = _get_data_client().get_stock_bars(request_params)
        
        if symbol not in bars_data or len(bars_data[symbol]) == 0:
            return {
                "error": f"No historical data found for {symbol}",
                "ticker": symbol,
                "requested_days": days,
                "trading_days": 0,
            }
        
        bars_list = bars_data[symbol]
        
        # Convert bars to dict format
        bars_output = []
        for bar in bars_list:
            bars_output.append({
                "date": bar.timestamp.date().isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            })
        
        # Calculate performance summary
        start_price = float(bars_list[0].open)
        end_price = float(bars_list[-1].close)
        price_change = round(end_price - start_price, 2)
        percent_change = round((price_change / start_price) * 100, 2) if start_price != 0 else 0.0
        
        # Find high and low across all bars
        all_highs = [float(bar.high) for bar in bars_list]
        all_lows = [float(bar.low) for bar in bars_list]
        period_high = round(max(all_highs), 2)
        period_low = round(min(all_lows), 2)
        
        return {
            "ticker": symbol,
            "requested_days": days,
            "trading_days": len(bars_list),
            "start_date": bars_list[0].timestamp.date().isoformat(),
            "end_date": bars_list[-1].timestamp.date().isoformat(),
            "bars": bars_output,
            "summary": {
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "price_change": price_change,
                "percent_change": percent_change,
                "period_high": period_high,
                "period_low": period_low,
            }
        }
        
    except Exception as e:
        return {
            "error": f"Failed to fetch historical data: {str(e)}",
            "ticker": symbol,
            "requested_days": days,
        }


def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get company fundamentals and key financial metrics for a stock.
    Uses yfinance to fetch data from Yahoo Finance (free, no API key required).
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    
    Returns:
        dict with:
          - ticker: The symbol
          - company_name: Full company name
          - sector: Business sector
          - industry: Industry classification
          - market_cap: Market capitalization (formatted string)
          - pe_ratio: Price-to-Earnings ratio
          - forward_pe: Forward P/E ratio
          - peg_ratio: Price/Earnings to Growth ratio
          - price_to_book: Price-to-Book ratio
          - dividend_yield: Dividend yield (as percentage)
          - revenue: Total revenue (formatted string)
          - earnings: Net income (formatted string)
          - profit_margin: Profit margin (as percentage)
          - debt_to_equity: Debt-to-Equity ratio
          - current_ratio: Current ratio (liquidity)
          - roe: Return on Equity (as percentage)
          - eps: Earnings per share (trailing twelve months)
          - beta: Stock volatility vs market
    """
    import yfinance as yf
    
    symbol = symbol.strip().upper()
    
    try:
        # Fetch ticker info
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Helper function to format large numbers
        def format_number(value):
            if value is None or value == 'N/A':
                return None
            try:
                value = float(value)
                if value >= 1_000_000_000_000:  # Trillion
                    return f"${value/1_000_000_000_000:.2f}T"
                elif value >= 1_000_000_000:  # Billion
                    return f"${value/1_000_000_000:.2f}B"
                elif value >= 1_000_000:  # Million
                    return f"${value/1_000_000:.2f}M"
                else:
                    return f"${value:,.0f}"
            except (ValueError, TypeError):
                return None
        
        # Helper to get value safely
        def safe_get(key, default=None):
            return info.get(key, default)
        
        # Helper to format percentage
        def format_pct(value):
            if value is None:
                return None
            try:
                return round(float(value) * 100, 2)  # Convert to percentage
            except (ValueError, TypeError):
                return None
        
        return {
            "ticker": symbol,
            "company_name": safe_get('longName'),
            "sector": safe_get('sector'),
            "industry": safe_get('industry'),
            "market_cap": format_number(safe_get('marketCap')),
            "market_cap_raw": safe_get('marketCap'),
            "pe_ratio": safe_get('trailingPE'),
            "forward_pe": safe_get('forwardPE'),
            "peg_ratio": safe_get('pegRatio'),
            "price_to_book": safe_get('priceToBook'),
            "dividend_yield": format_pct(safe_get('dividendYield')),
            "revenue": format_number(safe_get('totalRevenue')),
            "revenue_raw": safe_get('totalRevenue'),
            "earnings": format_number(safe_get('netIncomeToCommon')),
            "profit_margin": format_pct(safe_get('profitMargins')),
            "debt_to_equity": safe_get('debtToEquity'),
            "current_ratio": safe_get('currentRatio'),
            "roe": format_pct(safe_get('returnOnEquity')),
            "eps": safe_get('trailingEps'),
            "beta": safe_get('beta'),
            "fifty_two_week_high": safe_get('fiftyTwoWeekHigh'),
            "fifty_two_week_low": safe_get('fiftyTwoWeekLow'),
            "average_volume": safe_get('averageVolume'),
            "description": safe_get('longBusinessSummary'),
        }
        
    except Exception as e:
        return {
            "error": f"Failed to fetch fundamentals for {symbol}: {str(e)}",
            "ticker": symbol,
        }


def compare_stocks(symbols: list[str], days: int = 30) -> dict:
    """
    Compare performance of multiple stocks side-by-side over the same time period.
    Calls get_stock_performance for each ticker and returns combined results.
    
    Args:
        symbols: List of stock ticker symbols (e.g., ['AAPL', 'TSLA', 'MSFT'])
        days: Number of calendar days to look back (default 30)
    
    Returns:
        dict with:
          - requested_days: Calendar days requested
          - stocks: List of performance dicts, one per ticker (same structure as get_stock_performance)
          - comparison_summary: {best_performer, worst_performer, average_return}
    """
    if not symbols or len(symbols) == 0:
        return {"error": "At least one symbol is required"}
    
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
            "error": "No valid performance data for any of the requested symbols"
        }
    
    # Find best and worst performers by percent change
    percent_changes = [(r["ticker"], r["summary"]["percent_change"]) for r in valid_results]
    percent_changes.sort(key=lambda x: x[1], reverse=True)
    
    best_performer = percent_changes[0]
    worst_performer = percent_changes[-1]
    avg_return = round(sum(pc[1] for pc in percent_changes) / len(percent_changes), 2)
    
    return {
        "requested_days": days,
        "stocks": results,
        "comparison_summary": {
            "best_performer": {"ticker": best_performer[0], "percent_change": best_performer[1]},
            "worst_performer": {"ticker": worst_performer[0], "percent_change": worst_performer[1]},
            "average_return": avg_return,
            "stocks_compared": len(valid_results),
        }
    }
