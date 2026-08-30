"""
Alpaca Markets paper-trading MCP server.

Exposes paper-trading tools over MCP (Model Context Protocol) so a
Databricks Agent Bricks agent can call them like any other tool:
    - get_quote(symbol)
    - place_trade(account_id, symbol, side, quantity)
    - get_positions(account_id)
    - get_account_summary(account_id)
    - get_order_history(account_id, limit)
    - get_balance(account_id)
    - get_current_user()

These tools are backed by Alpaca Markets' real, hosted paper-trading
account (see alpaca_broker.py), so students can safely wire an Agent
Bricks agent to place real (but fake-money) trades without a real
brokerage account or risk of real money moving. account_id is accepted
for signature compatibility but is not used to select an account - Alpaca
paper trading is one account per API key pair.

Swap-in-a-real-broker note: to point this at a different broker instead,
keep the same 5 tool signatures below and replace the alpaca_broker.*
calls inside each tool with calls to that broker's SDK/API - the MCP
surface for the agent does not need to change. The original Lakebase-
simulated engine is preserved in paper_broker.py for reference.

Deploy this as its own Databricks App (same app.yaml + FastMCP entrypoint
pattern documented at
https://docs.databricks.com/aws/en/agents/mcp-tools/custom-mcp), separate
from the dashboard app, so an Agent Bricks agent (or any MCP client) can
register its URL as an external MCP server.

Run locally:
    python alpaca_mcp_server.py
"""

import os
import logging
import random
import uuid
from datetime import datetime
from contextvars import ContextVar
from functools import wraps

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

import alpaca_broker
import massive_broker

# Lazy imports for DB-dependent modules — only loaded when first needed,
# so the server starts and non-DB tools work even if Lakebase is down.
_lakebase = None
_sentence_transformers = None
_db_available = True  # Flipped to False after first DB failure to skip retries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpaca-mcp-server")

# Load embedding model once at startup
_embedding_model = None

def get_embedding_model():
    """Lazy-load the embedding model (expensive operation, only on first use)."""
    global _embedding_model, _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer as _ST
        _sentence_transformers = _ST
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = _sentence_transformers(EMBEDDING_MODEL)
    return _embedding_model

# Table names from environment variables
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "ticker_news_embeddings")
CHUNK_EMBEDDINGS_TABLE_NAME = os.environ.get("CHUNK_EMBEDDINGS_TABLE_NAME", "ticker_news_chunk_embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Context variable to store request headers for accessing end-user identity
_request_context: ContextVar[dict] = ContextVar('request_context', default={})

# Context variable to store session ID for each request
_session_id: ContextVar[str] = ContextVar('session_id', default=None)

# In-memory storage for staged trades awaiting confirmation
_staged_trades: dict[str, dict] = {}


def _get_end_user_email() -> str:
    """Get the actual end user's email from request headers, or fallback to service principal."""
    # Try to get from X-Forwarded-User header (Databricks App context)
    headers = _request_context.get()
    forwarded_user = headers.get('x-forwarded-user')
    if forwarded_user:
        return forwarded_user
    
    # Fallback: use service principal (local development or non-App contexts)
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    return w.current_user.me().user_name or 'zach@dataexpert.io'


def _get_lakebase():
    """Lazy-import the lakebase module. Returns None if DB is known to be down."""
    global _lakebase, _db_available
    if not _db_available:
        return None
    if _lakebase is None:
        import lakebase as _lb
        _lakebase = _lb
    return _lakebase


def _log_trace(tool_name: str, args: dict, result: any, error: str = None, duration_ms: float = None):
    """Log MCP tool call to Lakebase Postgres tracing table with standardized result format.
    Silently skips when DB is unavailable so non-DB tools are unaffected."""
    global _db_available
    if not _db_available:
        return  # DB already known to be down — skip silently
    
    lb = _get_lakebase()
    if lb is None:
        return
    
    try:
        import json
        
        # Get session_id from context, or generate a new one if not set
        session_id = _session_id.get()
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        user_email = _get_end_user_email()
        
        # Convert args to JSON string for storage
        args_json = json.dumps(args) if args else None
        
        # Create standardized result structure
        if error:
            standardized_result = {
                "tool_name": tool_name,
                "status": "error",
                "message": error,
                "data": result if result else None
            }
        else:
            standardized_result = {
                "tool_name": tool_name,
                "status": "success",
                "message": f"Successfully executed {tool_name}",
                "data": result
            }
        
        result_json = json.dumps(standardized_result)
        
        sql = """
        INSERT INTO mcp_traces (
            session_id, 
            tool_name, 
            user_email, 
            args, 
            result, 
            error, 
            duration_ms, 
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        lb.run_write(
            sql,
            (
                session_id,
                tool_name,
                user_email,
                args_json,
                result_json,
                error,
                duration_ms,
            ),
        )
        
    except Exception as e:
        # DB is down — flip the flag so future calls skip without retrying
        _db_available = False
        logger.warning(f"Tracing disabled (DB unavailable): {tool_name} — {str(e)}")


def traced_tool(func):
    """Decorator to add tracing to MCP tool calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        import inspect
        
        start_time = time.time()
        tool_name = func.__name__
        
        # Capture arguments
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        call_args = dict(bound_args.arguments)
        
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            _log_trace(tool_name, call_args, result, error=None, duration_ms=duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            _log_trace(tool_name, call_args, None, error=error_msg, duration_ms=duration_ms)
            raise
    
    return wrapper


mcp = FastMCP("alpaca-paper-trading")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to capture HTTP headers containing end-user identity and generate session ID."""
    async def dispatch(self, request: Request, call_next):
        # Generate unique session ID for this request
        session_id = str(uuid.uuid4())
        _session_id.set(session_id)
        
        # Capture headers that Databricks injects with user identity
        headers = {
            'x-forwarded-user': request.headers.get('x-forwarded-user'),
            'x-forwarded-email': request.headers.get('x-forwarded-email'),
        }
        _request_context.set(headers)
        response = await call_next(request)
        return response


@mcp.tool
def get_quote(symbol: str) -> dict:
    """
    Get the latest real quote for a stock ticker symbol from Massive.com.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".

    Returns:
        A dict with symbol, price, as_of (ISO timestamp), volume, change, and change_percent.
    """
    import time
    start_time = time.time()
    
    try:
        result = massive_broker.get_quote(symbol)
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('get_quote', {'symbol': symbol}, result, error=None, duration_ms=duration_ms)
        return result
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('get_quote', {'symbol': symbol}, None, error=str(e), duration_ms=duration_ms)
        raise


@mcp.tool
def get_stock_performance(symbol: str, days: int = 30) -> dict:
    """
    Get historical price performance for a stock ticker over a specified time period.
    Returns daily OHLCV data, performance summary, and distinguishes between
    trading days (actual bars) and calendar days (requested window).
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        days: Number of calendar days to look back (default 30, max 365)
    
    Returns:
        A dict with:
          - ticker: The stock symbol
          - requested_days: Calendar days requested
          - trading_days: Actual number of trading days with data
          - start_date: First trading day in the period
          - end_date: Last trading day in the period  
          - bars: List of daily OHLCV bars [{date, open, high, low, close, volume}]
          - summary: Performance metrics {start_price, end_price, price_change, 
                     percent_change, period_high, period_low}
    
    Example:
        get_stock_performance("AAPL", 7) returns 5-7 trading days for the past week,
        distinguishing that 7 calendar days != 7 trading days due to weekends/holidays.
    """
    import time
    start_time = time.time()
    
    # Validate and cap days parameter
    if days <= 0:
        return {"error": "days must be a positive integer"}
    days = min(days, 365)  # Cap at 1 year
    
    try:
        result = alpaca_broker.get_stock_performance(symbol, days)
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('get_stock_performance', {'symbol': symbol, 'days': days}, result, error=None, duration_ms=duration_ms)
        return result
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        _log_trace('get_stock_performance', {'symbol': symbol, 'days': days}, None, error=error_msg, duration_ms=duration_ms)
        return {"error": error_msg, "ticker": symbol, "requested_days": days}


@mcp.tool
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get company fundamentals and key financial metrics for a stock.
    Returns comprehensive fundamental data including market cap, P/E ratio, 
    revenue, earnings, profitability, and valuation metrics.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
    
    Returns:
        A dict with:
          - Company info: name, sector, industry
          - Valuation metrics: market cap, P/E ratio, PEG ratio, price-to-book
          - Profitability: profit margin, ROE, EPS
          - Financial health: debt-to-equity, current ratio
          - Dividends: dividend yield
          - Trading metrics: beta, 52-week high/low
    
    Example:
        get_stock_fundamentals("AAPL") returns comprehensive Apple Inc. fundamentals.
    """
    import time
    start_time = time.time()
    
    try:
        result = alpaca_broker.get_stock_fundamentals(symbol)
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('get_stock_fundamentals', {'symbol': symbol}, result, error=None, duration_ms=duration_ms)
        return result
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        _log_trace('get_stock_fundamentals', {'symbol': symbol}, None, error=error_msg, duration_ms=duration_ms)
        return {"error": error_msg, "ticker": symbol}


@mcp.tool
def compare_stocks(symbols: list[str], days: int = 30, include_fundamentals: bool = True) -> dict:
    """
    Compare performance of multiple stocks side-by-side.
    Returns both price performance and fundamental metrics for comprehensive comparison.
    
    Args:
        symbols: List of stock ticker symbols to compare (e.g., ["AAPL", "TSLA", "MSFT"])
        days: Number of calendar days for price performance (default 30, max 365)
        include_fundamentals: Whether to include fundamental metrics comparison (default True)
    
    Returns:
        A dict with:
          - requested_days: Calendar days requested
          - stocks: List containing both performance and fundamentals for each ticker
          - price_comparison: Summary of best/worst price performers
          - fundamentals_comparison: Summary comparing key metrics across stocks (if enabled)
    
    Example:
        compare_stocks(["AAPL", "MSFT"], 30) returns side-by-side 30-day performance
        plus fundamentals (P/E, market cap, profit margins) for both stocks.
    """
    import time
    start_time = time.time()
    
    # Validate symbols
    if not symbols or not isinstance(symbols, list) or len(symbols) == 0:
        return {"error": "symbols must be a non-empty list of ticker symbols"}
    
    # Validate and cap days parameter
    if days <= 0:
        return {"error": "days must be a positive integer"}
    days = min(days, 365)  # Cap at 1 year
    
    try:
        # Get price performance comparison
        price_result = alpaca_broker.compare_stocks(symbols, days)
        
        # Add fundamentals for each stock if requested
        if include_fundamentals and "stocks" in price_result:
            for stock_data in price_result["stocks"]:
                if "error" not in stock_data:
                    ticker = stock_data["ticker"]
                    fundamentals = alpaca_broker.get_stock_fundamentals(ticker)
                    stock_data["fundamentals"] = fundamentals
            
            # Create fundamentals comparison summary
            valid_stocks = [s for s in price_result["stocks"] if "error" not in s and "fundamentals" in s and "error" not in s["fundamentals"]]
            
            if valid_stocks:
                fundamentals_summary = {}
                
                # Compare market caps
                market_caps = [(s["ticker"], s["fundamentals"].get("market_cap_raw")) for s in valid_stocks if s["fundamentals"].get("market_cap_raw")]
                if market_caps:
                    market_caps.sort(key=lambda x: x[1], reverse=True)
                    fundamentals_summary["largest_market_cap"] = {"ticker": market_caps[0][0], "value": market_caps[0][1]}
                
                # Compare P/E ratios
                pe_ratios = [(s["ticker"], s["fundamentals"].get("pe_ratio")) for s in valid_stocks if s["fundamentals"].get("pe_ratio")]
                if pe_ratios:
                    pe_ratios.sort(key=lambda x: x[1])
                    fundamentals_summary["lowest_pe"] = {"ticker": pe_ratios[0][0], "value": pe_ratios[0][1]}
                    fundamentals_summary["highest_pe"] = {"ticker": pe_ratios[-1][0], "value": pe_ratios[-1][1]}
                
                # Compare profit margins
                profit_margins = [(s["ticker"], s["fundamentals"].get("profit_margin")) for s in valid_stocks if s["fundamentals"].get("profit_margin")]
                if profit_margins:
                    profit_margins.sort(key=lambda x: x[1], reverse=True)
                    fundamentals_summary["highest_profit_margin"] = {"ticker": profit_margins[0][0], "value": profit_margins[0][1]}
                
                # Compare ROE
                roes = [(s["ticker"], s["fundamentals"].get("roe")) for s in valid_stocks if s["fundamentals"].get("roe")]
                if roes:
                    roes.sort(key=lambda x: x[1], reverse=True)
                    fundamentals_summary["highest_roe"] = {"ticker": roes[0][0], "value": roes[0][1]}
                
                price_result["fundamentals_comparison"] = fundamentals_summary
        
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('compare_stocks', {'symbols': symbols, 'days': days, 'include_fundamentals': include_fundamentals}, price_result, error=None, duration_ms=duration_ms)
        return price_result
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        _log_trace('compare_stocks', {'symbols': symbols, 'days': days}, None, error=error_msg, duration_ms=duration_ms)
        return {"error": error_msg, "requested_days": days}


@mcp.tool
def get_stock_performance_massive(symbol: str, days: int = 30) -> dict:
    """
    Get historical price performance for a stock ticker using Massive Stocks API.
    This tool uses Massive.com's aggregate bars API, which provides reliable
    historical OHLCV data without the SIP data restrictions of the free tier.

    Returns daily OHLCV data, performance summary, and distinguishes between
    trading days (actual bars) and calendar days (requested window).

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "TSLA")
        days: Number of calendar days to look back (default 30, max 365)

    Returns:
        A dict with:
          - ticker: The stock symbol
          - requested_days: Calendar days requested
          - trading_days: Actual number of trading days with data
          - start_date: First trading day in the period
          - end_date: Last trading day in the period
          - bars: List of daily OHLCV bars [{date, open, high, low, close, volume, vwap, transactions}]
          - summary: Performance metrics {start_price, end_price, price_change,
                     percent_change, period_high, period_low}

    Example:
        get_stock_performance_massive("AAPL", 7) returns 5-7 trading days for the past week,
        distinguishing that 7 calendar days != 7 trading days due to weekends/holidays.
    """
    import time
    start_time = time.time()

    # Validate and cap days parameter
    if days <= 0:
        return {"error": "days must be a positive integer"}
    days = min(days, 365)  # Cap at 1 year

    try:
        result = massive_broker.get_stock_performance(symbol, days)
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('get_stock_performance_massive', {'symbol': symbol, 'days': days}, result, error=None, duration_ms=duration_ms)
        return result
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        _log_trace('get_stock_performance_massive', {'symbol': symbol, 'days': days}, None, error=error_msg, duration_ms=duration_ms)
        return {"error": error_msg, "ticker": symbol.upper(), "requested_days": days}


@mcp.tool
def compare_stocks_massive(symbols: list[str], days: int = 30, include_fundamentals: bool = True) -> dict:
    """
    Compare performance of multiple stocks side-by-side using Massive Stocks API.
    This tool uses Massive.com's aggregate bars API, which provides reliable
    historical OHLCV data without the SIP data restrictions of the free tier.

    Returns both price performance and fundamental metrics for comprehensive comparison.

    Args:
        symbols: List of stock ticker symbols to compare (e.g., ["AAPL", "TSLA", "MSFT"])
        days: Number of calendar days for price performance (default 30, max 365)
        include_fundamentals: Whether to include fundamental metrics comparison (default True)

    Returns:
        A dict with:
          - requested_days: Calendar days requested
          - stocks: List containing both performance and fundamentals for each ticker
          - comparison_summary: Summary of best/worst price performers
          - fundamentals_comparison: Summary comparing key metrics across stocks (if enabled)

    Example:
        compare_stocks_massive(["AAPL", "MSFT"], 30) returns side-by-side 30-day performance
        plus fundamentals (P/E, market cap, profit margins) for both stocks.
    """
    import time
    start_time = time.time()

    # Validate symbols
    if not symbols or not isinstance(symbols, list) or len(symbols) == 0:
        return {"error": "symbols must be a non-empty list of ticker symbols"}

    # Validate and cap days parameter
    if days <= 0:
        return {"error": "days must be a positive integer"}
    days = min(days, 365)  # Cap at 1 year

    try:
        # Get price performance comparison using Massive API
        price_result = massive_broker.compare_stocks(symbols, days)

        # Add fundamentals for each stock if requested
        if include_fundamentals and "stocks" in price_result:
            for stock_data in price_result["stocks"]:
                if "error" not in stock_data:
                    ticker = stock_data["ticker"]
                    fundamentals = alpaca_broker.get_stock_fundamentals(ticker)
                    stock_data["fundamentals"] = fundamentals

            # Create fundamentals comparison summary
            valid_stocks = [s for s in price_result["stocks"] if "error" not in s and "fundamentals" in s and "error" not in s["fundamentals"]]

            if valid_stocks:
                fundamentals_summary = {}

                # Compare market caps
                market_caps = [(s["ticker"], s["fundamentals"].get("market_cap_raw")) for s in valid_stocks if s["fundamentals"].get("market_cap_raw")]
                if market_caps:
                    market_caps.sort(key=lambda x: x[1], reverse=True)
                    fundamentals_summary["largest_market_cap"] = {"ticker": market_caps[0][0], "value": market_caps[0][1]}

                # Compare P/E ratios
                pe_ratios = [(s["ticker"], s["fundamentals"].get("pe_ratio")) for s in valid_stocks if s["fundamentals"].get("pe_ratio")]
                if pe_ratios:
                    pe_ratios.sort(key=lambda x: x[1])
                    fundamentals_summary["lowest_pe"] = {"ticker": pe_ratios[0][0], "value": pe_ratios[0][1]}
                    fundamentals_summary["highest_pe"] = {"ticker": pe_ratios[-1][0], "value": pe_ratios[-1][1]}

                # Compare profit margins
                profit_margins = [(s["ticker"], s["fundamentals"].get("profit_margin")) for s in valid_stocks if s["fundamentals"].get("profit_margin")]
                if profit_margins:
                    profit_margins.sort(key=lambda x: x[1], reverse=True)
                    fundamentals_summary["highest_profit_margin"] = {"ticker": profit_margins[0][0], "value": profit_margins[0][1]}

                # Compare ROE
                roes = [(s["ticker"], s["fundamentals"].get("roe")) for s in valid_stocks if s["fundamentals"].get("roe")]
                if roes:
                    roes.sort(key=lambda x: x[1], reverse=True)
                    fundamentals_summary["highest_roe"] = {"ticker": roes[0][0], "value": roes[0][1]}

                price_result["fundamentals_comparison"] = fundamentals_summary

        duration_ms = (time.time() - start_time) * 1000
        _log_trace('compare_stocks_massive', {'symbols': symbols, 'days': days, 'include_fundamentals': include_fundamentals}, price_result, error=None, duration_ms=duration_ms)
        return price_result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        _log_trace('compare_stocks_massive', {'symbols': symbols, 'days': days}, None, error=error_msg, duration_ms=duration_ms)
        return {"error": error_msg, "requested_days": days}


@mcp.tool
def stage_trade(account_id: str, symbol: str, side: str, quantity: float) -> dict:
    """
    Stage a trade for review - looks up the current price, calculates estimated
    cost, and generates a 5-digit confirmation code. No actual trade is executed.
    
    The user must call execute_trade with the confirmation code to complete the trade.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account (Alpaca paper trading is one account per
            API key pair).
        symbol: Stock ticker symbol, e.g. "AAPL".
        side: "BUY" or "SELL".
        quantity: Number of shares to trade (must be positive).

    Returns:
        A dict with trade summary (symbol, side, quantity, current_price,
        estimated_cost) and a 5-digit confirmation_code.
    """
    import time
    start_time = time.time()
    call_args = {'account_id': account_id, 'symbol': symbol, 'side': side, 'quantity': quantity}
    
    try:
        # Get current quote
        quote = massive_broker.get_quote(symbol)
        current_price = quote["price"]
        
        # Calculate estimated cost
        estimated_cost = current_price * quantity
        
        # Generate 5-digit confirmation code
        confirmation_code = f"{random.randint(10000, 99999)}"
        
        # Store staged trade
        _staged_trades[confirmation_code] = {
            "account_id": account_id,
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "staged_price": current_price,
            "estimated_cost": estimated_cost,
        }
        
        result = {
            "status": "staged",
            "message": f"Trade staged. Call execute_trade with confirmation code to proceed.",
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "current_price": current_price,
            "estimated_cost": estimated_cost,
            "confirmation_code": confirmation_code,
        }
        
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('stage_trade', call_args, result, error=None, duration_ms=duration_ms)
        return result
        
    except Exception as e:
        logger.exception(f"Failed to stage trade for {symbol}")
        error_result = {
            "status": "error",
            "message": f"Failed to stage trade: {str(e)}",
        }
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('stage_trade', call_args, error_result, error=str(e), duration_ms=duration_ms)
        return error_result


@mcp.tool
def execute_trade(confirmation_code: str) -> dict:
    """
    Execute a previously staged trade using the 5-digit confirmation code.
    
    The trade must have been staged first using stage_trade. This places a real
    market order (paper trade) against the configured Alpaca paper trading account.

    Args:
        confirmation_code: 5-digit code returned by stage_trade.

    Returns:
        A dict describing the executed order (id, symbol, side, quantity,
        price, notional, status, created_at) or an error if the code is invalid.
    """
    import time
    start_time = time.time()
    call_args = {'confirmation_code': confirmation_code}
    
    if confirmation_code not in _staged_trades:
        error_result = {
            "status": "error",
            "message": f"Invalid or expired confirmation code: {confirmation_code}",
        }
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('execute_trade', call_args, error_result, error="Invalid confirmation code", duration_ms=duration_ms)
        return error_result
    
    try:
        # Retrieve staged trade
        staged = _staged_trades[confirmation_code]
        
        # Execute the trade
        result = alpaca_broker.place_order(
            staged["account_id"],
            staged["symbol"],
            staged["side"],
            staged["quantity"],
        )
        
        # Remove from staged trades after successful execution
        del _staged_trades[confirmation_code]
        
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('execute_trade', call_args, result, error=None, duration_ms=duration_ms)
        return result
        
    except Exception as e:
        logger.exception(f"Failed to execute trade with code {confirmation_code}")
        # Keep the staged trade in case user wants to retry
        error_result = {
            "status": "error",
            "message": f"Failed to execute trade: {str(e)}",
        }
        duration_ms = (time.time() - start_time) * 1000
        _log_trace('execute_trade', call_args, error_result, error=str(e), duration_ms=duration_ms)
        return error_result


@mcp.tool
def get_positions(account_id: str) -> list[dict]:
    """
    Get all open positions for the Alpaca paper trading account.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A list of dicts, each with symbol, quantity, avg_cost, updated_at.
    """
    return alpaca_broker.get_positions(account_id)


@mcp.tool
def get_account_summary(account_id: str) -> dict:
    """
    Get a full account summary for the Alpaca paper trading account: cash
    balance, open positions marked-to-market, total market value, and
    total equity (cash + market value).

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A dict with account_id, cash_balance, positions, market_value,
        total_equity.
    """
    return alpaca_broker.get_account_summary(account_id)


@mcp.tool
def get_order_history(account_id: str, limit: int = 50) -> list[dict]:
    """
    Get recent orders for the Alpaca paper trading account, most recent first.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.
        limit: Max number of orders to return (default 50).

    Returns:
        A list of dicts, each with id, symbol, side, quantity, price,
        notional, status, created_at.
    """
    return alpaca_broker.get_order_history(account_id, limit)


@mcp.tool
def get_balance(account_id: str) -> dict:
    """
    Get the current cash balance and buying power for the Alpaca paper 
    trading account.

    Args:
        account_id: Accepted for signature compatibility; not used to
            select an account.

    Returns:
        A dict with account_id, cash_balance, buying_power, and currency.
    """
    return alpaca_broker.get_account_summary(account_id)


@mcp.tool
def get_current_user() -> dict:
    """
    Get information about the currently authenticated end user accessing the MCP server.
    
    When running as a Databricks App, this returns the actual end user making the
    request (from X-Forwarded-User header), not the service principal running the app.

    Returns:
        A dict with user_name (email from X-Forwarded-User header), 
        forwarded_email, and source ("request_header" or "service_principal").
    """
    try:
        # First, try to get the end user from the request headers
        # Databricks injects X-Forwarded-User with the actual user's email
        headers = _request_context.get()
        forwarded_user = headers.get('x-forwarded-user')
        forwarded_email = headers.get('x-forwarded-email')
        
        if forwarded_user:
            return {
                "status": "success",
                "user_name": forwarded_user,
                "forwarded_email": forwarded_email,
                "source": "request_header",
            }
        
        # Fallback: return the service principal if headers aren't available
        # (e.g., when running locally or in non-App contexts)
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        user = w.current_user.me()
        return {
            "status": "success",
            "user_name": user.user_name,
            "display_name": user.display_name,
            "active": user.active,
            "source": "service_principal",
        }
    except Exception as e:
        logger.exception("Failed to get current user")
        return {
            "status": "error",
            "message": f"Failed to get current user: {str(e)}",
        }


@mcp.tool
def add_to_watchlist(symbol: str, email: str = 'jiyue.gu2@gmail.com') -> dict:
    """
    Add a stock to the watchlist by fetching its current quote from Massive.com
    and storing it in the Lakebase watchlist table.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
    
    Returns:
        A dict with the quote data and confirmation that it was added to the watchlist.
    """
    lb = _get_lakebase()
    if lb is None:
        return {"status": "error", "message": "Database unavailable — watchlist features require a live Lakebase endpoint."}
    
    try:
        # Get the actual end user's email (not the service principal)
        user_email = email
        
        # Get quote from Massive.com
        quote = massive_broker.get_quote(symbol)
        
        # Store in Lakebase watchlist table
        sql = """
        INSERT INTO watchlist (email, symbol, latest_price, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (email, symbol) 
        DO UPDATE SET 
            latest_price = EXCLUDED.latest_price,
            updated_at = NOW()
        """
        
        lb.run_write(
            sql,
            (
                user_email,
                quote["symbol"],
                quote["price"]
            ),
        )
        
        return {
            "status": "success",
            "message": f"Added {symbol} to watchlist for {user_email}",
            "user_email": user_email,
            "quote": quote,
        }
    except Exception as e:
        logger.exception(f"Failed to add {symbol} to watchlist")
        return {
            "status": "error",
            "message": f"Failed to add {symbol} to watchlist: {str(e)}",
        }


@mcp.tool
def get_watchlist(limit: int = 100, email: str = 'zach@dataexpert.io') -> dict:
    """
    Retrieve all stocks in the authenticated user's watchlist from Lakebase.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        limit: Maximum number of entries to return (default: 100).
        email: authenticate user's email
    
    Returns:
        A dict with watchlist entries sorted by most recently added.
    """
    lb = _get_lakebase()
    if lb is None:
        return {"status": "error", "message": "Database unavailable — watchlist features require a live Lakebase endpoint."}
    
    try:
        sql = """
        SELECT 
            symbol,
            latest_price,
            updated_at
        FROM watchlist
        WHERE email = %s
        LIMIT %s
        """
        
        rows = lb.run_query(sql, (email, limit))
        
        return {
            "status": "success",
            "user_email": email,
            "count": len(rows),
            "watchlist": rows,
        }
    except Exception as e:
        logger.exception(f"Failed to retrieve watchlist")
        return {
            "status": "error",
            "message": f"Failed to retrieve watchlist: {str(e)}",
        }


@mcp.tool
def remove_from_watchlist(symbol: str) -> dict:
    """
    Remove a stock from the authenticated user's watchlist.
    
    Uses the authenticated user's email as the user_id.
    
    Args:
        symbol: Stock ticker symbol to remove, e.g. "AAPL".
    
    Returns:
        A dict with status and confirmation message.
    """
    lb = _get_lakebase()
    if lb is None:
        return {"status": "error", "message": "Database unavailable — watchlist features require a live Lakebase endpoint."}
    
    try:
        user_email = _get_end_user_email()
        
        symbol = symbol.strip().upper()
        
        sql = """
        DELETE FROM watchlist
        WHERE email = %s AND symbol = %s
        """
        
        rows_affected = lb.run_write(sql, (user_email, symbol))
        
        if rows_affected > 0:
            return {
                "status": "success",
                "message": f"Removed {symbol} from watchlist",
                "symbol": symbol,
                "user_email": user_email,
            }
        else:
            return {
                "status": "not_found",
                "message": f"{symbol} was not in the watchlist",
                "symbol": symbol,
                "user_email": user_email,
            }
    except Exception as e:
        logger.exception(f"Failed to remove {symbol} from watchlist")
        return {
            "status": "error",
            "message": f"Failed to remove {symbol} from watchlist: {str(e)}",
        }


@mcp.tool
def get_stock_information(stock_query: str, limit: int = 10, search_chunks: bool = True) -> dict:
    """
    Queries news information for a given stock
    
    Accepts a text query, computes its embedding, and returns the most similar
    documents and chunks from Lakebase using pgvector's cosine similarity.
    
    Args:
        query: Natural language search query (e.g. "tech company earnings")
        limit: Maximum number of results to return (default 10)
        search_chunks: Whether to search chunk-level embeddings in addition to documents
    
    Returns:
        A dict with query, documents, chunks, and model name
    """
    if not stock_query or not stock_query.strip():
        return {"error": "Query text is required"}
    
    lb = _get_lakebase()
    if lb is None:
        return {"error": "Database unavailable — news search requires a live Lakebase endpoint with pgvector embeddings."}
    
    try:
        # Compute embedding for the query
        model = get_embedding_model()
        query_embedding = model.encode(stock_query)
        
        # Convert to list for JSON serialization and postgres array format
        embedding_list = query_embedding.tolist()
        
        # Search document-level embeddings
        doc_results = lb.run_query(
            f"""
            SELECT 
                e.id,
                e.ticker,
                e.title,
                e.published_utc,
                e.model_name,
                1 - (e.embedding <=> %s::vector) as similarity,
                d.description,
                d.article_url,
                d.sentiment
            FROM {EMBEDDINGS_TABLE_NAME} e
            LEFT JOIN {NEWS_TABLE_NAME} d ON e.id = d.id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (str(embedding_list), str(embedding_list), limit),
        )
        
        chunk_results = []
        if search_chunks:
            # Search chunk-level embeddings
            chunk_results = lb.run_query(
                f"""
                SELECT 
                    c.id,
                    c.article_id,
                    c.ticker,
                    c.chunk_index,
                    c.chunk_text,
                    c.model_name,
                    1 - (c.embedding <=> %s::vector) as similarity,
                    d.title,
                    d.article_url,
                    d.published_utc
                FROM {CHUNK_EMBEDDINGS_TABLE_NAME} c
                LEFT JOIN {NEWS_TABLE_NAME} d ON c.article_id = d.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (str(embedding_list), str(embedding_list), limit),
            )
        
        return {
            "query": stock_query,
            "documents": doc_results,
            "chunks": chunk_results,
            "model": EMBEDDING_MODEL
        }
        
    except Exception as e:
        logger.exception("Vector search failed")
        return {"error": str(e)}


@mcp.tool
def save_research_note(ticker: str, note_type: str, content: str, email: str = 'jiyue.gu2@gmail.com') -> dict:
    """
    Save a research note or analysis report for a specific stock ticker.
    Notes are stored in Lakebase and associated with the user's email.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        note_type: Type of note - one of: "analysis", "thesis", "alert", "general"
        content: The note content (markdown or plain text)
        email: User email (defaults to authenticated user)
    
    Returns:
        A dict with status, note_id, and timestamp of the saved note.
    """
    from datetime import datetime
    import uuid
    
    ticker = ticker.strip().upper()
    note_type = note_type.strip().lower()
    
    # Validate note_type
    valid_types = ["analysis", "thesis", "alert", "general"]
    if note_type not in valid_types:
        return {
            "status": "error",
            "message": f"Invalid note_type. Must be one of: {', '.join(valid_types)}"
        }
    
    if not content or not content.strip():
        return {
            "status": "error",
            "message": "Note content cannot be empty"
        }
    
    lb = _get_lakebase()
    if lb is None:
        return {"status": "error", "message": "Database unavailable — research notes require a live Lakebase endpoint."}
    
    try:
        note_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        # Insert into research_notes table
        lb.run_write(
            """
            INSERT INTO research_notes (id, email, ticker, note_type, content, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (note_id, email, ticker, note_type, content, created_at)
        )
        
        return {
            "status": "success",
            "note_id": note_id,
            "ticker": ticker,
            "note_type": note_type,
            "created_at": created_at,
            "message": f"Research note saved for {ticker}"
        }
        
    except Exception as e:
        logger.exception(f"Failed to save research note for {ticker}")
        return {
            "status": "error",
            "message": f"Failed to save research note: {str(e)}"
        }


@mcp.tool
def get_research_notes(ticker: str = None, note_type: str = None, limit: int = 50, email: str = 'jiyue.gu2@gmail.com') -> dict:
    """
    Retrieve research notes for the authenticated user. Can filter by ticker and/or note type.
    
    Args:
        ticker: Optional - filter by stock ticker symbol
        note_type: Optional - filter by note type ("analysis", "thesis", "alert", "general")
        limit: Maximum number of notes to return (default 50)
        email: User email (defaults to authenticated user)
    
    Returns:
        A dict with list of notes, sorted by most recent first.
    """
    lb = _get_lakebase()
    if lb is None:
        return {"status": "error", "message": "Database unavailable — research notes require a live Lakebase endpoint."}
    
    try:
        # Build query with optional filters
        where_clauses = ["email = %s"]
        params = [email]
        
        if ticker:
            where_clauses.append("ticker = %s")
            params.append(ticker.strip().upper())
        
        if note_type:
            where_clauses.append("note_type = %s")
            params.append(note_type.strip().lower())
        
        where_sql = " AND ".join(where_clauses)
        params.append(limit)
        
        notes = lb.run_query(
            f"""
            SELECT id, ticker, note_type, content, created_at
            FROM research_notes
            WHERE {where_sql}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            tuple(params)
        )
        
        return {
            "status": "success",
            "count": len(notes),
            "notes": notes
        }
        
    except Exception as e:
        logger.exception("Failed to retrieve research notes")
        return {
            "status": "error",
            "message": f"Failed to retrieve notes: {str(e)}"
        }


@mcp.tool
def get_alerts_since_last_visit(email: str = 'jiyue.gu2@gmail.com', price_change_threshold: float = 5.0) -> dict:
    """
    Get alerts for significant price movements and new news since the user's last visit.
    Checks watchlist stocks for:
      1. Price changes exceeding the threshold (default 5%)
      2. New news articles published since last visit
    
    Args:
        email: User email (defaults to authenticated user)
        price_change_threshold: Percentage change threshold for alerts (default 5.0%)
    
    Returns:
        A dict with:
          - last_visit: Timestamp of user's previous visit
          - current_time: Current timestamp
          - price_alerts: List of stocks with significant price moves
          - news_alerts: List of new news articles since last visit
          - watchlist_count: Number of stocks checked
    """
    from datetime import datetime, timedelta
    
    lb = _get_lakebase()
    if lb is None:
        return {"status": "error", "message": "Database unavailable — alerts require a live Lakebase endpoint."}
    
    try:
        # Get user's last visit time
        visit_records = lb.run_query(
            "SELECT last_visit_at, previous_visit_at FROM user_visits WHERE email = %s",
            (email,)
        )
        
        if not visit_records or not visit_records[0].get('previous_visit_at'):
            # First visit or no previous visit recorded
            # Update visit time for next time
            lb.run_write(
                """
                INSERT INTO user_visits (email, last_visit_at, previous_visit_at)
                VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (email) DO UPDATE
                SET previous_visit_at = user_visits.last_visit_at,
                    last_visit_at = CURRENT_TIMESTAMP
                """,
                (email,)
            )
            
            return {
                "status": "success",
                "message": "First visit recorded. No previous alerts available.",
                "last_visit": None,
                "current_time": datetime.utcnow().isoformat(),
                "price_alerts": [],
                "news_alerts": [],
                "watchlist_count": 0
            }
        
        last_visit = visit_records[0]['previous_visit_at']
        current_time = datetime.utcnow()
        
        # Get user's watchlist
        watchlist = lb.run_query(
            "SELECT DISTINCT symbol FROM watchlist WHERE email = %s",
            (email,)
        )
        
        if not watchlist:
            return {
                "status": "success",
                "message": "Watchlist is empty",
                "last_visit": last_visit.isoformat() if hasattr(last_visit, 'isoformat') else str(last_visit),
                "current_time": current_time.isoformat(),
                "price_alerts": [],
                "news_alerts": [],
                "watchlist_count": 0
            }
        
        # Calculate days since last visit for price performance lookup
        days_since_visit = max(1, (current_time - last_visit).days + 1)
        
        # Check for significant price moves on watchlist stocks
        price_alerts = []
        for stock in watchlist:
            symbol = stock['symbol']
            try:
                perf = massive_broker.get_stock_performance(symbol, days_since_visit)
                
                if 'summary' in perf:
                    pct_change = perf['summary']['percent_change']
                    
                    if abs(pct_change) >= price_change_threshold:
                        direction = "up" if pct_change > 0 else "down"
                        price_alerts.append({
                            "ticker": symbol,
                            "percent_change": pct_change,
                            "price_change": perf['summary']['price_change'],
                            "current_price": perf['summary']['end_price'],
                            "direction": direction,
                            "alert_type": "significant_move",
                            "message": f"{symbol} moved {direction} {abs(pct_change):.2f}% since your last visit"
                        })
            except Exception as e:
                logger.warning(f"Failed to check price for {symbol}: {str(e)}")
        
        # Check for new news articles since last visit
        news_alerts = []
        try:
            new_news = lb.run_query(
                f"""
                SELECT DISTINCT ticker, title, published_utc, article_url, sentiment
                FROM {NEWS_TABLE_NAME}
                WHERE ticker IN %s
                  AND published_utc > %s
                ORDER BY published_utc DESC
                LIMIT 50
                """,
                (tuple(s['symbol'] for s in watchlist), last_visit)
            )
            
            news_alerts = [{
                "ticker": article['ticker'],
                "title": article['title'],
                "published_utc": article['published_utc'].isoformat() if hasattr(article['published_utc'], 'isoformat') else str(article['published_utc']),
                "article_url": article.get('article_url'),
                "sentiment": article.get('sentiment'),
                "alert_type": "new_news"
            } for article in new_news]
        except Exception as e:
            logger.warning(f"Failed to fetch news alerts: {str(e)}")
        
        # Update user's visit timestamp
        lb.run_write(
            """
            UPDATE user_visits
            SET previous_visit_at = last_visit_at,
                last_visit_at = CURRENT_TIMESTAMP
            WHERE email = %s
            """,
            (email,)
        )
        
        return {
            "status": "success",
            "last_visit": last_visit.isoformat() if hasattr(last_visit, 'isoformat') else str(last_visit),
            "current_time": current_time.isoformat(),
            "days_since_visit": days_since_visit,
            "price_alerts": price_alerts,
            "price_alerts_count": len(price_alerts),
            "news_alerts": news_alerts,
            "news_alerts_count": len(news_alerts),
            "watchlist_count": len(watchlist),
            "threshold": price_change_threshold
        }
        
    except Exception as e:
        logger.exception("Failed to generate alerts")
        return {
            "status": "error",
            "message": f"Failed to generate alerts: {str(e)}"
        }


if __name__ == "__main__":
    # Add middleware to capture request headers for end-user identity
    # This must be done before mcp.run() is called
    if hasattr(mcp, 'app') and mcp.app is not None:
        mcp.app.add_middleware(RequestContextMiddleware)
    
    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects
    # (see the "Host your own MCP" doc linked in the module docstring above).
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)
