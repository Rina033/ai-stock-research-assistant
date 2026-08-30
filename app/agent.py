"""
Agent module for the AI Stock Research Assistant.

Routes natural-language user messages to the appropriate capability:
  - Pull price data and summarize recent performance
  - Surface news/filings for a company
  - Compare multiple tickers on fundamentals or price action
  - Add/remove tickers from the user's watchlist
  - Save a research note or analysis report
  - Flag notable price moves or news since last visit
"""

import json
import re
from datetime import datetime

import massive_api
import db
import embeddings

# Try Databricks LLM for summaries
try:
    import mlflow.deployments  # type: ignore
    _USE_LLM = True
except Exception:
    _USE_LLM = False

_LLM_ENDPOINT = __import__("os").environ.get("LLM_ENDPOINT", "databricks-llama-4-maverick")

# Default user for single-user demo app
_DEFAULT_USER_ID = None


def _get_default_user() -> str:
    global _DEFAULT_USER_ID
    if _DEFAULT_USER_ID:
        return _DEFAULT_USER_ID
    user = db.get_or_create_user()
    _DEFAULT_USER_ID = user["id"]
    return _DEFAULT_USER_ID


def run_agent(message: str, user_id: str = None) -> str:
    """Process a user message and return a text response."""
    if not user_id:
        user_id = _get_default_user()

    msg = message.lower().strip()
    intent = _classify_intent(msg)

    handlers = {
        "add_ticker":        _handle_add_ticker,
        "remove_ticker":     _handle_remove_ticker,
        "list_watchlist":    _handle_list_watchlist,
        "price_summary":     _handle_price_summary,
        "news":              _handle_news,
        "compare":           _handle_compare,
        "fundamentals":      _handle_fundamentals,
        "save_note":         _handle_save_note,
        "flag_changes":      _handle_flag_changes,
        "semantic_search":   _handle_semantic_search,
        "help":              _handle_help,
    }

    handler = handlers.get(intent, _handle_general)
    return handler(message, user_id)


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _classify_intent(msg: str) -> str:
    """Classify user intent from message text."""
    # Watchlist management
    if re.search(r"\b(add|track|follow|watch)\b.*\b(ticker|stock|symbol|to)\b", msg) or \
       re.search(r"\badd\b\s+[A-Z]+\b", message_lookup(msg)):
        return "add_ticker"
    if re.search(r"\b(remove|delete|drop|unfollow)\b.*\b(ticker|stock|symbol|from)\b", msg):
        return "remove_ticker"
    if re.search(r"\b(show|list|view|what.*in)\b.*\bwatchlist\b", msg) or \
       re.search(r"\bmy (watchlist|stocks|tickers|holdings)\b", msg):
        return "list_watchlist"

    # Save note
    if re.search(r"\b(save|log|store|note|write)\b.*\b(note|research|analysis|report)\b", msg):
        return "save_note"

    # Compare
    if re.search(r"\b(compare|comparison|vs|versus|against|difference)\b", msg):
        return "compare"

    # News
    if re.search(r"\b(news|headline|filing|article|press|announcement)\b", msg):
        return "news"

    # Fundamentals
    if re.search(r"\b(fundamental|financial|earnings|revenue|pe ratio|market cap|balance sheet|income)\b", msg):
        return "fundamentals"

    # Flag changes
    if re.search(r"\b(what.*changed|what.*new|flag|since last|anything new|update)\b", msg):
        return "flag_changes"

    # Semantic search
    if re.search(r"\b(companies.*exposed|companies.*in|sectors? that|stocks? that|find.*similar)\b", msg):
        return "semantic_search"

    # Price summary (check after more specific intents)
    if re.search(r"\b(price|performance|chart|trend|how.*doing|summary|analysis)\b", msg):
        return "price_summary"

    # Help
    if re.search(r"\b(help|what can you do|commands|options|capabilities)\b", msg):
        return "help"

    return "price_summary"


def message_lookup(msg: str) -> str:
    """Return original-case message for ticker extraction."""
    return msg


def _extract_tickers(text: str) -> list[str]:
    """Extract uppercase ticker symbols (1-5 letters) from text."""
    # Match $TICKER or bare TICKER patterns
    found = set()
    # $TICKER pattern
    for m in re.finditer(r'\$([A-Z]{1,5})\b', text):
        found.add(m.group(1))
    # Bare ticker after keywords
    for m in re.finditer(r'\b(?:add|track|follow|watch|remove|delete|drop|compare|analyze|news for|price of|fundamentals of)\s+([A-Z]{1,5})\b', text, re.IGNORECASE):
        found.add(m.group(1).upper())
    # Multiple tickers separated by commas/and
    for m in re.finditer(r'\b([A-Z]{2,5})\b', text):
        t = m.group(1)
        # Filter out common words that look like tickers
        if t not in {"THE", "AND", "FOR", "NOT", "ALL", "NEW", "ONE", "TWO", "BUT", "ARE", "WAS", "HIS", "HER", "HAS", "HAD"}:
            found.add(t)
    return sorted(found)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _llm_summarize(prompt: str, max_tokens: int = 500) -> str:
    """Call Databricks LLM endpoint for text generation."""
    if not _USE_LLM:
        return ""
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        resp = client.predict_endpoint(
            name=_LLM_ENDPOINT,
            inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        )
        # Extract text from response
        if isinstance(resp, dict):
            choices = resp.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return resp.get("content", resp.get("text", ""))
        return str(resp)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_add_ticker(message: str, user_id: str) -> str:
    tickers = _extract_tickers(message)
    if not tickers:
        return "Which ticker would you like to add? e.g. 'Add AAPL to my watchlist'"

    # Get or create default watchlist
    watchlists = db.get_watchlists(user_id)
    if not watchlists:
        wl = db.create_watchlist(user_id, "My Watchlist", "Default watchlist")
        watchlist_id = wl if isinstance(wl, dict) else {"id": wl}
        # Re-fetch
        watchlists = db.get_watchlists(user_id)
    watchlist_id = watchlists[0]["id"]

    added = []
    for t in tickers:
        db.add_ticker(watchlist_id, t)
        added.append(t)
        # Fetch and store company info
        try:
            fund = massive_api.get_company_fundamentals(t)
            if "error" not in fund:
                emb = embeddings.create_embedding(
                    f"{fund.get('name', '')} {fund.get('sector', '')} {fund.get('industry', '')} {fund.get('description', '')}"
                )
                db.upsert_company(
                    ticker=t,
                    name=fund.get("name"),
                    sector=fund.get("sector"),
                    industry=fund.get("industry"),
                    description=fund.get("description"),
                    market_cap=fund.get("market_cap"),
                    profile=f"{fund.get('name', '')} - {fund.get('sector', '')} - {fund.get('industry', '')}",
                    embedding=embeddings.serialize_embedding(emb),
                )
        except Exception:
            pass

    return f"Added {', '.join(added)} to your watchlist."


def _handle_remove_ticker(message: str, user_id: str) -> str:
    tickers = _extract_tickers(message)
    if not tickers:
        return "Which ticker would you like to remove? e.g. 'Remove AAPL from my watchlist'"

    watchlists = db.get_watchlists(user_id)
    if not watchlists:
        return "You don't have any watchlists yet."
    watchlist_id = watchlists[0]["id"]

    removed = []
    for t in tickers:
        count = db.remove_ticker(watchlist_id, t)
        if count:
            removed.append(t)
    if removed:
        return f"Removed {', '.join(removed)} from your watchlist."
    return f"Ticker(s) {', '.join(tickers)} not found in your watchlist."


def _handle_list_watchlist(message: str, user_id: str) -> str:
    watchlists = db.get_watchlists(user_id)
    if not watchlists:
        return "You don't have any watchlists yet. Try 'Add AAPL to my watchlist'."

    lines = []
    for wl in watchlists:
        tickers = db.get_watchlist_tickers(wl["id"])
        ticker_symbols = [t["ticker"] for t in tickers]
        lines.append(f"**{wl['name']}** ({len(ticker_symbols)} tickers)")
        if ticker_symbols:
            # Get latest prices
            prices = db.get_latest_prices(ticker_symbols)
            for sym in ticker_symbols:
                p = prices.get(sym, {})
                if p and "price" in p:
                    change_str = f"{p['change_percent']:+.2f}%" if p.get("change_percent") is not None else ""
                    lines.append(f"  {sym}: ${p['price']:.2f} {change_str}")
                else:
                    lines.append(f"  {sym}: (no price data yet)")
        else:
            lines.append("  (empty)")
        lines.append("")
    return "\n".join(lines)


def _handle_price_summary(message: str, user_id: str) -> str:
    tickers = _extract_tickers(message)
    if not tickers:
        return "Which ticker would you like to analyze? e.g. 'Summarize AAPL performance'"

    t = tickers[0]
    quote = massive_api.get_quote(t)
    if "error" in quote:
        return f"Could not fetch data for {t}: {quote['error']}"

    # Save price snapshot
    db.save_price_snapshot(
        t, quote.get("price", 0), quote.get("change"),
        quote.get("change_percent"), quote.get("volume")
    )

    # Get historical data
    bars = massive_api.get_historical_prices(t, "1d", "3mo")

    summary_parts = [
        f"**{t}** — ${quote['price']:.2f}",
        f"Change: {quote['change']:+.2f} ({quote['change_percent']:+.2f}%)",
        f"Day Range: ${quote.get('low', 0):.2f} - ${quote.get('high', 0):.2f}",
        f"Volume: {quote.get('volume', 0):,}",
    ]

    if bars and len(bars) > 1 and "error" not in bars[0]:
        prices = [b["close"] for b in bars]
        high_3mo = max(prices)
        low_3mo = min(prices)
        start_price = prices[0]
        end_price = prices[-1]
        total_return = ((end_price - start_price) / start_price * 100) if start_price else 0
        summary_parts.append(f"\n3-Month Performance:")
        summary_parts.append(f"  Return: {total_return:+.2f}%")
        summary_parts.append(f"  Range: ${low_3mo:.2f} - ${high_3mo:.2f}")

    # Try LLM summary
    llm_prompt = f"""Summarize the following stock data for {t} in 2-3 sentences:
Current price: ${quote['price']:.2f}
Change: {quote['change']:+.2f} ({quote['change_percent']:+.2f}%)
Volume: {quote.get('volume', 0):,}
"""
    if bars and len(bars) > 1 and "error" not in bars[0]:
        llm_prompt += f"3-month return: {total_return:+.2f}%\n3-month range: ${low_3mo:.2f} - ${high_3mo:.2f}\n"

    llm_summary = _llm_summarize(llm_prompt)
    if llm_summary:
        summary_parts.append(f"\n{llm_summary}")

    # Save analysis report
    db.save_analysis_report(user_id, t, "price_summary", "\n".join(summary_parts))

    return "\n".join(summary_parts)


def _handle_news(message: str, user_id: str) -> str:
    tickers = _extract_tickers(message)
    if not tickers:
        return "Which ticker would you like news for? e.g. 'News for AAPL'"

    t = tickers[0]
    articles = massive_api.get_news(t, limit=5)
    if not articles or (len(articles) == 1 and "error" in articles[0]):
        return f"No recent news found for {t}."

    lines = [f"**Recent News for {t}:**"]
    for a in articles[:5]:
        title = a.get("title", "Untitled")
        source = a.get("source", "")
        url = a.get("url", "")
        summary = a.get("summary", "")
        lines.append(f"\n• {title}")
        if source:
            lines.append(f"  Source: {source}")
        if summary:
            lines.append(f"  {summary[:200]}" if len(summary) > 200 else f"  {summary}")
        if url:
            lines.append(f"  {url}")

        # Save to DB with embedding
        try:
            emb = embeddings.create_embedding(f"{title} {summary}")
            db.save_news_article(
                ticker=t, title=title, summary=summary, url=url,
                source=source, published_at=a.get("published_at"),
                embedding=embeddings.serialize_embedding(emb),
            )
        except Exception:
            pass

    return "\n".join(lines)


def _handle_compare(message: str, user_id: str) -> str:
    tickers = _extract_tickers(message)
    if len(tickers) < 2:
        return "Which tickers would you like to compare? e.g. 'Compare AAPL and MSFT'"

    lines = [f"**Comparing {', '.join(tickers)}:**\n"]

    # Fetch quotes and fundamentals
    for t in tickers:
        quote = massive_api.get_quote(t)
        fund = massive_api.get_company_fundamentals(t)

        if "error" in quote:
            lines.append(f"{t}: Unable to fetch data ({quote['error']})\n")
            continue

        db.save_price_snapshot(
            t, quote.get("price", 0), quote.get("change"),
            quote.get("change_percent"), quote.get("volume")
        )

        lines.append(f"**{t}** — ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")
        if "error" not in fund:
            lines.append(f"  Sector: {fund.get('sector', 'N/A')}")
            lines.append(f"  Industry: {fund.get('industry', 'N/A')}")
            lines.append(f"  Market Cap: ${fund.get('market_cap', 0) / 1e9:.1f}B" if fund.get('market_cap') else "  Market Cap: N/A")
            lines.append(f"  P/E Ratio: {fund.get('pe_ratio', 'N/A')}")
            lines.append(f"  Revenue: ${fund.get('revenue', 0) / 1e9:.1f}B" if fund.get('revenue') else "  Revenue: N/A")
        lines.append("")

    return "\n".join(lines)


def _handle_fundamentals(message: str, user_id: str) -> str:
    tickers = _extract_tickers(message)
    if not tickers:
        return "Which ticker would you like fundamentals for? e.g. 'Fundamentals for AAPL'"

    t = tickers[0]
    fund = massive_api.get_company_fundamentals(t)
    if "error" in fund:
        return f"Could not fetch fundamentals for {t}: {fund['error']}"

    # Store/update company in DB
    emb = embeddings.create_embedding(
        f"{fund.get('name', '')} {fund.get('sector', '')} {fund.get('industry', '')} {fund.get('description', '')}"
    )
    db.upsert_company(
        ticker=t, name=fund.get("name"), sector=fund.get("sector"),
        industry=fund.get("industry"), description=fund.get("description"),
        market_cap=fund.get("market_cap"),
        profile=f"{fund.get('name', '')} - {fund.get('sector', '')} - {fund.get('industry', '')}",
        embedding=embeddings.serialize_embedding(emb),
    )

    lines = [f"**{t} Fundamentals:**"]
    lines.append(f"  Company: {fund.get('name', 'N/A')}")
    lines.append(f"  Sector: {fund.get('sector', 'N/A')}")
    lines.append(f"  Industry: {fund.get('industry', 'N/A')}")
    if fund.get('market_cap'):
        lines.append(f"  Market Cap: ${fund['market_cap'] / 1e9:.1f}B")
    if fund.get('pe_ratio'):
        lines.append(f"  P/E Ratio: {fund['pe_ratio']:.2f}")
    if fund.get('eps'):
        lines.append(f"  EPS: ${fund['eps']:.2f}")
    if fund.get('dividend_yield'):
        lines.append(f"  Dividend Yield: {fund['dividend_yield']:.2f}%")
    if fund.get('beta'):
        lines.append(f"  Beta: {fund['beta']:.2f}")
    if fund.get('revenue'):
        lines.append(f"  Revenue: ${fund['revenue'] / 1e9:.1f}B")
    if fund.get('net_income'):
        lines.append(f"  Net Income: ${fund['net_income'] / 1e9:.1f}B")
    if fund.get('description'):
        lines.append(f"\n  {fund['description'][:500]}")

    return "\n".join(lines)


def _handle_save_note(message: str, user_id: str) -> str:
    tickers = _extract_tickers(message)
    t = tickers[0] if tickers else None

    # Extract a title and content from the message
    # Simple approach: use the message itself as the note content
    note_text = re.sub(r'^(save|log|store|write)\s+(a\s+)?(note|research|analysis|report)\s*(for|on|about)?\s*', '', message, flags=re.IGNORECASE).strip()
    if not note_text:
        return "What would you like to save? e.g. 'Save a note: AAPL looks overvalued given declining revenue'"

    title = note_text[:60] + ("..." if len(note_text) > 60 else "")
    db.save_research_note(user_id, t, title, note_text)
    return f"Research note saved{f' for {t}' if t else ''}: {title}"


def _handle_flag_changes(message: str, user_id: str) -> str:
    watchlists = db.get_watchlists(user_id)
    if not watchlists:
        return "You don't have any watchlists yet. Add tickers first!"

    watchlist_id = watchlists[0]["id"]
    tickers = db.get_watchlist_tickers(watchlist_id)
    if not tickers:
        return "Your watchlist is empty. Add tickers to start tracking!"

    lines = ["**Notable Changes Since Your Last Visit:**\n"]
    any_changes = False

    for wt in tickers:
        t = wt["ticker"]
        quote = massive_api.get_quote(t)
        if "error" in quote:
            continue

        # Save current snapshot
        db.save_price_snapshot(
            t, quote.get("price", 0), quote.get("change"),
            quote.get("change_percent"), quote.get("volume")
        )

        change_pct = quote.get("change_percent", 0)
        if abs(change_pct) >= 2.0:
            any_changes = True
            direction = "up" if change_pct > 0 else "down"
            lines.append(f"  **{t}** is {direction} {abs(change_pct):.2f}% to ${quote['price']:.2f}")

        # Check for recent news
        articles = massive_api.get_news(t, limit=3)
        for a in articles[:2]:
            if "error" not in a:
                any_changes = True
                lines.append(f"  {t} news: {a.get('title', '')}")

    if not any_changes:
        lines.append("  No significant changes detected. All quiet on the watchlist.")

    return "\n".join(lines)


def _handle_semantic_search(message: str, user_id: str) -> str:
    # Extract the search query
    query = re.sub(r'^(find|search|show|list)\s+(companies|stocks)\s*(that|exposed to|in)?', '', message, flags=re.IGNORECASE).strip()
    if not query:
        query = message

    query_emb = embeddings.create_embedding(query)
    companies = db.search_companies_by_embedding(query_emb, limit=5)

    if not companies:
        return f"No companies found matching '{query}'. Try adding tickers to your watchlist first to build the company database."

    lines = [f"**Companies matching '{query}':**\n"]
    for c in companies:
        lines.append(f"  {c['ticker']} — {c.get('name', 'N/A')} ({c.get('sector', 'N/A')})")
        if c.get('description'):
            lines.append(f"    {c['description'][:150]}")
        lines.append("")

    return "\n".join(lines)


def _handle_help(message: str, user_id: str) -> str:
    return """**AI Stock Research Assistant**\n\nI can help you with:\n\n* **Watchlist Management**\n  - "Add AAPL to my watchlist"\n  - "Remove MSFT from my watchlist"\n  - "Show my watchlist"\n\n* **Market Data & Analysis**\n  - "Summarize AAPL performance"\n  - "Compare AAPL and MSFT"\n  - "Fundamentals for GOOGL"\n\n* **News & Research**\n  - "News for AAPL"\n  - "Save a note: AAPL looks overvalued"\n  - "What's changed since my last visit?"\n\n* **Semantic Search**\n  - "Find companies exposed to rising interest rates"\n  - "Companies in the regional banking sector"\n\nJust type a question or command!"""


def _handle_general(message: str, user_id: str) -> str:
    return _handle_help(message, user_id)
