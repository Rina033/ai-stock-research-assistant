"""AI Stock Research Assistant — Flask entry point.

Users track a personal watchlist of tickers, describe an investing thesis
or question, and ask an agent to pull real market data, summarize company
fundamentals and news, and log the analysis.
"""
import os
import logging

from flask import Flask, jsonify, render_template, request

import agent
import db
import massive_api

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder=".")

# ---------------------------------------------------------------------------
# Initialize database schema on startup
# ---------------------------------------------------------------------------

_schema_initialized = False


def _ensure_schema():
    global _schema_initialized
    if not _schema_initialized:
        try:
            db.init_schema()
            _schema_initialized = True
            log.info("Database schema initialized.")
        except Exception as e:
            log.error("Failed to initialize schema: %s", e)


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    _ensure_schema()
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — Chat / Agent
# ---------------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():
    _ensure_schema()
    data = request.get_json()
    message = data.get("message", "").strip()
    user_id = data.get("user_id")
    if not message:
        return jsonify({"response": "Please enter a question or command."}), 400
    try:
        response = agent.run_agent(message, user_id)
        return jsonify({"response": response})
    except Exception as e:
        log.exception("Agent error")
        return jsonify({"response": f"Error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Routes — Watchlist API
# ---------------------------------------------------------------------------

@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    _ensure_schema()
    user = db.get_or_create_user()
    watchlists = db.get_watchlists(user["id"])
    if not watchlists:
        return jsonify({"watchlists": []})
    result = []
    for wl in watchlists:
        tickers = db.get_watchlist_tickers(wl["id"])
        result.append({
            "id": wl["id"],
            "name": wl["name"],
            "description": wl.get("description"),
            "tickers": [t["ticker"] for t in tickers],
        })
    return jsonify({"watchlists": result})


@app.route("/api/watchlist/tickers", methods=["POST"])
def add_ticker():
    _ensure_schema()
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    user = db.get_or_create_user()
    watchlists = db.get_watchlists(user["id"])
    if not watchlists:
        wl = db.create_watchlist(user["id"], "My Watchlist", "Default watchlist")
        watchlists = db.get_watchlists(user["id"])
    watchlist_id = watchlists[0]["id"]
    db.add_ticker(watchlist_id, ticker)
    # Fetch company info
    try:
        fund = massive_api.get_company_fundamentals(ticker)
        if "error" not in fund:
            import embeddings
            emb = embeddings.create_embedding(
                f"{fund.get('name', '')} {fund.get('sector', '')} {fund.get('industry', '')} {fund.get('description', '')}"
            )
            db.upsert_company(
                ticker=ticker, name=fund.get("name"), sector=fund.get("sector"),
                industry=fund.get("industry"), description=fund.get("description"),
                market_cap=fund.get("market_cap"),
                profile=f"{fund.get('name', '')} - {fund.get('sector', '')}",
                embedding=embeddings.serialize_embedding(emb),
            )
    except Exception:
        pass
    return jsonify({"status": "added", "ticker": ticker})


@app.route("/api/watchlist/tickers/<ticker>", methods=["DELETE"])
def delete_ticker(ticker):
    _ensure_schema()
    user = db.get_or_create_user()
    watchlists = db.get_watchlists(user["id"])
    if not watchlists:
        return jsonify({"error": "no watchlist"}), 404
    watchlist_id = watchlists[0]["id"]
    count = db.remove_ticker(watchlist_id, ticker.upper())
    return jsonify({"status": "removed" if count else "not_found", "ticker": ticker.upper()})


# ---------------------------------------------------------------------------
# Routes — Market Data
# ---------------------------------------------------------------------------

@app.route("/api/quotes/<ticker>")
def get_quote(ticker):
    _ensure_schema()
    quote = massive_api.get_quote(ticker)
    if "error" not in quote:
        db.save_price_snapshot(
            ticker, quote.get("price", 0), quote.get("change"),
            quote.get("change_percent"), quote.get("volume")
        )
    return jsonify(quote)


@app.route("/api/quotes")
def get_quotes():
    _ensure_schema()
    tickers = request.args.get("tickers", "")
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    quotes = massive_api.get_quotes(symbols)
    for q in quotes:
        if "error" not in q:
            db.save_price_snapshot(
                q["ticker"], q.get("price", 0), q.get("change"),
                q.get("change_percent"), q.get("volume")
            )
    return jsonify(quotes)


# ---------------------------------------------------------------------------
# Routes — Research Notes
# ---------------------------------------------------------------------------

@app.route("/api/research", methods=["GET"])
def get_research():
    _ensure_schema()
    user = db.get_or_create_user()
    ticker = request.args.get("ticker")
    notes = db.get_research_notes(user["id"], ticker)
    return jsonify({"notes": notes})


@app.route("/api/research", methods=["POST"])
def save_research():
    _ensure_schema()
    data = request.get_json()
    user = db.get_or_create_user()
    note = db.save_research_note(
        user["id"],
        data.get("ticker", ""),
        data.get("title", "Untitled"),
        data.get("content", ""),
    )
    return jsonify({"status": "saved", "note": note})


# ---------------------------------------------------------------------------
# Routes — Analysis Reports
# ---------------------------------------------------------------------------

@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    _ensure_schema()
    user = db.get_or_create_user()
    ticker = request.args.get("ticker")
    reports = db.get_analysis_reports(user["id"], ticker)
    return jsonify({"reports": reports})


# ---------------------------------------------------------------------------
# Routes — Health
# ---------------------------------------------------------------------------

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/debug")
def debug_env():
    """Debug endpoint — print PG* env vars for Lakebase connection."""
    import os
    pg_vars = {k: os.environ.get(k) for k in [
        "PGHOST", "PGDATABASE", "PGUSER", "PGPORT",
        "DATABRICKS_CLIENT_ID", "DATABRICKS_HOST",
    ]}
    return jsonify(pg_vars)


# ---------------------------------------------------------------------------
# Routes — Setup helper
# ---------------------------------------------------------------------------

@app.route("/setup")
def setup():
    """Debug endpoint — initialize schema and check DB connectivity."""
    _ensure_schema()
    info = {"schema": db.SCHEMA, "initialized": _schema_initialized}
    try:
        rows = db.lakebase.run_query("SELECT current_database(), current_user()")
        info["db"] = dict(rows[0]) if rows else "connected"
    except Exception as e:
        info["db_error"] = str(e)
    return jsonify(info)


@app.route("/enable-endpoint")
def enable_endpoint():
    """Enable the Lakebase endpoint if disabled."""
    import requests as _req
    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        return jsonify({"error": "DATABRICKS_HOST or DATABRICKS_TOKEN not set"}), 500
    base = f"{host}/api/2.0/postgres/projects/ai-stock-assistant/branches/production/endpoints"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # List endpoints
    resp = _req.get(base, headers=headers)
    results = []
    for ep in resp.json().get("endpoints", []):
        name = ep.get("name", "")
        disabled = ep.get("status", {}).get("disabled", False)
        ep_id = ep.get("endpoint_id", "")
        results.append({"endpoint": name, "disabled": disabled})
        if disabled:
            # Enable it
            patch_resp = _req.patch(
                f"{base}/{ep_id}",
                headers=headers,
                json={
                    "endpoint": {
                        "name": name,
                        "endpoint_id": ep_id,
                        "spec": {
                            "endpoint_type": "ENDPOINT_TYPE_READ_WRITE",
                            "autoscaling_limit_min_cu": 1.0,
                            "autoscaling_limit_max_cu": 1.0,
                            "disabled": False,
                        }
                    },
                    "update_mask": "spec.disabled,spec.endpoint_type,spec.autoscaling_limit_min_cu,spec.autoscaling_limit_max_cu",
                },
            )
            results.append({"action": "enable", "status_code": patch_resp.status_code, "response": patch_resp.text[:500]})
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", 8000)))
