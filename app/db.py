"""
Database module: schema initialization and CRUD for the AI Stock Research
Assistant. All tables live in a dedicated schema inside the Lakebase
Postgres database.
"""

import os
import json
import uuid
from datetime import datetime

import lakebase

SCHEMA = "stock_research"

# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

_SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {SCHEMA}.users (
    id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    email        TEXT UNIQUE,
    display_name TEXT,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.watchlists (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT NOT NULL REFERENCES {SCHEMA}.users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.watchlist_tickers (
    id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    watchlist_id TEXT NOT NULL REFERENCES {SCHEMA}.watchlists(id) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    added_at     TIMESTAMPTZ DEFAULT now(),
    UNIQUE (watchlist_id, ticker)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.companies (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    industry    TEXT,
    description TEXT,
    market_cap  BIGINT,
    profile     TEXT,
    embedding   TEXT,
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.price_snapshots (
    id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    ticker         TEXT NOT NULL,
    price          NUMERIC(18, 4),
    change         NUMERIC(18, 4),
    change_percent NUMERIC(10, 4),
    volume         BIGINT,
    snapshot_time  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.news_articles (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    ticker      TEXT,
    title       TEXT NOT NULL,
    summary     TEXT,
    url         TEXT,
    source      TEXT,
    published_at TIMESTAMPTZ,
    embedding   TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.research_notes (
    id         TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id    TEXT REFERENCES {SCHEMA}.users(id) ON DELETE SET NULL,
    ticker     TEXT,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.analysis_reports (
    id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id     TEXT REFERENCES {SCHEMA}.users(id) ON DELETE SET NULL,
    ticker      TEXT,
    report_type TEXT,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_wl_user    ON {SCHEMA}.watchlists(user_id);
CREATE INDEX IF NOT EXISTS idx_wlt_wl     ON {SCHEMA}.watchlist_tickers(watchlist_id);
CREATE INDEX IF NOT EXISTS idx_ps_ticker  ON {SCHEMA}.price_snapshots(ticker, snapshot_time DESC);
CREATE INDEX IF NOT EXISTS idx_na_ticker   ON {SCHEMA}.news_articles(ticker, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_rn_user    ON {SCHEMA}.research_notes(user_id);
CREATE INDEX IF NOT EXISTS idx_ar_user     ON {SCHEMA}.analysis_reports(user_id);
"""


def init_schema():
    """Create all tables and indexes if they don't exist."""
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
            conn.commit()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_or_create_user(user_id: str = None, email: str = None, display_name: str = None) -> dict:
    """Return an existing user or atomically create one by email."""
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(f"SELECT * FROM {SCHEMA}.users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    return dict(row)
            # Upsert by email so repeated and concurrent requests are safe.
            cur.execute(
                f"""
                INSERT INTO {SCHEMA}.users (email, display_name)
                VALUES (%s, %s)
                ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
                RETURNING *
                """,
                (email or "default@local", display_name or "Default User"),
            )
            conn.commit()
            return dict(cur.fetchone())


# ---------------------------------------------------------------------------
# Watchlists
# ---------------------------------------------------------------------------

def get_watchlists(user_id: str) -> list[dict]:
    rows = lakebase.run_query(
        f"SELECT * FROM {SCHEMA}.watchlists WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
    )
    return [dict(r) for r in rows]


def create_watchlist(user_id: str, name: str, description: str = None) -> dict:
    return lakebase.run_write(
        f"INSERT INTO {SCHEMA}.watchlists (user_id, name, description) VALUES (%s, %s, %s) RETURNING *",
        (user_id, name, description),
    )


def get_watchlist_tickers(watchlist_id: str) -> list[dict]:
    rows = lakebase.run_query(
        f"SELECT * FROM {SCHEMA}.watchlist_tickers WHERE watchlist_id = %s ORDER BY added_at DESC",
        (watchlist_id,),
    )
    return [dict(r) for r in rows]


def add_ticker(watchlist_id: str, ticker: str) -> bool:
    """Add a ticker to a watchlist. Returns True if inserted, False if already present."""
    try:
        lakebase.run_write(
            f"INSERT INTO {SCHEMA}.watchlist_tickers (watchlist_id, ticker) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (watchlist_id, ticker.upper()),
        )
        return True
    except Exception:
        return False


def remove_ticker(watchlist_id: str, ticker: str) -> int:
    return lakebase.run_write(
        f"DELETE FROM {SCHEMA}.watchlist_tickers WHERE watchlist_id = %s AND ticker = %s",
        (watchlist_id, ticker.upper()),
    )


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

def upsert_company(ticker: str, name: str = None, sector: str = None,
                    industry: str = None, description: str = None,
                    market_cap: int = None, profile: str = None,
                    embedding: str = None) -> dict:
    """Insert or update a company record."""
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SCHEMA}.companies (ticker, name, sector, industry, description, market_cap, profile, embedding, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (ticker) DO UPDATE SET
                    name = COALESCE(EXCLUDED.name, companies.name),
                    sector = COALESCE(EXCLUDED.sector, companies.sector),
                    industry = COALESCE(EXCLUDED.industry, companies.industry),
                    description = COALESCE(EXCLUDED.description, companies.description),
                    market_cap = COALESCE(EXCLUDED.market_cap, companies.market_cap),
                    profile = COALESCE(EXCLUDED.profile, companies.profile),
                    embedding = COALESCE(EXCLUDED.embedding, companies.embedding),
                    updated_at = now()
                RETURNING *
                """,
                (ticker.upper(), name, sector, industry, description, market_cap, profile, embedding),
            )
            conn.commit()
            return dict(cur.fetchone())


def get_company(ticker: str) -> dict | None:
    rows = lakebase.run_query(
        f"SELECT * FROM {SCHEMA}.companies WHERE ticker = %s",
        (ticker.upper(),),
    )
    return dict(rows[0]) if rows else None


# ---------------------------------------------------------------------------
# Price snapshots
# ---------------------------------------------------------------------------

def save_price_snapshot(ticker: str, price: float, change: float = None,
                         change_percent: float = None, volume: int = None):
    lakebase.run_write(
        f"INSERT INTO {SCHEMA}.price_snapshots (ticker, price, change, change_percent, volume) VALUES (%s, %s, %s, %s, %s)",
        (ticker.upper(), price, change, change_percent, volume),
    )


def get_price_snapshots(ticker: str, limit: int = 10) -> list[dict]:
    rows = lakebase.run_query(
        f"SELECT * FROM {SCHEMA}.price_snapshots WHERE ticker = %s ORDER BY snapshot_time DESC LIMIT %s",
        (ticker.upper(), limit),
    )
    return [dict(r) for r in rows]


def get_latest_prices(tickers: list[str]) -> dict:
    """Get the most recent price snapshot for each ticker."""
    if not tickers:
        return {}
    result = {}
    for t in tickers:
        rows = lakebase.run_query(
            f"SELECT * FROM {SCHEMA}.price_snapshots WHERE ticker = %s ORDER BY snapshot_time DESC LIMIT 1",
            (t.upper(),),
        )
        if rows:
            result[t.upper()] = dict(rows[0])
    return result


# ---------------------------------------------------------------------------
# News articles
# ---------------------------------------------------------------------------

def save_news_article(ticker: str, title: str, summary: str = None,
                       url: str = None, source: str = None,
                       published_at=None, embedding: str = None) -> dict:
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SCHEMA}.news_articles (ticker, title, summary, url, source, published_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *
                """,
                (ticker.upper() if ticker else None, title, summary, url, source, published_at, embedding),
            )
            conn.commit()
            return dict(cur.fetchone())


def get_news(ticker: str, limit: int = 10) -> list[dict]:
    rows = lakebase.run_query(
        f"SELECT * FROM {SCHEMA}.news_articles WHERE ticker = %s ORDER BY published_at DESC NULLS LAST LIMIT %s",
        (ticker.upper(), limit),
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Research notes
# ---------------------------------------------------------------------------

def save_research_note(user_id: str, ticker: str, title: str, content: str) -> dict:
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SCHEMA}.research_notes (user_id, ticker, title, content)
                VALUES (%s, %s, %s, %s) RETURNING *
                """,
                (user_id, ticker.upper() if ticker else None, title, content),
            )
            conn.commit()
            return dict(cur.fetchone())


def get_research_notes(user_id: str, ticker: str = None, limit: int = 20) -> list[dict]:
    if ticker:
        rows = lakebase.run_query(
            f"SELECT * FROM {SCHEMA}.research_notes WHERE user_id = %s AND ticker = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, ticker.upper(), limit),
        )
    else:
        rows = lakebase.run_query(
            f"SELECT * FROM {SCHEMA}.research_notes WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Analysis reports
# ---------------------------------------------------------------------------

def save_analysis_report(user_id: str, ticker: str, report_type: str, content: str) -> dict:
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {SCHEMA}.analysis_reports (user_id, ticker, report_type, content)
                VALUES (%s, %s, %s, %s) RETURNING *
                """,
                (user_id, ticker.upper() if ticker else None, report_type, content),
            )
            conn.commit()
            return dict(cur.fetchone())


def get_analysis_reports(user_id: str, ticker: str = None, limit: int = 20) -> list[dict]:
    if ticker:
        rows = lakebase.run_query(
            f"SELECT * FROM {SCHEMA}.analysis_reports WHERE user_id = %s AND ticker = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, ticker.upper(), limit),
        )
    else:
        rows = lakebase.run_query(
            f"SELECT * FROM {SCHEMA}.analysis_reports WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Semantic search (simple cosine on JSON-serialized vectors stored in TEXT)
# ---------------------------------------------------------------------------

def search_companies_by_embedding(query_embedding: list[float], limit: int = 5) -> list[dict]:
    """Search companies by embedding similarity. Embeddings stored as JSON text."""
    import math
    rows = lakebase.run_query(f"SELECT * FROM {SCHEMA}.companies WHERE embedding IS NOT NULL")
    scored = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
            sim = _cosine_sim(query_embedding, emb)
            scored.append((sim, dict(row)))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def search_news_by_embedding(query_embedding: list[float], limit: int = 5) -> list[dict]:
    """Search news articles by embedding similarity."""
    rows = lakebase.run_query(f"SELECT * FROM {SCHEMA}.news_articles WHERE embedding IS NOT NULL")
    scored = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
            sim = _cosine_sim(query_embedding, emb)
            scored.append((sim, dict(row)))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
