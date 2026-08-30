-- AI Stock Market Research Assistant - Combined Schema
-- Run this against your Lakebase Postgres database to set up all tables and indexes.

-- ============================================================================
-- 1. ticker_news_documents
-- ============================================================================

CREATE TABLE IF NOT EXISTS ticker_news_documents (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    author TEXT,
    article_url TEXT,
    publisher_name TEXT,
    keywords JSONB,
    sentiment TEXT,
    sentiment_reasoning TEXT,
    published_utc TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticker_news_documents_ticker
ON ticker_news_documents (ticker);

-- ============================================================================
-- 2. pgvector extension
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 3. ticker_news_embeddings (full-article embeddings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ticker_news_embeddings (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    published_utc TIMESTAMPTZ,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticker_news_embeddings_embedding
ON ticker_news_embeddings
USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- 4. ticker_news_chunk_embeddings (chunked embeddings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ticker_news_chunk_embeddings (
    id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticker_news_chunk_embeddings_embedding
ON ticker_news_chunk_embeddings
USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- 5. watchlist
-- ============================================================================

CREATE TABLE IF NOT EXISTS watchlist (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    latest_price DECIMAL(12, 2),
    updated_at TIMESTAMP,
    CONSTRAINT unique_user_symbol UNIQUE (email, symbol)
);

CREATE INDEX IF NOT EXISTS idx_watchlist_email ON watchlist(email);
CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol);
CREATE INDEX IF NOT EXISTS idx_watchlist_updated_at ON watchlist(updated_at DESC);