# AI Stock Market Research Assistant

An end-to-end system that ingests stock market news from Massive API, stores it in Databricks Lakebase (Postgres), generates vector embeddings for semantic search, and provides an AI agent interface for stock research and trading.

## Architecture

```
Massive API  →  src/ingestion/  →  Lakebase (ticker_news_documents)
                                       |
                                 src/embeddings/  →  Lakebase (ticker_news_embeddings)
                                       |
                                 src/agent/  →  MCP Server (Alpaca + Massive + Paper Trading)
                                       |
                                 app/  →  Dashboard UI
```

## Project Structure

```
ai-stock-research-assistant/
├── src/
│   ├── ingestion/          # Massive API client, Lakebase connector, secrets setup
│   ├── transformations/     # News processing and transformation pipeline
│   ├── embeddings/         # Text → vector embedding pipeline (sentence-transformers)
│   └── agent/              # MCP server with Alpaca/Massive/Paper brokers
├── sql/
│   ├── schema.sql          # Combined schema (all tables + indexes)
│   ├── 01_setup_news_table.sql
│   ├── 02_setup_embeddings_table.sql
│   ├── 03_setup_chunk_embeddings_table.sql
│   └── schema_watchlist.sql
├── app/                    # Databricks Apps (dashboard, ingestion, MCP server)
├── tests/                  # Test suite
├── docs/                   # Documentation for each component
├── requirements.txt
├── .env.example
└── README.md
```

## Components

### 1. Ingestion (`src/ingestion/`)

Fetches stock news from Massive API and stores in Lakebase.

* `massive_client.py` — Massive API client (news articles, ticker data)
* `lakebase.py` — Postgres connection helper via psycopg2
* `setup_secrets.py` — Stores API keys as Databricks secrets

### 2. Embeddings (`src/embeddings/`)

Generates vector embeddings for news articles using sentence-transformers.

* `ingest_ticker_news_embeddings.py` — Pipeline: reads unembedded docs, chunks text, generates embeddings, writes to Lakebase via psycopg2

Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

### 3. Agent (`src/agent/`)

MCP (Model Context Protocol) server for stock research and trading.

* `alpaca_mcp_server.py` — Main MCP server entry point
* `alpaca_broker.py` — Alpaca trading API integration
* `massive_broker.py` — Massive news data integration
* `paper_broker.py` — Paper trading simulation

### 4. Dashboard (`app/`)

Flask web UI for viewing stock data and managing watchlists.

* `dashboard.py` — Main dashboard app
* `ingestion_app.py` — Data ingestion interface
* `alpaca_broker.py` / `paper_broker.py` — Broker integrations

## Database Schema

### `ticker_news_documents`

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Article ID |
| `ticker` | TEXT | Stock ticker symbol |
| `title` | TEXT | Article title |
| `description` | TEXT | Article summary |
| `author` | TEXT | Author name |
| `article_url` | TEXT | Source URL |
| `publisher_name` | TEXT | Publisher |
| `keywords` | JSONB | Extracted keywords |
| `sentiment` | TEXT | Sentiment label |
| `sentiment_reasoning` | TEXT | Sentiment explanation |
| `published_utc` | TIMESTAMPTZ | Publication time |
| `payload` | JSONB | Full API response |
| `synced_at` | TIMESTAMPTZ | Ingestion time |

### `ticker_news_embeddings`

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Article ID |
| `ticker` | TEXT | Stock ticker |
| `title` | TEXT | Article title |
| `published_utc` | TIMESTAMPTZ | Publication time |
| `embedding` | vector(384) | MiniLM embedding |
| `model_name` | TEXT | Model identifier |
| `embedded_at` | TIMESTAMPTZ | Embedding time |

HNSW index on `embedding` using `vector_cosine_ops`.

### `ticker_news_chunk_embeddings`

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Chunk ID |
| `article_id` | TEXT | References article |
| `ticker` | TEXT | Stock ticker |
| `chunk_index` | INT | Chunk position |
| `chunk_text` | TEXT | Text chunk |
| `embedding` | vector(384) | MiniLM embedding |
| `model_name` | TEXT | Model identifier |
| `embedded_at` | TIMESTAMPTZ | Embedding time |

### `watchlist`

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment |
| `email` | VARCHAR(255) | User email |
| `symbol` | VARCHAR(10) | Stock symbol |
| `latest_price` | DECIMAL(12,2) | Latest price |
| `updated_at` | TIMESTAMP | Last update |

## Setup

### 1. Lakebase

Create a Lakebase instance in Databricks with native (password) authentication. Store the connection URL as a secret:

```python
from databricks.sdk import WorkspaceClient
import base64

w = WorkspaceClient()
w.secrets.create_scope(scope="database")
lakebase_url = "postgresql://role:password@host:5432/databricks_postgres?sslmode=require"
w.secrets.put_secret(scope="database", key="lakebase-url", string_value=base64.b64encode(lakebase_url.encode()).decode())
```

### 2. API Keys

Store Massive and Alpaca API keys as Databricks secrets. See `.env.example` for required variables.

### 3. Database Schema

Run `sql/schema.sql` against your Lakebase Postgres database to create all tables and indexes.

### 4. Deploy

Deploy the apps from `app/` directory using Databricks Apps.

## Tech Stack

* **Databricks Lakebase** — Postgres with pgvector
* **Massive API** — Stock news data source
* **Alpaca API** — Trading (paper and live)
* **sentence-transformers** — Embedding model (all-MiniLM-L6-v2, 384-dim)
* **FastMCP** — Model Context Protocol server
* **Flask** — Web dashboard
* **psycopg2** — Postgres driver (all DB operations, no Spark JDBC)

## Documentation

Detailed docs in `docs/`:
* `day-1-ingestion.md` — Ingestion pipeline setup
* `day-2-embeddings.md` — Embedding pipeline and vector search
* `day-3-dashboard.md` — Dashboard and MCP server

## Notes

* All database writes use psycopg2 directly — no Spark JDBC
* pgvector extension required for vector similarity search
* Alpaca supports both paper and live trading (configure in secrets)