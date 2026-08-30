# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 02 — News Ingestion
# MAGIC
# MAGIC Fetch news articles from the Massive Stocks API and write them directly to Lakebase Postgres.
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Read API key from Databricks secrets
# MAGIC 2. Call Massive API `/v2/reference/news` for watchlisted tickers
# MAGIC 3. Parse JSON response into Python rows
# MAGIC 4. Upgrade SDK for Lakebase `w.postgres` API
# MAGIC 5. Connect to `ai-stock-assistant` Lakebase project, create table, insert news data directly

# COMMAND ----------

# DBTITLE 1,Step 1: Read API key from secret
# Step 1: Read API key from Databricks secret
import base64
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
_secret = w.secrets.get_secret(scope="massive", key="api-key")
API_KEY = base64.b64decode(_secret.value).decode("utf-8")

print(f"API key loaded: {API_KEY[:4]}...{API_KEY[-4:]} (length={len(API_KEY)})")

# COMMAND ----------

# DBTITLE 1,Step 2: Call Massive news API
# Step 2: Call Massive API /v2/reference/news for AAPL
import requests
import json

ticker = "AAPL"
url = "https://api.massive.com/v2/reference/news"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
params = {
    "ticker": ticker,
    "limit": 50,
    "order": "desc",
    "sort": "published_utc",
}

response = requests.get(url, headers=headers, params=params, timeout=30)
print(f"Status: {response.status_code}")

data = response.json()
results = data.get("results", [])
print(f"Articles returned: {len(results)}")

# Show first article structure
if results:
    print(f"\nFirst article:")
    print(f"  id:            {results[0]['id']}")
    print(f"  title:         {results[0]['title']}")
    print(f"  publisher:     {results[0]['publisher']['name']}")
    print(f"  published_utc: {results[0]['published_utc']}")
    print(f"  tickers:       {results[0]['tickers']}")
    print(f"  description:   {results[0]['description'][:100]}...")

# COMMAND ----------

# DBTITLE 1,Step 3: JSON to Spark DataFrame
# Step 3: Parse JSON into Spark DataFrame
# Map API fields to news_articles schema:
#   article_id   = id
#   ticker       = ticker (from query param, since one article can mention multiple tickers)
#   headline     = title
#   article_text = description
#   published_at = published_utc
#   source       = publisher.name

from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType
)

rows = []
for article in results:
    rows.append((
        article.get("id"),
        ticker,
        article.get("title"),
        article.get("description"),
        article.get("published_utc"),
        article.get("publisher", {}).get("name"),
    ))

schema = StructType([
    StructField("article_id", StringType(), False),
    StructField("ticker", StringType(), False),
    StructField("headline", StringType(), True),
    StructField("article_text", StringType(), True),
    StructField("published_at", StringType(), True),
    StructField("source", StringType(), True),
])

news_df = spark.createDataFrame(rows, schema)
print(f"Spark DataFrame: {news_df.count()} rows")
news_df.printSchema()
news_df.show(truncate=30)

# COMMAND ----------

# DBTITLE 1,Step 10: Verify data and find correct project
# Step 4: Upgrade SDK to enable w.postgres API (required for Lakebase)
# If SDK was already upgraded in a prior run, no restart happens and variables are preserved.
import importlib.metadata as md
import subprocess, sys

before = md.version("databricks-sdk")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "databricks-sdk>=0.118.0"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
after = md.version("databricks-sdk")
print(f"databricks-sdk: {before} -> {after}  (changed={before != after})")

if before != after:
    print("Version changed — restarting Python to load new SDK...")
    dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Step 12: Connect to ai-stock-assistant project
# Step 5: Connect to ai-stock-assistant Lakebase, create table, insert news directly
# No Lakehouse intermediary — data goes straight from API → Lakebase Postgres
import psycopg2
from psycopg2.extras import execute_values
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# --- Connect to ai-stock-assistant Lakebase project ---
target_project = "ai-stock-assistant"
endpoint_name = None
endpoint_host = None

for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
    for endpoint in w.postgres.list_endpoints(parent=branch.name):
        endpoint_name = endpoint.name
        endpoint_host = endpoint.status.hosts.host if endpoint.status and endpoint.status.hosts else None

print(f"Endpoint: {endpoint_name}")
print(f"Host: {endpoint_host}")

token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
username = w.current_user.me().user_name

conn = psycopg2.connect(
    host=endpoint_host, port=5432, dbname="databricks_postgres",
    user=username, password=token, sslmode="require", connect_timeout=10,
)
cursor = conn.cursor()
cursor.execute("SELECT version();")
print(f"Connected: {cursor.fetchone()[0][:50]}...")

# --- Create bronze_news_articles table ---
cursor.execute("""
    CREATE TABLE IF NOT EXISTS bronze_news_articles (
        article_id   TEXT NOT NULL,
        ticker       TEXT NOT NULL,
        headline     TEXT,
        article_text TEXT,
        published_at TEXT,
        source       TEXT,
        ingested_at  TIMESTAMPTZ DEFAULT now(),
        source_api   TEXT
    )
""")
conn.commit()
print("\nTable ready: bronze_news_articles")

# --- Insert news data directly from API response ---
from datetime import datetime, timezone

rows = []
ingested_at = datetime.now(timezone.utc)
for article in results:
    rows.append((
        article.get("id"),
        ticker,
        article.get("title"),
        article.get("description"),
        article.get("published_utc"),
        article.get("publisher", {}).get("name"),
        ingested_at,
        "massive_stocks_api",
    ))

insert_sql = """
    INSERT INTO bronze_news_articles
        (article_id, ticker, headline, article_text, published_at, source, ingested_at, source_api)
    VALUES %s
    ON CONFLICT (article_id, ticker) DO NOTHING
"""
execute_values(cursor, insert_sql, rows, page_size=100)
conn.commit()
print(f"Inserted {len(rows)} articles into Lakebase bronze_news_articles")

# Verify
cursor.execute("SELECT COUNT(*) FROM bronze_news_articles")
print(f"Total rows in table: {cursor.fetchone()[0]}")

cursor.close()
conn.close()
print("\nDone! News data written directly to ai-stock-assistant Lakebase.")

# COMMAND ----------

# DBTITLE 1,Step 6: Read article and construct search_text
# Step 6: Read one article from Lakebase and construct search_text
# search_text = headline + " " + article_text  (for later embedding)
import psycopg2
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Reconnect to ai-stock-assistant Lakebase
target_project = "ai-stock-assistant"
for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
    for endpoint in w.postgres.list_endpoints(parent=branch.name):
        endpoint_name = endpoint.name
        endpoint_host = endpoint.status.hosts.host

token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
username = w.current_user.me().user_name

conn = psycopg2.connect(
    host=endpoint_host, port=5432, dbname="databricks_postgres",
    user=username, password=token, sslmode="require", connect_timeout=10,
)
cursor = conn.cursor()

# Fetch one article
cursor.execute("""
    SELECT article_id, ticker, headline, article_text, source, published_at
    FROM bronze_news_articles
    WHERE ticker = 'AAPL'
    ORDER BY published_at DESC
    LIMIT 1
""")
row = cursor.fetchone()
cursor.close()
conn.close()

if row:
    article_id, ticker, headline, article_text, source, published_at = row
    search_text = f"{headline} {article_text}"

    print(f"article_id:   {article_id}")
    print(f"ticker:       {ticker}")
    print(f"source:       {source}")
    print(f"published_at: {published_at}")
    print(f"headline:     {headline}")
    print(f"article_text: {article_text[:200]}...")
    print(f"\nsearch_text length: {len(search_text)} chars")
    print(f"\nsearch_text preview:")
    print(search_text[:500])
else:
    print("No articles found")

# COMMAND ----------

# DBTITLE 1,Step 7: Embed search_text
# Step 7: Generate embedding for search_text
# all-MiniLM-L6-v2: 384-dim, fast, lightweight model for semantic search
%pip install sentence-transformers --quiet

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding = model.encode(search_text)

print(type(embedding))
print(embedding.shape)
print(embedding[:10])

# COMMAND ----------

# DBTITLE 1,Step 9: Semantic search experiment
# Step 9: Semantic search experiment
# Encode a differently-worded query → pgvector cosine distance → find most relevant article
import psycopg2
import numpy as np
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# A query with completely different wording from the article
# Article headline: "SoundHound's Powerful No-Screen AI Opportunity Faces a High-Stakes Fight"
# Article text: voice AI, cars, restaurants, TVs, Apple competition
query = "Which company is building voice-controlled interfaces for automobiles and TVs?"

print(f"Query: {query}")

# Encode the query with the same model
query_embedding = model.encode(query)
query_str = np.array2string(query_embedding, separator=',', max_line_width=10000)

# Reconnect to Lakebase
target_project = "ai-stock-assistant"
for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
    for endpoint in w.postgres.list_endpoints(parent=branch.name):
        endpoint_name = endpoint.name
        endpoint_host = endpoint.status.hosts.host

token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
username = w.current_user.me().user_name

conn = psycopg2.connect(
    host=endpoint_host, port=5432, dbname="databricks_postgres",
    user=username, password=token, sslmode="require", connect_timeout=10,
)
cursor = conn.cursor()

# Semantic search: compare query embedding to stored article embeddings
cursor.execute("""
    SELECT
        e.article_id,
        b.headline,
        b.source,
        b.published_at,
        1 - (e.embedding <=> %s::vector) AS cosine_similarity
    FROM news_embeddings e
    JOIN bronze_news_articles b ON e.article_id = b.article_id
    ORDER BY e.embedding <=> %s::vector ASC
    LIMIT 5
""", (query_str, query_str))

results = cursor.fetchall()
print(f"\nTop {len(results)} semantic matches:")
print("-" * 80)
for i, (aid, headline, source, published_at, sim) in enumerate(results, 1):
    print(f"  #{i}  similarity: {sim:.4f}")
    print(f"      headline:  {headline}")
    print(f"      source:    {source}")
    print(f"      published: {published_at}")
    print(f"      article_id: {aid[:40]}...")

cursor.close()
conn.close()

# COMMAND ----------

# DBTITLE 1,Step 8: Insert embedding to Lakebase
# Step 8: Insert single embedding into Lakebase news_embeddings
# bronze_news_articles.article_id → news_embeddings.article_id + embedding(384)
# ON CONFLICT: re-running notebook updates existing record, no duplicates
import psycopg2
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Reconnect to ai-stock-assistant Lakebase
target_project = "ai-stock-assistant"
for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
    for endpoint in w.postgres.list_endpoints(parent=branch.name):
        endpoint_name = endpoint.name
        endpoint_host = endpoint.status.hosts.host

token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
username = w.current_user.me().user_name

conn = psycopg2.connect(
    host=endpoint_host, port=5432, dbname="databricks_postgres",
    user=username, password=token, sslmode="require", connect_timeout=10,
)

embedding_list = embedding.tolist()
with conn.cursor() as cur:
    cur.execute(
        """
        INSERT INTO news_embeddings (
            article_id,
            embedding
        )
        VALUES (%s, %s::vector)
        ON CONFLICT (article_id)
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            embedded_at = NOW()
        """,
        (article_id, embedding_list)
    )
conn.commit()
print(f"Inserted embedding for article_id: {article_id[:40]}...")

# Verify
cursor = conn.cursor()
cursor.execute("""
    SELECT article_id, vector_dims(embedding) AS dimensions, embedded_at
    FROM news_embeddings
""")
for row in cursor.fetchall():
    print(f"  article_id: {row[0][:40]}... | dimensions: {row[1]} | embedded_at: {row[2]}")

cursor.close()
conn.close()

# COMMAND ----------

# DBTITLE 1,Step 10: Batch embed 50 articles
# Step 10: Batch embed all 50 articles and write to news_embeddings
# 1. Read DISTINCT articles from bronze_news_articles
# 2. Build search_text = headline + " " + article_text for each
# 3. Batch encode all at once with all-MiniLM-L6-v2
# 4. Upsert all embeddings into news_embeddings (ON CONFLICT handles re-runs)

import psycopg2
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Reconnect to ai-stock-assistant Lakebase
target_project = "ai-stock-assistant"
for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
    for endpoint in w.postgres.list_endpoints(parent=branch.name):
        endpoint_name = endpoint.name
        endpoint_host = endpoint.status.hosts.host

token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
username = w.current_user.me().user_name

conn = psycopg2.connect(
    host=endpoint_host, port=5432, dbname="databricks_postgres",
    user=username, password=token, sslmode="require", connect_timeout=10,
)
cursor = conn.cursor()

# 1. Read distinct articles
cursor.execute("""
    SELECT DISTINCT article_id, headline, article_text
    FROM bronze_news_articles
    WHERE headline IS NOT NULL AND article_text IS NOT NULL
""")
articles = cursor.fetchall()
print(f"Articles to embed: {len(articles)}")

# 2. Build search_text for each article
article_ids = [a[0] for a in articles]
search_texts = [f"{a[1]} {a[2]}" for a in articles]

# 3. Batch encode all search_texts at once
print("Encoding embeddings...")
embeddings = model.encode(search_texts, show_progress_bar=True)
print(f"Encoded {len(embeddings)} embeddings, each {embeddings[0].shape[0]}-dim")

# 4. Upsert all into news_embeddings
print("Writing to Lakebase...")
for aid, emb in zip(article_ids, embeddings):
    emb_list = emb.tolist()
    cursor.execute(
        """
        INSERT INTO news_embeddings (article_id, embedding)
        VALUES (%s, %s::vector)
        ON CONFLICT (article_id)
        DO UPDATE SET embedding = EXCLUDED.embedding, embedded_at = NOW()
        """,
        (aid, emb_list)
    )
conn.commit()
print(f"Upserted {len(article_ids)} embeddings into news_embeddings")

# Verify
cursor.execute("SELECT COUNT(*), COUNT(DISTINCT article_id) FROM news_embeddings")
total, distinct = cursor.fetchone()
print(f"\nnews_embeddings: {total} rows ({distinct} distinct article_ids)")

cursor.execute("""
    SELECT vector_dims(embedding) AS dim, COUNT(*) AS cnt
    FROM news_embeddings
    GROUP BY vector_dims(embedding)
""")
for dim, cnt in cursor.fetchall():
    print(f"  dim={dim}: {cnt} rows")

cursor.close()
conn.close()

# COMMAND ----------

# DBTITLE 1,Step 11: Semantic search across 50 articles
# Step 11: Semantic search across all 50 articles
# Query with different wording → embed → pgvector cosine distance → top 5 matches
import psycopg2
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

query = "What are the biggest AI-related competitive risks facing technology companies?"
print(f"Query: {query}\n")

# Encode query with same model
query_embedding = model.encode(query)
query_list = query_embedding.tolist()

# Reconnect to Lakebase
target_project = "ai-stock-assistant"
for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
    for endpoint in w.postgres.list_endpoints(parent=branch.name):
        endpoint_name = endpoint.name
        endpoint_host = endpoint.status.hosts.host

token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
username = w.current_user.me().user_name

conn = psycopg2.connect(
    host=endpoint_host, port=5432, dbname="databricks_postgres",
    user=username, password=token, sslmode="require", connect_timeout=10,
)
cursor = conn.cursor()

cursor.execute("""
    SELECT
        b.article_id,
        b.headline,
        b.source,
        b.published_at,
        1 - (e.embedding <=> %s::vector) AS cosine_similarity
    FROM news_embeddings e
    JOIN bronze_news_articles b ON e.article_id = b.article_id
    ORDER BY e.embedding <=> %s::vector ASC
    LIMIT 5
""", (query_list, query_list))

results = cursor.fetchall()
print(f"Top {len(results)} semantic matches:")
print("-" * 80)
for i, (aid, headline, source, published_at, sim) in enumerate(results, 1):
    print(f"  #{i}  similarity: {sim:.4f}")
    print(f"      headline:  {headline}")
    print(f"      source:    {source}")
    print(f"      published: {published_at}")
    print()

cursor.close()
conn.close()

# COMMAND ----------

# DBTITLE 1,Step 12: search_news() Agent Read Tool
# Step 12: search_news() — Agent Read Tool
# Wraps the semantic search pipeline into a simple function.
# Agent only needs to know: "call search_news(query) when user asks about news"

def search_news(query, ticker=None, top_k=5):
    """
    Semantic search over news articles in Lakebase.
    
    Args:
        query:  natural language question (e.g. "AI competitive risks")
        ticker: optional filter, e.g. "AAPL" (default: search all tickers)
        top_k:  number of results to return (default: 5)
    
    Returns:
        list of dicts: [{headline, article_text, source, published_at, similarity}, ...]
    """
    # 1. Encode query → 384-dim vector
    query_embedding = model.encode(query)
    query_list = query_embedding.tolist()
    
    # 2. Connect to Lakebase
    w = WorkspaceClient()
    target_project = "ai-stock-assistant"
    for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
        for endpoint in w.postgres.list_endpoints(parent=branch.name):
            endpoint_name = endpoint.name
            endpoint_host = endpoint.status.hosts.host
    
    token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
    username = w.current_user.me().user_name
    
    conn = psycopg2.connect(
        host=endpoint_host, port=5432, dbname="databricks_postgres",
        user=username, password=token, sslmode="require", connect_timeout=10,
    )
    cursor = conn.cursor()
    
    # 3. pgvector cosine distance search
    if ticker:
        cursor.execute("""
            SELECT b.headline, b.article_text, b.source, b.published_at,
                   1 - (e.embedding <=> %s::vector) AS similarity
            FROM news_embeddings e
            JOIN bronze_news_articles b ON e.article_id = b.article_id
            WHERE b.ticker = %s
            ORDER BY e.embedding <=> %s::vector ASC
            LIMIT %s
        """, (query_list, ticker, query_list, top_k))
    else:
        cursor.execute("""
            SELECT b.headline, b.article_text, b.source, b.published_at,
                   1 - (e.embedding <=> %s::vector) AS similarity
            FROM news_embeddings e
            JOIN bronze_news_articles b ON e.article_id = b.article_id
            ORDER BY e.embedding <=> %s::vector ASC
            LIMIT %s
        """, (query_list, query_list, top_k))
    
    # 4. Format results
    results = []
    for headline, article_text, source, published_at, similarity in cursor.fetchall():
        results.append({
            "headline": headline,
            "article_text": article_text,
            "source": source,
            "published_at": published_at,
            "similarity": round(float(similarity), 4),
        })
    
    cursor.close()
    conn.close()
    return results


# --- Test it ---
print("=== Test 1: AI competitive risks (all tickers) ===")
results = search_news("What are the biggest AI-related competitive risks facing technology companies?")
for i, r in enumerate(results, 1):
    print(f"  #{i}  sim={r['similarity']}  {r['headline']}")

print(f"\n=== Test 2: Apple product launches (ticker=AAPL) ===")
results = search_news("new product announcements and launches", ticker="AAPL", top_k=3)
for i, r in enumerate(results, 1):
    print(f"  #{i}  sim={r['similarity']}  {r['headline']}")

# COMMAND ----------

# DBTITLE 1,Step 13: add_to_watchlist() Agent Write Tool
# Step 13: add_to_watchlist() — Agent Write Tool
# Lets the Agent modify data in Lakebase, not just read.
# User says "Add AAPL to my watchlist" → Agent calls add_to_watchlist("AAPL")

def add_to_watchlist(ticker, user_id="default"):
    """
    Add a ticker to the user's watchlist in Lakebase.
    
    Args:
        ticker:  stock symbol, e.g. "AAPL"
        user_id: user identifier (default: "default")
    
    Returns:
        dict: {ticker, user_id, status} — "added" or "already_exists"
    """
    w = WorkspaceClient()
    target_project = "ai-stock-assistant"
    for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
        for endpoint in w.postgres.list_endpoints(parent=branch.name):
            endpoint_name = endpoint.name
            endpoint_host = endpoint.status.hosts.host
    
    token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
    username = w.current_user.me().user_name
    
    conn = psycopg2.connect(
        host=endpoint_host, port=5432, dbname="databricks_postgres",
        user=username, password=token, sslmode="require", connect_timeout=10,
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO watchlist_items (user_id, ticker)
        VALUES (%s, %s)
        ON CONFLICT (user_id, ticker) DO NOTHING
        RETURNING ticker
    """, (user_id, ticker.upper()))
    
    inserted = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    
    if inserted:
        return {"ticker": ticker.upper(), "user_id": user_id, "status": "added"}
    else:
        return {"ticker": ticker.upper(), "user_id": user_id, "status": "already_exists"}


def get_watchlist(user_id="default"):
    """
    Retrieve the user's watchlist from Lakebase.
    
    Returns:
        list of dicts: [{ticker, created_at}, ...]
    """
    w = WorkspaceClient()
    target_project = "ai-stock-assistant"
    for branch in w.postgres.list_branches(parent=f"projects/{target_project}"):
        for endpoint in w.postgres.list_endpoints(parent=branch.name):
            endpoint_name = endpoint.name
            endpoint_host = endpoint.status.hosts.host
    
    token = w.postgres.generate_database_credential(endpoint=endpoint_name).token
    username = w.current_user.me().user_name
    
    conn = psycopg2.connect(
        host=endpoint_host, port=5432, dbname="databricks_postgres",
        user=username, password=token, sslmode="require", connect_timeout=10,
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticker, created_at
        FROM watchlist_items
        WHERE user_id = %s
        ORDER BY created_at ASC
    """, (user_id,))
    
    items = []
    for ticker, created_at in cursor.fetchall():
        items.append({"ticker": ticker, "created_at": created_at})
    
    cursor.close()
    conn.close()
    return items


# --- Test it ---
print("=== Test 1: Add AAPL ===")
result = add_to_watchlist("AAPL")
print(f"  {result}")

print("\n=== Test 2: Add AAPL again (should say already_exists) ===")
result = add_to_watchlist("aapl")  # lowercase → should normalize to AAPL
print(f"  {result}")

print("\n=== Test 3: Add NVDA ===")
result = add_to_watchlist("NVDA")
print(f"  {result}")

print("\n=== Test 4: Get watchlist ===")
watchlist = get_watchlist()
for item in watchlist:
    print(f"  {item['ticker']}  (added: {item['created_at']})")