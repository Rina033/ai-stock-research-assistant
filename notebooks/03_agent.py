# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 03 — AI Stock Research Agent
# MAGIC
# MAGIC Agent with two tools that can search/retrieve and take real actions (writes):
# MAGIC
# MAGIC 1. **search_news(query, ticker, top_k)** — Read Tool: semantic search over 50 news articles via pgvector
# MAGIC 2. **add_to_watchlist(ticker)** — Write Tool: add ticker to user's watchlist in Lakebase
# MAGIC
# MAGIC **Flow:**
# MAGIC ```
# MAGIC User message → LLM decides tool call → execute tool → return result to user
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Step 1: Setup
# Step 1: Install dependencies (kernel will restart after this)
%pip install sentence-transformers openai "databricks-sdk>=0.118.0" --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Step 1b: Setup
# Step 1b: Setup — imports, embedding model, Foundation Model client (run after kernel restart)
import os
import json
import psycopg2
import mlflow.deployments
from databricks.sdk import WorkspaceClient
from sentence_transformers import SentenceTransformer

# Embedding model (same as notebook 02)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Databricks Foundation Model API — mlflow.deployments handles auth automatically
w = WorkspaceClient()
deploy_client = mlflow.deployments.get_deploy_client("databricks")

import importlib.metadata as md
print(f"Setup complete — SDK: {md.version('databricks-sdk')}")

# COMMAND ----------

# DBTITLE 1,Step 2: Tool functions
# Step 2: Tool functions (same as notebook 02, self-contained for the agent)

def search_news(query, ticker=None, top_k=5):
    """Semantic search over news articles in Lakebase."""
    query_embedding = embedder.encode(query)
    query_list = query_embedding.tolist()
    
    w = WorkspaceClient()
    for branch in w.postgres.list_branches(parent="projects/ai-stock-assistant"):
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
    
    if ticker:
        cursor.execute("""
            SELECT b.headline, b.article_text, b.source, b.published_at,
                   1 - (e.embedding <=> %s::vector) AS similarity
            FROM news_embeddings e
            JOIN bronze_news_articles b ON e.article_id = b.article_id
            WHERE b.ticker = %s
            ORDER BY e.embedding <=> %s::vector ASC LIMIT %s
        """, (query_list, ticker, query_list, top_k))
    else:
        cursor.execute("""
            SELECT b.headline, b.article_text, b.source, b.published_at,
                   1 - (e.embedding <=> %s::vector) AS similarity
            FROM news_embeddings e
            JOIN bronze_news_articles b ON e.article_id = b.article_id
            ORDER BY e.embedding <=> %s::vector ASC LIMIT %s
        """, (query_list, query_list, top_k))
    
    results = []
    for headline, article_text, source, published_at, similarity in cursor.fetchall():
        results.append({
            "headline": headline,
            "article_text": article_text,
            "source": source,
            "published_at": str(published_at),
            "similarity": round(float(similarity), 4),
        })
    cursor.close()
    conn.close()
    return results


def add_to_watchlist(ticker, user_id="default"):
    """Add a ticker to the user's watchlist in Lakebase."""
    w = WorkspaceClient()
    for branch in w.postgres.list_branches(parent="projects/ai-stock-assistant"):
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


print("Tools defined: search_news(), add_to_watchlist()")

# COMMAND ----------

# DBTITLE 1,Step 3: Tool schemas
# Step 3: Tool schemas — tell the LLM what each tool does and what params it needs
# This is what makes the LLM an "agent" — it can decide which tool to call

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search news articles by semantic meaning. Use when the user asks about company news, risks, opportunities, market trends, or any question that can be answered from news articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g. 'AI competitive risks for Apple'",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional stock ticker to filter, e.g. 'AAPL'. Omit to search all tickers.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_watchlist",
            "description": "Add a stock ticker to the user's watchlist. Use when the user asks to watch, track, follow, or add a stock to their watchlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'NVDA'",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]

# Map tool names to actual functions
tool_functions = {
    "search_news": search_news,
    "add_to_watchlist": add_to_watchlist,
}

print(f"Tool schemas defined: {list(tool_functions.keys())}")

# COMMAND ----------

# DBTITLE 1,Step 4: Agent loop
# Step 4: The Agent — core loop
# user message → LLM decides tool call → execute tool → feed result back → LLM generates answer

MODEL = "databricks-gpt-oss-120b"

SYSTEM_PROMPT = """You are an AI Stock Research Assistant. You have two tools:

1. search_news(query, ticker, top_k) — search 50 news articles by semantic meaning
2. add_to_watchlist(ticker) — add a stock to the user's watchlist

When the user asks about news, risks, or market trends, call search_news.
When the user asks to add/follow/track a stock, call add_to_watchlist.
Be concise. Summarize search results as a short bullet list."""


def run_agent(user_message):
    """
    Run one turn of the agent.
    Returns the assistant's final response text.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Turn 1: LLM decides which tool to call
    response = deploy_client.predict(
        endpoint=MODEL,
        inputs={
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.1,
            "max_tokens": 1000,
        },
    )
    msg = response["choices"][0]["message"]
    messages.append(msg)

    # If LLM called a tool, execute it
    if msg.get("tool_calls"):
        for tool_call in msg["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])
            print(f"  → Calling: {func_name}({func_args})")

            # Execute the actual function
            result = tool_functions[func_name](**func_args)
            print(f"  → Result: {result}")

            # Feed result back to LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })

        # Turn 2: LLM generates final answer based on tool results
        response = deploy_client.predict(
            endpoint=MODEL,
            inputs={
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000,
            },
        )
        return response["choices"][0]["message"]["content"]
    else:
        # No tool call — LLM answered directly
        return msg["content"]


print("Agent ready. Use: run_agent(\"your question\")")

# COMMAND ----------

# DBTITLE 1,Step 5: Test the agent
# Step 5: Test the agent

print("=" * 60)
print("Test 1: Search news")
print("=" * 60)
answer = run_agent("What are the biggest AI-related competitive risks facing technology companies?")
print(f"\nAgent: {answer}")

print("\n" + "=" * 60)
print("Test 2: Add to watchlist")
print("=" * 60)
answer = run_agent("Please add TSLA to my watchlist.")
print(f"\nAgent: {answer}")

print("\n" + "=" * 60)
print("Test 3: Combined — search + add")
print("=" * 60)
answer = run_agent("What's the latest news about Apple? Also add MSFT to my watchlist.")
print(f"\nAgent: {answer}")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Check available models
# Check available Foundation Model endpoints
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

print("=== Custom Serving Endpoints ===")
for ep in w.serving_endpoints.list():
    print(f"  {ep.name}")

print("\n=== Testing Foundation Models ===")
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

for model in [
    "databricks-dbrx-instruct",
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-llama-3-3-70b-instruct",
    "databricks-mistral-large-2-instruct",
    "gpt-4o",
    "gpt-4o-mini",
]:
    try:
        r = deploy_client.predict(endpoint=model, inputs={
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 5,
        })
        print(f"  {model} — OK")
    except Exception as e:
        print(f"  {model} — {str(e)[:80]}")