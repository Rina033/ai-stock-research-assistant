# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # Massive Stocks API → JSON → Spark DataFrame
# MAGIC
# MAGIC This notebook demonstrates the full pipeline:
# MAGIC 1. Read the Massive API key from Databricks secrets (not hardcoded)
# MAGIC 2. Call the Massive Stocks API to fetch price data
# MAGIC 3. Parse the JSON response
# MAGIC 4. Convert to a Spark DataFrame for downstream processing

# COMMAND ----------

# DBTITLE 1,Step 1: Read API key from Databricks secret
import base64
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Read API key from Databricks secret scope
# The secret is stored base64-encoded in scope "massive", key "api-key"
_secret = w.secrets.get_secret(scope="massive", key="api-key")
API_KEY = base64.b64decode(_secret.value).decode("utf-8")

print(f"API key loaded: {API_KEY[:4]}...{API_KEY[-4:]} (length={len(API_KEY)})")

# Ticker to query
ticker = "AAPL"

# COMMAND ----------

# DBTITLE 1,Step 2: Call Massive API and get JSON
import requests

url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/prev"

# Try Bearer auth first (matching massive_client.py pattern)
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

response = requests.get(url, headers=headers, timeout=30)

print(f"Status code: {response.status_code}")

if response.status_code != 200:
    # Fallback: try apiKey as query parameter
    print("Bearer auth failed, trying apiKey query param...")
    response = requests.get(
        url,
        params={"adjusted": "true", "apiKey": API_KEY},
        timeout=30,
    )
    print(f"Status code (query param): {response.status_code}")

data = response.json()
print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
print(data)

# COMMAND ----------

# DBTITLE 1,Step 3: Parse JSON response
import json

# Parse the JSON response into a Python dict
# Massive /v2/aggs/ticker/{symbol}/prev returns:
# {"ticker": "AAPL", "results": [{"c": close, "h": high, "l": low, "o": open, "v": volume, "t": timestamp}]}

print(f"Ticker: {data.get('ticker', 'N/A')}")
print(f"Results count: {len(data.get('results', []))}")

# Flatten results into a list of dicts
results = data.get("results", [])
if results:
    for r in results:
        print(f"  Open: {r.get('o')}, Close: {r.get('c')}, High: {r.get('h')}, Low: {r.get('l')}, Volume: {r.get('v')}")
else:
    print("No results in response")
    print(f"Full response: {json.dumps(data, indent=2)}")

# COMMAND ----------

# DBTITLE 1,Step 4: Convert JSON to Spark DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType
import datetime

spark = SparkSession.builder.getOrCreate()

# Define schema matching the API response
schema = StructType([
    StructField("ticker", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("volume", LongType(), True),
    StructField("timestamp", StringType(), True),  # ISO string from API
])

# Flatten results into rows
results = data.get("results", [])
ticker_name = data.get("ticker", ticker)

rows = []
for r in results:
    ts_ms = r.get("t", 0)
    if ts_ms:
        ts_str = datetime.datetime.fromtimestamp(ts_ms / 1000).isoformat()
    else:
        ts_str = None
    rows.append((
        ticker_name,
        float(r.get("o", 0)),
        float(r.get("c", 0)),
        float(r.get("h", 0)),
        float(r.get("l", 0)),
        int(r.get("v", 0)),
        ts_str,
    ))

# Create Spark DataFrame
df = spark.createDataFrame(rows, schema)

print(f"Spark DataFrame created with {df.count()} rows")
df.printSchema()
df.show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 5: Spark DataFrame operations
# Demonstrate Spark DataFrame operations on the API data
from pyspark.sql.functions import col, expr

if df.count() > 0:
    # Calculate daily change
    df_with_change = df.withColumn(
        "change",
        col("close") - col("open")
    ).withColumn(
        "change_percent",
        expr("CASE WHEN open > 0 THEN ((close - open) / open) * 100 ELSE 0 END")
    )
    
    print("Price data with daily change:")
    df_with_change.select(
        "ticker", "open", "close", "high", "low", "volume",
        "change", "change_percent", "timestamp"
    ).show(truncate=False)
    
    # Show the DataFrame is ready for further processing
    print("\nDataFrame is ready for downstream processing (write to Lakebase, join with other data, etc.)")
else:
    print("No data to process. Check API response above.")

# COMMAND ----------

# DBTITLE 1,Step 6: Bronze layer with metadata
from pyspark.sql import functions as F

# Bronze layer: add ingestion metadata to the processed DataFrame
bronze_df = (
    df_with_change
    .withColumn("ingested_at", F.current_timestamp())
    .withColumn("source", F.lit("massive_stocks_api"))
)

display(bronze_df)

# COMMAND ----------

# DBTITLE 1,Step 7: Save to Delta table
# Confirm data looks correct in the display above, then save to Delta table
bronze_df.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("bronze_stock_prices")

print("Saved to bronze_stock_prices table")

# COMMAND ----------

# DBTITLE 1,Step 9: Silver layer dedup
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Silver layer: deduplicate by ticker + timestamp, keep latest ingested_at
window_spec = (
    Window
    .partitionBy("ticker", "timestamp")
    .orderBy(F.col("ingested_at").desc())
)

silver_df = (
    spark.table("bronze_stock_prices")
    .withColumn(
        "row_number",
        F.row_number().over(window_spec)
    )
    .filter(F.col("row_number") == 1)
    .drop("row_number")
)

display(silver_df)

# COMMAND ----------

# DBTITLE 1,Step 10: Verify silver dedup
# Verify silver_df has no duplicates
silver_df.groupBy("ticker", "timestamp") \
    .count() \
    .show()

# COMMAND ----------

# DBTITLE 1,Step 8: Check for duplicates
spark.sql("""
SELECT
    ticker,
    timestamp,
    COUNT(*) AS row_count
FROM bronze_stock_prices
GROUP BY ticker, timestamp
ORDER BY row_count DESC
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 11: Persist silver_df to Delta
# Persist silver_df to Delta table (overwrite = idempotent)
silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_stock_prices")

print("Saved to silver_stock_prices table")

# COMMAND ----------

# DBTITLE 1,Step 12: Verify silver table
# Verify silver_stock_prices: exactly 1 row per ticker + timestamp
spark.sql("""
SELECT
    ticker,
    timestamp,
    COUNT(*) AS row_count
FROM silver_stock_prices
GROUP BY ticker, timestamp
ORDER BY row_count DESC
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 13: Data-quality checks
# Data-quality checks on silver_stock_prices
from pyspark.sql import functions as F
from pyspark.sql.window import Window

silver = spark.table("silver_stock_prices")

# --- Check 1: Uniqueness — no duplicate (ticker, timestamp) pairs ---
dup_count = (
    silver
    .groupBy("ticker", "timestamp")
    .agg(F.count("*").alias("cnt"))
    .filter(F.col("cnt") > 1)
    .count()
)
print(f"[Check 1] Uniqueness (ticker, timestamp): {'PASS' if dup_count == 0 else f'FAIL — {dup_count} duplicate groups'}")

# --- Check 2: Price non-null — open, close, high, low must not be null ---
null_price_count = (
    silver
    .filter(
        F.col("open").isNull()
        | F.col("close").isNull()
        | F.col("high").isNull()
        | F.col("low").isNull()
    )
    .count()
)
print(f"[Check 2] Price non-null: {'PASS' if null_price_count == 0 else f'FAIL — {null_price_count} rows with null prices'}")

# --- Check 3: Price > 0 — open, close, high, low must be positive ---
nonpositive_count = (
    silver
    .filter(
        (F.col("open") <= 0)
        | (F.col("close") <= 0)
        | (F.col("high") <= 0)
        | (F.col("low") <= 0)
    )
    .count()
)
print(f"[Check 3] Price > 0: {'PASS' if nonpositive_count == 0 else f'FAIL — {nonpositive_count} rows with non-positive prices'}")

# --- Check 4: high >= low ---
high_low_violations = (
    silver
    .filter(F.col("high") < F.col("low"))
    .count()
)
print(f"[Check 4] high >= low: {'PASS' if high_low_violations == 0 else f'FAIL — {high_low_violations} rows where high < low'}")

# --- Summary ---
all_pass = (dup_count == 0 and null_price_count == 0 and nonpositive_count == 0 and high_low_violations == 0)
print(f"\n{'='*40}")
print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print(f"{'='*40}")

# Show the data for reference
print("\nSilver table contents:")
silver.show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 14: Quality checks dict
from pyspark.sql import functions as F

silver = spark.table("silver_stock_prices")

quality_checks = {
    # 1. 主业务字段不能为空
    "null_ticker":
        silver.filter(F.col("ticker").isNull()).count(),

    "null_timestamp":
        silver.filter(F.col("timestamp").isNull()).count(),

    # 2. 价格必须合理
    "invalid_open":
        silver.filter(F.col("open") <= 0).count(),

    "invalid_close":
        silver.filter(F.col("close") <= 0).count(),

    # 3. high 不应该低于 low
    "high_below_low":
        silver.filter(F.col("high") < F.col("low")).count(),

    # 4. volume 不应该为负数
    "negative_volume":
        silver.filter(F.col("volume") < 0).count(),

    # 5. ticker + timestamp 必须唯一
    "duplicate_grain":
        (
            silver
            .groupBy("ticker", "timestamp")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
}

quality_checks

# COMMAND ----------

# DBTITLE 1,Step 15: Fail-fast on quality checks
failed_checks = {
    check: count
    for check, count in quality_checks.items()
    if count > 0
}

if failed_checks:
    raise ValueError(
        f"Data quality checks failed: {failed_checks}"
    )
else:
    print("✅ All Silver data quality checks passed.")

# COMMAND ----------

# DBTITLE 1,Step 16: Build dim_company (SCD Type 2)
# Gold Layer: dim_company (SCD Type 2)
# Since Massive API /v2/aggs/ticker doesn't return company metadata,
# we build a lightweight dim_company from the tickers present in silver.
# In production, this would come from a company reference API or file.

from pyspark.sql import functions as F

silver = spark.table("silver_stock_prices")

# Extract distinct tickers — one row per company
dim_company = (
    silver
    .select(
        F.col("ticker").alias("ticker"),
        F.col("ticker").alias("company_name"),  # placeholder
        F.lit(None).cast("string").alias("sector"),
        F.lit(None).cast("string").alias("industry"),
        F.lit("1970-01-01T00:00:00").cast("timestamp").alias("effective_from"),
        F.lit(None).cast("timestamp").alias("effective_to"),
        F.lit(True).alias("is_current"),
    )
    .dropDuplicates(["ticker"])
)

# Generate a surrogate company_key (hash of ticker for stability)
dim_company = dim_company.withColumn(
    "company_key",
    F.sha2(F.col("ticker"), 256).substr(1, 16)
).select(
    "company_key",
    "ticker",
    "company_name",
    "sector",
    "industry",
    "effective_from",
    "effective_to",
    "is_current",
)

display(dim_company)

# COMMAND ----------

# DBTITLE 1,Step 17: Build fact_stock_price
# Gold Layer: fact_stock_price
# Join silver_stock_prices to dim_company to resolve company_key

fact_stock_price = (
    spark.table("silver_stock_prices")
    .join(
        dim_company,
        on="ticker",
        how="inner"
    )
    .select(
        "company_key",
        F.col("timestamp").alias("timestamp"),
        F.col("open"),
        F.col("high"),
        F.col("low"),
        F.col("close"),
        F.col("volume"),
        F.col("change"),
        F.col("change_percent"),
    )
)

display(fact_stock_price)

# COMMAND ----------

# DBTITLE 1,Step 18: Persist Gold tables
# Persist Gold tables to Delta
dim_company.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("dim_company")

fact_stock_price.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("fact_stock_price")

print("Saved dim_company and fact_stock_price tables")

# COMMAND ----------

# DBTITLE 1,Step 19: Verify Gold tables
# Verify Gold tables
print("=== dim_company ===")
spark.table("dim_company").show(truncate=False)

print("\n=== fact_stock_price ===")
spark.table("fact_stock_price").show(truncate=False)

print("\n=== Star schema join (dim_company ⋈ fact_stock_price) ===")
spark.sql("""
SELECT
    dc.ticker,
    dc.company_name,
    dc.sector,
    fp.timestamp,
    fp.open,
    fp.high,
    fp.low,
    fp.close,
    fp.volume,
    fp.change,
    fp.change_percent
FROM dim_company dc
JOIN fact_stock_price fp
    ON dc.company_key = fp.company_key
WHERE dc.is_current = true
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 20: Fetch company details from API
# Fetch company details from Massive API (GET /v3/reference/tickers/{ticker})
# This endpoint returns company_name, industry (via SIC), exchange, market cap, etc.

import requests

ticker = "AAPL"
details_url = f"https://api.massive.com/v3/reference/tickers/{ticker}"

resp = requests.get(details_url, headers=headers, timeout=30)
print(f"Status: {resp.status_code}")

company_data = resp.json().get("results", {})

print(f"Ticker:       {company_data.get('ticker')}")
print(f"Company name: {company_data.get('name')}")
print(f"Industry:     {company_data.get('sic_description')}")
print(f"SIC code:     {company_data.get('sic_code')}")
print(f"Exchange:     {company_data.get('primary_exchange')}")
print(f"Market cap:   {company_data.get('market_cap')}")
print(f"Active:       {company_data.get('active')}")

# COMMAND ----------

# DBTITLE 1,Step 21: Build dim_company v1 with SCD2
# Build dim_company v1 with SCD Type 2 fields
# Using company details fetched from Massive API /v3/reference/tickers/AAPL

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

company_df = spark.createDataFrame(
    [{
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "industry": "ELECTRONIC COMPUTERS",
        "sic_code": "3571",
        "exchange": "XNAS",
        "market_cap": 4.53e12,
        "active": True,
        "sector": None,
    }],
    schema=StructType([
        StructField("ticker", StringType(), True),
        StructField("company_name", StringType(), True),
        StructField("industry", StringType(), True),
        StructField("sic_code", StringType(), True),
        StructField("exchange", StringType(), True),
        StructField("market_cap", DoubleType(), True),
        StructField("active", BooleanType(), True),
        StructField("sector", StringType(), True),
    ]),
)

dim_company_df = (
    company_df
    .withColumn(
        "company_key",
        F.sha2(F.concat_ws("||", F.col("ticker")), 256)
    )
    .withColumn("effective_from", F.current_timestamp())
    .withColumn("effective_to", F.lit(None).cast("timestamp"))
    .withColumn("is_current", F.lit(True))
)

display(dim_company_df)

# COMMAND ----------

# DBTITLE 1,Step 22: Persist dim_company v1
# Persist dim_company to Delta (overwrite previous placeholder version)
dim_company_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("dim_company")

print("Saved dim_company table")

# COMMAND ----------

# DBTITLE 1,Step 23: Verify dim_company v1
# Verify dim_company v1
spark.sql("SELECT * FROM dim_company").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 24: Simulate industry change for SCD2
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType
)

company_schema = StructType([
    StructField("ticker", StringType(), False),
    StructField("company_name", StringType(), True),
    StructField("industry", StringType(), True),
    StructField("sic_code", StringType(), True),
    StructField("exchange", StringType(), True),
    StructField("market_cap", DoubleType(), True),
    StructField("active", BooleanType(), True),
    StructField("sector", StringType(), True)
])

updated_company_df = spark.createDataFrame(
    [(
        "AAPL",
        "Apple Inc.",
        "TECHNOLOGY HARDWARE",   # 故意改变
        "3571",
        "XNAS",
        4.53e12,
        True,
        None
    )],
    schema=company_schema
)

display(updated_company_df)

# COMMAND ----------

# DBTITLE 1,Step 25: SCD2 close old record
# SCD2 Step 1: Close the old current record
# This does NOT insert the new version — it only ends the old one.

from datetime import datetime
from delta.tables import DeltaTable
from pyspark.sql import functions as F

change_time = datetime.now()

dim_company = DeltaTable.forName(spark, "dim_company")

dim_company.update(
    condition="""
        ticker = 'AAPL'
        AND is_current = true
        AND industry <> 'TECHNOLOGY HARDWARE'
    """,
    set={
        "effective_to": F.lit(change_time),
        "is_current": F.lit(False)
    }
)

print(f"Old record closed at {change_time}")

# COMMAND ----------

# DBTITLE 1,Step 26: Verify old record closed
# Verify: old record should be closed (effective_to set, is_current = false)
spark.sql("""
SELECT
    ticker,
    industry,
    effective_from,
    effective_to,
    is_current
FROM dim_company
ORDER BY effective_from
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 27: SCD2 insert new version
from pyspark.sql import functions as F

new_version_df = (
    updated_company_df
    .withColumn(
        "company_key",
        F.sha2(F.concat_ws("||", F.col("ticker")), 256)
    )
    .withColumn(
        "effective_from",
        F.lit(change_time).cast("timestamp")
    )
    .withColumn(
        "effective_to",
        F.lit(None).cast("timestamp")
    )
    .withColumn(
        "is_current",
        F.lit(True)
    )
)

new_version_df.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("dim_company")

print("New version appended to dim_company")

# COMMAND ----------

# DBTITLE 1,Step 28: Verify SCD2 complete
# Verify: dim_company now has 2 rows — old (closed) + new (current)
spark.sql("""
SELECT
    ticker,
    company_name,
    industry,
    effective_from,
    effective_to,
    is_current
FROM dim_company
ORDER BY effective_from
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 29: Fix company_key per version
# SCD2 improvement: unique company_key per version
# Problem: current company_key = sha2(ticker) is the SAME for all versions
# Fix: company_key = sha2(ticker || effective_from) so each version is unique

from pyspark.sql import functions as F

# Show the problem: both versions share the same company_key
print("=== Before: shared company_key ===")
spark.sql("""
SELECT ticker, industry, company_key, is_current
FROM dim_company
ORDER BY effective_from
""").show(truncate=False)

# Fix: regenerate company_key with effective_from included
dim_company_fixed = (
    spark.table("dim_company")
    .withColumn(
        "company_key",
        F.sha2(
            F.concat_ws(
                "||",
                F.col("ticker"),
                F.col("effective_from").cast("string")
            ),
            256
        )
    )
)

# Overwrite the table with fixed keys
dim_company_fixed.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("dim_company")

print("=== After: unique company_key per version ===")
spark.sql("""
SELECT ticker, industry, company_key, is_current
FROM dim_company
ORDER BY effective_from
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 30: Dimension backfill v1 effective_from
# Step 30: Dimension backfill — set v1 effective_from to earliest fact timestamp
# Problem: v1 effective_from = Aug 25 08:51 (system scrape time)
#          earliest fact timestamp = Aug 24 20:00 (business event time)
# Fix: backfill v1 effective_from to cover the earliest price data
# Method: DeltaTable.update (NOT overwrite) — surgically modify one row

from delta.tables import DeltaTable
from pyspark.sql import functions as F

# 1. Find earliest fact timestamp for AAPL
earliest_fact = (
    spark.table("silver_stock_prices")
    .filter(F.col("ticker") == "AAPL")
    .select(F.min(F.col("timestamp").cast("timestamp")).alias("earliest_ts"))
    .collect()[0]["earliest_ts"]
)
print(f"Earliest fact timestamp for AAPL: {earliest_fact}")

# 2. Backfill v1: update effective_from AND company_key (since key includes effective_from)
dim_tbl = DeltaTable.forName(spark, "dim_company")

dim_tbl.update(
    condition="""
        ticker = 'AAPL'
        AND is_current = false
        AND industry = 'ELECTRONIC COMPUTERS'
    """,
    set={
        "effective_from": F.lit(earliest_fact).cast("timestamp"),
        "company_key": F.sha2(
            F.concat_ws("||",
                F.col("ticker"),
                F.lit(str(earliest_fact))
            ),
            256
        )
    }
)

print(f"v1 effective_from backfilled to {earliest_fact}")

# 3. Verify
spark.sql("""
SELECT
    ticker,
    company_key,
    industry,
    effective_from,
    effective_to,
    is_current
FROM dim_company
ORDER BY effective_from
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 31: Get v2 effective_from for gap fix
from pyspark.sql import functions as F

new_start_time = (
    spark.table("dim_company")
    .filter(
        (F.col("ticker") == "AAPL") &
        (F.col("is_current") == True)
    )
    .select("effective_from")
    .first()["effective_from"]
)

print(new_start_time)

# COMMAND ----------

# DBTITLE 1,Step 32: Eliminate SCD2 gap
# Step 32: Eliminate gap — set v1 effective_to = v2 effective_from
# This makes the two versions contiguous: v1 ends exactly when v2 begins

from delta.tables import DeltaTable
from pyspark.sql import functions as F

dim_table = DeltaTable.forName(spark, "dim_company")

dim_table.update(
    condition="""
        ticker = 'AAPL'
        AND is_current = false
    """,
    set={
        "effective_to": F.lit(new_start_time)
    }
)

print(f"v1 effective_to updated to {new_start_time} (matches v2 effective_from)")

# Verify: no gap, contiguous timeline
spark.sql("""
SELECT
    ticker,
    company_key,
    industry,
    effective_from,
    effective_to,
    is_current
FROM dim_company
ORDER BY effective_from
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Step 33: Gold fact_stock_price (point-in-time join)
# Gold Layer: fact_stock_price via point-in-time join
# silver_stock_prices ⋈ dim_company on ticker + effective_from/to
# Each fact row matches exactly one dimension version

from pyspark.sql import functions as F

p = spark.table("silver_stock_prices").alias("p")
d = spark.table("dim_company").alias("d")

gold_fact_df = (
    p.join(
        d,
        (
            (F.col("p.ticker") == F.col("d.ticker"))
            &
            (F.col("p.timestamp") >= F.col("d.effective_from"))
            &
            (
                (F.col("p.timestamp") < F.col("d.effective_to"))
                |
                F.col("d.effective_to").isNull()
            )
        ),
        "left"
    )
    .select(
        F.col("d.company_key"),
        F.col("p.ticker"),
        F.col("p.timestamp"),
        F.col("p.open"),
        F.col("p.high"),
        F.col("p.low"),
        F.col("p.close"),
        F.col("p.volume"),
        F.col("p.change"),
        F.col("p.change_percent")
    )
)

display(gold_fact_df)