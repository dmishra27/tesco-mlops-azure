# Databricks notebook: 02_feature_engineering
# Transforms bronze transactions into customer-level RFM + behavioural features
# written to the silver layer as a Delta table.

from pyspark.sql import functions as F, Window

# Bug fix #6: replaced hardcoded <STORAGE_ACCOUNT> with dynamic secret lookup
STORAGE_ACCOUNT = dbutils.secrets.get(scope="adls-scope", key="STORAGE_ACCOUNT")

BRONZE_PATH = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net/transactions"
SILVER_PATH = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_features"

SNAPSHOT_DATE = spark.sql("SELECT current_date()").collect()[0][0]

# ── Load bronze transactions ──────────────────────────────────────────────────
transactions = spark.read.format("delta").load(BRONZE_PATH)

# ── RFM features ─────────────────────────────────────────────────────────────
rfm = (
    transactions
    .groupBy("customer_id")
    .agg(
        F.datediff(F.lit(SNAPSHOT_DATE), F.max("timestamp")).alias("recency_days"),
        F.count("transaction_id").alias("frequency"),
        F.sum("total_amount").alias("monetary"),
        F.countDistinct("ingest_date").alias("active_days"),
        F.avg("total_amount").alias("avg_basket_size"),
        F.stddev("total_amount").alias("basket_std"),
    )
)

# ── Category affinity (top-3 categories by spend) ────────────────────────────
category_spend = (
    transactions
    .groupBy("customer_id", "category")
    .agg(F.sum("total_amount").alias("cat_spend"))
)

w_cat = Window.partitionBy("customer_id").orderBy(F.desc("cat_spend"))
top_categories = (
    category_spend
    .withColumn("rank", F.rank().over(w_cat))
    .filter(F.col("rank") <= 3)
    .groupBy("customer_id")
    .agg(F.collect_list("category").alias("top_categories"))
)

# ── Channel mix ───────────────────────────────────────────────────────────────
channel_mix = (
    transactions
    .groupBy("customer_id")
    .agg(
        F.count(F.when(F.col("channel") == "online", True)).alias("online_txns"),
        F.count(F.when(F.col("channel") == "in-store", True)).alias("instore_txns"),
    )
    .withColumn(
        "online_ratio",
        F.col("online_txns") / (F.col("online_txns") + F.col("instore_txns")),
    )
)

# ── Join all features ─────────────────────────────────────────────────────────
features = (
    rfm
    .join(top_categories, "customer_id", "left")
    .join(channel_mix, "customer_id", "left")
    .withColumn("snapshot_date", F.lit(SNAPSHOT_DATE))
    .fillna({"basket_std": 0.0, "online_ratio": 0.0})
)

# ── Write to silver ───────────────────────────────────────────────────────────
(
    features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("snapshot_date")
    .save(SILVER_PATH)
)

print(f"Feature engineering complete. Rows written: {features.count()}")
