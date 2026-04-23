# Databricks notebook: 01_ingest
# Ingests raw Tesco transaction events from Event Hub into ADLS Gen2 bronze layer.

import json
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType,
)

# Bug fix #6: replaced hardcoded <STORAGE_ACCOUNT> with dynamic secret lookup
STORAGE_ACCOUNT = dbutils.secrets.get(scope="adls-scope", key="STORAGE_ACCOUNT")
EVENTHUB_CS = dbutils.secrets.get(scope="adls-scope", key="EVENTHUB_CONNECTION_STRING")

BRONZE_PATH = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net/transactions"
CHECKPOINT_PATH = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net/_checkpoints/transactions"

EVENTHUB_NAME = "transactions"
CONSUMER_GROUP = "$Default"

# ── Schema ────────────────────────────────────────────────────────────────────
transaction_schema = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("store_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("category", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("channel", StringType(), True),  # online / in-store
])

# ── Read from Event Hub (Structured Streaming) ────────────────────────────────
eh_conf = {
    "eventhubs.connectionString": sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(
        EVENTHUB_CS
    ),
    "eventhubs.eventHubName": EVENTHUB_NAME,
    "eventhubs.consumerGroup": CONSUMER_GROUP,
    "eventhubs.startingPosition": json.dumps({"offset": "-1", "seqNo": -1, "enqueuedTime": None, "isInclusive": True}),
}

raw_stream = (
    spark.readStream.format("eventhubs")
    .options(**eh_conf)
    .load()
)

# ── Parse JSON payload ────────────────────────────────────────────────────────
parsed_stream = (
    raw_stream
    .select(
        F.col("enqueuedTime").alias("enqueued_at"),
        F.col("partitionId").cast("int").alias("partition_id"),
        F.from_json(F.col("body").cast("string"), transaction_schema).alias("data"),
    )
    .select("enqueued_at", "partition_id", "data.*")
    .withColumn("ingest_date", F.to_date("timestamp"))
)

# ── Write to ADLS Gen2 bronze (Delta, partitioned by date) ────────────────────
query = (
    parsed_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .partitionBy("ingest_date")
    .start(BRONZE_PATH)
)

query.awaitTermination()
