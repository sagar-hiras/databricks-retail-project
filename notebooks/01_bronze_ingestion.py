# Databricks notebook source
"""
Bronze Layer: Raw Data Ingestion
Ingests raw customer churn data from external sources into the bronze layer.
"""

from pyspark.sql import SparkSession

# COMMAND ----------

def ingest_raw_data(source_path: str, bronze_table: str):
    """
    Ingest raw data from source and write to bronze table.
    
    Args:
        source_path: Path to raw data source
        bronze_table: Target bronze table name
    """
    spark = SparkSession.builder.appName("BronzeIngestion").getOrCreate()
    
    # Read raw data
    df = spark.read.csv(source_path, header=True, inferSchema=True)
    
    # Write to bronze table
    df.write.mode("overwrite").format("delta").saveAsTable(bronze_table)
    
    print(f"Ingested {df.count()} records to {bronze_table}")

# COMMAND ----------

if __name__ == "__main__":
    ingest_raw_data("data/sample_churn_data.csv", "bronze_churn_raw")
