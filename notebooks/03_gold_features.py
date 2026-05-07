# Databricks notebook source
"""
Gold Layer: Feature Engineering & Aggregation
Creates analytical features and aggregations for ML model training.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# COMMAND ----------

def create_ml_features(silver_table: str, gold_table: str):
    """
    Engineer features and create ML-ready dataset.
    
    Args:
        silver_table: Source silver table
        gold_table: Target gold table
    """
    spark = SparkSession.builder.appName("GoldFeatures").getOrCreate()
    
    # Read silver data
    df = spark.read.table(silver_table)
    
    # Feature engineering
    df_features = df.select(
        col("customer_id"),
        col("age"),
        col("tenure_months"),
        col("monthly_charges"),
        col("total_charges"),
        when(col("contract_type") == "month-to-month", 1).otherwise(0).alias("month_to_month"),
        when(col("internet_service") == "fiber", 1).otherwise(0).alias("has_fiber"),
        col("churn").alias("target")
    )
    
    # Write to gold table
    df_features.write.mode("overwrite").format("delta").saveAsTable(gold_table)
    
    print(f"ML features written to {gold_table}")

# COMMAND ----------

if __name__ == "__main__":
    create_ml_features("silver_churn_clean", "gold_churn_features")
