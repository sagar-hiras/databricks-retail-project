# Databricks notebook source
"""
Silver Layer: Data Cleaning & Transformation
Cleans and standardizes raw data from bronze layer.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, lower

# COMMAND ----------

def clean_and_transform(bronze_table: str, silver_table: str):
    """
    Clean and transform data from bronze to silver layer.
    
    Args:
        bronze_table: Source bronze table
        silver_table: Target silver table
    """
    spark = SparkSession.builder.appName("SilverCleaning").getOrCreate()
    
    # Read bronze data
    df = spark.read.table(bronze_table)
    
    # Data cleaning transformations
    df_clean = df.select(
        col("customer_id").cast("bigint"),
        col("age").cast("int"),
        col("tenure_months").cast("int"),
        col("monthly_charges").cast("double"),
        col("total_charges").cast("double"),
        lower(trim(col("contract_type"))).alias("contract_type"),
        lower(trim(col("internet_service"))).alias("internet_service"),
        col("churn").cast("int")
    )
    
    # Write to silver table
    df_clean.write.mode("overwrite").format("delta").saveAsTable(silver_table)
    
    print(f"Cleaned data written to {silver_table}")

# COMMAND ----------

if __name__ == "__main__":
    clean_and_transform("bronze_churn_raw", "silver_churn_clean")
