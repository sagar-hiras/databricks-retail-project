# Databricks notebook source
"""
Model Serving: Load and serve the trained churn prediction model.
Provides functions for batch and real-time predictions.
"""

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# COMMAND ----------

def load_and_score(model_path: str, input_table: str, output_table: str):
    """
    Load trained model and generate predictions on new data.
    
    Args:
        model_path: Path to saved model
        input_table: Table with new data to score
        output_table: Table to write predictions
    """
    spark = SparkSession.builder.appName("ModelServing").getOrCreate()
    
    # Load saved model
    model = PipelineModel.load(model_path)
    
    # Read input data
    df = spark.read.table(input_table)
    
    # Generate predictions
    predictions = model.transform(df)
    
    # Write predictions
    predictions.write.mode("overwrite").format("delta").saveAsTable(output_table)
    
    print(f"Predictions written to {output_table}")

# COMMAND ----------

def batch_predict(model_path: str, data_path: str) -> dict:
    """
    Perform batch predictions on new data.
    
    Args:
        model_path: Path to saved model
        data_path: Path to input data
        
    Returns:
        Dictionary with prediction statistics
    """
    spark = SparkSession.builder.appName("BatchPredict").getOrCreate()
    
    model = PipelineModel.load(model_path)
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    predictions = model.transform(df)
    
    results = {
        "total_predictions": predictions.count(),
        "predicted_churners": predictions.filter("prediction = 1").count()
    }
    
    return results

# COMMAND ----------

if __name__ == "__main__":
    load_and_score("/tmp/churn_model", "gold_churn_features", "predictions_churn")
