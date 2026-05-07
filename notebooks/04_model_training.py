# Databricks notebook source
"""
Model Training: Train and evaluate churn prediction model.
Uses MLlib for distributed model training.
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

def train_churn_model(gold_table: str, model_path: str):
    """
    Train a churn prediction model using Random Forest.
    
    Args:
        gold_table: Source gold table with features
        model_path: Path to save trained model
    """
    spark = SparkSession.builder.appName("ModelTraining").getOrCreate()
    
    # Read feature data
    df = spark.read.table(gold_table)
    
    # Split data
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    # Prepare features
    feature_cols = ["age", "tenure_months", "monthly_charges", "total_charges", 
                   "month_to_month", "has_fiber"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Train Random Forest model
    rf = RandomForestClassifier(labelCol="target", featuresCol="features", 
                               numTrees=100, maxDepth=10, seed=42)
    
    pipeline = Pipeline(stages=[assembler, rf])
    model = pipeline.fit(train_data)
    
    # Evaluate
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="target")
    auc = evaluator.evaluate(predictions)
    
    print(f"Model AUC: {auc:.4f}")
    
    # Save model
    model.write().overwrite().save(model_path)
    print(f"Model saved to {model_path}")

# COMMAND ----------

if __name__ == "__main__":
    train_churn_model("gold_churn_features", "/tmp/churn_model")
