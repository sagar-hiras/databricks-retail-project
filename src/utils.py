"""
Utility functions for Databricks churn ML project.
"""

from typing import List, Dict, Any
from pyspark.sql import DataFrame, SparkSession


def get_spark_session(app_name: str = "ChurnML") -> SparkSession:
    """
    Get or create a Spark session.
    
    Args:
        app_name: Application name for the Spark session
        
    Returns:
        SparkSession object
    """
    return SparkSession.builder.appName(app_name).getOrCreate()


def validate_data_schema(df: DataFrame, expected_columns: List[str]) -> bool:
    """
    Validate that a DataFrame has expected columns.
    
    Args:
        df: Spark DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        True if all expected columns present, False otherwise
    """
    actual_columns = set(df.columns)
    expected = set(expected_columns)
    
    return expected.issubset(actual_columns)


def print_data_summary(df: DataFrame) -> Dict[str, Any]:
    """
    Print summary statistics for a DataFrame.
    
    Args:
        df: Spark DataFrame
        
    Returns:
        Dictionary with row count and column count
    """
    summary = {
        "row_count": df.count(),
        "column_count": len(df.columns),
        "columns": df.columns
    }
    
    print(f"DataFrame Summary:")
    print(f"  Rows: {summary['row_count']}")
    print(f"  Columns: {summary['column_count']}")
    print(f"  Column Names: {summary['columns']}")
    
    return summary


def log_transformation(operation: str, input_rows: int, output_rows: int):
    """
    Log data transformation details.
    
    Args:
        operation: Description of the operation
        input_rows: Number of rows before transformation
        output_rows: Number of rows after transformation
    """
    print(f"[{operation}]")
    print(f"  Input rows: {input_rows}")
    print(f"  Output rows: {output_rows}")
    print(f"  Row loss: {input_rows - output_rows}")
