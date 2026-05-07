"""
Unit tests for utility functions.
"""

import pytest
from pyspark.sql import SparkSession
from src.utils import validate_data_schema, print_data_summary


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    return SparkSession.builder.appName("test").getOrCreate()


@pytest.fixture
def sample_df(spark):
    """Create a sample DataFrame for testing."""
    data = [
        (1, 32, 24, 65.50),
        (2, 45, 12, 89.25),
        (3, 28, 6, 55.00),
    ]
    columns = ["customer_id", "age", "tenure_months", "monthly_charges"]
    return spark.createDataFrame(data, schema=columns)


class TestValidateDataSchema:
    """Tests for validate_data_schema function."""
    
    def test_valid_schema(self, sample_df):
        """Test validation with all expected columns present."""
        expected = ["customer_id", "age", "tenure_months"]
        assert validate_data_schema(sample_df, expected) is True
    
    def test_missing_column(self, sample_df):
        """Test validation with missing column."""
        expected = ["customer_id", "age", "missing_column"]
        assert validate_data_schema(sample_df, expected) is False
    
    def test_extra_columns(self, sample_df):
        """Test validation with more columns than expected."""
        expected = ["customer_id", "age"]
        assert validate_data_schema(sample_df, expected) is True


class TestPrintDataSummary:
    """Tests for print_data_summary function."""
    
    def test_summary_output(self, sample_df):
        """Test that summary returns correct counts."""
        summary = print_data_summary(sample_df)
        assert summary["row_count"] == 3
        assert summary["column_count"] == 4
        assert len(summary["columns"]) == 4
