# Churn ML Project

A Databricks-based machine learning project for predicting customer churn using the medallion architecture (bronze, silver, gold layers).

## Project Overview

This project implements a complete ML pipeline on Databricks:
- **Bronze Layer**: Raw data ingestion
- **Silver Layer**: Data cleaning and transformation
- **Gold Layer**: Feature engineering and ML-ready datasets
- **Model Training**: Random Forest classifier for churn prediction
- **Model Serving**: Batch and real-time prediction capabilities

## Project Structure

```
churn-ml-project/
├── data/                        # Sample raw data (CSV)
├── notebooks/
│   ├── 01_bronze_ingestion.py   # Data ingestion
│   ├── 02_silver_cleaning.py    # Data cleaning
│   ├── 03_gold_features.py      # Feature engineering
│   ├── 04_model_training.py     # Model training
│   └── 05_model_serving.py      # Model serving
├── src/
│   └── utils.py                 # Helper functions
├── tests/
│   └── test_utils.py            # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Databricks Runtime 11.0+
- PySpark 3.5.0+
- Python 3.9+

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Upload project to Databricks workspace

3. Run notebooks in sequence:
   - Start with `01_bronze_ingestion.py`
   - Continue through `02`, `03`, `04`, `05`

## Data Schema

The sample data includes:
- `customer_id`: Unique customer identifier
- `age`: Customer age
- `tenure_months`: Number of months as customer
- `monthly_charges`: Monthly subscription charges
- `total_charges`: Total lifetime charges
- `contract_type`: Type of service contract
- `internet_service`: Type of internet service
- `churn`: Target variable (1=churned, 0=retained)

## Model Performance

The Random Forest model achieves competitive AUC on the test set. Performance metrics are logged during training.

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src
```

## Contributing

1. Create a feature branch
2. Make changes and write tests
3. Run test suite to verify
4. Submit pull request

## License

MIT License