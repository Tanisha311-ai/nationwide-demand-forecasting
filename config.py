
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Create directories if they do not exist
for directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    REPORT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Dataset file paths
CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
SALES_TRAIN_FILE = RAW_DATA_DIR / "sales_train_evaluation.csv"
SALES_VALIDATION_FILE = RAW_DATA_DIR / "sales_train_validation.csv"
SELL_PRICES_FILE = RAW_DATA_DIR / "sell_prices.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"

# Forecasting configuration
TOP_N_ITEMS = 50
FORECAST_HORIZON_DAYS = 28
HOLDOUT_DAYS = 28
HISTORY_WINDOW_DAYS = 56
SEASONAL_LAG_DAYS = 7

# Inventory configuration
LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL_Z = 1.645

# Output file paths
DATASET_SUMMARY_FILE = OUTPUT_DIR / "dataset_summary.csv"
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.csv"
VALIDATION_FEATURES_FILE = PROCESSED_DATA_DIR / "validation_features.csv"
FUTURE_FORECAST_FILE = OUTPUT_DIR / "future_28_day_forecast.csv"
FUTURE_METRICS_FILE = OUTPUT_DIR / "future_28_day_metrics.csv"
INVENTORY_OPTIMIZATION_FILE = OUTPUT_DIR / "inventory_optimization.csv"
INVENTORY_RISK_FILE = OUTPUT_DIR / "inventory_risk_assessment.csv"
REPLENISHMENT_FILE = OUTPUT_DIR / "replenishment_recommendations.csv"

