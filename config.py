

from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Create required directories if they do not exist
for directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    REPORT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# M5 dataset paths
CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"

SALES_TRAIN_FILE = (
    RAW_DATA_DIR / "sales_train_evaluation.csv"
)

SALES_VALIDATION_FILE = (
    RAW_DATA_DIR / "sales_train_validation.csv"
)

SELL_PRICES_FILE = RAW_DATA_DIR / "sell_prices.csv"

SAMPLE_SUBMISSION_FILE = (
    RAW_DATA_DIR / "sample_submission.csv"
)


# Forecasting configuration
TOP_N_ITEMS = 50
FORECAST_HORIZON_DAYS = 28
HOLDOUT_DAYS = 28
HISTORY_WINDOW_DAYS = 56
SEASONAL_LAG_DAYS = 7


# Inventory policy assumptions
LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL_Z = 1.645


# Forecasting outputs
FUTURE_FORECAST_FILE = (
    OUTPUT_DIR / "future_28_day_forecast.csv"
)

FUTURE_METRICS_FILE = (
    OUTPUT_DIR / "future_28_day_metrics.csv"
)

MODEL_METRICS_FILE = (
    OUTPUT_DIR / "model_metrics.csv"
)

PREDICTION_INTERVAL_METRICS_FILE = (
    OUTPUT_DIR / "prediction_interval_metrics.csv"
)


# Model comparison outputs
STORE_PERFORMANCE_FILE = (
    OUTPUT_DIR / "store_forecast_performance.csv"
)

MODEL_COMPARISON_FILE = (
    OUTPUT_DIR / "model_comparison_prophet_lightgbm.csv"
)

TFT_METRICS_FILE = OUTPUT_DIR / "tft_metrics.csv"

TFT_LIGHTGBM_COMPARISON_FILE = (
    OUTPUT_DIR / "tft_lightgbm_comparison.csv"
)


# Hierarchy and reconciliation outputs
HIERARCHY_FILE = OUTPUT_DIR / "hierarchy.csv"

RECONCILED_FORECAST_FILE = (
    OUTPUT_DIR / "reconciled_forecasts.csv"
)

RECONCILIATION_METRICS_FILE = (
    OUTPUT_DIR / "reconciliation_metrics.csv"
)


# Inventory outputs
INVENTORY_OPTIMIZATION_FILE = (
    OUTPUT_DIR / "inventory_optimization.csv"
)

INVENTORY_RISK_FILE = (
    OUTPUT_DIR / "inventory_risk_assessment.csv"
)

REPLENISHMENT_FILE = (
    OUTPUT_DIR / "replenishment_recommendations.csv"
)

ORTOOLS_OPTIMIZATION_FILE = (
    OUTPUT_DIR / "ortools_optimized_orders.csv"
)

INVENTORY_SENSITIVITY_FILE = (
    OUTPUT_DIR / "inventory_sensitivity_results.csv"
)

SERVICE_LEVEL_BY_SCENARIO_FILE = (
    OUTPUT_DIR / "inventory_service_level_by_scenario.csv"
)

SERVICE_LEVEL_SUMMARY_FILE = (
    OUTPUT_DIR / "inventory_service_level_summary.csv"
)
