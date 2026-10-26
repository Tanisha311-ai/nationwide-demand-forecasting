
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

LIGHTGBM_PATH = OUTPUT_DIR / "forecast_validation.csv"
PROPHET_PATH = OUTPUT_DIR / "prophet_forecasts.csv"

print("=" * 60)
print("FAIR MODEL COMPARISON: PROPHET VS LIGHTGBM")
print("=" * 60)


# 2. LOAD FORECASTS

lgb = pd.read_csv(LIGHTGBM_PATH)
prophet = pd.read_csv(PROPHET_PATH)

lgb["date"] = pd.to_datetime(lgb["date"])
prophet["ds"] = pd.to_datetime(prophet["ds"])

# Aggregate LightGBM predictions across stores for each item and date.
lgb_daily = (
    lgb.groupby(["item_id", "date"], as_index=False)
    .agg(
        actual_sales=("actual_sales", "sum"),
        predicted_sales=("predicted_sales", "sum")
    )
)

# Prophet forecasts are already aggregated by item and date.
prophet_daily = prophet[
    ["item_id", "ds", "actual_sales", "predicted_sales"]
].rename(columns={"ds": "date"})

# Keep only item-date combinations present in both datasets.
comparison = lgb_daily.merge(
    prophet_daily[["item_id", "date", "predicted_sales"]],
    on=["item_id", "date"],
    how="inner",
    suffixes=("_lightgbm", "_prophet")
)

# Check that both models use the same actual demand.
actual_difference = (
    comparison["actual_sales"] -
    prophet_daily.merge(
        lgb_daily[["item_id", "date", "actual_sales"]],
        on=["item_id", "date"],
        how="inner",
        suffixes=("_prophet", "_lightgbm")
    )["actual_sales_lightgbm"]
)

if not np.allclose(actual_difference.to_numpy(), 0):
    raise ValueError("Actual sales differ between model datasets.")

actual = comparison["actual_sales"]
predictions = {
    "LightGBM": comparison["predicted_sales_lightgbm"],
    "Prophet": comparison["predicted_sales_prophet"]
}


# 3. CALCULATE METRICS

results = []

for model_name, pred in predictions.items():
    absolute_errors = (actual - pred).abs()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    wape = (
        absolute_errors.sum() / actual.abs().sum() * 100
        if actual.abs().sum() > 0
        else np.nan
    )

    results.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape
    })

metrics = pd.DataFrame(results).sort_values("WAPE_percent")

print("\nComparison level: item aggregated across stores")
print("Matched item-date rows:", len(comparison))
print("Unique products:", comparison["item_id"].nunique())
print("Unique dates:", comparison["date"].nunique())
print("\nMODEL COMPARISON")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 4. SAVE RESULTS

output_path = OUTPUT_DIR / "model_comparison_prophet_lightgbm.csv"
metrics.to_csv(output_path, index=False)

print("\nSaved comparison:", output_path)
print("\nCOMPARISON COMPLETED")
