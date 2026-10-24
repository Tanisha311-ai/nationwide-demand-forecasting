
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Evaluating demand forecasting models...", flush=True)

forecast_path = OUTPUT_DIR / "forecast_validation.csv"
valid_path = PROCESSED_DIR / "validation_features.csv"

if not forecast_path.exists():
    raise FileNotFoundError(
        "Forecast file not found. Run 03_forecasting.py first."
    )

if not valid_path.exists():
    raise FileNotFoundError(
        "Validation features not found. Run 02_features.py first."
    )

forecasts = pd.read_csv(forecast_path)
valid = pd.read_csv(valid_path)

if len(forecasts) != len(valid):
    raise ValueError(
        "Forecast and validation row counts do not match."
    )

actual = forecasts["actual_sales"].to_numpy(dtype=float)
lightgbm_pred = forecasts["predicted_sales"].to_numpy(dtype=float)

if "lag_7" not in valid.columns:
    raise ValueError("lag_7 is missing from validation features.")

baseline_pred = valid["lag_7"].to_numpy(dtype=float)

def calculate_metrics(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_actual = np.abs(actual).sum()

    if total_actual > 0:
        wape = np.abs(actual - predicted).sum() / total_actual * 100
    else:
        wape = np.nan

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape
    }

results = pd.DataFrame([
    calculate_metrics(actual, baseline_pred, "Seven-day baseline"),
    calculate_metrics(actual, lightgbm_pred, "LightGBM")
])

print("\nModel Comparison")
print(results.round(4).to_string(index=False))

baseline_wape = results.loc[
    results["model"] == "Seven-day baseline", "WAPE_percent"
].iloc[0]

lightgbm_wape = results.loc[
    results["model"] == "LightGBM", "WAPE_percent"
].iloc[0]

if baseline_wape > 0 and np.isfinite(baseline_wape):
    improvement = (
        (baseline_wape - lightgbm_wape) / baseline_wape
    ) * 100

    print(f"\nLightGBM WAPE: {lightgbm_wape:.2f}%")
    print(f"Baseline WAPE: {baseline_wape:.2f}%")
    print(f"Relative WAPE improvement: {improvement:.2f}%")

    if improvement > 0:
        print("LightGBM performs better than the baseline.")
    elif improvement < 0:
        print("The baseline performs better than LightGBM.")
    else:
        print("Both models have the same WAPE.")
else:
    print("Baseline WAPE could not be used for comparison.")

output_path = OUTPUT_DIR / "model_comparison.csv"
results.to_csv(output_path, index=False)

print("\nComparison saved to:", output_path)
print("Evaluation completed successfully.")

