
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
train_path = PROCESSED_DIR / "train_features.csv"

for path in (forecast_path, valid_path, train_path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

forecasts = pd.read_csv(forecast_path)
valid = pd.read_csv(valid_path)
train = pd.read_csv(train_path)

# Validate required columns.
required_forecast = {"actual_sales", "predicted_sales"}
required_valid = {
    "id", "item_id", "store_id", "day_num",
    "sales", "lag_1", "lag_7", "lag_28"
}
required_train = {"item_id", "store_id", "day_num", "sales"}

if not required_forecast.issubset(forecasts.columns):
    raise ValueError(
        f"Forecast file is missing: {required_forecast - set(forecasts.columns)}"
    )

if not required_valid.issubset(valid.columns):
    raise ValueError(
        f"Validation file is missing: {required_valid - set(valid.columns)}"
    )

if not required_train.issubset(train.columns):
    raise ValueError(
        f"Training file is missing: {required_train - set(train.columns)}"
    )

# Align rows using identifiers rather than relying only on row count.
if "id" not in forecasts.columns or "day_num" not in forecasts.columns:
    raise ValueError(
        "forecast_validation.csv must contain id and day_num columns. "
        "Update 03_forecasting.py to save these identifiers with predictions."
    )

valid = valid.merge(
    forecasts[["id", "day_num", "actual_sales", "predicted_sales"]],
    on=["id", "day_num"],
    how="left",
    validate="one_to_one"
)

if valid["predicted_sales"].isna().any():
    raise ValueError("Some validation rows do not have matching predictions.")

actual = np.maximum(valid["actual_sales"].to_numpy(dtype=float), 0)

predictions = {
    "Naive baseline (lag 1)": valid["lag_1"].to_numpy(dtype=float),
    "7-day seasonal baseline": valid["lag_7"].to_numpy(dtype=float),
    "28-day seasonal baseline": valid["lag_28"].to_numpy(dtype=float),
    "LightGBM": valid["predicted_sales"].to_numpy(dtype=float),
}

for name in predictions:
    predictions[name] = np.maximum(predictions[name], 0)

# Compute the RMSSE denominator from consecutive historical daily sales.
train = train.sort_values(["id", "day_num"]).copy()

train["squared_difference"] = (
    train.groupby("id")["sales"].diff().pow(2)
)

scale_by_id = (
    train.groupby("id")["squared_difference"]
    .mean()
    .rename("scale")
)

valid = valid.join(scale_by_id, on="id")
scales = valid["scale"].to_numpy(dtype=float)


def calculate_metrics(actual, predicted, scales, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_actual = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    squared_errors = np.square(actual - predicted)
    mask = np.isfinite(scales) & (scales > 0)

    rmsse = (
        np.sqrt(np.mean(squared_errors[mask] / scales[mask]))
        if mask.any()
        else np.nan
    )

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "RMSSE": rmsse,
    }


results = pd.DataFrame([
    calculate_metrics(actual, pred, scales, name)
    for name, pred in predictions.items()
])

print("\nModel Comparison")
print(results.round(4).to_string(index=False))

# Compare LightGBM against the 7-day seasonal baseline.
baseline_wape = results.loc[
    results["model"] == "7-day seasonal baseline",
    "WAPE_percent"
].iloc[0]

lightgbm_wape = results.loc[
    results["model"] == "LightGBM",
    "WAPE_percent"
].iloc[0]

if np.isfinite(baseline_wape) and baseline_wape > 0:
    improvement = (
        (baseline_wape - lightgbm_wape) / baseline_wape
    ) * 100

    print(f"\n7-day baseline WAPE: {baseline_wape:.2f}%")
    print(f"LightGBM WAPE: {lightgbm_wape:.2f}%")
    print(f"Relative WAPE improvement: {improvement:.2f}%")

output_path = OUTPUT_DIR / "model_comparison.csv"
results.to_csv(output_path, index=False)

print("\nComparison saved to:", output_path)
print("Evaluation completed successfully.")



