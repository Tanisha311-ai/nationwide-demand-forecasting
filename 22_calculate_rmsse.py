import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent

forecast_path = BASE_DIR / "data/outputs/future_28_day_forecast.csv"
training_path = BASE_DIR / "data/raw/sales_train_validation.csv"
metrics_path = BASE_DIR / "data/outputs/future_28_day_metrics.csv"
details_path = BASE_DIR / "data/outputs/future_28_day_rmsse_by_series.csv"

# Load forecast data
forecast = pd.read_csv(forecast_path)

# Identify the 500 item-store combinations
target_keys = set(
    zip(forecast["item_id"], forecast["store_id"])
)

# Historical training columns: d_1 to d_1913
sales_columns = [f"d_{i}" for i in range(1, 1914)]
use_columns = ["item_id", "store_id"] + sales_columns

# Read training data in chunks to reduce memory usage
histories = {}

for chunk in pd.read_csv(
    training_path,
    usecols=use_columns,
    chunksize=250
):
    mask = [
        (item, store) in target_keys
        for item, store in zip(chunk["item_id"], chunk["store_id"])
    ]

    selected = chunk.loc[mask]

    for _, row in selected.iterrows():
        key = (row["item_id"], row["store_id"])
        histories[key] = row[sales_columns].to_numpy(dtype=float)

print("Historical series found:", len(histories))
print("Forecast series:", len(target_keys))

if len(histories) != len(target_keys):
    raise ValueError(
        "Some forecast series could not be matched to training history."
    )

# Calculate RMSSE for each item-store combination
results = []

for (item, store), group in forecast.groupby(["item_id", "store_id"]):

    group = group.sort_values("day_num")

    actual = group["actual_sales"].to_numpy(dtype=float)
    predicted = group["predicted_sales"].to_numpy(dtype=float)
    baseline = group["baseline_forecast"].to_numpy(dtype=float)

    history = histories[(item, store)]

    # Seven-day seasonal-naive training error
    seasonal_errors = history[7:] - history[:-7]
    denominator = np.mean(seasonal_errors ** 2)

    if denominator == 0:
        continue

    # Forecast-period mean squared errors
    lgb_mse = np.mean((actual - predicted) ** 2)
    baseline_mse = np.mean((actual - baseline) ** 2)

    results.append({
        "item_id": item,
        "store_id": store,
        "LightGBM_RMSSE": np.sqrt(lgb_mse / denominator),
        "Baseline_RMSSE": np.sqrt(baseline_mse / denominator)
    })

# Save item-store-level results
rmsse_df = pd.DataFrame(results)
rmsse_df.to_csv(details_path, index=False)

if rmsse_df.empty:
    raise ValueError("No RMSSE values could be calculated.")

# Mean of item-store RMSSE values
lgb_rmsse = rmsse_df["LightGBM_RMSSE"].mean()
baseline_rmsse = rmsse_df["Baseline_RMSSE"].mean()

# Update the existing metrics file
metrics = pd.read_csv(metrics_path)

rmsse_values = {
    "LightGBM recursive forecast": lgb_rmsse,
    "Seven-day seasonal baseline": baseline_rmsse
}

metrics["RMSSE"] = metrics["model"].map(rmsse_values)
metrics.to_csv(metrics_path, index=False)

print("\nRMSSE RESULTS")
print("------------------------------")
print("LightGBM RMSSE:", round(lgb_rmsse, 4))
print("Baseline RMSSE:", round(baseline_rmsse, 4))
print("Series included:", len(rmsse_df))
print("Series skipped (zero denominator):", len(target_keys) - len(rmsse_df))

print("\nUpdated metrics:")
print(metrics.to_string(index=False))

print("\nFiles saved:")
print(metrics_path)
print(details_path)