
from pathlib import Path
import time

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PROPHET DEMAND FORECASTING")
print("=" * 60)


# 2. LOAD DATA

train_path = PROCESSED_DIR / "train_features.csv"
validation_path = PROCESSED_DIR / "validation_features.csv"

train = pd.read_csv(train_path)
validation = pd.read_csv(validation_path)

train["date"] = pd.to_datetime(train["date"])
validation["date"] = pd.to_datetime(validation["date"])

print("Training rows:", len(train))
print("Validation rows:", len(validation))

# Prophet requires columns named ds and y.
# Aggregate across stores for each product to reduce the
# number of individual Prophet models that must be trained.

train_daily = (
    train.groupby(["item_id", "date"], as_index=False)["sales"]
    .sum()
    .rename(columns={"date": "ds", "sales": "y"})
)

validation_daily = (
    validation.groupby(["item_id", "date"], as_index=False)["sales"]
    .sum()
    .rename(columns={"date": "ds", "sales": "actual_sales"})
)

items = sorted(train_daily["item_id"].unique())

print("Products to forecast:", len(items))
print("Training Prophet models...")


# 3. TRAIN ONE MODEL PER PRODUCT

all_forecasts = []
start_time = time.time()

for index, item in enumerate(items, start=1):

    item_train = train_daily[train_daily["item_id"] == item][
        ["ds", "y"]
    ].copy()

    item_validation = validation_daily[
        validation_daily["item_id"] == item
    ][["ds", "actual_sales"]].copy()

    # Prophet requires finite numeric targets.
    item_train = item_train.replace([np.inf, -np.inf], np.nan)
    item_train = item_train.dropna(subset=["ds", "y"])

    if item_train.empty:
        print(f"Skipping {item}: no usable training data")
        continue

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.80
    )

    model.fit(item_train)

    future = item_validation[["ds"]].copy()
    predictions = model.predict(future)

    result = item_validation.copy()
    result["item_id"] = item
    result["predicted_sales"] = predictions["yhat"].to_numpy()
    result["lower_bound_80"] = predictions["yhat_lower"].to_numpy()
    result["upper_bound_80"] = predictions["yhat_upper"].to_numpy()

    # Demand cannot be negative.
    result["predicted_sales"] = result["predicted_sales"].clip(lower=0)
    result["lower_bound_80"] = result["lower_bound_80"].clip(lower=0)
    result["upper_bound_80"] = result["upper_bound_80"].clip(lower=0)

    all_forecasts.append(result)

    print(f"Completed {index}/{len(items)}: {item}")


# 4. COMBINE FORECASTS

if not all_forecasts:
    raise RuntimeError("Prophet did not produce any forecasts.")

forecast_df = pd.concat(all_forecasts, ignore_index=True)

forecast_df["absolute_error"] = (
    forecast_df["actual_sales"] - forecast_df["predicted_sales"]
).abs()

forecast_df["inside_interval"] = (
    (forecast_df["actual_sales"] >= forecast_df["lower_bound_80"])
    & (forecast_df["actual_sales"] <= forecast_df["upper_bound_80"])
)


# 5. EVALUATE

actual = forecast_df["actual_sales"].to_numpy()
predicted = forecast_df["predicted_sales"].to_numpy()

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

total_actual = np.abs(actual).sum()
wape = (
    forecast_df["absolute_error"].sum() / total_actual * 100
    if total_actual > 0
    else np.nan
)

coverage = forecast_df["inside_interval"].mean() * 100

mean_interval_width = (
    forecast_df["upper_bound_80"] - forecast_df["lower_bound_80"]
).mean()

metrics = pd.DataFrame([{
    "model": "Prophet",
    "MAE": mae,
    "RMSE": rmse,
    "WAPE_percent": wape,
    "nominal_interval_coverage_percent": 80,
    "observed_interval_coverage_percent": coverage,
    "mean_interval_width": mean_interval_width
}])

print("\nPROPHET RESULTS")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")
print(f"Observed 80% interval coverage: {coverage:.2f}%")
print(f"Mean interval width: {mean_interval_width:.4f}")
print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")


# 6. SAVE OUTPUTS

forecast_path = OUTPUT_DIR / "prophet_forecasts.csv"
metrics_path = OUTPUT_DIR / "prophet_metrics.csv"

forecast_df.to_csv(forecast_path, index=False)
metrics.to_csv(metrics_path, index=False)

print("\nSaved forecasts:", forecast_path)
print("Saved metrics:", metrics_path)
print("\nPROPHET FORECASTING COMPLETED")
