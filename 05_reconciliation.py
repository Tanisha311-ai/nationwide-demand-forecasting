
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

forecast_path = OUTPUT_DIR / "forecast_validation.csv"

if not forecast_path.exists():
    raise FileNotFoundError(
        "Forecast file not found. Run 03_forecasting.py first."
    )

print("Starting hierarchical forecast reconciliation...", flush=True)

df = pd.read_csv(forecast_path)

required_cols = [
    "item_id",
    "store_id",
    "state_id",
    "actual_sales",
    "predicted_sales"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df["predicted_sales"] = pd.to_numeric(
    df["predicted_sales"], errors="coerce"
).fillna(0).clip(lower=0)

df["actual_sales"] = pd.to_numeric(
    df["actual_sales"], errors="coerce"
)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

elif "d" in df.columns:
    df["date"] = df["d"]

df = df.dropna(subset=["actual_sales", "item_id", "store_id", "state_id"])

print("Creating store-level forecasts...", flush=True)

group_cols = ["state_id", "store_id"]

if "date" in df.columns:
    group_cols.append("date")

store_forecasts = (
    df.groupby(group_cols, as_index=False, observed=True)
    .agg(
        actual_sales=("actual_sales", "sum"),
        predicted_sales=("predicted_sales", "sum")
    )
)

store_forecasts["absolute_error"] = np.abs(
    store_forecasts["actual_sales"]
    - store_forecasts["predicted_sales"]
)

store_forecasts.to_csv(
    OUTPUT_DIR / "store_level_forecasts.csv",
    index=False
)

print("Creating state-level forecasts...", flush=True)

state_group_cols = ["state_id"]

if "date" in df.columns:
    state_group_cols.append("date")

state_forecasts = (
    df.groupby(state_group_cols, as_index=False, observed=True)
    .agg(
        actual_sales=("actual_sales", "sum"),
        predicted_sales=("predicted_sales", "sum")
    )
)

state_forecasts["absolute_error"] = np.abs(
    state_forecasts["actual_sales"]
    - state_forecasts["predicted_sales"]
)

state_forecasts.to_csv(
    OUTPUT_DIR / "state_level_forecasts.csv",
    index=False
)

print("Creating product-level forecasts...", flush=True)

product_group_cols = ["state_id", "store_id", "item_id"]

if "date" in df.columns:
    product_group_cols.append("date")

product_forecasts = (
    df.groupby(product_group_cols, as_index=False, observed=True)
    .agg(
        actual_sales=("actual_sales", "sum"),
        predicted_sales=("predicted_sales", "sum")
    )
)

product_forecasts.to_csv(
    OUTPUT_DIR / "product_level_forecasts.csv",
    index=False
)

print("\nReconciliation Summary")
print("Product-level rows:", len(product_forecasts))
print("Store-level rows:", len(store_forecasts))
print("State-level rows:", len(state_forecasts))

print("\nTotal predicted sales:", round(df["predicted_sales"].sum(), 2))
print("Total actual sales:", round(df["actual_sales"].sum(), 2))

print("\nSaved files:")
print("product_level_forecasts.csv")
print("store_level_forecasts.csv")
print("state_level_forecasts.csv")

print("\nHierarchical aggregation completed successfully.")
