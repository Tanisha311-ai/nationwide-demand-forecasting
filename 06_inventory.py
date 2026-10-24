
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

SALES_PATH = RAW_DIR / "sales_train_evaluation.csv"
FORECAST_PATH = OUTPUT_DIR / "future_28_day_forecast.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
HISTORY_WINDOW_DAYS = 56
SERVICE_LEVEL_Z = 1.645

print("Starting inventory optimization...")

if not SALES_PATH.exists():
    raise FileNotFoundError(f"Historical sales file not found: {SALES_PATH}")

if not FORECAST_PATH.exists():
    raise FileNotFoundError(
        f"Future forecast file not found: {FORECAST_PATH}. "
        "Run 07_future_validation.py first."
    )

forecast = pd.read_csv(FORECAST_PATH)

required_forecast_cols = [
    "item_id",
    "store_id",
    "state_id",
    "actual_sales",
    "predicted_sales",
]

missing = [
    col for col in required_forecast_cols
    if col not in forecast.columns
]

if missing:
    raise ValueError(f"Missing forecast columns: {missing}")

forecast["predicted_sales"] = forecast["predicted_sales"].clip(lower=0)
forecast["actual_sales"] = forecast["actual_sales"].clip(lower=0)

forecast_summary = (
    forecast.groupby(
        ["item_id", "store_id", "state_id"],
        as_index=False
    )
    .agg(
        actual_demand_28d=("actual_sales", "sum"),
        forecast_demand_28d=("predicted_sales", "sum"),
        forecast_days=("predicted_sales", "size"),
    )
)

forecast_summary["forecast_daily_demand"] = (
    forecast_summary["forecast_demand_28d"]
    / forecast_summary["forecast_days"].replace(0, np.nan)
)

zero_demand = forecast_summary["actual_demand_28d"].eq(0)

forecast_summary["zero_actual_demand_flag"] = zero_demand

forecast_summary["forecast_overprediction_units"] = (
    forecast_summary["forecast_demand_28d"]
    - forecast_summary["actual_demand_28d"]
).clip(lower=0)

print("Loading recent historical sales...")

history_start_day = 1914 - HISTORY_WINDOW_DAYS
history_days = [
    f"d_{day}"
    for day in range(history_start_day, 1914)
]

use_columns = ["item_id", "store_id"] + history_days

history = pd.read_csv(
    SALES_PATH,
    usecols=use_columns
)

selected_items = forecast_summary["item_id"].unique()

history = history[
    history["item_id"].isin(selected_items)
].copy()

if history.empty:
    raise ValueError(
        "No matching historical products were found in the sales file."
    )

history_long = history.melt(
    id_vars=["item_id", "store_id"],
    value_vars=history_days,
    var_name="day",
    value_name="daily_sales"
)

history_long["daily_sales"] = pd.to_numeric(
    history_long["daily_sales"],
    errors="coerce"
).fillna(0).clip(lower=0)

demand_stats = (
    history_long.groupby(
        ["item_id", "store_id"],
        as_index=False
    )
    .agg(
        historical_avg_daily_demand=("daily_sales", "mean"),
        historical_demand_std=("daily_sales", "std"),
        days_of_history=("daily_sales", "count"),
    )
)

demand_stats["historical_demand_std"] = (
    demand_stats["historical_demand_std"].fillna(0)
)

inventory = forecast_summary.merge(
    demand_stats,
    on=["item_id", "store_id"],
    how="left"
)

inventory["historical_avg_daily_demand"] = (
    inventory["historical_avg_daily_demand"].fillna(0)
)

inventory["historical_demand_std"] = (
    inventory["historical_demand_std"].fillna(0)
)

inventory["days_of_history"] = (
    inventory["days_of_history"].fillna(0)
)

inventory["safety_stock_units"] = np.ceil(
    SERVICE_LEVEL_Z
    * inventory["historical_demand_std"]
    * np.sqrt(LEAD_TIME_DAYS)
).astype(int)

inventory["reorder_point_units"] = np.ceil(
    inventory["forecast_daily_demand"] * LEAD_TIME_DAYS
    + inventory["safety_stock_units"]
).astype(int)

inventory["target_stock_units"] = np.ceil(
    inventory["forecast_daily_demand"]
    * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS)
    + inventory["safety_stock_units"]
).astype(int)

inventory["simulated_stock_on_hand"] = np.ceil(
    inventory["historical_avg_daily_demand"]
    * REVIEW_PERIOD_DAYS
).astype(int)

inventory["recommended_order_units"] = (
    inventory["target_stock_units"]
    - inventory["simulated_stock_on_hand"]
).clip(lower=0).astype(int)

inventory["inventory_status"] = np.where(
    inventory["simulated_stock_on_hand"]
    <= inventory["reorder_point_units"],
    "Reorder",
    "Stock sufficient"
)

inventory["review_reason"] = ""

inventory.loc[
    inventory["zero_actual_demand_flag"],
    "review_reason"
] = "Zero actual demand in holdout period"

inventory.loc[
    inventory["historical_avg_daily_demand"].eq(0)
    & ~inventory["zero_actual_demand_flag"],
    "review_reason"
] = "No recent historical demand"

inventory.loc[
    inventory["days_of_history"] < HISTORY_WINDOW_DAYS,
    "review_reason"
] = "Insufficient historical observations"

review_mask = (
    inventory["zero_actual_demand_flag"]
    | inventory["historical_avg_daily_demand"].eq(0)
    | (inventory["days_of_history"] < HISTORY_WINDOW_DAYS)
)

inventory.loc[
    review_mask,
    "recommended_order_units"
] = 0

inventory.loc[
    review_mask,
    "inventory_status"
] = "Manual review required"

inventory["inventory_assumption"] = "Simulated stock; actual stock unavailable"

inventory = inventory.sort_values(
    ["inventory_status", "recommended_order_units"],
    ascending=[True, False]
)

output_path = OUTPUT_DIR / "inventory_optimization.csv"

inventory.to_csv(output_path, index=False)

review_path = OUTPUT_DIR / "inventory_review_required.csv"

inventory.loc[review_mask].to_csv(
    review_path,
    index=False
)

reorder_path = OUTPUT_DIR / "replenishment_recommendations.csv"

inventory.loc[
    (inventory["recommended_order_units"] > 0)
    & ~review_mask
].to_csv(
    reorder_path,
    index=False
)

print("\nInventory optimization summary")
print(f"Product-store combinations: {len(inventory)}")
print(f"Combinations requiring manual review: {review_mask.sum()}")
print(
    "Combinations with zero actual holdout demand: "
    f"{inventory['zero_actual_demand_flag'].sum()}"
)
print(
    "Combinations with a positive order recommendation: "
    f"{(inventory['recommended_order_units'] > 0).sum()}"
)
print(
    "Total recommended units: "
    f"{inventory['recommended_order_units'].sum():,.0f}"
)

print("\nInventory status counts")
print(inventory["inventory_status"].value_counts().to_string())

print("\nSaved files:")
print(output_path)
print(review_path)
print(reorder_path)
