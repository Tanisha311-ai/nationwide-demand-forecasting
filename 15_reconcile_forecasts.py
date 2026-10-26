
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_features.csv"
FORECAST_PATH = BASE_DIR / "data" / "outputs" / "forecast_validation.csv"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training history and LightGBM forecasts...", flush=True)

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["item_id", "store_id", "state_id", "date", "sales"],
)

forecast = pd.read_csv(FORECAST_PATH)

train["date"] = pd.to_datetime(train["date"])
forecast["date"] = pd.to_datetime(forecast["date"])

# Actual observations available for rolling one-day-ahead evaluation.
validation_actuals = forecast[
    ["item_id", "store_id", "state_id", "date", "actual_sales"]
].rename(columns={"actual_sales": "sales"})

history = pd.concat(
    [train, validation_actuals],
    ignore_index=True,
)

history = history.sort_values(
    ["item_id", "store_id", "date"]
)

# Define the 500 bottom-level product-store series.
leaves = (
    history[["item_id", "store_id", "state_id"]]
    .drop_duplicates()
    .sort_values(["state_id", "store_id", "item_id"])
    .reset_index(drop=True)
)

leaves["leaf_id"] = leaves["item_id"] + "__" + leaves["store_id"]

if len(leaves) != 500:
    raise ValueError(
        f"Expected 500 product-store series, found {len(leaves)}."
    )

# Define all 564 nodes in the same order as the saved matrix.
leaves["state_node"] = "STATE__" + leaves["state_id"]
leaves["store_node"] = "STORE__" + leaves["store_id"]
leaves["product_node"] = "PRODUCT__" + leaves["item_id"]

nodes = (
    ["TOTAL"]
    + sorted(leaves["state_node"].unique())
    + sorted(leaves["store_node"].unique())
    + sorted(leaves["product_node"].unique())
    + sorted(leaves["leaf_id"].unique())
)

node_index = {node: i for i, node in enumerate(nodes)}

# Build the summation matrix: each row describes a hierarchy node,
# and each column represents one product-store series.
S = np.zeros((len(nodes), len(leaves)), dtype=float)

for j, row in leaves.iterrows():
    memberships = [
        "TOTAL",
        row["state_node"],
        row["store_node"],
        row["product_node"],
        row["leaf_id"],
    ]

    for node in memberships:
        S[node_index[node], j] = 1.0

print("Summation matrix:", S.shape)

# Aggregate actual historical demand independently at each node.
history["leaf_id"] = history["item_id"] + "__" + history["store_id"]
history["state_node"] = "STATE__" + history["state_id"]
history["store_node"] = "STORE__" + history["store_id"]
history["product_node"] = "PRODUCT__" + history["item_id"]

parts = []

def add_node_aggregates(group_col):
    grouped = (
        history.groupby([group_col, "date"], as_index=False)["sales"]
        .sum()
        .rename(columns={group_col: "node", "sales": "actual_sales"})
    )
    parts.append(grouped)

add_node_aggregates("state_node")
add_node_aggregates("store_node")
add_node_aggregates("product_node")
add_node_aggregates("leaf_id")

total = (
    history.groupby("date", as_index=False)["sales"]
    .sum()
    .rename(columns={"sales": "actual_sales"})
)
total["node"] = "TOTAL"
parts.append(total[["node", "date", "actual_sales"]])

node_history = pd.concat(parts, ignore_index=True)
node_history = node_history.sort_values(["node", "date"])

# Rolling one-day-ahead aggregate baseline:
# forecast each date using actual demand from seven days earlier.
node_history["base_forecast"] = (
    node_history.groupby("node", sort=False)["actual_sales"]
    .shift(7)
)

# Keep only validation dates.
validation_dates = sorted(forecast["date"].unique())

node_history = node_history[
    node_history["date"].isin(validation_dates)
].copy()

# Use LightGBM as the bottom-level base forecast.
forecast["leaf_id"] = forecast["item_id"] + "__" + forecast["store_id"]

leaf_predictions = forecast.set_index(["date", "leaf_id"])["predicted_sales"]

# Build base forecast vector for each validation date.
# Aggregate nodes use their own 7-day seasonal forecasts;
# leaf nodes use the existing LightGBM predictions.
aggregate_lookup = node_history.set_index(["date", "node"])["base_forecast"]

reconciled_rows = []
leaf_rows = []

print("Reconciling forecasts...", flush=True)

# OLS reconciliation:
# bottom forecasts = (S' S)^(-1) S' times all-node base forecasts.
# Solve the linear system instead of explicitly inverting S' S.
G = np.linalg.solve(S.T @ S, S.T)

for date in validation_dates:
    y_hat = np.zeros(len(nodes), dtype=float)

    for node in nodes:
        if node in set(leaves["leaf_id"]):
            key = (pd.Timestamp(date), node)
            if key not in leaf_predictions.index:
                raise ValueError(f"Missing LightGBM forecast: {key}")
            value = float(leaf_predictions.loc[key])
        else:
            key = (pd.Timestamp(date), node)
            if key not in aggregate_lookup.index:
                raise ValueError(f"Missing aggregate forecast: {key}")
            value = float(aggregate_lookup.loc[key])

        if not np.isfinite(value):
            raise ValueError(f"Invalid forecast for {node} on {date}")
        y_hat[node_index[node]] = value

    # Estimate reconciled bottom-level forecasts using OLS.
    bottom_reconciled = G @ y_hat

    # Demand cannot be negative. Clip bottom-level forecasts, then
    # aggregate them again to preserve exact hierarchy consistency.
    bottom_reconciled = np.clip(bottom_reconciled, 0, None)
    coherent_all_nodes = S @ bottom_reconciled

    for j, row in leaves.iterrows():
        leaf_rows.append({
            "date": pd.Timestamp(date),
            "item_id": row["item_id"],
            "store_id": row["store_id"],
            "state_id": row["state_id"],
            "actual_sales": float(
                forecast.loc[
                    (forecast["date"] == pd.Timestamp(date))
                    & (forecast["leaf_id"] == row["leaf_id"]),
                    "actual_sales",
                ].iloc[0]
            ),
            "original_lightgbm": float(
                leaf_predictions.loc[(pd.Timestamp(date), row["leaf_id"])]
            ),
            "reconciled_forecast": float(bottom_reconciled[j]),
        })

    for i, node in enumerate(nodes):
        reconciled_rows.append({
            "date": pd.Timestamp(date),
            "node": node,
            "reconciled_forecast": float(coherent_all_nodes[i]),
        })

leaf_output = pd.DataFrame(leaf_rows)
hierarchy_output = pd.DataFrame(reconciled_rows)

leaf_path = OUTPUT_DIR / "reconciled_leaf_forecasts.csv"
hierarchy_path = OUTPUT_DIR / "reconciled_hierarchical_forecasts.csv"

leaf_output.to_csv(leaf_path, index=False)
hierarchy_output.to_csv(hierarchy_path, index=False)

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    wape = (
        np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
        if np.sum(np.abs(actual)) > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }

original_metrics = calculate_metrics(
    leaf_output["actual_sales"],
    leaf_output["original_lightgbm"],
)

reconciled_metrics = calculate_metrics(
    leaf_output["actual_sales"],
    leaf_output["reconciled_forecast"],
)

metrics = pd.DataFrame([
    {"model": "Original LightGBM", **original_metrics},
    {"model": "OLS reconciled LightGBM", **reconciled_metrics},
])

metrics_path = OUTPUT_DIR / "reconciliation_metrics.csv"
metrics.to_csv(metrics_path, index=False)

print("\nRECONCILIATION RESULTS")
print(metrics.to_string(index=False))

print("\nHierarchy consistency check")
wide = hierarchy_output.pivot(
    index="date",
    columns="node",
    values="reconciled_forecast",
)

max_error = 0.0

for date in validation_dates:
    day = leaf_output[leaf_output["date"] == pd.Timestamp(date)]

    # Check that the total equals the sum of all reconciled leaves.
    leaf_total = day["reconciled_forecast"].sum()
    hierarchy_total = wide.loc[pd.Timestamp(date), "TOTAL"]

    max_error = max(max_error, abs(leaf_total - hierarchy_total))

print("Maximum total consistency error:", round(max_error, 8))
print("Saved leaf forecasts:", leaf_path)
print("Saved all-node forecasts:", hierarchy_path)
print("Saved metrics:", metrics_path)
