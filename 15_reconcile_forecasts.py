

import os
import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

FORECAST_HORIZON = 28

TRAIN_PATH = "data/processed/train_features.csv"
VALIDATION_PATH = "data/outputs/forecast_validation.csv"
HIERARCHY_PATH = "data/outputs/hierarchy_summation_matrix.csv"

OUTPUT_DIR = "data/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_wape(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(actual - predicted)) / denominator
    ) * 100


def calculate_metrics(actual, predicted, level):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "level": level,
        "MAE": np.mean(np.abs(actual - predicted)),
        "RMSE": np.sqrt(np.mean((actual - predicted) ** 2)),
        "WAPE": calculate_wape(actual, predicted)
    }


def validate_file(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL FORECAST RECONCILIATION")
print("=" * 70)

print("\nLoading project files...")

for path in [TRAIN_PATH, VALIDATION_PATH, HIERARCHY_PATH]:
    validate_file(path)


# Load only the required training columns.
train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "item_id",
        "store_id",
        "state_id"
    ]
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

summation_df = pd.read_csv(
    HIERARCHY_PATH,
    index_col=0
)

print("\nTraining data:", train_df.shape)
print("Validation data:", validation_df.shape)
print("Summation matrix:", summation_df.shape)


# ============================================================
# 4. VALIDATE INPUT COLUMNS
# ============================================================

required_train = [
    "item_id",
    "store_id",
    "state_id"
]

required_validation = [
    "item_id",
    "store_id",
    "date",
    "predicted_sales",
    "actual_sales"
]

for col in required_train:

    if col not in train_df.columns:
        raise ValueError(
            f"Missing training column: {col}"
        )


for col in required_validation:

    if col not in validation_df.columns:
        raise ValueError(
            f"Missing validation column: {col}"
        )


if summation_df.empty:
    raise ValueError("Hierarchy matrix is empty.")


# ============================================================
# 5. VALIDATE SUMMATION MATRIX
# ============================================================

print("\nValidating summation matrix...")

summation_df.index = summation_df.index.astype(str)

summation_df.columns = summation_df.columns.astype(str)

summation_df = summation_df.apply(
    pd.to_numeric,
    errors="coerce"
)

if summation_df.isna().any().any():

    raise ValueError(
        "Hierarchy matrix contains invalid values."
    )


S = summation_df.values.astype(float)

if not np.isin(S, [0, 1]).all():

    raise ValueError(
        "Summation matrix must contain only 0 and 1."
    )


all_nodes = summation_df.index.tolist()

bottom_series = summation_df.columns.tolist()

if len(set(all_nodes)) != len(all_nodes):
    raise ValueError("Duplicate hierarchy nodes found.")

if len(set(bottom_series)) != len(bottom_series):
    raise ValueError("Duplicate bottom-level series found.")


if S.shape != (len(all_nodes), len(bottom_series)):

    raise ValueError(
        "Summation matrix dimensions are inconsistent."
    )


if "TOTAL" not in all_nodes:

    raise ValueError(
        "TOTAL node is missing from the hierarchy."
    )


total_row = summation_df.loc["TOTAL"].values

if not np.all(total_row == 1):

    raise ValueError(
        "TOTAL node must include every bottom-level series."
    )


print("Bottom-level series:", len(bottom_series))
print("Hierarchy nodes:", len(all_nodes))
print("Summation matrix validation: PASSED")


# ============================================================
# 6. PREPARE SERIES METADATA
# ============================================================

print("\nPreparing series metadata...")

train_df["item_id"] = train_df["item_id"].astype(str)

train_df["store_id"] = train_df["store_id"].astype(str)

train_df["state_id"] = train_df["state_id"].astype(str)


train_df["series_key"] = (
    train_df["item_id"]
    + "__"
    + train_df["store_id"]
)


# Check that each series has only one item, store and state.

for col in ["item_id", "store_id", "state_id"]:

    counts = train_df.groupby("series_key")[col].nunique()

    if (counts > 1).any():

        raise ValueError(
            f"A series maps to multiple {col} values."
        )


metadata_df = (
    train_df[
        [
            "series_key",
            "item_id",
            "store_id",
            "state_id"
        ]
    ]
    .drop_duplicates(subset="series_key")
    .set_index("series_key")
)


missing_metadata = [
    series
    for series in bottom_series
    if series not in metadata_df.index
]

if missing_metadata:

    raise ValueError(
        "Missing series metadata: "
        f"{missing_metadata[:10]}"
    )


extra_metadata = [
    series
    for series in metadata_df.index
    if series not in bottom_series
]

if extra_metadata:

    print(
        "Warning: Training metadata contains series "
        "not present in the hierarchy:",
        len(extra_metadata)
    )


# Validate store-to-state mapping.

store_state_counts = (
    metadata_df.reset_index()
    .groupby("store_id")["state_id"]
    .nunique()
)

if (store_state_counts > 1).any():

    raise ValueError(
        "A store is mapped to multiple states."
    )


print("Metadata matching: PASSED")


# ============================================================
# 7. PREPARE VALIDATION FORECASTS
# ============================================================

print("\nPreparing validation forecasts...")

validation_df["item_id"] = (
    validation_df["item_id"].astype(str)
)

validation_df["store_id"] = (
    validation_df["store_id"].astype(str)
)


validation_df["series_key"] = (
    validation_df["item_id"]
    + "__"
    + validation_df["store_id"]
)


validation_df["date"] = pd.to_datetime(
    validation_df["date"],
    errors="coerce"
)


for col in ["predicted_sales", "actual_sales"]:

    validation_df[col] = pd.to_numeric(
        validation_df[col],
        errors="coerce"
    )


# Reject invalid records instead of silently dropping them.

if validation_df[
    ["date", "predicted_sales", "actual_sales"]
].isna().any().any():

    raise ValueError(
        "Validation data contains missing or invalid dates "
        "or sales values."
    )


# Check duplicate date-series combinations.

duplicates = validation_df.duplicated(
    subset=["date", "series_key"],
    keep=False
)

if duplicates.any():

    raise ValueError(
        "Duplicate date and series combinations found."
    )


# Check that all validation series belong to the hierarchy.

unexpected_series = set(
    validation_df["series_key"].unique()
) - set(bottom_series)

if unexpected_series:

    raise ValueError(
        "Validation data contains series not found in "
        f"the hierarchy: {list(unexpected_series)[:10]}"
    )


validation_dates = sorted(
    validation_df["date"].unique()
)


if len(validation_dates) != FORECAST_HORIZON:

    raise ValueError(
        f"Expected {FORECAST_HORIZON} validation dates, "
        f"found {len(validation_dates)}."
    )


# Check that every date has every bottom-level series.

expected_rows = (
    FORECAST_HORIZON * len(bottom_series)
)

if len(validation_df) != expected_rows:

    raise ValueError(
        f"Expected {expected_rows} validation rows, "
        f"found {len(validation_df)}."
    )


# ============================================================
# 8. CREATE BOTTOM-LEVEL FORECAST MATRICES
# ============================================================

print("\nCreating bottom-level forecast matrices...")

leaf_forecasts = validation_df.pivot(
    index="date",
    columns="series_key",
    values="predicted_sales"
)

leaf_actuals = validation_df.pivot(
    index="date",
    columns="series_key",
    values="actual_sales"
)


# Enforce the exact hierarchy ordering.

leaf_forecasts = leaf_forecasts.reindex(
    index=validation_dates,
    columns=bottom_series
)

leaf_actuals = leaf_actuals.reindex(
    index=validation_dates,
    columns=bottom_series
)


if leaf_forecasts.isna().any().any():

    raise ValueError(
        "Missing bottom-level forecast values."
    )


if leaf_actuals.isna().any().any():

    raise ValueError(
        "Missing bottom-level actual values."
    )


original_leaf_values = (
    leaf_forecasts.values.astype(float)
)

actual_values = (
    leaf_actuals.values.astype(float)
)


print("Validation dates:", len(validation_dates))

print(
    "Bottom-level forecast matrix:",
    original_leaf_values.shape
)


# ============================================================
# 9. BOTTOM-UP HIERARCHICAL RECONCILIATION
# ============================================================

print("\nPerforming bottom-up reconciliation...")

# Summation matrix:
# 564 hierarchy nodes x 500 bottom-level series
#
# Bottom-level forecasts:
# 28 dates x 500 series
#
# Aggregated hierarchy forecasts:
# 28 dates x 564 nodes

coherent_all_values = (
    original_leaf_values @ S.T
)


coherent_df = pd.DataFrame(
    coherent_all_values,
    index=pd.to_datetime(validation_dates),
    columns=all_nodes
)


print(
    "Reconciled hierarchy forecast shape:",
    coherent_all_values.shape
)

print("Bottom-up aggregation completed.")


# ============================================================
# 10. VALIDATE HIERARCHICAL COHERENCE
# ============================================================

print("\nValidating hierarchical coherence...")

# Independently reconstruct all hierarchy values.

expected_values = (
    coherent_df[bottom_series].values @ S.T
)


coherence_error = np.max(
    np.abs(
        coherent_df[all_nodes].values - expected_values
    )
)


print(
    "Maximum coherence error:",
    coherence_error
)


if coherence_error > 1e-8:

    raise ValueError(
        "Hierarchy coherence validation failed."
    )


print("Hierarchy coherence: PASSED")


# ============================================================
# 11. CALCULATE HIERARCHY ACTUAL VALUES
# ============================================================

print("\nAggregating actual demand...")

actual_all_values = (
    actual_values @ S.T
)


actual_all_df = pd.DataFrame(
    actual_all_values,
    index=pd.to_datetime(validation_dates),
    columns=all_nodes
)


print("Actual hierarchy aggregation completed.")


# ============================================================
# 12. CALCULATE FORECAST METRICS
# ============================================================

print("\nCalculating forecast metrics...")

metrics = []


# Bottom-level performance.

metrics.append(
    calculate_metrics(
        actual_values,
        original_leaf_values,
        "Bottom level"
    )
)


# All hierarchy nodes.

metrics.append(
    calculate_metrics(
        actual_all_values,
        coherent_all_values,
        "All hierarchy nodes"
    )
)


# State-level performance.

state_nodes = [
    node for node in all_nodes
    if node.startswith("STATE__")
]

if state_nodes:

    metrics.append(
        calculate_metrics(
            actual_all_df[state_nodes].values,
            coherent_df[state_nodes].values,
            "State level"
        )
    )


# Store-level performance.

store_nodes = [
    node for node in all_nodes
    if node.startswith("STORE__")
]

if store_nodes:

    metrics.append(
        calculate_metrics(
            actual_all_df[store_nodes].values,
            coherent_df[store_nodes].values,
            "Store level"
        )
    )


# Product-level performance.

product_nodes = [
    node for node in all_nodes
    if node.startswith("PRODUCT__")
]

if product_nodes:

    metrics.append(
        calculate_metrics(
            actual_all_df[product_nodes].values,
            coherent_df[product_nodes].values,
            "Product level"
        )
    )


metrics_df = pd.DataFrame(metrics)


print("\nForecast evaluation:")

print(
    metrics_df.to_string(index=False)
)


# ============================================================
# 13. SAVE HIERARCHICAL FORECASTS
# ============================================================

print("\nSaving hierarchy forecasts...")

coherent_df.index.name = "date"

coherent_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reconciled_hierarchical_forecasts.csv"
    )
)


# ============================================================
# 14. SAVE BOTTOM-LEVEL FORECASTS
# ============================================================

print("\nSaving bottom-level forecast comparison...")

bottom_output = pd.DataFrame(
    {
        "date": np.repeat(
            validation_dates,
            len(bottom_series)
        ),

        "series_key": np.tile(
            bottom_series,
            len(validation_dates)
        ),

        "actual_sales": actual_values.flatten(),

        "original_forecast": original_leaf_values.flatten(),

        "reconciled_forecast": (
            original_leaf_values.flatten()
        )
    }
)


bottom_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reconciled_bottom_level_forecasts.csv"
    ),
    index=False
)


# ============================================================
# 15. SAVE METRICS
# ============================================================

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reconciliation_metrics.csv"
    ),
    index=False
)


# ============================================================
# 16. SAVE COHERENCE VALIDATION
# ============================================================

validation_summary = pd.DataFrame(
    {
        "maximum_coherence_error": [
            coherence_error
        ],

        "bottom_level_series": [
            len(bottom_series)
        ],

        "hierarchy_nodes": [
            len(all_nodes)
        ],

        "forecast_horizon": [
            FORECAST_HORIZON
        ],

        "reconciliation_method": [
            "bottom_up"
        ],

        "hierarchy_coherent": [
            coherence_error <= 1e-8
        ]
    }
)


validation_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reconciliation_validation.csv"
    ),
    index=False
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("HIERARCHICAL RECONCILIATION COMPLETED")

print("=" * 70)

print("\nProject summary:")

print("Bottom-level series:", len(bottom_series))

print("Hierarchy nodes:", len(all_nodes))

print("Forecast horizon:", FORECAST_HORIZON)

print("Reconciliation method: Bottom-Up")

print("Maximum coherence error:", coherence_error)

print("Hierarchy coherence: PASSED")


print("\nSaved files:")

print("1. reconciled_hierarchical_forecasts.csv")

print("2. reconciled_bottom_level_forecasts.csv")

print("3. reconciliation_metrics.csv")

print("4. reconciliation_validation.csv")


print("\nAll steps finished successfully.")
