
from pathlib import Path
import gc
import pandas as pd
import numpy as np


# 1. PROJECT PATHS


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


N_ITEMS = 50
VALIDATION_DAYS = 28

print("=" * 60)
print("M5 FEATURE ENGINEERING")
print("=" * 60)


# 2. LOADING SALES DATA


sales_path = RAW_DIR / "sales_train_validation.csv"

if not sales_path.exists():
    sales_path = RAW_DIR / "sales_train_evaluation.csv"

if not sales_path.exists():
    raise FileNotFoundError(
        "Sales CSV not found inside data/raw/"
    )

print("\nLoading sales data...")
sales = pd.read_csv(sales_path, low_memory=False)

day_cols = [c for c in sales.columns if c.startswith("d_")]
day_cols = sorted(day_cols, key=lambda c: int(c.split("_")[1]))

if not day_cols:
    raise ValueError("No daily sales columns such as d_1 were found.")

print("Original sales shape:", sales.shape)
print("Number of daily columns:", len(day_cols))


# 3. SELECTING 50 POPULAR ITEMS

print(f"\nSelecting the top {N_ITEMS} items by total sales...")

row_totals = sales[day_cols].sum(axis=1)

item_totals = (
    pd.DataFrame({
        "item_id": sales["item_id"],
        "total_sales": row_totals
    })
    .groupby("item_id")["total_sales"]
    .sum()
)

top_items = item_totals.nlargest(N_ITEMS).index

# Keeping all available stores for these selected items
id_cols = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

sales_small = sales.loc[
    sales["item_id"].isin(top_items),
    id_cols + day_cols
].copy()

print("Selected sales shape:", sales_small.shape)
print("Selected items:", sales_small["item_id"].nunique())
print("Selected stores:", sales_small["store_id"].nunique())

del sales, row_totals, item_totals
gc.collect()



print("\nConverting sales to long format...")

df = sales_small.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales"
)

del sales_small
gc.collect()

df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
df["day_num"] = df["d"].str.replace("d_", "", regex=False).astype("int32")

print("Long-format shape:", df.shape)


# 5. MERGING CALENDAR DATA


print("\nMerging calendar data...")

calendar = pd.read_csv(RAW_DIR / "calendar.csv")

calendar["date"] = pd.to_datetime(
    calendar["date"], errors="coerce"
)

calendar_cols = [
    "d", "date", "wm_yr_wk", "weekday", "wday",
    "month", "year", "event_name_1", "event_name_2",
    "snap_CA", "snap_TX", "snap_WI"
]

calendar_cols = [
    c for c in calendar_cols if c in calendar.columns
]

df = df.merge(
    calendar[calendar_cols],
    on="d",
    how="left"
)

# Basic calendar features
df["is_event"] = (
    df[["event_name_1", "event_name_2"]]
    .notna()
    .any(axis=1)
    .astype("int8")
)

df["is_weekend"] = (
    df["weekday"].isin(["Saturday", "Sunday"])
    .astype("int8")
)

# Selecting the SNAP indicator matching each item's state
snap_map = {
    "CA": "snap_CA",
    "TX": "snap_TX",
    "WI": "snap_WI"
}

df["snap"] = 0

for state, column in snap_map.items():
    if column in df.columns:
        mask = df["state_id"].eq(state)
        df.loc[mask, "snap"] = df.loc[mask, column].fillna(0)


# 6. MERGING SELLING PRICES


print("\nMerging selling prices...")

prices = pd.read_csv(RAW_DIR / "sell_prices.csv")

df = df.merge(
    prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)


df = df.sort_values(["id", "day_num"]).reset_index(drop=True)

df["sell_price"] = (
    df.groupby("id", sort=False)["sell_price"]
    .transform(lambda x: x.ffill().bfill())
)


# 7. CREATE LAG AND ROLLING FEATURES


print("\nCreating lag and rolling-average features...")

grouped_sales = df.groupby("id", sort=False)["sales"]

df["lag_7"] = grouped_sales.shift(7)
df["lag_28"] = grouped_sales.shift(28)

df["rolling_mean_7"] = grouped_sales.transform(
    lambda x: x.shift(1).rolling(7, min_periods=7).mean()
)

df["rolling_mean_28"] = grouped_sales.transform(
    lambda x: x.shift(1).rolling(28, min_periods=28).mean()
)

# Releasing references to reduce memory use
del grouped_sales, calendar, prices
gc.collect()


# 8. CREATING TRAINING AND VALIDATION SETS


print("\nSplitting training and validation data...")

max_day = int(df["day_num"].max())
validation_start = max_day - VALIDATION_DAYS

# Removing rows without sufficient lag history
feature_cols = [
    "lag_7", "lag_28",
    "rolling_mean_7", "rolling_mean_28"
]

df = df.dropna(subset=feature_cols + ["sales"]).copy()

# Keeping useful numeric features for modeling
model_columns = [
    "id", "item_id", "dept_id", "cat_id",
    "store_id", "state_id", "d", "date",
    "day_num", "sales", "sell_price",
    "wday", "month", "year", "is_event",
    "is_weekend", "snap",
    "lag_7", "lag_28",
    "rolling_mean_7", "rolling_mean_28"
]

model_columns = [c for c in model_columns if c in df.columns]
df = df[model_columns]

train = df[df["day_num"] <= validation_start].copy()
validation = df[df["day_num"] > validation_start].copy()



print("\nSaving processed datasets...")

train_path = PROCESSED_DIR / "train_features.csv"
validation_path = PROCESSED_DIR / "validation_features.csv"

train.to_csv(train_path, index=False)
validation.to_csv(validation_path, index=False)

print("\n--- RESULTS ---")
print("Training rows:", f"{len(train):,}")
print("Validation rows:", f"{len(validation):,}")
print("Training file:", train_path)
print("Validation file:", validation_path)
print("Items:", df["item_id"].nunique())
print("Stores:", df["store_id"].nunique())

print("\nSTEP 2 COMPLETED SUCCESSFULLY!")

del df, train, validation
gc.collect()
