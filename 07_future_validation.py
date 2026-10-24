
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_ITEMS = 50
TRAIN_END = 1913
FORECAST_START = 1914
FORECAST_END = 1941

ID_COLS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id"
]

CATEGORICAL_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id"
]

FEATURE_COLS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "day_num", "sell_price", "wday", "month", "year",
    "is_event", "is_weekend", "snap",
    "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"
]


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    wape = (
        np.abs(actual - predicted).sum() / actual.sum() * 100
        if actual.sum() != 0
        else np.nan
    )

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WAPE_percent": wape
    }


print("Loading the M5 dataset...")

sales_path = RAW_DIR / "sales_train_evaluation.csv"
calendar_path = RAW_DIR / "calendar.csv"
prices_path = RAW_DIR / "sell_prices.csv"

for path in [sales_path, calendar_path, prices_path]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

sales = pd.read_csv(sales_path)

train_days = [f"d_{i}" for i in range(1, TRAIN_END + 1)]
future_days = [
    f"d_{i}" for i in range(FORECAST_START, FORECAST_END + 1)
]

missing_days = [
    col for col in train_days + future_days
    if col not in sales.columns
]

if missing_days:
    raise ValueError(f"Missing required sales columns: {missing_days[:5]}")

print("Selecting the top-selling products...")

item_totals = sales.groupby("item_id")[train_days].sum().sum(axis=1)
top_items = item_totals.nlargest(N_ITEMS).index

selected = sales[sales["item_id"].isin(top_items)].copy()

calendar = pd.read_csv(calendar_path)
prices = pd.read_csv(prices_path)

calendar["date"] = pd.to_datetime(calendar["date"])

calendar_cols = [
    "d", "date", "wm_yr_wk", "weekday", "wday",
    "month", "year", "event_name_1", "event_name_2",
    "snap_CA", "snap_TX", "snap_WI"
]

calendar_cols = [
    col for col in calendar_cols if col in calendar.columns
]

calendar = calendar[calendar_cols].copy()
calendar["day_num"] = calendar["d"].str.extract(r"(\d+)").astype(int)

calendar["is_event"] = (
    calendar.get("event_name_1", pd.Series(index=calendar.index, dtype=object)).notna()
    | calendar.get("event_name_2", pd.Series(index=calendar.index, dtype=object)).notna()
).astype(int)

calendar["is_weekend"] = (
    calendar["weekday"].isin(["Saturday", "Sunday"])
).astype(int)

calendar["d"] = calendar["d"].astype(str)

print("Preparing historical training features...")

historical = selected[ID_COLS + train_days].melt(
    id_vars=ID_COLS,
    value_vars=train_days,
    var_name="d",
    value_name="sales"
)

historical["day_num"] = historical["d"].str.extract(r"(\d+)").astype(int)

historical = historical.merge(
    calendar,
    on=["d", "day_num"],
    how="left"
)

historical = historical.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

snap_by_state = {
    "CA": "snap_CA",
    "TX": "snap_TX",
    "WI": "snap_WI"
}

historical["snap"] = 0

for state, snap_col in snap_by_state.items():
    if snap_col in historical.columns:
        mask = historical["state_id"] == state
        historical.loc[mask, "snap"] = historical.loc[mask, snap_col].fillna(0)

historical = historical.sort_values(
    ["item_id", "store_id", "day_num"]
).reset_index(drop=True)

series_keys = ["item_id", "store_id"]
grouped_sales = historical.groupby(series_keys)["sales"]

historical["lag_7"] = grouped_sales.shift(7)
historical["lag_28"] = grouped_sales.shift(28)

historical["rolling_mean_7"] = grouped_sales.transform(
    lambda values: values.shift(1).rolling(7).mean()
)

historical["rolling_mean_28"] = grouped_sales.transform(
    lambda values: values.shift(1).rolling(28).mean()
)

historical["sell_price"] = historical.groupby(
    series_keys
)["sell_price"].transform(
    lambda values: values.ffill().bfill()
)

historical["sell_price"] = historical["sell_price"].fillna(0)

historical = historical.dropna(
    subset=["lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"]
).copy()

X_train = historical[FEATURE_COLS].copy()
y_train = historical["sales"].astype(float)

category_levels = {}

for col in CATEGORICAL_COLS:
    X_train[col] = X_train[col].astype("category")
    category_levels[col] = X_train[col].cat.categories

print("Training the LightGBM model...")

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=2,
    verbosity=-1
)

model.fit(X_train, y_train)

model.booster_.save_model(
    str(MODEL_DIR / "lightgbm_future_holdout.txt")
)

with open(MODEL_DIR / "future_feature_columns.json", "w") as file:
    json.dump(FEATURE_COLS, file, indent=4)

print("Preparing recursive 28-day forecasts...")

metadata_cols = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id"
]

series_metadata = selected[metadata_cols].drop_duplicates(
    ["item_id", "store_id"]
).reset_index(drop=True)

history = {}
baseline_history = {}
last_prices = {}
actual_future = {}

for _, row in selected.iterrows():
    key = (row["item_id"], row["store_id"])

    values = row[train_days].astype(float).tolist()
    history[key] = values.copy()
    baseline_history[key] = values.copy()

    actual_future[key] = row[future_days].astype(float).tolist()

series_price = historical.groupby(
    ["item_id", "store_id"]
)["sell_price"].last()

for key, value in series_price.items():
    last_prices[key] = float(value)

price_lookup = {}

for row in prices.itertuples(index=False):
    price = getattr(row, "sell_price")
    if pd.notna(price):
        price_lookup[
            (str(row.store_id), str(row.item_id), int(row.wm_yr_wk))
        ] = float(price)

calendar_by_day = calendar.set_index("day_num").to_dict("index")

forecast_records = []

for day in range(FORECAST_START, FORECAST_END + 1):
    cal = calendar_by_day.get(day)

    if cal is None:
        raise ValueError(f"Calendar information missing for d_{day}")

    feature_rows = []
    row_keys = []
    row_prices = []

    for _, meta in series_metadata.iterrows():
        key = (meta["item_id"], meta["store_id"])
        past = history[key]

        week_key = (
            str(meta["store_id"]),
            str(meta["item_id"]),
            int(cal["wm_yr_wk"])
        )

        price = price_lookup.get(week_key, last_prices.get(key, 0.0))
        last_prices[key] = price

        state = meta["state_id"]
        snap_col = snap_by_state.get(state)
        snap_value = cal.get(snap_col, 0) if snap_col else 0

        feature_rows.append({
            "item_id": meta["item_id"],
            "dept_id": meta["dept_id"],
            "cat_id": meta["cat_id"],
            "store_id": meta["store_id"],
            "state_id": state,
            "day_num": day,
            "sell_price": price,
            "wday": cal["wday"],
            "month": cal["month"],
            "year": cal["year"],
            "is_event": cal["is_event"],
            "is_weekend": cal["is_weekend"],
            "snap": snap_value,
            "lag_7": past[-7],
            "lag_28": past[-28],
            "rolling_mean_7": np.mean(past[-7:]),
            "rolling_mean_28": np.mean(past[-28:])
        })

        row_keys.append(key)
        row_prices.append(price)

    X_future = pd.DataFrame(feature_rows, columns=FEATURE_COLS)

    for col in CATEGORICAL_COLS:
        X_future[col] = pd.Categorical(
            X_future[col],
            categories=category_levels[col]
        )

    predictions = np.maximum(model.predict(X_future), 0)

    for i, key in enumerate(row_keys):
        predicted = float(predictions[i])
        baseline = float(max(baseline_history[key][-7], 0))
        actual = actual_future[key][day - FORECAST_START]

        meta = series_metadata[
            (series_metadata["item_id"] == key[0])
            & (series_metadata["store_id"] == key[1])
        ].iloc[0]

        forecast_records.append({
            "id": meta["id"],
            "item_id": key[0],
            "store_id": key[1],
            "state_id": meta["state_id"],
            "d": f"d_{day}",
            "day_num": day,
            "date": cal["date"],
            "actual_sales": actual,
            "predicted_sales": predicted,
            "baseline_forecast": baseline,
            "sell_price": row_prices[i]
        })

        history[key].append(predicted)
        baseline_history[key].append(baseline)

    print(f"Completed forecast for d_{day}")

forecast_df = pd.DataFrame(forecast_records)

forecast_path = OUTPUT_DIR / "future_28_day_forecast.csv"
forecast_df.to_csv(forecast_path, index=False)

print("Calculating future-holdout metrics...")

actual = forecast_df["actual_sales"]

metrics = []

for model_name, prediction_col in [
    ("Seven-day seasonal baseline", "baseline_forecast"),
    ("LightGBM recursive forecast", "predicted_sales")
]:
    scores = calculate_metrics(actual, forecast_df[prediction_col])
    scores["model"] = model_name
    metrics.append(scores)

metrics_df = pd.DataFrame(metrics)[
    ["model", "MAE", "RMSE", "WAPE_percent"]
]

metrics_path = OUTPUT_DIR / "future_28_day_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "future_feature_importance.csv",
    index=False
)

print("\nFuture Holdout Model Comparison")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nForecast saved to: {forecast_path}")
print(f"Metrics saved to: {metrics_path}")
print(f"Feature importance saved to: {OUTPUT_DIR / 'future_feature_importance.csv'}")
print(f"Model saved to: {MODEL_DIR / 'lightgbm_future_holdout.txt'}")
