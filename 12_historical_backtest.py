
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading historical sales...", flush=True)

df = pd.read_csv(
    DATA_PATH,
    usecols=["item_id", "store_id", "state_id", "date", "sales"],
)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["item_id", "store_id", "date"])

# Hold out the last 28 training days for historical error estimation.
# These dates are earlier than the final validation period.
dates = sorted(df["date"].unique())
holdout_dates = dates[-28:]

# Seven-day seasonal naive forecasts:
# forecast each date using sales from exactly seven days earlier.
df["baseline_forecast"] = (
    df.groupby(["item_id", "store_id"], sort=False)["sales"]
    .shift(7)
)

backtest = df[df["date"].isin(holdout_dates)].copy()
backtest = backtest.dropna(subset=["baseline_forecast"])

backtest = backtest.rename(
    columns={
        "sales": "actual_sales",
        "baseline_forecast": "predicted_sales",
    }
)

backtest["absolute_error"] = (
    backtest["actual_sales"] - backtest["predicted_sales"]
).abs()

output_path = OUTPUT_DIR / "historical_backtest_errors.csv"
backtest.to_csv(output_path, index=False)

print("\nHistorical backtest completed.")
print("Backtest dates:", backtest["date"].nunique())
print("Date range:", backtest["date"].min(), "to", backtest["date"].max())
print("Rows:", len(backtest))
print("MAE:", round(backtest["absolute_error"].mean(), 4))
print("Saved:", output_path)
