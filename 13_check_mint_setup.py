
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

train_path = BASE_DIR / "data" / "processed" / "train_features.csv"
backtest_path = BASE_DIR / "data" / "outputs" / "historical_backtest_errors.csv"

train = pd.read_csv(
    train_path,
    usecols=["item_id", "store_id", "state_id", "date", "sales"],
)

backtest = pd.read_csv(backtest_path)

train["date"] = pd.to_datetime(train["date"])
backtest["date"] = pd.to_datetime(backtest["date"])

print("TRAINING HISTORY")
print("Date range:", train["date"].min().date(), "to", train["date"].max().date())
print("Unique dates:", train["date"].nunique())
print("Product-store series:", train[["item_id", "store_id"]].drop_duplicates().shape[0])

print("\nBACKTEST")
print("Date range:", backtest["date"].min().date(), "to", backtest["date"].max().date())
print("Unique dates:", backtest["date"].nunique())
print("Bottom-level series:", backtest[["item_id", "store_id"]].drop_duplicates().shape[0])

print("\nHIERARCHY")
print("States:", train["state_id"].nunique())
print("Stores:", train["store_id"].nunique())
print("Products:", train["item_id"].nunique())

print("\nStore-to-state mapping:")
print(
    train[["store_id", "state_id"]]
    .drop_duplicates()
    .sort_values(["state_id", "store_id"])
    .to_string(index=False)
)
