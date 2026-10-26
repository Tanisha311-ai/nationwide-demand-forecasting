
import pandas as pd

path = "data/outputs/historical_backtest_errors.csv"
df = pd.read_csv(path)

print("Rows:", len(df))
print("Dates:", df["date"].nunique())
print("Products:", df["item_id"].nunique())
print("Stores:", df["store_id"].nunique())
print("States:", df["state_id"].nunique())

print("\nStores per state:")
print(
    df[["state_id", "store_id"]]
    .drop_duplicates()
    .groupby("state_id")["store_id"]
    .nunique()
)

print("\nProducts per store:")
print(
    df[["store_id", "item_id"]]
    .drop_duplicates()
    .groupby("store_id")["item_id"]
    .nunique()
)

print("\nSales totals by state:")
print(
    df.groupby("state_id")["actual_sales"]
    .sum()
    .sort_values(ascending=False)
)
