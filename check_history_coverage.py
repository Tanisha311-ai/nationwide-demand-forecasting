
import pandas as pd

path = "data/processed/train_features.csv"

df = pd.read_csv(
    path,
    usecols=["item_id", "store_id", "state_id", "date", "sales"]
)

df["date"] = pd.to_datetime(df["date"])

print("Unique dates:", df["date"].nunique())
print("Date range:", df["date"].min(), "to", df["date"].max())

coverage = (
    df.groupby(["item_id", "store_id"])["date"]
    .nunique()
)

print("Product-store combinations:", len(coverage))
print("Minimum dates per combination:", coverage.min())
print("Maximum dates per combination:", coverage.max())

print(
    "Combinations with incomplete history:",
    (coverage != df["date"].nunique()).sum()
)
