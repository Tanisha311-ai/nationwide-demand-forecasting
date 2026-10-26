
import pandas as pd
from pathlib import Path

files = [
    "data/processed/train_features.csv",
    "data/processed/validation_features.csv",
]

columns_to_check = [
    "item_id",
    "store_id",
    "state_id",
    "date",
    "d",
    "sales",
]

for file in files:
    path = Path(file)
    print(f"\nFILE: {path}")

    if not path.exists():
        print("Not found")
        continue

    df = pd.read_csv(
        path,
        usecols=lambda col: col in columns_to_check
    )

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    date_col = "date" if "date" in df.columns else "d"

    if date_col in df.columns:
        print("Date range:", df[date_col].min(), "to", df[date_col].max())

    if "item_id" in df.columns:
        print("Unique products:", df["item_id"].nunique())

    if "store_id" in df.columns:
        print("Unique stores:", df["store_id"].nunique())

    if "state_id" in df.columns:
        print("Unique states:", df["state_id"].nunique())

    if "sales" in df.columns:
        print("Total sales:", df["sales"].sum())
