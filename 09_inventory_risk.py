
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

INPUT_PATH = OUTPUT_DIR / "inventory_optimization.csv"
OUTPUT_PATH = OUTPUT_DIR / "inventory_risk_assessment.csv"

df = pd.read_csv(INPUT_PATH)

required = [
    "item_id",
    "store_id",
    "reorder_point_units",
    "target_stock_units",
    "simulated_stock_on_hand",
    "recommended_order_units",
    "inventory_status",
    "review_reason",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["stock_vs_reorder_point"] = (
    df["simulated_stock_on_hand"] - df["reorder_point_units"]
)

df["stock_vs_target"] = (
    df["simulated_stock_on_hand"] - df["target_stock_units"]
)

df["stockout_risk"] = np.where(
    df["simulated_stock_on_hand"] <= df["reorder_point_units"],
    "High",
    "Low"
)

df["overstock_risk"] = np.where(
    df["simulated_stock_on_hand"] > df["target_stock_units"] * 1.5,
    "Potential high",
    np.where(
        df["simulated_stock_on_hand"] > df["target_stock_units"],
        "Potential",
        "Low"
    )
)

df["risk_classification"] = "Normal"

df.loc[
    df["stockout_risk"] == "High",
    "risk_classification"
] = "High stockout risk"

df.loc[
    df["overstock_risk"].isin(["Potential", "Potential high"]),
    "risk_classification"
] = "Potential overstock risk"

df.loc[
    df["inventory_status"] == "Manual review required",
    "risk_classification"
] = "Manual review"

df.to_csv(OUTPUT_PATH, index=False)

print("Inventory risk assessment completed.")
print(f"Combinations assessed: {len(df)}")

print("\nRisk classification counts:")
print(df["risk_classification"].value_counts().to_string())

print("\nHighest-priority combinations:")
cols = [
    "item_id",
    "store_id",
    "simulated_stock_on_hand",
    "reorder_point_units",
    "target_stock_units",
    "recommended_order_units",
    "risk_classification",
]

print(
    df.sort_values(
        "recommended_order_units",
        ascending=False
    )[cols].head(10).to_string(index=False)
)

print(f"\nSaved: {OUTPUT_PATH}")
