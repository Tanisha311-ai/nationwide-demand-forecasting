
import pandas as pd

file_path = "data/outputs/ortools_optimized_orders.csv"
df = pd.read_csv(file_path)

print("Total records:", len(df))
print("\nOptimization statuses:")
print(df["optimization_status"].value_counts(dropna=False))

print("\nOrder quantity comparison:")
print("Original units:",
      df["recommended_order_units"].sum())
print("Optimized units:",
      df["ortools_order_qty"].sum())
print("Net change:",
      df["order_qty_change"].sum())

print("\nLargest order quantity increases:")
print(
    df.sort_values("order_qty_change", ascending=False)[
        [
            "item_id",
            "store_id",
            "recommended_order_units",
            "ortools_order_qty",
            "order_qty_change",
            "optimization_status",
        ]
    ].head(10).to_string(index=False)
)

print("\nLargest order quantity decreases:")
print(
    df.sort_values("order_qty_change")[
        [
            "item_id",
            "store_id",
            "recommended_order_units",
            "ortools_order_qty",
            "order_qty_change",
            "optimization_status",
        ]
    ].head(10).to_string(index=False)
)

optimized = df[df["optimization_status"] == "Optimized"]

print("\nMean estimated objective cost:",
      round(optimized["expected_cost"].mean(), 2))

print("\nReminder: costs and current stock are assumed,")
print("so these results require sensitivity testing.")
