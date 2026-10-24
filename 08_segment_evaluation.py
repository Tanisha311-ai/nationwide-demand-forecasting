

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

forecast_path = OUTPUT_DIR / "future_28_day_forecast.csv"
df = pd.read_csv(forecast_path)

required_cols = [
    "item_id",
    "store_id",
    "state_id",
    "actual_sales",
    "predicted_sales",
    "baseline_forecast"
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


def calculate_group_metrics(data, group_cols):
    rows = []

    for group, part in data.groupby(group_cols, observed=True):
        actual = part["actual_sales"].sum()
        predicted = part["predicted_sales"].sum()
        baseline = part["baseline_forecast"].sum()

        absolute_error = (
            part["actual_sales"] - part["predicted_sales"]
        ).abs()

        baseline_error = (
            part["actual_sales"] - part["baseline_forecast"]
        ).abs()

        wape = (
            absolute_error.sum() / actual * 100
            if actual > 0
            else np.nan
        )

        baseline_wape = (
            baseline_error.sum() / actual * 100
            if actual > 0
            else np.nan
        )

        group_values = group if isinstance(group, tuple) else (group,)

        result = dict(zip(group_cols, group_values))
        result.update({
            "actual_demand": actual,
            "predicted_demand": predicted,
            "forecast_bias_units": predicted - actual,
            "MAE": absolute_error.mean(),
            "WAPE_percent": wape,
            "baseline_WAPE_percent": baseline_wape,
            "WAPE_improvement_percentage_points": (
                baseline_wape - wape
                if pd.notna(wape) and pd.notna(baseline_wape)
                else np.nan
            )
        })

        rows.append(result)

    return pd.DataFrame(rows)


store_metrics = calculate_group_metrics(
    df, ["state_id", "store_id"]
).sort_values("WAPE_percent")

store_metrics.to_csv(
    OUTPUT_DIR / "store_forecast_performance.csv",
    index=False
)

item_metrics = calculate_group_metrics(
    df, ["item_id", "store_id"]
)

zero_demand = item_metrics[item_metrics["actual_demand"] == 0].copy()

positive_demand = item_metrics[
    item_metrics["actual_demand"] > 0
].copy()

positive_demand = positive_demand.sort_values("WAPE_percent")

positive_demand.to_csv(
    OUTPUT_DIR / "item_store_forecast_performance.csv",
    index=False
)

zero_demand.to_csv(
    OUTPUT_DIR / "zero_actual_demand_combinations.csv",
    index=False
)

print("\nForecast performance by store")
print(
    store_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n10 product-store combinations with the highest WAPE")
print(
    positive_demand.nlargest(10, "WAPE_percent").to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n10 product-store combinations with the lowest WAPE")
print(
    positive_demand.nsmallest(10, "WAPE_percent").to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nProduct-store combinations with zero actual demand")
print(f"Number of combinations: {len(zero_demand)}")

if not zero_demand.empty:
    print(
        zero_demand[
            [
                "item_id",
                "store_id",
                "actual_demand",
                "predicted_demand",
                "forecast_bias_units"
            ]
        ].sort_values(
            "predicted_demand",
            ascending=False
        ).head(10).to_string(index=False)
    )

print("\nSaved reports:")
print(OUTPUT_DIR / "store_forecast_performance.csv")
print(OUTPUT_DIR / "item_store_forecast_performance.csv")
print(OUTPUT_DIR / "zero_actual_demand_combinations.csv")

