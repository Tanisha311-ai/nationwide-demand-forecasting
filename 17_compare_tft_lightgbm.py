
import numpy as np
import pandas as pd

TFT_PATH = "data/outputs/tft_forecasts.csv"
LIGHTGBM_PATH = "data/outputs/forecast_validation.csv"
OUTPUT_PATH = "data/outputs/tft_lightgbm_comparison.csv"

tft = pd.read_csv(TFT_PATH, parse_dates=["date"])
lightgbm = pd.read_csv(LIGHTGBM_PATH, parse_dates=["date"])

# Select exactly the series evaluated by TFT
tft_series = set(tft["id"].unique())
lightgbm = lightgbm[lightgbm["id"].isin(tft_series)].copy()

# Check that the comparison uses matching series and dates
tft_keys = set(zip(tft["id"], tft["date"]))
lgbm_keys = set(zip(lightgbm["id"], lightgbm["date"]))

if tft_keys != lgbm_keys:
    raise ValueError(
        f"Mismatch in series-date rows: "
        f"TFT={len(tft_keys)}, LightGBM={len(lgbm_keys)}"
    )

# Merge on series and date to prevent misaligned comparisons
comparison = tft[
    ["id", "date", "actual_sales", "predicted_sales"]
].rename(columns={
    "predicted_sales": "tft_prediction",
    "actual_sales": "tft_actual",
}).merge(
    lightgbm[["id", "date", "actual_sales", "predicted_sales"]].rename(
        columns={
            "predicted_sales": "lightgbm_prediction",
            "actual_sales": "lightgbm_actual",
        }
    ),
    on=["id", "date"],
    how="inner",
    validate="one_to_one",
)

if not np.allclose(
    comparison["tft_actual"],
    comparison["lightgbm_actual"],
):
    raise ValueError("Actual sales differ between the two files.")

actual = comparison["tft_actual"].to_numpy()
total_actual = np.sum(np.abs(actual))

def calculate_metrics(name, predictions):
    errors = actual - predictions
    return {
        "model": name,
        "series": comparison["id"].nunique(),
        "forecast_days": comparison["date"].nunique(),
        "rows": len(comparison),
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "WAPE_percent": (
            np.sum(np.abs(errors)) / total_actual * 100
            if total_actual > 0 else np.nan
        ),
    }

metrics = pd.DataFrame([
    calculate_metrics(
        "TFT",
        comparison["tft_prediction"].to_numpy(),
    ),
    calculate_metrics(
        "LightGBM",
        comparison["lightgbm_prediction"].to_numpy(),
    ),
])

metrics.to_csv(OUTPUT_PATH, index=False)

print("\nFAIR MODEL COMPARISON")
print("Same product-store series:", comparison["id"].nunique())
print("Same validation dates:", comparison["date"].nunique())
print("Matched rows:", len(comparison))
print(metrics.to_string(index=False))
print("\nSaved:", OUTPUT_PATH)
