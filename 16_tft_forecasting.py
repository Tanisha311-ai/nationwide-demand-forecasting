
import os
import warnings

import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")

# Keep CPU training manageable
torch.set_num_threads(2)

TRAIN_PATH = "data/processed/train_features.csv"
VALIDATION_PATH = "data/processed/validation_features.csv"
OUTPUT_DIR = "data/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ENCODER_LENGTH = 56
MAX_PREDICTION_LENGTH = 28
MAX_EPOCHS = 3

print("Loading data...")

train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
validation = pd.read_csv(VALIDATION_PATH, parse_dates=["date"])

# Select a small pilot sample: 5 products across 4 stores
selected_items = sorted(train["item_id"].unique())[:5]
selected_stores = sorted(train["store_id"].unique())[:4]

train = train[
    train["item_id"].isin(selected_items)
    & train["store_id"].isin(selected_stores)
].copy()

validation = validation[
    validation["item_id"].isin(selected_items)
    & validation["store_id"].isin(selected_stores)
].copy()

# Use only features available at forecast time.
# Do not use lag/rolling features calculated from validation sales.
known_categoricals = ["wday", "month"]
static_categoricals = [
    "item_id",
    "store_id",
    "state_id",
    "cat_id",
    "dept_id",
]
known_reals = [
    "day_num",
    "sell_price",
    "year",
    "is_event",
    "is_weekend",
    "snap",
]

for df in [train, validation]:
    df["time_idx"] = df["day_num"].astype(int)

    for col in known_categoricals:
        df[col] = df[col].astype(str)

    for col in static_categoricals:
        df[col] = df[col].astype(str)

    for col in known_reals:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sales"] = pd.to_numeric(
    df["sales"], errors="coerce"
).astype("float32")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure feature values are usable
    df["sell_price"] = df["sell_price"].fillna(0)
    df[known_reals] = df[known_reals].fillna(0)
    df["sales"] = df["sales"].fillna(0)

train = train.sort_values(["id", "time_idx"]).reset_index(drop=True)
validation = validation.sort_values(["id", "time_idx"]).reset_index(drop=True)

print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Product-store series:", train["id"].nunique())

# Build training dataset
training_dataset = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="sales",
    group_ids=["id"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=static_categoricals,
    time_varying_known_categoricals=known_categoricals,
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=["sales"],
    target_normalizer=GroupNormalizer(
        groups=["id"],
        transformation="softplus",
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=False,
)

# Append the validation period so the model sees the 28 future dates.
# Validation sales are targets for scoring, not forecast-time input features.
combined = pd.concat(
    [train, validation],
    ignore_index=True,
).sort_values(["id", "time_idx"]).reset_index(drop=True)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    combined,
    predict=True,
    stop_randomization=True,
)

train_loader = training_dataset.to_dataloader(
    train=True,
    batch_size=64,
    num_workers=0,
)

validation_loader = validation_dataset.to_dataloader(
    train=False,
    batch_size=64,
    num_workers=0,
)

print("Building TFT model...")

tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=0.003,
    hidden_size=8,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=4,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    optimizer="adam",
    reduce_on_plateau_patience=2,
)

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="cpu",
    devices=1,
    gradient_clip_val=0.1,
    limit_train_batches=50,
    limit_val_batches=10,
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
)

print("Training TFT...")
trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=validation_loader,
)

print("Generating forecasts...")

# One prediction per series for the 28-day horizon.
# Quantile order: 10th percentile, median, 90th percentile.
predictions = tft.predict(
    validation_loader,
    mode="quantiles",
)

predictions = predictions.detach().cpu().numpy()

# The prediction dataset creates one final forecast window per series.
forecast_rows = []

series_ids = sorted(validation["id"].unique())

if predictions.shape[0] != len(series_ids):
    raise ValueError(
        "Prediction count does not match series count. "
        "Inspect the prediction dataset and series ordering."
    )

for i, series_id in enumerate(series_ids):
    series_validation = (
        validation[validation["id"] == series_id]
        .sort_values("time_idx")
        .tail(MAX_PREDICTION_LENGTH)
        .reset_index(drop=True)
    )

    if len(series_validation) != MAX_PREDICTION_LENGTH:
        raise ValueError(
            f"{series_id} does not have 28 validation dates."
        )

    for j, row in series_validation.iterrows():
        forecast_rows.append({
            "id": series_id,
            "item_id": row["item_id"],
            "store_id": row["store_id"],
            "state_id": row["state_id"],
            "date": row["date"],
            "actual_sales": row["sales"],
            "lower_bound_80": max(0, predictions[i, j, 0]),
            "predicted_sales": max(0, predictions[i, j, 1]),
            "upper_bound_80": max(0, predictions[i, j, 2]),
        })

results = pd.DataFrame(forecast_rows)

# Evaluation metrics
actual = results["actual_sales"].to_numpy()
predicted = results["predicted_sales"].to_numpy()

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
wape = (
    np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    if np.sum(np.abs(actual)) > 0
    else np.nan
)

inside = (
    (actual >= results["lower_bound_80"].to_numpy())
    & (actual <= results["upper_bound_80"].to_numpy())
)

coverage = np.mean(inside) * 100
mean_width = np.mean(
    results["upper_bound_80"] - results["lower_bound_80"]
)

results.to_csv(
    f"{OUTPUT_DIR}/tft_forecasts.csv",
    index=False,
)

metrics = pd.DataFrame([{
    "model": "TFT",
    "series_count": results["id"].nunique(),
    "forecast_days": results["date"].nunique(),
    "MAE": mae,
    "RMSE": rmse,
    "WAPE_percent": wape,
    "interval_coverage_percent": coverage,
    "mean_interval_width": mean_width,
}])

metrics.to_csv(
    f"{OUTPUT_DIR}/tft_metrics.csv",
    index=False,
)

print("\nTFT EVALUATION")
print(metrics.to_string(index=False))
print("\nSaved data/outputs/tft_forecasts.csv")
print("Saved data/outputs/tft_metrics.csv")
