
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("LightGBM Demand Forecasting")

train_path = PROCESSED_DIR / "train_features.csv"
valid_path = PROCESSED_DIR / "validation_features.csv"

if not train_path.exists() or not valid_path.exists():
    raise FileNotFoundError(
        "Training or validation data not found. Run 02_features.py first."
    )

print("Loading training data...")
train = pd.read_csv(train_path, low_memory=False)

print("Loading validation data...")
valid = pd.read_csv(valid_path, low_memory=False)

print("Training shape:", train.shape)
print("Validation shape:", valid.shape)

categorical_cols = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

numeric_cols = [
    "day_num",
    "sell_price",
    "wday",
    "month",
    "year",
    "is_event",
    "is_weekend",
    "snap",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

feature_cols = [
    col for col in categorical_cols + numeric_cols
    if col in train.columns and col in valid.columns
]

target_col = "sales"

if not feature_cols:
    raise ValueError("No usable features found in the datasets.")

train = train.dropna(subset=[target_col]).copy()
valid = valid.dropna(subset=[target_col]).copy()

for col in categorical_cols:
    if col in feature_cols:
        categories = pd.Index(
            pd.concat(
                [train[col], valid[col]],
                ignore_index=True
            ).dropna().unique()
        )

        train[col] = pd.Categorical(
            train[col], categories=categories
        )
        valid[col] = pd.Categorical(
            valid[col], categories=categories
        )

for col in numeric_cols:
    if col in feature_cols:
        train[col] = pd.to_numeric(
            train[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        valid[col] = pd.to_numeric(
            valid[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        median_value = train[col].median()

        if pd.isna(median_value):
            median_value = 0

        train[col] = train[col].fillna(median_value)
        valid[col] = valid[col].fillna(median_value)

X_train = train[feature_cols]
y_train = train[target_col].astype("float32")

X_valid = valid[feature_cols]
y_valid = valid[target_col].astype("float32")

print("Features:", feature_cols)
print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))

print("Training LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=2,
    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=[
        col for col in categorical_cols
        if col in feature_cols
    ]
)

print("Model training completed.")

print("Generating validation predictions...")

predictions = model.predict(X_valid)
predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))

total_actual = y_valid.sum()

if total_actual > 0:
    wape = (
        np.abs(y_valid.to_numpy() - predictions).sum()
        / total_actual
    ) * 100
else:
    wape = np.nan

print("\nValidation Metrics")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

model_path = MODEL_DIR / "lightgbm_model.txt"
model.booster_.save_model(str(model_path))

with open(MODEL_DIR / "feature_columns.json", "w") as file:
    json.dump(feature_cols, file, indent=4)

result_cols = [
    col for col in [
        "id", "item_id", "store_id", "state_id",
        "d", "date", "sales"
    ]
    if col in valid.columns
]

results = valid[result_cols].copy()
results["actual_sales"] = y_valid.to_numpy()
results["predicted_sales"] = predictions
results["absolute_error"] = np.abs(
    results["actual_sales"] - results["predicted_sales"]
)

forecast_path = OUTPUT_DIR / "forecast_validation.csv"
results.to_csv(forecast_path, index=False)

metrics = pd.DataFrame([{
    "model": "LightGBM",
    "MAE": mae,
    "RMSE": rmse,
    "WAPE_percent": wape,
    "training_rows": len(X_train),
    "validation_rows": len(X_valid)
}])

metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop Features")
print(importance.head(10).to_string(index=False))

print("\nSaved model:", model_path)
print("Saved forecasts:", forecast_path)
print("Saved metrics:", OUTPUT_DIR / "model_metrics.csv")
print("Forecasting step completed successfully.")

del train, valid, X_train, X_valid, y_train, y_valid
gc.collect()