
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


# Project paths
BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("LightGBM Demand Forecasting with Prediction Intervals")


# Load data
train_path = PROCESSED_DIR / "train_features.csv"
valid_path = PROCESSED_DIR / "validation_features.csv"

if not train_path.exists() or not valid_path.exists():
    raise FileNotFoundError(
        "Training or validation data not found. "
        "Run 02_features.py first."
    )

print("Loading training data...")
train = pd.read_csv(train_path, low_memory=False)

print("Loading validation data...")
valid = pd.read_csv(valid_path, low_memory=False)

print("Training shape:", train.shape)
print("Validation shape:", valid.shape)


# Define features
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
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

target_col = "sales"

required_cols = {"sales", "id", "day_num"}

missing_train = required_cols - set(train.columns)
missing_valid = required_cols - set(valid.columns)

if missing_train:
    raise ValueError(
        f"Training data is missing columns: {missing_train}"
    )

if missing_valid:
    raise ValueError(
        f"Validation data is missing columns: {missing_valid}"
    )

feature_cols = [
    col
    for col in categorical_cols + numeric_cols
    if col in train.columns and col in valid.columns
]

if not feature_cols:
    raise ValueError("No usable features found.")

train = train.dropna(subset=[target_col]).copy()
valid = valid.dropna(subset=[target_col]).copy()

train = train[np.isfinite(train[target_col])].copy()
valid = valid[np.isfinite(valid[target_col])].copy()

if train.empty or valid.empty:
    raise ValueError("Training or validation data is empty.")

if (train[target_col] < 0).any() or (valid[target_col] < 0).any():
    raise ValueError("Negative sales values found. Check the source data.")

print("Features:", feature_cols)
print("Training rows:", len(train))
print("Validation rows:", len(valid))


# Prepare categorical features
for col in categorical_cols:
    if col in feature_cols:
        categories = pd.Index(train[col].dropna().unique())

        train[col] = pd.Categorical(
            train[col],
            categories=categories
        )

        valid[col] = pd.Categorical(
            valid[col],
            categories=categories
        )


# Prepare numeric features
for col in numeric_cols:
    if col in feature_cols:
        train[col] = pd.to_numeric(
            train[col],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        valid[col] = pd.to_numeric(
            valid[col],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        median_value = train[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        train[col] = train[col].fillna(median_value)
        valid[col] = valid[col].fillna(median_value)


# Create training and validation matrices
X_train = train[feature_cols]
y_train = train[target_col].astype("float32")

X_valid = valid[feature_cols]
y_valid = valid[target_col].astype("float32")

actual_values = y_valid.to_numpy(dtype=float)

categorical_features = [
    col for col in categorical_cols
    if col in feature_cols
]


# Train the main LightGBM model
print("\nTraining LightGBM point forecast model...")

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
    categorical_feature=categorical_features
)

print("Point forecast model trained.")


# Generate point forecasts
print("\nGenerating point forecasts...")

predictions = np.maximum(
    model.predict(X_valid),
    0
)


# Train lower quantile model
print("Training lower quantile model...")

lower_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.10,
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

lower_model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features
)

print("Lower quantile model trained.")


# Train upper quantile model
print("Training upper quantile model...")

upper_model = lgb.LGBMRegressor(
    objective="quantile",
    alpha=0.90,
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

upper_model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features
)

print("Upper quantile model trained.")


# Generate prediction intervals
print("\nGenerating prediction intervals...")

lower_predictions = np.maximum(
    lower_model.predict(X_valid),
    0
)

upper_predictions = np.maximum(
    upper_model.predict(X_valid),
    0
)

# Ensure lower bound does not exceed upper bound
lower_bound = np.minimum(
    lower_predictions,
    upper_predictions
)

upper_bound = np.maximum(
    lower_predictions,
    upper_predictions
)


# Evaluate point forecasts
mae = mean_absolute_error(
    actual_values,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        predictions
    )
)

total_actual = np.abs(actual_values).sum()

if total_actual > 0:
    wape = (
        np.abs(actual_values - predictions).sum()
        / total_actual
    ) * 100
else:
    wape = np.nan

print("\nPoint Forecast Metrics")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")


# Evaluate prediction intervals
inside_interval = (
    (actual_values >= lower_bound)
    & (actual_values <= upper_bound)
)

coverage = inside_interval.mean() * 100

mean_interval_width = np.mean(
    upper_bound - lower_bound
)

print("\nPrediction Interval Metrics")
print("Nominal interval coverage: 80%")
print(f"Observed coverage: {coverage:.2f}%")
print(f"Mean interval width: {mean_interval_width:.4f}")

interval_metrics = pd.DataFrame([{
    "nominal_coverage_percent": 80.0,
    "actual_coverage_percent": coverage,
    "mean_interval_width": mean_interval_width,
    "validation_rows": len(actual_values)
}])

interval_metrics.to_csv(
    OUTPUT_DIR / "prediction_interval_metrics.csv",
    index=False
)


# Save trained models
print("\nSaving models...")

model_path = MODEL_DIR / "lightgbm_model.txt"
lower_model_path = MODEL_DIR / "lightgbm_lower_quantile.txt"
upper_model_path = MODEL_DIR / "lightgbm_upper_quantile.txt"

model.booster_.save_model(str(model_path))
lower_model.booster_.save_model(str(lower_model_path))
upper_model.booster_.save_model(str(upper_model_path))

with open(
    MODEL_DIR / "feature_columns.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(feature_cols, file, indent=4)


# Save forecast results
result_cols = [
    col for col in [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "date",
        "day_num"
    ]
    if col in valid.columns
]

results = valid[result_cols].copy()

results["actual_sales"] = actual_values
results["predicted_sales"] = predictions
results["lower_bound_80"] = lower_bound
results["upper_bound_80"] = upper_bound
results["inside_interval"] = inside_interval

results["absolute_error"] = np.abs(
    results["actual_sales"] - results["predicted_sales"]
)

forecast_path = OUTPUT_DIR / "forecast_validation.csv"

results.to_csv(
    forecast_path,
    index=False
)


# Save model metrics
metrics = pd.DataFrame([{
    "model": "LightGBM",
    "MAE": mae,
    "RMSE": rmse,
    "WAPE_percent": wape,
    "training_rows": len(X_train),
    "validation_rows": len(X_valid)
}])

metrics.to_csv(
    OUTPUT_DIR / "model_metrics.csv",
    index=False
)


# Save feature importance
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop Features")
print(importance.head(10).to_string(index=False))

print("\nSaved outputs:")
print("Forecasts:", forecast_path)
print("Model metrics:", OUTPUT_DIR / "model_metrics.csv")
print("Interval metrics:", OUTPUT_DIR / "prediction_interval_metrics.csv")
print("Feature importance:", OUTPUT_DIR / "feature_importance.csv")

print("\nSaved models:")
print(model_path)
print(lower_model_path)
print(upper_model_path)

print("\nForecasting completed successfully.")

del (
    train,
    valid,
    X_train,
    X_valid,
    y_train,
    y_valid,
    lower_model,
    upper_model
)

gc.collect()

