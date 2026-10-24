
from pathlib import Path
import pandas as pd

# 1. Defining project folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("M5 WALMART DATASET - INITIAL VALIDATION")
print("=" * 60)


required_files = [
    "calendar.csv",
    "sell_prices.csv"
]

for filename in required_files:
    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file: {file_path}\n"
            "Check that the CSV is inside data/raw/."
        )


validation_path = RAW_DIR / "sales_train_validation.csv"
evaluation_path = RAW_DIR / "sales_train_evaluation.csv"

if validation_path.exists():
    sales_path = validation_path
elif evaluation_path.exists():
    sales_path = evaluation_path
else:
    raise FileNotFoundError(
        "Could not find sales_train_validation.csv "
        "or sales_train_evaluation.csv in data/raw/."
    )

# 3. Loading datasets

print("\nLoading sales data...")
sales = pd.read_csv(sales_path, low_memory=False)

print("Loading calendar data...")
calendar = pd.read_csv(RAW_DIR / "calendar.csv")

print("Loading selling prices...")
prices = pd.read_csv(RAW_DIR / "sell_prices.csv")

# 4. Displaying dataset dimensions
print("\n--- Dataset dimensions ---")
print(f"Sales:   {sales.shape[0]:,} rows x {sales.shape[1]:,} columns")
print(f"Calendar:{calendar.shape[0]:,} rows x {calendar.shape[1]:,} columns")
print(f"Prices:  {prices.shape[0]:,} rows x {prices.shape[1]:,} columns")

# 5. Inspecting sales structure
print("\n--- Sales columns (first 10) ---")
print(sales.columns[:10].tolist())

print("\n--- Sales preview ---")
print(sales.iloc[:5, :8].to_string(index=False))

# 6. Checking required columns
sales_required = {"id", "item_id", "dept_id", "cat_id",
                  "store_id", "state_id"}
calendar_required = {"date", "d"}
prices_required = {"store_id", "item_id", "wm_yr_wk", "sell_price"}

for name, df, required in [
    ("Sales", sales, sales_required),
    ("Calendar", calendar, calendar_required),
    ("Prices", prices, prices_required)
]:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

print("\nRequired columns are present.")

# 7. Validating calendar dates
calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")

print("\n--- Calendar date range ---")
print("Start:", calendar["date"].min())
print("End:  ", calendar["date"].max())

# 8. Checking missing values
print("\n--- Missing values in key columns ---")
print("Sales identifiers missing:", sales["id"].isna().sum())
print("Calendar dates missing:", calendar["date"].isna().sum())
print("Price values missing:", prices["sell_price"].isna().sum())

print("\n--- Hierarchy ---")
print("States:", sorted(sales["state_id"].dropna().unique().tolist()))
print("Number of stores:", sales["store_id"].nunique())
print("Number of items:", sales["item_id"].nunique())
print("Number of departments:", sales["dept_id"].nunique())
print("Number of categories:", sales["cat_id"].nunique())

# 9. Save a compact dataset summary
summary = pd.DataFrame([
    {
        "dataset": "sales",
        "file": sales_path.name,
        "rows": sales.shape[0],
        "columns": sales.shape[1]
    },
    {
        "dataset": "calendar",
        "file": "calendar.csv",
        "rows": calendar.shape[0],
        "columns": calendar.shape[1]
    },
    {
        "dataset": "prices",
        "file": "sell_prices.csv",
        "rows": prices.shape[0],
        "columns": prices.shape[1]
    }
])

summary.to_csv(
    OUTPUT_DIR / "dataset_summary.csv",
    index=False
)

print("\nSaved: data/outputs/dataset_summary.csv")
print("\nSTEP 1 COMPLETED SUCCESSFULLY!")
