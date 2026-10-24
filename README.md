
# Nationwide Demand Forecasting and Inventory Optimization

An end-to-end data science project that forecasts retail product demand, evaluates predictions on unseen data, and translates forecasts into inventory replenishment recommendations.

This project uses the M5 retail sales dataset, LightGBM forecasting, time-series holdout validation, inventory risk assessment, and an interactive Streamlit dashboard.

## Project Objectives

- Forecast daily demand for selected retail products across stores.
- Compare machine learning forecasts against a seven-day seasonal baseline.
- Evaluate model performance using MAE, RMSE, and WAPE.
- Analyze forecasting performance across stores and product-store combinations.
- Estimate safety stock, reorder points, target stock levels, and replenishment quantities.
- Identify potential stockout and overstock risks.
- Present results through an interactive dashboard.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM
- Matplotlib and Seaborn
- Streamlit
- Git and GitHub

## Project Workflow

1. Data preprocessing and dataset summary.
2. Feature engineering, including lag features and rolling averages.
3. LightGBM demand forecasting.
4. Model evaluation against a seasonal baseline.
5. Bottom-up aggregation of forecasts to higher levels.
6. Future holdout validation using a 28-day forecast horizon.
7. Inventory optimization and replenishment recommendations.
8. Store and product-store segment evaluation.
9. Inventory risk assessment.
10. Interactive dashboard for reviewing results.

## Model Evaluation

The model was evaluated on a future 28-day holdout period for the selected product-store combinations.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seven-day seasonal baseline | 6.1914 | 10.2670 | 46.6779% |
| LightGBM recursive forecast | 5.2740 | 8.2487 | 39.7613% |

LightGBM reduced WAPE by approximately 14.82% relative to the seasonal baseline on this evaluation set.

These results represent offline evaluation on the selected combinations and do not guarantee production performance.

### Evaluation Metrics

- **MAE:** Measures the average absolute difference between actual and predicted demand.
- **RMSE:** Penalizes larger prediction errors more heavily.
- **WAPE:** Measures total absolute forecast error relative to total actual demand.

## Inventory Optimization

The inventory module estimates:

- Safety stock
- Reorder points
- Target stock levels
- Recommended replenishment quantities
- Potential stockout and overstock risks

The current prototype uses the following assumptions:

- Lead time: 3 days
- Review period: 7 days
- Historical demand window: 56 days
- Service-level z-score: 1.645

**Important limitation:** The M5 dataset does not provide actual stock-on-hand quantities. Inventory levels in this project are simulated from historical demand. Therefore, replenishment quantities and inventory risk classifications are illustrative recommendations, not live operational decisions.

## Interactive Dashboard

The Streamlit dashboard contains three sections:

### 1. Forecast Evaluation

- Displays model evaluation metrics.
- Compares LightGBM performance against the seasonal baseline.
- Supports review of forecast performance.

### 2. Store Performance

- Displays store-level forecast performance.
- Allows filtering using WAPE.
- Helps identify stores where forecasting errors require further investigation.

### 3. Inventory Optimization

- Displays inventory status and recommendations.
- Supports filtering by store and inventory status.
- Shows inventory risk assessments.
- Provides CSV downloads for further analysis.

## Dataset

This project uses the M5 retail sales forecasting dataset, which contains historical Walmart sales data alongside calendar and selling-price information.

The raw dataset files are not included in this repository because of their size.

Download the dataset from:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Place the required files in the `data/raw/` directory.

Required files:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`
- `sample_submission.csv`

## Project Structure

```text
nationwide-demand-forecasting/
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── models/
├── notebooks/
├── reports/
├── 01_preprocessing.py
├── 02_features.py
├── 03_forecasting.py
├── 04_evaluation.py
├── 05_reconciliation.py
├── 06_inventory.py
├── 07_future_validation.py
├── 08_segment_evaluation.py
├── 09_inventory_risk.py
├── app.py
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tanisha311-ai/nationwide-demand-forecasting.git
cd nationwide-demand-forecasting
```

### 2. Create a virtual environment

On Windows, run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python executable directly instead.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Download the required M5 CSV files and place them in `data/raw/`.

Ensure that the directory and filenames match the paths expected by the scripts.

### 5. Run the pipeline

Run these commands from the project root in the following order:

```powershell
python 01_preprocessing.py
python 02_features.py
python 03_forecasting.py
python 04_evaluation.py
python 05_reconciliation.py
python 07_future_validation.py
python 06_inventory.py
python 08_segment_evaluation.py
python 09_inventory_risk.py
```

Each script depends on files generated by earlier steps. If a script fails, inspect the error and verify that its required input files exist.

### 6. Launch the dashboard

After the required output files have been generated, run:

```powershell
python -m streamlit run app.py
```

The dashboard will open in your browser.

## Limitations and Future Improvements

- Expand evaluation to more products, stores, and forecast horizons.
- Tune LightGBM hyperparameters and compare additional forecasting models.
- Add automated tests and data-quality checks.
- Incorporate real inventory, supplier lead-time, ordering-cost, and holding-cost data when available.
- Evaluate advanced hierarchical forecast reconciliation methods such as MinT.
- Improve deployment, experiment tracking, and model monitoring.
- Explore probabilistic forecasts and service-level optimization.

## Author

Developed as a data science portfolio project focused on demand forecasting, model evaluation, inventory analytics, and decision support.

