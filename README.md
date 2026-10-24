
# Nationwide Demand Forecasting and Inventory Optimization

An end-to-end data science project that forecasts product demand, evaluates predictions on unseen data, and translates forecasts into inventory replenishment recommendations.

The project uses the M5 retail sales dataset, LightGBM forecasting, time-series holdout validation, inventory risk assessment, and an interactive Streamlit dashboard.

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
6. Inventory optimization and replenishment recommendations.
7. Future holdout validation using a 28-day forecast horizon.
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

These are offline evaluation results for the selected combinations, not a guarantee of production performance.

## Inventory Optimization

The inventory module estimates:

- Safety stock
- Reorder points
- Target stock levels
- Recommended replenishment quantities
- Potential stockout and overstock risks

The current prototype uses a three-day lead time, a seven-day review period, and a target service-level z-score of 1.645.

**Important:** The M5 dataset does not provide actual stock-on-hand quantities. Inventory levels in this project are simulated from historical demand. Consequently, order quantities and inventory risk classifications are illustrative recommendations, not live operational decisions.

## Dashboard

The Streamlit dashboard contains three sections:

- **Forecast Evaluation:** model metrics and forecast evaluation.
- **Store Performance:** store-level forecast performance and filtering.
- **Inventory Optimization:** inventory status, replenishment recommendations, and risk assessment.

## Dataset

This project uses the M5 retail sales forecasting dataset, which contains historical Walmart sales data alongside calendar and selling-price information.

The raw CSV files are not included in this repository because of their size and licensing considerations. Obtain the dataset from its original source and place the required files in `data/raw/`.

Required files include:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`
- `sample_submission.csv`

Dataset source: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

## Project Structure

```text
nationwide-demand-forecasting/
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── models/
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
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tanisha311-ai/nationwide-demand-forecasting.git
cd nationwide-demand-forecasting
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place the required M5 CSV files in `data/raw/`.

### 5. Run the pipeline

Run the scripts in the following order from the project root:

```powershell
python 01_preprocessing.py
python 02_features.py
python 03_forecasting.py
python 04_evaluation.py
python 05_reconciliation.py
python 06_inventory.py
python 07_future_validation.py
python 08_segment_evaluation.py
python 09_inventory_risk.py
```

Run the dashboard after generating the required outputs:

```powershell
python -m streamlit run app.py
```

Check each script's output and dependencies before running the full pipeline on a fresh installation.

## Limitations and Future Improvements

- Expand evaluation beyond the selected 50 products and across more forecasting horizons.
- Tune model hyperparameters and compare additional forecasting approaches.
- Add automated tests and data-quality checks.
- Incorporate real inventory, supplier lead-time, ordering-cost, and holding-cost data when available.
- Evaluate advanced hierarchical reconciliation methods such as MinT.
- Improve deployment, experiment tracking, and monitoring.

## Author

Developed as a data science portfolio project focused on demand forecasting, model evaluation, inventory analytics, and decision support.
