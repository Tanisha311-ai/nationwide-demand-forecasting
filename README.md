
# Nationwide Demand Forecasting and Inventory Optimization

An end-to-end retail demand forecasting and inventory optimization project using the M5 Walmart sales dataset. This project combines machine learning, time-series forecasting, forecast evaluation, inventory risk analysis, optimization, and an interactive Plotly Dash dashboard.

## Project Overview

The goal is to forecast product demand across stores and use those forecasts to support inventory and replenishment decisions.

### Key Features

- Retail demand forecasting using LightGBM, Prophet, and a pilot Temporal Fusion Transformer (TFT).
- Feature engineering using lag values, rolling averages, calendar variables, prices, and event-related features.
- Forecast evaluation using MAE, RMSE, WAPE, and RMSSE where available.
- Comparison with seasonal-naive forecasting baselines.
- Prediction intervals and empirical interval coverage analysis.
- Hierarchical aggregation and forecast reconciliation experiments.
- Inventory risk assessment and replenishment recommendations.
- OR-Tools optimization using assumed holding, ordering, and stockout costs.
- Interactive Plotly Dash dashboard for forecasts, model comparisons, inventory, and service-level scenarios.

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- LightGBM
- Prophet
- PyTorch and PyTorch Forecasting
- Darts
- Google OR-Tools
- Plotly and Dash
- Matplotlib and Seaborn

## Selected Forecasting Results

The 28-day holdout evaluation produced the following results:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seven-day seasonal baseline | 6.1914 | 10.2670 | 46.6779% |
| LightGBM recursive forecast | 5.2740 | 8.2487 | 39.7613% |

LightGBM reduced WAPE by approximately **14.82% relative to the seven-day seasonal baseline** in this evaluation.

Results depend on the selected products, stores, evaluation period, and aggregation level. The TFT experiment is a limited pilot and has not been fully tuned.

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
├── 10_prophet_forecasting.py
├── 11_compare_models.py
├── 12_historical_backtest.py
├── 13_check_mint_setup.py
├── 14_build_hierarchy.py
├── 15_reconcile_forecasts.py
├── 16_tft_forecasting.py
├── 17_compare_tft_lightgbm.py
├── 18_ortools_optimization.py
├── 19_validate_ortools.py
├── 20_inventory_sensitivity.py
├── 21_validate_service_level.py
├── app.py
├── dash_app.py
├── config.py
├── requirements.txt
└── README.md
```

## Dataset

This project uses the [M5 Forecasting Accuracy dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

Download the required CSV files and place them in `data/raw/`:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`
- `sample_submission.csv`

The raw dataset is large and is not included in this repository. Do not upload large raw data files or model binaries to GitHub unless necessary.

## Installation

Open the terminal in the project root directory.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some deep-learning and time-series libraries may have platform or version compatibility requirements.

### 3. Prepare the dataset

Place the required M5 CSV files inside `data/raw/`. Check `config.py` and the relevant scripts if your local paths differ.

## Running the Project

Run the core workflow from the project root, following the dependencies between scripts:

```powershell
python 01_preprocessing.py
python 02_features.py
python 03_forecasting.py
python 04_evaluation.py
python 07_future_validation.py
python 06_inventory.py
python 08_segment_evaluation.py
python 09_inventory_risk.py
```

Optional experiments include:

```powershell
python 10_prophet_forecasting.py
python 11_compare_models.py
python 14_build_hierarchy.py
python 15_reconcile_forecasts.py
python 16_tft_forecasting.py
python 17_compare_tft_lightgbm.py
python 18_ortools_optimization.py
python 19_validate_ortools.py
python 20_inventory_sensitivity.py
python 21_validate_service_level.py
```

Some scripts depend on outputs from earlier steps. Run only the experiments whose input files are available.

## Launching the Dashboard

The current interactive dashboard is built with Plotly Dash.

```powershell
python dash_app.py
```

Open the local address shown in the terminal, usually:

`http://127.0.0.1:8050`

Keep the terminal running while using the dashboard. Press `Ctrl + C` to stop the server.

The earlier Streamlit dashboard is retained in `app.py` and can be started with:

```powershell
python -m streamlit run app.py
```

## Dashboard Sections

- **Overview:** forecasting KPIs and project summary.
- **Demand Forecast:** comparison of actual and forecast demand.
- **Model Comparison:** comparison of available forecasting models.
- **Inventory:** replenishment recommendations and inventory risk.
- **Service Levels:** scenario-based inventory service-level analysis.

The dashboard reads generated CSV files from `data/outputs/`. Run the corresponding scripts if an expected chart or table is empty.

## Inventory Optimization Methodology

The inventory experiments use assumed lead times, review periods, holding costs, ordering costs, stockout penalties, and demand scenarios. OR-Tools is used to explore replenishment quantities under these assumptions.

**Important limitations:**

- The M5 dataset does not provide actual stock-on-hand information.
- Inventory quantities and stock levels are simulated or derived from project assumptions.
- Cost reductions are estimated model outputs, not measured real-world savings.
- Service-level results are scenario-based and do not represent verified live operational performance.

The project is intended for learning, experimentation, and portfolio demonstration rather than live purchasing decisions.

## Evaluation Metrics

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **WAPE:** Weighted Absolute Percentage Error.
- **RMSSE:** Root Mean Squared Scaled Error.
- **Prediction interval coverage:** Percentage of actual observations falling inside a forecast interval.

Compare models using the same evaluation period and aggregation level.

## Future Improvements

- Tune forecasting models and compare them on consistent holdout periods.
- Validate hierarchical reconciliation methods against unreconciled forecasts.
- Incorporate real stock levels, supplier lead times, and business-approved costs.
- Improve probabilistic forecasting and inventory policy evaluation.
- Add automated testing, experiment tracking, and