# Nationwide Demand Forecasting and Inventory Optimization

An end-to-end retail demand forecasting and inventory optimization project using the **M5 Walmart Sales Forecasting dataset**. The project combines machine learning, time-series forecasting, hierarchical reconciliation, inventory risk analysis, mathematical optimization, and interactive dashboards.

## Project Overview

The objective is to forecast product demand across multiple stores and use the forecasts to support data-driven inventory and replenishment decisions.

The project explores how forecasting models can help retailers anticipate demand, identify potential stockout risks, and make more informed replenishment decisions.

### Key Features

- Retail demand forecasting using LightGBM, Prophet, and an experimental Temporal Fusion Transformer (TFT).
- Feature engineering using lag features, rolling averages, calendar variables, prices, and event-related information.
- Comparison with seasonal-naive forecasting baselines.
- Forecast evaluation using MAE, RMSE, WAPE, and RMSSE.
- Prediction intervals and empirical coverage analysis.
- Hierarchical aggregation and reconciliation using Bottom-Up forecasting.
- Inventory risk assessment and replenishment recommendations.
- OR-Tools optimization using assumed ordering, holding, and stockout costs.
- Interactive dashboards for forecast evaluation, store performance, and inventory optimization.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM |
| Time-Series Forecasting | Prophet, Darts |
| Deep Learning | PyTorch, PyTorch Forecasting |
| Optimization | Google OR-Tools |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit, Plotly Dash |
| Development Tools | VS Code, Git, GitHub |

## Forecasting Results

### 28-Day Holdout Evaluation

The LightGBM recursive forecasting model was evaluated against a seven-day seasonal-naive baseline.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seven-day seasonal baseline | 6.1914 | 10.2670 | 46.6779% |
| LightGBM recursive forecast | 5.2740 | 8.2487 | 39.7613% |

LightGBM achieved approximately **14.82% relative WAPE improvement** over the seasonal baseline.

These results apply to the selected product-store subset and evaluation period, not the complete Walmart retail dataset.

### Additional Model Experiments

- **Prophet:** Evaluated at an aggregated product level. In a matched product-level comparison, LightGBM achieved approximately 20.08% WAPE compared with Prophet's 26.17%.
- **TFT:** Implemented as an experimental pilot using a limited number of series and training epochs. Its performance has not yet been fully tuned or optimized.
- **RMSSE:** The 28-day LightGBM evaluation achieved an average per-series RMSSE of approximately 0.6278. This is not the official M5 weighted RMSSE.

Results from different aggregation levels and evaluation subsets should not be compared as though they were the same experiment.

## Hierarchical Forecast Reconciliation

Hierarchical forecasting was implemented using a **Bottom-Up reconciliation approach** to ensure consistency across item-store, product, store, state, and total-level forecasts.

The hierarchy was constructed using:

- 500 bottom-level item-store series.
- 51 products.
- 10 stores.
- 3 states.
- 564 total hierarchy nodes.

A 28-day validation horizon was used to evaluate forecast consistency.

### Reconciliation Results

| Evaluation Level | WAPE |
|---|---:|
| Bottom-level series | 39.2068% |
| All hierarchy nodes | 20.3892% |

**Hierarchy validation:**

- Reconciliation method: Bottom-Up aggregation.
- Maximum hierarchy coherence error: 0.0.
- Hierarchy consistency validation: Passed.

The Bottom-Up approach preserves the original bottom-level forecasts and aggregates them to calculate forecasts for higher hierarchy levels. This ensures that store, state, product, and total-level forecasts are consistent with their underlying item-store forecasts.

Forecast accuracy at the bottom level is preserved rather than improved through reconciliation. The main benefit demonstrated in this experiment is hierarchical consistency.

Generated outputs:

- `reconciled_hierarchical_forecasts.csv`
- `reconciled_bottom_level_forecasts.csv`
- `reconciliation_metrics.csv`
- `reconciliation_validation.csv`


## Inventory Optimization

The inventory optimization component uses demand forecasts to generate replenishment recommendations and assess potential inventory risks.

Google OR-Tools is used to explore ordering quantities under assumed inventory constraints and cost parameters.

### Methodology

- Forecast-based demand estimation.
- Inventory risk classification.
- Reorder quantity recommendations.
- Lead-time and review-period assumptions.
- Holding cost and stockout penalty scenarios.
- Service-level sensitivity analysis.
- Optimization using Google OR-Tools.

### Inventory Assumptions

The prototype uses:

- Lead time: 3 days.
- Review period: 7 days.
- Target service-level z-score: 1.645.
- Simulated stock-on-hand values.

### Optimization Experiment

The OR-Tools experiment compared the initial replenishment recommendations with optimized quantities.

| Measure | Initial | Optimized |
|---|---:|---:|
| Recommended units | 29,739 | 30,372 |
| Estimated service level | 97.42% | 97.66% |
| Expected shortage units | 1,681.25 | 1,528.75 |

The optimization results are scenario-based outputs under the project's assumptions. They do not represent verified Walmart inventory savings or actual operational performance.

**Important limitations:**

- The M5 dataset does not contain actual stock-on-hand information.
- Inventory quantities are simulated or derived from project assumptions.
- Ordering, holding, and stockout costs are assumed inputs.
- Real-world savings and service levels would require actual inventory and supplier data.

## Project Structure

```text
nationwide-demand-forecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── models/
├── notebooks/
├── reports/
│
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
├── 22_calculate_rmsse.py
│
├── app.py
├── dash_app.py
├── config.py
├── requirements.txt
└── README.md
```

## Dataset

This project uses the [M5 Forecasting Accuracy dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

Download the required CSV files and place them inside `data/raw/`:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`
- `sample_submission.csv`

The project uses a selected subset of the dataset to keep experimentation manageable on limited hardware.

The raw dataset is not included in this repository because of its size.

## Installation

Open PowerShell in the project root directory.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some deep-learning and time-series libraries may require compatible Python and package versions.

### 3. Prepare the dataset

Place the required M5 CSV files inside `data/raw/`.

Check `config.py` and the individual scripts if your local file paths differ.

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

### Additional Experiments

Prophet forecasting and model comparison:

```powershell
python 10_prophet_forecasting.py
python 11_compare_models.py
python 12_historical_backtest.py
```

Hierarchical forecasting:

```powershell
python 14_build_hierarchy.py
python 15_reconcile_forecasts.py
```

TFT experiments:

```powershell
python 16_tft_forecasting.py
python 17_compare_tft_lightgbm.py
```

Inventory optimization:

```powershell
python 18_ortools_optimization.py
python 19_validate_ortools.py
python 20_inventory_sensitivity.py
python 21_validate_service_level.py
```

RMSSE evaluation:

```powershell
python 22_calculate_rmsse.py
```

Some scripts depend on outputs from earlier steps. Run them only when their required input files are available.

## Launching the Dashboard

### Streamlit dashboard

The Streamlit dashboard is available through `app.py`.

```powershell
python -m streamlit run app.py
```

### Plotly Dash application

The separate Dash application is available through `dash_app.py`.

```powershell
python dash_app.py
```

Open the local address displayed in the terminal. Keep the terminal running while using the dashboard.

Press `Ctrl + C` to stop the application.

## Dashboard Sections

The dashboard implementations provide visual access to project outputs such as:

- Forecast evaluation and model performance.
- Actual versus predicted demand.
- Store-level performance.
- Inventory optimization recommendations.
- Inventory risk and service-level scenarios.

The dashboard reads generated data from `data/outputs/`. Run the relevant forecasting or optimization scripts if an expected chart or table is empty.

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| WAPE | Weighted Absolute Percentage Error |
| RMSSE | Root Mean Squared Scaled Error |
| Prediction Interval Coverage | Percentage of actual observations falling inside the forecast interval |

Models should be compared using the same evaluation period, series, and aggregation level.

## Limitations

- The forecasting experiments use a selected subset of the M5 dataset.
- TFT is an experimental pilot and requires further tuning.
- Hierarchical reconciliation demonstrated coherence but did not materially improve forecast accuracy in the current experiment.
- Inventory levels and cost assumptions are simulated.
- The optimization results are not verified real-world savings.
- The project is intended for learning, experimentation, and portfolio demonstration rather than live purchasing decisions.

## Future Improvements

- Tune forecasting models and evaluate them across additional products and stores.
- Explore probabilistic forecasting and improved prediction intervals.
- Investigate alternative hierarchical reconciliation methods.
- Incorporate real stock levels, supplier lead times, and business-approved inventory costs.
- Improve inventory policy evaluation under changing demand conditions.
- Add automated testing, experiment tracking, and model monitoring.

## Author

**Tanisha Kapoor**

Data Science | Machine Learning | Forecasting | Data Analytics

GitHub: [Tanisha311-ai](https://github.com/Tanisha311-ai)
