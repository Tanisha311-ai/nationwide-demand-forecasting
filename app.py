

from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)

st.title("Nationwide Demand Forecasting & Inventory Optimization")
st.caption(
    "M5 retail sales dataset | 28-day holdout evaluation | "
    "Inventory quantities are simulated"
)


@st.cache_data
def load_csv(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


metrics = load_csv("future_28_day_metrics.csv")
forecasts = load_csv("future_28_day_forecast.csv")
stores = load_csv("store_forecast_performance.csv")
inventory = load_csv("inventory_optimization.csv")
risk = load_csv("inventory_risk_assessment.csv")
orders = load_csv("replenishment_recommendations.csv")


if metrics.empty:
    st.error(
        "Forecast metrics not found. Run 07_future_validation.py "
        "and check data/outputs."
    )
    st.stop()


# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

st.header("Project Overview")

model_col = next(
    (c for c in ["model", "Model"] if c in metrics.columns),
    None
)

wape_col = next(
    (c for c in ["WAPE_percent", "wape_percent"] if c in metrics.columns),
    None
)

model_wape = {}

if model_col and wape_col:
    model_wape = dict(zip(metrics[model_col], metrics[wape_col]))

lightgbm_wape = next(
    (
        v for k, v in model_wape.items()
        if "lightgbm" in str(k).lower()
    ),
    None
)

baseline_wape = next(
    (
        v for k, v in model_wape.items()
        if "baseline" in str(k).lower()
    ),
    None
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Holdout forecast WAPE",
    f"{lightgbm_wape:.2f}%"
    if lightgbm_wape is not None else "N/A"
)

col2.metric(
    "Seasonal baseline WAPE",
    f"{baseline_wape:.2f}%"
    if baseline_wape is not None else "N/A"
)

if lightgbm_wape is not None and baseline_wape not in (None, 0):
    improvement = baseline_wape - lightgbm_wape
    col3.metric("WAPE improvement", f"{improvement:.2f} pp")
else:
    col3.metric("WAPE improvement", "N/A")

col4.metric(
    "Forecast rows",
    f"{len(forecasts):,}" if not forecasts.empty else "N/A"
)

st.divider()


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

forecast_tab, store_tab, inventory_tab = st.tabs(
    [
        "Forecast Evaluation",
        "Store Performance",
        "Inventory Optimization"
    ]
)


# --------------------------------------------------
# TAB 1: FORECAST EVALUATION
# --------------------------------------------------

with forecast_tab:
    st.subheader("Future Holdout Model Comparison")

    st.write(
        "The model is evaluated on the future 28-day holdout period. "
        "Lower WAPE, MAE, and RMSE generally indicate better accuracy."
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    if not forecasts.empty:

        if "item_id" in forecasts.columns:
            item_options = sorted(
                forecasts["item_id"].dropna().unique()
            )

            selected_item = st.selectbox(
                "Select a product",
                ["All products"] + item_options,
                key="forecast_product_filter"
            )

            view = forecasts.copy()

            if selected_item != "All products":
                view = view[view["item_id"] == selected_item]

        else:
            view = forecasts.copy()

        actual_col = "actual_sales"
        predicted_col = "predicted_sales"
        baseline_col = "baseline_forecast"

        day_col = next(
            (
                c for c in ["day", "d", "date", "day_id"]
                if c in view.columns
            ),
            None
        )

        if (
            day_col
            and actual_col in view.columns
            and predicted_col in view.columns
        ):
            chart_columns = [actual_col, predicted_col]

            if baseline_col in view.columns:
                chart_columns.append(baseline_col)

            chart_data = (
                view.groupby(day_col, as_index=False)[chart_columns]
                .sum()
            )

            chart_data = chart_data.set_index(day_col)

            st.subheader("Actual vs Forecast Demand")
            st.line_chart(chart_data)

        else:
            st.info(
                "A daily trend chart requires a day/date column in the "
                "forecast output. The evaluation table is still available."
            )


# --------------------------------------------------
# TAB 2: STORE PERFORMANCE
# --------------------------------------------------

with store_tab:
    st.subheader("Forecast Performance by Store")

    if stores.empty:
        st.warning(
            "Store metrics not found. Run 08_segment_evaluation.py first."
        )

    else:
        if "WAPE_percent" in stores.columns:
            max_wape = float(stores["WAPE_percent"].max())
            min_wape = float(stores["WAPE_percent"].min())

            chosen_wape = st.slider(
                "Maximum WAPE (%)",
                min_value=0.0,
                max_value=max(100.0, max_wape),
                value=max_wape,
                key="store_wape_filter"
            )

            filtered_stores = stores[
                stores["WAPE_percent"] <= chosen_wape
            ]

        else:
            filtered_stores = stores.copy()

        st.dataframe(
            filtered_stores,
            use_container_width=True,
            hide_index=True
        )

        if (
            not filtered_stores.empty
            and "store_id" in filtered_stores.columns
            and "WAPE_percent" in filtered_stores.columns
        ):
            chart = filtered_stores.set_index("store_id")[
                ["WAPE_percent"]
            ]

            st.subheader("WAPE by Store")
            st.bar_chart(chart)


# --------------------------------------------------
# TAB 3: INVENTORY OPTIMIZATION
# --------------------------------------------------

with inventory_tab:
    st.subheader("Inventory Recommendations")

    if inventory.empty:
        st.warning(
            "Inventory results not found. Run 06_inventory.py first."
        )

    else:
        # Summary metrics use the complete inventory dataset.
        col_a, col_b, col_c = st.columns(3)

        col_a.metric(
            "Product-store combinations",
            f"{len(inventory):,}"
        )

        if "recommended_order_units" in inventory.columns:
            total_order = inventory["recommended_order_units"].sum()

            col_b.metric(
                "Recommended units",
                f"{total_order:,.0f}"
            )
        else:
            col_b.metric("Recommended units", "N/A")

        if "inventory_status" in inventory.columns:
            review_count = (
                inventory["inventory_status"]
                == "Manual review required"
            ).sum()

            col_c.metric("Manual review", f"{review_count:,}")

        else:
            col_c.metric("Manual review", "N/A")

        st.divider()

        # ------------------------------------------
        # FILTERS
        # ------------------------------------------

        st.subheader("Filter Replenishment Recommendations")

        inventory_view = inventory.copy()

        filter_col1, filter_col2 = st.columns(2)

        # Store filter
        if "store_id" in inventory_view.columns:
            store_options = sorted(
                inventory_view["store_id"].dropna().unique()
            )

            with filter_col1:
                selected_store = st.selectbox(
                    "Filter by store",
                    ["All stores"] + store_options,
                    key="inventory_store_filter"
                )

            if selected_store != "All stores":
                inventory_view = inventory_view[
                    inventory_view["store_id"] == selected_store
                ]

        # Status filter
        if "inventory_status" in inventory_view.columns:
            status_options = sorted(
                inventory_view["inventory_status"]
                .dropna()
                .unique()
            )

            with filter_col2:
                selected_status = st.multiselect(
                    "Filter by inventory status",
                    status_options,
                    default=status_options,
                    key="inventory_status_filter"
                )

            inventory_view = inventory_view[
                inventory_view["inventory_status"].isin(selected_status)
            ]

        # ------------------------------------------
        # FIXED INVENTORY STATUS BAR CHART
        # Uses inventory_view AFTER applying filters.
        # ------------------------------------------

        if "inventory_status" in inventory_view.columns:
            st.subheader("Inventory Status")

            status_order = [
                "Stock sufficient",
                "Reorder",
                "Manual review required",
                "High stockout risk",
                "Potential overstock risk",
                "Normal"
            ]

            status_counts = (
                inventory_view["inventory_status"]
                .value_counts()
                .reindex(
                    [
                        status
                        for status in status_order
                        if status in inventory_view["inventory_status"].values
                    ],
                    fill_value=0
                )
            )

            # Show a helpful message when filters return no rows.
            if inventory_view.empty:
                st.info("No inventory records match the selected filters.")

            else:
                st.bar_chart(status_counts)

        # ------------------------------------------
        # FILTERED INVENTORY TABLE
        # ------------------------------------------

        if "recommended_order_units" in inventory_view.columns:
            inventory_view = inventory_view.sort_values(
                "recommended_order_units",
                ascending=False
            )

        st.subheader("Filtered Inventory Records")

        st.dataframe(
            inventory_view,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # DOWNLOAD FILTERED DATA
        # ------------------------------------------

        st.download_button(
            "Download filtered inventory CSV",
            data=inventory_view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_inventory_recommendations.csv",
            mime="text/csv",
            key="download_filtered_inventory"
        )

    # ----------------------------------------------
    # INVENTORY RISK ASSESSMENT
    # ----------------------------------------------

    if not risk.empty:
        st.divider()
        st.subheader("Inventory Risk Assessment")

        if "risk_classification" in risk.columns:
            risk_counts = risk["risk_classification"].value_counts()
            st.bar_chart(risk_counts)

        st.dataframe(
            risk,
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Important: stock-on-hand is simulated because actual inventory "
    "data is unavailable. Replenishment quantities are illustrative, "
    "not live purchase orders."
)
