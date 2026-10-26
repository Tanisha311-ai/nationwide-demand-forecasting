

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, dash_table


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

app = Dash(__name__)
app.title = "Nationwide Demand Forecasting"


# =========================================================
# LOAD DATA
# =========================================================

def load_csv(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print(f"Could not load {filename}: {exc}")

    print(f"File not found: {path}")
    return pd.DataFrame()


metrics = load_csv("future_28_day_metrics.csv")
forecasts = load_csv("future_28_day_forecast.csv")
stores = load_csv("store_forecast_performance.csv")
inventory = load_csv("inventory_optimization.csv")
risk = load_csv("inventory_risk_assessment.csv")
orders = load_csv("replenishment_recommendations.csv")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_column(df, candidates):
    """Return the first matching column name."""
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def make_table(df, table_id, page_size=10):
    """Create a reusable Dash data table."""
    if df.empty:
        return html.P("No data available for this section.")

    display_df = df.copy()

    # DataTable displays null values as blank.
    display_df = display_df.where(pd.notna(display_df), None)

    return dash_table.DataTable(
        id=table_id,
        columns=[
            {"name": str(col), "id": str(col)}
            for col in display_df.columns
        ],
        data=display_df.to_dict("records"),
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={
            "overflowX": "auto",
            "minWidth": "100%"
        },
        style_cell={
            "textAlign": "left",
            "padding": "9px",
            "minWidth": "100px",
            "maxWidth": "260px",
            "whiteSpace": "normal",
            "height": "auto"
        },
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#eaf0f6"
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#f8fafc"
            }
        ]
    )


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15}
    )
    fig.update_layout(
        template="plotly_white",
        height=350,
        xaxis={"visible": False},
        yaxis={"visible": False}
    )
    return fig


def metric_value(df, model_candidates, value_candidates, keyword):
    """Find a model metric without assuming capitalization."""
    model_col = find_column(df, model_candidates)
    value_col = find_column(df, value_candidates)

    if model_col is None or value_col is None:
        return None

    matched = df[
        df[model_col].astype(str).str.contains(
            keyword, case=False, na=False
        )
    ]

    if matched.empty:
        return None

    try:
        return float(matched.iloc[0][value_col])
    except (TypeError, ValueError):
        return None


def format_percent(value):
    return f"{value:.2f}%" if value is not None else "N/A"


# =========================================================
# PREPARE DROPDOWN OPTIONS
# =========================================================

store_col = find_column(inventory, ["store_id", "store", "store_name"])
status_col = find_column(
    inventory,
    ["inventory_status", "status", "stock_status"]
)

store_options = []

if store_col:
    store_options = [
        {"label": str(s), "value": str(s)}
        for s in sorted(inventory[store_col].dropna().unique())
    ]

status_options = []

if status_col:
    status_options = [
        {"label": str(s), "value": str(s)}
        for s in sorted(inventory[status_col].dropna().unique())
    ]


# =========================================================
# OVERVIEW METRICS
# =========================================================

lightgbm_wape = metric_value(
    metrics,
    ["model", "Model"],
    ["WAPE_percent", "wape_percent", "WAPE"],
    "lightgbm"
)

baseline_wape = metric_value(
    metrics,
    ["model", "Model"],
    ["WAPE_percent", "wape_percent", "WAPE"],
    "baseline"
)

recommended_col = find_column(
    inventory,
    ["recommended_order_units", "recommended_units"]
)

recommended_total = (
    pd.to_numeric(inventory[recommended_col], errors="coerce")
    .sum()
    if recommended_col else None
)

# =========================================================
# DASHBOARD LAYOUT
# =========================================================

CARD_STYLE = {
    "padding": "18px",
    "border": "1px solid #dce3eb",
    "borderRadius": "10px",
    "backgroundColor": "white",
    "flex": "1",
    "minWidth": "180px"
}

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f4f7fb",
        "minHeight": "100vh",
        "padding": "24px"
    },
    children=[
        html.H1(
            "Nationwide Demand Forecasting & Inventory Optimization",
            style={"color": "#17365d", "marginBottom": "5px"}
        ),

        html.P(
            "M5 retail demand | 28-day holdout evaluation | "
            "Forecasting and replenishment analytics",
            style={"color": "#566573"}
        ),

        html.Div(
            style={
                "display": "flex",
                "gap": "15px",
                "flexWrap": "wrap",
                "margin": "20px 0"
            },
            children=[
                html.Div([
                    html.H4("LightGBM Holdout WAPE"),
                    html.H2(format_percent(lightgbm_wape))
                ], style=CARD_STYLE),

                html.Div([
                    html.H4("Baseline Holdout WAPE"),
                    html.H2(format_percent(baseline_wape))
                ], style=CARD_STYLE),

                html.Div([
                    html.H4("Recommended Order Units"),
                    html.H2(
                        f"{recommended_total:,.0f}"
                        if recommended_total is not None else "N/A"
                    )
                ], style=CARD_STYLE),

                html.Div([
                    html.H4("Inventory Records"),
                    html.H2(f"{len(inventory):,}")
                ], style=CARD_STYLE)
            ]
        ),

        dcc.Tabs(
            id="dashboard-tabs",
            value="forecast",
            children=[
                dcc.Tab(
                    label="Forecast Evaluation",
                    value="forecast",
                    children=[
                        html.Div(
                            style={"padding": "20px"},
                            children=[
                                html.H3("Model Evaluation Metrics"),
                                make_table(metrics, "metrics-table"),

                                html.H3(
                                    "Actual vs Predicted Demand",
                                    style={"marginTop": "25px"}
                                ),

                                html.Label("Filter by product"),

                                dcc.Dropdown(
                                    id="forecast-product",
                                    options=[
                                        {
                                            "label": str(x),
                                            "value": str(x)
                                        }
                                        for x in sorted(
                                            forecasts["item_id"]
                                            .dropna().unique()
                                        )
                                    ] if "item_id" in forecasts.columns
                                    else [],
                                    placeholder="All products",
                                    clearable=True
                                ),

                                dcc.Graph(id="forecast-chart")
                            ]
                        )
                    ]
                ),

                dcc.Tab(
                    label="Store Performance",
                    value="stores",
                    children=[
                        html.Div(
                            style={"padding": "20px"},
                            children=[
                                html.H3("Store Forecast Performance"),
                                make_table(stores, "stores-table"),

                                dcc.Graph(id="store-chart")
                            ]
                        )
                    ]
                ),

                dcc.Tab(
                    label="Inventory Optimization",
                    value="inventory",
                    children=[
                        html.Div(
                            style={"padding": "20px"},
                            children=[
                                html.H3("Inventory Filters"),

                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns":
                                            "repeat(auto-fit, minmax(240px, 1fr))",
                                        "gap": "18px"
                                    },
                                    children=[
                                        html.Div([
                                            html.Label("Select store"),
                                            dcc.Dropdown(
                                                id="inventory-store",
                                                options=store_options,
                                                placeholder="All stores",
                                                clearable=True
                                            )
                                        ]),

                                        html.Div([
                                            html.Label(
                                                "Select inventory status"
                                            ),
                                            dcc.Dropdown(
                                                id="inventory-status",
                                                options=status_options,
                                                value=[
                                                    x["value"]
                                                    for x in status_options
                                                ],
                                                multi=True,
                                                placeholder="All statuses"
                                            )
                                        ])
                                    ]
                                ),

                                html.H3(
                                    "Inventory Status",
                                    style={"marginTop": "25px"}
                                ),

                                # This chart updates from the filtered data.
                                dcc.Graph(id="inventory-status-chart"),

                                html.H3("Filtered Inventory Records"),

                                html.Div(id="inventory-record-count"),

                                html.Div(
                                    id="inventory-table-container"
                                ),

                                html.Button(
                                    "Download Filtered Inventory CSV",
                                    id="download-inventory-button",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "15px",
                                        "padding": "10px 16px",
                                        "backgroundColor": "#17365d",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "cursor": "pointer"
                                    }
                                ),

                                dcc.Download(id="download-inventory"),

                                html.Hr(
                                    style={"margin": "30px 0"}
                                ),

                                html.H3("Inventory Risk Assessment"),

                                dcc.Graph(id="risk-chart"),

                                make_table(risk, "risk-table"),

                                html.P(
                                    "Inventory levels and recommendations "
                                    "are simulated, not live stock data.",
                                    style={
                                        "color": "#8a4b08",
                                        "marginTop": "20px"
                                    }
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)


# =========================================================
# CALLBACK 1: FORECAST CHART
# =========================================================

@app.callback(
    Output("forecast-chart", "figure"),
    Input("forecast-product", "value")
)
def update_forecast_chart(selected_product):
    if forecasts.empty:
        return empty_figure("Forecast data is unavailable.")

    df = forecasts.copy()

    item_col = find_column(df, ["item_id", "product_id", "item"])

    if selected_product and item_col:
        df = df[
            df[item_col].astype(str) == str(selected_product)
        ]

    day_col = find_column(
        df, ["day", "d", "date", "day_id", "ds"]
    )
    actual_col = find_column(
        df, ["actual_sales", "actual", "actual_units"]
    )
    predicted_col = find_column(
        df, ["predicted_sales", "prediction", "predicted", "forecast"]
    )
    baseline_col = find_column(
        df, ["baseline_forecast", "baseline_prediction"]
    )

    if not day_col or not actual_col or not predicted_col:
        return empty_figure(
            "Required day, actual-sales, or predicted-sales columns "
            "were not found. Check future_28_day_forecast.csv."
        )

    columns = [actual_col, predicted_col]

    if baseline_col:
        columns.append(baseline_col)

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    chart_df = (
        df.groupby(day_col, as_index=False)[columns]
        .sum()
        .sort_values(day_col)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df[day_col],
        y=chart_df[actual_col],
        mode="lines",
        name="Actual demand"
    ))

    fig.add_trace(go.Scatter(
        x=chart_df[day_col],
        y=chart_df[predicted_col],
        mode="lines",
        name="LightGBM forecast"
    ))

    if baseline_col:
        fig.add_trace(go.Scatter(
            x=chart_df[day_col],
            y=chart_df[baseline_col],
            mode="lines",
            name="Baseline forecast"
        ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Day",
        yaxis_title="Units sold",
        hovermode="x unified",
        legend_title="Series"
    )

    return fig


# =========================================================
# CALLBACK 2: STORE PERFORMANCE CHART
# =========================================================

@app.callback(
    Output("store-chart", "figure"),
    Input("dashboard-tabs", "value")
)
def update_store_chart(active_tab):
    if stores.empty:
        return empty_figure("Store performance data is unavailable.")

    store_id_col = find_column(
        stores, ["store_id", "store", "store_name"]
    )
    wape_col = find_column(
        stores, ["WAPE_percent", "wape_percent", "WAPE"]
    )

    if not store_id_col or not wape_col:
        return empty_figure(
            "Store ID or WAPE column was not found."
        )

    df = stores.copy()
    df[wape_col] = pd.to_numeric(df[wape_col], errors="coerce")
    df = df.dropna(subset=[wape_col])

    fig = px.bar(
        df,
        x=store_id_col,
        y=wape_col,
        title="Forecast WAPE by Store",
        labels={
            store_id_col: "Store",
            wape_col: "WAPE (%)"
        }
    )

    fig.update_layout(template="plotly_white")
    return fig


# =========================================================
# CALLBACK 3: FILTERED INVENTORY, STATUS CHART AND TABLE
# =========================================================

@app.callback(
    Output("inventory-status-chart", "figure"),
    Output("inventory-record-count", "children"),
    Output("inventory-table-container", "children"),
    Input("inventory-store", "value"),
    Input("inventory-status", "value")
)
def update_inventory_dashboard(selected_store, selected_statuses):
    if inventory.empty:
        return (
            empty_figure("Inventory data is unavailable."),
            "No inventory records available.",
            html.P("Run the inventory optimization script first.")
        )

    df = inventory.copy()

    store_column = find_column(
        df, ["store_id", "store", "store_name"]
    )
    status_column = find_column(
        df, ["inventory_status", "status", "stock_status"]
    )
    order_column = find_column(
        df, ["recommended_order_units", "recommended_units"]
    )

    # Apply the store filter.
    if selected_store and store_column:
        df = df[
            df[store_column].astype(str) == str(selected_store)
        ]

    # Apply the status filter.
    # An empty selection means no statuses are selected.
    if status_column:
        if selected_statuses is not None:
            df = df[
                df[status_column].astype(str).isin(
                    [str(s) for s in selected_statuses]
                )
            ]
        else:
            df = df.iloc[0:0]

    # The chart uses df AFTER both filters.
    if df.empty:
        fig = empty_figure(
            "No inventory records match the selected filters."
        )
    elif status_column:
        status_counts = (
            df[status_column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Inventory status")
            .reset_index(name="Record count")
        )

        fig = px.bar(
            status_counts,
            x="Inventory status",
            y="Record count",
            color="Inventory status",
            text="Record count",
            title="Inventory Status for Selected Filters"
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            xaxis_title="Inventory status",
            yaxis_title="Number of records",
            yaxis_rangemode="tozero"
        )
    else:
        fig = empty_figure(
            "The inventory status column was not found."
        )

    if order_column:
        df[order_column] = pd.to_numeric(
            df[order_column], errors="coerce"
        )
        df = df.sort_values(
            order_column, ascending=False, na_position="last"
        )

    count_text = f"Matching inventory records: {len(df):,}"

    table = make_table(df, "filtered-inventory-table")

    return fig, count_text, table


# =========================================================
# CALLBACK 4: INVENTORY RISK CHART
# =========================================================

@app.callback(
    Output("risk-chart", "figure"),
    Input("dashboard-tabs", "value")
)
def update_risk_chart(active_tab):
    if risk.empty:
        return empty_figure("Inventory risk data is unavailable.")

    risk_col = find_column(
        risk,
        ["risk_classification", "risk_category", "risk_level"]
    )

    if not risk_col:
        return empty_figure(
            "A risk classification column was not found."
        )

    counts = (
        risk[risk_col]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .rename_axis("Risk classification")
        .reset_index(name="Record count")
    )

    fig = px.bar(
        counts,
        x="Risk classification",
        y="Record count",
        color="Risk classification",
        text="Record count",
        title="Inventory Risk Distribution"
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis_rangemode="tozero"
    )

    return fig


# =========================================================
# CALLBACK 5: DOWNLOAD FILTERED INVENTORY
# =========================================================

@app.callback(
    Output("download-inventory", "data"),
    Input("download-inventory-button", "n_clicks"),
    Input("inventory-store", "value"),
    Input("inventory-status", "value"),
    prevent_initial_call=True
)
def download_filtered_inventory(
    n_clicks, selected_store, selected_statuses
):
    from dash import ctx

    if not n_clicks or ctx.triggered_id != "download-inventory-button":
        return None

    if inventory.empty:
        return None

    df = inventory.copy()

    store_column = find_column(
        df, ["store_id", "store", "store_name"]
    )
    status_column = find_column(
        df, ["inventory_status", "status", "stock_status"]
    )

    if selected_store and store_column:
        df = df[
            df[store_column].astype(str) == str(selected_store)
        ]

    if status_column:
        if selected_statuses is not None:
            df = df[
                df[status_column].astype(str).isin(
                    [str(s) for s in selected_statuses]
                )
            ]
        else:
            df = df.iloc[0:0]

    return dcc.send_data_frame(
        df.to_csv,
        "filtered_inventory_recommendations.csv",
        index=False
    )


# =========================================================
# RUN DASH APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
