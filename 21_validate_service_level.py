
import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
INPUT_FILE = "data/outputs/ortools_optimized_orders.csv"
OUTPUT_DIR = "data/outputs"

LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
PLANNING_DAYS = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS

# Demand scenario probabilities
SCENARIOS = {
    "Low": 0.25,
    "Mean": 0.50,
    "High": 0.25,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

required_columns = [
    "item_id",
    "store_id",
    "historical_avg_daily_demand",
    "historical_demand_std",
    "simulated_stock_on_hand",
    "recommended_order_units",
    "ortools_order_qty",
    "zero_actual_demand_flag",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep combinations with usable demand and stock data
df = df.dropna(subset=[
    "historical_avg_daily_demand",
    "historical_demand_std",
    "simulated_stock_on_hand",
    "recommended_order_units",
    "ortools_order_qty",
]).copy()

# Negative demand estimates are not allowed
df["historical_avg_daily_demand"] = (
    df["historical_avg_daily_demand"].clip(lower=0)
)
df["historical_demand_std"] = df["historical_demand_std"].clip(lower=0)

# Exclude zero-demand combinations for the main service-level analysis
# because they were designated for manual review.
df = df[~df["zero_actual_demand_flag"].astype(bool)].copy()

print(f"Eligible item-store combinations: {len(df)}")
print(f"Planning horizon: {PLANNING_DAYS} days")


# --------------------------------------------------
# 3. BUILD DEMAND SCENARIOS
# --------------------------------------------------
mean_demand = (
    df["historical_avg_daily_demand"] * PLANNING_DAYS
)

demand_std = (
    df["historical_demand_std"] * np.sqrt(PLANNING_DAYS)
)

df["low_demand"] = np.maximum(
    0, np.round(mean_demand - demand_std)
)

df["mean_demand"] = np.maximum(
    0, np.round(mean_demand)
)

df["high_demand"] = np.maximum(
    0, np.round(mean_demand + demand_std)
)


# --------------------------------------------------
# 4. EVALUATE EACH INVENTORY POLICY
# --------------------------------------------------
results = []

policies = {
    "Original": "recommended_order_units",
    "OR-Tools Optimized": "ortools_order_qty",
}

for scenario_name, probability in SCENARIOS.items():

    demand_column = f"{scenario_name.lower()}_demand"
    demand = df[demand_column].to_numpy(dtype=float)
    stock = df["simulated_stock_on_hand"].to_numpy(dtype=float)

    for policy_name, order_column in policies.items():

        order = df[order_column].to_numpy(dtype=float)

        # Available units = stock on hand + incoming order
        available = stock + order

        # Units that can be supplied
        units_fulfilled = np.minimum(demand, available)

        # Unmet demand
        shortage = np.maximum(demand - available, 0)

        # Excess stock remaining at the end of the horizon
        ending_excess = np.maximum(available - demand, 0)

        # Cycle service indicator:
        # 1 when all demand is fulfilled; otherwise 0
        cycle_service = (shortage == 0).astype(int)

        total_demand = demand.sum()
        total_fulfilled = units_fulfilled.sum()

        results.append({
            "scenario": scenario_name,
            "scenario_probability": probability,
            "policy": policy_name,
            "combinations": len(df),
            "total_demand_units": round(total_demand, 2),
            "total_available_units": round(available.sum(), 2),
            "fulfilled_units": round(total_fulfilled, 2),
            "shortage_units": round(shortage.sum(), 2),
            "ending_excess_units": round(ending_excess.sum(), 2),
            "cycle_service_level_percent": round(
                cycle_service.mean() * 100, 2
            ),
            "unit_fill_rate_percent": round(
                total_fulfilled / total_demand * 100
                if total_demand > 0 else 100,
                2,
            ),
            "combinations_with_shortage": int(
                (shortage > 0).sum()
            ),
            "combinations_fully_served": int(
                (shortage == 0).sum()
            ),
        })


scenario_results = pd.DataFrame(results)


# --------------------------------------------------
# 5. PROBABILITY-WEIGHTED RESULTS
# --------------------------------------------------
weighted_results = []

for policy_name in policies:

    policy_data = scenario_results[
        scenario_results["policy"] == policy_name
    ].copy()

    # Weighted expected units across scenarios
    weighted_demand = (
        policy_data["total_demand_units"]
        * policy_data["scenario_probability"]
    ).sum()

    weighted_fulfilled = (
        policy_data["fulfilled_units"]
        * policy_data["scenario_probability"]
    ).sum()

    weighted_shortage = (
        policy_data["shortage_units"]
        * policy_data["scenario_probability"]
    ).sum()

    weighted_excess = (
        policy_data["ending_excess_units"]
        * policy_data["scenario_probability"]
    ).sum()

    # Weighted cycle service: expected share of combinations
    # fully served within each scenario
    weighted_cycle_service = (
        policy_data["cycle_service_level_percent"]
        * policy_data["scenario_probability"]
    ).sum()

    weighted_results.append({
        "policy": policy_name,
        "expected_demand_units": round(weighted_demand, 2),
        "expected_fulfilled_units": round(weighted_fulfilled, 2),
        "expected_shortage_units": round(weighted_shortage, 2),
        "expected_ending_excess_units": round(weighted_excess, 2),
        "weighted_cycle_service_level_percent": round(
            weighted_cycle_service, 2
        ),
        "weighted_unit_fill_rate_percent": round(
            weighted_fulfilled / weighted_demand * 100
            if weighted_demand > 0 else 100,
            2,
        ),
    })

weighted_summary = pd.DataFrame(weighted_results)


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------
scenario_path = os.path.join(
    OUTPUT_DIR, "inventory_service_level_by_scenario.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR, "inventory_service_level_summary.csv"
)

scenario_results.to_csv(scenario_path, index=False)
weighted_summary.to_csv(summary_path, index=False)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------
print("\nSERVICE LEVEL BY SCENARIO")
print(
    scenario_results[
        [
            "scenario",
            "policy",
            "total_demand_units",
            "shortage_units",
            "cycle_service_level_percent",
            "unit_fill_rate_percent",
            "combinations_with_shortage",
        ]
    ].to_string(index=False)
)

print("\nPROBABILITY-WEIGHTED SUMMARY")
print(weighted_summary.to_string(index=False))

print("\nSaved:")
print(scenario_path)
print(summary_path)
