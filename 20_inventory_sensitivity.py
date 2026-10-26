
import os
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

INPUT_FILE = "data/outputs/ortools_optimized_orders.csv"
OUTPUT_FILE = "data/outputs/inventory_sensitivity_results.csv"

# Assumptions, not actual Walmart business costs
HOLDING_COST = 1.0
ORDER_COST = 0.25

# Include low and high penalties to test sensitivity properly
STOCKOUT_PENALTIES = [0.5, 1.0, 2.0, 5.0, 10.0]

LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
PLANNING_DAYS = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS


def is_zero_demand(value):
    """Handle boolean, numeric, and text values safely."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def build_scenarios(row):
    daily_demand = max(
        0.0, float(row["forecast_daily_demand"])
    )
    daily_std = max(
        0.0, float(row["historical_demand_std"])
    )

    mean_demand = daily_demand * PLANNING_DAYS
    std_demand = daily_std * np.sqrt(PLANNING_DAYS)

    scenarios = [
        (max(0, int(round(mean_demand - std_demand))), 0.25),
        (max(0, int(round(mean_demand))), 0.50),
        (max(0, int(round(mean_demand + std_demand))), 0.25),
    ]

    return scenarios


def calculate_expected_cost(row, order_qty, stockout_penalty):
    """Calculate expected cost independently using the same scenarios."""
    stock = max(0, int(row["simulated_stock_on_hand"]))
    order_qty = max(0, int(order_qty))

    expected_cost = 0.0

    for demand, probability in build_scenarios(row):
        available = stock + order_qty
        excess = max(0, available - demand)
        shortage = max(0, demand - available)

        scenario_cost = (
            HOLDING_COST * excess
            + stockout_penalty * shortage
            + ORDER_COST * order_qty
        )

        expected_cost += probability * scenario_cost

    return expected_cost


def optimize_order(row, stockout_penalty):
    """Minimize expected holding, shortage, and ordering costs."""
    stock = max(0, int(row["simulated_stock_on_hand"]))

    if is_zero_demand(row["zero_actual_demand_flag"]):
        return None, "Manual review required"

    scenarios = build_scenarios(row)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is unavailable.")

    mean_demand = max(
        0.0, float(row["forecast_daily_demand"])
    ) * PLANNING_DAYS

    std_demand = max(
        0.0, float(row["historical_demand_std"])
    ) * np.sqrt(PLANNING_DAYS)

    max_order = max(
        100,
        int(np.ceil(mean_demand + 3 * std_demand))
    )

    order_qty = solver.IntVar(0, max_order, "order_qty")
    objective = solver.Objective()

    for i, (demand, probability) in enumerate(scenarios):
        excess = solver.NumVar(
            0, solver.infinity(), f"excess_{i}"
        )
        shortage = solver.NumVar(
            0, solver.infinity(), f"shortage_{i}"
        )

        solver.Add(excess >= stock + order_qty - demand)
        solver.Add(shortage >= demand - stock - order_qty)

        objective.SetCoefficient(
            excess, probability * HOLDING_COST
        )
        objective.SetCoefficient(
            shortage, probability * stockout_penalty
        )

    # Order cost is paid once; scenario costs are probability-weighted.
    objective.SetCoefficient(order_qty, ORDER_COST)
    objective.SetMinimization()

    status = solver.Solve()

    if status not in (
        pywraplp.Solver.OPTIMAL,
        pywraplp.Solver.FEASIBLE,
    ):
        return None, "Optimization failed"

    return int(round(order_qty.solution_value())), "Optimized"


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "item_id",
        "store_id",
        "forecast_daily_demand",
        "historical_demand_std",
        "simulated_stock_on_hand",
        "zero_actual_demand_flag",
        "recommended_order_units",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    for penalty in STOCKOUT_PENALTIES:
        print(f"Testing stockout penalty: {penalty}")

        for _, row in df.iterrows():
            original_qty = max(
                0, int(row["recommended_order_units"])
            )

            original_cost = calculate_expected_cost(
                row, original_qty, penalty
            )

            if is_zero_demand(row["zero_actual_demand_flag"]):
                results.append({
                    "item_id": row["item_id"],
                    "store_id": row["store_id"],
                    "stockout_penalty": penalty,
                    "original_order_qty": original_qty,
                    "optimized_order_qty": np.nan,
                    "original_expected_cost": round(original_cost, 2),
                    "optimized_expected_cost": np.nan,
                    "cost_difference": np.nan,
                    "status": "Manual review required",
                })
                continue

            optimized_qty, status = optimize_order(row, penalty)

            if optimized_qty is None:
                optimized_cost = np.nan
                cost_difference = np.nan
            else:
                optimized_cost = calculate_expected_cost(
                    row, optimized_qty, penalty
                )
                cost_difference = optimized_cost - original_cost

            results.append({
                "item_id": row["item_id"],
                "store_id": row["store_id"],
                "stockout_penalty": penalty,
                "original_order_qty": original_qty,
                "optimized_order_qty": optimized_qty,
                "original_expected_cost": round(original_cost, 2),
                "optimized_expected_cost": (
                    round(optimized_cost, 2)
                    if pd.notna(optimized_cost) else np.nan
                ),
                "cost_difference": (
                    round(cost_difference, 2)
                    if pd.notna(cost_difference) else np.nan
                ),
                "status": status,
            })

    output = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\nSensitivity analysis complete.")
    print("Saved:", OUTPUT_FILE)

    valid = output[output["status"] == "Optimized"].copy()

    summary = valid.groupby("stockout_penalty").agg(
        combinations=("item_id", "size"),
        original_units=("original_order_qty", "sum"),
        optimized_units=("optimized_order_qty", "sum"),
        original_estimated_cost=("original_expected_cost", "sum"),
        optimized_estimated_cost=("optimized_expected_cost", "sum"),
        total_cost_difference=("cost_difference", "sum"),
    ).reset_index()

    print("\nScenario comparison:")
    print(summary.round(2).to_string(index=False))

    quantities = valid.pivot_table(
        index=["item_id", "store_id"],
        columns="stockout_penalty",
        values="optimized_order_qty",
    )

    print("\nQuantity sensitivity:")
    for low, high in zip(STOCKOUT_PENALTIES, STOCKOUT_PENALTIES[1:]):
        changed = int((quantities[low] != quantities[high]).sum())
        print(f"Penalty {low} -> {high}: {changed} combinations changed")

    print(
        "\nNegative cost difference means lower estimated cost "
        "for the optimized policy."
    )
    print(
        "All costs and stock levels are assumptions. "
        "Results are not measured real-world savings."
    )


if __name__ == "__main__":
    main()

