
import os
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

# Input and output files
INPUT_FILE = "data/outputs/replenishment_recommendations.csv"
OUTPUT_FILE = "data/outputs/ortools_optimized_orders.csv"

# Assumptions for this optimization experiment (not actual business costs)
LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
HOLDING_COST_PER_UNIT = 1.0
STOCKOUT_PENALTY_PER_UNIT = 10.0
ORDER_COST_PER_UNIT = 0.25

PLANNING_DAYS = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS


def optimize_inventory(row):
    """Choose an order quantity by minimizing expected inventory costs."""

    current_stock = max(0, int(row["simulated_stock_on_hand"]))
    daily_demand = max(0.0, float(row["forecast_daily_demand"]))
    daily_std = max(0.0, float(row["historical_demand_std"]))

    # Keep zero-demand holdout cases for manual review.
    if bool(row["zero_actual_demand_flag"]):
        return pd.Series({
            "ortools_order_qty": 0,
            "expected_cost": np.nan,
            "optimization_status": "Manual review required"
        })

    mean_demand = daily_demand * PLANNING_DAYS
    demand_std = daily_std * np.sqrt(PLANNING_DAYS)

    # Three demand scenarios: low, expected and high.
    scenarios = [
        (max(0, int(round(mean_demand - demand_std))), 0.25),
        (max(0, int(round(mean_demand))), 0.50),
        (max(0, int(round(mean_demand + demand_std))), 0.25),
    ]

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is unavailable.")

    # Integer order quantity, capped at a documented planning limit.
    max_order = max(
        100,
        int(np.ceil(mean_demand + 3 * demand_std))
    )
    order_qty = solver.IntVar(0, max_order, "order_qty")

    expected_cost_terms = []

    for i, (demand, probability) in enumerate(scenarios):
        excess = solver.NumVar(0, solver.infinity(), f"excess_{i}")
        shortage = solver.NumVar(0, solver.infinity(), f"shortage_{i}")

        # Inventory after ordering compared with scenario demand.
        solver.Add(
            excess >= current_stock + order_qty - demand
        )
        solver.Add(
            shortage >= demand - current_stock - order_qty
        )

        scenario_cost = (
            HOLDING_COST_PER_UNIT * excess
            + STOCKOUT_PENALTY_PER_UNIT * shortage
        )
        expected_cost_terms.append(probability * scenario_cost)

    # Penalize ordering too, rather than ordering unlimited stock.
    objective = solver.Objective()
    for term in expected_cost_terms:
        objective.SetCoefficient(
            term, 1
        ) if False else None

    # Build the expected-cost objective from the scenario variables.
    # The terms above are linear expressions, so combine them explicitly.
    # Recreate the objective coefficients using the variables.
    objective.Clear()

    for i, (demand, probability) in enumerate(scenarios):
        # Retrieve scenario variables by their names.
        excess = solver.LookupVariable(f"excess_{i}")
        shortage = solver.LookupVariable(f"shortage_{i}")

        objective.SetCoefficient(
            excess, probability * HOLDING_COST_PER_UNIT
        )
        objective.SetCoefficient(
            shortage, probability * STOCKOUT_PENALTY_PER_UNIT
        )

    objective.SetCoefficient(order_qty, ORDER_COST_PER_UNIT)
    objective.SetMinimization()

    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return pd.Series({
            "ortools_order_qty": 0,
            "expected_cost": np.nan,
            "optimization_status": "Optimization failed"
        })

    return pd.Series({
        "ortools_order_qty": int(round(order_qty.solution_value())),
        "expected_cost": round(objective.Value(), 2),
        "optimization_status": "Optimized"
    })


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "item_id",
        "store_id",
        "forecast_daily_demand",
        "historical_demand_std",
        "simulated_stock_on_hand",
        "zero_actual_demand_flag",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} inventory records.")
    print("Running OR-Tools inventory optimization...")

    results = df.apply(optimize_inventory, axis=1)
    output = pd.concat([df, results], axis=1)

    output["order_qty_change"] = (
        output["ortools_order_qty"]
        - output["recommended_order_units"]
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("\nOptimization complete.")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nOptimization status:")
    print(output["optimization_status"].value_counts())

    optimized = output[
        output["optimization_status"] == "Optimized"
    ]

    if not optimized.empty:
        print("\nSummary")
        print("Optimized records:", len(optimized))
        print("Total optimized order units:",
              int(optimized["ortools_order_qty"].sum()))
        print("Original recommended units:",
              int(optimized["recommended_order_units"].sum()))
        print("Average estimated cost:",
              round(optimized["expected_cost"].mean(), 2))

    print("\nReminder: stock is simulated and costs are assumptions.")


if __name__ == "__main__":
    main()
