
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load unique product-store-state combinations
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    usecols=["item_id", "store_id", "state_id"]
).drop_duplicates()

# --------------------------------------------------
# 2. Validate store-to-state mapping
# --------------------------------------------------

store_state_counts = df.groupby("store_id")["state_id"].nunique()

if not (store_state_counts == 1).all():
    raise ValueError("A store maps to multiple states.")

# --------------------------------------------------
# 3. Create bottom-level product-store series
# --------------------------------------------------

leaves = (
    df[["item_id", "store_id", "state_id"]]
    .drop_duplicates()
    .sort_values(["state_id", "store_id", "item_id"])
    .reset_index(drop=True)
)

leaves["leaf_id"] = (
    leaves["item_id"].astype(str)
    + "__"
    + leaves["store_id"].astype(str)
)

if leaves["leaf_id"].duplicated().any():
    raise ValueError("Duplicate product-store leaf IDs detected.")

# --------------------------------------------------
# 4. Define hierarchy memberships
# --------------------------------------------------

leaves["state_node"] = "STATE__" + leaves["state_id"]
leaves["store_node"] = "STORE__" + leaves["store_id"]
leaves["product_node"] = "PRODUCT__" + leaves["item_id"]

# All aggregate and bottom-level nodes
nodes = (
    ["TOTAL"]
    + sorted(leaves["state_node"].unique().tolist())
    + sorted(leaves["store_node"].unique().tolist())
    + sorted(leaves["product_node"].unique().tolist())
    + sorted(leaves["leaf_id"].unique().tolist())
)

# Remove duplicate nodes, preserving order
nodes = list(dict.fromkeys(nodes))

# --------------------------------------------------
# 5. Build summation matrix
# --------------------------------------------------

S = np.zeros((len(nodes), len(leaves)), dtype=np.int8)

node_index = {
    node: i for i, node in enumerate(nodes)
}

for j, row in leaves.iterrows():

    memberships = [
        "TOTAL",
        row["state_node"],
        row["store_node"],
        row["product_node"],
        row["leaf_id"]
    ]

    for node in memberships:
        S[node_index[node], j] = 1

# --------------------------------------------------
# 6. Save matrix
# --------------------------------------------------

matrix_df = pd.DataFrame(
    S,
    index=nodes,
    columns=leaves["leaf_id"]
)

matrix_path = OUTPUT_DIR / "hierarchy_summation_matrix.csv"

matrix_df.to_csv(matrix_path, index_label="node")

# --------------------------------------------------
# 7. Validate hierarchy
# --------------------------------------------------

total_row = S[node_index["TOTAL"]]

assert np.all(total_row == 1), "Total node membership is incorrect."

state_nodes = [
    node for node in nodes if node.startswith("STATE__")
]

store_nodes = [
    node for node in nodes if node.startswith("STORE__")
]

product_nodes = [
    node for node in nodes if node.startswith("PRODUCT__")
]

leaf_nodes = [
    node for node in nodes if "__" in node and node not in
    state_nodes + store_nodes + product_nodes
]

assert len(state_nodes) == leaves["state_id"].nunique()
assert len(store_nodes) == leaves["store_id"].nunique()
assert len(product_nodes) == leaves["item_id"].nunique()
assert len(leaf_nodes) == len(leaves)

# Each leaf must belong to exactly one state and one store
state_membership = S[[node_index[n] for n in state_nodes]].sum(axis=0)
store_membership = S[[node_index[n] for n in store_nodes]].sum(axis=0)

assert np.all(state_membership == 1)
assert np.all(store_membership == 1)

# Each leaf must have exactly one corresponding product node
product_membership = S[[node_index[n] for n in product_nodes]].sum(axis=0)

assert np.all(product_membership == 1)

# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\nHIERARCHY VALIDATION SUCCESSFUL")

print("Bottom-level series:", len(leaves))
print("States:", len(state_nodes))
print("Stores:", len(store_nodes))
print("Products:", len(product_nodes))
print("Total hierarchy nodes:", len(nodes))
print("Summation matrix shape:", S.shape)

print("\nMembership checks:")
print("Each leaf belongs to one state: PASSED")
print("Each leaf belongs to one store: PASSED")
print("Each leaf belongs to one product: PASSED")
print("All leaves contribute to total: PASSED")

print("\nSaved:", matrix_path)

