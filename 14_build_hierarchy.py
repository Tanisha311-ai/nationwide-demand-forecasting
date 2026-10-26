
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    DATA_PATH,
    usecols=["item_id", "store_id", "state_id"],
).drop_duplicates()

# Each store must belong to exactly one state.
store_state_counts = df.groupby("store_id")["state_id"].nunique()

if not (store_state_counts == 1).all():
    raise ValueError("At least one store maps to multiple states.")

# Bottom-level series: one product sold in one store.
leaves = (
    df[["item_id", "store_id", "state_id"]]
    .drop_duplicates()
    .sort_values(["state_id", "store_id", "item_id"])
    .reset_index(drop=True)
)

leaves["leaf_id"] = (
    leaves["item_id"] + "__" + leaves["store_id"]
)

if leaves["leaf_id"].duplicated().any():
    raise ValueError("Duplicate product-store leaf IDs detected.")

# Build aggregate membership for every leaf.
leaves["state_node"] = "STATE__" + leaves["state_id"]
leaves["store_node"] = "STORE__" + leaves["store_id"]
leaves["product_node"] = "PRODUCT__" + leaves["item_id"]
leaves["total_node"] = "TOTAL"

nodes = (
    ["TOTAL"]
    + sorted(leaves["state_node"].unique())
    + sorted(leaves["store_node"].unique())
    + sorted(leaves["product_node"].unique())
    + sorted(leaves["leaf_id"].unique())
)

S = np.zeros((len(nodes), len(leaves)), dtype=np.int8)
node_index = {node: i for i, node in enumerate(nodes)}

for j, row in leaves.iterrows():
    memberships = [
        "TOTAL",
        row["state_node"],
        row["store_node"],
        row["product_node"],
        row["leaf_id"],
    ]

    for node in memberships:
        S[node_index[node], j] = 1

matrix_df = pd.DataFrame(S, index=nodes, columns=leaves["leaf_id"])
matrix_path = OUTPUT_DIR / "hierarchy_summation_matrix.csv"
matrix_df.to_csv(matrix_path, index_label="node")

print("Hierarchy validated.")
print("Bottom-level product-store series:", len(leaves))
print("States:", leaves["state_id"].nunique())
print("Stores:", leaves["store_id"].nunique())
print("Products:", leaves["item_id"].nunique())
print("Total hierarchy nodes:", len(nodes))
print("Summation matrix shape:", S.shape)
print("Saved:", matrix_path)

# Verify every leaf belongs to exactly one state and one store.
print("\nLeaf membership check:")
print("Each leaf has one state:", (S[1:1 + leaves["state_id"].nunique()].sum(axis=0) == 1).all())
