"""
Graph builder: converts road segment DataFrame into a PyTorch Geometric Data object.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import joblib
import os

# FIX: Added 'age_factor' — critical predictor missing from original FEATURE_COLS.
# Notebook uses 10 features; original backend only had 9 (no age_factor),
# causing a shape mismatch with the model's input projection layer.
FEATURE_COLS = [
    "lat",
    "lon",
    "road_type_id",
    "lanes",
    "traffic_volume",
    "rainfall_mm",
    "length_m",
    "speed_limit",
    "iri_current",
    "age_factor",   # FIX: was missing
]


def build_pyg_graph(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit: bool = True,           # FIX: param renamed from 'fit_scaler' to 'fit' (matches notebook)
) -> tuple:
    """
    Build a PyTorch Geometric Data object from the road segments DataFrame.

    Returns:
        (data, scaler, df_indexed)
    """
    df = df.reset_index(drop=True)

    X = df[FEATURE_COLS].values.astype(np.float32)

    if fit or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    y = df["iri_future"].values.astype(np.float32)

    # Build adjacency from shared road nodes
    node_to_segs = defaultdict(list)
    for idx, row in df.iterrows():
        node_to_segs[row["node_u"]].append(idx)
        node_to_segs[row["node_v"]].append(idx)

    edge_src = []
    edge_dst = []
    seen = set()
    for segs in node_to_segs.values():
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    edge_src += [a, b]
                    edge_dst += [b, a]

    if len(edge_src) == 0:
        # fallback: sequential edges
        edge_src = list(range(len(df) - 1)) + list(range(1, len(df)))
        edge_dst = list(range(1, len(df))) + list(range(len(df) - 1))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    data = Data(
        x          = torch.tensor(X_scaled, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.float),
        x_raw      = torch.tensor(X, dtype=torch.float),  # unscaled for physics loss
    )

    return data, scaler, df


def save_scaler(scaler: StandardScaler, path: str = "saved_models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str = "saved_models/scaler.pkl") -> StandardScaler:
    return joblib.load(path)