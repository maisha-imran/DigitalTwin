"""
Inference utilities for the trained PI-GNN model.
"""

import torch
import numpy as np
import pandas as pd
from model.gnn_model import PIGNN
from data.graph_builder import build_pyg_graph, load_scaler, FEATURE_COLS
from data.synthetic_data import iri_to_condition, condition_to_urgency, estimate_repair_cost
import os


def load_model(
    model_path: str = "saved_models/pignn_model.pt",
    in_channels: int = 10,          # FIX: was 9; must match the 10-feature FEATURE_COLS
    hidden_channels: int = 128,     # FIX: was 64; must match trained model architecture
    num_heads: int = 4,
) -> PIGNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PIGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def run_inference(df: pd.DataFrame, model: PIGNN, scaler) -> pd.DataFrame:
    """
    Run inference on a road segment DataFrame.

    Returns df with predicted columns added.
    """
    device = next(model.parameters()).device

    # FIX: param renamed from fit_scaler=False → fit=False to match build_pyg_graph signature
    data, _, df_out = build_pyg_graph(df, scaler=scaler, fit=False)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)

    preds_np = preds.cpu().numpy()

    df_out = df_out.copy()
    df_out["iri_predicted"]    = np.clip(preds_np, 0.5, 10.0)   # FIX: was 12.0
    df_out["condition_predicted"] = df_out["iri_predicted"].apply(iri_to_condition)
    df_out["urgency_predicted"]   = df_out["condition_predicted"].apply(condition_to_urgency)
    df_out["repair_cost_usd"]     = df_out.apply(
        lambda r: estimate_repair_cost(r["iri_predicted"], r["length_m"], r["road_type"]),
        axis=1,
    )

    return df_out


def get_maintenance_ranking(df_inferred: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Return top-N roads ranked by predicted IRI (worst first)."""
    cols = [
        "edge_id", "lat", "lon", "road_type", "length_m",
        "iri_current", "iri_predicted", "condition_predicted",
        "urgency_predicted", "repair_cost_usd",
    ]
    available = [c for c in cols if c in df_inferred.columns]
    ranked = df_inferred[available].sort_values("iri_predicted", ascending=False).head(top_n)
    return ranked.reset_index(drop=True)