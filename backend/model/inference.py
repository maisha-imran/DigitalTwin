"""
Inference utilities for the trained PI-GNN model.
"""

from __future__ import annotations

import os
import torch
import numpy as np
import pandas as pd

from model.gnn_model import PIGNN
from data.graph_builder import build_pyg_graph, load_scaler, FEATURE_COLS
from data.synthetic_data import iri_to_condition, condition_to_urgency, estimate_repair_cost


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str = "saved_models/pignn_model.pt",
    in_channels: int = 10,
    hidden_channels: int = 128,
    num_heads: int = 4,
) -> PIGNN:
    """Load a trained PIGNN checkpoint from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PIGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    df: pd.DataFrame,
    model: PIGNN,
    scaler,
    iri_clip_max: float = 10.0,
) -> pd.DataFrame:
    """
    Run the PI-GNN on a road-segment DataFrame and return it enriched
    with predicted columns.

    Added columns
    -------------
    iri_predicted       – clipped GNN output
    condition_predicted – iri_to_condition(iri_predicted)
    urgency_predicted   – condition_to_urgency(condition_predicted)
    repair_cost_usd     – physics-based repair cost estimate
    deterioration_delta – iri_predicted - iri_current
    """
    device = next(model.parameters()).device

    # fit=False: use the pre-fitted scaler from training, never refit here
    data, _, df_out = build_pyg_graph(df, scaler=scaler, fit=False)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)

    preds_np = np.clip(preds.cpu().numpy(), 0.5, iri_clip_max)

    df_out = df_out.copy()
    df_out["iri_predicted"]       = preds_np
    df_out["condition_predicted"] = df_out["iri_predicted"].apply(iri_to_condition)
    df_out["urgency_predicted"]   = df_out["condition_predicted"].apply(condition_to_urgency)
    df_out["repair_cost_usd"]     = df_out.apply(
        lambda r: estimate_repair_cost(r["iri_predicted"], r["length_m"], r["road_type"]),
        axis=1,
    )
    df_out["deterioration_delta"] = (
        df_out["iri_predicted"] - df_out["iri_current"]
    ).round(4)

    return df_out


# ---------------------------------------------------------------------------
# Maintenance ranking
# ---------------------------------------------------------------------------

def get_maintenance_ranking(
    df_inferred: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Return the top-N road segments ranked by predicted IRI (worst first).
    Includes a normalised priority score in [0, 1].
    """
    cols = [
        "edge_id", "lat", "lon", "road_type", "length_m",
        "traffic_volume", "rainfall_mm", "age_factor",
        "iri_current", "iri_predicted", "deterioration_delta",
        "condition_predicted", "urgency_predicted", "repair_cost_usd",
    ]
    available = [c for c in cols if c in df_inferred.columns]
    ranked = (
        df_inferred[available]
        .sort_values("iri_predicted", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Normalised priority score: combines IRI and deterioration velocity
    iri_norm   = (ranked["iri_predicted"] - 0.5) / (10.0 - 0.5)
    delta_col  = "deterioration_delta" if "deterioration_delta" in ranked.columns else None
    delta_norm = (ranked[delta_col] / ranked[delta_col].abs().max()) if delta_col else 0
    ranked["priority_score"] = (0.7 * iri_norm + 0.3 * delta_norm).clip(0, 1).round(4)

    return ranked