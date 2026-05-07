"""
FastAPI backend for the Road Digital Twin system.

Endpoints:
  POST /train     - Train the PI-GNN model
  POST /predict   - Run inference on road segments
  GET  /metrics   - Return evaluation metrics
  GET  /roads     - Return road segment data with predictions
  GET  /maintenance - Return maintenance priority ranking
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from data.preprocess import load_data, train_val_split
from data.graph_builder import build_pyg_graph, save_scaler, load_scaler
from model.gnn_model import PIGNN
from model.trainer import train_model
from model.inference import load_model, run_inference, get_maintenance_ranking
from utils.metrics import compute_metrics, compute_confusion
from utils.helpers import df_to_records, NumpyEncoder

app = FastAPI(
    title="Road Digital Twin API",
    description="Physics-Informed GNN for road deterioration prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Global state ----
_state: dict = {
    "model": None,
    "scaler": None,
    "df_inferred": None,
    "history": None,
    "metrics": None,
    "confusion": None,
    "trained": False,
}

MODEL_PATH = "saved_models/pignn_model.pt"
SCALER_PATH = "saved_models/scaler.pkl"


# ---- Request / Response models ----
class TrainRequest(BaseModel):
    epochs: int = 80
    hidden_channels: int = 64
    lr: float = 1e-3
    lambda_physics: float = 0.3
    use_osmnx: bool = False


class PredictRequest(BaseModel):
    traffic_scale: float = 1.0   # multiplier for traffic volume
    rainfall_scale: float = 1.0  # multiplier for rainfall


# ---- Helpers ----
def _ensure_model_loaded():
    if _state["model"] is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            _state["model"] = load_model(MODEL_PATH)
            _state["scaler"] = load_scaler(SCALER_PATH)
            # Run default inference
            df = load_data()
            _state["df_inferred"] = run_inference(df, _state["model"], _state["scaler"])
        else:
            raise HTTPException(
                status_code=404,
                detail="Model not found. Please train first via POST /train.",
            )


# ---- Routes ----
@app.get("/")
def root():
    return {"message": "Road Digital Twin API is running", "version": "1.0.0"}


@app.post("/train")
def train(req: TrainRequest):
    """Train the PI-GNN model on synthetic road data."""
    try:
        df = load_data(use_osmnx=req.use_osmnx)
        df_train, df_val = train_val_split(df, val_ratio=0.2)

        train_data, scaler, _ = build_pyg_graph(df_train, fit_scaler=True)
        val_data, _, _ = build_pyg_graph(df_val, scaler=scaler, fit_scaler=False)

        model, history = train_model(
            train_data=train_data,
            val_data=val_data,
            hidden_channels=req.hidden_channels,
            lr=req.lr,
            epochs=req.epochs,
            lambda_physics=req.lambda_physics,
            save_path=MODEL_PATH,
            verbose=True,
        )

        save_scaler(scaler, SCALER_PATH)

        # Compute final metrics on full val set
        device = next(model.parameters()).device
        val_data_d = val_data.to(device)
        with torch.no_grad():
            val_preds = model(val_data_d.x, val_data_d.edge_index).cpu().numpy()
        val_y = val_data.y.numpy()

        metrics = compute_metrics(val_y, val_preds)
        confusion = compute_confusion(val_y, val_preds)

        # Inference on full dataset
        df_inferred = run_inference(df, model, scaler)

        _state.update({
            "model": model,
            "scaler": scaler,
            "df_inferred": df_inferred,
            "history": history,
            "metrics": metrics,
            "confusion": confusion,
            "trained": True,
        })

        return {
            "status": "success",
            "epochs": req.epochs,
            "metrics": metrics,
            "total_roads": len(df),
            "history_summary": {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "best_val_mae": min(history["val_mae"]),
                "best_val_r2": max(history["val_r2"]),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """Run inference with environmental parameter scaling."""
    _ensure_model_loaded()
    try:
        df = load_data()
        df["traffic_volume"] = df["traffic_volume"] * req.traffic_scale
        df["rainfall_mm"] = df["rainfall_mm"] * req.rainfall_scale

        df_inferred = run_inference(df, _state["model"], _state["scaler"])
        _state["df_inferred"] = df_inferred

        return {
            "status": "success",
            "total_roads": len(df_inferred),
            "condition_distribution": df_inferred["condition_predicted"].value_counts().to_dict(),
            "avg_iri_predicted": round(float(df_inferred["iri_predicted"].mean()), 3),
            "total_repair_cost_usd": round(float(df_inferred["repair_cost_usd"].sum()), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Return model evaluation metrics and training history."""
    if _state["metrics"] is None:
        _ensure_model_loaded()
        # Compute metrics if we loaded from disk
        if _state["metrics"] is None:
            return {
                "metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0},
                "confusion": {"matrix": [], "labels": []},
                "history": {},
                "trained": _state["trained"],
            }

    return json.loads(json.dumps({
        "metrics": _state["metrics"],
        "confusion": _state["confusion"],
        "history": _state["history"],
        "trained": _state["trained"],
    }, cls=NumpyEncoder))


@app.get("/roads")
def get_roads(limit: int = 500):
    """Return road segments with predictions."""
    _ensure_model_loaded()
    df = _state["df_inferred"]
    if df is None:
        raise HTTPException(status_code=404, detail="No inference data available.")

    sample = df.head(limit) if len(df) > limit else df
    return {
        "roads": df_to_records(sample),
        "total": len(df),
        "condition_distribution": df["condition_predicted"].value_counts().to_dict(),
        "stats": {
            "avg_iri_current": round(float(df["iri_current"].mean()), 3),
            "avg_iri_predicted": round(float(df["iri_predicted"].mean()), 3),
            "total_repair_cost_usd": round(float(df["repair_cost_usd"].sum()), 2),
            "critical_roads": int((df["condition_predicted"] == "Critical").sum()),
            "poor_roads": int((df["condition_predicted"] == "Poor").sum()),
        },
    }


@app.get("/maintenance")
def get_maintenance(top_n: int = 50):
    """Return maintenance priority ranking."""
    _ensure_model_loaded()
    df = _state["df_inferred"]
    if df is None:
        raise HTTPException(status_code=404, detail="No inference data available.")

    ranked = get_maintenance_ranking(df, top_n=top_n)
    total_cost = float(ranked["repair_cost_usd"].sum())

    return {
        "roads": df_to_records(ranked),
        "total_estimated_cost_usd": round(total_cost, 2),
        "urgency_breakdown": ranked["urgency_predicted"].value_counts().to_dict(),
        "top_n": top_n,
    }
