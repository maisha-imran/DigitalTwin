"""
FastAPI backend for the Road Digital Twin system.

The model is pre-trained in a notebook and loaded at startup.
No training is performed here.

Endpoints
---------
Core
  GET  /                          – liveness check
  GET  /health                    – detailed system-health check
  GET  /metrics                   – GNN evaluation metrics + confusion matrix

Roads
  GET  /roads                     – paginated, filterable road list
  GET  /roads/{edge_id}           – single road details + per-road forecast
  GET  /roads/search              – filter by condition, road_type, urgency

Network
  GET  /network/health            – network health score, grade, breakdown

Predictions & Scenarios
  POST /predict                   – re-run inference (with env-param scaling)
  POST /scenario                  – what-if comparison vs baseline
  POST /budget-plan               – repair plan optimised for a budget

Forecasting
  GET  /forecast                  – network-wide multi-year IRI projection
  GET  /forecast/{edge_id}        – per-road year-by-year IRI forecast

Statistics
  GET  /maintenance               – maintenance priority ranking
  GET  /stats/by-road-type        – per road-type aggregations
  GET  /stats/by-condition        – per condition-band aggregations
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import pandas as pd
import math
import random
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from data.graph_builder import build_pyg_graph, load_scaler
from data.preprocess import load_data
from model.gnn_model import PIGNN
from model.inference import load_model, run_inference, get_maintenance_ranking
from utils.analysis import (
    compare_scenarios,
    compute_network_health,
    forecast_network,
    forecast_single_road,
    optimize_budget,
    stats_by_condition,
    stats_by_road_type,
)
from utils.helpers import NumpyEncoder, df_to_records
from utils.metrics import compute_confusion, compute_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH  = os.getenv("MODEL_PATH",  "saved_models/pignn_model.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "saved_models/scaler.pkl")
HISTORY_PATH = os.getenv("HISTORY_PATH", "saved_models/training_history.json")

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_state: dict = {
    "model":        None,
    "scaler":       None,
    "df_base":      None,   # raw loaded DataFrame (pre-inference, FULL GRAPH)
    "df_inferred":  None,   # inference output (baseline, FILTERED FOR DASHBOARD)
    "metrics":      None,
    "confusion":    None,
    "model_ready":  False,
}


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _boot_model() -> None:
    """Load model + scaler and run baseline inference at startup."""
    if not os.path.exists(MODEL_PATH):
        log.warning("Model checkpoint not found at %s — /predict endpoints will return 503.", MODEL_PATH)
        return
    if not os.path.exists(SCALER_PATH):
        log.warning("Scaler not found at %s — /predict endpoints will return 503.", SCALER_PATH)
        return

    log.info("Loading model from %s …", MODEL_PATH)
    _state["model"]  = load_model(MODEL_PATH)
    _state["scaler"] = load_scaler(SCALER_PATH)

    log.info("Loading road data and running baseline inference …")
    df = load_data()
    _state["df_base"] = df
    
    # 1. Run inference on the FULL network so the GNN has proper context
    full_inferred_df = run_inference(df, _state["model"], _state["scaler"])
    
    # 2. Filter down to major arteries FOR THE DASHBOARD
    major_roads = ["motorway", "trunk", "primary", "secondary", "tertiary"]
    arterial_df = full_inferred_df[full_inferred_df["road_type"].isin(major_roads)].copy()
    
    _state["df_inferred"] = arterial_df
    _state["model_ready"] = True
    log.info("Startup complete — %d arterial segments isolated from full network.", len(arterial_df))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _boot_model()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Road Digital Twin API",
    description=(
        "Physics-Informed GNN (PI-GNN) for road deterioration prediction.\n\n"
        "The model is pre-trained externally and loaded at startup. "
        "This API exposes inference, scenario analysis, forecasting, and "
        "maintenance optimisation endpoints."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Guards & helpers
# ---------------------------------------------------------------------------

def _require_model() -> None:
    if not _state["model_ready"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Ensure saved_models/pignn_model.pt and saved_models/scaler.pkl exist "
                "and restart the server."
            ),
        )


def _inferred_df() -> pd.DataFrame:
    _require_model()
    df = _state["df_inferred"]
    if df is None:
        raise HTTPException(status_code=503, detail="Inference data not yet available.")
    return df


def _json(payload) -> JSONResponse:
    """Serialize through NumpyEncoder to handle numpy scalars."""
    return JSONResponse(content=json.loads(json.dumps(payload, cls=NumpyEncoder)))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    traffic_scale:  float = Field(1.0, ge=0.1, le=5.0,  description="Multiplier applied to traffic_volume.")
    rainfall_scale: float = Field(1.0, ge=0.0, le=5.0,  description="Multiplier applied to rainfall_mm.")
    age_offset:     float = Field(0.0, ge=0.0, le=2.0,  description="Additive offset applied to age_factor.")


class ScenarioRequest(BaseModel):
    traffic_scale:  float = Field(1.0, ge=0.1, le=5.0)
    rainfall_scale: float = Field(1.0, ge=0.0, le=5.0)
    age_offset:     float = Field(0.0, ge=0.0, le=2.0)
    label:          str   = Field("scenario", description="Human-readable label for this scenario.")


class BudgetRequest(BaseModel):
    budget_usd: float = Field(..., gt=0, description="Available repair budget in USD.")
    strategy: Literal["worst_first", "cost_effective", "critical_only", "length_weighted"] = (
        Field("worst_first", description="Road-selection strategy.")
    )


# ---------------------------------------------------------------------------
# Routes — core
# ---------------------------------------------------------------------------

@app.get("/", tags=["Core"], summary="Liveness check")
def root():
    return {
        "message":     "Road Digital Twin API",
        "version":     "2.0.0",
        "model_ready": _state["model_ready"],
    }


@app.get("/health", tags=["Core"], summary="Detailed system health check")
def health():
    df = _state["df_inferred"]
    return {
        "status":        "ok" if _state["model_ready"] else "degraded",
        "model_ready":   _state["model_ready"],
        "model_path":    MODEL_PATH,
        "scaler_path":   SCALER_PATH,
        "model_exists":  os.path.exists(MODEL_PATH),
        "scaler_exists": os.path.exists(SCALER_PATH),
        "roads_loaded":  len(df) if df is not None else 0,
        "device":        str(next(_state["model"].parameters()).device)
                         if _state["model"] else "n/a",
    }


@app.get("/metrics", tags=["Core"], summary="Model evaluation metrics")
def get_metrics():
    """
    Return evaluation metrics computed at load time on the full inferred dataset
    (predicted vs ground-truth iri_future).
    """
    df = _inferred_df()

    # Recompute lazily if not cached
    if _state["metrics"] is None:
        y_true = np.asarray(df["iri_future"].values  if "iri_future"  in df.columns else df["iri_current"].values)
        y_pred = np.asarray(df["iri_predicted"].values)
        _state["metrics"]  = compute_metrics(y_true, y_pred)
        _state["confusion"] = compute_confusion(y_true, y_pred)

    return _json({
        "metrics":   _state["metrics"],
        "confusion": _state["confusion"],
    })


@app.get("/training-history", tags=["Core"], summary="Model training loss history")
def get_training_history():
    """
    Returns the actual epoch history from the training phase loaded from disk.
    """
    if not os.path.exists(HISTORY_PATH):
        raise HTTPException(
            status_code=404, 
            detail="Training history not found. Please run the notebook and save training_history.json."
        )
        
    with open(HISTORY_PATH, "r") as f:
        raw_history = json.load(f)
        
    formatted_history = []
    
    epochs = len(raw_history.get("train_loss", []))
    
    for e in range(epochs):
        formatted_history.append({
            "epoch": e + 1,
            "train": round(raw_history["train_loss"][e], 4),
            "val":   round(raw_history["val_loss"][e], 4),
            "phys":  round(raw_history["phys_loss"][e], 4)
        })
        
    return _json(formatted_history)


# ---------------------------------------------------------------------------
# Routes — roads
# ---------------------------------------------------------------------------

@app.get("/roads", tags=["Roads"], summary="List road segments (filterable, paginated)")
def list_roads(
    limit:      int           = Query(500,  ge=1,  le=5_000),
    offset:     int           = Query(0,    ge=0),
    condition:  Optional[str] = Query(None, description="Filter by predicted condition (Good/Fair/Moderate/Poor/Critical)"),
    road_type:  Optional[str] = Query(None, description="Filter by road type (motorway/trunk/primary/…)"),
    urgency:    Optional[str] = Query(None, description="Filter by urgency (None/Low/Medium/High/Immediate)"),
    sort_by:    str           = Query("iri_predicted", description="Column to sort by"),
    descending: bool          = Query(True),
):
    df = _inferred_df()

    if condition:
        df = df[df["condition_predicted"] == condition]
    if road_type:
        df = df[df["road_type"] == road_type]
    if urgency:
        df = df[df["urgency_predicted"] == urgency]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)

    total_filtered = len(df)
    page = df.iloc[offset : offset + limit]

    return _json({
        "total":    total_filtered,
        "offset":   offset,
        "limit":    limit,
        "roads":    df_to_records(page),
        "stats": {
            "avg_iri_predicted":     round(float(df["iri_predicted"].mean()), 3) if not df.empty else 0.0,
            "total_repair_cost_usd": round(float(df["repair_cost_usd"].sum()), 2) if not df.empty else 0.0,
        },
    })


@app.get("/roads/{edge_id}", tags=["Roads"], summary="Single road details + multi-year forecast")
def get_road(edge_id: str, forecast_years: int = Query(10, ge=1, le=20)):
    df = _inferred_df()
    matches = df[df["edge_id"] == edge_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Road '{edge_id}' not found.")

    row      = matches.iloc[0]
    forecast = forecast_single_road(row, years=forecast_years)

    return _json({
        "road":     row.to_dict(),
        "forecast": forecast,
    })


# ---------------------------------------------------------------------------
# Routes — network
# ---------------------------------------------------------------------------

@app.get("/network/health", tags=["Network"], summary="Network-wide health score and breakdown")
def network_health():
    df = _inferred_df()
    return _json(compute_network_health(df))


# ---------------------------------------------------------------------------
# Routes — predictions & scenarios
# ---------------------------------------------------------------------------

@app.post("/predict", tags=["Predictions"], summary="Re-run inference with environmental scaling")
def predict(req: PredictRequest):
    """
    Scale environmental inputs and re-run the GNN on the modified dataset.
    Updates the server's baseline inferred DataFrame.
    """
    _require_model()
    try:
        df = _state["df_base"].copy()
        df["traffic_volume"] = (df["traffic_volume"] * req.traffic_scale).clip(lower=0)
        df["rainfall_mm"]    = (df["rainfall_mm"]    * req.rainfall_scale).clip(lower=0)
        if "age_factor" in df.columns:
            df["age_factor"] = (df["age_factor"] + req.age_offset).clip(lower=1.0, upper=4.5)

        # Run inference on FULL graph
        full_inferred_df = run_inference(df, _state["model"], _state["scaler"])
        
        # Filter down to major arteries
        major_roads = ["motorway", "trunk", "primary", "secondary", "tertiary"]
        df_inferred = full_inferred_df[full_inferred_df["road_type"].isin(major_roads)].copy()
        
        _state["df_inferred"] = df_inferred

        return _json({
            "status":                 "success",
            "inputs": {
                "traffic_scale":  req.traffic_scale,
                "rainfall_scale": req.rainfall_scale,
                "age_offset":     req.age_offset,
            },
            "total_roads":            len(df_inferred),
            "condition_distribution": df_inferred["condition_predicted"].value_counts().to_dict(),
            "avg_iri_predicted":      round(float(df_inferred["iri_predicted"].mean()), 3),
            "total_repair_cost_usd":  round(float(df_inferred["repair_cost_usd"].sum()), 2),
        })
    except Exception as exc:
        log.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/scenario", tags=["Predictions"], summary="What-if scenario vs current baseline")
def scenario(req: ScenarioRequest):
    """
    Run a what-if scenario and return a diff against the current baseline
    *without* updating the server state.
    """
    _require_model()
    try:
        df = _state["df_base"].copy()
        df["traffic_volume"] = (df["traffic_volume"] * req.traffic_scale).clip(lower=0)
        df["rainfall_mm"]    = (df["rainfall_mm"]    * req.rainfall_scale).clip(lower=0)
        if "age_factor" in df.columns:
            df["age_factor"] = (df["age_factor"] + req.age_offset).clip(lower=1.0, upper=4.5)

        # Run inference on FULL graph
        full_scenario_df = run_inference(df, _state["model"], _state["scaler"])
        
        # Filter down to major arteries
        major_roads = ["motorway", "trunk", "primary", "secondary", "tertiary"]
        df_scenario = full_scenario_df[full_scenario_df["road_type"].isin(major_roads)].copy()
        
        df_baseline  = _state["df_inferred"]
        comparison   = compare_scenarios(df_baseline, df_scenario, label=req.label)

        return _json(comparison)
    except Exception as exc:
        log.exception("Scenario failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/budget-plan", tags=["Predictions"], summary="Budget-constrained repair optimisation")
def budget_plan(req: BudgetRequest):
    """
    Given a budget and strategy, return the optimal set of road segments
    to prioritise for repair.
    """
    df = _inferred_df()
    try:
        result = optimize_budget(df, budget_usd=req.budget_usd, strategy=req.strategy)
        return _json(result)
    except Exception as exc:
        log.exception("Budget plan failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Routes — forecasting
# ---------------------------------------------------------------------------

@app.get("/forecast", tags=["Forecasting"], summary="Network-wide multi-year IRI projection")
def forecast(
    years:          int  = Query(5,    ge=1, le=20),
    include_roads:  bool = Query(False, description="Include per-road trajectory table (large payload)."),
):
    df     = _inferred_df()
    result = forecast_network(df, years=years)

    if not include_roads:
        result.pop("trajectory", None)

    return _json(result)


@app.get("/forecast/{edge_id}", tags=["Forecasting"], summary="Per-road year-by-year IRI forecast")
def forecast_road(
    edge_id: str,
    years:   int = Query(10, ge=1, le=20),
):
    df      = _inferred_df()
    matches = df[df["edge_id"] == edge_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Road '{edge_id}' not found.")

    row      = matches.iloc[0]
    forecast = forecast_single_road(row, years=years)

    return _json({
        "edge_id":   edge_id,
        "road_type": row["road_type"],
        "lat":       round(float(row["lat"]), 6),
        "lon":       round(float(row["lon"]), 6),
        "forecast":  forecast,
    })


# ---------------------------------------------------------------------------
# Routes — statistics
# ---------------------------------------------------------------------------

@app.get("/maintenance", tags=["Statistics"], summary="Maintenance priority ranking")
def get_maintenance(
    top_n:       int           = Query(50,   ge=1, le=500),
    road_type:   Optional[str] = Query(None, description="Filter to a specific road type."),
    min_iri:     Optional[float] = Query(None, description="Only include roads with iri_predicted ≥ this value."),
):
    df = _inferred_df()

    if road_type:
        df = df[df["road_type"] == road_type]
    if min_iri is not None:
        df = df[df["iri_predicted"] >= min_iri]

    ranked     = get_maintenance_ranking(df, top_n=top_n)
    total_cost = float(ranked["repair_cost_usd"].sum())

    return _json({
        "total_estimated_cost_usd": round(total_cost, 2),
        "urgency_breakdown":        ranked["urgency_predicted"].value_counts().to_dict(),
        "top_n":                    top_n,
        "returned":                 len(ranked),
        "roads":                    df_to_records(ranked),
    })


@app.get("/stats/by-road-type", tags=["Statistics"], summary="Aggregated metrics per road type")
def stats_road_type():
    df = _inferred_df()
    return _json({"data": stats_by_road_type(df)})


@app.get("/stats/by-condition", tags=["Statistics"], summary="Aggregated metrics per condition band")
def stats_condition():
    df = _inferred_df()
    return _json({"data": stats_by_condition(df)})
