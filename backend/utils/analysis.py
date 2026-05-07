"""
Advanced analytics for the Road Digital Twin system.

Modules:
  - Network health scoring
  - Multi-year IRI deterioration forecasting
  - Budget-constrained repair optimization
  - Scenario (what-if) comparison engine
  - Road-type and geospatial aggregations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from data.synthetic_data import (
    iri_to_condition,
    condition_to_urgency,
    estimate_repair_cost,
    ROAD_DET_MULT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IRI_TARGET_GOOD = 2.0          # IRI threshold for "Good" condition
IRI_MAX         = 10.0
DET_TRAFFIC_W   = 0.35
DET_RAIN_W      = 0.25
DET_AGE_W       = 0.15
DET_LEN_W       = 0.10
DET_TYPE_W      = 0.05

CONDITION_HEALTH_SCORE = {
    "Good":     100,
    "Fair":      75,
    "Moderate":  50,
    "Poor":      25,
    "Critical":   0,
}

RepairStrategy = Literal["worst_first", "cost_effective", "critical_only", "length_weighted"]


# ---------------------------------------------------------------------------
# Utility: per-segment physics deterioration rate
# ---------------------------------------------------------------------------

def _deterioration_rate(row: pd.Series) -> float:
    """
    Compute annualised IRI deterioration rate for a single road segment
    using the same physics model as synthetic_data.py.
    """
    det = (
        DET_TRAFFIC_W * (row["traffic_volume"] / 30_000)
        + DET_RAIN_W  * (row["rainfall_mm"]    / 2_500)
        + DET_AGE_W   * (float(row.get("age_factor", 1.5)) - 1.0)
        + DET_LEN_W   * min(row["length_m"] / 5_000, 1.0)
        + DET_TYPE_W  * ROAD_DET_MULT.get(row["road_type"], 1.0)
    )
    return float(np.clip(det * 1.5, 0.05, 1.5))


# ---------------------------------------------------------------------------
# 1. Network health
# ---------------------------------------------------------------------------

def compute_network_health(df: pd.DataFrame) -> dict:
    """
    Return an overall health score (0–100), letter grade, and detailed
    condition/cost breakdown for the entire road network.
    """
    total = len(df)
    if total == 0:
        return {}

    cond_col = "condition_predicted" if "condition_predicted" in df.columns else "condition_current"
    cond_counts = df[cond_col].value_counts().to_dict()

    raw_score = sum(
        CONDITION_HEALTH_SCORE.get(cond, 50) * cnt
        for cond, cnt in cond_counts.items()
    ) / total

    if   raw_score >= 80: grade = "A"
    elif raw_score >= 65: grade = "B"
    elif raw_score >= 50: grade = "C"
    elif raw_score >= 35: grade = "D"
    else:                 grade = "F"

    total_len_km      = df["length_m"].sum() / 1_000
    critical_mask     = df[cond_col] == "Critical"
    poor_mask         = df[cond_col] == "Poor"
    actionable_mask   = critical_mask | poor_mask

    iri_col_pred    = "iri_predicted" if "iri_predicted" in df.columns else "iri_current"
    avg_iri_current = float(df["iri_current"].mean())
    avg_iri_pred    = float(df[iri_col_pred].mean())
    iri_delta       = round(avg_iri_pred - avg_iri_current, 3)

    repair_col = "repair_cost_usd"
    total_repair  = float(df[repair_col].sum()) if repair_col in df.columns else 0.0
    urgent_repair = float(df.loc[actionable_mask, repair_col].sum()) if repair_col in df.columns else 0.0

    return {
        "health_score":        round(raw_score, 1),
        "health_grade":        grade,
        "total_roads":         total,
        "total_length_km":     round(total_len_km, 2),
        "condition_breakdown": cond_counts,
        "condition_pct": {
            k: round(v / total * 100, 1) for k, v in cond_counts.items()
        },
        "avg_iri_current":        round(avg_iri_current, 3),
        "avg_iri_predicted":      round(avg_iri_pred, 3),
        "avg_iri_delta":          iri_delta,
        "critical_roads":         int(critical_mask.sum()),
        "poor_roads":             int(poor_mask.sum()),
        "actionable_roads":       int(actionable_mask.sum()),
        "critical_length_km":     round(float(df.loc[critical_mask, "length_m"].sum() / 1_000), 2),
        "total_repair_cost_usd":  round(total_repair, 2),
        "urgent_repair_cost_usd": round(urgent_repair, 2),
        "urgency_breakdown":      df["urgency_predicted"].value_counts().to_dict()
                                  if "urgency_predicted" in df.columns else {},
    }


# ---------------------------------------------------------------------------
# 2. Multi-year deterioration forecast
# ---------------------------------------------------------------------------

def forecast_network(
    df: pd.DataFrame,
    years: int = 5,
) -> dict:
    """
    Project IRI and condition distribution across the network for each
    year from 0 (baseline) up to `years`.

    Returns a dict keyed by year with per-year summaries plus a
    per-road long-format table as 'trajectory'.
    """
    base_col = "iri_predicted" if "iri_predicted" in df.columns else "iri_current"
    det_rates = df.apply(_deterioration_rate, axis=1).values

    yearly_summaries = []
    road_trajectories: list[dict] = []

    iri_matrix = np.zeros((len(df), years + 1), dtype=np.float32)
    iri_matrix[:, 0] = df[base_col].values.astype(np.float32)

    for yr in range(1, years + 1):
        iri_matrix[:, yr] = np.clip(
            iri_matrix[:, yr - 1] + det_rates,
            iri_matrix[:, yr - 1] + 0.01,   # always deteriorates
            IRI_MAX,
        )

    for yr in range(years + 1):
        iri_yr    = iri_matrix[:, yr]
        conds     = [iri_to_condition(v) for v in iri_yr]
        costs     = [
            estimate_repair_cost(iri_yr[i], df.iloc[i]["length_m"], df.iloc[i]["road_type"])
            for i in range(len(df))
        ]
        cond_counts = pd.Series(conds).value_counts().to_dict()
        total = len(df)

        score = sum(
            CONDITION_HEALTH_SCORE.get(c, 50) * cond_counts.get(c, 0)
            for c in CONDITION_HEALTH_SCORE
        ) / total

        yearly_summaries.append({
            "year":                  yr,
            "health_score":          round(float(score), 1),
            "avg_iri":               round(float(iri_yr.mean()), 3),
            "max_iri":               round(float(iri_yr.max()), 3),
            "condition_breakdown":   cond_counts,
            "critical_roads":        int(sum(1 for c in conds if c == "Critical")),
            "total_repair_cost_usd": round(float(sum(costs)), 2),
        })

    # Per-road trajectories (useful for sparklines in the UI)
    for i, row in df.iterrows():
        traj = {
            "edge_id":   row["edge_id"],
            "road_type": row["road_type"],
            "lat":       round(float(row["lat"]), 6),
            "lon":       round(float(row["lon"]), 6),
        }
        for yr in range(years + 1):
            traj[f"iri_year_{yr}"] = round(float(iri_matrix[i, yr]), 3)
        road_trajectories.append(traj)

    return {
        "years":            years,
        "yearly_summaries": yearly_summaries,
        "trajectory":       road_trajectories,
    }


def forecast_single_road(
    row: pd.Series,
    years: int = 10,
) -> list[dict]:
    """
    Year-by-year IRI forecast for a single road segment.
    Returns a list of {year, iri, condition, urgency, repair_cost_usd}.
    """
    base_col = "iri_predicted" if "iri_predicted" in row.index else "iri_current"
    det_rate = _deterioration_rate(row)
    iri = float(row[base_col])
    result = []
    for yr in range(years + 1):
        condition   = iri_to_condition(iri)
        repair_cost = estimate_repair_cost(iri, float(row["length_m"]), row["road_type"])
        result.append({
            "year":           yr,
            "iri":            round(iri, 3),
            "condition":      condition,
            "urgency":        condition_to_urgency(condition),
            "repair_cost_usd": round(repair_cost, 2),
        })
        iri = float(np.clip(iri + det_rate, iri + 0.01, IRI_MAX))
    return result


# ---------------------------------------------------------------------------
# 3. Budget-constrained repair optimisation
# ---------------------------------------------------------------------------

def optimize_budget(
    df: pd.DataFrame,
    budget_usd: float,
    strategy: RepairStrategy = "worst_first",
) -> dict:
    """
    Select which road segments to repair given a finite budget.

    Strategies
    ----------
    worst_first     : highest predicted IRI first (greedy)
    cost_effective  : maximise IRI-improvement per USD (bang-for-buck)
    critical_only   : only Critical/Poor roads, worst-first
    length_weighted : prioritise long critical roads (IRI × length)
    """
    df = df.copy()
    df["_iri_improvement"] = (df["iri_predicted"] - IRI_TARGET_GOOD).clip(lower=0)

    if strategy == "worst_first":
        candidates = df.sort_values("iri_predicted", ascending=False)

    elif strategy == "cost_effective":
        df["_value_score"] = df["_iri_improvement"] / (df["repair_cost_usd"] + 1.0)
        candidates = df.sort_values("_value_score", ascending=False)

    elif strategy == "critical_only":
        candidates = df[
            df["condition_predicted"].isin(["Critical", "Poor"])
        ].sort_values("iri_predicted", ascending=False)

    elif strategy == "length_weighted":
        df["_priority"] = df["iri_predicted"] * (df["length_m"] / 1_000)
        candidates = df.sort_values("_priority", ascending=False)

    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    selected_rows, total_cost = [], 0.0
    for _, row in candidates.iterrows():
        cost = float(row["repair_cost_usd"])
        if total_cost + cost <= budget_usd:
            selected_rows.append(row)
            total_cost += cost

    selected_df = pd.DataFrame(selected_rows)

    iri_improvement_total = float(selected_df["_iri_improvement"].sum()) if not selected_df.empty else 0.0
    roads_by_condition    = selected_df["condition_predicted"].value_counts().to_dict() if not selected_df.empty else {}

    output_cols = [
        "edge_id", "lat", "lon", "road_type", "length_m",
        "iri_predicted", "condition_predicted", "urgency_predicted", "repair_cost_usd",
    ]
    available = [c for c in output_cols if c in selected_df.columns]

    return {
        "strategy":               strategy,
        "budget_usd":             budget_usd,
        "roads_selected":         len(selected_df),
        "total_cost_usd":         round(total_cost, 2),
        "budget_utilised_pct":    round(total_cost / budget_usd * 100, 1) if budget_usd > 0 else 0.0,
        "remaining_budget_usd":   round(budget_usd - total_cost, 2),
        "total_iri_improvement":  round(iri_improvement_total, 3),
        "condition_breakdown":    roads_by_condition,
        "avg_iri_before":         round(float(selected_df["iri_predicted"].mean()), 3) if not selected_df.empty else 0.0,
        "roads":                  selected_df[available].to_dict(orient="records") if not selected_df.empty else [],
    }


# ---------------------------------------------------------------------------
# 4. Scenario comparison
# ---------------------------------------------------------------------------

def compare_scenarios(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
    label: str = "scenario",
) -> dict:
    """
    Diff two inferred DataFrames (baseline vs scenario) and return
    a summary of how conditions shift.
    """
    assert len(baseline) == len(scenario), "DataFrames must have the same length."

    delta_iri = scenario["iri_predicted"].values - baseline["iri_predicted"].values

    baseline_health = compute_network_health(baseline)
    scenario_health = compute_network_health(scenario)

    worsened = int((delta_iri > 0.1).sum())
    improved = int((delta_iri < -0.1).sum())
    unchanged = len(delta_iri) - worsened - improved

    return {
        "label":                 label,
        "baseline_health_score": baseline_health["health_score"],
        "scenario_health_score": scenario_health["health_score"],
        "health_score_delta":    round(scenario_health["health_score"] - baseline_health["health_score"], 1),
        "avg_iri_delta":         round(float(delta_iri.mean()), 4),
        "max_iri_delta":         round(float(delta_iri.max()), 4),
        "roads_worsened":        worsened,
        "roads_improved":        improved,
        "roads_unchanged":       unchanged,
        "baseline_repair_cost":  baseline_health["total_repair_cost_usd"],
        "scenario_repair_cost":  scenario_health["total_repair_cost_usd"],
        "repair_cost_delta":     round(
            scenario_health["total_repair_cost_usd"] - baseline_health["total_repair_cost_usd"], 2
        ),
        "baseline_condition_breakdown": baseline_health["condition_breakdown"],
        "scenario_condition_breakdown": scenario_health["condition_breakdown"],
    }


# ---------------------------------------------------------------------------
# 5. Aggregated stats
# ---------------------------------------------------------------------------

def stats_by_road_type(df: pd.DataFrame) -> list[dict]:
    """Per road-type aggregation of key IRI and cost metrics."""
    iri_col = "iri_predicted" if "iri_predicted" in df.columns else "iri_current"
    rows = []
    for rtype, grp in df.groupby("road_type"):
        cond_counts = grp["condition_predicted"].value_counts().to_dict() if "condition_predicted" in grp else {}
        rows.append({
            "road_type":              rtype,
            "count":                  len(grp),
            "total_length_km":        round(float(grp["length_m"].sum() / 1_000), 2),
            "avg_iri_current":        round(float(grp["iri_current"].mean()), 3),
            "avg_iri_predicted":      round(float(grp[iri_col].mean()), 3),
            "max_iri_predicted":      round(float(grp[iri_col].max()), 3),
            "avg_traffic_volume":     round(float(grp["traffic_volume"].mean()), 1),
            "avg_repair_cost_usd":    round(float(grp["repair_cost_usd"].mean()), 2),
            "total_repair_cost_usd":  round(float(grp["repair_cost_usd"].sum()), 2),
            "condition_breakdown":    cond_counts,
        })
    return sorted(rows, key=lambda r: r["avg_iri_predicted"], reverse=True)


def stats_by_condition(df: pd.DataFrame) -> list[dict]:
    """Per condition-band aggregation."""
    cond_col = "condition_predicted" if "condition_predicted" in df.columns else "condition_current"
    iri_col  = "iri_predicted"       if "iri_predicted"       in df.columns else "iri_current"
    order    = ["Critical", "Poor", "Moderate", "Fair", "Good"]
    rows     = []
    for cond in order:
        grp = df[df[cond_col] == cond]
        if grp.empty:
            continue
        rows.append({
            "condition":              cond,
            "urgency":                condition_to_urgency(cond),
            "count":                  len(grp),
            "pct":                    round(len(grp) / len(df) * 100, 1),
            "total_length_km":        round(float(grp["length_m"].sum() / 1_000), 2),
            "avg_iri":                round(float(grp[iri_col].mean()), 3),
            "avg_traffic_volume":     round(float(grp["traffic_volume"].mean()), 1),
            "total_repair_cost_usd":  round(float(grp["repair_cost_usd"].sum()), 2),
            "road_type_breakdown":    grp["road_type"].value_counts().to_dict(),
        })
    return rows