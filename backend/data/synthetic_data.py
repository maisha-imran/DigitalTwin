"""
Synthetic road network data generator.
Creates realistic road segment data with physics-consistent IRI values.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple

np.random.seed(42)

ROAD_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
ROAD_TYPE_IDS = {r: i for i, r in enumerate(ROAD_TYPES)}

ROAD_TYPE_TRAFFIC = {
    "motorway": (30000, 80000),
    "trunk": (15000, 40000),
    "primary": (8000, 20000),
    "secondary": (3000, 10000),
    "tertiary": (1000, 5000),
    "residential": (100, 1500),
}

ROAD_TYPE_LANES = {
    "motorway": (4, 8),
    "trunk": (2, 4),
    "primary": (2, 4),
    "secondary": (2, 2),
    "tertiary": (1, 2),
    "residential": (1, 2),
}

ROAD_TYPE_SPEED = {
    "motorway": (80, 120),
    "trunk": (60, 80),
    "primary": (40, 60),
    "secondary": (30, 50),
    "tertiary": (20, 40),
    "residential": (10, 30),
}

# FIX: Added road deterioration multipliers (used in det_rate calculation)
ROAD_DET_MULT = {
    "motorway": 0.55,
    "trunk": 0.70,
    "primary": 0.85,
    "secondary": 1.00,
    "tertiary": 1.20,
    "residential": 1.40,
}

# FIX: Added calibrated IRI base values per road type
ROAD_IRI_BASE = {
    "motorway": 0.8,
    "trunk": 1.0,
    "primary": 1.2,
    "secondary": 1.5,
    "tertiary": 1.8,
    "residential": 2.2,
}

def _iri_from_features(
    traffic: float,
    rainfall: float,
    length: float,
    road_type: str,
    age_factor: float,
) -> float:
    """
    Calibrated physics-consistent IRI calculation.
    Ensures a realistic bell curve of road conditions.
    """
    base = ROAD_IRI_BASE[road_type]

    # Dampen the extreme penalties
    traf_effect = 1.0 * (traffic / 80000)          
    rain_effect = 0.5 * (rainfall / 2500)           
    len_effect  = 0.2 * min(length / 5000, 1.0)     
    
    # Age penalty only applies to the time *after* it was paved (age_factor - 1.0)
    age_effect  = (age_factor - 1.0) * ROAD_DET_MULT.get(road_type, 1.2) * 0.8  
    
    # Add a slightly larger noise spread (0.3) to create a few organic "Poor" anomalies
    noise       = np.random.normal(0, 0.3)

    iri = base + traf_effect + rain_effect + len_effect + age_effect + noise
    return float(np.clip(iri, 0.5, 10.0))


def generate_synthetic_graph(
    n_nodes: int = 500,                 # FIX: was 120; notebook uses 500
    center_lat: float = 12.9716,
    center_lon: float = 77.5946,
    lat_spread: float = 0.12,           # FIX: was 0.08; notebook uses 0.12
    lon_spread: float = 0.14,           # FIX: was 0.10; notebook uses 0.14
) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Generate a synthetic road network graph with realistic features.
    Returns (G, edges_df).
    """
    lats = center_lat + np.random.uniform(-lat_spread, lat_spread, n_nodes)
    lons = center_lon + np.random.uniform(-lon_spread, lon_spread, n_nodes)

    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, lat=lats[i], lon=lons[i])

    edges = set()
    for i in range(n_nodes):
        dists = np.sqrt((lats - lats[i]) ** 2 + (lons - lons[i]) ** 2)
        # FIX: connect 4 nearest neighbours (was 3); matches notebook
        for j in np.argsort(dists)[1:5]:
            edges.add((min(i, j), max(i, j)))

    # FIX: more random long-range edges (n//3 vs n//5); matches notebook
    for _ in range(n_nodes // 3):
        i, j = np.random.choice(n_nodes, 2, replace=False)
        edges.add((min(i, j), max(i, j)))

    rows = []
    for (u, v) in edges:
        road_type = np.random.choice(
            ROAD_TYPES,
            p=[0.05, 0.08, 0.15, 0.20, 0.22, 0.30],
        )
        dlat = (lats[v] - lats[u]) * 111000
        dlon = (lons[v] - lons[u]) * 111000 * np.cos(np.radians((lats[u] + lats[v]) / 2))
        length = float(max(50.0, np.sqrt(dlat ** 2 + dlon ** 2)))

        traffic  = float(np.random.uniform(*ROAD_TYPE_TRAFFIC[road_type]))
        rainfall = float(np.random.uniform(400, 2500))

        ll, lh = ROAD_TYPE_LANES[road_type]
        lanes = ll if ll == lh else int(np.random.randint(ll, lh + 1))
        sl, sh = ROAD_TYPE_SPEED[road_type]
        speed = sl if sl == sh else int(np.random.randint(sl, sh + 1))

        age_factor = float(np.random.uniform(1.0, 2.5))

        iri_current = _iri_from_features(traffic, rainfall, length, road_type, age_factor)

        # FIX: deterministic physics-based deterioration rate matching notebook
        det_rate = (
            0.35 * (traffic  / 30000)        +
            0.25 * (rainfall / 2500)         +
            0.15 * (age_factor - 1.0)        +
            0.10 * min(length / 5000, 1.0)   +
            0.05 * ROAD_DET_MULT[road_type]
        )
        det_rate = float(np.clip(det_rate * 1.5, 0.05, 1.5))

        # FIX: tiny noise (σ=0.05) so model can learn signal; IRI always increases
        iri_future = float(np.clip(
            iri_current + det_rate + np.random.normal(0, 0.05),
            iri_current + 0.02,   # always deteriorates
            10.0,                 # FIX: was 12.0
        ))

        mid_lat = (lats[u] + lats[v]) / 2
        mid_lon = (lons[u] + lons[v]) / 2

        G.add_edge(u, v, road_type=road_type, length=length, iri_current=iri_current)

        rows.append({
            "edge_id":       f"{u}_{v}",
            "node_u":        u,
            "node_v":        v,
            "lat":           mid_lat,
            "lon":           mid_lon,
            "road_type":     road_type,
            "road_type_id":  ROAD_TYPE_IDS[road_type],
            "lanes":         lanes,
            "traffic_volume": traffic,
            "rainfall_mm":   rainfall,
            "length_m":      length,
            "speed_limit":   speed,
            "iri_current":   iri_current,
            "iri_future":    iri_future,
            "age_factor":    age_factor,   # FIX: was missing in old version
        })

    df = pd.DataFrame(rows)
    return G, df


def iri_to_condition(iri: float) -> str:
    if iri < 2.0:
        return "Good"
    elif iri < 3.5:
        return "Fair"
    elif iri < 5.0:
        return "Moderate"
    elif iri < 7.0:
        return "Poor"
    else:
        return "Critical"


def condition_to_urgency(condition: str) -> str:
    mapping = {
        "Good": "None",
        "Fair": "Low",
        "Moderate": "Medium",
        "Poor": "High",
        "Critical": "Immediate",
    }
    return mapping.get(condition, "Unknown")


def estimate_repair_cost(iri: float, length_m: float, road_type: str) -> float:
    """Estimate repair cost in USD based on IRI and road type."""
    base_per_km = {
        "motorway": 120000,
        "trunk": 90000,
        "primary": 60000,
        "secondary": 40000,
        "tertiary": 25000,
        "residential": 15000,
    }.get(road_type, 30000)

    severity = max(0, (iri - 2.0) / 10.0)
    cost = base_per_km * (length_m / 1000) * severity
    return round(cost, 2)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["condition_current"] = df["iri_current"].apply(iri_to_condition)
    df["condition_future"]  = df["iri_future"].apply(iri_to_condition)
    df["urgency"]           = df["condition_future"].apply(condition_to_urgency)
    df["repair_cost_usd"]   = df.apply(
        lambda r: estimate_repair_cost(r["iri_future"], r["length_m"], r["road_type"]),
        axis=1,
    )
    return df
