"""
Preprocessing utilities for road network data.
Handles both OSMnx-sourced and synthetic data.
"""

import pandas as pd
import numpy as np
from data.synthetic_data import (
    generate_synthetic_graph,
    add_derived_columns,
    ROAD_TYPE_IDS,
)


def try_osmnx_load(place_name: str = "Bengaluru, India", network_type: str = "drive"):
    """
    Attempt to load road network from OSMnx.
    Returns (G, df) or (None, None) on failure.
    """
    try:
        import osmnx as ox
        G = ox.graph_from_place(place_name, network_type=network_type)
        gdf_edges = ox.graph_to_gdfs(G, nodes=False)

        rows = []
        for idx, row in gdf_edges.iterrows():
            road_type = row.get("highway", "residential")
            if isinstance(road_type, list):
                road_type = road_type[0]
            road_type = road_type if road_type in ROAD_TYPE_IDS else "residential"

            length    = float(row.get("length", 100))
            lanes_raw = row.get("lanes", 2)
            lanes     = int(lanes_raw) if not isinstance(lanes_raw, list) else int(lanes_raw[0])
            speed_raw = row.get("maxspeed", "30")
            speed     = int(str(speed_raw).split()[0]) if speed_raw else 30

            # FIX: age_factor was completely missing from OSMnx rows, causing a
            # KeyError in build_pyg_graph when constructing the feature matrix.
            age_factor = float(np.random.uniform(1.0, 2.5))

            rows.append({
                "edge_id":       f"{idx[0]}_{idx[1]}",
                "node_u":        idx[0],
                "node_v":        idx[1],
                "lat":           row.geometry.centroid.y,
                "lon":           row.geometry.centroid.x,
                "road_type":     road_type,
                "road_type_id":  ROAD_TYPE_IDS[road_type],
                "lanes":         lanes,
                "traffic_volume": np.random.uniform(500, 30000),
                "rainfall_mm":   900.0,
                "length_m":      length,
                "speed_limit":   speed,
                "iri_current":   np.random.uniform(1.5, 7.0),
                "iri_future":    np.random.uniform(1.5, 8.0),
                "age_factor":    age_factor,   # FIX: was missing
            })

        df = pd.DataFrame(rows)
        df = add_derived_columns(df)
        return G, df

    except Exception as e:
        print(f"OSMnx load failed ({e}), falling back to synthetic data.")
        return None, None


def load_data(use_osmnx: bool = False, place_name: str = "Bengaluru, India") -> pd.DataFrame:
    """Load road data. Tries OSMnx first if requested, else synthetic."""
    if use_osmnx:
        G, df = try_osmnx_load(place_name)
        if df is not None:
            return df

    # FIX: n_nodes=150 → 500 to match notebook dataset size
    G, df = generate_synthetic_graph(n_nodes=500)
    df = add_derived_columns(df)
    return df


def train_val_split(df: pd.DataFrame, val_ratio: float = 0.2):
    """Split dataframe into train/val sets."""
    idx      = np.random.permutation(len(df))          # FIX: use permutation (not shuffle-in-place)
    val_size = int(len(df) * val_ratio)
    val_idx  = idx[:val_size]
    train_idx = idx[val_size:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )