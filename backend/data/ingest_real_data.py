import os
import numpy as np
import pandas as pd
import geopandas as gpd
from data.synthetic_data import (
    ROAD_TYPE_IDS,
    ROAD_TYPE_TRAFFIC,
    ROAD_DET_MULT,
    _iri_from_features,
    add_derived_columns
)

def load_bengaluru_geopackage(gpkg_path: str = "blr_roads_raw.gpkg") -> pd.DataFrame:
    """
    Parses the real Bengaluru GeoPackage and applies physics-consistent features 
    to align with the PI-GNN pipeline. Retains all road segments to ensure a 
    fully connected graph for proper message passing.
    """
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"Could not find the GeoPackage file at: {gpkg_path}")

    print(f" Reading GeoPackage layer from {gpkg_path}...")
    gdf_edges = gpd.read_file(gpkg_path, layer="blr_roads_raw")
    
    rows = []
    print(f" Total segments in raw file: {len(gdf_edges)}")
    print(" Processing full network to preserve graph connectivity for GNN...")
    
    for idx, row in gdf_edges.iterrows():
        # 1. Parse and standardize road type classification
        road_type = row.get("highway", "residential")
        if isinstance(road_type, list):
            road_type = road_type[0]
            
        # FIX: The filter blocking minor residential lanes has been removed.
        # The GNN needs these edges to prevent disconnected nodes and overfitting.

        # 2. Extract topological node mapping
        u_node = row.get("u", idx)
        v_node = row.get("v", str(idx + 1) if isinstance(idx, int) else f"{idx}_1")
        length = float(row.get("length", 100.0))
        
        # 3. Clean up messy lane formats like "['3', '4']", None, or comma-separated lists
        lanes_raw = row.get("lanes", 2)
        if lanes_raw is None or pd.isna(lanes_raw):
            lanes = 2
        elif isinstance(lanes_raw, list):
            lanes = int(lanes_raw[0]) if lanes_raw else 2
        else:
            # Convert to string and clean out brackets, quotes, or trailing spaces
            lanes_str = str(lanes_raw).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            # If it's a comma-separated string like "3, 4", take the first value
            if "," in lanes_str:
                lanes_str = lanes_str.split(",")[0]
            
            lanes_str = lanes_str.strip()
            lanes = int(lanes_str) if lanes_str.isdigit() else 2
        
        # 4. Extract speed limit with fallback
        speed_raw = row.get("maxspeed", "30")
        try:
            speed = int(str(speed_raw).split()[0]) if speed_raw else 30
        except ValueError:
            speed = 30

        # 5. Populate structural physics features
        traffic_bounds = ROAD_TYPE_TRAFFIC.get(road_type, (100, 1500))
        traffic_volume = float(np.random.uniform(*traffic_bounds))
        rainfall_mm = float(np.random.uniform(900.0, 1200.0))
        age_factor = float(np.random.uniform(1.0, 2.5))

        # 6. Calculate physics loss targets using baseline formula logic
        iri_current = _iri_from_features(traffic_volume, rainfall_mm, length, road_type, age_factor)
        
        det_rate = (
            0.35 * (traffic_volume / 30000)   +
            0.25 * (rainfall_mm / 2500)       +
            0.15 * (age_factor - 1.0)         +
            0.10 * min(length / 5000, 1.0)    +
            0.05 * ROAD_DET_MULT.get(road_type, 1.40) # Default to residential multiplier
        )
        det_rate = float(np.clip(det_rate * 1.5, 0.05, 1.5))
        
        iri_future = float(np.clip(
            iri_current + det_rate + np.random.normal(0, 0.05),
            iri_current + 0.02,
            10.0,
        ))

        # 7. Extract spatial center-mass coordinates
        centroid = row.geometry.centroid
        
        rows.append({
            "edge_id":        f"{u_node}_{v_node}",
            "node_u":         u_node,
            "node_v":         v_node,
            "lat":            centroid.y,
            "lon":            centroid.x,
            "road_type":      road_type,
            "road_type_id":   ROAD_TYPE_IDS.get(road_type, ROAD_TYPE_IDS["residential"]),
            "lanes":          lanes,
            "traffic_volume": traffic_volume,
            "rainfall_mm":    rainfall_mm,
            "length_m":       length,
            "speed_limit":    speed,
            "iri_current":    iri_current,
            "iri_future":     iri_future,
            "age_factor":     age_factor,
        })

    df = pd.DataFrame(rows)
    df = add_derived_columns(df)
    print(f" Ingestion successful! Full network dataframe generated with shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_bengaluru_geopackage("blr_roads_raw.gpkg")
    print("\nFirst 5 Rows of Parsed Network:")
    print(df[["edge_id", "road_type", "lanes", "iri_current", "repair_cost_usd"]].head())
