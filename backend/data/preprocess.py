"""
Preprocessing utilities for road network data.
Handles both real GeoPackage maps and synthetic data fallbacks.
"""

import pandas as pd
import numpy as np
from data.synthetic_data import (
    generate_synthetic_graph,
    add_derived_columns,
)
from data.ingest_real_data import load_bengaluru_geopackage


def load_data(use_real_gpkg: bool = True, gpkg_path: str = "blr_roads_raw.gpkg") -> pd.DataFrame:
    """
    Load road network data. Tries the real Bengaluru GeoPackage first,
    falling back to synthetic generator if an error occurs.
    """
    if use_real_gpkg:
        try:
            df = load_bengaluru_geopackage(gpkg_path)
            return df
        except Exception as e:
            print(f"⚠️ GeoPackage ingestion failed ({e}). Falling back to synthetic layout.")

    # Synthetic fallback logic if the file is missing or corrupted
    G, df = generate_synthetic_graph(n_nodes=500)
    df = add_derived_columns(df)
    return df


def train_val_split(df: pd.DataFrame, val_ratio: float = 0.2):
    """Split dataframe into train/val sets."""
    idx = np.random.permutation(len(df))
    val_size = int(len(df) * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )