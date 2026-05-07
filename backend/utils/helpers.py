"""
General helper functions.
"""

import json
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-serialisable list of dicts."""
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return json.loads(json.dumps(records, cls=NumpyEncoder))