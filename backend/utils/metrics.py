"""
Evaluation metrics for IRI prediction.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from data.synthetic_data import iri_to_condition


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute confusion matrix over condition classes."""
    classes = ["Good", "Fair", "Moderate", "Poor", "Critical"]
    true_labels = [iri_to_condition(v) for v in y_true]
    pred_labels = [iri_to_condition(v) for v in y_pred]
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    return {
        "matrix": cm.tolist(),
        "labels": classes,
    }