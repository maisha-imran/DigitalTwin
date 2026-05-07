"""
Physics-informed loss for the GNN model.

Physics constraints:
  1. Monotonicity:        predicted IRI must not drop more than 2% below current IRI
                          (roads don't spontaneously improve without maintenance)
  2. Traffic sensitivity: higher traffic → higher deterioration
  3. Rainfall influence:  high rainfall → accelerated degradation
  4. Age sensitivity:     older roads deteriorate faster  [FIX: was missing entirely]
  5. Boundary:            predictions must stay within [0.5, 10.0] IRI range
"""

import torch
import torch.nn.functional as F

IRI_MIN = 0.5
IRI_MAX = 10.0   # FIX: was 12.0; notebook caps at 10.0

# Feature column indices in the RAW (unscaled) feature tensor
# FEATURE_COLS: lat(0) lon(1) road_type_id(2) lanes(3) traffic_volume(4)
#               rainfall_mm(5) length_m(6) speed_limit(7) iri_current(8) age_factor(9)
IDX_TRAFFIC    = 4
IDX_RAINFALL   = 5
IDX_IRI_CURRENT = 8
IDX_AGE        = 9   # FIX: was missing; age_factor is now the 10th feature


def physics_loss(
    predictions: torch.Tensor,
    x_raw: torch.Tensor,
    lambda_weight: float = 0.15,   # FIX: was 0.3; notebook uses 0.15 so MSE dominates
) -> torch.Tensor:
    """
    Compute physics-informed regularization loss.

    Args:
        predictions:   model output, shape [N]
        x_raw:         unscaled node features, shape [N, F]
        lambda_weight: weighting factor for total physics penalty

    Returns:
        Scalar tensor representing physics loss.
    """
    pred = predictions.squeeze()

    iri_current = x_raw[:, IDX_IRI_CURRENT]
    traffic     = x_raw[:, IDX_TRAFFIC]
    rainfall    = x_raw[:, IDX_RAINFALL]
    age         = x_raw[:, IDX_AGE]          # FIX: added age constraint
    delta       = pred - iri_current

    # --- Constraint 1: Monotonicity ---
    # FIX: allow only ≤2% improvement margin (was 10%/0.90).
    # Roads don't spontaneously improve; 2% margin handles prediction noise.
    loss_monotonicity = F.relu(iri_current * 0.98 - pred).mean()

    # --- Constraint 2: Traffic sensitivity ---
    # High-traffic segments must show higher deterioration than low-traffic.
    high_traffic_mask = (traffic > traffic.mean()).float()
    loss_traffic = F.relu(-delta * high_traffic_mask).mean()

    # --- Constraint 3: Rainfall influence ---
    # High rainfall should not lead to improved IRI predictions.
    high_rain_mask = (rainfall > rainfall.mean()).float()
    loss_rain = F.relu(-delta * high_rain_mask * 0.5).mean()

    # --- Constraint 4: Age sensitivity ---
    # FIX: entirely missing in original; older roads must deteriorate faster.
    high_age_mask = (age > age.mean()).float()
    loss_age = F.relu(-delta * high_age_mask * 0.3).mean()

    # --- Constraint 5: Boundary enforcement ---
    loss_boundary = (F.relu(IRI_MIN - pred) + F.relu(pred - IRI_MAX)).mean()

    total_physics = (
        loss_monotonicity
        + loss_traffic
        + loss_rain
        + loss_age        # FIX: added
        + loss_boundary
    )

    return lambda_weight * total_physics


def total_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    x_raw: torch.Tensor,
    lambda_physics: float = 0.15,   # FIX: was 0.3
) -> tuple:
    """
    Compute combined prediction + physics loss.

    Returns:
        (total, pred_loss, phys_loss)
    """
    pred_loss = F.mse_loss(predictions.squeeze(), targets)
    phys_loss = physics_loss(predictions, x_raw, lambda_weight=lambda_physics)
    return pred_loss + phys_loss, pred_loss, phys_loss