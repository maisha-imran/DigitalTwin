"""
Training loop for the PI-GNN model.
Supports full-graph training (transductive setting).
"""

import torch
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from model.gnn_model import PIGNN
from model.physics_loss import total_loss
from utils.metrics import compute_metrics
import os


def train_model(
    train_data: Data,
    val_data: Data,
    in_channels: int = 10,          # FIX: was 9; age_factor adds the 10th feature
    hidden_channels: int = 128,     # FIX: was 64; notebook uses 128
    num_heads: int = 4,
    dropout: float = 0.2,           # FIX: was 0.3; notebook uses 0.2
    lr: float = 3e-3,               # FIX: was 1e-3; notebook uses 3e-3
    weight_decay: float = 1e-4,
    epochs: int = 300,              # FIX: was 100; notebook trains for 300 epochs
    lambda_physics: float = 0.15,  # FIX: was 0.3; notebook uses 0.15
    save_path: str = "saved_models/pignn_model.pt",
    verbose: bool = True,
) -> tuple:
    """
    Train the PI-GNN model.

    Returns:
        (model, history_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PIGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # FIX: replaced CosineAnnealingLR with warmup + cosine decay matching notebook.
    # Original scheduler (CosineAnnealingLR) had no warmup phase, causing unstable
    # early training with the higher lr=3e-3.
    def lr_lambda(epoch: int) -> float:
        warmup = 20
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_data = train_data.to(device)
    val_data   = val_data.to(device)

    # FIX: added 'pred_loss' and 'phys_loss' keys to history (were missing)
    history = {
        "train_loss":  [],
        "val_loss":    [],
        "pred_loss":   [],   # FIX: was missing
        "phys_loss":   [],   # FIX: was missing
        "val_mae":     [],
        "val_rmse":    [],
        "val_r2":      [],
    }

    best_val_loss  = float("inf")
    best_state     = None             # FIX: store in-memory instead of only disk save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Training step ---
        model.train()
        optimizer.zero_grad()

        preds = model(train_data.x, train_data.edge_index)
        loss, pred_l, phys_l = total_loss(
            preds, train_data.y, train_data.x_raw, lambda_physics
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_preds = model(val_data.x, val_data.edge_index)
            val_loss, _, _ = total_loss(
                val_preds, val_data.y, val_data.x_raw, lambda_physics
            )

            val_preds_np = val_preds.cpu().numpy()
            val_y_np     = val_data.y.cpu().numpy()
            metrics      = compute_metrics(val_y_np, val_preds_np)

        history["train_loss"].append(float(loss))
        history["pred_loss"].append(float(pred_l))    # FIX: was missing
        history["phys_loss"].append(float(phys_l))    # FIX: was missing
        history["val_loss"].append(float(val_loss))
        history["val_mae"].append(metrics["mae"])
        history["val_rmse"].append(metrics["rmse"])
        history["val_r2"].append(metrics["r2"])

        if float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            # FIX: keep in-memory clone (like notebook) AND save to disk
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), save_path)

        if verbose and (epoch % 30 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {loss:.4f} (pred={pred_l:.4f}, phys={phys_l:.4f}) | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}"
            )

    # Reload best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(torch.load(save_path, map_location=device))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model parameters: {model.count_parameters():,}")

    return model, history