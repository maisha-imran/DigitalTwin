"""
Physics-Informed Graph Neural Network (PI-GNN) for road deterioration prediction.

Architecture:
  - Input projection layer
  - 3 × Graph Attention (GAT) layers with residual connections
  - Dropout regularization
  - MLP regression head → predicted IRI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


class PIGNN(nn.Module):
    """
    Physics-Informed Graph Attention Network for IRI prediction.

    Args:
        in_channels:     Number of input node features.
                         FIX: default changed 9 → 10 (age_factor added to FEATURE_COLS).
        hidden_channels: Hidden dimension size.
                         FIX: default changed 64 → 128 (matches notebook).
        out_channels:    Output dimension (1 for IRI regression).
        num_heads:       Number of attention heads in GAT layers.
        dropout:         Dropout probability.
                         FIX: default changed 0.3 → 0.2 (matches notebook).
    """

    def __init__(
        self,
        in_channels: int = 10,      # FIX: was 9; age_factor is the 10th feature
        hidden_channels: int = 128, # FIX: was 64; notebook uses 128
        out_channels: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2,       # FIX: was 0.3; notebook uses 0.2
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection — named 'proj' to match the saved checkpoint from the notebook
        self.proj = nn.Linear(in_channels, hidden_channels)

        # GAT Layer 1
        self.gat1 = GATConv(
            hidden_channels,
            hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.bn1 = BatchNorm(hidden_channels)

        # GAT Layer 2
        self.gat2 = GATConv(
            hidden_channels,
            hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.bn2 = BatchNorm(hidden_channels)

        # GAT Layer 3 (final aggregation, single head)
        self.gat3 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.bn3 = BatchNorm(hidden_channels)

        # FIX: Regression head expanded to 3 layers (hidden→64→16→1) matching notebook.
        # Original had only 2 linear layers (hidden→32→1), limiting expressiveness.
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Input projection
        h = F.relu(self.proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GAT Block 1
        h1 = self.bn1(self.gat1(h, edge_index))
        h1 = F.elu(h1) + h  # residual

        # GAT Block 2
        h2 = self.bn2(self.gat2(h1, edge_index))
        h2 = F.elu(h2) + h1  # residual

        # GAT Block 3
        h3 = self.bn3(self.gat3(h2, edge_index))
        h3 = F.elu(h3)

        # Regression head
        out = self.head(h3)
        return out.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)