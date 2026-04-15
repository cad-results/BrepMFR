"""Graph pooling strategies for BrepFormer."""

from typing import Optional

import torch
import torch.nn as nn


class GraphPooling(nn.Module):
    """Graph pooling module to get graph-level representation.

    Supports multiple pooling strategies:
    - cls: Use the [CLS] token (virtual node) representation
    - mean: Mean pooling over all node representations
    - max: Max pooling over all node representations
    - attention: Attention-weighted pooling
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        pooling_type: str = "cls",
    ):
        """Initialize GraphPooling.

        Args:
            hidden_dim: Hidden dimension of node features.
            pooling_type: Type of pooling ('cls', 'mean', 'max', 'attention').
        """
        super().__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim

        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        node_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: Node features of shape (batch, num_nodes + 1, hidden_dim).
                           Position 0 is the [CLS] token.
            mask: Mask for valid nodes of shape (batch, num_nodes + 1).
                  True = valid, False = padding.

        Returns:
            Graph-level representation of shape (batch, hidden_dim).
        """
        if self.pooling_type == "cls":
            # Use [CLS] token at position 0
            return node_features[:, 0, :]

        elif self.pooling_type == "mean":
            # Mean pooling over face nodes (excluding [CLS])
            face_features = node_features[:, 1:, :]
            if mask is not None:
                face_mask = mask[:, 1:]  # Exclude [CLS] mask
                face_features = face_features * face_mask.unsqueeze(-1)
                pooled = face_features.sum(dim=1) / face_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = face_features.mean(dim=1)
            return pooled

        elif self.pooling_type == "max":
            # Max pooling over face nodes (excluding [CLS])
            face_features = node_features[:, 1:, :]
            if mask is not None:
                face_mask = mask[:, 1:]
                face_features = face_features.masked_fill(~face_mask.unsqueeze(-1), float("-inf"))
            pooled, _ = face_features.max(dim=1)
            return pooled

        elif self.pooling_type == "attention":
            # Attention-weighted pooling over face nodes
            face_features = node_features[:, 1:, :]
            attn_scores = self.attention(face_features).squeeze(-1)  # (batch, num_nodes)

            if mask is not None:
                face_mask = mask[:, 1:]
                attn_scores = attn_scores.masked_fill(~face_mask, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, num_nodes)
            pooled = (face_features * attn_weights.unsqueeze(-1)).sum(dim=1)
            return pooled

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
