"""Basic building blocks for BrepFormer.

Implementation follows the official BrepFormer blocks.py with:
- RMSNorm (not LayerNorm)
- SwiGLU FFN
- MLP and NonLinear modules
- NonLinearClassifier (4-layer with BN)
- EdgeConv for graph convolution on edge features
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Dimension to normalize.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    Reference: https://arxiv.org/abs/2002.05202
    Implements: SwiGLU(x, W, V, b, c) = Swish(xW + b) * (xV + c)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Initialize SwiGLU.

        Args:
            in_dim: Input dimension.
            hidden_dim: Hidden dimension (will use 2/3 * 4 * in_dim formula if not specified).
            out_dim: Output dimension (defaults to in_dim).
            dropout: Dropout probability.
        """
        super().__init__()
        out_dim = out_dim or in_dim

        # Following paper: hidden_dim * 2/3 * 4 formula
        # But we use the provided hidden_dim directly
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)  # Output projection
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Swish(xW1) * (xW3), then W2."""
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    """Multi-layer Perceptron with configurable activation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """Initialize MLP.

        Args:
            in_dim: Input dimension.
            hidden_dim: Hidden dimension.
            out_dim: Output dimension.
            num_layers: Number of layers.
            activation: Activation function name.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_features = in_dim if i == 0 else hidden_dim
            out_features = out_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_features, out_features))

            if i < num_layers - 1:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(x)


class NonLinear(nn.Module):
    """Non-linear transformation with BatchNorm.

    From official BrepFormer: 2-layer MLP with BatchNorm.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """Initialize NonLinear.

        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            hidden_dim: Hidden dimension (defaults to out_dim).
        """
        super().__init__()
        hidden_dim = hidden_dim or out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        return x


class NonLinearClassifier(nn.Module):
    """Non-linear classifier head.

    From official BrepFormer blocks.py:
    Linear(hidden_dim, 512) -> BatchNorm -> ReLU -> Dropout ->
    Linear(512, 512) -> BatchNorm -> ReLU -> Dropout ->
    Linear(512, 256) -> BatchNorm -> ReLU -> Dropout ->
    Linear(256, num_classes)
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        """Initialize NonLinearClassifier.

        Args:
            in_dim: Input dimension.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)


class FaceSegmentationClassifier(nn.Module):
    """Face-level segmentation classifier head.

    3-layer MLP operating on per-face embeddings:
    Linear(in_dim -> hidden_dim) -> BN -> ReLU -> Dropout ->
    Linear(hidden_dim -> hidden_dim) -> BN -> ReLU -> Dropout ->
    Linear(hidden_dim -> num_classes)

    Handles (batch, N_faces, dim) input by reshaping for BatchNorm1d.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        """Initialize FaceSegmentationClassifier.

        Args:
            in_dim: Input dimension (per-face embedding dim).
            num_classes: Number of face-level classes.
            hidden_dim: Hidden dimension for MLP layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Per-face embeddings of shape (batch, N_faces, dim).

        Returns:
            Logits of shape (batch, N_faces, num_classes).
        """
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


class EdgeConv(nn.Module):
    """Edge Convolution for aggregating edge features along paths.

    Used in GraphAttnBias for multi-hop edge path encoding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        """Initialize EdgeConv.

        Args:
            in_dim: Input edge feature dimension.
            out_dim: Output edge feature dimension.
        """
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, out_dim)

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for edge convolution.

        Aggregates features from neighboring edges (edges sharing a node)
        via scatter-mean on both endpoints, then concatenates with the
        original edge feature.

        Args:
            edge_features: Edge features of shape (num_edges, in_dim).
            edge_index: Edge indices of shape (2, num_edges).

        Returns:
            Updated edge features of shape (num_edges, out_dim).
        """
        src, dst = edge_index
        num_edges = edge_features.shape[0]
        feat_dim = edge_features.shape[-1]
        num_nodes = max(src.max().item(), dst.max().item()) + 1

        # Aggregate edge features to nodes via scatter-mean on source endpoint
        node_feat = torch.zeros(num_nodes, feat_dim, device=edge_features.device)
        node_count = torch.zeros(num_nodes, 1, device=edge_features.device)
        node_feat.scatter_add_(0, src.unsqueeze(-1).expand(-1, feat_dim), edge_features)
        node_count.scatter_add_(0, src.unsqueeze(-1), torch.ones(num_edges, 1, device=edge_features.device))
        node_feat = node_feat / (node_count + 1e-8)

        # For each edge, gather aggregated neighbor features from dst node
        neighbor_feat = node_feat[dst]  # (num_edges, feat_dim)

        out = torch.cat([edge_features, neighbor_feat], dim=-1)
        return F.relu(self.fc(out))


def init_params_global(module: nn.Module):
    """Initialize parameters following official BrepFormer.

    Args:
        module: Module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
