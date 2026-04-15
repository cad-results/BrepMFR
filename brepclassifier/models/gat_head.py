"""GAT Classification Head for pipe fitting classification.

Converts BrepFormer padded batch format to PyG Batch, applies
GATv2Conv layers with residual connections, then GlobalAttention
pooling and a Dense classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, GlobalAttention
from torch_geometric.data import Data, Batch


class GATClassificationHead(nn.Module):
    """GAT-based classification head.

    Architecture:
        Face embeddings (batch, N, D) + edge_index + attn_mask
            -> convert padded tensors to PyG Batch
            -> GATv2Conv layers with skip connections
            -> GlobalAttention pooling
            -> Dense + Softmax
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_classes: int = 8,
        gat_num_layers: int = 3,
        gat_heads: int = 4,
        gat_hidden_dim: int = 256,
        gat_v2: bool = True,
        gat_dropout: float = 0.3,
        gat_pooling: str = "global_attention",
        dense_dims: list = None,
        dense_dropout: float = 0.3,
    ):
        """Initialize GATClassificationHead.

        Args:
            in_dim: Input feature dimension (BrepFormer hidden_dim).
            num_classes: Number of output classes.
            gat_num_layers: Number of GAT layers.
            gat_heads: Number of attention heads per GAT layer.
            gat_hidden_dim: Total hidden dimension of GAT layers.
            gat_v2: Use GATv2Conv (default True).
            gat_dropout: Dropout in GAT layers.
            gat_pooling: Pooling strategy ('global_attention').
            dense_dims: Hidden dimensions for dense head.
            dense_dropout: Dropout in dense head.
        """
        super().__init__()

        if dense_dims is None:
            dense_dims = [512, 256]

        self.in_dim = in_dim
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_heads = gat_heads
        head_dim = gat_hidden_dim // gat_heads

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()

        for i in range(gat_num_layers):
            layer_in_dim = in_dim if i == 0 else gat_hidden_dim
            gat = GATv2Conv(
                in_channels=layer_in_dim,
                out_channels=head_dim,
                heads=gat_heads,
                concat=True,  # output = head_dim * heads = gat_hidden_dim
                dropout=gat_dropout,
                add_self_loops=True,
            )
            self.gat_layers.append(gat)
            self.gat_norms.append(nn.BatchNorm1d(gat_hidden_dim))
            self.gat_dropouts.append(nn.Dropout(gat_dropout))

        # Skip projection if input dim != gat_hidden_dim
        if in_dim != gat_hidden_dim:
            self.skip_proj = nn.Linear(in_dim, gat_hidden_dim)
        else:
            self.skip_proj = None

        # Global attention pooling
        gate_nn = nn.Sequential(
            nn.Linear(gat_hidden_dim, gat_hidden_dim),
            nn.ReLU(),
            nn.Linear(gat_hidden_dim, 1),
        )
        self.pool = GlobalAttention(gate_nn)

        # Dense classification head (LayerNorm instead of BatchNorm to support batch_size=1)
        dense_layers = []
        prev_dim = gat_hidden_dim
        for dim in dense_dims:
            dense_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dense_dropout),
            ])
            prev_dim = dim
        dense_layers.append(nn.Linear(prev_dim, num_classes))
        self.dense = nn.Sequential(*dense_layers)

    def _padded_to_pyg_batch(
        self,
        face_emb: torch.Tensor,
        edge_index: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Batch:
        """Convert BrepFormer padded batch to PyG Batch.

        Args:
            face_emb: (B, N, D) face embeddings (already skipped CLS).
            edge_index: (B, 2, E) edge indices.
            attn_mask: (B, N) mask for valid faces (True=valid).

        Returns:
            PyG Batch object.
        """
        B, N, D = face_emb.shape
        data_list = []

        for b in range(B):
            # Get valid face mask
            mask = attn_mask[b]  # (N,)
            valid_count = mask.sum().item()

            if valid_count == 0:
                # Edge case: no valid faces
                x = torch.zeros(1, D, device=face_emb.device)
                ei = torch.zeros(2, 0, dtype=torch.long, device=face_emb.device)
                data_list.append(Data(x=x, edge_index=ei))
                continue

            # Extract valid face embeddings
            valid_indices = mask.nonzero(as_tuple=True)[0]
            x = face_emb[b, valid_indices]  # (valid_count, D)

            # Create index mapping: old_idx -> new_idx
            idx_map = torch.full((N,), -1, dtype=torch.long, device=face_emb.device)
            idx_map[valid_indices] = torch.arange(valid_count, device=face_emb.device)

            # Filter edge_index to valid range
            ei = edge_index[b]  # (2, E)
            src, dst = ei[0], ei[1]

            # Keep edges where both endpoints are valid and in range
            valid_edges = (src < N) & (dst < N) & mask[src.clamp(max=N-1)] & mask[dst.clamp(max=N-1)]
            # Also filter out zero-padded edges (both src and dst are 0 and not the first valid node)
            nonzero_edges = (src > 0) | (dst > 0) | (valid_count > 0)
            valid_edges = valid_edges & nonzero_edges

            if valid_edges.any():
                src_valid = idx_map[src[valid_edges]]
                dst_valid = idx_map[dst[valid_edges]]
                # Remove any -1 entries (shouldn't happen but safety check)
                keep = (src_valid >= 0) & (dst_valid >= 0)
                ei_new = torch.stack([src_valid[keep], dst_valid[keep]], dim=0)
            else:
                ei_new = torch.zeros(2, 0, dtype=torch.long, device=face_emb.device)

            data_list.append(Data(x=x, edge_index=ei_new))

        return Batch.from_data_list(data_list)

    def forward(
        self,
        face_emb: torch.Tensor,
        edge_index: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            face_emb: (B, N+1, D) node embeddings from BrepEncoder (includes CLS).
            edge_index: (B, 2, E) edge indices.
            attn_mask: (B, N+1) attention mask (includes CLS position).

        Returns:
            logits: (B, num_classes)
        """
        # Skip CLS token
        face_emb = face_emb[:, 1:, :]  # (B, N, D)
        face_mask = attn_mask[:, 1:]    # (B, N)

        # Convert to PyG batch
        pyg_batch = self._padded_to_pyg_batch(face_emb, edge_index, face_mask)

        x = pyg_batch.x
        ei = pyg_batch.edge_index
        batch_vec = pyg_batch.batch

        # Apply GAT layers with skip connections
        h = x
        for i, (gat, norm, drop) in enumerate(
            zip(self.gat_layers, self.gat_norms, self.gat_dropouts)
        ):
            h_in = h
            h = gat(h, ei)
            h = norm(h)
            h = F.elu(h)
            h = drop(h)

            # Skip connection
            if i == 0 and self.skip_proj is not None:
                h = h + self.skip_proj(h_in)
            elif i > 0:
                h = h + h_in

        # Global attention pooling -> (B, gat_hidden_dim)
        graph_emb = self.pool(h, batch_vec)

        # Dense head -> logits
        logits = self.dense(graph_emb)

        return logits
