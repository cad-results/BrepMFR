"""BrepEncoder: Stacked Transformer encoder for B-rep graphs.

Implementation follows the official BrepFormer encoder.py with:
- GraphNodeFeature for face encoding
- GraphAttnBias for edge/topology attention bias
- Stack of GraphEncoderLayer
- Final RMSNorm
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from brepformer.configs.config import BrepClassifierConfig
from brepformer.models.layers.embedding import GraphNodeFeature, GraphAttnBias
from brepformer.models.layers.encoder_layer import GraphEncoderLayer
from brepformer.models.layers.blocks import RMSNorm, init_params_global


class BrepEncoder(nn.Module):
    """BrepEncoder: Transformer encoder for B-rep graphs.

    Architecture:
    1. GraphNodeFeature: Encode faces + prepend [CLS] token
    2. GraphAttnBias: Compute attention bias from edges/topology
    3. Stack of GraphEncoderLayers with attention bias injection
    4. Final RMSNorm

    Returns node embeddings and graph embedding ([CLS] token).
    """

    def __init__(self, config: BrepClassifierConfig):
        """Initialize BrepEncoder.

        Args:
            config: Configuration object with all hyperparameters.
        """
        super().__init__()
        self.config = config
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)

        # Node feature encoder
        self.node_feature = GraphNodeFeature(
            hidden_dim=config.hidden_dim,
            num_face_types=config.num_face_types,
            face_grid_channels=config.face_grid_channels,
            surface_encoder_channels=config.surface_encoder_channels,
            surface_encoder_output_dim=config.surface_encoder_output_dim,
            num_degree=config.num_degree,
        )

        # Attention bias from edges/topology
        self.attn_bias = GraphAttnBias(
            num_heads=config.num_heads,
            num_spatial=config.num_spatial,
            num_edge_dis=config.num_edge_dis,
            multi_hop_max_dist=config.multi_hop_max_dist,
            num_edge_types=config.num_edge_types,
            num_convexity_types=config.num_convexity_types,
            edge_grid_channels=config.edge_grid_channels,
            curve_encoder_channels=config.curve_encoder_channels,
            d2_descriptor_dim=config.d2_descriptor_dim,
            angle_descriptor_dim=config.angle_descriptor_dim,
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            GraphEncoderLayer(
                hidden_dim=config.hidden_dim,
                ffn_dim=config.ffn_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation_dropout=config.activation_dropout,
                use_rope=config.use_rope,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim)

        # Initialize parameters
        self.apply(init_params_global)

        # Re-zero padding_idx entries that init_params_global corrupted
        for module in self.modules():
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(
        self,
        face_grid: torch.Tensor,
        face_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_grid: torch.Tensor,
        spatial_pos: torch.Tensor,
        in_degree: Optional[torch.Tensor] = None,
        edge_path: Optional[torch.Tensor] = None,
        d2_distance: Optional[torch.Tensor] = None,
        angle_distance: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            face_grid: UV-grid tensor of shape (batch, num_faces, C, H, W).
            face_attr: Face attributes of shape (batch, num_faces, attr_dim).
            edge_index: Edge indices of shape (batch, 2, num_edges).
            edge_attr: Edge attributes of shape (batch, num_edges, attr_dim).
            edge_grid: Edge UV-grid of shape (batch, num_edges, C, L).
            spatial_pos: Shortest path distances of shape (batch, N+1, N+1).
            in_degree: In-degree of each node of shape (batch, num_faces).
            edge_path: Edge indices along shortest paths.
            d2_distance: D2 shape descriptors.
            angle_distance: Angle descriptors.
            attn_mask: Attention mask of shape (batch, num_faces + 1).

        Returns:
            Tuple of:
            - node_emb: Node embeddings of shape (batch, num_faces + 1, hidden_dim)
            - graph_emb: Graph embedding of shape (batch, hidden_dim)
        """
        # Encode node features (faces + [CLS] token)
        node_emb = self.node_feature(face_grid, face_attr, in_degree)
        # node_emb shape: (batch, num_faces + 1, hidden_dim)

        # Compute attention bias from edges/topology
        bias = self.attn_bias(
            spatial_pos=spatial_pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_grid=edge_grid,
            edge_path=edge_path,
            d2_distance=d2_distance,
            angle_distance=angle_distance,
            attn_mask=attn_mask,
        )
        # bias shape: (batch, num_heads, num_faces + 1, num_faces + 1)

        # Free large input tensors no longer needed
        del face_grid, face_attr, in_degree, edge_index, edge_attr, edge_grid
        del edge_path, spatial_pos, d2_distance, angle_distance

        # Pass through transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                node_emb = grad_checkpoint(
                    layer, node_emb, bias, attn_mask,
                    use_reentrant=False,
                )
            else:
                node_emb = layer(node_emb, attn_bias=bias, attn_mask=attn_mask)

        # Final normalization
        node_emb = self.final_norm(node_emb)

        # Extract graph embedding from [CLS] token
        graph_emb = node_emb[:, 0, :]

        return node_emb, graph_emb
