"""Embedding modules for BrepFormer.

Implementation follows the official BrepFormer embedding.py with:
- SurfaceEncoder: 3-layer 2D CNN for UV-grid encoding
- CurveEncoder: 3-layer 1D CNN for curve grid encoding
- GraphNodeFeature: Face encoding with virtual [CLS] token
- GraphAttnBias: Edge/topology features to attention bias (KEY INNOVATION)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brepformer.models.layers.blocks import NonLinear, EdgeConv


class SurfaceEncoder(nn.Module):
    """Surface Encoder using 2D CNN on UV-grid samples.

    From official BrepFormer:
    Conv2d(7, 64, k=3) -> BN -> ReLU ->
    Conv2d(64, 128, k=3) -> BN -> ReLU ->
    Conv2d(128, 256, k=3) -> BN -> ReLU ->
    AdaptiveAvgPool2d(1) -> Flatten -> Linear(256, 128)
    """

    def __init__(
        self,
        in_channels: int = 7,
        channels: list = None,
        output_dim: int = 128,
    ):
        """Initialize SurfaceEncoder.

        Args:
            in_channels: Number of input channels (default 7: x,y,z,nx,ny,nz,mask).
            channels: List of channel sizes for conv layers.
            output_dim: Output embedding dimension.
        """
        super().__init__()
        channels = channels or [64, 128, 256]

        layers = []
        prev_channels = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_channels, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_channels = ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: UV-grid tensor of shape (batch, num_faces, channels, H, W)
               or (num_faces, channels, H, W).

        Returns:
            Face embeddings of shape (batch, num_faces, output_dim)
            or (num_faces, output_dim).
        """
        has_batch = x.dim() == 5
        if has_batch:
            batch_size, num_faces = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # (batch * num_faces, C, H, W)

        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        if has_batch:
            x = x.view(batch_size, num_faces, -1)

        return x


class CurveEncoder(nn.Module):
    """Curve Encoder using 1D CNN on curve grid samples.

    From official BrepFormer:
    Conv1d(12, 64, k=3) -> BN -> ReLU ->
    Conv1d(64, 128, k=3) -> BN -> ReLU ->
    Conv1d(128, num_heads, k=3) -> BN -> ReLU ->
    AdaptiveAvgPool1d(1)

    Note: Output dim is num_heads for direct attention bias.
    """

    def __init__(
        self,
        in_channels: int = 12,
        channels: list = None,
        output_dim: int = 32,  # num_heads
    ):
        """Initialize CurveEncoder.

        Args:
            in_channels: Number of input channels (default 12).
            channels: List of channel sizes for conv layers.
            output_dim: Output dimension (typically num_heads).
        """
        super().__init__()
        channels = channels or [64, 128]

        layers = []
        prev_channels = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv1d(prev_channels, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_channels = ch

        # Final layer outputs num_heads
        layers.extend([
            nn.Conv1d(prev_channels, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        ])

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Curve grid tensor of shape (batch, num_edges, channels, L)
               or (num_edges, channels, L).

        Returns:
            Edge embeddings of shape (batch, num_edges, output_dim)
            or (num_edges, output_dim).
        """
        has_batch = x.dim() == 4
        if has_batch:
            batch_size, num_edges = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # (batch * num_edges, C, L)

        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        if has_batch:
            x = x.view(batch_size, num_edges, -1)

        return x


class GraphNodeFeature(nn.Module):
    """Graph Node Feature encoder for faces.

    Encodes face features from:
    - UV-grid samples via SurfaceEncoder
    - Face attributes (type, area, centroid, rationality, num_loops)
    - Fuses all features via NonLinear layer
    - Prepends virtual [CLS] token for graph-level representation
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_face_types: int = 6,
        face_grid_channels: int = 7,
        surface_encoder_channels: list = None,
        surface_encoder_output_dim: int = 128,
        num_degree: int = 128,
    ):
        """Initialize GraphNodeFeature.

        Args:
            hidden_dim: Output hidden dimension.
            num_face_types: Number of face/surface types.
            face_grid_channels: Number of UV-grid channels.
            surface_encoder_channels: Channel sizes for SurfaceEncoder.
            surface_encoder_output_dim: SurfaceEncoder output dimension.
            num_degree: Number of in-degree embedding types.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Surface encoder for UV-grid
        self.surface_encoder = SurfaceEncoder(
            in_channels=face_grid_channels,
            channels=surface_encoder_channels,
            output_dim=surface_encoder_output_dim,
        )

        # Face attribute encoders
        self.face_type_encoder = nn.Embedding(num_face_types + 1, hidden_dim)  # +1 for padding
        self.face_area_encoder = nn.Linear(1, hidden_dim)
        self.face_centroid_encoder = nn.Linear(3, hidden_dim)
        self.face_rationality_encoder = nn.Embedding(2, hidden_dim)  # rational/non-rational
        self.face_loop_encoder = nn.Linear(1, hidden_dim)  # number of loops
        self.face_normal_encoder = nn.Linear(3, hidden_dim)  # normal at centroid
        self.face_bbox_encoder = nn.Linear(3, hidden_dim)  # bounding box extent
        self.face_reversed_encoder = nn.Embedding(2, hidden_dim)  # is_reversed

        # In-degree encoder
        self.in_degree_encoder = nn.Embedding(num_degree + 1, hidden_dim, padding_idx=0)

        # Fusion layer: surface + 8 attribute encodings
        fusion_input_dim = surface_encoder_output_dim + hidden_dim * 8
        self.fusion = NonLinear(fusion_input_dim, hidden_dim)

        # Virtual graph token ([CLS])
        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(
        self,
        face_grid: torch.Tensor,
        face_attr: torch.Tensor,
        in_degree: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            face_grid: UV-grid tensor of shape (batch, num_faces, C, H, W).
            face_attr: Face attributes of shape (batch, num_faces, attr_dim).
                       Expected attrs: [type, area, cx, cy, cz, rationality, num_loops, ...]
            in_degree: In-degree of each node of shape (batch, num_faces).

        Returns:
            Node features with prepended [CLS] token.
            Shape: (batch, num_faces + 1, hidden_dim).
        """
        batch_size, num_faces = face_grid.shape[:2]

        # Encode UV-grid
        surface_feat = self.surface_encoder(face_grid)  # (batch, num_faces, surface_dim)

        # Extract and encode face attributes
        # face_attr layout: [type, area, cx, cy, cz, rationality, num_loops, nx, ny, nz, bx, by, bz, is_reversed]
        face_type = face_attr[:, :, 0].long().clamp(0, self.face_type_encoder.num_embeddings - 1)
        face_area = face_attr[:, :, 1:2]  # (batch, num_faces, 1)
        face_centroid = face_attr[:, :, 2:5]  # (batch, num_faces, 3)
        face_rationality = face_attr[:, :, 5].long().clamp(0, 1)
        face_num_loops = face_attr[:, :, 6:7]  # (batch, num_faces, 1)
        face_normal = face_attr[:, :, 7:10]  # (batch, num_faces, 3)
        face_bbox = face_attr[:, :, 10:13]  # (batch, num_faces, 3)
        face_reversed = face_attr[:, :, 13].long().clamp(0, 1)

        # Encode attributes
        type_feat = self.face_type_encoder(face_type)  # (batch, num_faces, hidden_dim)
        area_feat = self.face_area_encoder(face_area)
        centroid_feat = self.face_centroid_encoder(face_centroid)
        rationality_feat = self.face_rationality_encoder(face_rationality)
        loop_feat = self.face_loop_encoder(face_num_loops)
        normal_feat = self.face_normal_encoder(face_normal)
        bbox_feat = self.face_bbox_encoder(face_bbox)
        reversed_feat = self.face_reversed_encoder(face_reversed)

        # In-degree encoding
        if in_degree is not None:
            in_degree = in_degree.clamp(0, self.in_degree_encoder.num_embeddings - 1)
            degree_feat = self.in_degree_encoder(in_degree)
        else:
            degree_feat = torch.zeros(batch_size, num_faces, self.hidden_dim, device=face_grid.device)

        # Concatenate all features
        all_feat = torch.cat([
            surface_feat,
            type_feat,
            area_feat,
            centroid_feat,
            rationality_feat,
            loop_feat,
            normal_feat,
            bbox_feat,
            reversed_feat,
        ], dim=-1)

        # Fuse features
        node_feat = self.fusion(all_feat)  # (batch, num_faces, hidden_dim)

        # Add in-degree encoding
        node_feat = node_feat + degree_feat

        # Prepend virtual [CLS] token
        cls_token = self.graph_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        node_feat = torch.cat([cls_token, node_feat], dim=1)  # (batch, num_faces + 1, hidden_dim)

        return node_feat


class GraphAttnBias(nn.Module):
    """Graph Attention Bias from edge and topology features.

    This is the KEY INNOVATION of BrepFormer: edge/topology features
    are converted to attention bias and added to attention scores.

    Components:
    - Spatial position encoding (shortest path distance)
    - D2 shape descriptor encoding
    - Angle descriptor encoding
    - CurveEncoder for edge UV-grid
    - Edge attribute encoders (type, length, angle, convexity)
    - Multi-hop edge aggregation via EdgeConv
    - Edge distance decay based on hop distance
    """

    def __init__(
        self,
        num_heads: int = 32,
        num_spatial: int = 64,
        num_edge_dis: int = 64,
        multi_hop_max_dist: int = 16,
        num_edge_types: int = 6,
        num_convexity_types: int = 3,
        edge_grid_channels: int = 12,
        curve_encoder_channels: list = None,
        d2_descriptor_dim: int = 64,
        angle_descriptor_dim: int = 64,
    ):
        """Initialize GraphAttnBias.

        Args:
            num_heads: Number of attention heads.
            num_spatial: Number of spatial position types.
            num_edge_dis: Number of edge distance types.
            multi_hop_max_dist: Maximum hop distance for edge paths.
            num_edge_types: Number of edge/curve types.
            num_convexity_types: Number of convexity types.
            edge_grid_channels: Number of edge grid channels.
            curve_encoder_channels: Channel sizes for CurveEncoder.
            d2_descriptor_dim: Dimension of D2 shape descriptor.
            angle_descriptor_dim: Dimension of angle descriptor.
        """
        super().__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        # Spatial position encoder (shortest path distance -> attention bias)
        self.spatial_pos_encoder = nn.Embedding(num_spatial + 1, num_heads, padding_idx=0)

        # D2 shape descriptor encoder
        self.d2_pos_encoder = NonLinear(d2_descriptor_dim, num_heads)

        # Angle descriptor encoder
        self.ang_pos_encoder = NonLinear(angle_descriptor_dim, num_heads)

        # Curve encoder for edge UV-grid
        self.curve_encoder = CurveEncoder(
            in_channels=edge_grid_channels,
            channels=curve_encoder_channels,
            output_dim=num_heads,
        )

        # Edge attribute encoders
        self.edge_type_encoder = nn.Embedding(num_edge_types + 1, num_heads, padding_idx=0)
        self.edge_len_encoder = NonLinear(1, num_heads)
        self.edge_ang_encoder = NonLinear(1, num_heads)
        self.edge_conv_encoder = nn.Embedding(num_convexity_types + 1, num_heads, padding_idx=0)

        # Edge distance decay encoder
        self.edge_dis_encoder = nn.Embedding(num_edge_dis + 1, num_heads, padding_idx=0)

        # Edge convolution for multi-hop aggregation
        self.edge_conv_layer = EdgeConv(num_heads, num_heads)

    def forward(
        self,
        spatial_pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_grid: torch.Tensor,
        edge_path: Optional[torch.Tensor] = None,
        d2_distance: Optional[torch.Tensor] = None,
        angle_distance: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to compute attention bias.

        Args:
            spatial_pos: Shortest path distances of shape (batch, N+1, N+1).
            edge_index: Edge indices of shape (batch, 2, num_edges).
            edge_attr: Edge attributes of shape (batch, num_edges, attr_dim).
                       Expected: [type, length, angle, convexity, ...]
            edge_grid: Edge UV-grid of shape (batch, num_edges, C, L).
            edge_path: Edge indices along shortest paths of shape (batch, N+1, N+1, max_dist).
            d2_distance: D2 shape descriptors of shape (batch, N+1, N+1, d2_dim).
            angle_distance: Angle descriptors of shape (batch, N+1, N+1, ang_dim).
            attn_mask: Attention mask of shape (batch, N+1).

        Returns:
            Attention bias of shape (batch, num_heads, N+1, N+1).
        """
        batch_size = spatial_pos.shape[0]
        num_nodes = spatial_pos.shape[1]

        # Initialize attention bias
        attn_bias = torch.zeros(batch_size, self.num_heads, num_nodes, num_nodes,
                                device=spatial_pos.device, dtype=torch.float32)

        # 1. Spatial position encoding
        spatial_pos_clamped = spatial_pos.clamp(0, self.spatial_pos_encoder.num_embeddings - 1).long()
        spatial_bias = self.spatial_pos_encoder(spatial_pos_clamped)  # (batch, N+1, N+1, num_heads)
        attn_bias += spatial_bias.permute(0, 3, 1, 2)
        del spatial_bias, spatial_pos_clamped

        # 2. D2 shape descriptor encoding
        if d2_distance is not None:
            d2_bias = self.d2_pos_encoder(d2_distance)  # (batch, N+1, N+1, num_heads)
            attn_bias += d2_bias.permute(0, 3, 1, 2)
            del d2_bias

        # 3. Angle descriptor encoding
        if angle_distance is not None:
            ang_bias = self.ang_pos_encoder(angle_distance)  # (batch, N+1, N+1, num_heads)
            attn_bias += ang_bias.permute(0, 3, 1, 2)
            del ang_bias

        # 4. Edge feature encoding
        if edge_grid.shape[1] > 0:  # Has edges
            # Encode edge UV-grid
            curve_feat = self.curve_encoder(edge_grid)  # (batch, num_edges, num_heads)

            # Encode edge attributes
            edge_type = (edge_attr[:, :, 0].long() + 1).clamp(0, self.edge_type_encoder.num_embeddings - 1)
            edge_len = edge_attr[:, :, 1:2]
            edge_ang = edge_attr[:, :, 8:9]
            edge_convexity = (edge_attr[:, :, 9].long() + 1).clamp(0, self.edge_conv_encoder.num_embeddings - 1)

            type_feat = self.edge_type_encoder(edge_type)  # (batch, num_edges, num_heads)
            len_feat = self.edge_len_encoder(edge_len)
            ang_feat = self.edge_ang_encoder(edge_ang)
            conv_feat = self.edge_conv_encoder(edge_convexity)

            # Combine edge features
            edge_feat = curve_feat + type_feat + len_feat + ang_feat + conv_feat  # (batch, num_edges, num_heads)
            del curve_feat, type_feat, len_feat, ang_feat, conv_feat

            # Zero out padding edge features (padding edges have all-zero grids)
            edge_is_padding = (edge_grid.abs().sum(dim=(2, 3)) == 0)  # (batch, num_edges)
            edge_feat = edge_feat.masked_fill(edge_is_padding.unsqueeze(-1), 0.0)

            # Add edge features to attention bias using scatter_add
            # edge_index has shape (batch, 2, num_edges)
            # We need to add +1 to account for [CLS] token at position 0
            src = (edge_index[:, 0, :] + 1).clamp(0, num_nodes - 1)  # (batch, num_edges)
            dst = (edge_index[:, 1, :] + 1).clamp(0, num_nodes - 1)  # (batch, num_edges)
            num_edges_actual = edge_index.shape[2]

            # Flatten for scatter_add operation
            # attn_bias shape: (batch, num_heads, num_nodes, num_nodes)
            # We want to add edge_feat[b, e, h] to attn_bias[b, h, src[b,e], dst[b,e]]

            # Create batch and edge indices
            batch_idx = torch.arange(batch_size, device=edge_index.device).view(-1, 1).expand(-1, num_edges_actual)

            # Flatten everything for scatter_add
            # Linear index into (num_nodes, num_nodes) matrix: src * num_nodes + dst
            linear_idx_fwd = src * num_nodes + dst  # (batch, num_edges)

            # edge_feat: (batch, num_edges, num_heads) -> need to scatter into (batch, num_heads, num_nodes * num_nodes)
            flat_bias = attn_bias.view(batch_size, self.num_heads, -1)  # (batch, num_heads, num_nodes^2)

            # Create the contribution tensor
            edge_feat_t = edge_feat.permute(0, 2, 1)  # (batch, num_heads, num_edges)

            # Expand indices for all heads
            linear_idx_fwd_exp = linear_idx_fwd.unsqueeze(1).expand(-1, self.num_heads, -1)  # (batch, num_heads, num_edges)

            # Use scatter_add to accumulate edge features (forward only;
            # edges are already bidirectional from build_face_adjacency)
            flat_bias = flat_bias.scatter_add(2, linear_idx_fwd_exp, edge_feat_t)

            # Reshape back
            attn_bias = flat_bias.view(batch_size, self.num_heads, num_nodes, num_nodes)

        # 5. Edge path encoding with distance decay
        if edge_path is not None:
            # edge_path: (batch, N+1, N+1, max_dist) contains edge indices along paths
            # Apply distance decay based on hop count
            for hop in range(min(self.multi_hop_max_dist, edge_path.shape[-1])):
                hop_edges = edge_path[:, :, :, hop]  # (batch, N+1, N+1)
                valid_mask = hop_edges >= 0

                # Distance decay
                hop_decay = self.edge_dis_encoder(
                    torch.full_like(hop_edges, hop + 1).clamp(0, self.edge_dis_encoder.num_embeddings - 1)
                )  # (batch, N+1, N+1, num_heads)

                # Apply decay where valid
                hop_decay = hop_decay * valid_mask.unsqueeze(-1).float()
                attn_bias = attn_bias + hop_decay.permute(0, 3, 1, 2)

        # Apply attention mask (set bias to -inf for padded positions)
        if attn_mask is not None:
            # attn_mask: (batch, N+1) where True = valid, False = padded
            mask_2d = attn_mask.unsqueeze(1) & attn_mask.unsqueeze(2)  # (batch, N+1, N+1)
            attn_bias = attn_bias.masked_fill(~mask_2d.unsqueeze(1), float("-inf"))

        return attn_bias
