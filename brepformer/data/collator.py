"""Batch collation for BrepFormer.

Handles variable-size B-rep graphs by padding to the maximum size in each batch.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F


class BrepCollator:
    """Collator for batching B-rep graph samples.

    Pads variable-size tensors to the maximum size in the batch:
    - face_grid: pad to max_faces
    - face_attr: pad to max_faces
    - edge_index: pad to max_edges
    - edge_attr: pad to max_edges
    - edge_grid: pad to max_edges
    - spatial_pos: pad to max_faces + 1
    - in_degree: pad to max_faces
    - d2_distance: pad to max_faces + 1
    - angle_distance: pad to max_faces + 1
    """

    def __init__(
        self,
        pad_value: float = 0.0,
        max_faces: int = None,
        max_edges: int = None,
    ):
        """Initialize BrepCollator.

        Args:
            pad_value: Value to use for padding.
            max_faces: Maximum number of faces (optional, for truncation).
            max_edges: Maximum number of edges (optional, for truncation).
        """
        self.pad_value = pad_value
        self.max_faces = max_faces
        self.max_edges = max_edges

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch: List of sample dictionaries from dataset.

        Returns:
            Batched dictionary with padded tensors.
        """
        batch_size = len(batch)

        # Find maximum sizes in batch
        max_faces = max(s["num_faces"] for s in batch)
        max_edges = max(s["num_edges"] for s in batch)

        # Apply limits if specified
        if self.max_faces is not None:
            max_faces = min(max_faces, self.max_faces)
        if self.max_edges is not None:
            max_edges = min(max_edges, self.max_edges)

        # Initialize output tensors
        face_grid_shape = batch[0]["face_grid"].shape[1:]  # (C, H, W)
        edge_grid_shape = batch[0]["edge_grid"].shape[1:]  # (C, L)
        face_attr_dim = batch[0]["face_attr"].shape[-1]
        edge_attr_dim = batch[0]["edge_attr"].shape[-1]

        # Padded tensors
        face_grid = torch.zeros(batch_size, max_faces, *face_grid_shape)
        face_attr = torch.zeros(batch_size, max_faces, face_attr_dim)
        edge_index = torch.zeros(batch_size, 2, max_edges, dtype=torch.long)
        edge_attr = torch.zeros(batch_size, max_edges, edge_attr_dim)
        edge_grid = torch.zeros(batch_size, max_edges, *edge_grid_shape)
        spatial_pos = torch.zeros(batch_size, max_faces + 1, max_faces + 1, dtype=torch.long)
        in_degree = torch.zeros(batch_size, max_faces, dtype=torch.long)

        # Attention mask (True = valid, False = padding)
        attn_mask = torch.zeros(batch_size, max_faces + 1, dtype=torch.bool)

        # Labels
        if batch[0]["label"].dim() == 0:
            # Single-label (scalar)
            labels = torch.stack([s["label"] for s in batch])
        else:
            # Multi-label (vector)
            labels = torch.stack([s["label"] for s in batch])

        # Face labels (for face segmentation)
        has_face_labels = "face_labels" in batch[0]
        if has_face_labels:
            face_labels = torch.full((batch_size, max_faces), -1, dtype=torch.long)
        else:
            face_labels = None

        # Optional tensors
        has_d2 = "d2_distance" in batch[0]
        has_angle = "angle_distance" in batch[0]

        if has_d2:
            d2_dim = batch[0]["d2_distance"].shape[-1]
            d2_distance = torch.zeros(batch_size, max_faces + 1, max_faces + 1, d2_dim)
        else:
            d2_distance = None

        if has_angle:
            angle_dim = batch[0]["angle_distance"].shape[-1]
            angle_distance = torch.zeros(batch_size, max_faces + 1, max_faces + 1, angle_dim)
        else:
            angle_distance = None

        # Model IDs
        model_ids = []

        # Fill tensors
        for i, sample in enumerate(batch):
            n_faces = min(sample["num_faces"], max_faces)
            n_edges = min(sample["num_edges"], max_edges)

            # Face features
            face_grid[i, :n_faces] = sample["face_grid"][:n_faces]
            face_attr[i, :n_faces] = sample["face_attr"][:n_faces]

            # Edge features
            # Need to filter edges that connect truncated nodes
            valid_edges = (sample["edge_index"][0] < n_faces) & (sample["edge_index"][1] < n_faces)
            valid_edge_count = min(valid_edges.sum().item(), max_edges)

            if valid_edge_count > 0:
                valid_indices = valid_edges.nonzero(as_tuple=True)[0][:valid_edge_count]
                edge_index[i, :, :valid_edge_count] = sample["edge_index"][:, valid_indices]
                edge_attr[i, :valid_edge_count] = sample["edge_attr"][valid_indices]
                edge_grid[i, :valid_edge_count] = sample["edge_grid"][valid_indices]

            # Graph structure
            # spatial_pos has shape (N+1, N+1) where position 0 is [CLS]
            sp_size = min(sample["spatial_pos"].shape[0], max_faces + 1)
            spatial_pos[i, :sp_size, :sp_size] = sample["spatial_pos"][:sp_size, :sp_size]

            # In-degree
            in_degree[i, :n_faces] = sample["in_degree"][:n_faces]

            # Attention mask: [CLS] + valid faces
            attn_mask[i, 0] = True  # [CLS] token always valid
            attn_mask[i, 1:n_faces + 1] = True

            # Face labels
            if has_face_labels and "face_labels" in sample:
                face_labels[i, :n_faces] = sample["face_labels"][:n_faces]

            # Optional descriptors
            if has_d2 and "d2_distance" in sample:
                d2_size = min(sample["d2_distance"].shape[0], max_faces + 1)
                d2_distance[i, :d2_size, :d2_size] = sample["d2_distance"][:d2_size, :d2_size]

            if has_angle and "angle_distance" in sample:
                ang_size = min(sample["angle_distance"].shape[0], max_faces + 1)
                angle_distance[i, :ang_size, :ang_size] = sample["angle_distance"][:ang_size, :ang_size]

            model_ids.append(sample["model_id"])

        # Create output dictionary
        output = {
            "face_grid": face_grid,
            "face_attr": face_attr,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_grid": edge_grid,
            "spatial_pos": spatial_pos,
            "in_degree": in_degree,
            "attn_mask": attn_mask,
            "label": labels,
            "model_ids": model_ids,
        }

        if face_labels is not None:
            output["face_labels"] = face_labels
        if d2_distance is not None:
            output["d2_distance"] = d2_distance
        if angle_distance is not None:
            output["angle_distance"] = angle_distance

        return output
