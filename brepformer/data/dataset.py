"""Dataset for MFTRCAD B-rep data.

Loads B-rep graph data from JSON files and optionally external labels.
Supports multi-label classification (from per-face labels) or single-label
classification (from external label file).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from brepformer.data.preprocessing import precompute_graph_features


class MTFRCADDataset(Dataset):
    """Dataset for MFTRCAD B-rep classification.

    Loads B-rep graphs from JSON files in the format:
    [model_name, {graph, graph_face_attr, graph_face_grid, graph_edge_attr, graph_edge_grid}]

    Labels:
    - Default (multi-label): Derived from per-face labels in separate label files
    - External: Loaded from a JSON file mapping model_id -> class
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_file: Optional[str] = None,
        label_file: Optional[str] = None,
        num_classes: int = 27,
        multi_label: bool = True,
        num_spatial: int = 64,
        d2_bins: int = 64,
        angle_bins: int = 64,
        precompute_features: bool = True,
    ):
        """Initialize MTFRCADDataset.

        Args:
            data_dir: Directory containing graphs/ and labels/ subdirectories.
            split: Dataset split ('train', 'val', 'test').
            split_file: Path to split file listing model IDs. If None, uses all data.
            label_file: Path to external labels JSON file. If None, uses per-face labels.
            num_classes: Number of classes for classification.
            multi_label: Whether to use multi-label classification.
            num_spatial: Maximum spatial distance for preprocessing.
            d2_bins: Number of D2 histogram bins.
            angle_bins: Number of angle histogram bins.
            precompute_features: Whether to precompute graph features.
        """
        self.data_dir = Path(data_dir)
        self.graphs_dir = self.data_dir / "graphs"
        self.labels_dir = self.data_dir / "labels"
        self.split = split
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.num_spatial = num_spatial
        self.d2_bins = d2_bins
        self.angle_bins = angle_bins
        self.precompute_features = precompute_features

        # Load external labels if provided
        self.external_labels = None
        if label_file is not None:
            with open(label_file, "r") as f:
                self.external_labels = json.load(f)
            self.multi_label = False  # External labels are single-label

        # Get list of graph files
        self.graph_files = self._get_graph_files(split_file)

        # Build model_id to label file mapping
        self.label_file_map = self._build_label_map()

    def _get_graph_files(self, split_file: Optional[str]) -> List[Path]:
        """Get list of graph files for this split.

        Args:
            split_file: Path to split file, or None to use all files.

        Returns:
            List of graph file paths.
        """
        if split_file is not None and os.path.exists(split_file):
            # Load model IDs from split file
            with open(split_file, "r") as f:
                model_ids = [line.strip() for line in f if line.strip()]

            graph_files = []
            for model_id in model_ids:
                # Try to find matching graph file
                pattern = f"*{model_id}*.json"
                matches = list(self.graphs_dir.glob(pattern))
                if matches:
                    graph_files.append(matches[0])

            return graph_files
        else:
            # Use all graph files
            return sorted(self.graphs_dir.glob("*.json"))

    def _build_label_map(self) -> Dict[str, Path]:
        """Build mapping from model ID to label file.

        Returns:
            Dictionary mapping model_id (extracted from filename) to label file path.
        """
        label_map = {}
        for label_file in self.labels_dir.glob("*.json"):
            # Skip _rel files
            if "_rel" in label_file.name:
                continue
            # Extract model ID from filename
            model_id = label_file.stem.replace("_result", "")
            label_map[model_id] = label_file
        return label_map

    def _extract_model_id(self, graph_file: Path) -> str:
        """Extract model ID from graph filename.

        Args:
            graph_file: Path to graph file.

        Returns:
            Model ID string.
        """
        return graph_file.stem.replace("_result", "")

    def _load_label(self, model_id: str) -> torch.Tensor:
        """Load label for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Label tensor (multi-hot for multi-label, scalar for single-label).
        """
        if self.external_labels is not None:
            # Single-label from external file
            label = self.external_labels.get(model_id, 0)
            return torch.tensor(label, dtype=torch.long)

        # Multi-label from per-face labels
        label_file = self.label_file_map.get(model_id)
        if label_file is None:
            # No label file found, return zeros
            return torch.zeros(self.num_classes, dtype=torch.float32)

        with open(label_file, "r") as f:
            label_data = json.load(f)

        # Convert per-face labels to multi-hot vector
        multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        if "cls" in label_data:
            cls_dict = label_data["cls"]
            for face_id, class_id in cls_dict.items():
                if 0 <= class_id < self.num_classes:
                    multi_hot[class_id] = 1.0

        return multi_hot

    def _load_face_labels(self, model_id: str, num_faces: int) -> torch.Tensor:
        """Load per-face class labels for a model.

        Args:
            model_id: Model identifier.
            num_faces: Number of faces in the model.

        Returns:
            Tensor of shape (num_faces,) with class IDs per face, or -1 for missing.
        """
        face_labels = torch.full((num_faces,), -1, dtype=torch.long)

        label_file = self.label_file_map.get(model_id)
        if label_file is None:
            return face_labels

        with open(label_file, "r") as f:
            label_data = json.load(f)

        if "cls" in label_data:
            for face_id_str, class_id in label_data["cls"].items():
                face_id = int(face_id_str)
                if 0 <= face_id < num_faces and 0 <= class_id < self.num_classes:
                    face_labels[face_id] = class_id

        return face_labels

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
            - face_grid: UV-grid tensor (num_faces, C, H, W)
            - face_attr: Face attributes (num_faces, attr_dim)
            - edge_index: Edge indices (2, num_edges)
            - edge_attr: Edge attributes (num_edges, attr_dim)
            - edge_grid: Edge grid (num_edges, C, L)
            - spatial_pos: Shortest path distances (N+1, N+1)
            - in_degree: Node in-degrees (num_faces,)
            - label: Label tensor
            - model_id: Model identifier string
        """
        graph_file = self.graph_files[idx]
        model_id = self._extract_model_id(graph_file)

        # Load graph data
        with open(graph_file, "r") as f:
            data = json.load(f)

        # Data format: [model_name, {graph, graph_face_attr, ...}]
        model_name = data[0]
        content = data[1]

        # Extract graph structure
        graph = content["graph"]
        edge_index = np.array(graph["edges"], dtype=np.int64)  # (2, num_edges)
        num_nodes = graph["num_nodes"]

        # Extract face features
        face_attr = np.array(content["graph_face_attr"], dtype=np.float32)  # (N, 14)
        face_grid = np.array(content["graph_face_grid"], dtype=np.float32)  # (N, 7, 10, 10)

        # Extract edge features
        edge_attr = np.array(content["graph_edge_attr"], dtype=np.float32)  # (E, 15)
        edge_grid = np.array(content["graph_edge_grid"], dtype=np.float32)  # (E, 12, 10)

        # Precompute graph features
        if self.precompute_features:
            # Extract face centroids and normals for descriptors
            # Assuming face_attr layout includes centroid at indices 2:5
            # and we can estimate normals from face_grid
            face_centroids = face_attr[:, 2:5] if face_attr.shape[1] >= 5 else None

            # Estimate face normals from grid (average of grid normals)
            # face_grid has normals at channels 3:6
            if face_grid.shape[1] >= 6:
                face_normals = face_grid[:, 3:6, :, :].mean(axis=(2, 3))
            else:
                face_normals = None

            graph_features = precompute_graph_features(
                edge_index=edge_index,
                num_nodes=num_nodes,
                face_centroids=face_centroids,
                face_normals=face_normals,
                num_spatial=self.num_spatial,
                d2_bins=self.d2_bins,
                angle_bins=self.angle_bins,
            )
        else:
            # Minimal preprocessing
            from brepformer.data.preprocessing import compute_shortest_paths, compute_in_degree
            spatial_pos, _ = compute_shortest_paths(edge_index, num_nodes, self.num_spatial)
            in_degree = compute_in_degree(edge_index, num_nodes)
            graph_features = {
                "spatial_pos": spatial_pos,
                "in_degree": in_degree,
            }

        # Load label
        label = self._load_label(model_id)

        # Load per-face labels
        face_labels = self._load_face_labels(model_id, num_nodes)

        # Create sample dictionary
        sample = {
            "face_grid": torch.from_numpy(face_grid),
            "face_attr": torch.from_numpy(face_attr),
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
            "edge_grid": torch.from_numpy(edge_grid),
            "spatial_pos": torch.from_numpy(graph_features["spatial_pos"]),
            "in_degree": torch.from_numpy(graph_features["in_degree"]),
            "label": label,
            "face_labels": face_labels,
            "model_id": model_id,
            "num_faces": num_nodes,
            "num_edges": edge_index.shape[1],
        }

        # Add optional features
        if "d2_distance" in graph_features:
            sample["d2_distance"] = torch.from_numpy(graph_features["d2_distance"])
        if "angle_distance" in graph_features:
            sample["angle_distance"] = torch.from_numpy(graph_features["angle_distance"])

        return sample


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Create train/val/test splits from graph files.

    Args:
        data_dir: Directory containing graphs/ subdirectory.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.
        output_dir: Directory to save split files. If None, uses data_dir.

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing model IDs.
    """
    graphs_dir = Path(data_dir) / "graphs"
    output_dir = Path(output_dir or data_dir)

    # Get all model IDs
    model_ids = []
    for graph_file in graphs_dir.glob("*.json"):
        model_id = graph_file.stem.replace("_result", "")
        model_ids.append(model_id)

    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(model_ids)

    # Split
    n = len(model_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": model_ids[:n_train],
        "val": model_ids[n_train:n_train + n_val],
        "test": model_ids[n_train + n_val:],
    }

    # Save split files
    for split_name, split_ids in splits.items():
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, "w") as f:
            for model_id in split_ids:
                f.write(f"{model_id}\n")
        print(f"Saved {split_name} split with {len(split_ids)} samples to {split_file}")

    return splits
