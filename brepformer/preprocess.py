#!/usr/bin/env python3
"""Preprocess MFTRCAD data for BrepFormer training.

This script loads raw B-rep graph data and creates preprocessed pickle files
that can be efficiently loaded during training with multi-worker dataloaders.

Usage:
    python brepformer/preprocess.py --data_dir brepformer/data/mftrcad --output_dir brepformer/data/mftrcad_processed

The preprocessed data includes:
- Face UV-grids and attributes
- Edge UV-grids and attributes
- Graph structure (edge_index, spatial_pos, in_degree)
- Optional: D2 distance descriptors, angle descriptors
- Labels (multi-hot from per-face labels or external file)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.data.preprocessing import precompute_graph_features


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess MFTRCAD data for BrepFormer")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing graphs/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Path to external labels JSON file (optional)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=27,
        help="Number of classes for classification",
    )
    parser.add_argument(
        "--compute_descriptors",
        action="store_true",
        default=False,
        help="Compute D2 and angle descriptors (slower but may improve quality)",
    )
    parser.add_argument(
        "--num_spatial",
        type=int,
        default=64,
        help="Maximum spatial distance for shortest path computation",
    )
    parser.add_argument(
        "--d2_bins",
        type=int,
        default=64,
        help="Number of bins for D2 shape descriptor",
    )
    parser.add_argument(
        "--angle_bins",
        type=int,
        default=64,
        help="Number of bins for angle descriptor",
    )
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test split ratios (comma-separated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_graph_file(graph_path: Path) -> Dict:
    """Load a single graph JSON file.

    Args:
        graph_path: Path to the graph JSON file.

    Returns:
        Dictionary with parsed graph data.
    """
    with open(graph_path, "r") as f:
        data = json.load(f)

    model_name = data[0]
    content = data[1]

    graph = content["graph"]
    edge_index = np.array(graph["edges"], dtype=np.int64)
    num_nodes = graph["num_nodes"]

    face_attr = np.array(content["graph_face_attr"], dtype=np.float32)
    face_grid = np.array(content["graph_face_grid"], dtype=np.float32)
    edge_attr = np.array(content["graph_edge_attr"], dtype=np.float32)
    edge_grid = np.array(content["graph_edge_grid"], dtype=np.float32)

    return {
        "model_name": model_name,
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "face_attr": face_attr,
        "face_grid": face_grid,
        "edge_attr": edge_attr,
        "edge_grid": edge_grid,
    }


def load_label(
    model_id: str,
    labels_dir: Path,
    external_labels: Optional[Dict],
    num_classes: int,
) -> np.ndarray:
    """Load label for a model.

    Args:
        model_id: Model identifier.
        labels_dir: Directory containing label files.
        external_labels: External labels dictionary (or None).
        num_classes: Number of classes.

    Returns:
        Label array (multi-hot for multi-label, scalar for single-label).
    """
    if external_labels is not None:
        label = external_labels.get(model_id, 0)
        return np.array(label, dtype=np.int64)

    # Multi-label from per-face labels
    label_file = labels_dir / f"{model_id}_result.json"
    if not label_file.exists():
        # Try alternative naming
        for f in labels_dir.glob(f"*{model_id}*.json"):
            if "_rel" not in f.name:
                label_file = f
                break

    multi_hot = np.zeros(num_classes, dtype=np.float32)
    if label_file.exists():
        with open(label_file, "r") as f:
            label_data = json.load(f)

        if "cls" in label_data:
            for face_id, class_id in label_data["cls"].items():
                if 0 <= class_id < num_classes:
                    multi_hot[class_id] = 1.0

    return multi_hot


def load_face_labels(
    model_id: str,
    labels_dir: Path,
    num_faces: int,
    num_classes: int,
) -> np.ndarray:
    """Load per-face class labels for a model.

    Args:
        model_id: Model identifier.
        labels_dir: Directory containing label files.
        num_faces: Number of faces in the model.
        num_classes: Number of classes.

    Returns:
        Array of shape (num_faces,) with class IDs per face, or -1 for missing.
    """
    face_labels = np.full(num_faces, -1, dtype=np.int64)

    label_file = labels_dir / f"{model_id}_result.json"
    if not label_file.exists():
        for f in labels_dir.glob(f"*{model_id}*.json"):
            if "_rel" not in f.name:
                label_file = f
                break

    if not label_file.exists():
        return face_labels

    with open(label_file, "r") as f:
        label_data = json.load(f)

    if "cls" in label_data:
        for face_id_str, class_id in label_data["cls"].items():
            face_id = int(face_id_str)
            if 0 <= face_id < num_faces and 0 <= class_id < num_classes:
                face_labels[face_id] = class_id

    return face_labels


def preprocess_sample(
    graph_path: Path,
    labels_dir: Path,
    external_labels: Optional[Dict],
    num_classes: int,
    compute_descriptors: bool,
    num_spatial: int,
    d2_bins: int,
    angle_bins: int,
) -> Dict:
    """Preprocess a single sample.

    Args:
        graph_path: Path to graph JSON file.
        labels_dir: Directory containing label files.
        external_labels: External labels dictionary (or None).
        num_classes: Number of classes.
        compute_descriptors: Whether to compute D2/angle descriptors.
        num_spatial: Maximum spatial distance.
        d2_bins: Number of D2 bins.
        angle_bins: Number of angle bins.

    Returns:
        Preprocessed sample dictionary.
    """
    # Load raw data
    data = load_graph_file(graph_path)

    # Extract model ID
    model_id = graph_path.stem.replace("_result", "")

    # Load label
    label = load_label(model_id, labels_dir, external_labels, num_classes)

    # Compute graph features
    face_centroids = data["face_attr"][:, 2:5] if data["face_attr"].shape[1] >= 5 else None
    if data["face_grid"].shape[1] >= 6:
        face_normals = data["face_grid"][:, 3:6, :, :].mean(axis=(2, 3))
    else:
        face_normals = None

    graph_features = precompute_graph_features(
        edge_index=data["edge_index"],
        num_nodes=data["num_nodes"],
        face_centroids=face_centroids if compute_descriptors else None,
        face_normals=face_normals if compute_descriptors else None,
        face_grids=data["face_grid"] if compute_descriptors else None,
        num_spatial=num_spatial,
        d2_bins=d2_bins,
        angle_bins=angle_bins,
    )

    # Load per-face labels
    face_labels = load_face_labels(model_id, labels_dir, data["num_nodes"], num_classes)

    # Create sample
    sample = {
        "model_id": model_id,
        "face_grid": data["face_grid"],
        "face_attr": data["face_attr"],
        "edge_index": data["edge_index"],
        "edge_attr": data["edge_attr"],
        "edge_grid": data["edge_grid"],
        "spatial_pos": graph_features["spatial_pos"],
        "in_degree": graph_features["in_degree"],
        "label": label,
        "face_labels": face_labels,
        "num_faces": data["num_nodes"],
        "num_edges": data["edge_index"].shape[1],
    }

    if "d2_distance" in graph_features:
        sample["d2_distance"] = graph_features["d2_distance"]
    if "angle_distance" in graph_features:
        sample["angle_distance"] = graph_features["angle_distance"]

    return sample


def main():
    """Main preprocessing function."""
    import gc

    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    graphs_dir = data_dir / "graphs"
    labels_dir = data_dir / "labels"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load external labels if provided
    external_labels = None
    if args.label_file:
        with open(args.label_file, "r") as f:
            external_labels = json.load(f)
        print(f"Loaded external labels from {args.label_file}")

    # Get all graph files
    graph_files = sorted(graphs_dir.glob("*.json"))
    num_files = len(graph_files)
    print(f"Found {num_files} graph files")

    # Assign split indices with stratification by dominant class
    ratios = [float(r) for r in args.split_ratio.split(",")]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6

    np.random.seed(args.seed)

    # Determine dominant class per model for stratification
    dominant_classes = []
    for graph_path in graph_files:
        label_path = labels_dir / graph_path.name
        dominant_cls = 0
        if label_path.exists():
            try:
                with open(label_path, "r") as lf:
                    ldata = json.load(lf)
                if isinstance(ldata, list) and len(ldata) > 1:
                    ldata = ldata[1]
                if isinstance(ldata, dict) and "cls" in ldata:
                    from collections import Counter
                    cls_counts = Counter(ldata["cls"].values())
                    if cls_counts:
                        dominant_cls = cls_counts.most_common(1)[0][0]
            except Exception:
                pass
        dominant_classes.append(dominant_cls)

    dominant_classes = np.array(dominant_classes)

    # Stratified split
    split_assignment = {}
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        # First split: train vs (val+test)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=ratios[1] + ratios[2], random_state=args.seed)
        train_idx, valtest_idx = next(sss1.split(np.arange(num_files), dominant_classes))
        # Second split: val vs test
        val_ratio = ratios[1] / (ratios[1] + ratios[2])
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_ratio, random_state=args.seed)
        val_idx, test_idx = next(sss2.split(valtest_idx, dominant_classes[valtest_idx]))
        val_idx = valtest_idx[val_idx]
        test_idx = valtest_idx[test_idx]
        for i in train_idx:
            split_assignment[i] = "train"
        for i in val_idx:
            split_assignment[i] = "val"
        for i in test_idx:
            split_assignment[i] = "test"
    except (ImportError, ValueError):
        # Fallback to random split if sklearn unavailable or stratification fails
        indices = np.random.permutation(num_files)
        n_train = int(num_files * ratios[0])
        n_val = int(num_files * ratios[1])
        for i in indices[:n_train]:
            split_assignment[i] = "train"
        for i in indices[n_train:n_train + n_val]:
            split_assignment[i] = "val"
        for i in indices[n_train + n_val:]:
            split_assignment[i] = "test"

    # Process one sample at a time, writing each to its own file to limit RAM
    split_sizes = {}
    total_success = 0
    for split_name in ["train", "val", "test"]:
        split_file_indices = sorted(i for i, s in split_assignment.items() if s == split_name)
        print(f"\nPreprocessing {split_name} split ({len(split_file_indices)} files)...")

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for file_idx in tqdm(split_file_indices):
            graph_path = graph_files[file_idx]
            try:
                sample = preprocess_sample(
                    graph_path=graph_path,
                    labels_dir=labels_dir,
                    external_labels=external_labels,
                    num_classes=args.num_classes,
                    compute_descriptors=args.compute_descriptors,
                    num_spatial=args.num_spatial,
                    d2_bins=args.d2_bins,
                    angle_bins=args.angle_bins,
                )
                sample_path = split_dir / f"{count:05d}.pkl"
                with open(sample_path, "wb") as f:
                    pickle.dump(sample, f)
                del sample
                count += 1
            except Exception as e:
                print(f"Error processing {graph_path.name}: {e}")
                continue

        split_sizes[split_name] = count
        total_success += count
        print(f"Saved {split_name} split ({count} samples) to {split_dir}/")

        gc.collect()

    print(f"\nSuccessfully preprocessed {total_success} samples")

    # Save metadata
    metadata = {
        "num_classes": args.num_classes,
        "num_samples": total_success,
        "compute_descriptors": args.compute_descriptors,
        "num_spatial": args.num_spatial,
        "d2_bins": args.d2_bins,
        "angle_bins": args.angle_bins,
        "external_labels": args.label_file is not None,
        "split_sizes": split_sizes,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_dir / 'metadata.json'}")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
