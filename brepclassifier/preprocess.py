#!/usr/bin/env python3
"""Preprocess pipe fitting data for training.

Loads graph JSON files, computes graph structure features, performs
stratified train/val/test splitting, computes class weights, and
saves preprocessed pickle files.

Usage:
    python brepclassifier/preprocess.py \
        --data_dir brepclassifier/data/ssdata1 \
        --output_dir brepclassifier/data/ssdata1_processed
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brepformer.data.preprocessing import precompute_graph_features


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess pipe fitting data")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing graphs/ and labels.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--compute_descriptors",
        action="store_true",
        default=False,
        help="Compute D2 and angle descriptors",
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
        help="Train/val/test split ratios",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_weight",
        type=float,
        default=10.0,
        help="Maximum class weight cap (inverse frequency)",
    )

    return parser.parse_args()


def load_graph_file(graph_path: Path) -> Dict:
    """Load a single graph JSON file."""
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


def preprocess_sample(
    graph_path: Path,
    label: int,
    compute_descriptors: bool,
    num_spatial: int,
    d2_bins: int,
    angle_bins: int,
) -> Dict:
    """Preprocess a single sample."""
    data = load_graph_file(graph_path)
    model_id = graph_path.stem

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
        num_spatial=num_spatial,
        d2_bins=d2_bins,
        angle_bins=angle_bins,
    )

    sample = {
        "model_id": model_id,
        "face_grid": data["face_grid"],
        "face_attr": data["face_attr"],
        "edge_index": data["edge_index"],
        "edge_attr": data["edge_attr"],
        "edge_grid": data["edge_grid"],
        "spatial_pos": graph_features["spatial_pos"],
        "in_degree": graph_features["in_degree"],
        "label": np.array(label, dtype=np.int64),
        "num_faces": data["num_nodes"],
        "num_edges": data["edge_index"].shape[1],
    }

    if "d2_distance" in graph_features:
        sample["d2_distance"] = graph_features["d2_distance"]
    if "angle_distance" in graph_features:
        sample["angle_distance"] = graph_features["angle_distance"]

    return sample


def stratified_split(labels: np.ndarray, ratios: List[float], seed: int):
    """Perform stratified train/val/test split.

    Uses sklearn StratifiedShuffleSplit for proper class-ratio preservation.

    Args:
        labels: Array of class labels.
        ratios: [train, val, test] ratios.
        seed: Random seed.

    Returns:
        train_indices, val_indices, test_indices
    """
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        # First split: train vs (val+test)
        test_val_ratio = ratios[1] + ratios[2]
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=seed)
        train_idx, temp_idx = next(sss1.split(np.zeros(len(labels)), labels))

        # Second split: val vs test
        temp_labels = labels[temp_idx]
        val_ratio_of_temp = ratios[1] / test_val_ratio
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_ratio_of_temp, random_state=seed)
        val_idx_local, test_idx_local = next(sss2.split(np.zeros(len(temp_labels)), temp_labels))

        val_idx = temp_idx[val_idx_local]
        test_idx = temp_idx[test_idx_local]

        return train_idx, val_idx, test_idx

    except ImportError:
        print("Warning: sklearn not available, using random split (not stratified)")
        np.random.seed(seed)
        indices = np.random.permutation(len(labels))
        n_train = int(len(labels) * ratios[0])
        n_val = int(len(labels) * ratios[1])
        return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def compute_class_weights(labels: np.ndarray, num_classes: int, max_weight: float = 10.0) -> np.ndarray:
    """Compute inverse-frequency class weights with cap.

    Args:
        labels: Array of class labels.
        num_classes: Number of classes.
        max_weight: Maximum weight cap.

    Returns:
        Array of class weights.
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero

    # Inverse frequency
    weights = len(labels) / (num_classes * counts)

    # Cap at max_weight
    weights = np.minimum(weights, max_weight)

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    return weights


def main():
    """Main preprocessing function."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    graphs_dir = data_dir / "graphs"
    labels_path = data_dir / "labels.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    if not labels_path.exists():
        print(f"Error: {labels_path} not found. Run convert_steps.py first.")
        sys.exit(1)

    with open(labels_path, "r") as f:
        labels_dict = json.load(f)
    print(f"Loaded labels for {len(labels_dict)} models")

    # Get graph files that have labels
    graph_files = []
    graph_labels = []
    for graph_path in sorted(graphs_dir.glob("*.json")):
        model_id = graph_path.stem
        if model_id in labels_dict:
            graph_files.append(graph_path)
            graph_labels.append(labels_dict[model_id])

    print(f"Found {len(graph_files)} graph files with labels")

    # Process all samples
    print("Preprocessing samples...")
    samples = []
    for graph_path, label in tqdm(zip(graph_files, graph_labels), total=len(graph_files)):
        try:
            sample = preprocess_sample(
                graph_path=graph_path,
                label=label,
                compute_descriptors=args.compute_descriptors,
                num_spatial=args.num_spatial,
                d2_bins=args.d2_bins,
                angle_bins=args.angle_bins,
            )
            samples.append(sample)
        except Exception as e:
            print(f"Error processing {graph_path.name}: {e}")
            continue

    print(f"Successfully preprocessed {len(samples)} samples")

    # Extract labels array
    all_labels = np.array([s["label"] for s in samples])

    # Stratified split
    ratios = [float(r) for r in args.split_ratio.split(",")]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6

    train_idx, val_idx, test_idx = stratified_split(all_labels, ratios, args.seed)

    splits = {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }

    # Compute class weights from training set
    train_labels = all_labels[train_idx]
    class_weights = compute_class_weights(train_labels, num_classes=8, max_weight=args.max_weight)

    # Print split statistics
    print("\nSplit statistics:")
    for split_name, split_samples in splits.items():
        split_labels = np.array([s["label"] for s in split_samples])
        counts = np.bincount(split_labels.astype(int), minlength=8)
        print(f"  {split_name}: {len(split_samples)} samples, class distribution: {counts.tolist()}")

    print(f"\nClass weights: {class_weights.tolist()}")

    # Save preprocessed data
    for split_name, split_samples in splits.items():
        output_path = output_dir / f"{split_name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(split_samples, f)
        print(f"Saved {split_name} split ({len(split_samples)} samples) to {output_path}")

    # Load existing metadata if available
    src_metadata_path = data_dir / "metadata.json"
    if src_metadata_path.exists():
        with open(src_metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Update metadata
    metadata.update({
        "num_classes": 8,
        "num_samples": len(samples),
        "compute_descriptors": args.compute_descriptors,
        "num_spatial": args.num_spatial,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "class_weights": class_weights.tolist(),
        "class_counts": np.bincount(all_labels.astype(int), minlength=8).tolist(),
        "class_names": [
            "Elbow - Weld Fitting",
            "Elbow - Pipe End Fitting",
            "Elbow - Socket Fitting",
            "Tee - Weld Fitting",
            "Tee - Pipe End Fitting",
            "Tee - Socket Fitting",
            "Elbow - Miscellaneous",
            "Tee - Miscellaneous",
        ],
    })

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_dir / 'metadata.json'}")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
