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
        "--descriptors_ignore_mask",
        action="store_true",
        help="Sample the D2 and angle descriptors from EVERY UV grid point, ignoring the "
             "validity mask in face_grid channel 6. Exists to isolate one facet of Bug 8: "
             "the descriptors are built from `mask > 0.5` points, so fixing the mask changes "
             "the attention-bias descriptors as well as the SurfaceEncoder input, and a "
             "single corpus cannot tell those two apart. Building one corpus with this flag "
             "reproduces the legacy descriptor content on top of a fixed mask, so the pair "
             "attributes the effect. Legacy corpora had an all-ones mask, which is exactly "
             "what this flag emulates -- so it is a no-op on them.",
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
        "--no_group_split", action="store_true",
        help="Disable group-aware splitting. By default models sharing a "
             "'<slug>__' prefix are kept within one split, so parametric variants "
             "of one part cannot leak from train into val and inflate val F1. "
             "Grouping is already a no-op for mftrcad ids (they have no '__').",
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


def _numpy_group_split(groups: np.ndarray, ratios: List[float], seed: int):
    """Split by whole group into train/val/test without sklearn.

    Assigns each group entirely to one split, filling train, then val, then test
    to their target *model* counts (groups vary in size, so counting models
    rather than groups keeps the ratios honest).

    This exists because the outer fallback in the caller used to be a plain
    random split over models, which would leak parametric variants of one part
    across splits — the exact failure the group logic is there to prevent. A
    missing optional dependency must not silently reintroduce it.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_total = len(groups)
    targets = [ratios[0] * n_total, (ratios[0] + ratios[1]) * n_total]

    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    train, val, test = [], [], []
    running = 0
    for g in uniq:
        members = idx_by_group[g]
        bucket = train if running < targets[0] else (val if running < targets[1] else test)
        bucket.extend(members.tolist())
        running += len(members)
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


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
        label_data = _read_label_payload(label_file)

        if label_data is None:
            return multi_hot
        if "cls" in label_data:
            for face_id, class_id in label_data["cls"].items():
                if 0 <= class_id < num_classes:
                    multi_hot[class_id] = 1.0
        elif "labels" in label_data:
            # Same two-schema problem as load_face_labels; -1 masked faces
            # contribute nothing, so a fully masked model gets an all-zero
            # target, matching what the old code produced for every model.
            for class_id in label_data["labels"]:
                if 0 <= class_id < num_classes:
                    multi_hot[class_id] = 1.0

    return multi_hot


def _read_label_payload(label_file: Path) -> Optional[Dict]:
    """Read a label file, tolerating both on-disk label schemas.

    Two writers produce label JSONs and they do not agree:

    - mftrcad / defeature ground truth is a *dict* with a ``"cls"`` map of
      face-index (as a string) -> class id.
    - the pseudo-label writer (``data/merge_suitable_samples.py``, and
      ``data/arbitrate_labels.py`` after it) emits a *list*,
      ``[model_id, {"labels": [...], "num_faces": N, "source": ...}]``, whose
      ``labels`` is dense per-face with -1 for masked faces.

    Reading only the first form is how an entire pseudo-labelled corpus
    preprocessed to all -1: on a list, ``"cls" in label_data`` is a membership
    test over the elements, which is False, so every face silently kept its
    initial -1 and the corpus became zero-gradient padding with no error.
    """
    with open(label_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
        return data[1]
    return None


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

    label_data = _read_label_payload(label_file)
    if label_data is None:
        return face_labels

    if "cls" in label_data:
        for face_id_str, class_id in label_data["cls"].items():
            face_id = int(face_id_str)
            if 0 <= face_id < num_faces and 0 <= class_id < num_classes:
                face_labels[face_id] = class_id
    elif "labels" in label_data:
        labels = label_data["labels"]
        # A length mismatch means the labels were fitted against a different
        # tessellation than the graph being preprocessed, so every face index
        # would be off. Fail loudly: the caller catches this per-sample and
        # prints it, which is far better than writing a plausible-looking
        # pickle whose labels are shifted.
        if len(labels) != num_faces:
            raise ValueError(
                f"{label_file.name}: {len(labels)} labels for a {num_faces}-face "
                "graph — labels and graph disagree, refusing to guess an alignment"
            )
        for face_id, class_id in enumerate(labels):
            # Out-of-range (including the -1 masked sentinel) stays -1, which is
            # the ignore_index the face loss and every face metric already use.
            if 0 <= class_id < num_classes:
                face_labels[face_id] = class_id

    return face_labels


def _descriptor_grids(face_grid, ignore_mask: bool):
    """The grids the D2/angle descriptors should sample from.

    ``compute_d2_descriptor`` and ``compute_angle_descriptor`` select points with
    ``face_grid[i, 6] > 0.5``, so the validity mask decides which geometry the attention-bias
    descriptors are built from -- not just what the SurfaceEncoder sees. Fixing the mask
    therefore moves two things at once. Forcing channel 6 to 1 here reproduces the legacy
    descriptor content on top of a fixed mask, which is the only way to attribute the two
    apart. Copies rather than mutating: the returned array is handed to the descriptor code
    only, and the real mask must still reach the pickle.
    """
    if not ignore_mask:
        return face_grid
    g = np.array(face_grid, copy=True)
    g[:, 6, :, :] = 1.0
    return g



def preprocess_sample(
    graph_path: Path,
    labels_dir: Path,
    external_labels: Optional[Dict],
    num_classes: int,
    compute_descriptors: bool,
    num_spatial: int,
    d2_bins: int,
    angle_bins: int,
    descriptors_ignore_mask: bool = False,
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
        descriptors_ignore_mask: Sample descriptors from all UV points, not just valid ones.

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
    # Centroids live at 11:14 post-normalisation. Cols 2:5 are the cone/sphere/torus one-hot
    # bits, which are zero on most parts -- so the pairwise max came out at 1e-6 and 99.2% of
    # face pairs got an ALL-ZERO D2 histogram (measured on defeature_processed/test/00000).
    # The point clouds D2 actually samples come from face_grid and were always correct; only
    # the max_dist derived from these centroids was wrong, which is why the whole channel
    # fell outside the histogram range.
    face_centroids = data["face_attr"][:, 11:14] if data["face_attr"].shape[1] >= 14 else None
    if data["face_grid"].shape[1] >= 6:
        face_normals = data["face_grid"][:, 3:6, :, :].mean(axis=(2, 3))
    else:
        face_normals = None

    graph_features = precompute_graph_features(
        edge_index=data["edge_index"],
        num_nodes=data["num_nodes"],
        face_centroids=face_centroids if compute_descriptors else None,
        face_normals=face_normals if compute_descriptors else None,
        face_grids=_descriptor_grids(data["face_grid"], descriptors_ignore_mask)
        if compute_descriptors else None,
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

    # Group key per model, so correlated variants of one part cannot straddle the
    # split. Pseudo-labelled GrabCAD samples are named "<slug>__<part>", and a slug
    # is one uploader's parametric family — the corpus contains e.g. one clamp in 12
    # sizes. Splitting those at random leaks near-duplicates from train into val,
    # which *raises* validation F1 while real generalisation gets worse: the worst
    # kind of bug, because it hides itself behind a better-looking number.
    #
    # mftrcad ids ("20240116_231044_1009") contain no "__", so each becomes its own
    # group and grouping is a no-op for them — safe to leave enabled by default.
    groups = np.array([p.stem.split("__")[0] for p in graph_files])
    n_groups = len(set(groups.tolist()))
    if not args.no_group_split and n_groups < num_files:
        print(f"Group-aware split: {num_files} models in {n_groups} groups "
              f"(largest group {max(np.bincount(np.unique(groups, return_inverse=True)[1]))})")

    # Stratified split
    split_assignment = {}
    try:
        if args.no_group_split or n_groups >= num_files:
            # Every model is its own group; plain stratification is already correct.
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
        else:
            # Group-aware: no group appears in more than one split. Stratification is
            # kept where possible (StratifiedGroupKFold honours both), falling back to
            # GroupShuffleSplit if class counts are too thin to stratify.
            val_ratio = ratios[1] / (ratios[1] + ratios[2])
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                def _grouped(idx, frac_out, seed):
                    """Split idx into (keep, out) with ~frac_out held out, by group."""
                    n_splits = max(2, int(round(1.0 / max(frac_out, 1e-9))))
                    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                               random_state=seed)
                    keep, out = next(sgk.split(idx, dominant_classes[idx], groups[idx]))
                    return idx[keep], idx[out]

                all_idx = np.arange(num_files)
                train_idx, valtest_idx = _grouped(all_idx, ratios[1] + ratios[2], args.seed)
                val_idx, test_idx = _grouped(valtest_idx, 1.0 - val_ratio, args.seed + 1)
            except (ImportError, ValueError):
                # No sklearn (or stratification impossible). Fall back to a
                # dependency-free split that is still group-aware: losing
                # stratification costs some class balance, but losing grouping
                # would silently inflate val F1, which is far worse. sklearn is
                # genuinely absent in some envs here, so this path is real.
                train_idx, val_idx, test_idx = _numpy_group_split(
                    groups, ratios, args.seed)

            # Assert the invariant rather than trust it: a silent leak here is
            # exactly the failure this code exists to prevent.
            gtr, gva, gte = (set(groups[i].tolist()) for i in (train_idx, val_idx, test_idx))
            leaks = (gtr & gva) | (gtr & gte) | (gva & gte)
            if leaks:
                raise RuntimeError(
                    f"group leak across splits ({len(leaks)} groups, e.g. "
                    f"{sorted(leaks)[:3]}) — refusing to write a corpus whose "
                    "validation score would be inflated"
                )
            print(f"  train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} "
                  f"models; groups {len(gtr)}/{len(gva)}/{len(gte)}, no overlap")

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
                    descriptors_ignore_mask=args.descriptors_ignore_mask,
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
        "descriptors_ignore_mask": args.descriptors_ignore_mask,
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
