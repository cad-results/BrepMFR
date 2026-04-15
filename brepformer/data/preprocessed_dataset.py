"""Dataset for preprocessed MFTRCAD data.

Loads preprocessed pickle files created by preprocess.py for efficient
multi-worker data loading during training.
"""

import gc
import json
import os
import pickle
import random as _random
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """Dataset for preprocessed B-rep data stored in pickle files.

    This dataset is designed for efficient multi-worker loading by keeping
    all data in memory after loading from pickle.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        max_faces: int = None,
    ):
        """Initialize PreprocessedDataset.

        Args:
            data_path: Path to directory containing {split}/ subdirs or {split}.pkl files,
                       or path to a specific .pkl file.
            split: Dataset split ('train', 'val', 'test'). Ignored if data_path points to a .pkl file.
            transform: Optional transform to apply to samples.
            max_faces: If set, skip samples that likely exceed this face count
                       (estimated from file size to avoid loading huge files).
        """
        self.transform = transform
        self.samples = None
        self._sample_files = None

        data_path = Path(data_path)
        if data_path.suffix == ".pkl":
            # Direct .pkl file path (legacy)
            pkl_path = data_path
            if not pkl_path.exists():
                raise FileNotFoundError(f"Preprocessed data file not found: {pkl_path}")
            print(f"Loading preprocessed data from {pkl_path}...")
            with open(pkl_path, "rb") as f:
                self.samples = pickle.load(f)
            gc.collect()
            if max_faces is not None:
                before = len(self.samples)
                self.samples = [s for s in self.samples if s["num_faces"] <= max_faces]
                filtered = before - len(self.samples)
                if filtered > 0:
                    print(f"Filtered {filtered} samples with >{max_faces} faces")
            print(f"Loaded {len(self.samples)} samples")
        else:
            split_dir = data_path / split
            pkl_path = data_path / f"{split}.pkl"

            if split_dir.is_dir():
                # Per-sample files: load lazily from split directory
                self._sample_files = sorted(split_dir.glob("*.pkl"))
                if not self._sample_files:
                    raise FileNotFoundError(f"No .pkl files found in {split_dir}")
                total = len(self._sample_files)

                if max_faces is not None:
                    # Filter by file size to avoid loading huge samples.
                    # d2+angle distance dominate: ~2*(N+1)^2*64*4 bytes.
                    # Use 3x safety margin to avoid false positives.
                    max_bytes = 3 * (max_faces + 1) ** 2 * 64 * 4
                    self._sample_files = [
                        f for f in self._sample_files
                        if os.path.getsize(f) <= max_bytes
                    ]
                    filtered = total - len(self._sample_files)
                    if filtered > 0:
                        print(f"Filtered {filtered}/{total} samples exceeding ~{max_faces} faces (by file size)")

                print(f"Found {len(self._sample_files)} preprocessed samples in {split_dir}")
            elif pkl_path.exists():
                # Legacy single .pkl file
                print(f"Loading preprocessed data from {pkl_path}...")
                with open(pkl_path, "rb") as f:
                    self.samples = pickle.load(f)
                gc.collect()
                if max_faces is not None:
                    before = len(self.samples)
                    self.samples = [s for s in self.samples if s["num_faces"] <= max_faces]
                    filtered = before - len(self.samples)
                    if filtered > 0:
                        print(f"Filtered {filtered} samples with >{max_faces} faces")
                print(f"Loaded {len(self.samples)} samples")
            else:
                raise FileNotFoundError(
                    f"Preprocessed data not found: looked for {split_dir}/ and {pkl_path}"
                )

    def __len__(self) -> int:
        """Return number of samples."""
        if self._sample_files is not None:
            return len(self._sample_files)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing preprocessed tensors.
        """
        if self._sample_files is not None:
            with open(self._sample_files[idx], "rb") as f:
                sample = pickle.load(f)
        else:
            sample = self.samples[idx]

        # Convert numpy arrays to tensors
        output = {
            "face_grid": torch.from_numpy(sample["face_grid"]),
            "face_attr": torch.from_numpy(sample["face_attr"]),
            "edge_index": torch.from_numpy(sample["edge_index"]),
            "edge_attr": torch.from_numpy(sample["edge_attr"]),
            "edge_grid": torch.from_numpy(sample["edge_grid"]),
            "spatial_pos": torch.from_numpy(sample["spatial_pos"]),
            "in_degree": torch.from_numpy(sample["in_degree"]),
            "label": torch.from_numpy(sample["label"]) if sample["label"].ndim > 0 else torch.tensor(sample["label"]),
            "model_id": sample["model_id"],
            "num_faces": sample["num_faces"],
            "num_edges": sample["num_edges"],
        }

        # Optional features
        if "d2_distance" in sample:
            output["d2_distance"] = torch.from_numpy(sample["d2_distance"])
        if "angle_distance" in sample:
            output["angle_distance"] = torch.from_numpy(sample["angle_distance"])
        if "face_labels" in sample:
            output["face_labels"] = torch.from_numpy(sample["face_labels"])

        if self.transform is not None:
            output = self.transform(output)

        return output


def create_limit_data_manifest(data_dir: str, limit_data: int, seed: int, output_path: str) -> dict:
    """Create a manifest for reproducible dataset subsetting.

    Proportionally selects samples from train/val/test splits and saves
    the selection to a JSON manifest. Model IDs are extracted from the
    selected pickles so downstream scripts (e.g. infer.py) can filter
    by model_id without re-loading pickles.

    Args:
        data_dir: Preprocessed data directory with train/val/test subdirs.
        limit_data: Total number of samples to select across all splits.
        seed: Random seed for reproducibility.
        output_path: Path to write the manifest JSON.

    Returns:
        The manifest dict.
    """
    data_dir = Path(data_dir)
    rng = _random.Random(seed)

    # Collect filenames per split
    split_files = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.is_dir():
            split_files[split] = sorted([f.name for f in split_dir.glob("*.pkl")])
        else:
            split_files[split] = []

    total = sum(len(v) for v in split_files.values())
    if total == 0:
        raise ValueError(f"No samples found in {data_dir}")

    # Proportional selection per split
    selected = {}
    remaining = limit_data
    splits = [s for s in ["train", "val", "test"] if split_files[s]]

    for i, split in enumerate(splits):
        if i == len(splits) - 1:
            n = remaining
        else:
            n = round(limit_data * len(split_files[split]) / total)
        n = min(n, len(split_files[split]))
        n = max(n, 0)
        remaining -= n

        chosen = sorted(rng.sample(split_files[split], n))
        selected[split] = chosen

    # Extract model_ids from selected pickles
    all_model_ids = []
    for split_name, filenames in selected.items():
        split_dir = data_dir / split_name
        for fname in filenames:
            with open(split_dir / fname, "rb") as f:
                sample = pickle.load(f)
            all_model_ids.append(sample["model_id"])
            del sample

    manifest = {
        "limit_data": limit_data,
        "seed": seed,
        "data_dir": str(data_dir),
        "splits": selected,
        "model_ids": sorted(all_model_ids),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Limit data manifest saved: {out}")
    for split_name, files in selected.items():
        print(f"  {split_name}: {len(files)} samples")

    return manifest


def apply_limit_data_manifest(dataset: "PreprocessedDataset", manifest: dict, split: str):
    """Filter a PreprocessedDataset to only include samples listed in a manifest.

    Args:
        dataset: PreprocessedDataset instance to filter in-place.
        manifest: Manifest dict (loaded from JSON).
        split: Split name ("train", "val", "test").
    """
    allowed: Set[str] = set(manifest["splits"].get(split, []))
    if not allowed:
        print(f"Limit data: no entries for split '{split}' in manifest")
        return

    if dataset._sample_files is not None:
        before = len(dataset._sample_files)
        dataset._sample_files = [f for f in dataset._sample_files if f.name in allowed]
        print(f"Limit data: {split} {before} -> {len(dataset._sample_files)} samples")
    elif dataset.samples is not None:
        model_ids = set(manifest.get("model_ids", []))
        before = len(dataset.samples)
        dataset.samples = [s for s in dataset.samples if s.get("model_id") in model_ids]
        print(f"Limit data: {split} {before} -> {len(dataset.samples)} samples")


def load_limit_data_manifest(manifest_path: str) -> dict:
    """Load a limit_data manifest from JSON file."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_preprocessed_datasets(
    data_dir: str,
    splits: List[str] = None,
) -> Dict[str, PreprocessedDataset]:
    """Load preprocessed datasets for specified splits.

    Args:
        data_dir: Directory containing preprocessed pickle files.
        splits: List of splits to load (default: ['train', 'val', 'test']).

    Returns:
        Dictionary mapping split names to datasets.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    datasets = {}
    for split in splits:
        split_dir = Path(data_dir) / split
        pkl_path = Path(data_dir) / f"{split}.pkl"
        if split_dir.is_dir() or pkl_path.exists():
            datasets[split] = PreprocessedDataset(data_dir, split=split)
        else:
            print(f"Warning: {split_dir}/ and {pkl_path} not found, skipping {split} split")

    return datasets
