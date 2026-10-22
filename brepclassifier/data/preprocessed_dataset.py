"""Dataset for preprocessed pipe fitting data.

Loads preprocessed pickle files created by preprocess.py for efficient
multi-worker data loading during training.

Identical to brepformer version except label is always torch.long scalar.
"""

import pickle
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """Dataset for preprocessed B-rep data stored in pickle files."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        max_faces: int = 0,
    ):
        """Initialize PreprocessedDataset.

        Args:
            data_path: Path to directory containing {split}.pkl files,
                       or path to a specific .pkl file.
            split: Dataset split ('train', 'val', 'test').
            transform: Optional transform to apply to samples.
            max_faces: Maximum number of faces per sample (0 = no limit).
        """
        self.transform = transform

        data_path = Path(data_path)
        if data_path.suffix == ".pkl":
            pkl_path = data_path
        else:
            pkl_path = data_path / f"{split}.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(f"Preprocessed data file not found: {pkl_path}")

        print(f"Loading preprocessed data from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self.samples = pickle.load(f)

        if max_faces > 0:
            before = len(self.samples)
            self.samples = [s for s in self.samples if s["num_faces"] <= max_faces]
            print(f"Filtered {before - len(self.samples)} samples with >{max_faces} faces")

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns dictionary with tensors. Label is always torch.long scalar
        (single-label classification).
        """
        sample = self.samples[idx]

        output = {
            "face_grid": torch.from_numpy(sample["face_grid"]),
            "face_attr": torch.from_numpy(sample["face_attr"]),
            "edge_index": torch.from_numpy(sample["edge_index"]),
            "edge_attr": torch.from_numpy(sample["edge_attr"]),
            "edge_grid": torch.from_numpy(sample["edge_grid"]),
            "spatial_pos": torch.from_numpy(sample["spatial_pos"]),
            "in_degree": torch.from_numpy(sample["in_degree"]),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "model_id": sample["model_id"],
            "num_faces": sample["num_faces"],
            "num_edges": sample["num_edges"],
        }

        if "d2_distance" in sample:
            output["d2_distance"] = torch.from_numpy(sample["d2_distance"])
        if "angle_distance" in sample:
            output["angle_distance"] = torch.from_numpy(sample["angle_distance"])

        if self.transform is not None:
            output = self.transform(output)

        return output


def load_preprocessed_datasets(
    data_dir: str,
    splits: List[str] = None,
) -> Dict[str, PreprocessedDataset]:
    """Load preprocessed datasets for specified splits."""
    if splits is None:
        splits = ["train", "val", "test"]

    datasets = {}
    for split in splits:
        pkl_path = Path(data_dir) / f"{split}.pkl"
        if pkl_path.exists():
            datasets[split] = PreprocessedDataset(data_dir, split=split)
        else:
            print(f"Warning: {pkl_path} not found, skipping {split} split")

    return datasets
