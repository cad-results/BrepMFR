#!/usr/bin/env python3
"""Test script for pipe fitting classifier.

Usage:
    python brepclassifier/test.py \
        --data_dir brepclassifier/data/ssdata1_processed \
        --checkpoint results/pipe_classifier/best.ckpt
"""

import argparse
import json
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
# Compatibility shim: older pytorch_lightning uses np.Inf, removed in NumPy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from brepclassifier.models.pipe_classifier import PipeFittingClassifier
from brepclassifier.data.preprocessed_dataset import PreprocessedDataset
from brepformer.data.collator import BrepCollator


CLASS_NAMES = [
    "Elbow - Weld Fitting",
    "Elbow - Pipe End Fitting",
    "Elbow - Socket Fitting",
    "Tee - Weld Fitting",
    "Tee - Pipe End Fitting",
    "Tee - Socket Fitting",
    "Elbow - Miscellaneous",
    "Tee - Miscellaneous",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test pipe fitting classifier")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed data directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_faces", type=int, default=0,
                        help="Max faces per sample (0=no limit). Use 500 to avoid OOM on large models.")
    parser.add_argument("--output_file", type=str, default=None)

    return parser.parse_args()


def main():
    """Main test function."""
    args = parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    from brepformer.configs.config import BrepClassifierConfig
    from brepclassifier.configs.config import PipeFittingConfig
    torch.serialization.add_safe_globals([
        pathlib.PosixPath, BrepClassifierConfig, PipeFittingConfig,
    ])
    model = PipeFittingClassifier.load_from_checkpoint(args.checkpoint)
    model.eval()

    config = model.config

    # Load test dataset
    print("Loading preprocessed test dataset...")
    test_dataset = PreprocessedDataset(args.data_dir, split="test", max_faces=args.max_faces)
    print(f"Test samples: {len(test_dataset)}")

    # Create data loader
    collator = BrepCollator()
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Run test via trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )

    print("Running evaluation...")
    results = trainer.test(model, dataloaders=test_loader)

    # Print results
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for key, value in results[0].items():
        print(f"  {key}: {value:.4f}")

    # Per-class metrics
    print("\nComputing per-class metrics...")
    compute_per_class_metrics(model, test_loader, config.num_classes)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"\nResults saved to {output_path}")


def compute_per_class_metrics(model, dataloader, num_classes):
    """Compute and print per-class precision, recall, F1."""
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            logits = model(batch)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(batch["label"].cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Confusion matrix
    cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    conf_matrix = cm(all_preds, all_targets)

    # Per-class metrics
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    support = conf_matrix.sum(dim=1)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\nPer-class metrics:")
    print("-" * 70)
    print(f"{'Class':>3} {'Name':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 70)
    for i in range(num_classes):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"{i:>3} {name:<30} {precision[i].item():>10.4f} {recall[i].item():>10.4f} "
              f"{f1[i].item():>10.4f} {int(support[i].item()):>8}")
    print("-" * 70)
    print(f"{'':>3} {'Macro Average':<30} {precision.mean().item():>10.4f} "
          f"{recall.mean().item():>10.4f} {f1.mean().item():>10.4f}")


if __name__ == "__main__":
    main()
