#!/usr/bin/env python3
"""Test script for BrepFormer B-rep classification.

Usage:
    python brepformer/test.py --data_dir brepformer/data/mftrcad \
        --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import pathlib
import torch
torch.serialization.add_safe_globals([pathlib.PosixPath])
from brepformer.configs.config import BrepClassifierConfig
torch.serialization.add_safe_globals([BrepClassifierConfig])
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from brepformer.models.brep_classifier import BrepClassifier
from brepformer.data.dataset import MTFRCADDataset
from brepformer.data.collator import BrepCollator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test BrepFormer for B-rep classification")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing graphs/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=None,
        help="Path to test split file",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Path to external labels JSON file (optional)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=27, help="Number of classes")
    parser.add_argument("--multi_label", action="store_true", default=True, help="Use multi-label classification")
    parser.add_argument("--no_multi_label", action="store_false", dest="multi_label", help="Use single-label classification")

    # Other arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for results")

    return parser.parse_args()


def main():
    """Main test function."""
    args = parse_args()

    # Data paths
    data_dir = Path(args.data_dir)
    test_split = args.test_split or str(data_dir / "test.txt")

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = BrepClassifier.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Get config from model
    config = model.config

    # Create test dataset
    print("Loading test dataset...")
    test_dataset = MTFRCADDataset(
        data_dir=str(data_dir),
        split="test",
        split_file=test_split if Path(test_split).exists() else None,
        label_file=args.label_file,
        num_classes=config.num_classes,
        multi_label=config.multi_label,
    )

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

    # Trainer for testing
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )

    # Run test
    print("Running evaluation...")
    results = trainer.test(model, dataloaders=test_loader)

    # Print results
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for key, value in results[0].items():
        print(f"  {key}: {value:.4f}")

    # Compute per-class metrics if single-label
    if not config.multi_label:
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
    """Compute per-class precision, recall, F1.

    Args:
        model: Trained model.
        dataloader: Test data loader.
        num_classes: Number of classes.
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            logits = model(batch)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(batch["label"].cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Compute confusion matrix
    cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    conf_matrix = cm(all_preds, all_targets)

    # Compute per-class metrics
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\nPer-class metrics:")
    print("-" * 50)
    print(f"{'Class':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)
    for i in range(num_classes):
        print(f"{i:>10} {precision[i].item():>10.4f} {recall[i].item():>10.4f} {f1[i].item():>10.4f}")
    print("-" * 50)
    print(f"{'Macro Avg':>10} {precision.mean().item():>10.4f} {recall.mean().item():>10.4f} {f1.mean().item():>10.4f}")


if __name__ == "__main__":
    main()
