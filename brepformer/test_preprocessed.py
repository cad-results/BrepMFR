#!/usr/bin/env python3
"""Test script for BrepFormer using preprocessed data.

Usage:
    python brepformer/test_preprocessed.py --data_dir brepformer/data/mftrcad_processed \
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

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import numpy as np

from brepformer.models.brep_classifier import BrepClassifier
from brepformer.data.preprocessed_dataset import PreprocessedDataset
from brepformer.data.collator import BrepCollator
from brepformer.data.classes import (
    CLASS_TO_REAL_CLASS, REAL_CLASS_NAMES, REAL_NUM_CLASSES,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test BrepFormer with preprocessed data")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed test.pkl file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Other arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for results")
    parser.add_argument("--output_face_preds", type=str, default=None,
                        help="Output JSON file for per-face predictions (face segmentation mode)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--real_classes", action="store_true",
                        help="Remap 27 MFTRCAD classes to 8 real machining feature categories")
    parser.add_argument("--limit_data_manifest", type=str, default=None,
                        help="Path to limit_data_manifest.json for reproducible dataset subsetting")

    return parser.parse_args()


def main():
    """Main test function."""
    args = parse_args()

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    # Add safe globals for PyTorch 2.6+ compatibility with Lightning checkpoints
    import pathlib
    from brepformer.configs.config import BrepClassifierConfig
    torch.serialization.add_safe_globals([pathlib.PosixPath, BrepClassifierConfig])
    model = BrepClassifier.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Get config from model
    config = model.config

    # Create dataset for the requested split
    print(f"Loading preprocessed {args.split} dataset...")
    test_dataset = PreprocessedDataset(args.data_dir, split=args.split)

    if args.limit_data_manifest:
        from brepformer.data.preprocessed_dataset import (
            load_limit_data_manifest,
            apply_limit_data_manifest,
        )
        manifest = load_limit_data_manifest(args.limit_data_manifest)
        apply_limit_data_manifest(test_dataset, manifest, args.split)

    print(f"{args.split.capitalize()} samples: {len(test_dataset)}")

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
        compute_per_class_metrics(model, test_loader, config.num_classes,
                                  real_classes=args.real_classes)

    # Face segmentation per-face predictions
    if config.face_segmentation and args.output_face_preds:
        print("\nComputing per-face predictions...")
        compute_face_predictions(model, test_loader, config, args.output_face_preds,
                                 real_classes=args.real_classes)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"\nResults saved to {output_path}")


def compute_per_class_metrics(model, dataloader, num_classes, real_classes=False):
    """Compute per-class precision, recall, F1.

    Args:
        model: Trained model.
        dataloader: Test data loader.
        num_classes: Number of classes.
        real_classes: If True, remap predictions/targets to 8 real classes.
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

    if real_classes:
        remap = torch.tensor(CLASS_TO_REAL_CLASS, dtype=torch.long)
        all_preds = remap[all_preds]
        all_targets = remap[all_targets]
        num_classes = REAL_NUM_CLASSES
        class_names = REAL_CLASS_NAMES
    else:
        class_names = None

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

    label_str = "real classes" if real_classes else "classes"
    print(f"\nPer-class metrics ({label_str}):")
    print("-" * 60)
    print(f"{'Class':>25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for i in range(num_classes):
        name = class_names[i] if class_names else str(i)
        print(f"{name:>25} {precision[i].item():>10.4f} {recall[i].item():>10.4f} {f1[i].item():>10.4f}")
    print("-" * 60)
    print(f"{'Macro Avg':>25} {precision.mean().item():>10.4f} {recall.mean().item():>10.4f} {f1.mean().item():>10.4f}")


def compute_face_predictions(model, dataloader, config, output_path, real_classes=False):
    """Compute per-face predictions and metrics for face segmentation.

    Args:
        model: Trained model with face_segmentation=True.
        dataloader: Test data loader.
        config: Model config.
        output_path: Path to save JSON output.
        real_classes: If True, remap predictions/targets to 8 real classes.
    """
    device = next(model.parameters()).device
    model.eval()

    all_face_preds = []
    all_face_targets = []
    per_model_results = []

    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
            output = model(batch_on_device)
            face_logits = output["face_logits"]  # (B, N, C)
            face_probs = torch.softmax(face_logits, dim=-1)
            face_preds = face_logits.argmax(dim=-1)  # (B, N)
            face_targets = batch["face_labels"]  # (B, N) on CPU

            B = face_logits.size(0)
            for i in range(B):
                model_id = batch["model_ids"][i] if "model_ids" in batch else f"sample"
                n_faces = batch.get("num_faces", None)
                # Determine actual face count from attn_mask
                mask = batch["attn_mask"][i]  # (N+1,)
                n = mask.sum().item() - 1  # subtract CLS token

                fp = face_preds[i, :n].cpu().tolist()
                ft = face_targets[i, :n].tolist()
                probs = face_probs[i, :n].cpu().tolist()

                per_model_results.append({
                    "model_id": model_id,
                    "num_faces": n,
                    "face_preds": fp,
                    "face_targets": ft,
                    "face_probs": probs,
                })

                valid_mask = torch.tensor(ft) != -1
                if valid_mask.any():
                    all_face_preds.append(torch.tensor(fp)[valid_mask])
                    all_face_targets.append(torch.tensor(ft)[valid_mask])

    # Compute aggregate metrics
    if all_face_preds:
        all_fp = torch.cat(all_face_preds)
        all_ft = torch.cat(all_face_targets)

        if real_classes:
            remap = torch.tensor(CLASS_TO_REAL_CLASS, dtype=torch.long)
            all_fp = remap[all_fp]
            all_ft = remap[all_ft]
            num_face_classes = REAL_NUM_CLASSES
        else:
            num_face_classes = config.num_face_classes

        # Per-face accuracy
        face_acc = (all_fp == all_ft).float().mean().item()

        # Per-class IoU
        per_class_iou = []
        for c in range(num_face_classes):
            pred_c = all_fp == c
            true_c = all_ft == c
            intersection = (pred_c & true_c).sum().item()
            union = (pred_c | true_c).sum().item()
            iou = intersection / union if union > 0 else float('nan')
            per_class_iou.append(iou)

        valid_ious = [x for x in per_class_iou if x == x]  # filter NaN
        mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

        metrics = {
            "face_accuracy": face_acc,
            "mean_iou": mean_iou,
            "per_class_iou": per_class_iou,
        }

        print(f"\n--- Face Segmentation Metrics ---")
        print(f"  Per-face accuracy: {face_acc:.4f}")
        print(f"  Mean IoU: {mean_iou:.4f}")
    else:
        metrics = {}

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "predictions": per_model_results}, f, indent=2)
    print(f"  Face predictions saved to {output_path}")


if __name__ == "__main__":
    main()
