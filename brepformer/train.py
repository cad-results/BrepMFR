#!/usr/bin/env python3
"""Training script for BrepFormer B-rep classification.

Usage:
    python brepformer/train.py --data_dir brepformer/data/mftrcad --max_epochs 100

With external labels:
    python brepformer/train.py --data_dir brepformer/data/mftrcad \
        --label_file path/to/labels.json --num_classes 10
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf  # numpy 2.0 compat for pytorch_lightning 1.x

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from brepformer.configs.config import BrepClassifierConfig
from brepformer.models.brep_classifier import BrepClassifier
from brepformer.data.dataset import MTFRCADDataset, create_data_splits
from brepformer.data.collator import BrepCollator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BrepFormer for B-rep classification")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing graphs/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Path to external labels JSON file (optional)",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Path to train split file (optional, will create if not exists)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help="Path to validation split file",
    )

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=27, help="Number of classes")
    parser.add_argument("--multi_label", action="store_true", default=True, help="Use multi-label classification")
    parser.add_argument("--no_multi_label", action="store_false", dest="multi_label", help="Use single-label classification")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV heads for GQA")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup steps")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--exp_name", type=str, default="brepformer", help="Experiment name")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="32", help="Training precision (16 or 32)")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development test")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--precompute_features", action="store_true", default=False, help="Precompute D2/angle descriptors (slower loading)")
    parser.add_argument("--no_precompute_features", action="store_false", dest="precompute_features", help="Skip precomputing D2/angle descriptors")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    pl.seed_everything(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data splits if needed
    data_dir = Path(args.data_dir)
    train_split = args.train_split or str(data_dir / "train.txt")
    val_split = args.val_split or str(data_dir / "val.txt")

    if not os.path.exists(train_split):
        print("Creating data splits...")
        create_data_splits(
            data_dir=str(data_dir),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=args.seed,
            output_dir=str(data_dir),
        )

    # Create config
    config = BrepClassifierConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        activation_dropout=args.dropout,
        num_classes=args.num_classes,
        multi_label=args.multi_label and args.label_file is None,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip_val=args.gradient_clip_val,
        data_dir=str(data_dir),
        label_file=args.label_file,
    )

    # Create datasets
    print("Loading training dataset...")
    train_dataset = MTFRCADDataset(
        data_dir=str(data_dir),
        split="train",
        split_file=train_split if os.path.exists(train_split) else None,
        label_file=args.label_file,
        num_classes=args.num_classes,
        multi_label=config.multi_label,
        precompute_features=args.precompute_features,
    )

    print("Loading validation dataset...")
    val_dataset = MTFRCADDataset(
        data_dir=str(data_dir),
        split="val",
        split_file=val_split if os.path.exists(val_split) else None,
        label_file=args.label_file,
        num_classes=args.num_classes,
        multi_label=config.multi_label,
        precompute_features=args.precompute_features,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create data loaders
    collator = BrepCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    model = BrepClassifier(config)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-{val/f1:.4f}",
            monitor="val/f1",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=20,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.exp_name,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        precision=int(args.precision),
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )

    # Train
    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )

    print(f"Training complete! Best model saved to {output_dir}")


if __name__ == "__main__":
    main()
