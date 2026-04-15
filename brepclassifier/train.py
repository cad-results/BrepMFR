#!/usr/bin/env python3
"""Training script for pipe fitting classifier.

Uses preprocessed pickle data with BrepFormer encoder + GAT head.

Usage:
    # Basic training
    python brepclassifier/train.py \
        --data_dir brepclassifier/data/ssdata1_processed \
        --max_epochs 300

    # With pretrained encoder (main brepformer checkpoint, 27-class model-only)
    python brepclassifier/train.py \
        --data_dir brepclassifier/data/ssdata1_processed \
        --pretrained_encoder "results/brepformer/best-epoch=99-val/f1=0.8466.ckpt" \
        --max_epochs 300

    # With pretrained encoder (face seg heavy checkpoint, 27-class + face seg)
    python brepclassifier/train.py \
        --data_dir brepclassifier/data/ssdata1_processed \
        --pretrained_encoder "results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt" \
        --max_epochs 300

    # Frozen encoder (train only GAT head)
    python brepclassifier/train.py \
        --data_dir brepclassifier/data/ssdata1_processed \
        --pretrained_encoder "results/brepformer/best-epoch=99-val/f1=0.8466.ckpt" \
        --freeze_encoder \
        --max_epochs 300
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
# Compatibility shim: older pytorch_lightning uses np.Inf, removed in NumPy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from brepclassifier.configs.config import PipeFittingConfig
from brepclassifier.models.pipe_classifier import PipeFittingClassifier
from brepclassifier.data.preprocessed_dataset import PreprocessedDataset
from brepformer.data.collator import BrepCollator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train pipe fitting classifier")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Preprocessed data directory")

    # Pretrained weights
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to pretrained brepformer checkpoint")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder parameters")
    parser.add_argument("--classifier_ckpt", type=str, default=None,
                        help="Resume from brepclassifier checkpoint")

    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    # GAT arguments
    parser.add_argument("--gat_num_layers", type=int, default=3)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--gat_hidden_dim", type=int, default=256)
    parser.add_argument("--gat_dropout", type=float, default=0.3)

    # Dense head arguments
    parser.add_argument("--dense_dropout", type=float, default=0.3)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--encoder_lr_factor", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/pipe_classifier")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--max_faces", type=int, default=0,
                        help="Max faces per sample (0=no limit)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from Lightning checkpoint")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    pl.seed_everything(args.seed)

    # Load metadata
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.json"
    metadata = {}
    class_weights = None

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {json.dumps(metadata, indent=2)}")

        if "class_weights" in metadata:
            class_weights = metadata["class_weights"]
            print(f"Using class weights from metadata: {class_weights}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = PipeFittingConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        activation_dropout=args.dropout,
        num_classes=8,
        multi_label=False,
        gat_num_layers=args.gat_num_layers,
        gat_heads=args.gat_heads,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_dropout=args.gat_dropout,
        dense_dropout=args.dense_dropout,
        pretrained_encoder_ckpt=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        classifier_ckpt=args.classifier_ckpt,
        learning_rate=args.learning_rate,
        encoder_lr_factor=args.encoder_lr_factor,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epochs,
        class_weights=class_weights,
    )

    # Create datasets
    print("Loading preprocessed training dataset...")
    train_dataset = PreprocessedDataset(str(data_dir), split="train", max_faces=args.max_faces)

    print("Loading preprocessed validation dataset...")
    val_dataset = PreprocessedDataset(str(data_dir), split="val", max_faces=args.max_faces)

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
    model = PipeFittingClassifier(config)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    gat_params = sum(p.numel() for p in model.gat_head.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"GAT head parameters: {gat_params:,}")

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
            monitor="val/f1",
            patience=30,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir="results",
        name="pipe_classifier",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
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
