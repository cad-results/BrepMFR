#!/usr/bin/env python3
"""Fine-tune a pre-trained MFTRCAD (27-class) BrepFormer on the defeature (5-class) dataset.

Loads encoder weights from a pre-trained checkpoint and creates new 5-class
classifier heads, optionally warm-started by averaging the 27-class weights
according to the CLASS_TO_DEFEATURE mapping. Supports differential learning
rates (lower for encoder, higher for new heads) and optional encoder freezing.

Usage:
    # Fine-tune trial5 on defeature data
    python brepformer/fine_tune.py \
        --pretrained results/trial5/best-epoch=39-val/f1=0.8832.ckpt \
        --data_dir brepformer/data/defeature_processed \
        --exp_name finetune_v1

    # With encoder frozen for first 10 epochs
    python brepformer/fine_tune.py \
        --pretrained results/trial5/best-epoch=39-val/f1=0.8832.ckpt \
        --data_dir brepformer/data/defeature_processed \
        --freeze_encoder_epochs 10 \
        --exp_name finetune_frozen
"""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from brepformer.configs.config import BrepClassifierConfig
from brepformer.models.brep_classifier import BrepClassifier
from brepformer.data.preprocessed_dataset import PreprocessedDataset
from brepformer.data.collator import BrepCollator
from brepformer.data.classes import CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BrepFormer from MFTRCAD to defeature")

    # Pre-trained model
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pre-trained 27-class checkpoint")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed defeature data directory")

    # Target classes
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of target classes")
    parser.add_argument("--num_face_classes", type=int, default=5,
                        help="Number of target face classes")

    # Fine-tuning strategy
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0,
                        help="Freeze encoder for this many initial epochs (0 = never freeze)")
    parser.add_argument("--encoder_lr_factor", type=float, default=0.1,
                        help="Learning rate multiplier for encoder params (relative to head LR)")
    parser.add_argument("--no_warm_start", action="store_true",
                        help="Don't warm-start heads from 27-class weights (use random init)")

    # Face segmentation
    parser.add_argument("--face_segmentation", action="store_true", default=True,
                        help="Enable face segmentation (default: True for fine-tuning)")
    parser.add_argument("--no_face_segmentation", action="store_false", dest="face_segmentation")
    parser.add_argument("--face_seg_weight", type=float, default=1.0)
    parser.add_argument("--model_cls_weight", type=float, default=1.0)
    parser.add_argument("--face_seg_hidden_dim", type=int, default=512)
    parser.add_argument("--face_seg_dropout", type=float, default=0.3)
    parser.add_argument("--weighted_crossentropy", action="store_true", default=False,
                        help="Use inverse-frequency weighted CrossEntropyLoss for face segmentation")

    # Training
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto-detected if not set)")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate for classifier heads")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup steps (lower than pre-training)")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser.add_argument("--max_faces", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--exp_name", type=str, default="finetune")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--limit_data", type=int, default=None)

    return parser.parse_args()


def warm_start_linear(old_weight, old_bias, mapping, num_new_classes):
    """Create new linear layer weights by averaging old weights per mapping.

    Args:
        old_weight: (old_classes, in_features) weight matrix.
        old_bias: (old_classes,) bias vector, or None.
        mapping: list of length old_classes, mapping[i] = new class ID.
        num_new_classes: number of new output classes.

    Returns:
        (new_weight, new_bias) tensors.
    """
    in_features = old_weight.shape[1]
    new_weight = torch.zeros(num_new_classes, in_features)
    new_bias = torch.zeros(num_new_classes) if old_bias is not None else None
    counts = torch.zeros(num_new_classes)

    for old_cls, new_cls in enumerate(mapping):
        if old_cls < old_weight.shape[0]:
            new_weight[new_cls] += old_weight[old_cls]
            if old_bias is not None:
                new_bias[new_cls] += old_bias[old_cls]
            counts[new_cls] += 1

    # Average
    for c in range(num_new_classes):
        if counts[c] > 0:
            new_weight[c] /= counts[c]
            if new_bias is not None:
                new_bias[c] /= counts[c]

    return new_weight, new_bias


def transfer_weights(pretrained_ckpt_path, new_model, warm_start=True):
    """Load pre-trained weights into the new model, remapping classifier heads.

    Transfers:
    - Encoder weights (exact copy)
    - Pooling weights (exact copy)
    - Classifier head: reusable layers copied, final layer warm-started
    - Face classifier: reusable layers copied, final layer warm-started

    Args:
        pretrained_ckpt_path: Path to the pre-trained checkpoint.
        new_model: New BrepClassifier with 5-class heads.
        warm_start: If True, warm-start final layers from 27→5 mapping.
    """
    ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
    old_state = ckpt["state_dict"]

    new_state = new_model.state_dict()
    transferred = 0
    skipped = 0
    warm_started = 0

    for key, param in old_state.items():
        if key in new_state and new_state[key].shape == param.shape:
            new_state[key] = param
            transferred += 1
        elif key in new_state:
            # Shape mismatch — this is a classifier output layer
            skipped += 1
        else:
            skipped += 1

    # Warm-start the final classifier layers
    if warm_start:
        mapping = CLASS_TO_DEFEATURE
        num_new = DEFEATURE_NUM_CLASSES

        # Model-level classifier: last Linear layer
        # NonLinearClassifier.classifier has layers: [0-11], final is [12] (Linear 256→27)
        old_w_key = "classifier.classifier.12.weight"
        old_b_key = "classifier.classifier.12.bias"
        if old_w_key in old_state:
            w, b = warm_start_linear(
                old_state[old_w_key], old_state.get(old_b_key),
                mapping, num_new,
            )
            new_state[old_w_key.replace("12", "12")] = w  # same key since structure matches
            if b is not None:
                new_state[old_b_key.replace("12", "12")] = b
            warm_started += 1
            print(f"  Warm-started model classifier final layer: {old_w_key}")

        # Face classifier: fc3 (Linear 512→27)
        old_fw_key = "face_classifier.fc3.weight"
        old_fb_key = "face_classifier.fc3.bias"
        if old_fw_key in old_state and "face_classifier.fc3.weight" in new_state:
            w, b = warm_start_linear(
                old_state[old_fw_key], old_state.get(old_fb_key),
                mapping, num_new,
            )
            new_state["face_classifier.fc3.weight"] = w
            if b is not None:
                new_state["face_classifier.fc3.bias"] = b
            warm_started += 1
            print(f"  Warm-started face classifier final layer: {old_fw_key}")

    new_model.load_state_dict(new_state, strict=False)
    print(f"Weight transfer: {transferred} copied, {skipped} skipped, {warm_started} warm-started")


class EncoderUnfreezeCallback(Callback):
    """Unfreezes encoder parameters after a specified number of epochs."""

    def __init__(self, unfreeze_at_epoch: int):
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_at_epoch:
            for name, param in pl_module.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
            print(f"\n*** Encoder unfrozen at epoch {self.unfreeze_at_epoch} ***\n")


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # Load pre-trained checkpoint to get architecture config
    print(f"Loading pre-trained checkpoint: {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    old_hp = ckpt["hyper_parameters"]["config"]
    del ckpt
    gc.collect()

    # Load metadata from defeature data
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata}")
        if args.num_classes is None:
            args.num_classes = metadata.get("num_classes", 5)

    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve batch size
    if args.batch_size is not None:
        actual_batch_size = args.batch_size
        accumulate = args.accumulate_grad_batches or 1
    else:
        accumulate = args.accumulate_grad_batches
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            if gpu_mem_mb <= 10240:
                actual_batch_size = 8
                accumulate = accumulate or 4
            elif gpu_mem_mb <= 16384:
                actual_batch_size = 16
                accumulate = accumulate or 2
            else:
                actual_batch_size = 32
                accumulate = accumulate or 1
            print(f"GPU has {gpu_mem_mb:.0f} MB — auto batch_size={actual_batch_size}, "
                  f"accumulate_grad_batches={accumulate}")
        else:
            actual_batch_size = 32
            accumulate = accumulate or 1

    # Create config: keep encoder architecture from pre-trained, change classification
    config = BrepClassifierConfig(
        # Encoder architecture — must match pre-trained
        hidden_dim=old_hp.hidden_dim,
        ffn_dim=old_hp.ffn_dim,
        num_layers=old_hp.num_layers,
        num_heads=old_hp.num_heads,
        num_kv_heads=old_hp.num_kv_heads,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        activation_dropout=args.dropout,
        # Classification — new target classes
        num_classes=args.num_classes,
        multi_label=True,
        batch_size=actual_batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip_val=args.gradient_clip_val,
        # Face segmentation — new target classes
        face_segmentation=args.face_segmentation,
        face_seg_weight=args.face_seg_weight,
        model_cls_weight=args.model_cls_weight,
        num_face_classes=args.num_face_classes,
        face_seg_hidden_dim=args.face_seg_hidden_dim,
        face_seg_dropout=args.face_seg_dropout,
        use_rope=old_hp.use_rope,
        gradient_checkpointing=old_hp.gradient_checkpointing,
    )

    # Create datasets
    print("Loading preprocessed training dataset...")
    train_dataset = PreprocessedDataset(str(data_dir), split="train", max_faces=args.max_faces)

    print("Loading preprocessed validation dataset...")
    val_dataset = PreprocessedDataset(str(data_dir), split="val", max_faces=args.max_faces)

    if args.limit_data:
        from brepformer.data.preprocessed_dataset import (
            create_limit_data_manifest,
            apply_limit_data_manifest,
        )
        manifest_path = output_dir / "limit_data_manifest.json"
        manifest = create_limit_data_manifest(
            data_dir=str(data_dir),
            limit_data=args.limit_data,
            seed=args.seed,
            output_path=str(manifest_path),
        )
        apply_limit_data_manifest(train_dataset, manifest, "train")
        apply_limit_data_manifest(val_dataset, manifest, "val")

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    gc.collect()

    # Create data loaders
    collator = BrepCollator(max_faces=args.max_faces)
    loader_kwargs = dict(
        collate_fn=collator,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = False

    train_loader = DataLoader(
        train_dataset, batch_size=actual_batch_size,
        shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=actual_batch_size,
        shuffle=False, **loader_kwargs,
    )

    # Compute face class weights if requested
    if args.face_segmentation and args.weighted_crossentropy:
        print("Computing face class weights from training data...")
        face_class_counts = np.zeros(args.num_face_classes, dtype=np.float64)
        if train_dataset.samples is not None:
            for sample in train_dataset.samples:
                fl = sample.get("face_labels")
                if fl is not None:
                    for c in fl:
                        if 0 <= c < args.num_face_classes:
                            face_class_counts[c] += 1
        total = face_class_counts.sum()
        if total > 0:
            freq = np.maximum(face_class_counts, 1.0) / total
            weights = 1.0 / (freq * args.num_face_classes)
            config.face_class_weights = weights.tolist()
            print(f"Face class weights: {dict(zip(range(args.num_face_classes), weights.round(3)))}")

    # Create new model with 5-class heads
    print("Creating fine-tune model...")
    model = BrepClassifier(config)

    # Transfer pre-trained weights
    print("Transferring pre-trained weights...")
    transfer_weights(args.pretrained, model, warm_start=not args.no_warm_start)

    # Freeze encoder if requested
    if args.freeze_encoder_epochs > 0:
        print(f"Freezing encoder for {args.freeze_encoder_epochs} epochs")
        for name, param in model.named_parameters():
            if name.startswith("encoder.") or name.startswith("pooling."):
                param.requires_grad = False

    # Override configure_optimizers for differential learning rates
    original_configure = model.configure_optimizers

    def configure_optimizers_finetune():
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder.") or name.startswith("pooling."):
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": config.learning_rate * args.encoder_lr_factor,
                "name": "encoder",
            })
        param_groups.append({
            "params": head_params,
            "lr": config.learning_rate,
            "name": "heads",
        })

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=config.lr_factor, patience=config.lr_patience,
            min_lr=config.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            },
        }

    model.configure_optimizers = configure_optimizers_finetune

    # Callbacks
    class GarbageCollectionCallback(Callback):
        def __init__(self, every_n_batches=50):
            self._every = every_n_batches
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if (batch_idx + 1) % self._every == 0:
                gc.collect()
        def on_train_epoch_end(self, trainer, pl_module):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        def on_validation_epoch_end(self, trainer, pl_module):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    callbacks = [
        GarbageCollectionCallback(),
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
    if args.freeze_encoder_epochs > 0:
        callbacks.append(EncoderUnfreezeCallback(args.freeze_encoder_epochs))

    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.exp_name,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=accumulate,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )

    # Save fine-tuning config for reproducibility
    ft_config = {
        "pretrained": args.pretrained,
        "data_dir": args.data_dir,
        "num_classes": args.num_classes,
        "num_face_classes": args.num_face_classes,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "encoder_lr_factor": args.encoder_lr_factor,
        "learning_rate": args.learning_rate,
        "warm_start": not args.no_warm_start,
        "max_epochs": args.max_epochs,
        "batch_size": actual_batch_size,
    }
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(ft_config, f, indent=2)

    print("Starting fine-tuning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Fine-tuning complete! Best model saved to {output_dir}")


if __name__ == "__main__":
    main()
