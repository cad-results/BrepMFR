#!/usr/bin/env python3
"""Training script for BrepFormer using preprocessed data.

This script uses preprocessed pickle files for efficient multi-worker data loading.

Usage:
    # First preprocess the data:
    python brepformer/preprocess.py --data_dir brepformer/data/mftrcad --output_dir brepformer/data/mftrcad_processed

    # Then train:
    python brepformer/train_preprocessed.py --data_dir brepformer/data/mftrcad_processed --max_epochs 100
"""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf  # numpy 2.0 compat for pytorch_lightning 1.x

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
from brepformer.data.preprocessed_dataset import PreprocessedDataset
from brepformer.data.collator import BrepCollator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BrepFormer with preprocessed data")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed train.pkl, val.pkl, test.pkl files",
    )

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes (auto-detect from metadata if None)")
    parser.add_argument("--multi_label", action="store_true", default=True, help="Use multi-label classification")
    parser.add_argument("--no_multi_label", action="store_false", dest="multi_label", help="Use single-label classification")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--ffn_dim", type=int, default=512, help="FFN dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="Number of KV heads for GQA")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")

    # Face segmentation arguments
    parser.add_argument("--face_segmentation", action="store_true", default=False,
                        help="Enable face-level segmentation head")
    parser.add_argument("--face_seg_weight", type=float, default=1.0,
                        help="Loss weight for face segmentation")
    parser.add_argument("--model_cls_weight", type=float, default=1.0,
                        help="Loss weight for model classification")
    parser.add_argument("--num_face_classes", type=int, default=27,
                        help="Number of face-level classes")
    parser.add_argument("--face_seg_hidden_dim", type=int, default=512,
                        help="Hidden dimension for face segmentation MLP")
    parser.add_argument("--face_seg_dropout", type=float, default=0.3,
                        help="Dropout for face segmentation MLP")
    parser.add_argument("--weighted_crossentropy", action="store_true", default=False,
                        help="Use inverse-frequency weighted CrossEntropyLoss for face segmentation")

    # Architecture arguments
    parser.add_argument("--use_rope", action="store_true", default=False,
                        help="Enable Rotary Position Embeddings (disabled by default for graph nodes)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to reduce GPU memory (enabled by default)")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Disable gradient checkpointing")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-detected from GPU memory if not set)")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup steps")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers (0 = main process only, saves RAM)")
    parser.add_argument("--accumulate_grad_batches", type=int, default=None, help="Gradient accumulation steps (auto-detected if batch_size not set)")
    parser.add_argument("--max_faces", type=int, default=500, help="Max faces per sample; filters extreme outliers by file size")
    parser.add_argument("--lr_monitor", type=str, default="train/loss_epoch",
                        help="Metric ReduceLROnPlateau watches. Default is the historical "
                             "train/loss_epoch, which decays LR on training loss and so "
                             "keeps cutting it while the model overfits. Prefer val/loss "
                             "or val/face_f1.")
    parser.add_argument("--early_stop_monitor", type=str, default="val/loss",
                        help="Metric EarlyStopping watches. val/loss is the historical "
                             "default, but on a pseudo-labelled corpus it mixes ground "
                             "truth with teacher noise; val/face_f1 stops on the axis "
                             "the model is actually deployed for.")
    parser.add_argument("--early_stop_patience", type=int, default=20,
                        help="EarlyStopping patience in epochs")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--exp_name", type=str, default="brepformer", help="Experiment name")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", type=int, default=32, help="Training precision (16 or 32)")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development test")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Warm-start from a checkpoint's WEIGHTS only, on a fresh "
                             "optimizer and LR schedule. Layers whose name or shape "
                             "does not match (typically the classifier heads when the "
                             "class count differs) are left at their initial values. "
                             "Use this to continue a trained model on a new corpus; use "
                             "--resume_from only to restart an interrupted run.")
    parser.add_argument("--attr_layout", default="onehot", choices=["onehot", "categorical"],
                        help="How the model decodes face_attr/edge_attr columns. 'onehot' is "
                             "the layout actually written to disk and is correct. "
                             "'categorical' reproduces the historical indexing, under which "
                             "surface type collapses to an is-it-a-plane bit, face area becomes "
                             "an is-it-a-cylinder bit, edge length reads the convexity flag and "
                             "edge convexity is a constant; use it only to reproduce a pre-fix "
                             "run. NOTE: weights trained under one layout are not valid under "
                             "the other -- the incumbent checkpoint drops 82.95%% -> 32.75%% "
                             "face accuracy if decoded the other way -- so --init_weights "
                             "across layouts warm-starts the body but forces the attribute "
                             "encoders to re-adapt.")
    parser.add_argument("--mask_policy", default="asis",
                        choices=["asis", "const", "zero"],
                        help="Overwrite face_grid channel 6, the UV validity mask, with a constant. 'asis' keeps the pickle's mask. This is candidate fix C for Bug 8: channel 6 is identically 1 on every STEP-derived corpus and 44.5%% masked in MFTRCAD pretraining, so the encoder learns it as a validity gate on one corpus and is handed a constant on the other. Rather than repairing the mask (--mftrcad_uv in steps_to_graphs), 'const'/'zero' delete the channel's information on BOTH sides, needing no re-preprocessing. Use it on train AND eval or the shift is merely moved.")
    parser.add_argument("--rotate_aug", default="none", choices=["none", "octahedral", "so3"],
                        help="Rotation augmentation on the TRAIN split only. The corpus is "
                             "axis-aligned and the pipeline had no augmentation of any kind, so "
                             "a model can key on world-frame normals; so3_probe measured `hole` "
                             "face F1 falling 0.971 -> 0.120 in the worst frame, and that worst "
                             "frame was axis-aligned. 'octahedral' draws from the 24 axis-aligned "
                             "90-degree rotations, 'so3' uniformly from SO(3), which is how real "
                             "GrabCAD parts arrive. Val stays unrotated so the early-stop metric "
                             "remains comparable across runs.")
    parser.add_argument("--limit_data", type=int, default=None,
                        help="Limit total dataset to N samples (proportionally across splits). "
                             "Saves a manifest JSON for reproducible subsetting in test/analyze/infer.")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    pl.seed_everything(args.seed)

    # Load metadata
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata}")
        if args.num_classes is None:
            args.num_classes = metadata.get("num_classes", 27)
    else:
        if args.num_classes is None:
            args.num_classes = 27
        metadata = {}

    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve batch size: explicit flag overrides, otherwise auto-detect from GPU memory
    if args.batch_size is not None:
        actual_batch_size = args.batch_size
        accumulate = args.accumulate_grad_batches or 1
    else:
        accumulate = args.accumulate_grad_batches
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
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

    # Create config
    config = BrepClassifierConfig(
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        activation_dropout=args.dropout,
        num_classes=args.num_classes,
        multi_label=args.multi_label,
        batch_size=actual_batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip_val=args.gradient_clip_val,
        face_segmentation=args.face_segmentation,
        face_seg_weight=args.face_seg_weight,
        model_cls_weight=args.model_cls_weight,
        num_face_classes=args.num_face_classes,
        face_seg_hidden_dim=args.face_seg_hidden_dim,
        face_seg_dropout=args.face_seg_dropout,
        use_rope=args.use_rope,
        # The RoPE frequency table must cover the longest sequence the model can see, which
        # is max_faces + 1 (the extra slot is the graph token). The default 512 is smaller
        # than the max_faces=1000 every run in SCRIPTS.md uses, so --use_rope would have
        # crashed on the first part with >511 faces. Size the table from max_faces instead
        # of leaving a constant that silently disagrees with the data.
        max_seq_len=max(512, args.max_faces + 1),
        gradient_checkpointing=args.gradient_checkpointing,
        lr_monitor=args.lr_monitor,
        attr_layout=args.attr_layout,
    )

    # Create datasets (max_faces filters extreme outliers by file size)
    print("Loading preprocessed training dataset...")
    train_dataset = PreprocessedDataset(str(data_dir), split="train", max_faces=args.max_faces,
                                        rotate_aug=args.rotate_aug, rotate_seed=args.seed,
                                        mask_policy=args.mask_policy)
    if args.rotate_aug != "none":
        print(f"Rotation augmentation: {args.rotate_aug} (train split only)")
    if args.mask_policy != "asis":
        # On val too, unlike rotate_aug: this is an input-distribution change, not an
        # augmentation, and val has to see the same inputs the model will be deployed on.
        print(f"UV validity mask (face_grid ch6) forced to {args.mask_policy} on train AND val")

    print("Loading preprocessed validation dataset...")
    # Deliberately NOT augmented: rotating val changes what val/face_f1 measures, and the
    # early-stop monitor has to stay comparable to every run in SCRIPTS.md. Measure
    # rotation robustness after the fact with `python -m brepformer.so3_probe`.
    val_dataset = PreprocessedDataset(str(data_dir), split="val", max_faces=args.max_faces,
                                      mask_policy=args.mask_policy)

    # Apply limit_data subsetting if requested
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
    # max_faces on collator caps padding so one large sample doesn't blow up the batch
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
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Compute inverse-frequency class weights for face segmentation (only with --weighted_crossentropy)
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
        else:
            import pickle as _pkl
            for f in train_dataset._sample_files:
                with open(f, "rb") as fh:
                    sample = _pkl.load(fh)
                fl = sample.get("face_labels")
                if fl is not None:
                    for c in fl:
                        if 0 <= c < args.num_face_classes:
                            face_class_counts[c] += 1
                del sample, fl
            gc.collect()
        total = face_class_counts.sum()
        if total > 0:
            freq = np.maximum(face_class_counts, 1.0) / total
            weights = 1.0 / (freq * args.num_face_classes)
            config.face_class_weights = weights.tolist()
            print(f"Face class weights: min={weights.min():.3f}, max={weights.max():.3f}")
    elif args.face_segmentation:
        print("Using unweighted CrossEntropyLoss for face segmentation")

    # Create model
    print("Creating model...")
    model = BrepClassifier(config)

    if args.init_weights:
        # Weights-only warm start. Deliberately NOT --resume_from: that restores the
        # optimizer state, epoch counter and LR schedule, so continuing a 58-epoch run
        # would start at epoch 58 and stop immediately against --max_epochs. Here we
        # want the learned representation on a fresh schedule over a different corpus.
        ckpt = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        src = ckpt.get("state_dict", ckpt)
        own = model.state_dict()
        loaded, skipped = [], []
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        model.load_state_dict(own)
        print(f"init_weights {args.init_weights}: loaded {len(loaded)} tensors, "
              f"skipped {len(skipped)} (shape/name mismatch)")
        if skipped:
            # Heads are the expected mismatch when the class count differs; anything
            # else means the checkpoint is from a different architecture and the run
            # is quietly training a mostly-random model.
            heads = [k for k in skipped if "classifier" in k or "head" in k]
            print(f"  skipped heads: {len(heads)}; other: {len(skipped) - len(heads)}")
            for k in skipped[:8]:
                print(f"    {k}")

    # Callbacks
    class GarbageCollectionCallback(pl.Callback):
        """Periodic gc to bound RAM without the overhead of per-batch gc.collect()."""
        def __init__(self, every_n_batches: int = 50):
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
    ]
    if args.face_segmentation:
        # val/f1 above is the MODEL-level multi-label F1, not per-face segmentation
        # quality — the two rank epochs differently, so selecting on val/f1 alone can
        # hand back a checkpoint that is worse at the task the model is used for
        # (defeature_v2 consumes face_preds, not model_classes). Keep a separate
        # best-face-* checkpoint so face segmentation can be compared across runs
        # without re-scoring every epoch.
        callbacks.append(ModelCheckpoint(
            dirpath=output_dir,
            filename="bestface-{epoch:02d}-{val/face_f1:.4f}",
            monitor="val/face_f1",
            mode="max",
            save_top_k=3,
        ))
    callbacks += [
        EarlyStopping(
            monitor=args.early_stop_monitor,
            patience=args.early_stop_patience,
            mode="max" if args.early_stop_monitor.rsplit("/", 1)[-1]
            in ("f1", "face_f1", "acc", "face_acc") else "min",
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
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=accumulate,
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
