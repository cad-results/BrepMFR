# -*- coding: utf-8 -*-
"""
Training wrapper for BrepMFR with compatibility fixes for newer NumPy and PyTorch Lightning.

This wrapper addresses:
1. NumPy np.int/np.float deprecation (removed in NumPy 1.24+)
2. PyTorch Lightning API changes (deprecated lifecycle methods and trainer args)

Usage:
    Training:
        python train_wrapper.py train --dataset_path /path/to/dataset --max_epochs 100

    Testing:
        python train_wrapper.py test --dataset_path /path/to/dataset --checkpoint path/to/best.ckpt
"""

# =============================================================================
# NumPy compatibility shim - MUST be first before any other imports
# =============================================================================
import warnings
import numpy as np

# Suppress deprecation warnings for np.int, np.float, np.bool in NumPy 1.20-1.23
# and add aliases for NumPy 1.24+ where they were removed
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    try:
        # Test if np.int works (will fail on NumPy 1.24+)
        _ = np.int
    except AttributeError:
        np.int = np.int64
    try:
        _ = np.float
    except AttributeError:
        np.float = np.float64
    try:
        _ = np.bool
    except AttributeError:
        np.bool = np.bool_

# =============================================================================
# Standard imports
# =============================================================================
import argparse
import pathlib
import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# =============================================================================
# Import original modules (after NumPy patching)
# =============================================================================
from data.dataset import CADSynth
from models.brepseg_model import BrepSeg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# =============================================================================
# Compatible BrepSeg class with updated PyTorch Lightning API
# =============================================================================
class CompatibleBrepSeg(BrepSeg):
    """
    BrepSeg subclass with PyTorch Lightning 2.x compatible lifecycle methods.

    Changes from parent class:
    - training_epoch_end -> on_train_epoch_end
    - validation_epoch_end -> on_validation_epoch_end
    - test_epoch_end -> on_test_epoch_end
    - optimizer_step signature updated (removed deprecated params)
    """

    def on_train_epoch_end(self):
        """Called at the end of training epoch (replaces training_epoch_end)."""
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch (replaces validation_epoch_end)."""
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []
        per_face_comp = (preds_np == labels_np).astype(np.int64)
        self.log("per_face_accuracy", np.mean(per_face_comp))

    def on_test_epoch_end(self):
        """Called at the end of test epoch (replaces test_epoch_end)."""
        print("num_classes: %s" % self.num_classes)
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []

        per_face_comp = (preds_np == labels_np).astype(np.int64)
        self.log("per_face_accuracy", np.mean(per_face_comp))
        print("per_face_accuracy: %s" % np.mean(per_face_comp))

        # Per-class accuracy
        per_class_acc = []
        for i in range(0, self.num_classes):
            class_pos = np.where(labels_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = preds_np[class_pos]
                class_i_label = labels_np[class_pos]
                per_face_comp = (class_i_preds == class_i_label).astype(np.int64)
                per_class_acc.append(np.mean(per_face_comp))
                print("class_%s_acc: %s" % (i + 1, np.mean(per_face_comp)))
        self.log("per_class_accuracy", np.mean(per_class_acc))
        print("per_class_accuracy: %s" % np.mean(per_class_acc))

        # IoU
        per_class_iou = []
        for i in range(0, self.num_classes):
            label_pos = np.where(labels_np == i)
            pred_pos = np.where(preds_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = preds_np[label_pos]
                class_i_label = labels_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int64)
                Union = (class_i_preds != class_i_label).astype(np.int64)
                class_i_preds_ = preds_np[pred_pos]
                class_i_label_ = labels_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int64)
                per_class_iou.append(
                    np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_))
                )
        self.log("IoU", np.mean(per_class_iou))
        print("IoU: %s" % np.mean(per_class_iou))

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx=0,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Custom optimizer step with learning rate warmup.

        Compatible with PyTorch Lightning 1.x signature.
        """
        # Update params
        optimizer.step(closure=optimizer_closure)

        # Manually warm up lr without a scheduler
        if self.trainer.global_step < 5000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.001

    # Disable parent's deprecated methods to prevent warnings
    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass


# =============================================================================
# Argument parser (manual, without Trainer.add_argparse_args)
# =============================================================================
def create_parser():
    parser = argparse.ArgumentParser("BrepMFR Training Wrapper")
    parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
    parser.add_argument("--num_classes", type=int, default=25, help="Number of feature classes")
    parser.add_argument("--dataset", choices=("cadsynth",), default="cadsynth", help="Dataset to use")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers for dataloader (set to 0 on Windows)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to load weights from for testing",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="BrepMFR",
        help="Experiment name (folder inside ./results/)",
    )

    # Transformer parameters
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--act_dropout", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_node", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers_encode", type=int, default=8)

    # Trainer parameters (manual instead of Trainer.add_argparse_args)
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")

    return parser


# =============================================================================
# Dataloader factory (handles num_workers=0 compatibility and PL 1.7.x bug)
# =============================================================================
def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    Create a DataLoader with proper prefetch_factor handling.

    Uses standard DataLoader instead of DataLoaderX to avoid PyTorch Lightning
    1.7.x bug with DataLoader subclasses (_old_init AttributeError).

    When num_workers=0, prefetch_factor must be None (PyTorch requirement).
    """
    from torch.utils.data import DataLoader

    if num_workers > 0:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset._collate,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )
    else:
        # num_workers=0: prefetch_factor must be None
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset._collate,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
        )


# =============================================================================
# Trainer factory
# =============================================================================
def create_trainer(args, checkpoint_callback, logger):
    """
    Create PyTorch Lightning Trainer without deprecated arguments.

    Replaces:
    - Trainer.from_argparse_args() (deprecated)
    - auto_select_gpus=True (removed)
    """
    # Determine accelerator and devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.devices
    else:
        accelerator = "cpu"
        devices = 1

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        num_sanity_val_steps=0,  # Disable sanity check to avoid PL 1.7.x dataloader bug
    )

    return trainer


# =============================================================================
# Main function
# =============================================================================
def main():
    parser = create_parser()
    args = parser.parse_args()

    # Setup paths
    results_path = pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # Define checkpoint path based on date and time
    month_day = time.strftime("%m%d")
    hour_min_second = time.strftime("%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        monitor="eval_loss",
        dirpath=str(results_path.joinpath(month_day, hour_min_second)),
        filename="best",
        save_top_k=10,
        save_last=True,
    )

    logger = TensorBoardLogger(
        str(results_path),
        name=month_day,
        version=hour_min_second,
    )

    trainer = create_trainer(args, checkpoint_callback, logger)

    # Select dataset
    if args.dataset == "cadsynth":
        Dataset = CADSynth
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.traintest == "train":
        print(
            f"""
-----------------------------------------------------------------------------------
B-rep model feature recognition (Compatibility Wrapper)
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
        """
        )

        model = CompatibleBrepSeg(args)

        train_data = Dataset(
            root_dir=args.dataset_path,
            split="train",
            random_rotate=True,
            num_class=args.num_classes,
        )
        val_data = Dataset(
            root_dir=args.dataset_path,
            split="val",
            random_rotate=False,
            num_class=args.num_classes,
        )

        train_loader = create_dataloader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = create_dataloader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        trainer.fit(model, train_loader, val_loader)

    else:
        # Test mode
        assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"

        test_data = Dataset(
            root_dir=args.dataset_path,
            split="test",
            random_rotate=False,
            num_class=args.num_classes,
        )
        test_loader = create_dataloader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Load checkpoint with CompatibleBrepSeg class
        model = CompatibleBrepSeg.load_from_checkpoint(args.checkpoint)
        trainer.test(model, dataloaders=[test_loader], ckpt_path=args.checkpoint, verbose=False)


if __name__ == "__main__":
    main()
