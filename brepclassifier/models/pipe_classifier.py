"""PipeFittingClassifier: Full model for pipe fitting classification.

PyTorch Lightning module combining:
- BrepEncoder (from brepformer, optionally pretrained)
- GATClassificationHead for graph-level classification
- Training/validation/test logic with metrics
"""

import pathlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall

from brepformer.models.brep_encoder import BrepEncoder
from brepclassifier.configs.config import PipeFittingConfig
from brepclassifier.models.gat_head import GATClassificationHead


class PipeFittingClassifier(pl.LightningModule):
    """PipeFittingClassifier: BrepEncoder + GAT for pipe fitting classification.

    Two-stage architecture:
    1. BrepEncoder (reused from brepformer, optionally pretrained)
    2. GATClassificationHead (new, trained from scratch)
    """

    def __init__(self, config: PipeFittingConfig):
        """Initialize PipeFittingClassifier.

        Args:
            config: PipeFittingConfig with all hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Build encoder using brepformer's BrepEncoder
        encoder_config = config.to_encoder_config()
        self.encoder = BrepEncoder(encoder_config)

        # GAT classification head
        self.gat_head = GATClassificationHead(
            in_dim=config.hidden_dim,
            num_classes=config.num_classes,
            gat_num_layers=config.gat_num_layers,
            gat_heads=config.gat_heads,
            gat_hidden_dim=config.gat_hidden_dim,
            gat_v2=config.gat_v2,
            gat_dropout=config.gat_dropout,
            gat_pooling=config.gat_pooling,
            dense_dims=config.dense_dims,
            dense_dropout=config.dense_dropout,
        )

        # Load pretrained encoder weights if specified
        if config.pretrained_encoder_ckpt:
            self._load_pretrained_encoder(config.pretrained_encoder_ckpt)

        # Freeze encoder if specified
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Loss function with class weights
        if config.class_weights is not None:
            weight = torch.tensor(config.class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self._setup_metrics()

        # For warmup scheduling
        self.warmup_steps = config.warmup_steps
        self.current_step = 0

    def _load_pretrained_encoder(self, ckpt_path: str):
        """Load pretrained encoder weights from a brepformer checkpoint.

        Extracts 'encoder.*' keys from the checkpoint state_dict.

        Args:
            ckpt_path: Path to brepformer checkpoint (.ckpt).
        """
        print(f"Loading pretrained encoder from {ckpt_path}...")

        # Add safe globals for loading
        from brepformer.configs.config import BrepClassifierConfig
        torch.serialization.add_safe_globals([pathlib.PosixPath, BrepClassifierConfig])

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract encoder keys
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder."):]] = value

        if not encoder_state:
            print(f"Warning: No encoder.* keys found in {ckpt_path}")
            return

        # Load with strict=False to handle minor mismatches
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"  Loaded {len(encoder_state) - len(unexpected)} encoder parameters")

    def _setup_metrics(self):
        """Setup metrics for training/validation/test."""
        num_classes = self.config.num_classes

        # Training metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Validation metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

        # Test metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: Dictionary with BrepFormer batch format.

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Run encoder
        node_emb, graph_emb = self.encoder(
            face_grid=batch["face_grid"],
            face_attr=batch["face_attr"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            edge_grid=batch["edge_grid"],
            spatial_pos=batch["spatial_pos"],
            in_degree=batch.get("in_degree"),
            edge_path=batch.get("edge_path"),
            d2_distance=batch.get("d2_distance"),
            angle_distance=batch.get("angle_distance"),
            attn_mask=batch.get("attn_mask"),
        )

        # GAT head: takes node_emb (B, N+1, D), edge_index, attn_mask
        logits = self.gat_head(
            face_emb=node_emb,
            edge_index=batch["edge_index"],
            attn_mask=batch.get("attn_mask"),
        )

        return logits

    def _compute_loss_and_preds(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Compute loss and predictions."""
        logits = self(batch)
        targets = batch["label"]
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=-1)
        return loss, preds, targets

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, targets = self._compute_loss_and_preds(batch)

        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)

        self.current_step += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, preds, targets = self._compute_loss_and_preds(batch)

        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, targets = self._compute_loss_and_preds(batch)

        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with differential learning rates."""
        # Differential LR: encoder at lower rate, GAT head at full rate
        encoder_params = list(self.encoder.parameters())
        gat_params = list(self.gat_head.parameters())

        param_groups = [
            {
                "params": [p for p in encoder_params if p.requires_grad],
                "lr": self.config.learning_rate * self.config.encoder_lr_factor,
                "name": "encoder",
            },
            {
                "params": gat_params,
                "lr": self.config.learning_rate,
                "name": "gat_head",
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.min_lr,
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

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int = 0,
        optimizer_closure=None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        """Custom optimizer step with warmup."""
        if self.current_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.current_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                if pg.get("name") == "encoder":
                    pg["lr"] = lr_scale * self.config.learning_rate * self.config.encoder_lr_factor
                else:
                    pg["lr"] = lr_scale * self.config.learning_rate

        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_end(self):
        """Reset training metrics."""
        self.train_acc.reset()
        self.train_f1.reset()
