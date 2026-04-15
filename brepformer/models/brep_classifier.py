"""BrepClassifier: Full model for B-rep classification.

PyTorch Lightning module combining:
- BrepEncoder for feature extraction
- GraphPooling for graph-level representation
- NonLinearClassifier for classification
- Training/validation/test logic with metrics
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall

from brepformer.configs.config import BrepClassifierConfig
from brepformer.models.brep_encoder import BrepEncoder
from brepformer.models.pooling import GraphPooling
from brepformer.models.layers.blocks import NonLinearClassifier, FaceSegmentationClassifier


class BrepClassifier(pl.LightningModule):
    """BrepClassifier: Full model for B-rep whole-model classification.

    Combines BrepEncoder, pooling, and classification head into a
    PyTorch Lightning module with training logic.
    """

    def __init__(self, config: BrepClassifierConfig):
        """Initialize BrepClassifier.

        Args:
            config: Configuration object with all hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Encoder
        self.encoder = BrepEncoder(config)

        # Pooling
        self.pooling = GraphPooling(
            hidden_dim=config.hidden_dim,
            pooling_type="cls",  # Use [CLS] token
        )

        # Classification head
        self.classifier = NonLinearClassifier(
            in_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

        # Loss function
        if config.multi_label:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Face segmentation head (conditional)
        self.face_segmentation = config.face_segmentation
        if self.face_segmentation:
            self.face_classifier = FaceSegmentationClassifier(
                in_dim=config.hidden_dim,
                num_classes=config.num_face_classes,
                hidden_dim=config.face_seg_hidden_dim,
                dropout=config.face_seg_dropout,
            )
            face_weight_tensor = None
            if config.face_class_weights is not None:
                face_weight_tensor = torch.tensor(config.face_class_weights, dtype=torch.float32)
            self.face_loss_fn = nn.CrossEntropyLoss(
                weight=face_weight_tensor, ignore_index=-1,
            )
            self.model_cls_weight = config.model_cls_weight
            self.face_seg_weight = config.face_seg_weight

        # Metrics
        self._setup_metrics()

        # For warmup scheduling
        self.warmup_steps = config.warmup_steps
        self.current_step = 0

    def _setup_metrics(self):
        """Setup metrics for training/validation/test."""
        num_classes = self.config.num_classes
        task = "multilabel" if self.config.multi_label else "multiclass"

        # Training metrics
        self.train_acc = Accuracy(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
        )
        self.train_f1 = F1Score(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
            average="macro",
        )

        # Validation metrics
        self.val_acc = Accuracy(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
        )
        self.val_f1 = F1Score(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
            average="macro",
        )
        self.val_precision = Precision(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
            average="macro",
        )
        self.val_recall = Recall(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
            average="macro",
        )

        # Test metrics
        self.test_acc = Accuracy(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
        )
        self.test_f1 = F1Score(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
            average="macro",
        )
        self.test_precision = Precision(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
            average="macro",
        )
        self.test_recall = Recall(
            task=task,
            num_labels=num_classes if self.config.multi_label else None,
            num_classes=None if self.config.multi_label else num_classes,
            average="macro",
        )

        # Face segmentation metrics
        if self.face_segmentation:
            num_face_classes = self.config.num_face_classes
            self.train_face_acc = Accuracy(
                task="multiclass", num_classes=num_face_classes, ignore_index=-1,
            )
            self.train_face_f1 = F1Score(
                task="multiclass", num_classes=num_face_classes, average="macro", ignore_index=-1,
            )
            self.val_face_acc = Accuracy(
                task="multiclass", num_classes=num_face_classes, ignore_index=-1,
            )
            self.val_face_f1 = F1Score(
                task="multiclass", num_classes=num_face_classes, average="macro", ignore_index=-1,
            )
            self.test_face_acc = Accuracy(
                task="multiclass", num_classes=num_face_classes, ignore_index=-1,
            )
            self.test_face_f1 = F1Score(
                task="multiclass", num_classes=num_face_classes, average="macro", ignore_index=-1,
            )

    def forward(self, batch: Dict[str, torch.Tensor]):
        """Forward pass.

        Args:
            batch: Dictionary containing:
                - face_grid: (batch, num_faces, C, H, W)
                - face_attr: (batch, num_faces, attr_dim)
                - edge_index: (batch, 2, num_edges)
                - edge_attr: (batch, num_edges, attr_dim)
                - edge_grid: (batch, num_edges, C, L)
                - spatial_pos: (batch, N+1, N+1)
                - in_degree: (batch, num_faces) [optional]
                - edge_path: (batch, N+1, N+1, max_dist) [optional]
                - d2_distance: (batch, N+1, N+1, d2_dim) [optional]
                - angle_distance: (batch, N+1, N+1, ang_dim) [optional]
                - attn_mask: (batch, num_faces + 1) [optional]

        Returns:
            Dict with "model_logits" and optionally "face_logits",
            or just model logits tensor when face_segmentation is disabled.
        """
        # Extract features
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

        # Pool to graph level (using [CLS] token)
        graph_repr = self.pooling(node_emb, mask=batch.get("attn_mask"))

        # Model-level classification
        model_logits = self.classifier(graph_repr)

        if not self.face_segmentation:
            return model_logits

        # Face-level segmentation from per-face embeddings (positions 1..N)
        face_emb = node_emb[:, 1:, :]  # (B, N_faces, hidden_dim)
        face_logits = self.face_classifier(face_emb)  # (B, N_faces, num_face_classes)

        return {"model_logits": model_logits, "face_logits": face_logits}

    def _compute_loss_and_preds(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple:
        """Compute loss and predictions.

        Args:
            batch: Input batch dictionary.

        Returns:
            Tuple of (loss, preds, targets) when face_segmentation is disabled.
            Tuple of (loss, preds, targets, face_preds, face_targets) when enabled.
        """
        output = self(batch)
        targets = batch["label"]

        if not self.face_segmentation:
            logits = output
            loss = self.loss_fn(logits, targets.float() if self.config.multi_label else targets)
            if self.config.multi_label:
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                preds = logits.argmax(dim=-1)
            return loss, preds, targets

        # Face segmentation mode: combined loss
        model_logits = output["model_logits"]
        face_logits = output["face_logits"]  # (B, N_faces, num_face_classes)

        # Model-level loss
        model_loss = self.loss_fn(
            model_logits, targets.float() if self.config.multi_label else targets
        )

        # Face-level loss: reshape to (B*N, C) vs (B*N,)
        face_targets = batch["face_labels"]  # (B, N_faces) with -1 for padding/missing
        B, N, C = face_logits.shape
        face_loss = self.face_loss_fn(
            face_logits.reshape(B * N, C),
            face_targets[:, :N].reshape(B * N),
        )

        # Combined loss
        loss = self.model_cls_weight * model_loss + self.face_seg_weight * face_loss

        # Log individual losses for diagnosis
        self.log("loss/model", model_loss.detach(), on_step=False, on_epoch=True)
        self.log("loss/face", face_loss.detach(), on_step=False, on_epoch=True)

        # Model predictions
        if self.config.multi_label:
            preds = (torch.sigmoid(model_logits) > 0.5).long()
        else:
            preds = model_logits.argmax(dim=-1)

        # Face predictions
        face_preds = face_logits.argmax(dim=-1)  # (B, N_faces)

        return loss, preds, targets, face_preds, face_targets[:, :N]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch dictionary.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        result = self._compute_loss_and_preds(batch)

        if self.face_segmentation:
            loss, preds, targets, face_preds, face_targets = result
        else:
            loss, preds, targets = result

        # Update metrics
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)

        # Face segmentation metrics
        if self.face_segmentation:
            valid = face_targets != -1
            if valid.any():
                self.train_face_acc(face_preds[valid], face_targets[valid])
                self.train_face_f1(face_preds[valid], face_targets[valid])
            self.log("train/face_acc", self.train_face_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/face_f1", self.train_face_f1, on_step=False, on_epoch=True)

        self.current_step += 1

        del preds, targets
        if self.face_segmentation:
            del face_preds, face_targets

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Input batch dictionary.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        result = self._compute_loss_and_preds(batch)

        if self.face_segmentation:
            loss, preds, targets, face_preds, face_targets = result
        else:
            loss, preds, targets = result

        # Update metrics
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)

        # Face segmentation metrics
        if self.face_segmentation:
            valid = face_targets != -1
            if valid.any():
                self.val_face_acc(face_preds[valid], face_targets[valid])
                self.val_face_f1(face_preds[valid], face_targets[valid])
            self.log("val/face_acc", self.val_face_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/face_f1", self.val_face_f1, on_step=False, on_epoch=True)

        del preds, targets
        if self.face_segmentation:
            del face_preds, face_targets

        # Detach to prevent PL from accumulating computation graphs
        return loss.detach()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Input batch dictionary.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        result = self._compute_loss_and_preds(batch)

        if self.face_segmentation:
            loss, preds, targets, face_preds, face_targets = result
        else:
            loss, preds, targets = result

        # Update metrics
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)

        # Face segmentation metrics
        if self.face_segmentation:
            valid = face_targets != -1
            if valid.any():
                self.test_face_acc(face_preds[valid], face_targets[valid])
                self.test_face_f1(face_preds[valid], face_targets[valid])
            self.log("test/face_acc", self.test_face_acc, on_step=False, on_epoch=True)
            self.log("test/face_f1", self.test_face_f1, on_step=False, on_epoch=True)

        return loss.detach()

    def configure_optimizers(self):
        """Configure optimizer and scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration.
        """
        # AdamW optimizer (from official implementation)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

        # ReduceLROnPlateau scheduler (from official implementation)
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
                "monitor": "train/loss_epoch",  # Use train loss if no validation
                "interval": "epoch",
                "frequency": 1,
                "strict": False,  # Don't error if metric not found
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
        """Custom optimizer step with warmup.

        Args:
            epoch: Current epoch.
            batch_idx: Current batch index.
            optimizer: Optimizer instance.
            optimizer_idx: Index of optimizer (for multiple optimizers).
            optimizer_closure: Closure for optimizer.
            on_tpu: Whether running on TPU.
            using_native_amp: Whether using native AMP.
            using_lbfgs: Whether using LBFGS optimizer.
        """
        # Linear warmup
        if self.current_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.current_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.config.learning_rate

        # Update weights
        optimizer.step(closure=optimizer_closure)

    # Metric resets are handled automatically by PyTorch Lightning when
    # torchmetrics objects are passed to self.log() with on_epoch=True.
