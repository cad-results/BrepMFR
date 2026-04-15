#!/usr/bin/env python3
"""Analysis script for BrepFormer models and predictions.

This script provides various analysis tools:
- Model architecture summary
- Per-class performance analysis
- Prediction visualization
- Attention weight analysis
- Feature embedding visualization (t-SNE/PCA)

Usage:
    # Analyze model architecture
    python brepformer/analyze.py --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt --mode architecture

    # Per-class performance analysis
    python brepformer/analyze.py --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
        --data_dir brepformer/data/mftrcad_processed --mode per_class

    # Embedding visualization
    python brepformer/analyze.py --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
        --data_dir brepformer/data/mftrcad_processed --mode embeddings --output_dir analysis_results
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add safe globals for checkpoint loading
torch.serialization.add_safe_globals([pathlib.PosixPath])


# ── Data preparation & VRAM helpers ────────────────────────────────────────


def prepare_dataset(data_dir: str, split: str, limit_data_manifest=None):
    """Prepare and cache the dataset. Called BEFORE model loading.

    Validates that the data directory exists and loads the dataset into CPU
    memory so any data issues are caught early, before expensive model loading.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Preparing {split} dataset from {data_dir}...")
    dataset = _make_dataset(data_dir, split, limit_data_manifest)
    print(f"Dataset ready: {len(dataset)} samples")
    return dataset


def _make_loader(dataset, batch_size: int, num_workers: int):
    """Create a DataLoader with memory-efficient settings."""
    from brepformer.data.collator import BrepCollator

    collator = BrepCollator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


def _free_vram():
    """Free unused VRAM between analysis steps."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _autocast_ctx(device):
    """Autocast context for memory-efficient inference on GPU."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


# ── Existing helpers ───────────────────────────────────────────────────────


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze BrepFormer models")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing preprocessed data (required for some modes)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["architecture", "per_class", "embeddings", "predictions", "attention",
                 "confusion_matrix", "face_segmentation", "step_inference", "all"],
        default="architecture",
        help="Analysis mode",
    )
    parser.add_argument(
        "--step_dir",
        type=str,
        default=None,
        help="Directory of STEP files (for step_inference mode)",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=50,
        help="Max models to evaluate in step_inference mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default 1, conservative for 4GB VRAM)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples for embedding visualization",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to analyze",
    )

    parser.add_argument(
        "--real_classes",
        action="store_true",
        help="Remap 27 MFTRCAD classes to 8 real machining feature categories",
    )
    parser.add_argument(
        "--limit_data_manifest",
        type=str,
        default=None,
        help="Path to limit_data_manifest.json for reproducible dataset subsetting",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    from brepformer.configs.config import BrepClassifierConfig
    from brepformer.models.brep_classifier import BrepClassifier

    torch.serialization.add_safe_globals([BrepClassifierConfig])
    model = BrepClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def _make_dataset(data_dir, split, limit_data_manifest=None):
    """Load a PreprocessedDataset with optional limit_data filtering."""
    from brepformer.data.preprocessed_dataset import PreprocessedDataset, apply_limit_data_manifest, load_limit_data_manifest
    dataset = PreprocessedDataset(data_dir, split=split)
    if limit_data_manifest:
        manifest = load_limit_data_manifest(limit_data_manifest) if isinstance(limit_data_manifest, str) else limit_data_manifest
        apply_limit_data_manifest(dataset, manifest, split)
    return dataset


# ── Analysis functions ─────────────────────────────────────────────────────


def analyze_architecture(model, output_dir: Path):
    """Analyze and print model architecture details."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("=" * 60)

    # Config summary
    config = model.config
    print("\n--- Configuration ---")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"FFN dimension: {config.ffn_dim}")
    print(f"Number of layers: {config.num_layers}")
    print(f"Number of attention heads: {config.num_heads}")
    print(f"Number of KV heads (GQA): {config.num_kv_heads}")
    print(f"Number of classes: {config.num_classes}")
    print(f"Multi-label: {config.multi_label}")
    print(f"Dropout: {config.dropout}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- Parameter Count ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")

    # Per-module breakdown
    print("\n--- Module Breakdown ---")
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
        print(f"  {name}: {params:,} ({params/total_params*100:.1f}%)")

    # Encoder breakdown
    print("\n--- Encoder Breakdown ---")
    encoder = model.encoder
    for name, module in encoder.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,}")

    # Save architecture info
    output_dir.mkdir(parents=True, exist_ok=True)
    arch_info = {
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
        "total_params": total_params,
        "trainable_params": trainable_params,
        "module_params": module_params,
    }
    with open(output_dir / "architecture.json", "w") as f:
        json.dump(arch_info, f, indent=2, default=str)
    print(f"\nArchitecture info saved to {output_dir / 'architecture.json'}")


def analyze_per_class(model, dataset, output_dir: Path, batch_size: int, num_workers: int,
                      real_classes: bool = False):
    """Analyze per-class performance metrics."""
    from brepformer.data.classes import CLASS_TO_REAL_CLASS, REAL_CLASS_NAMES, REAL_NUM_CLASSES

    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("=" * 60)

    loader = _make_loader(dataset, batch_size, num_workers)

    device = next(model.parameters()).device
    model.eval()

    # Collect predictions
    all_preds = []
    all_targets = []
    all_probs = []

    print(f"\nRunning inference on {len(dataset)} samples...")
    with torch.no_grad(), _autocast_ctx(device):
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            output = model(batch)
            logits = output["model_logits"] if isinstance(output, dict) else output

            if model.config.multi_label:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

            all_probs.append(probs.float().cpu())
            all_preds.append(preds.cpu())
            all_targets.append(batch["label"].cpu())

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    num_classes = model.config.num_classes

    if real_classes and not model.config.multi_label:
        remap = torch.tensor(CLASS_TO_REAL_CLASS, dtype=torch.long)
        all_preds = remap[all_preds]
        all_targets = remap[all_targets]
        num_classes = REAL_NUM_CLASSES
        class_names = REAL_CLASS_NAMES
    elif real_classes:
        print("  Warning: --real_classes not supported for multi-label per_class analysis")
        class_names = None
    else:
        class_names = None

    if model.config.multi_label:
        # Multi-label metrics
        tp = (all_preds * all_targets).sum(dim=0)
        fp = (all_preds * (1 - all_targets)).sum(dim=0)
        fn = ((1 - all_preds) * all_targets).sum(dim=0)
        tn = ((1 - all_preds) * (1 - all_targets)).sum(dim=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        support = all_targets.sum(dim=0)

        print("\n--- Per-Class Metrics (Multi-Label) ---")
        print(f"{'Class':>6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 50)

        results = []
        for i in range(num_classes):
            print(f"{i:>6} {precision[i].item():>10.4f} {recall[i].item():>10.4f} "
                  f"{f1[i].item():>10.4f} {int(support[i].item()):>10}")
            results.append({
                "class": i,
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "f1": f1[i].item(),
                "support": int(support[i].item()),
            })

        print("-" * 50)
        macro_p = precision.mean().item()
        macro_r = recall.mean().item()
        macro_f1 = f1.mean().item()
        print(f"{'Macro':>6} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")

        # Weighted metrics
        weights = support / support.sum()
        weighted_p = (precision * weights).sum().item()
        weighted_r = (recall * weights).sum().item()
        weighted_f1 = (f1 * weights).sum().item()
        print(f"{'Weighted':>6} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f}")

        # Multi-label confusion matrices
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Co-occurrence matrix (27, 27): entry (i,j) = count of samples where class i is GT and class j is predicted
            co_occurrence = np.zeros((num_classes, num_classes), dtype=np.int64)
            targets_np = all_targets.numpy()
            preds_np = all_preds.numpy()
            for sample_idx in range(targets_np.shape[0]):
                gt_classes = np.where(targets_np[sample_idx] > 0.5)[0]
                pred_classes = np.where(preds_np[sample_idx] > 0.5)[0]
                for gi in gt_classes:
                    for pj in pred_classes:
                        co_occurrence[gi, pj] += 1
            np.save(output_dir / "confusion_matrix_multilabel.npy", co_occurrence)
            print(f"\nCo-occurrence matrix saved to {output_dir / 'confusion_matrix_multilabel.npy'}")

            # Per-class 2x2 confusion: (27, 2, 2) with [[TN, FP], [FN, TP]]
            per_class_conf = np.zeros((num_classes, 2, 2), dtype=np.int64)
            for c in range(num_classes):
                per_class_conf[c, 0, 0] = int(tn[c].item())   # TN
                per_class_conf[c, 0, 1] = int(fp[c].item())   # FP
                per_class_conf[c, 1, 0] = int(fn[c].item())   # FN
                per_class_conf[c, 1, 1] = int(tp[c].item())   # TP
            np.save(output_dir / "per_class_confusion.npy", per_class_conf)
            print(f"Per-class confusion saved to {output_dir / 'per_class_confusion.npy'}")
        except Exception as e:
            print(f"\nError generating confusion matrices in per_class mode: {e}")
            import traceback
            traceback.print_exc()

    else:
        # Single-label metrics
        from torchmetrics import ConfusionMatrix
        cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        conf_matrix = cm(all_preds, all_targets)

        tp = conf_matrix.diag()
        fp = conf_matrix.sum(dim=0) - tp
        fn = conf_matrix.sum(dim=1) - tp
        support = conf_matrix.sum(dim=1)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        label_str = "Single-Label, Real Classes" if real_classes else "Single-Label"
        print(f"\n--- Per-Class Metrics ({label_str}) ---")
        print(f"{'Class':>25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 65)

        results = []
        for i in range(num_classes):
            name = class_names[i] if class_names else str(i)
            print(f"{name:>25} {precision[i].item():>10.4f} {recall[i].item():>10.4f} "
                  f"{f1[i].item():>10.4f} {int(support[i].item()):>10}")
            results.append({
                "class": i,
                "class_name": name,
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "f1": f1[i].item(),
                "support": int(support[i].item()),
            })

        # Save confusion matrix
        np.save(output_dir / "confusion_matrix.npy", conf_matrix.numpy())

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "per_class_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-class metrics saved to {output_dir / 'per_class_metrics.json'}")


def analyze_embeddings(model, dataset, output_dir: Path, batch_size: int,
                       num_workers: int, num_samples: int):
    """Extract and visualize embeddings using t-SNE/PCA."""
    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)

    # Use Subset for sampling to preserve the cached dataset
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices.tolist())
    else:
        subset = dataset

    loader = _make_loader(subset, batch_size, num_workers)

    device = next(model.parameters()).device
    model.eval()

    # Extract embeddings
    embeddings = []
    labels = []

    print(f"\nExtracting embeddings from {len(subset)} samples...")
    with torch.no_grad(), _autocast_ctx(device):
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get graph embeddings from encoder
            _, graph_emb = model.encoder(
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

            embeddings.append(graph_emb.float().cpu().numpy())
            labels.append(batch["label"].cpu().numpy())

    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Save raw embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "labels.npy", labels)

    # Try dimensionality reduction
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        print("\nRunning PCA...")
        n_pca = min(50, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_pca)
        embeddings_pca = pca.fit_transform(embeddings)
        np.save(output_dir / "embeddings_pca50.npy", embeddings_pca)
        print(f"PCA explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")

        print("\nRunning t-SNE...")
        perplexity = min(30, embeddings_pca.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=max(perplexity, 2), random_state=42, n_iter=1000)
        embeddings_tsne = tsne.fit_transform(embeddings_pca[:, :n_pca])
        np.save(output_dir / "embeddings_tsne.npy", embeddings_tsne)

        print(f"\nEmbeddings saved to {output_dir}")
        print("  - embeddings.npy: Raw embeddings")
        print("  - embeddings_pca50.npy: PCA-reduced (50D)")
        print("  - embeddings_tsne.npy: t-SNE (2D)")

    except ImportError:
        print("\nWarning: scikit-learn not installed. Skipping dimensionality reduction.")
        print("Install with: pip install scikit-learn")


def analyze_confusion_matrix(model, dataset, output_dir: Path, batch_size: int,
                             num_workers: int, real_classes: bool = False):
    """Generate confusion matrix analysis."""
    from brepformer.data.classes import CLASS_TO_REAL_CLASS, REAL_NUM_CLASSES

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    loader = _make_loader(dataset, batch_size, num_workers)

    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    print(f"\nRunning inference on {len(dataset)} samples...")
    with torch.no_grad(), _autocast_ctx(device):
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            output = model(batch)
            logits = output["model_logits"] if isinstance(output, dict) else output

            if model.config.multi_label:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(batch["label"].cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    num_classes = model.config.num_classes

    if real_classes and not model.config.multi_label:
        remap = torch.tensor(CLASS_TO_REAL_CLASS, dtype=torch.long)
        all_preds = remap[all_preds]
        all_targets = remap[all_targets]
        num_classes = REAL_NUM_CLASSES

    output_dir.mkdir(parents=True, exist_ok=True)

    if model.config.multi_label:
        targets_np = all_targets.numpy()
        preds_np = all_preds.numpy()

        # Co-occurrence matrix
        co_occurrence = np.zeros((num_classes, num_classes), dtype=np.int64)
        for idx in range(targets_np.shape[0]):
            gt_classes = np.where(targets_np[idx] > 0.5)[0]
            pred_classes = np.where(preds_np[idx] > 0.5)[0]
            for gi in gt_classes:
                for pj in pred_classes:
                    co_occurrence[gi, pj] += 1
        np.save(output_dir / "confusion_matrix_multilabel.npy", co_occurrence)
        print(f"Co-occurrence matrix saved to {output_dir / 'confusion_matrix_multilabel.npy'}")

        # Per-class 2x2 confusion
        tp = (all_preds * all_targets).sum(dim=0)
        fp = (all_preds * (1 - all_targets)).sum(dim=0)
        fn = ((1 - all_preds) * all_targets).sum(dim=0)
        tn = ((1 - all_preds) * (1 - all_targets)).sum(dim=0)

        per_class_conf = np.zeros((num_classes, 2, 2), dtype=np.int64)
        for c in range(num_classes):
            per_class_conf[c, 0, 0] = int(tn[c].item())
            per_class_conf[c, 0, 1] = int(fp[c].item())
            per_class_conf[c, 1, 0] = int(fn[c].item())
            per_class_conf[c, 1, 1] = int(tp[c].item())
        np.save(output_dir / "per_class_confusion.npy", per_class_conf)
        print(f"Per-class confusion saved to {output_dir / 'per_class_confusion.npy'}")
    else:
        from torchmetrics import ConfusionMatrix
        cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        conf_matrix = cm(all_preds, all_targets)
        np.save(output_dir / "confusion_matrix.npy", conf_matrix.numpy())
        print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.npy'}")

    print("Confusion matrix analysis complete.")


def analyze_predictions(model, dataset, output_dir: Path, batch_size: int,
                        num_workers: int, real_classes: bool = False):
    """Analyze model predictions in detail."""
    from brepformer.data.classes import CLASS_TO_REAL_CLASS, REAL_NUM_CLASSES

    print("\n" + "=" * 60)
    print("PREDICTION ANALYSIS")
    print("=" * 60)

    loader = _make_loader(dataset, batch_size, num_workers)

    device = next(model.parameters()).device
    model.eval()

    # Collect predictions
    predictions = []

    print(f"\nRunning inference on {len(dataset)} samples...")
    sample_idx = 0
    with torch.no_grad(), _autocast_ctx(device):
        for batch in tqdm(loader):
            batch_size_actual = batch["face_grid"].size(0)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            output = model(batch)
            logits = output["model_logits"] if isinstance(output, dict) else output

            if model.config.multi_label:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

            for i in range(batch_size_actual):
                model_id = batch["model_ids"][i] if "model_ids" in batch else f"sample_{sample_idx}"
                target = batch["label"][i].cpu().numpy()

                if model.config.multi_label:
                    pred = preds[i].cpu().numpy()
                    prob = probs[i].float().cpu().numpy()
                    pred_classes = np.where(pred > 0.5)[0].tolist()
                    target_classes = np.where(target > 0.5)[0].tolist()

                    # Full class probabilities for all classes
                    class_probs = {int(c): float(prob[c]) for c in range(len(prob))}

                    if real_classes:
                        pred_classes = sorted(set(CLASS_TO_REAL_CLASS[c] for c in pred_classes))
                        target_classes = sorted(set(CLASS_TO_REAL_CLASS[c] for c in target_classes))

                    # Jaccard similarity (IoU of predicted vs target class sets)
                    pred_set = set(pred_classes)
                    target_set = set(target_classes)
                    intersection = len(pred_set & target_set)
                    union = len(pred_set | target_set)
                    jaccard = intersection / union if union > 0 else 1.0

                    predictions.append({
                        "model_id": model_id,
                        "predicted_classes": pred_classes,
                        "target_classes": target_classes,
                        "num_pred": len(pred_classes),
                        "num_target": len(target_classes),
                        "class_probabilities": class_probs,
                        "jaccard_similarity": jaccard,
                        "correct": pred_classes == target_classes,
                    })
                else:
                    pred = preds[i].item()
                    prob = probs[i].float().cpu().numpy()
                    tgt = int(target)
                    if real_classes:
                        pred = CLASS_TO_REAL_CLASS[pred]
                        tgt = CLASS_TO_REAL_CLASS[tgt]
                    predictions.append({
                        "model_id": model_id,
                        "predicted_class": pred,
                        "target_class": tgt,
                        "confidence": float(prob[preds[i].item()]),
                        "correct": pred == tgt,
                    })

                sample_idx += 1

    # Compute summary statistics
    correct_count = sum(1 for p in predictions if p["correct"])
    accuracy = correct_count / len(predictions)

    print(f"\n--- Prediction Summary ---")
    print(f"Total samples: {len(predictions)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump({
            "summary": {
                "total_samples": len(predictions),
                "correct": correct_count,
                "accuracy": accuracy,
            },
            "predictions": predictions,
        }, f, indent=2)

    print(f"\nPredictions saved to {output_dir / 'predictions.json'}")


def analyze_face_segmentation(model, dataset, output_dir: Path, batch_size: int,
                               num_workers: int, real_classes: bool = False):
    """Analyze face-level segmentation performance.

    Computes per-class IoU, mean IoU, per-face accuracy, and confusion matrix.
    """
    from brepformer.data.classes import (
        CLASS_NAMES, NUM_CLASSES,
        CLASS_TO_REAL_CLASS, REAL_CLASS_NAMES, REAL_NUM_CLASSES,
    )

    print("\n" + "=" * 60)
    print("FACE SEGMENTATION ANALYSIS")
    print("=" * 60)

    if not model.face_segmentation:
        print("Model does not have face segmentation head enabled.")
        return

    loader = _make_loader(dataset, batch_size, num_workers)

    device = next(model.parameters()).device
    model.eval()

    all_face_preds = []
    all_face_targets = []

    print(f"\nRunning face-level inference on {len(dataset)} samples...")
    with torch.no_grad(), _autocast_ctx(device):
        for batch in tqdm(loader):
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
            output = model(batch_on_device)
            face_logits = output["face_logits"]  # (B, N, C)
            face_preds = face_logits.argmax(dim=-1).cpu()  # (B, N)
            face_targets = batch["face_labels"]  # (B, N)

            B, N = face_preds.shape
            for i in range(B):
                mask = batch["attn_mask"][i]
                n = mask.sum().item() - 1  # subtract CLS

                fp = face_preds[i, :n]
                ft = face_targets[i, :n]
                valid = ft != -1
                if valid.any():
                    all_face_preds.append(fp[valid])
                    all_face_targets.append(ft[valid])

    if not all_face_preds:
        print("No valid face predictions found.")
        return

    all_fp = torch.cat(all_face_preds)
    all_ft = torch.cat(all_face_targets)

    if real_classes:
        remap = torch.tensor(CLASS_TO_REAL_CLASS, dtype=torch.long)
        all_fp = remap[all_fp]
        all_ft = remap[all_ft]
        num_face_classes = REAL_NUM_CLASSES
        active_class_names = REAL_CLASS_NAMES
    else:
        num_face_classes = model.config.num_face_classes
        active_class_names = CLASS_NAMES

    # Per-face accuracy
    face_acc = (all_fp == all_ft).float().mean().item()

    # Per-class IoU, precision, recall, F1
    per_class_metrics = []
    confusion = np.zeros((num_face_classes, num_face_classes), dtype=np.int64)

    for i in range(len(all_fp)):
        p, t = all_fp[i].item(), all_ft[i].item()
        if 0 <= p < num_face_classes and 0 <= t < num_face_classes:
            confusion[t, p] += 1

    for c in range(num_face_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else float('nan')
        support = int(confusion[c, :].sum())

        per_class_metrics.append({
            "class_id": c,
            "class_name": active_class_names[c] if c < len(active_class_names) else f"class_{c}",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support": support,
        })

    valid_ious = [m["iou"] for m in per_class_metrics if m["iou"] == m["iou"]]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    # Print results
    print(f"\n--- Face Segmentation Results ---")
    print(f"Per-face accuracy: {face_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"\n{'Class':>25} {'Prec':>8} {'Recall':>8} {'F1':>8} {'IoU':>8} {'Support':>8}")
    print("-" * 75)
    for m in per_class_metrics:
        iou_str = f"{m['iou']:.4f}" if m['iou'] == m['iou'] else "N/A"
        print(f"{m['class_name']:>25} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {iou_str:>8} {m['support']:>8}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "face_accuracy": face_acc,
        "mean_iou": mean_iou,
        "per_class": per_class_metrics,
    }
    with open(output_dir / "face_seg_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nMetrics saved to {output_dir / 'face_seg_metrics.json'}")

    np.save(output_dir / "face_seg_confusion.npy", confusion)
    print(f"Confusion matrix saved to {output_dir / 'face_seg_confusion.npy'}")


def analyze_step_inference(model, data_dir: str, output_dir: Path, step_dir: str,
                           max_models: int = 50, real_classes: bool = False,
                           limit_data_manifest=None):
    """Analyze inference quality on STEP files vs preprocessed ground truth.

    Compares two inference paths:
    1. preprocessed_pickle: loads from preprocessed data (exact training match)
    2. step_to_graph: converts STEP files on-the-fly (tests deployment path)

    Requires both --data_dir (for GT labels) and --step_dir (for STEP files).
    """
    import pickle
    from sklearn.metrics import f1_score, precision_score, recall_score
    from brepformer.infer import prepare_batch, load_sample_from_preprocessed
    from brepformer.data.step_to_graph import step_to_preprocessed_sample
    from brepformer.data.classes import CLASS_NAMES

    print("\n" + "=" * 60)
    print("STEP INFERENCE ANALYSIS")
    print("=" * 60)

    if not model.face_segmentation:
        print("Model does not have face segmentation head. Skipping.")
        return

    # Load manifest filter if provided
    allowed_files = None
    if limit_data_manifest:
        from brepformer.data.preprocessed_dataset import load_limit_data_manifest
        manifest = load_limit_data_manifest(limit_data_manifest) if isinstance(limit_data_manifest, str) else limit_data_manifest
        allowed_files = set(manifest["splits"].get("test", []))

    # Load GT from preprocessed test data
    test_dir = Path(data_dir) / "test"
    if not test_dir.is_dir():
        print(f"No test/ directory found in {data_dir}")
        return

    gt = {}
    for f in sorted(test_dir.glob("*.pkl"))[:max_models]:
        if allowed_files is not None and f.name not in allowed_files:
            continue
        with open(f, "rb") as fh:
            s = pickle.load(fh)
        gt[s["model_id"]] = np.array(s.get("face_labels", []))

    steps_path = Path(step_dir)
    step_map = {}
    for mid in gt:
        for c in [steps_path / f"{mid}.step", steps_path / f"{mid}_result.step"]:
            if c.exists():
                step_map[mid] = str(c)
                break

    print(f"Test models: {len(step_map)}, max: {max_models}")
    device = next(model.parameters()).device

    results_all = {}
    for path_name, use_pickle in [("preprocessed_pickle", True), ("step_to_graph", False)]:
        all_preds, all_targets = [], []
        n_models, n_perfect = 0, 0

        for mid, sp in list(step_map.items())[:max_models]:
            fl = gt[mid]
            valid = fl >= 0
            if valid.sum() == 0:
                continue

            if use_pickle:
                sample = load_sample_from_preprocessed(sp)
            else:
                sample = step_to_preprocessed_sample(sp)
            if sample is None:
                continue

            batch = prepare_batch(sample, device)
            nf = sample["num_faces"]
            with torch.no_grad(), _autocast_ctx(device):
                out = model(batch)
            preds = out["face_logits"][0, :nf].argmax(dim=-1).cpu().numpy()

            pv, tv = preds[valid], fl[valid]
            all_preds.extend(pv.tolist())
            all_targets.extend(tv.tolist())
            n_models += 1
            if (pv == tv).all():
                n_perfect += 1

        ap = np.array(all_preds)
        at = np.array(all_targets)
        acc = float((ap == at).mean())
        f1m = float(f1_score(at, ap, average='macro', zero_division=0))
        f1w = float(f1_score(at, ap, average='weighted', zero_division=0))
        prec = float(precision_score(at, ap, average='macro', zero_division=0))
        rec = float(recall_score(at, ap, average='macro', zero_division=0))

        print(f"\n--- {path_name} ---")
        print(f"  Models: {n_models}, Faces: {len(ap)}")
        print(f"  Face accuracy:     {acc:.4f}")
        print(f"  F1 (macro):        {f1m:.4f}")
        print(f"  F1 (weighted):     {f1w:.4f}")
        print(f"  Precision (macro): {prec:.4f}")
        print(f"  Recall (macro):    {rec:.4f}")
        print(f"  Perfect models:    {n_perfect}/{n_models} ({100*n_perfect/max(n_models,1):.1f}%)")

        results_all[path_name] = {
            "n_models": n_models, "n_faces": len(ap),
            "face_accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
            "precision_macro": prec, "recall_macro": rec,
            "perfect_models": n_perfect,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "step_inference_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {output_dir / 'step_inference_results.json'}")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    """Main analysis function."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    ldm = args.limit_data_manifest

    # ── Step 1: Prepare/validate data FIRST (before model loading) ─────
    # This catches data issues early and caches the dataset for reuse
    # across multiple analysis modes.
    modes_needing_dataset = {"per_class", "embeddings", "predictions",
                             "confusion_matrix", "face_segmentation"}

    dataset = None
    if args.data_dir is not None:
        needs_dataset = args.mode in modes_needing_dataset or args.mode == "all"
        if needs_dataset:
            dataset = prepare_dataset(args.data_dir, args.split, ldm)

    # Validate step_inference requirements early
    if args.mode == "step_inference":
        if args.data_dir is None or args.step_dir is None:
            print("\nError: --data_dir and --step_dir required for step_inference analysis")
            sys.exit(1)
        # Validate directories exist
        if not Path(args.data_dir).exists():
            print(f"\nError: Data directory not found: {args.data_dir}")
            sys.exit(1)
        if not Path(args.step_dir).exists():
            print(f"\nError: STEP directory not found: {args.step_dir}")
            sys.exit(1)

    # ── Step 2: Load model ─────────────────────────────────────────────
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # ── Step 3: Run analyses with cached dataset ───────────────────────
    if args.mode in ["architecture", "all"]:
        analyze_architecture(model, output_dir)

    if args.mode in ["per_class", "all"]:
        if dataset is None:
            print("\nError: --data_dir required for per_class analysis")
        else:
            analyze_per_class(model, dataset, output_dir, args.batch_size,
                            args.num_workers, args.real_classes)
            _free_vram()

    if args.mode in ["embeddings", "all"]:
        if dataset is None:
            print("\nError: --data_dir required for embeddings analysis")
        else:
            analyze_embeddings(model, dataset, output_dir, args.batch_size,
                             args.num_workers, args.num_samples)
            _free_vram()

    if args.mode in ["predictions", "all"]:
        if dataset is None:
            print("\nError: --data_dir required for predictions analysis")
        else:
            analyze_predictions(model, dataset, output_dir, args.batch_size,
                              args.num_workers, args.real_classes)
            _free_vram()

    if args.mode in ["confusion_matrix", "all"]:
        if dataset is None:
            print("\nError: --data_dir required for confusion_matrix analysis")
        else:
            analyze_confusion_matrix(model, dataset, output_dir, args.batch_size,
                                    args.num_workers, args.real_classes)
            _free_vram()

    if args.mode in ["face_segmentation", "all"]:
        if dataset is None:
            print("\nError: --data_dir required for face_segmentation analysis")
        else:
            analyze_face_segmentation(model, dataset, output_dir, args.batch_size,
                                      args.num_workers, args.real_classes)
            _free_vram()

    if args.mode in ["step_inference"]:
        analyze_step_inference(model, args.data_dir, output_dir, args.step_dir,
                               args.max_models, args.real_classes, ldm)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
