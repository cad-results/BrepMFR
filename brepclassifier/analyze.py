#!/usr/bin/env python3
"""Analysis script for pipe fitting classifier.

6 analysis modes:
- architecture: Model summary with GAT head breakdown
- per_class: 8-class per-class precision/recall/F1
- confusion_matrix: 8x8 confusion matrix
- predictions: Per-sample predictions
- embeddings: Encoder CLS + GAT pooled embeddings -> PCA -> t-SNE
- all: Run everything

Usage:
    python brepclassifier/analyze.py \
        --checkpoint results/pipe_classifier/best.ckpt \
        --data_dir brepclassifier/data/ssdata1_processed \
        --mode all \
        --output_dir analysis_results/pipe_classifier
"""

import argparse
import json
import pathlib
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
# Compatibility shim: older pytorch_lightning uses np.Inf, removed in NumPy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.serialization.add_safe_globals([pathlib.PosixPath])

CLASS_NAMES = [
    "Elbow - Weld Fitting",
    "Elbow - Pipe End Fitting",
    "Elbow - Socket Fitting",
    "Tee - Weld Fitting",
    "Tee - Pipe End Fitting",
    "Tee - Socket Fitting",
    "Elbow - Miscellaneous",
    "Tee - Miscellaneous",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze pipe fitting classifier")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--mode", type=str,
                        choices=["architecture", "per_class", "embeddings",
                                 "predictions", "confusion_matrix", "all"],
                        default="architecture")
    parser.add_argument("--output_dir", type=str, default="analysis_results/pipe_classifier")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max_faces", type=int, default=0,
                        help="Max faces per sample (0=no limit). Use 500 to avoid OOM.")

    return parser.parse_args()


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    from brepformer.configs.config import BrepClassifierConfig
    from brepclassifier.configs.config import PipeFittingConfig
    from brepclassifier.models.pipe_classifier import PipeFittingClassifier

    torch.serialization.add_safe_globals([BrepClassifierConfig, PipeFittingConfig])
    model = PipeFittingClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def analyze_architecture(model, output_dir: Path):
    """Analyze and print model architecture details."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("=" * 60)

    config = model.config
    print("\n--- Configuration ---")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"FFN dimension: {config.ffn_dim}")
    print(f"Encoder layers: {config.num_layers}")
    print(f"Attention heads: {config.num_heads}")
    print(f"KV heads (GQA): {config.num_kv_heads}")
    print(f"Number of classes: {config.num_classes}")
    print(f"GAT layers: {config.gat_num_layers}")
    print(f"GAT heads: {config.gat_heads}")
    print(f"GAT hidden dim: {config.gat_hidden_dim}")
    print(f"GAT pooling: {config.gat_pooling}")
    print(f"Dense dims: {config.dense_dims}")
    print(f"Freeze encoder: {config.freeze_encoder}")

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
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = params
        print(f"  {name}: {params:,} ({params/total_params*100:.1f}%) "
              f"[trainable: {trainable:,}]")

    # Encoder breakdown
    print("\n--- Encoder Breakdown ---")
    for name, module in model.encoder.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,}")

    # GAT head breakdown
    print("\n--- GAT Head Breakdown ---")
    for name, module in model.gat_head.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    arch_info = {
        "config": {k: v for k, v in vars(config).items() if not k.startswith("_")},
        "total_params": total_params,
        "trainable_params": trainable_params,
        "module_params": module_params,
    }
    with open(output_dir / "architecture.json", "w") as f:
        json.dump(arch_info, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'architecture.json'}")


def analyze_per_class(model, data_dir: str, output_dir: Path, batch_size: int,
                      num_workers: int, split: str, max_faces: int = 0):
    """Analyze per-class performance metrics."""
    from brepclassifier.data.preprocessed_dataset import PreprocessedDataset
    from brepformer.data.collator import BrepCollator

    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("=" * 60)

    dataset = PreprocessedDataset(data_dir, split=split, max_faces=max_faces)
    collator = BrepCollator()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collator)

    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"\nRunning inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(batch["label"].cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    num_classes = model.config.num_classes

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

    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':>3} {'Name':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 70)

    results = []
    for i in range(num_classes):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
        print(f"{i:>3} {name:<30} {precision[i].item():>10.4f} {recall[i].item():>10.4f} "
              f"{f1[i].item():>10.4f} {int(support[i].item()):>8}")
        results.append({
            "class": i,
            "name": name,
            "precision": precision[i].item(),
            "recall": recall[i].item(),
            "f1": f1[i].item(),
            "support": int(support[i].item()),
        })

    print("-" * 70)
    print(f"{'':>3} {'Macro Average':<30} {precision.mean().item():>10.4f} "
          f"{recall.mean().item():>10.4f} {f1.mean().item():>10.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "per_class_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir / 'per_class_metrics.json'}")


def analyze_confusion_matrix(model, data_dir: str, output_dir: Path, batch_size: int,
                             num_workers: int, split: str, max_faces: int = 0):
    """Generate and save confusion matrix."""
    from brepclassifier.data.preprocessed_dataset import PreprocessedDataset
    from brepformer.data.collator import BrepCollator
    from torchmetrics import ConfusionMatrix

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    dataset = PreprocessedDataset(data_dir, split=split, max_faces=max_faces)
    collator = BrepCollator()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collator)

    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            logits = model(batch)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(batch["label"].cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cm = ConfusionMatrix(task="multiclass", num_classes=model.config.num_classes)
    conf_matrix = cm(all_preds, all_targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "confusion_matrix.npy", conf_matrix.numpy())
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.npy'}")


def analyze_predictions(model, data_dir: str, output_dir: Path, batch_size: int,
                        num_workers: int, split: str, max_faces: int = 0):
    """Generate per-sample predictions."""
    from brepclassifier.data.preprocessed_dataset import PreprocessedDataset
    from brepformer.data.collator import BrepCollator

    print("\n" + "=" * 60)
    print("PREDICTION ANALYSIS")
    print("=" * 60)

    dataset = PreprocessedDataset(data_dir, split=split, max_faces=max_faces)
    collator = BrepCollator()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collator)

    device = next(model.parameters()).device
    model.eval()

    predictions = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            batch_size_actual = batch["face_grid"].size(0)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            for i in range(batch_size_actual):
                model_id = batch["model_ids"][i] if "model_ids" in batch else f"sample_{sample_idx}"
                target = batch["label"][i].item()
                pred = preds[i].item()
                prob = probs[i].cpu().numpy()

                predictions.append({
                    "model_id": model_id,
                    "predicted_class": pred,
                    "predicted_name": CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred),
                    "target_class": target,
                    "target_name": CLASS_NAMES[target] if target < len(CLASS_NAMES) else str(target),
                    "confidence": float(prob[pred]),
                    "correct": pred == target,
                })
                sample_idx += 1

    correct_count = sum(1 for p in predictions if p["correct"])
    accuracy = correct_count / len(predictions)

    print(f"\n--- Prediction Summary ---")
    print(f"Total samples: {len(predictions)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")

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
    print(f"Saved to {output_dir / 'predictions.json'}")


def analyze_embeddings(model, data_dir: str, output_dir: Path, batch_size: int,
                       num_workers: int, num_samples: int, split: str,
                       max_faces: int = 0):
    """Extract embeddings and run PCA + t-SNE."""
    from brepclassifier.data.preprocessed_dataset import PreprocessedDataset
    from brepformer.data.collator import BrepCollator

    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)

    dataset = PreprocessedDataset(data_dir, split=split, max_faces=max_faces)

    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset.samples = [dataset.samples[i] for i in indices]

    collator = BrepCollator()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collator)

    device = next(model.parameters()).device
    model.eval()

    cls_embeddings = []
    labels = []

    print(f"\nExtracting embeddings from {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get encoder CLS embedding
            _, graph_emb = model.encoder(
                face_grid=batch["face_grid"],
                face_attr=batch["face_attr"],
                edge_index=batch["edge_index"],
                edge_attr=batch["edge_attr"],
                edge_grid=batch["edge_grid"],
                spatial_pos=batch["spatial_pos"],
                in_degree=batch.get("in_degree"),
                attn_mask=batch.get("attn_mask"),
            )

            cls_embeddings.append(graph_emb.cpu().numpy())
            labels.append(batch["label"].cpu().numpy())

    cls_embeddings = np.vstack(cls_embeddings)
    labels = np.concatenate(labels)

    print(f"Embeddings shape: {cls_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", cls_embeddings)
    np.save(output_dir / "labels.npy", labels)

    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        print("\nRunning PCA...")
        n_components = min(50, cls_embeddings.shape[1], cls_embeddings.shape[0])
        pca = PCA(n_components=n_components)
        embeddings_pca = pca.fit_transform(cls_embeddings)
        np.save(output_dir / "embeddings_pca50.npy", embeddings_pca)

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(labels) - 1),
                     random_state=42, n_iter=1000)
        embeddings_tsne = tsne.fit_transform(embeddings_pca)
        np.save(output_dir / "embeddings_tsne.npy", embeddings_tsne)

        print(f"Saved embeddings to {output_dir}")

    except ImportError:
        print("Warning: scikit-learn not installed. Skipping dimensionality reduction.")


def main():
    """Main analysis function."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    if args.mode in ["architecture", "all"]:
        analyze_architecture(model, output_dir)

    if args.mode in ["per_class", "all"]:
        if args.data_dir is None:
            print("\nError: --data_dir required for per_class analysis")
        else:
            analyze_per_class(model, args.data_dir, output_dir, args.batch_size,
                            args.num_workers, args.split, args.max_faces)

    if args.mode in ["confusion_matrix", "all"]:
        if args.data_dir is None:
            print("\nError: --data_dir required for confusion_matrix analysis")
        else:
            analyze_confusion_matrix(model, args.data_dir, output_dir, args.batch_size,
                                    args.num_workers, args.split, args.max_faces)

    if args.mode in ["predictions", "all"]:
        if args.data_dir is None:
            print("\nError: --data_dir required for predictions analysis")
        else:
            analyze_predictions(model, args.data_dir, output_dir, args.batch_size,
                              args.num_workers, args.split, args.max_faces)

    if args.mode in ["embeddings", "all"]:
        if args.data_dir is None:
            print("\nError: --data_dir required for embeddings analysis")
        else:
            analyze_embeddings(model, args.data_dir, output_dir, args.batch_size,
                             args.num_workers, args.num_samples, args.split,
                             args.max_faces)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
