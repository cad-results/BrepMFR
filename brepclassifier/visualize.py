#!/usr/bin/env python3
"""Visualization script for pipe fitting classifier analysis results.

Generates:
- Per-class F1 bar chart (8 classes)
- Precision-Recall scatter
- Class distribution bar chart
- 8x8 confusion matrix heatmap
- t-SNE scatter colored by 8 classes
- Training curves from TensorBoard logs

Usage:
    python brepclassifier/visualize.py --mode all \
        --input_dir analysis_results/pipe_classifier \
        --log_dir results/pipe_classifier \
        --output_dir plots/pipe_classifier
"""

import argparse
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "Elbow-WF", "Elbow-PEF", "Elbow-SF",
    "Tee-WF", "Tee-PEF", "Tee-SF",
    "Elbow-Misc", "Tee-Misc",
]

CLASS_NAMES_FULL = [
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
    parser = argparse.ArgumentParser(description="Visualize pipe classifier results")
    parser.add_argument("--mode", type=str,
                        choices=["embeddings", "metrics", "training", "confusion",
                                 "distribution", "all"],
                        default="all")
    parser.add_argument("--input_dir", type=str,
                        default="analysis_results/pipe_classifier")
    parser.add_argument("--log_dir", type=str, default="results/pipe_classifier")
    parser.add_argument("--output_dir", type=str, default="plots/pipe_classifier")
    parser.add_argument("--format", type=str, choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def visualize_metrics(input_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize per-class F1 scores and precision/recall."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed")
        return

    print("\n--- Per-Class Metrics ---")

    metrics_path = input_dir / "per_class_metrics.json"
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [m["class"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    recall = [m["recall"] for m in metrics]
    f1 = [m["f1"] for m in metrics]
    support = [m["support"] for m in metrics]
    names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in classes]

    # Color palette for 8 classes
    colors = plt.colormaps.get_cmap("tab10")(np.linspace(0, 1, 10))[:8]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1 bar chart
    ax = axes[0, 0]
    bars = ax.bar(names, f1, color=colors, alpha=0.8)
    ax.axhline(y=np.mean(f1), color='red', linestyle='--', label=f'Mean: {np.mean(f1):.3f}')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45)

    # Precision-Recall scatter
    ax = axes[0, 1]
    for i, name in enumerate(names):
        ax.scatter(recall[i], precision[i], color=colors[i], s=100, alpha=0.8,
                   label=name, zorder=5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='lower left')

    # Class distribution
    ax = axes[1, 0]
    ax.bar(names, support, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution (shows imbalance)')
    ax.tick_params(axis='x', rotation=45)

    # Grouped bar chart
    ax = axes[1, 1]
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1', alpha=0.8)
    ax.set_ylabel('Score')
    ax.set_title('Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / f"per_class_metrics.{fmt}", dpi=dpi)
    plt.close()
    print(f"Saved to {output_dir}")


def visualize_confusion_matrix(input_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize 8x8 confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Error: matplotlib/seaborn not installed")
        return

    print("\n--- Confusion Matrix ---")

    cm_path = input_dir / "confusion_matrix.npy"
    if not cm_path.exists():
        print(f"Error: {cm_path} not found")
        return

    cm = np.load(cm_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    names = CLASS_NAMES[:cm.shape[0]]

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=axes[0],
                xticklabels=names, yticklabels=names)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=names, yticklabels=names)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix.{fmt}", dpi=dpi)
    plt.close()
    print(f"Saved to {output_dir}")


def visualize_embeddings(input_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize t-SNE embeddings colored by 8 classes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed")
        return

    print("\n--- Embedding Visualization ---")

    tsne_path = input_dir / "embeddings_tsne.npy"
    labels_path = input_dir / "labels.npy"

    if not tsne_path.exists():
        print(f"Error: {tsne_path} not found")
        return

    embeddings = np.load(tsne_path)
    labels = np.load(labels_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.colormaps.get_cmap("tab10")(np.linspace(0, 1, 10))[:8]

    fig, ax = plt.subplots(figsize=(12, 10))

    for cls_id in range(8):
        mask = labels == cls_id
        if mask.any():
            name = CLASS_NAMES[cls_id]
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      s=15, alpha=0.7, color=colors[cls_id], label=name)

    ax.set_title('t-SNE Embeddings (8 Pipe Fitting Classes)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=9, markerscale=2, loc='best')
    plt.tight_layout()
    plt.savefig(output_dir / f"embeddings_tsne.{fmt}", dpi=dpi)
    plt.close()
    print(f"Saved to {output_dir}")


def visualize_training(log_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize training curves from TensorBoard logs."""
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Error: matplotlib or tensorboard not installed")
        return

    print("\n--- Training Curves ---")

    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"Error: No TensorBoard event files found in {log_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_scalars = {}
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(str(event_file.parent))
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            if tag not in all_scalars:
                all_scalars[tag] = {"steps": [], "values": []}
            for event in ea.Scalars(tag):
                all_scalars[tag]["steps"].append(event.step)
                all_scalars[tag]["values"].append(event.value)

    if not all_scalars:
        print("No scalar data found")
        return

    loss_metrics = {k: v for k, v in all_scalars.items() if "loss" in k.lower()}
    acc_metrics = {k: v for k, v in all_scalars.items() if "acc" in k.lower()}
    f1_metrics = {k: v for k, v in all_scalars.items() if "f1" in k.lower()}
    lr_metrics = {k: v for k, v in all_scalars.items() if "lr" in k.lower()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, metrics, ylabel, title in [
        (axes[0, 0], loss_metrics, "Loss", "Loss"),
        (axes[0, 1], acc_metrics, "Accuracy", "Accuracy"),
        (axes[1, 0], f1_metrics, "F1 Score", "F1 Score"),
        (axes[1, 1], lr_metrics, "Learning Rate", "Learning Rate"),
    ]:
        for name, data in metrics.items():
            steps, values = zip(*sorted(zip(data["steps"], data["values"])))
            label = name.replace("_", " ").title()
            ax.plot(steps, values, label=label, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"training_curves.{fmt}", dpi=dpi)
    plt.close()
    print(f"Saved to {output_dir}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    if args.mode in ["metrics", "all"]:
        visualize_metrics(input_dir, output_dir, args.format, args.dpi)

    if args.mode in ["confusion", "all"]:
        visualize_confusion_matrix(input_dir, output_dir, args.format, args.dpi)

    if args.mode in ["embeddings", "all"]:
        visualize_embeddings(input_dir, output_dir, args.format, args.dpi)

    if args.mode in ["training", "all"]:
        visualize_training(log_dir, output_dir, args.format, args.dpi)

    if args.mode in ["distribution", "all"]:
        visualize_metrics(input_dir, output_dir, args.format, args.dpi)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
