#!/usr/bin/env python3
"""Visualization script for BrepFormer analysis results.

Generates plots and visualizations from analysis outputs.

Usage:
    # Visualize embeddings (requires analysis results)
    python brepformer/visualize.py --mode embeddings --input_dir analysis_results --output_dir plots

    # Visualize per-class metrics
    python brepformer/visualize.py --mode metrics --input_dir analysis_results --output_dir plots

    # Visualize training curves from TensorBoard logs
    python brepformer/visualize.py --mode training --log_dir results/brepformer --output_dir plots
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize BrepFormer analysis results")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["embeddings", "metrics", "training", "confusion", "all"],
        default="all",
        help="Visualization mode",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="analysis_results",
        help="Input directory containing analysis results",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results/brepformer",
        help="TensorBoard log directory for training visualization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output images",
    )

    return parser.parse_args()


def visualize_embeddings(input_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize t-SNE embeddings."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        return

    print("\n--- Embedding Visualization ---")

    # Load embeddings
    tsne_path = input_dir / "embeddings_tsne.npy"
    labels_path = input_dir / "labels.npy"

    if not tsne_path.exists():
        print(f"Error: {tsne_path} not found. Run analyze.py --mode embeddings first.")
        return

    embeddings = np.load(tsne_path)
    labels = np.load(labels_path)

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels: {labels.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # For multi-label, color by number of active classes
    if labels.ndim > 1:
        num_active = labels.sum(axis=1)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                            c=num_active, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Number of Active Classes')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('B-rep Embeddings (colored by number of machining features)')
        plt.tight_layout()
        plt.savefig(output_dir / f"embeddings_tsne_multiclass.{fmt}", dpi=dpi)
        plt.close()

        # Also create plots for most common classes
        class_counts = labels.sum(axis=0)
        top_classes = np.argsort(class_counts)[-5:]  # Top 5 classes

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, cls in enumerate(top_classes):
            ax = axes[idx]
            mask = labels[:, cls] > 0.5
            ax.scatter(embeddings[~mask, 0], embeddings[~mask, 1],
                      c='lightgray', alpha=0.3, s=10, label='Other')
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c='red', alpha=0.6, s=20, label=f'Class {cls}')
            ax.set_title(f'Class {cls} (n={int(class_counts[cls])})')
            ax.legend()

        # Overall view in last subplot
        axes[5].scatter(embeddings[:, 0], embeddings[:, 1],
                       c=num_active, cmap='viridis', alpha=0.6, s=20)
        axes[5].set_title('All Classes (by count)')

        plt.tight_layout()
        plt.savefig(output_dir / f"embeddings_tsne_top_classes.{fmt}", dpi=dpi)
        plt.close()

    else:
        # Single-label visualization
        unique_labels = np.unique(labels)
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                            c=labels, cmap='tab20', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Class')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('B-rep Embeddings (colored by class)')
        plt.tight_layout()
        plt.savefig(output_dir / f"embeddings_tsne.{fmt}", dpi=dpi)
        plt.close()

    print(f"Embedding plots saved to {output_dir}")


def visualize_metrics(input_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize per-class metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        return

    print("\n--- Per-Class Metrics Visualization ---")

    metrics_path = input_dir / "per_class_metrics.json"
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found. Run analyze.py --mode per_class first.")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [m["class"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    recall = [m["recall"] for m in metrics]
    f1 = [m["f1"] for m in metrics]
    support = [m["support"] for m in metrics]

    # Bar plot of F1 scores
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1 Score
    ax = axes[0, 0]
    bars = ax.bar(classes, f1, color='steelblue', alpha=0.8)
    ax.axhline(y=np.mean(f1), color='red', linestyle='--', label=f'Mean: {np.mean(f1):.3f}')
    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score')
    ax.legend()

    # Precision vs Recall
    ax = axes[0, 1]
    ax.scatter(recall, precision, c=f1, cmap='RdYlGn', s=100, alpha=0.7)
    for i, cls in enumerate(classes):
        ax.annotate(str(cls), (recall[i], precision[i]), fontsize=8)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (colored by F1)')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    # Support distribution
    ax = axes[1, 0]
    ax.bar(classes, support, color='coral', alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Support (# samples)')
    ax.set_title('Class Distribution')

    # Grouped bar chart
    ax = axes[1, 1]
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1', alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1 by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"per_class_metrics.{fmt}", dpi=dpi)
    plt.close()

    print(f"Metrics plots saved to {output_dir}")


def visualize_confusion_matrix(input_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Error: matplotlib/seaborn not installed.")
        return

    print("\n--- Confusion Matrix Visualization ---")

    # Try candidate filenames for backward compatibility
    cm_candidates = [
        "confusion_matrix.npy",
        "face_seg_confusion.npy",
    ]
    cm_path = None
    for name in cm_candidates:
        candidate = input_dir / name
        if candidate.exists():
            cm_path = candidate
            break
    if cm_path is None:
        print(f"Error: No confusion matrix found in {input_dir}. "
              f"Expected one of: {', '.join(cm_candidates)}. "
              f"Run analyze.py --mode per_class or --mode face_segmentation first.")
        return
    print(f"Using confusion matrix: {cm_path.name}")

    cm = np.load(cm_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize by row (true labels)
    cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')

    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix.{fmt}", dpi=dpi)
    plt.close()

    print(f"Confusion matrix plot saved to {output_dir}")


def visualize_training(log_dir: Path, output_dir: Path, fmt: str, dpi: int):
    """Visualize training curves from TensorBoard logs."""
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Error: matplotlib or tensorboard not installed.")
        return

    print("\n--- Training Curve Visualization ---")

    # Find event files
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"Error: No TensorBoard event files found in {log_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all events
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
        print("No scalar data found in TensorBoard logs")
        return

    print(f"Found {len(all_scalars)} scalar metrics")

    # Group related metrics
    loss_metrics = {k: v for k, v in all_scalars.items() if "loss" in k.lower()}
    acc_metrics = {k: v for k, v in all_scalars.items() if "acc" in k.lower()}
    f1_metrics = {k: v for k, v in all_scalars.items() if "f1" in k.lower()}
    lr_metrics = {k: v for k, v in all_scalars.items() if "lr" in k.lower()}

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    for name, data in loss_metrics.items():
        steps, values = zip(*sorted(zip(data["steps"], data["values"])))
        label = name.replace("_", " ").title()
        ax.plot(steps, values, label=label, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    for name, data in acc_metrics.items():
        steps, values = zip(*sorted(zip(data["steps"], data["values"])))
        label = name.replace("_", " ").title()
        ax.plot(steps, values, label=label, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score
    ax = axes[1, 0]
    for name, data in f1_metrics.items():
        steps, values = zip(*sorted(zip(data["steps"], data["values"])))
        label = name.replace("_", " ").title()
        ax.plot(steps, values, label=label, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("F1 Score")
    ax.set_title("Training and Validation F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    for name, data in lr_metrics.items():
        steps, values = zip(*sorted(zip(data["steps"], data["values"])))
        label = name.replace("_", " ").title()
        ax.plot(steps, values, label=label, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"training_curves.{fmt}", dpi=dpi)
    plt.close()

    print(f"Training curves saved to {output_dir}")


def main():
    """Main visualization function."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    if args.mode in ["embeddings", "all"]:
        visualize_embeddings(input_dir, output_dir, args.format, args.dpi)

    if args.mode in ["metrics", "all"]:
        visualize_metrics(input_dir, output_dir, args.format, args.dpi)

    if args.mode in ["confusion", "all"]:
        visualize_confusion_matrix(input_dir, output_dir, args.format, args.dpi)

    if args.mode in ["training", "all"]:
        visualize_training(log_dir, output_dir, args.format, args.dpi)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
