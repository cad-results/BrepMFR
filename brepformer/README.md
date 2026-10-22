# BrepFormer: Transformer-Based B-rep Classification

This is an implementation of the BrepFormer architecture for whole B-rep (Boundary Representation) classification, based on the paper ["BrepFormer: Transformer-Based B-rep Geometric Feature Recognition"](https://arxiv.org/abs/2504.07378).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Data Format](#data-format)
- [Scripts Reference](#scripts-reference)
  - [Data Preprocessing](#data-preprocessing)
  - [Training](#training)
  - [Testing](#testing)
  - [Analysis](#analysis)
  - [Visualization](#visualization)
  - [TensorBoard](#tensorboard)
  - [Model Export](#model-export)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Defeature Dataset (5 classes)](#defeature-dataset-5-classes)

---

## Overview

BrepFormer is a Graph Transformer architecture designed for CAD B-rep models. Key features:

- **Multi-label classification**: Recognizes 27 machining feature classes per model
- **Graph attention with topology bias**: Edge and topology features modulate attention scores
- **Rotary Position Embeddings (RoPE)**: For better sequence modeling
- **Grouped Query Attention (GQA)**: 32 query heads with 8 KV heads for efficiency
- **SwiGLU FFN**: Advanced feed-forward network with gated linear units

### Performance

**Trial 5** (best checkpoint, 43 epochs on MFTRCAD with face segmentation):

| Metric | Preprocessed Test | STEP Inference (dataset) |
|--------|------------------|-------------------------|
| Face accuracy | 92.5% | 94.7% (via pickle) |
| F1 (macro) | 87.3% | 86.4% |
| F1 (weighted) | — | 94.6% |
| Mean IoU | 78.6% | — |
| Perfect models | 50% | 48% |

Model-level: test accuracy 96.7%, F1 87.2%.

### Face Segmentation on Real Parts (defeature 5-class)

Scored with `scan_checkpoints.py` on the full `defeature_processed/test` split — 157
hand-labelled real parts (`navin_defeaturing`), 13,445 labelled faces. This is the
number that matters for `defeature_v2.py`, which consumes `face_preds`.

Every checkpoint below can be downloaded from HuggingFace. See [Model Weights](../README.md#model-weights) for the links and local paths.

| rank | checkpoint | macro F1 | weighted F1 | worst class | random | hole | chamfer | fillet | cut |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `trial1_finetune_ss/best-epoch=32` | **0.8886** | **0.8951** | **0.820** | 0.820 | 0.954 | 0.874 | 0.887 | 0.908 |
| 2 | `trial1_finetune_ss/last` | 0.8881 | 0.8924 | 0.814 | 0.814 | 0.954 | 0.882 | 0.887 | 0.904 |
| 3 | `trial1_finetune_ss/best-epoch=26` | 0.8853 | 0.8900 | 0.811 | 0.811 | 0.956 | 0.875 | 0.885 | 0.900 |
| 4 | `trial1_ss/last` | 0.8789 | 0.8845 | 0.811 | 0.811 | 0.960 | 0.862 | 0.873 | 0.888 |
| 5 | `trial1_ss1500/best-epoch=43` | 0.8770 | 0.8889 | 0.810 | 0.810 | 0.972 | 0.839 | 0.868 | 0.896 |
| 7 | `trial2_finetune_ss/last` | 0.8662 | 0.8735 | 0.788 | 0.788 | 0.946 | 0.854 | 0.856 | 0.887 |
| 13 | `trial1_ss1500/best-epoch=57` ← `defeature_v2.py` default | 0.8564 | 0.8619 | 0.778 | 0.778 | 0.976 | 0.836 | 0.839 | 0.853 |
| 21 | `trial5/best-epoch=43` (27-class, collapsed) | 0.6344 | 0.6385 | 0.348 | 0.348 | 0.809 | 0.720 | 0.579 | 0.715 |

**`DEFAULT_CHECKPOINT` in `defeature_v2.py` ranks 13th of 26.** Switching to
`trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt` is +3.2 points macro F1 and lifts
the worst class from 0.778 to 0.820, at zero training cost.

**Checkpoint filenames rank the wrong metric.** `ModelCheckpoint` monitors `val/f1`,
which is model-level multi-label F1, not per-face segmentation. The two orderings
disagree badly: `trial1_finetune_ss` posts a *lower* `val/f1` (0.9354) than
`trial1_ss1500` (0.9415) while segmenting faces substantially better. `train_preprocessed.py`
now also writes `bestface-{epoch}-{val/face_f1}` checkpoints when `--face_segmentation`
is set, so future runs can be compared on the right axis.

**Fine-tuning from a 27-class model beats direct 5-class training.** All seven
`*_finetune_ss` checkpoints outrank every direct 5-class checkpoint. The earlier
conclusion that the fine-tune route had "lost" came from comparing `val/f1` and is wrong
for face segmentation.

Caveats worth keeping: 28 of 58 saved checkpoints (`trial2`–`trial4`, `brepformer`,
`face_seg_heavy`) are architecturally obsolete — they use BatchNorm and a single FFN per
layer, where the current `BrepClassifier` uses LayerNorm and dual FFNs — so they cannot
be loaded and are excluded, not scored. `pipe_classifier` (4 more) is a different model
class. And these checkpoints trained on `defeature/train`, so this split is in-domain for
them; it is genuinely held out, but it is not a cross-dataset generalization measurement.

Reproduce:

```bash
python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/trial1_ss results/trial1_ss1000 results/trial1_ss1500 \
                results/trial1_finetune_ss results/trial2_finetune_ss results/trial5 \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
    --out brepformer/analysis_results/ckpt_scan_defeature_test.json
```

### Known Issue: Inference on Unknown STEP Files

The current model was trained on MFTRCAD dataset graph JSONs which use a different
face UV-grid sampling algorithm than `step_to_graph.py`. For **dataset models**, the
inference pipeline loads preprocessed pickles and achieves full accuracy. For
**unknown STEP files** not in the dataset, inference falls back to `step_to_graph()`
which produces mismatched UV-grids, degrading face accuracy to ~28%. To fix this
for production, repreprocess the training data using `step_to_graph()` and retrain.
See `pipeline_analysis.md` Issues #28-31 for details.

---

## Installation

### Prerequisites

```bash
# Create conda environment (recommended)
conda create -n brep_mfr python=3.9
conda activate brep_mfr

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install dependencies
pip install pytorch-lightning==1.7.7
pip install torchmetrics
pip install tensorboard
pip install tqdm
pip install numpy
pip install scipy

# Optional: for visualization and analysis
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"
```

---

## Quick Start

### 1. Preprocess Data

```bash
# Preprocess raw data to pickle format (required for multi-worker training)
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed
```

### 2. Train Model

```bash
# Train with preprocessed data
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 100 \
    --batch_size 32 \
    --num_workers 4
```

### 3. Monitor Training

```bash
# Launch TensorBoard
python brepformer/tensorboard_server.py --log_dir results
```

### 4. Evaluate Model

```bash
# Test on test set
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt
```

---

## Project Structure

```
brepformer/
├── configs/
│   ├── __init__.py
│   └── config.py                 # BrepClassifierConfig dataclass (89 hyperparameters)
│
├── models/
│   ├── __init__.py
│   ├── brep_classifier.py        # Main PyTorch Lightning module
│   ├── brep_encoder.py           # Core transformer encoder
│   ├── pooling.py                # Graph pooling strategies (cls/mean/max/attention)
│   └── layers/
│       ├── __init__.py
│       ├── attention.py          # Multi-head attention with RoPE and GQA
│       ├── blocks.py             # RMSNorm, SwiGLU, MLP, NonLinearClassifier
│       ├── embedding.py          # SurfaceEncoder, CurveEncoder, GraphNodeFeature, GraphAttnBias
│       └── encoder_layer.py      # Single transformer layer
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                # MFTRCADDataset (loads raw JSON)
│   ├── preprocessed_dataset.py   # PreprocessedDataset (loads pickle)
│   ├── collator.py               # BrepCollator for batching
│   ├── preprocessing.py          # Graph feature computation
│   ├── classes.py                # 27-class names + colors (single source of truth)
│   └── step_to_graph.py          # STEP → graph conversion (pythonOCC)
│
├── data/mftrcad/                 # Raw dataset
│   ├── graphs/                   # JSON graph files (~28,555 files)
│   ├── labels/                   # Per-face label files
│   └── steps/                    # STEP CAD files
│
├── data/mftrcad_processed/       # Preprocessed dataset (after running preprocess.py)
│   ├── train.pkl                 # Training samples (~3.3GB)
│   ├── val.pkl                   # Validation samples (~400MB)
│   ├── test.pkl                  # Test samples (~400MB)
│   └── metadata.json             # Dataset metadata
│
├── data/defeature/               # Defeature dataset (5 classes, 1561 models)
│   ├── steps/                    # STEP CAD files
│   ├── labels/                   # Per-face JSON labels (remapped 0-4)
│   ├── graphs/                   # Graph JSONs (after step_to_graph conversion)
│   └── dataset_info.json         # Dataset metadata
│
├── data/defeature_processed/     # Preprocessed defeature data (after preprocess.py)
│
├── preprocess.py                 # Data preprocessing script
├── train.py                      # Training with raw data (single-worker)
├── train_preprocessed.py         # Training with preprocessed data (multi-worker)
├── test.py                       # Testing with raw data
├── test_preprocessed.py          # Testing with preprocessed data
├── analyze.py                    # Model and prediction analysis (+ face segmentation mode)
├── infer.py                      # End-to-end STEP → predictions inference
├── export_freecad.py             # Colored STEP export for FreeCAD
├── defeature.py                  # Automatic defeaturing (removes predicted features from STEP)
├── visualize_seg.py              # Qt+pythonOCC face segmentation viewer
├── visualize_defeature.py        # Qt+pythonOCC original vs defeatured comparison viewer
├── visualize.py                  # Visualization from analysis results
├── tensorboard_server.py         # TensorBoard launcher
├── export_model.py               # Model export (state_dict/TorchScript/ONNX)
├── scripts.md                    # Scripts reference documentation
├── viewer.md                     # Visualization & FreeCAD workflow docs
└── README.md                     # This file
```

---

## Model Architecture

### Overview

```
Input B-rep Graph
       │
       ▼
┌─────────────────────────────────────┐
│ GraphNodeFeature                    │
│  - SurfaceEncoder: 2D CNN [7,10,10] │
│  - Face attribute encoders          │
│  - Virtual [CLS] token              │
│  Output: [batch, N+1, 256]          │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│ GraphAttnBias (KEY INNOVATION)      │
│  - Spatial position embedding       │
│  - D2/angle descriptors             │
│  - CurveEncoder on edge grids       │
│  - Multi-hop edge aggregation       │
│  Output: [batch, heads, N+1, N+1]   │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│ BrepEncoder (8 transformer layers)  │
│  For each layer:                    │
│    - RMSNorm (pre-norm)             │
│    - MultiheadAttention + RoPE      │
│    - Attention bias injection       │
│    - SwiGLU FFN                     │
│  Output: [batch, N+1, 256]          │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│ GraphPooling (CLS token)            │
│  Output: [batch, 256]               │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│ NonLinearClassifier                 │
│  256 → 512 → 512 → 256 → num_classes│
│  with BatchNorm + ReLU + Dropout    │
│  Output: [batch, 27]                │
└─────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **SurfaceEncoder** | 3-layer 2D CNN (7→64→128→256→128) on face UV-grids |
| **CurveEncoder** | 3-layer 1D CNN (12→64→128→num_heads) on edge grids |
| **GraphAttnBias** | Converts edge/topology features to attention bias |
| **RoPE** | Rotary Position Embeddings for sequence modeling |
| **GQA** | Grouped Query Attention (32 Q heads, 8 KV heads) |
| **RMSNorm** | Root Mean Square normalization (not LayerNorm) |
| **SwiGLU** | Swish-Gated Linear Unit FFN |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 256 | Embedding dimension |
| `ffn_dim` | 512 | FFN hidden dimension |
| `num_layers` | 8 | Number of transformer layers |
| `num_heads` | 32 | Number of attention heads |
| `num_kv_heads` | 8 | Number of KV heads (GQA) |
| `num_classes` | 27 | Number of machining feature classes |
| `dropout` | 0.3 | Dropout probability |
| `learning_rate` | 0.002 | Initial learning rate |
| `warmup_steps` | 5000 | LR warmup steps |
| `batch_size` | 32 | Training batch size |

---

## Data Format

### Input Graph (JSON)

Each B-rep model is stored as a JSON file with format:
```json
[
  "model_name",
  {
    "graph": {
      "edges": [[src1, src2, ...], [dst1, dst2, ...]],
      "num_nodes": N
    },
    "graph_face_attr": [[14 attributes per face], ...],
    "graph_face_grid": [[[7 channels, 10x10 UV grid]], ...],
    "graph_edge_attr": [[15 attributes per edge], ...],
    "graph_edge_grid": [[[12 channels, 10 points]], ...]
  }
]
```

### Feature Dimensions

| Feature | Shape | Description |
|---------|-------|-------------|
| `face_grid` | (N, 7, 10, 10) | UV-sampled points + normals |
| `face_attr` | (N, 14) | Face type, area, centroid, etc. |
| `edge_grid` | (E, 12, 10) | Curve-sampled points + tangents |
| `edge_attr` | (E, 15) | Edge type, length, angles, etc. |
| `spatial_pos` | (N+1, N+1) | Shortest path distances |
| `label` | (27,) | Multi-hot machining feature vector |

### Label Format

**Multi-label (default)**: Derived from per-face annotations, each model has a multi-hot vector of 27 machining features.

**Single-label (optional)**: Use `--label_file` to provide external whole-model labels:
```json
{
  "model_id_1": 0,
  "model_id_2": 5,
  ...
}
```

---

## Scripts Reference

### Data Preprocessing

#### `preprocess.py`

Converts raw JSON data to pickle format for efficient multi-worker loading.

```bash
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed \
    --num_classes 27 \
    --split_ratio "0.8,0.1,0.1" \
    --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Input directory with graphs/ and labels/ |
| `--output_dir` | required | Output directory for pickle files |
| `--label_file` | None | External labels JSON (optional) |
| `--num_classes` | 27 | Number of classes |
| `--compute_descriptors` | False | Compute D2/angle descriptors (slower) |
| `--num_spatial` | 64 | Max spatial distance |
| `--d2_bins` | 64 | D2 descriptor bins |
| `--angle_bins` | 64 | Angle descriptor bins |
| `--split_ratio` | "0.8,0.1,0.1" | Train/val/test split |
| `--seed` | 42 | Random seed |

---

### Training

#### `train_preprocessed.py` (Recommended)

Training with preprocessed pickle data. Supports multi-worker data loading.

```bash
# Basic training
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 100

# Full configuration
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.002 \
    --num_workers 4 \
    --hidden_dim 256 \
    --num_layers 8 \
    --num_heads 32 \
    --dropout 0.3 \
    --output_dir results \
    --exp_name my_experiment \
    --devices 1 \
    --precision "32"

# Resume from checkpoint
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --resume_from results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --max_epochs 200

# Fast development test
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --fast_dev_run
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Preprocessed data directory |
| `--num_classes` | auto | Number of classes (from metadata) |
| `--multi_label` | True | Multi-label classification |
| `--hidden_dim` | 256 | Hidden dimension |
| `--ffn_dim` | 512 | FFN dimension |
| `--num_layers` | 8 | Transformer layers |
| `--num_heads` | 32 | Attention heads |
| `--num_kv_heads` | 8 | KV heads (GQA) |
| `--dropout` | 0.3 | Dropout probability |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 0.002 | Learning rate |
| `--max_epochs` | 200 | Maximum epochs |
| `--warmup_steps` | 5000 | LR warmup steps |
| `--gradient_clip_val` | 1.0 | Gradient clipping |
| `--num_workers` | 4 | Data loader workers |
| `--accumulate_grad_batches` | 1 | Gradient accumulation |
| `--output_dir` | "results" | Output directory |
| `--exp_name` | "brepformer" | Experiment name |
| `--seed` | 42 | Random seed |
| `--devices` | 1 | Number of GPUs |
| `--precision` | "32" | Training precision ("16" or "32") |
| `--fast_dev_run` | False | Quick test run |
| `--resume_from` | None | Checkpoint to resume |

#### `train.py` (Alternative)

Training with raw JSON data. Use `num_workers=0` only.

```bash
python brepformer/train.py \
    --data_dir brepformer/data/mftrcad \
    --max_epochs 100 \
    --num_workers 0

# With external labels (single-label)
python brepformer/train.py \
    --data_dir brepformer/data/mftrcad \
    --label_file path/to/labels.json \
    --num_classes 10 \
    --no_multi_label
```

---

### Testing

#### `test_preprocessed.py`

Evaluate model on test set.

```bash
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --batch_size 32 \
    --num_workers 4 \
    --output_file test_results.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Preprocessed data directory |
| `--checkpoint` | required | Model checkpoint path |
| `--batch_size` | 32 | Batch size |
| `--num_workers` | 4 | Data loader workers |
| `--output_file` | None | Output JSON file |

---

### Analysis

#### `scan_checkpoints.py`

Ranks every saved checkpoint of one or more runs by **face-level** F1 on a held-out
split. Use it before trusting a `best-*.ckpt` filename: those are ranked by model-level
`val/f1`, which orders epochs differently (see
[Face Segmentation on Real Parts](#face-segmentation-on-real-parts-defeature-5-class) —
the shipped default ranked 13th of 26). The test split is loaded and prepared once and
reused across checkpoints, so scanning ~30 of them costs about as much as 2 or 3
`eval_per_class.py` runs.

Reports macro F1 (every class counts equally, so a dead rare class shows up) next to
weighted F1 (dominated by `cut`/`hole`) — a run can gain one and lose the other.

```bash
python -m brepformer.scan_checkpoints \
    --ckpt_dirs results/trial1_ss1500 results/trial1_finetune_ss \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
    --out brepformer/analysis_results/ckpt_scan_defeature_test.json
```

| Argument | Default | Description |
|---|---|---|
| `--ckpt_dirs` | required | Run directories; searched recursively (`val/f1` in the filename makes each checkpoint a *directory*) |
| `--data_dir` | `defeature_processed` | Preprocessed corpus to score against |
| `--split` | test | Split to score |
| `--limit` | none | Cap test models; leave unset — subsampling reorders near-ties |
| `--label_space` | 5 | `5` collapses a 27-class head's predictions via `CLASS_TO_DEFEATURE`; targets are left alone if the corpus is already 5-class |
| `--skip_last` | False | Exclude `last*.ckpt`; included by default since the metric that pruned the others was the wrong one |

#### `eval_per_class.py`

Per-class precision/recall/F1 for a single checkpoint, plus what each weak class is
confused *with*, its mean face area, and the model's confidence when it predicts it.
`--focus <class_id>` breaks recall down by face-area decile — the number that predicts
whether a feature class transfers to real parts, since a class can score well overall
and still be dead in deployment if training only showed it at one scale.

```bash
python -m brepformer.eval_per_class \
    --checkpoint "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --data_dir brepformer/data/defeature_processed --split test --label_space 5

# why is chamfer weak? recall vs face area
python -m brepformer.eval_per_class \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/mftrcad_processed --split test --focus 26
```

#### `analyze.py`

Comprehensive model and prediction analysis.

```bash
# Architecture analysis
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --mode architecture \
    --output_dir analysis_results

# Per-class performance
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode per_class \
    --output_dir analysis_results

# Embedding visualization (t-SNE)
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode embeddings \
    --num_samples 1000 \
    --output_dir analysis_results

# Detailed predictions
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode predictions \
    --output_dir analysis_results

# Run all analyses
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode all \
    --output_dir analysis_results
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Model checkpoint |
| `--data_dir` | None | Data directory (required for most modes) |
| `--mode` | "architecture" | Analysis mode |
| `--output_dir` | "analysis_results" | Output directory |
| `--batch_size` | 32 | Batch size |
| `--num_workers` | 4 | Data loader workers |
| `--num_samples` | 1000 | Samples for embedding analysis |
| `--split` | "test" | Data split to analyze |

**Output files:**
- `architecture.json` - Model architecture details
- `per_class_metrics.json` - Per-class precision/recall/F1
- `embeddings.npy` - Raw embeddings
- `embeddings_tsne.npy` - t-SNE embeddings
- `predictions.json` - Detailed predictions

---

### Interactive Viewer

#### `viewer.py` (project root)

Interactive 3D point cloud viewer for browsing MFTRCAD models with ground-truth labels, model predictions, and analysis overlays. Builds point clouds from face UV-grids, supports per-face GT coloring, model-level prediction coloring, side-by-side comparison, and non-blocking matplotlib popups for metrics/confusion/embeddings.

```bash
# Browse test split
./run_viewer.sh browse --split test --sort index

# View predictions sorted by worst accuracy
./run_viewer.sh predictions --sort worst

# Analysis plots only (no 3D viewer needed)
./run_viewer.sh analysis --metrics
./run_viewer.sh analysis --confusion
./run_viewer.sh analysis --embeddings
```

See [viewer.md](viewer.md) for full documentation including keyboard controls, CLI arguments, visualization details, and troubleshooting.

---

### Visualization

#### `visualize.py`

Generate plots from analysis results.

```bash
# Embedding visualization
python brepformer/visualize.py \
    --mode embeddings \
    --input_dir analysis_results \
    --output_dir plots

# Per-class metrics
python brepformer/visualize.py \
    --mode metrics \
    --input_dir analysis_results \
    --output_dir plots

# Training curves
python brepformer/visualize.py \
    --mode training \
    --log_dir results/brepformer \
    --output_dir plots

# All visualizations
python brepformer/visualize.py \
    --mode all \
    --input_dir analysis_results \
    --log_dir results/brepformer \
    --output_dir plots \
    --format png \
    --dpi 150
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | "all" | Visualization mode |
| `--input_dir` | "analysis_results" | Analysis results directory |
| `--log_dir` | "results/brepformer" | TensorBoard logs |
| `--output_dir` | "plots" | Output directory |
| `--format` | "png" | Image format (png/pdf/svg) |
| `--dpi` | 150 | Image DPI |

---

### TensorBoard

#### `tensorboard_server.py`

Launch TensorBoard for training monitoring.

```bash
# Basic usage
python brepformer/tensorboard_server.py --log_dir results

# Custom port
python brepformer/tensorboard_server.py --log_dir results --port 8080

# Remote access (bind to all interfaces)
python brepformer/tensorboard_server.py --log_dir results --host 0.0.0.0

# Compare multiple experiments
python brepformer/tensorboard_server.py --log_dir results --compare
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--log_dir` | "results" | Log directory |
| `--port` | 6006 | TensorBoard port |
| `--host` | "localhost" | Host to bind |
| `--compare` | False | Experiment comparison mode |
| `--reload_interval` | 30 | Reload interval (seconds) |

**Alternative: Direct TensorBoard**
```bash
tensorboard --logdir results --port 6006
```

---

### Model Export

#### `export_model.py`

Export trained models for deployment.

```bash
# Export state dict (PyTorch)
python brepformer/export_model.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --format state_dict \
    --output_path model_weights.pt

# Export TorchScript
python brepformer/export_model.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --format torchscript \
    --output_path model_scripted.pt \
    --max_faces 100 \
    --max_edges 200

# Export ONNX
python brepformer/export_model.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --format onnx \
    --output_path model.onnx
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Model checkpoint |
| `--format` | "state_dict" | Export format |
| `--output_path` | required | Output file path |
| `--max_faces` | 100 | Max faces for tracing |
| `--max_edges` | 200 | Max edges for tracing |

---

## Configuration

### BrepClassifierConfig

All hyperparameters are defined in `configs/config.py`:

```python
from brepformer.configs.config import BrepClassifierConfig

config = BrepClassifierConfig(
    # Model architecture
    hidden_dim=256,
    ffn_dim=512,
    num_layers=8,
    num_heads=32,
    num_kv_heads=8,

    # Classification
    num_classes=27,
    multi_label=True,

    # Regularization
    dropout=0.3,
    attention_dropout=0.3,
    activation_dropout=0.3,

    # Training
    batch_size=32,
    learning_rate=0.002,
    warmup_steps=5000,
    max_epochs=200,
)
```

---

## Usage Examples

### Complete Training Pipeline

```bash
# 1. Preprocess data
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed

# 2. Start TensorBoard (in separate terminal)
python brepformer/tensorboard_server.py --log_dir results &

# 3. Train model
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --exp_name my_experiment

# 4. Test model
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/my_experiment/last.ckpt

# 5. Analyze results
python brepformer/analyze.py \
    --checkpoint results/my_experiment/last.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode all \
    --output_dir analysis_my_experiment

# 6. Generate visualizations
python brepformer/visualize.py \
    --mode all \
    --input_dir analysis_my_experiment \
    --log_dir results/my_experiment \
    --output_dir plots_my_experiment
```

### Training with External Labels

```bash
# For single-label classification with external labels
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_custom \
    --label_file path/to/my_labels.json \
    --num_classes 10

python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_custom \
    --num_classes 10 \
    --no_multi_label \
    --max_epochs 100
```

### Hyperparameter Tuning

```bash
# Experiment 1: Larger model
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --hidden_dim 512 \
    --num_layers 12 \
    --exp_name large_model

# Experiment 2: Higher dropout
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --dropout 0.5 \
    --exp_name high_dropout

# Experiment 3: Different learning rate
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --learning_rate 0.001 \
    --exp_name lr_0001
```

---

## Face-Level Segmentation

BrepFormer supports an optional **face-level segmentation head** that predicts per-face machining feature classes (27 MFTRCAD classes) in addition to whole-model classification.

### Multi-Task Architecture

```
BrepEncoder (8 transformer layers)
  Output: [batch, N+1, 256]  (node_emb)
         │
         ├── GraphPooling ([CLS] token) ──> NonLinearClassifier ──> model_logits [batch, 27]
         │
         └── Face embeddings (pos 1..N) ──> FaceSegmentationClassifier ──> face_logits [batch, N, 27]

Combined Loss = model_cls_weight * model_loss + face_seg_weight * face_loss
```

### Training with Face Segmentation

```bash
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --face_segmentation \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --max_epochs 100
```

### Inference on STEP Files

```bash
# Single file inference
python -m brepformer.infer \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output preds.seg

# Batch inference
python -m brepformer.infer \
    --step_dir brepformer/data/sample/steps/ \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output_dir face_inference_results/

# With 8 grouped machining feature categories (instead of 27)
python -m brepformer.infer \
    --step brepformer/data/defeature/steps/uh62382.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output preds.seg --8_classes

# With 5 defeature categories (random, hole, chamfer, fillet, cut)
python -m brepformer.infer \
    --step brepformer/data/defeature/steps/uh62382.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output preds.seg --5_classes
```

### FreeCAD Colored Export

```bash
# Export colored STEP (per-face colors viewable in FreeCAD)
python -m brepformer.export_freecad --step model.step --seg preds.seg --output colored.step

# Export with 8-class colors
python -m brepformer.export_freecad --step model.step --seg preds.seg --output colored.step --8_classes

# Export with 5-class defeature colors
python -m brepformer.export_freecad --step model.step --seg preds.seg --output colored.step --5_classes

# From predicted .seg files in inference_results/
python -m brepformer.export_freecad \
    --step_dir brepformer/data/sample/steps/ \
    --seg_dir face_inference_results/ \
    --output_dir colored_steps/

# Or run inference and export in one step
python -m brepformer.export_freecad \
    --step model.step \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output colored.step
```

### Visualization

```bash
# Qt+pythonOCC viewer with 27-class display
python -m brepformer.visualize_seg --step model.step --seg preds.seg

# Viewer with 8-class grouped display
python -m brepformer.visualize_seg --step model.step --seg preds.seg --8_classes

# Viewer with 5-class defeature display
python -m brepformer.visualize_seg --step model.step --seg preds.seg --5_classes

# Browse dataset with GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --labels_dir brepformer/data/sample/labels/

# Browse with predicted .seg files from face inference
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --seg_dir face_inference_results/
```

### Automatic Defeaturing

```bash
# Defeature a single model
python -m brepformer.defeature --step model.step

# Batch defeaturing
python -m brepformer.defeature --step_dir steps/ --output_dir brepformer/defeatured_output/

# From pre-computed .seg predictions
python -m brepformer.defeature --step model.step --seg preds.seg --save_colored

# Export colored STEP + defeature in one command
python -m brepformer.export_freecad \
    --step model.step --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt \
    --defeature
```

### Face Segmentation Analysis

```bash
python brepformer/analyze.py \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/mftrcad_processed \
    --mode face_segmentation \
    --output_dir analysis_results
```

### Step Inference Analysis

Compares inference quality between preprocessed pickles (exact training match)
and on-the-fly STEP conversion (deployment path):

```bash
python brepformer/analyze.py \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/mftrcad_processed \
    --step_dir brepformer/data/mftrcad/steps \
    --mode step_inference \
    --max_models 50 \
    --output_dir analysis_results/trial5
```

See [scripts.md](scripts.md) for full CLI reference and [viewer.md](viewer.md) for visualization documentation.

---

## Inference Pipeline (Single STEP File)

`inference_pipeline.sh` is an end-to-end script that takes **any STEP file** and runs the full BrepFormer pipeline — from inference through defeaturing — saving all outputs to a single folder.

### Quick Start

```bash
# Basic: run the full pipeline on a STEP file
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/uh62382.step

# With full analysis (requires preprocessed dataset)
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/uh62382.step \
    --data_dir brepformer/data/defeature_processed

# Headless (skip interactive viewers)
./brepformer/scripts/inference_pipeline.sh brepformer/data/defeature/steps/uh62382.step --skip_viewers
```

### Pipeline Steps

| Step | Tool | Conda Env | Output |
|------|------|-----------|--------|
| 1. Inference | `brepformer.infer` | brepmfr | `preds.json`, `preds.seg` |
| 2. FreeCAD export | `brepformer.export_freecad` | brepmfr | `<name>_colored.step` |
| 3. Analysis | `brepformer.analyze --mode all` | brepmfr | `analysis/` |
| 4. Visualize seg | `brepformer.visualize_seg` | brepmfr | (interactive viewer) |
| 5. Defeature v2 | `brepformer.defeature_v2` | new_brepmfr | `<name>_defeatured.step`, `<name>_report.json` |
| 6. Visualize defeature | `brepformer.visualize_defeature` | new_brepmfr | (interactive viewer) |

The script automatically switches from the `brepmfr` conda environment to `new_brepmfr` before the defeature steps (which require pythonocc >= 7.9).

### Output Structure

```
brepformer/pipeline_output/<filename>/
├── preds.json                    # Full predictions (per-face probs, class names)
├── preds.seg                     # Per-face labels (one label per line)
├── <filename>_colored.step       # Colored STEP for FreeCAD
├── analysis/                     # Analysis outputs
│   ├── architecture.json         # Model architecture details
│   └── ...                       # (per_class, embeddings, etc. if --data_dir)
├── <filename>_defeatured.step    # Defeatured STEP (features removed)
├── <filename>_colored.step       # Colored STEP from defeature
└── <filename>_report.json        # Defeaturing report (removed/failed/valid)
```

### Note on the Analysis Step (Step 3)

The analysis step does **not** analyze your individual STEP file. It evaluates the model checkpoint against an entire preprocessed test dataset (e.g. `defeature_processed/test.pkl`), computing per-class precision/recall/F1, confusion matrices, t-SNE embeddings, face segmentation IoU, etc. This is useful for assessing the overall quality of the checkpoint you're using for inference.

- **Without `--data_dir`**: only architecture analysis runs (parameter counts, layer shapes, config) since there is no test set to evaluate against.
- **With `--data_dir`**: full analysis runs (`--mode all`) — the model is evaluated on every sample in the test split.

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `<step_file>` | required | Path to STEP (.step/.stp) file |
| `--checkpoint` | `results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt` | Model checkpoint |
| `--data_dir` | None | Preprocessed dataset dir for full analysis (e.g. `brepformer/data/defeature_processed`) |
| `--skip_viewers` | false | Skip interactive viewer steps (4 and 6) |
| `--output_dir` | `brepformer/pipeline_output/<filename>/` | Override output directory |

See [SCRIPTS.md](SCRIPTS.md) for the full CLI reference.

---

## Defeature Dataset (5 classes)

The defeature dataset (navin_defeaturing) contains 1561 industrial CAD models with per-face labels for 5 defeaturing categories:

| Class | Name | Distribution |
|-------|------|-------------|
| 0 | Random (Other) | 27.3% |
| 1 | Hole | 23.6% |
| 2 | Chamfer | 3.8% |
| 3 | Fillet | 14.9% |
| 4 | Cut | 30.5% |

### Quick Start (Defeature)

```bash
# 1. Prepare data (copy STEP files, remap 7-class labels to 5 classes)
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature

# 2. Convert STEP to graph JSONs (requires pythonocc)
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature \
    --convert_graphs

# 3. Preprocess (5 classes)
python brepformer/preprocess.py \
    --data_dir brepformer/data/defeature \
    --output_dir brepformer/data/defeature_processed \
    --num_classes 5 \
    --compute_descriptors

# 4. Train face segmentation model (5 classes)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_face_classes 5 \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir results \
    --exp_name defeature

# 5. Test
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/defeature/best-*.ckpt

# 6. Inference on new STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint results/defeature/best-*.ckpt \
    --output_dir inference/defeature/
```

### Limited Data Training (Subset Testing)

Use `--limit_data N` to train on a random subset of the preprocessed data without re-preprocessing. A manifest JSON is saved so that test, analyze, and infer scripts use the exact same subset.

```bash
# Train on 1000 samples from the defeature dataset
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation --num_face_classes 5 \
    --output_dir results --exp_name defeature_1k \
    --limit_data 1000

# Test/analyze/infer on the same subset using the manifest
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/defeature_1k/best-*.ckpt \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

python brepformer/analyze.py \
    --checkpoint results/defeature_1k/best-*.ckpt \
    --data_dir brepformer/data/defeature_processed \
    --mode all --output_dir results/defeature_1k/analysis \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint results/defeature_1k/best-*.ckpt \
    --output_dir results/defeature_1k/inference/ \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json
```

The `--limit_data` flag on `train_preprocessed.py` creates the manifest; downstream scripts accept `--limit_data_manifest` to filter to the same samples. When using `run_pipeline.py`, pass `--limit_data N` and the manifest is auto-propagated to all stages.

See [SCRIPTS.md](SCRIPTS.md) for the complete defeature pipeline reference.

### Automatic Defeaturing

`defeature.py` takes a STEP file (or directory of them), runs the BrepFormer model to classify every face into one of the 5 defeature categories, then uses OpenCASCADE to remove every face that isn't "random" — producing a clean stock shape with all manufacturing features erased.

#### How it works

1. **Load model** — loads the checkpoint (default: `results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt`). Verifies it has a face segmentation head. If a 27-class MFTRCAD model is used instead of a 5-class defeature model, predictions are automatically remapped via `CLASS_TO_DEFEATURE`.

2. **Read STEP and infer** — reads the STEP file with `STEPControl_Reader`, converts it to a graph via `step_to_preprocessed_sample`, and runs inference to get a per-face class (0-4). Can also skip inference and read pre-computed `.seg` files instead.

3. **Enumerate and group faces** — walks the shape with `TopExp_Explorer` in the same order the model uses, so prediction index *i* maps to face *i*. Faces predicted as `random` (0) are kept. The rest are grouped by type: hole (1), chamfer (2), fillet (3), cut (4).

4. **Progressive defeaturing** — calls `BRepAlgoAPI_Defeaturing` to remove feature faces. This does not just delete faces — it **extends adjacent faces to fill the gap**, which is what makes each operation geometrically correct:

   - **Holes** (circular, polygonal, blind, through): the hole wall faces and bottom face (if blind) are removed; the surrounding face is extended across the opening to fill it.
   - **Fillets**: the blend surface is removed; the two adjacent faces are extended until they intersect, restoring the original sharp edge.
   - **Chamfers**: same as fillets — the angled face is removed and adjacent faces are extended to meet at a sharp edge.
   - **Cuts** (slots, pockets, passages, steps): all faces forming the cavity are removed; surrounding faces are extended inward to fill the volume.

   The removal algorithm has four phases:

   | Phase | Strategy |
   |-------|----------|
   | **1. Batch** | Try removing ALL feature faces in a single call. If OCCT can heal the shape, this succeeds instantly. |
   | **2. Type-by-type** | If batch fails, progressively accumulate feature types in order: fillet, chamfer, hole, cut. The order is strategic — fillets and chamfers (smooth blends) are easiest for the kernel to heal; cuts are largest and most likely to cause failures. |
   | **3. Connected components** | For any type that fails as a group, find connected components (faces sharing edges). Multi-face features like blind holes (cylinder + bottom disk) or rectangular pockets (4 walls + floor) are tried as complete units — removing half a feature always fails. |
   | **4. Second pass** | After some features are removed, retry the remaining ones on the modified shape. Removing a fillet may unblock an adjacent hole that previously couldn't be healed. |

   Phases 1-3 operate on the **original shape** with an accumulating face set, so face references never go stale. Phase 4 re-enumerates on the modified shape.

5. **Heal, unify, and validate** — runs `ShapeFix_Shape` to repair geometric imprecisions, `ShapeUpgrade_UnifySameDomain` to merge faces that now lie on the same surface (e.g. a plate face that was split by a hole is unified back into one face), then `BRepCheck_Analyzer` to confirm the result is a valid solid.

6. **Write output** — writes the defeatured shape as a new STEP file. Saves a JSON report with face counts (kept, removed, failed) and validity status.

#### Quick start

```bash
# Defeature a single STEP file
python -m brepformer.defeature --step model.step

# Defeature a directory of STEP files
python -m brepformer.defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --output_dir brepformer/defeatured_output/

# From pre-computed .seg predictions (skip inference)
python -m brepformer.defeature --step model.step --seg preds.seg

# Save a colored STEP alongside the defeatured one for comparison
python -m brepformer.defeature --step model.step --save_colored --verbose
```

Output goes to `brepformer/defeatured_output/` by default. The `--defeature` flag is also available in `export_freecad.py` and `visualize_seg.py` (press **F** key in the viewer).

See [SCRIPTS.md](SCRIPTS.md) for full CLI reference.

### Defeaturing Visual Comparison

Compare original and defeatured models side by side:

```bash
# Browse all defeatured results with GT labels
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --labels_dir brepformer/data/defeature/labels/

# Single model comparison
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/uh62382.step \
    --defeatured brepformer/defeatured_output/uh62382_defeatured.step

# With predictions on the original model
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --seg_dir inference/trial1_ss1500/
```

See [SCRIPTS.md](SCRIPTS.md) for the full CLI reference.

---

## Statistics Explanation

The `preds/*/test_face_results.json` and `preds/*/train_face_results.json` files contain the following metrics produced by evaluating a trained model on a dataset split.

### Metric definitions

| Key | Description |
|-----|-------------|
| `loss/model` | Cross-entropy loss for the **model-level** classification head — how well the model predicts the overall machining-feature class of the entire CAD model. |
| `loss/face` | Cross-entropy loss for the **face-level** segmentation head — how well the model assigns a machining-feature class to each individual B-rep face. |
| `test/loss` | Combined total loss: `model_cls_weight × loss/model + face_seg_weight × loss/face`. |
| `test/acc` | **Model-level accuracy** — fraction of whole CAD models whose predicted class matches the ground-truth label. Each model counts as a single sample regardless of how many faces it has. |
| `test/f1` | Macro-averaged F1 score across all model-level classes. Every class contributes equally to the average, independent of its frequency in the dataset. |
| `test/precision` | Macro-averaged precision across all model-level classes. |
| `test/recall` | Macro-averaged recall across all model-level classes. |
| `test/face_acc` | **Face-level accuracy** — fraction of individual B-rep faces (across every model in the split) whose predicted machining-feature class matches the ground-truth label. |
| `test/face_f1` | Macro-averaged F1 across all face-level classes. |

In `train_face_results.json` the same metrics appear under the same `test/*` keys (the evaluation is run through the model's test step regardless of which data split is used); the file name distinguishes the split.

### Model-level vs Face-level

The two granularities measure fundamentally different things:

- **Model-level** (`test/acc`, `test/f1`, `test/precision`, `test/recall`): the entire CAD model receives one predicted label. A model is "correct" when that single label matches the ground truth. This is a coarse signal — it collapses all face-level detail into one number.
- **Face-level** (`test/face_acc`, `test/face_f1`): every face on every model receives its own predicted label. Accuracy and F1 are computed across the full population of faces, so a model with 100 faces contributes 100 data points rather than 1.

Because many CAD models are dominated by a single feature class, model-level accuracy (`test/acc`) tends to be higher than face-level accuracy (`test/face_acc`). The face-level metric is the stricter of the two: the model must correctly segment each face even when multiple feature classes coexist on a single model. Face-level macro F1 (`test/face_f1`) is especially informative because rare feature classes on individual faces are weighted equally to common ones, making it sensitive to under-represented geometry.

Comparing train vs test values (e.g. `train_face_results.json` vs `test_face_results.json`) reveals overfitting: a large gap between `train/face_acc` and `test/face_acc` indicates the model has memorised training-set faces but does not generalise to unseen geometries.

## Feature Representation: What the Model Actually Sees

This section documents the B-rep feature layout, because a mismatch between the layout the
data uses and the layout the model reads was found on 2026-09-01 and is the largest single
correctness issue in the codebase. Anyone touching `embedding.py`, `step_to_graph.py` or
`preprocess.py` needs this table.

### The MFTRCAD attribute layout

`normalize_geometry` in `data/step_to_graph.py` converts STEP-derived geometry into the
layout MFTRCAD itself uses. This was verified against the raw
`data/mftrcad/graphs/*_result.json` files, which are bit-identical to the preprocessed
pickles — so the layout below is the dataset's own format and the preprocessing is correct:

```
face_attr[14] = [plane, cylinder, cone, sphere, torus, other,   # one-hot surface type
                 0, 0, 0,                                        # unused
                 area,                                           # col 9, normalised by scale^2
                 0,                                              # unused
                 cx, cy, cz]                                     # cols 11-13, centroid

edge_attr[15] = [0, convex, smooth,                              # convexity one-hot (1, 2, 4)
                 length,                                         # col 3, normalised by scale
                 concave,
                 circle, 0, line, other_type,                    # curve-type one-hot (5, 7, 8)
                 0, 0, 0, 0, 0, 0]                               # unused
```

Two consequences are worth internalising. First, the layout is **sparse and positional**,
not packed — a reader that assumes `[type, area, cx, cy, cz, ...]` will silently get
one-hot bits where it expects floats, with no shape error to catch it. Second, the format
**discards** attributes that `step_to_graph` computed upstream: face normals, bounding-box
extents, rationality, loop count, edge dihedral angle, midpoint, tangent and curvature are
all extracted and then dropped. They cannot be recovered from a pickle; restoring them
means changing `normalize_geometry` and re-preprocessing.

### The mismatch (found 2026-09-01, **fixed 2026-09-04** — see "The fix, as applied" below)

`GraphNodeFeature.forward` (`models/layers/embedding.py:243`) documents and reads an
index-based layout — `[type, area, cx, cy, cz, rational, n_loops, nx, ny, nz, bbx, bby,
bbz, reversed]` — that no data in this repo has ever had. Because both layouts are 14 wide,
nothing raises. What each encoder receives in practice:

| Encoder | Intended input | Actual input |
|---|---|---|
| `face_type_encoder` | surface type index 0-5 | col 0, the *plane* bit — **6 types collapse to plane/not-plane** |
| `face_area_encoder` | face area | col 1, the cylinder bit |
| `face_centroid_encoder` | `(cx, cy, cz)` | cols 2:5, the cone/sphere/torus bits (~96% zero) |
| `face_rationality_encoder` | 0/1 | col 5, constant 0 |
| `face_loop_encoder` | loop count | col 6, constant 0 |
| `face_normal_encoder` | `(nx, ny, nz)` | cols 7:10 = `[0, 0, area]` |
| `face_bbox_encoder` | bbox extent | cols 10:13 = `[0, cx, cy]` |
| `face_reversed_encoder` | 0/1 | col 13 = `cz`, truncated to int then clamped to 0..1 |
| `edge_type_encoder` | curve type index | col 0 + 1, effectively binary |
| `edge_len_encoder` | edge length | col 1, the convex flag (real length is col 3) |
| `edge_ang_encoder` | dihedral angle | col 8, the other-type flag (angle is not in the format at all) |
| `edge_conv_encoder` | convexity | col 9, **constant 0 — convexity is entirely dead** |

Ranked by how much it costs:

1. **Convexity is dead.** Whether a blend is convex or concave is the single most
   diagnostic signal separating a fillet from a chamfer or a pocket wall, and the model
   never receives it.
2. **Surface type is binary.** Distinguishing a cylinder from a cone from a torus is what
   separates `hole` from `chamfer` from `fillet`; the model only learns "planar or not".
3. **The D2 descriptor channel is empty** (fixed 2026-09-04; see "The D2 channel" below).
   `preprocess.py:346` reads the same wrong
   columns as centroids, so pairwise distances are computed between one-hot bits. Measured
   over 20 pickles per corpus: 51% (mftrcad) / 78% (trial6) of face pairs put all their D2
   mass in bin 0, and the remainder are uniform 1/64 across all 64 bins. Either way the
   channel transmits nothing.

The model still reaches ~0.94 model-level F1 because `face_grid` — 7 channels of UV-sampled
positions, normals and a mask — feeds a CNN (`SurfaceEncoder`) that recovers most of the
geometry independently of these attributes. **The attributes are not carrying the model;
they are close to inert.** That also means every trial number recorded so far is a floor
rather than a ceiling.

**Why this is not a trial6 confound.** The defect is byte-identical across every arm,
treatment and control alike, so the trial6/trial6a A/B comparisons remain valid — they
measure what the pcq corpus contributes over a uniformly degraded representation. Do not
discard the sweep over this.

**Why the fix was deferred.** `trial6_sweep.py` launches a fresh `train_preprocessed.py`
process per arm. Editing the model mid-sweep would give later arms a different architecture
from earlier ones and destroy the comparison. The fix also changes what the learned weights
mean, so warm-start arms have to be re-run rather than fine-tuned. Sequence: finish the
sweep, then apply and re-run — which is what happened.

### The fix, as applied (2026-09-04)

**No re-preprocessing was needed.** I had assumed it would be; that was wrong. Column-by-column
inspection of `face_attr`/`edge_attr` across `mftrcad_processed`, `defeature_processed` and
`trial6_processed` confirmed every field the encoder wants is *already on disk*, just at other
indices. Only the reader was broken.

`GraphNodeFeature` and `GraphAttnBias` now take an `attr_layout`, driven by
`BrepClassifierConfig.attr_layout` and a `--attr_layout {onehot,categorical}` training flag:

- `onehot` (default, correct): type from `argmax(face_attr[..., 0:6])` with a 7th index for
  all-zero rows, area from col 9, centroid from cols 11:14, edge length from col 3, edge type
  from the line/circle/other bits, convexity from the convex/smooth/concave bits.
- `categorical`: the historical indexing, kept **only** so pre-fix checkpoints can be scored
  under the inputs they were actually trained on.

Fields `normalize_geometry` genuinely discards — normals, bbox, loops, rationality, orientation,
dihedral angle — are fed **explicit zeros** rather than reinterpreted from unrelated columns. The
encoder then contributes a constant bias, which is honest about the absence and keeps every
parameter shape identical, so existing checkpoints still load (243/243 tensors, 0 skipped).

**Old checkpoints must keep the old decoding, and the difference is enormous.** On 12 models /
516 faces the incumbent `trial1_finetune_ss` ep32 scores **82.95% under `categorical` and 32.75%
under `onehot`**. Flipping the default without guarding it would have silently destroyed every
number in this file. `infer.load_model` therefore auto-detects the era.

**The detection trap.** `getattr(saved_config, "attr_layout")` cannot detect a pre-fix config:
unpickling a dataclass rebuilds it against the *current* class definition, so the new field's
default is returned for configs pickled before the field existed. The first version of the
detection chose `onehot` for old checkpoints because of this and was caught only by scoring.
Probe `vars(saved).get("attr_layout")` — the instance `__dict__` holds only keys really restored.

The consequence for training is that a warm start across layouts is not a fine-tune: the
attribute encoders are handed inputs whose meaning changed and must re-adapt, so the A/B run
starts at face F1 **0.3266**, not ~0.88, and has to climb back past the incumbent's **0.8886** on
`defeature_processed/test` to prove the correct attributes are worth more than the body's
adaptation to the broken ones. That run (`fix4_onehot_def`, same corpus / init / hyperparameters
as `t6ctl_init_ft`, layout the only difference) has since finished — see the verdict below.


### The A/B verdict: the correct decoding is worse, and it loses on the right classes (2026-09-05)

`fix4_onehot_def` — the incumbent's exact configuration with nothing changed but the layout —
early-stopped at epoch 106 having climbed from face F1 0.3266 at epoch 0 to **0.8714 on val**, which
*beats* its categorical twin `t6ctl_init_ft` (0.8637). On the 157 real hand-labelled test parts it
scores **0.8699 against the incumbent's 0.8886**.

The bar and its interpretation were both written down before the run finished, which is the only
reason this reads as a result rather than as a choice of metric: *anything under 0.8886 means the
transformer body was more tuned to the broken inputs than the correct attributes are worth.*

| | random | hole | chamfer | fillet | cut | macro |
|---|---|---|---|---|---|---|
| `trial1_finetune_ss` ep32 (`categorical`) | 0.820 | 0.954 | **0.874** | **0.887** | 0.908 | **0.8886** |
| `fix4_onehot_def` ep106 (`onehot`) | 0.813 | 0.950 | 0.845 | 0.850 | 0.891 | 0.8699 |

`random`, `hole` and `cut` are within run-to-run noise. The entire deficit is `chamfer` and
`fillet` — the two classes edge **convexity** discriminates, and convexity is precisely what the
broken layout misread (it took `edge_attr[3]`, the convex flag, as edge length). Giving the model
correct convexity made it worse at the only two classes convexity exists to separate.

That is not as paradoxical as it sounds, and it is the reason the arm cannot settle the question. It
warm-started from weights trained under the broken decoding, so what it tests is *re-decode and
fine-tune*, not *train the fixed model*: the attribute encoders' weights are matched to the old
column meanings, and the classes leaning hardest on a re-interpreted column are exactly the ones
that fail to re-adapt. A clean test needs an init that never saw the broken layout — and the only
such init is from-scratch, which collapses on a corpus this size. So the defensible conclusion is
narrow and worth stating as such: **there is currently no initialization under which the correct
layout pays for itself.** `attr_layout` stays a flag, `categorical` stays the reproduction pin, the
incumbent stays in production, and `defeature_v2.py` does not change.


### The D2 channel: an empty histogram range, not missing geometry (2026-09-04)

The D2 defect was filed as the expensive one — "needs re-preprocessing, hours for 28.5k models".
It does not, and the mechanism is narrower than the diagnosis assumed.

`compute_d2_descriptor` builds its histogram from **point clouds read out of `face_grid`**, which
every pickle already stores and which were always correct. The centroids it is also handed serve
exactly one purpose: estimating `max_dist`, the top of the histogram range. Reading them from
`face_attr[:, 2:5]` — the cone/sphere/torus one-hot bits, zero on most parts — gives:

| centroid column | `max_dist` | empty face pairs |
|---|---|---|
| `2:5` (what was read) | **1e-06** | **99.2%** |
| `11:14` (the real column) | 2.17053 | 0% |

With a 1e-6 range every real distance falls outside the histogram and `np.histogram` returns
zeros. So the geometry was never missing — the range it was binned into was. Over 8 test files
the mean empty-pair fraction is **0.892 → 0.000**, and each pair goes from ~0 to ~7 populated
bins. `angle_distance` was never affected: its range is a fixed `[0, π]` and its normals come
from the grid.

Three call sites read `face_attr[:, 2:5]` and are fixed: `preprocess.py:346`,
`data/dataset.py:247`, `data/step_to_graph.py:652`. The last also read normals from
`face_attr[:, 7:10]`, which post-normalisation is `[0, 0, area]`; it now takes them from the grid
and passes `face_grids` through, as `preprocess.py` already did.

Because the inputs are all on disk, **`brepformer/data/repair_descriptors.py` rewrites
`d2_distance` in existing pickles** with no STEP and no OCC, via temp-file + atomic rename so an
interrupted run cannot leave a truncated pickle. `--check_only` reports the empty-pair fraction
without writing.

**Run on the 27-class side, 2026-09-05; the 5-class side is deliberately left broken.**
`mftrcad_processed` (28,555 files) and the 807 non-symlinked `trial6a_processed` files took ~1.5 h
of CPU alongside training, and the mean empty-pair fraction went **0.44–0.49 → 0.004**. The
27-class corpus was *less* broken than the 5-class one (0.892) for a reason worth knowing:
`max_dist` came from the cone/sphere/torus one-hot bits, so a corpus with more curved features
accidentally got a non-degenerate range more often. The residual 0.4% is genuine — coincident faces
at zero separation. `defeature_processed` stays unrepaired so that repairing it and re-running one
5-class arm is a clean attribution of the D2 fix alone, against a 0.8886 baseline measured on the
unrepaired corpus.

**One trap, and it would have reported success while doing the opposite.** `trial6_processed` and
`trial6a_processed` are **symlink farms** into `defeature_processed` and `mftrcad_processed`.
`os.replace` onto a symlink path replaces the *symlink* with a regular file — so the first version
of the script would have de-linked both corpora, left the 17 GB of real pickles still broken, grown
the disk by another 17 GB, and printed a perfect 0.44 → 0.000. It now resolves the path before
writing and dedupes by resolved target, so naming both corpora in one `--dirs` repairs each real
file exactly once and says so (`N already handled via another path`).

### Reproducing the diagnosis

```bash
# The columns read as centroids are all zero, and the pickle matches the raw graph
# exactly -- proving the one-hot layout is MFTRCAD's, not a preprocessing bug.
python - <<'EOF'
import pickle, glob, json, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d = pickle.load(open(sorted(glob.glob('brepformer/data/mftrcad_processed/train/*.pkl'))[0],'rb'))
fa = np.asarray(d['face_attr'])
print(fa[:4])
print('read as centroids (cols 2:5):', fa[:4, 2:5].ravel())
print('real centroids  (cols 11:14):', fa[:4, 11:14].ravel())
_, g = json.load(open('brepformer/data/mftrcad/graphs/20240116_231044_0_result.json'))
print('identical to raw MFTRCAD graph:', np.allclose(np.asarray(g['graph_face_attr']), fa))
EOF

# Column-wise occupancy over 200 models: cols 0-5 one-hot, 6-8 and 10 dead, 9 area,
# 11-13 centroid.
python - <<'EOF'
import pickle, glob, numpy as np
cols = np.zeros(14); n = 0
for f in sorted(glob.glob('brepformer/data/mftrcad_processed/train/*.pkl'))[:200]:
    d = pickle.load(open(f,'rb')); fa = np.asarray(d['face_attr'])[:int(d['num_faces'])]
    cols += (fa != 0).sum(0); n += fa.shape[0]
print('nonzero fraction per column:', (cols/n).round(3))
EOF
```

### Two smaller model-side issues

- `GraphAttnBias.edge_conv_layer` (`embedding.py:369`) is constructed and never called, so
  the multi-hop edge aggregation its docstring advertises does not happen and its
  parameters sit dead in every checkpoint.
- `MultiheadAttention` is constructed with `max_seq_len=512` while training runs
  `--max_faces 1000`. Latent only because `use_rope` defaults to `False`; enabling RoPE
  without raising `max_seq_len` would crash in `reshape_for_broadcast`.

Checked and *not* bugs: the fully-masked-row guard in `attention.py:217` does fire, because
`GraphAttnBias` writes `-inf` into both the rows and the columns of padded pairs even though
`attn_mask` alone masks only columns. And `edge_path` is `None` throughout — no pickle
carries it — so the hop-decay loop in `GraphAttnBias.forward` is dead code rather than a
wrong computation.

## Coarse Label Spaces: Marginalize, Don't Argmax-Then-Map

The 27-class model is routinely evaluated and shipped in a coarser space (`--5_classes`,
`--8_classes`). Collapsing it correctly is not obvious, and `infer.py` got it wrong until
2026-09-01.

The collapse is many-to-one, so mapping the 27-class argmax down is **not** the same as
summing each coarse group's probability mass and taking the largest. Only the latter
minimises 0-1 loss in the coarse space: five sibling fine classes holding 0.15 each should
beat a single rival holding 0.25, but argmax-then-map picks the rival. The two rules
disagree on 35,626 of 334,175 faces (10.7%). Separately, `face_probs` used to stay 27-wide
while `face_preds` became 5-class, so any downstream confidence gate was thresholding a
different label space from the predictions it gated.

`_remap_result_to_defeature` now marginalizes first, rewrites `face_probs` into the coarse
space, and preserves the fine distribution as `face_probs_27` for auditing.

### The group-size trap

Fixing this exposed something more important. The 27→5 groups are extremely uneven:

| coarse class | fine classes absorbed |
|---|---|
| `random` | 1 |
| `hole` | 2 |
| `chamfer` | 2 |
| `fillet` | 1 |
| `cut` | **21** |

A completely **uniform** — that is, entirely uninformative — 27-class posterior therefore
marginalizes to `[0.037, 0.074, 0.074, 0.037, 0.778]`, and the plain argmax confidently
returns `cut`. Diffuse probability mass lands on `cut` by construction, and the rare
classes this project exists to recover are crushed by the shape of the mapping rather than
by the evidence.

That is correct behaviour if the metric is accuracy, and wrong if it is macro-F1. Both
rules are therefore available:

```bash
# Bayes-optimal for 0-1 loss / accuracy. Default.
python -m brepformer.infer --step part.step --checkpoint CKPT --5_classes \
    --coarse_decision argmax

# Divides each coarse marginal by its group size before the argmax, cancelling the
# structural advantage a large group gets purely by absorbing diffuse mass. Use this
# whenever the reported number is macro-F1.
python -m brepformer.infer --step part.step --checkpoint CKPT --5_classes \
    --coarse_decision balanced
```

Verify the asymmetry directly:

```bash
python - <<'EOF'
from brepformer.infer import _marginalize_probs, _coarse_argmax
from brepformer.data.classes import CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES
c = _marginalize_probs([[1/27]*27], CLASS_TO_DEFEATURE, DEFEATURE_NUM_CLASSES)
print('uniform posterior ->', [round(p, 3) for p in c[0]])
print('argmax   ->', _coarse_argmax(c, CLASS_TO_DEFEATURE, 5, 'argmax'))    # [4] = cut
print('balanced ->', _coarse_argmax(c, CLASS_TO_DEFEATURE, 5, 'balanced'))  # [0]
EOF
```

## Geometric Rules: Selector, Never Corrector

`geometric_rules.py` checks each predicted face class against what the underlying surface
geometry can actually support, returning `CONFIRM`, `DENY` (with a suggested correction) or
`UNCERTAIN`. It must run as `--geo_mode advise`, where denials only *select* faces for
review. `rewrite` mode exists for comparison and corrupts labels.

### How it is validated

Feed mftrcad's ground-truth labels in as if they were predictions. Every face is then
correct by construction, so **any** `DENY` is provably false, and its rate is the rate at
which the rules destroy correct labels:

```bash
python -m brepformer.data.validate_geometric_rules --limit 60 --label_space 27 \
    --out /tmp/geo.json
```

Every verdict now carries a `rule` field naming the rule that produced it, so a corruption
rate can be attributed to a specific rule rather than matched out of a prose string. The
validator prints that breakdown.

### Two fixed defects

**An axis is a line, not a direction.** Co-axiality was tested as
`abs(dot(axis_i, axis_j)) > 0.95` — satisfied by every pair of parallel cylinders on a
part, wherever they sit. The axis *location* was never even extracted. `_axes_coaxial` now
also requires the offset between a point on each axis to be parallel to the axis, with the
perpendicular component compared against a radius-scaled tolerance so the test is
size-invariant. On a 300×400 mm fixture plate with M8 holes on a 25 mm pattern, 740 axial
faces formed **one** axis cluster before and **189** after — against roughly 12×16 = 192
actual holes.

**A chamfer is not always planar.** A chamfer is planar only when it breaks a *straight*
edge; chamfer a hole, boss or shaft — a circular edge — and the face is a cone.
`must_be_planar` denied all of those unconditionally. Classes 0/26 and 5-class 2 now carry
`may_be_conical`, exempting cones from the planarity gate.

Measured effect, 60 models / 1412 labelled faces:

| metric | before | after |
|---|---|---|
| overall false-denial rate | 15.0% | **13.8%** |
| `chamfer_2` falsely denied | 100% (53/53) | **71.7% (38/53)** |

### Two more fixed defects: every threshold was guessed

Per-rule attribution of the residual false denials at that point:

```
must_be_planar        82  (rectangular_passage_2 70, 2sides_blind_step 11, ...)
must_be_curved        69  (circular_end_pocket 51, blind_hole 13, ...)
area_fraction_max     38  (chamfer_2 38)
large_planar_stock     6  (rectangular_through_step 2, ...)
```

All 38 residual `chamfer_2` denials came from `area_fraction_max`, which **refutes** the
previously recorded belief that the area caps are negligible: they bind on 0.34% of faces
corpus-wide and on 72% of this one class. A corpus-wide rate can hide a per-class
catastrophe completely. Always break these results out per class.

That attribution pointed at two thresholds nobody had ever measured, so both were measured
over 300 mftrcad models (`/tmp/measure_area.py`, `/tmp/measure_stype.py`, ~2 min each).

**Area caps.** `chamfer_2` has a *median* area fraction of **0.158** and a maximum of
0.861, against a 0.10 cap — no cap small enough to mean anything survives contact with the
real distribution. Removed from `chamfer_2` and from 5-class `chamfer`, which absorbs it.
The caps on `chamfer` (max 0.038) and `round` (max 0.012) are measurably inert and kept as
cheap guards.

**Type constraints.** These were assigned from each class's *name*, and that is wrong in
principle: a 27-class label names a **feature**, and a feature owns all of its faces.
Measured mix:

| class | planar | conical | curved | old constraint denied |
|---|---|---|---|---|
| `rectangular_passage_2` | 0% | 0% | 100% | `must_be_planar` — **100%**, exactly inverted |
| `circular_end_pocket` | 58% | 1% | 41% | `must_be_curved` — 58% (floor and walls) |
| `blind_hole` | 52% | 0% | 48% | `must_be_curved` — 52% (the hole bottom) |
| `2sides_blind_step` | 48% | 0% | 52% | `must_be_planar` — 52% |
| `circular_blind_slot` | 46% | 0% | 54% | `must_be_curved` — 46% |
| `through_hole` | 3% | 0% | 97% | `must_be_curved` — 3%, kept |
| `chamfer` | 85% | 13% | 2% | `must_be_planar` — 2%, kept |

"Circular" describes the feature's silhouette and says nothing about any one of its faces.
A constraint is now present only where it was measured to deny <5%. In the 5-class space
`hole` and `cut` lost theirs entirely — `cut` unions 21 fine classes and has no
characteristic type at all.

### Where it ended up

| state | false-denial rate |
|---|---|
| as shipped | 15.0% |
| + axis-line co-axiality, cone exemption | 13.8% |
| + area caps measured | 11.3% |
| + type constraints measured | **0.6%** (16/2544 over 100 models) |
| same, `--label_space 5` | **0.4%** |

13 of the residual 16 are `large_planar_stock`: a real `rectangular_through_step` face can
legitimately exceed 15% of total area. That is a genuine heuristic tension rather than a
misencoded assumption, so it stays.

**None of this changes the verdict on rewrite mode.** `benefit` mode still reports 0 of
374 corrupted faces recovered — see "The Label Harness" below. Cheap and accurate denial
makes the rules a usable *detector*; it does not make them a corrector.

---

## The Label Harness: Three Questions, Not One

`brepformer/data/label_harness.py`

Every labelling idea in this project has looked good in isolation and several have been
actively destructive. The geometric rules rewrote a third of all faces and corrupted 15%
of the correct ones. The vision pass performed at or below chance on feature faces. In
both cases the failure was invisible for weeks, and invisible for the *same* reason: they
were evaluated on pseudo-labels, where there is no correct answer for a bad heuristic to
disagree with. Confident-looking output is the default state of an unmeasured labeller.

mftrcad's ~28.5k exactly-labelled models remove that excuse, and the harness makes using
them the cheap path — one command, one report format, so a new heuristic is scored the
same way as the last one instead of growing its own bespoke validator.

### Why three modes

Asking "is this labeller good?" hides three independent questions, and a component can
pass any one of them while failing badly on another.

**`--mode harm` — does it destroy correct labels?** Feed the ground truth in as if it
were the component's input. Every face is then correct by construction, so *any* change
the component proposes is provably a false one, and the change rate is exactly the rate at
which it corrupts correct labels. There is no modelling and no ambiguity, which is why
this is the first test to run and the one that catches disasters.

**`--mode benefit` — does it fix wrong ones?** Harm alone cannot separate a harmful
component from a useless one: a component that changes nothing scores a perfect 0% harm.
So corrupt the truth first and measure recovery. The corruption is deliberately *not*
uniform-random. A real teacher confuses `chamfer` with `fillet` and `through_hole` with
`Oring`; it does not confuse `chamfer` with `6sides_passage`. Uniform noise produces
geometrically absurd labels that any surface-type check catches, which would flatter every
component. `--noise confusable` draws only from pairs listed in `CONFUSABLE_27` /
`CONFUSABLE_5`. Outcomes are split three ways — **fixed**, **churned** (changed to a
third wrong answer) and **missed** — because "changed a lot of labels" is not evidence of
anything, and clean-face breakage is tracked alongside so a component that fixes ten
labels while breaking twenty is visibly a net loss.

**`--mode select` — can predictions be auto-accepted?** This is the mode that serves the
actual goal. Auto-labelling does not need a model that is right everywhere; it needs one
that **knows when it is right**, so high-confidence faces can be accepted untouched and
the rest routed to review. That is a *ranking* property, not an accuracy property, and
accuracy cannot see it. The harness sorts faces by confidence, and for each coverage level
k reports the error rate among the accepted top-k — a risk–coverage curve. The scalar is
**AUGRC** (area under the generalised risk–coverage curve): the mean error count over *all*
faces, averaged over every coverage level. It charges heavily for an error that slips
through at high confidence and lightly for a low-confidence mistake review would have caught
anyway. 0 means every error ranks below every correct prediction.

The harness reports **both AUGRC and AURC**, and the distinction is not cosmetic. AURC divides
by the *accepted* count, AUGRC by the total. AURC is the incumbent (Jaeger et al., ICLR 2023,
arXiv:2211.15259), but its `1/k` denominator gives high-confidence failures a harmonic weight,
which breaks monotonicity — it can prefer a scorer with both lower accuracy *and* worse error
ranking — and it **amplifies the influence of label noise**, which this corpus has. AUGRC
(Traub et al., NeurIPS 2024, arXiv:2407.01032) fixes that, but is a 2024 proposal rather than
the accepted standard, so both are printed.

**This function used to compute AURC and call it AUGRC**, and every AUGRC figure published in
this file before 2026-09-02 was really AURC. Each metric also has its *own* uninformative
baseline and they are different numbers: a random ranking has `AUROC_f = 0.5`, so
`augrc_base = (1 - acc) / 2` — *half* the error rate — while `aurc_base` is the error rate
itself. The corrected implementation is verified against the closed form
`(1 - AUROC_f)·acc·(1 - acc) + ½(1 - acc)²` to within 0.004.

Alongside each empirical auto-accept cut the harness now prints a **certified** one: a
Clopper–Pearson lower bound on precision at `DELTA = 0.05`, with a `delta/log2(m)` multiplicity
correction (Geifman & El-Yaniv, arXiv:1705.08500) because scanning every threshold and keeping
the best is selecting on the very data the bound is computed from. Exact binomial rather than
normal or Hoeffding, since for a 0/1 loss the exact inversion is both valid and strictly
tighter (RCPS, JACM 2021). This matters more than it sounds: **at ≥95% precision no labeller
certifies any coverage at all** — "3.8% of faces at 95%" was a precision measured on ~53 faces.
Only the fused labeller certifies anything, 11.7% at ≥90%.

### Everything is reported per class, always

The single most repeated lesson in this project. The `area_fraction_max` rule binds on
0.34% of faces corpus-wide — negligible — and on 72% of `chamfer_2` — total. A component
that is fine on `stock` and catastrophic on `chamfer` looks acceptable in aggregate while
destroying exactly the rare classes this corpus exists to rescue. Aggregate numbers on a
long-tailed label distribution are dominated by `stock` and `cut` and say almost nothing.
`harm` mode additionally attributes each corruption to the specific rule that caused it,
so the output names the thing to change rather than reporting that something is wrong.

### What it found immediately

**Every geometric threshold in the rules was wrong, and measuring them took four
minutes.** `harm` mode localised the damage to a specific rule and class, which turned a
vague "the rules are harmful" into two concrete measurements — the per-class area
distribution and the per-class surface-type mix. Both contradicted the shipped
assumptions outright: `chamfer_2` has a *median* area fraction of 0.158 against a 0.10
cap, and `rectangular_passage_2` is 0% planar under a `must_be_planar` constraint. The
false-denial rate went **15.0% → 13.8% → 11.3% → 0.6%** as each guessed constant was
replaced by a measured one.

The general lesson is worth more than the fix. The constraints were assigned from each
class's *name*, and a 27-class label names a **feature**, not a face. A feature owns all
of its faces, so `circular_end_pocket` is 58% planar (floor and walls) and `blind_hole` is
52% planar (its bottom). "Circular" describes the feature's silhouette and says nothing
about any individual face. Per-face type constraints are therefore only valid for classes
that happen to be single-typed — which the measurement identifies and the class name
cannot. In the 5-class space they are invalid almost everywhere: `cut` unions 21 fine
classes and has no characteristic surface type at all.

**But the rules still cannot fix anything.** After all of that, `benefit` mode reports
**0 of 374** corrupted faces recovered, with 31% churned into a third wrong answer. This
is structural, not a threshold: every DENY suggests `stock` or `through_hole` and nothing
else, so there is no vocabulary in which to restore a `chamfer`↔`chamfer_2` confusion.
This settles the `--geo_mode` question that was previously a judgement call. The rules are
a **detector with a 0.6% false-alarm rate, not a corrector** — worth using to flag a face
for review, never to rewrite one.

**Confidence is anti-correlated with correctness on the two classes that matter most.**
On 59 mftrcad models, mean confidence on `chamfer` errors (0.898) is *higher* than on
`chamfer` hits (0.892); `hole` behaves the same way (0.811 vs 0.785). Raising a global
threshold therefore admits those classes' errors preferentially — no single confidence cut
can auto-accept them. The auto-accept budget at ≥95% precision is 11.5% of faces overall.
The conclusion is architectural, not a tuning note: accept thresholds must be **per
class**, with `chamfer` routed to mandatory review regardless of confidence, and this
table re-measured for every new checkpoint rather than assumed to carry over.

### Re-measured on real parts, and the budget is 7× larger (2026-09-04)

The 11.5% above is on **remapped mftrcad**, while every checkpoint macro F1 in this README is on
`defeature_processed/test`. Those are different parts, so the two numbers could not be quoted
together. Two additions put them on one footing — `scan_checkpoints.py --dump_probs`, which
writes `face_preds`/`face_probs` for the pickles it is already scoring, and
`label_harness.py --gt_pkl_dir`, which reads select-mode truth from a preprocessed split's
`face_labels`. The dump reproduces macro F1 0.8886 exactly, so nothing else moved.

On the 157 real hand-labelled parts / 13,445 supervised faces, the incumbent's confidence is
**informative, not anti-correlated**: AUGRC **0.0208** against 0.0521 for uninformative ranking
(AURC 0.0285 vs 0.1043).

| target precision | accept (observed) | accept (95%-certified) | threshold |
|---|---|---|---|
| ≥99% | 22.3% | 16.6% | 0.998 |
| ≥98% | 39.1% | 30.4% | 0.994 |
| **≥95%** | **82.6%** | **80.0%** | 0.891 |
| ≥90% | 98.9% | 97.1% | 0.504 |

**80% of faces auto-accept at a certified 95% precision**, against 11.5% on mftrcad. This is the
central number for the review-and-override workflow: it means a human sees 20% of faces, not all
of them, and it is a far better operating point than a flat 0.7 mask applied to every class.

| true class | faces | acc | mean conf | conf when wrong | margin |
|---|---|---|---|---|---|
| `cut` | 5425 | 89% | 0.936 | 0.785 | 0.151 |
| `hole` | 3032 | 98% | 0.977 | 0.737 | 0.240 |
| `random` | 3011 | **80%** | 0.899 | **0.816** | **0.083** |
| `fillet` | 1568 | 92% | 0.928 | 0.749 | 0.179 |
| `chamfer` | 409 | 94% | 0.958 | 0.801 | 0.157 |

Two conclusions differ from the mftrcad reading and supersede it for real parts:

- **`chamfer` does not need mandatory review.** 94% accuracy with a 0.157 confidence margin. The
  anti-correlation was a remapped-mftrcad artifact.
- **`random` is the bottleneck**, and it is the worst-calibrated class on both counts — lowest
  accuracy and the smallest margin between mean confidence and confidence-when-wrong. Confident
  `random` errors are what caps the ≥99% band at 22%. Seed the review queue with faces
  *predicted* `random` rather than with the globally least-confident faces, and set the accept
  threshold per class: `hole` and `fillet` errors are cheaply separable, `random` errors are not.

The architectural conclusion — **per-class thresholds, re-measured per checkpoint** — is
unchanged. Only which class needs the strict threshold changed.

### Per-class thresholds, built and measured — and they accept *fewer* faces on purpose (2026-09-05)

`brepformer/data/accept_queue.py` turns the analysis above into the decision the workflow runs on:
`calibrate` fits one certified threshold per **predicted** class against ground truth, `apply` splits
an unlabelled corpus into accept/review and writes a per-model queue. Nothing is masked — an
unreviewed face still gets a label, with a certified precision attached.

The first thing it showed contradicts the plan it was built from. Per-class thresholds certify
**72.9%** of the 13,445 faces at 95% precision; a single global threshold certifies **80.0%**. The
7.1-point gap is what honest multiplicity costs: delta 0.05 split five ways by Bonferroni, plus the
weaker bound each class gets from its own smaller sample. A per-class table that *beats* global is
usually a table that searched more thresholds without paying for the search.

Raw coverage is the wrong comparison, and the reason is what the global number is a guarantee
*about*. Inside the globally-accepted set at threshold 0.917:

| predicted class | accepted | observed | certified | vs 95% target |
|---|---|---|---|---|
| `cut` | 4316 | 96.3% | 95.5% | OK |
| `hole` | 2898 | 96.2% | 95.2% | OK |
| `random` | 1853 | 95.6% | 94.2% | **under** |
| `fillet` | 1301 | 93.6% | 91.7% | **under** |
| `chamfer` | 393 | 88.5% | 83.9% | **under** |

The global 95% is **pooled**, carried by `cut` and `hole` (55% of faces). Three of five classes sit
under target inside the set it accepts, and the worst is `chamfer`: one wrong auto-accepted chamfer
in roughly nine, in exactly the class where that is most damaging, since chamfer/fillet confusion is
this model's characteristic failure. The extra coverage is coverage the global cut was not entitled
to. Per-class spends 7.1 points to make the guarantee hold *per class*, which is the only form a
reviewer can act on.

| predicted class | faces | acc | threshold | accepted | certified |
|---|---|---|---|---|---|
| `cut` | 5252 | 92% | 0.904 | 83.5% | 95.0% |
| `hole` | 3182 | 93% | 0.917 | 91.1% | 95.0% |
| `random` | 2845 | 84% | 0.944 | 58.2% | 95.0% |
| `fillet` | 1692 | 85% | 0.976 | 51.1% | 95.0% |
| `chamfer` | 474 | 81% | **none** | 0.0% | — |

**So the operating rule is: every face predicted `chamfer` is reviewed.** No confidence certifies
95% for it — 474 faces, 3.5% of the corpus, the cheapest review anyone will buy. This also corrects
the section above: those tables are keyed on the **true** class, where `chamfer` looks well
calibrated at 94%. A threshold can only key on the **predicted** class, and there `chamfer` is the
worst of the five at 81%. Both numbers are right; only one of them is actionable.

Two implementation points worth knowing. Thresholds are only valid at a confidence boundary where
the next face is *strictly* less confident — accepting at `>= c` admits every tie, so evaluating
mid-tie certifies a set larger than the one that gets accepted. And a class with no certified
threshold accepts nothing at any confidence; falling back to the global cut for it, the obvious
convenience, would re-admit precisely the class calibration refused.

This also surfaced a latent bug: `binom_precision_lower_bound(n_ok, n, delta)` accepted a `delta`
and ignored it, because `_delta_eff` read the module-level 0.05 unconditionally. Only `mode_select`
called it and only at the default, so no published number moves — but a per-class search has to
spend a smaller delta per class, and cannot if the argument is inert.

### Two silent traps, both now guarded

A checkpoint's label space is not recoverable from its filename, and `infer.py` will
print 27-class *names* over a 5-class posterior if `--5_classes` is omitted. Scoring those
predictions against 27-class ground truth produced **7.6% accuracy with `stock` at 0%**
for a checkpoint that is genuinely fine — a units error that is indistinguishable from a
broken model. The harness now aborts when the posterior width disagrees with
`--label_space`, and collapses mftrcad's 27-class truth through `CLASS_TO_DEFEATURE`
whenever `--label_space 5`. Separately, `infer.py` aborts with a double-free in the
`brep_mfr` env; run inference under `new_brepmfr`. Exact commands and full output tables
are in `SCRIPTS.md` under "Label harness".

## Synthetic Labels: Stop Masking, Get a Second Opinion

The pseudo-label pipeline that built the expanded corpus was one chain with one voice:
`infer.py` teacher → per-class-quantile confidence mask → train. It has two faults that
are structural, not tunable.

**No second opinion.** Every pseudo-label is the teacher's, so the student can only inherit
the teacher's errors, and self-distillation ratchets them in — the measured mechanism behind
`chamfer` degrading across generations.

**Masking deletes the hard faces instead of resolving them.** Setting an uncertain face to
`-1` looks conservative, but uncertainty is not spread evenly. It concentrates on exactly
the rare classes the corpus needs: a 0.7 global confidence cut left `chamfer` with a 0.4%
keep rate. The mask preferentially throws away the labels worth having, which is why the
requirement was to *look at a face and override it with confidence* rather than mask it.

### The measured surprise: exact geometry beats the trained network

`geo_label.py` labels a face from the B-Rep alone, using the `P(class | geometric fact)`
table that `measure_class_geometry.py` had already measured over 12,000 models and 305,785
labelled faces — 24 facts covering surface type, accessibility, edge convexity, tangential
edges and AAG component composition. That table existed for months and **nothing consumed
it as a predictor**; `vision_label.py` only pasted a few rows into a prompt.

Scored against exact ground truth on 59 mftrcad models / 1388 faces, 5-class:

| labeller | accuracy | `chamfer` | AUGRC (base) | AURC (base) | @≥95% observed | @≥95% **certified** |
| --- | --- | --- | --- | --- | --- | --- |
| network (`f1=0.9415.ckpt`) | 63.9% | 12% | 0.1451 (0.1805) | 0.2310 (0.3610) | 11.5% | none |
| geometry | **72.6%** | **86%** | **0.1069** (0.1369) | 0.1804 (0.2738) | 3.8% | none |
| fused | **72.6%** | **86%** | 0.1070 (0.1369) | **0.1697** (0.2738) | **12.0%** | none |

A hundred lines of arithmetic over a lookup table beats the graph transformer by 8.7 points,
and by **7×** on `chamfer` — the class the defeaturing task exists to find. Compare each
figure only against its own base; across label spaces the number is meaningless, which is the
same units trap the harness already guards.

**Fuse for the operating point, not for the ranking.** An earlier version of this table claimed
fusion improved the ranking, 0.1804 → 0.1697. That was an AURC artifact: on AUGRC, fusion and
geometry are a dead tie (0.1070 vs 0.1069, geometry marginally ahead). What fusion actually buys
is the auto-accept budget — 12.0% at ≥95% against geometry's 3.8% — and it is the only labeller
that certifies any coverage at all (11.7% at ≥90%).

This is not really a claim about geometry. It is a measurement of what the model was denied:
`face_attr` reads an index layout the data never had, so **edge convexity is identically
zero in the network's input** — and convexity is the single best fillet-vs-chamfer
discriminator. The network is being asked to separate chamfers from fillets with the
decisive fact deleted. Fixing that layout bug is now the highest-value model change
available, and this result is the estimate of what it is worth.

### Fusion: agreement is the accept signal, and the queue is binary

```
agree     65.2% of faces   80.6% precision
disagree  34.8% of faces   geo right 57.8%   net right 32.7%   BOTH WRONG only 9.5%
```

Everything `fuse_labels.py` does follows from those three numbers:

1. **Accept on agreement, not on confidence.** Within the agreeing set, tightening network
   confidence from 0.0 to 0.95 moves precision 80.6% → 81.5% while coverage collapses 65% →
   41%. The network's confidence is nearly worthless as a discriminator here — it is
   *anti-correlated* with correctness on `chamfer`. Two independent views agreeing is what
   carries the information.
2. **Geometry is the default on disagreement**, right nearly 2:1 over the network. The
   network's answer is still recorded, because it is right on a third of what geometry misses.
3. **Both are wrong on only 9.5% of disagreements.** So for 90.5% of the queue, the correct
   label is one of exactly two named candidates, and an arbiter that merely picks A-or-B
   would reach ~84% overall on this sample.

That last point is what rescues the Claude imaging pass. Decomposing the 27-class label into
(family, kind, depth) showed dense-view vision scoring `depth` (through vs blind) at
**45.8% against a 38.1% chance baseline — pure chance**, because through-vs-blind is not a
property of any exterior view and no amount of zooming or hole-navigation can recover it. It
must be computed from topology. But "is this face a chamfer or a fillet" *is* answerable from
a picture. The queue changes the question from an open 27-way guess to a binary choice
between two specific candidates, which is the difference between a pass that measured at
chance and one worth rendering images for.

`ALWAYS_REVIEW` overrides agreement for `chamfer` on measurement, not caution: the two
labellers agree on only 16% of chamfer faces and are still just 60% right when they do — the
worst agreed-precision of any class, against 86% for `random` and 100% for `fillet`.
Agreement on `chamfer` more often means two labellers sharing a blind spot.

### Three design choices measurement reversed

Each was implemented the obviously-correct way and was wrong; the diagnostic that overturned
it is kept in the script so no one has to trust the conclusion.

- **Sum the evidence, don't average it.** Averaging log-likelihood ratios stops `k`
  correlated facts voting `k` times — but it also rescales every informative fact by
  `k/(k+1)` when an uninformative one joins, so it can only dilute. Macro-recall over all 24
  facts: **0.307 summed, 0.242 half, 0.128 averaged**. Correlation is a calibration problem,
  not a ranking one.
- **Use every fact.** Greedy forward selection beat all-24 on the half it was selected on
  (0.3092 vs 0.2998) and **lost** on held-out models (0.2877 vs 0.3121). Excluding a fact is
  a stronger claim than including one: a weak fact costs almost nothing, a missing fact costs
  everything it knew. The split must be *by model* — faces of one part share a component and
  a bounding box.
- **Rate the decision on a different scale than you make it on.** Summed evidence saturates
  confidence, so at `alpha=0` the top-decile precision (69.6%) came out *below* overall
  accuracy (72.6%) — the most confident faces were the worst ones. Because the gate can never
  change a label, decoupling costs nothing and cut 5-class **AURC** from 0.2345 to 0.1851.
  (Those are AURC, not AUGRC — see the metric note above. Re-checked against the corrected
  AUGRC and the choice survives: the production decoupled setting scores 0.1069, better than
  every row of the joint calibration sweep, whose own best is 0.1112 and which gives up macro
  recall to get there.)

### Where geometry's ceiling is, and it is the same category error as vision

The 27-class table (12,000 models) reaches 49.1% accuracy, **AURC** 0.2182 against a 0.5086
base, and a **29.6%** auto-accept budget at ≥95% precision (empirical, not certified) — it calls `chamfer_2` correctly
on all 53 faces at confidence 1.000. But it scores **0%** on every polygonal class
(`triangular_passage`, `6sides_pocket`, `rectangular_passage`). No fact in the table counts
the sides of a feature, so one wall of a hex pocket is identical to one wall of a rectangular
pocket.

That is the same mistake as the face-centred vision crops: a 27-class label names a
**feature**, and a feature owns all of its faces, so a per-face question cannot recover a
feature-level property.

### Correction: clustering into AAG components does *not* fix it

The obvious fix — `face_geometry.part_adjacency` already computes AAG concave components, so
cluster faces into features, ask the question of the feature, propagate to its faces — was
recommended here and then measured. **It loses badly.** Majority-vote a component's
`geo_label` predictions onto its faces and 27-class accuracy falls from 49.1% to **31.8%**.
The reason is not the voting rule, it is that the components are not features:

| | with `stock` | oracle `stock` removed |
|---|---|---|
| components that are label-pure | 25.5% | 51.1% |
| faces in their component's majority label (**voting ceiling**) | 57.3% | 63.7% |
| component majority vote | 31.8% | 22.5% |
| `geo_label` per-face on the same faces | 49.1% | 35.4% |

Purity by component size is 100% at size 1, 19% at 2, 44% at 3, 27% at 4-6 and **0% at 7+**.
The dominant impurity is `<feature> + stock` (`chamfer_2 + stock` 19×, `rectangular_passage_2
+ stock` 15×, `Oring + stock` 12×): a concave component is a feature *plus the host material
it is cut into*. Removing `stock` with an oracle doubles purity — and voting still loses,
because 30 non-stock components still have 8+ faces and are genuine interacting-feature blobs
(`6sides_pocket + rectangular_passage_2`, `chamfer_2 + triangular_passage`). A chamfer on the
rim of a passage is concave-connected to the passage; that is the textbook AAG limitation, not
a bug in the implementation. **Do not spend effort on hard label propagation within concave
components.** Component identity is worth having as an *input* — a fact the classifier may
weigh — never as a constraint on the output.

### What the polygonal classes are actually missing, measured

The side count is not absent from the geometry. It is present and destroyed by the binning.
Two existing facts each come within one step of it:

- `strict_component_surfaces` collapses surface types to a **set**, so a triangular pocket and
  a hex pocket are both just `plane`.
- `strict_component_size` bins as `0-2 / 2 / 3-5 / 5-9 / >=9`, so a rectangular pocket (4
  walls + floor = 5) and a hex pocket (6 + 1 = 7) **share the `5-9` bin**.

Counting the component's planar and cylindrical faces at integer resolution instead
(`/tmp/test_sidecount.py`, 1412 mftrcad faces with exact GT):

```
I(class ; n_planes, n_cyls) = 0.840 bits of H(class) = 3.605   -> 23.3% of the label entropy
same signal through the EXISTING coarse size bin = 0.217 bits  ->  6.0%
integer resolution recovers +0.622 bits that the current table throws away
```

And where a feature does sit in its own component the fact points straight at the answer:
`6sides_pocket` → `(6 planes, 0 cyls)`, `triangular_pocket` → `(3, 0)`, `chamfer_2` → `(1, 0)`.

But the same table shows the real blocker. The modal bin for `6sides_passage` is the saturated
`(10+, 3+)` at **80%** of its faces — the component has merged into most of the part. So the
two findings agree: re-binning the count is a free +0.62 bits, and beyond that the limit is
**component construction**, not the classifier. Through-vs-blind must likewise come from
topology, not from a picture — vision scored `depth` at 45.8% against a 38.1% chance baseline.

### The fix: don't hand-build the partition, learn the prune

`brepformer/data/edge_prune_probe.py` tests the route FeatureFox takes (arXiv:2604.26770,
verified — Fuchs, Kacan, Haag, Lohse, Apr 2026): rather than taking connected components of the
raw face-adjacency graph, train a binary **"do these two faces belong to the same feature?"**
classifier on edge attributes, prune the edges it rejects, and take components of what remains.
On 400 mftrcad models with exact GT, split by model:

| partition | pure % | **ceiling %** | components |
|---|---|---|---|
| existing strict concave (baseline) | 24.8 | 56.3 | 659 |
| **learned edge prune** | **83.5** | **88.0** | 1786 |
| oracle same-label prune | 100.0 | 100.0 | 1977 |

Held-out edge AUC **0.945**, from 11 attributes `face_geometry` already emits and seconds of
CPU. It improves with scale (at 60 models: AUC 0.895, ceiling 83.3%) while the baseline stays
flat, so it is not a sample-size artifact.

**And it explains the negative result above.** Dihedral angle alone scores AUC **0.471** and
convexity alone **0.531** — both *at chance*. Those are precisely the two quantities
`part_adjacency` thresholds to build concave components. The signal that separates features is
real but exists only in the combination of attributes, which a learned prune recovers and a
hand-tuned angle threshold cannot. The loose/strict threshold hierarchy was never a weak
version of the right idea; it was the wrong feature space.

A ceiling is not an accuracy, so `--classify` closes the loop: a component-level classifier over
the pruned partition, using descriptors no per-face fact can express — integer plane/cylinder/cone
counts (the side count the `5-9` bin destroys) and a count of fully occluded planar faces as a
bottom-face, hence through-vs-blind, proxy — propagating one label to every face of a component:

| partition | per-face accuracy (same held-out models) |
|---|---|
| existing strict concave | 34.9% |
| **learned edge prune** | **62.8%** |

The partition is the whole difference: the same classifier on the old partition gains only ~3
points over hard voting's 31.8%, and ~28 on the new one. `geo_label`'s per-face 27-class accuracy
is 49.1% for scale, but that came from a different 60-model sample with a table built on 12,000
models, so 62.8 vs 34.9 is the controlled comparison and +13.7 is indicative only.

This reframes the whole feature-level plan. Clustering was not the mistake — *hand-building the
clusters* was. With a ceiling of 88.0% instead of 56.3%, asking the question of the feature
becomes worth doing, and the side-count and bottom-face descriptors above become computable on
components that are actually features. Two method details are easy to get wrong and both matter:
split **by model, never by edge** (edges of one part share faces and a bounding box), and encode
edge features **symmetrically** (`min`/`max`, not iteration order) since the pair is undirected.
Ground truth here is same-*semantic-label*, not same-instance, because mftrcad has no instance
ids — so the ceiling is optimistic for instance segmentation but exact for the per-face
labelling task we actually run.

## Training: Initialization Dwarfs Every Other Knob

Measured 2026-09-02 across the nine-arm family-5 sweep. **Read the `bestface-*` checkpoint
filename, not `best-epoch-*`** — the latter ranks `val/f1`, the *model-level* multi-label
metric, which is high for models whose face segmentation is dead.

| arm | data | init | best **face_f1** | best val/f1 (model-level) |
|---|---|---|---|---|
| `t6ctl_base` / `t6ctl_wf` | defeature GT | scratch | 0.1001 / 0.0887 | 0.706 |
| **`t6ctl_init_ft`** | defeature GT | trial1_finetune_ss | **0.8637** | 0.930 |
| `t6_base` / `t6_f` | trial6 masked | scratch | 0.1255 | 0.795 |
| `t6_w` / `t6_wf` | trial6 masked | scratch | 0.1982 / 0.1942 | 0.690 / 0.795 |
| `t6_init_def` | trial6 masked | trial1_ss1500 | 0.6958 | 0.853 |
| `t6_init_def_wf` | trial6 masked | trial1_ss1500 | **0.8177** (still training) | 0.851 |

**Every from-scratch arm collapsed to a single class.** `t6_base` held `val/face_acc` at
*exactly* 0.457 for all 65 epochs and 286 minutes while `train/face_acc` climbed 0.366 → 0.510.
0.457 is the validation majority-class share (`cut` = 46.4% of 22,110 supervised faces), and a
face macro F1 of 0.1255 ≈ 1/5 is the arithmetic signature of a constant prediction; the
`bestface-*` checkpoints stop being written at epoch 02.

This is not a pseudo-label problem — **the controls, training on exact defeature ground truth,
are the worst arms of all** (0.089–0.100). Roughly one to five hours from scratch is simply not
enough for this model to learn face segmentation. So the sweep's designed variables — masking
strategy, `--weighted_crossentropy`, `--face_seg_weight` — were being compared *between two
collapsed models* and mean nothing there. **Always pass `--init_weights`; treat any from-scratch
number as void.**

Among the arms that did learn, loss weighting is worth about **+12 points** (0.6958 → 0.8177),
which nearly closes the gap to the defeature-GT arm despite training on masked pseudo-labels.
A caution that cost us a wrong number once already: that arm read 0.7065 at epoch 6, so never
quote a metric off a live run without checking the epoch.

**The comparison everyone will want to make is invalid as run.** `t6ctl_init_ft` 0.8637 versus
`t6_init_def` 0.6958 crosses two different validation sets: `defeature_processed` val is
**99.6%** supervised (12,521 of 12,570 faces), `trial6_processed` val is **63.0%** (22,110 of
35,070). Run `brepformer/data/supervision_census.py` on any corpus before training on it — it
prints supervision density, the class mix, and the majority-class collapse baseline to compare
`val/face_acc` against.

### Scored on one common split, the sweep produced nothing

Every surviving checkpoint re-scored on **`defeature_processed/test`** — 157 real hand-labelled
parts, 5-class face macro F1, one forward pass each, no training
(`scan_checkpoints.py --device cpu`, so it does not touch the GPU a trainer is using):

| macro F1 | chamfer | `random` | checkpoint |
|---|---|---|---|
| **0.8886** | 0.874 | 0.820 | `trial1_finetune_ss` ep32 — **the initialization** |
| 0.8860 | 0.857 | 0.819 | `t6ctl_init_ft` ep01 (defeature GT) |
| **0.8564** | 0.836 | 0.778 | `trial1_ss1500` ep57 — **the initialization** |
| 0.8454 | 0.810 | 0.777 | `t6_init_def_wf` ep29 (trial6 masked + weighting) |
| 0.7669 | 0.821 | 0.639 | `t6_init_def` ep00 |
| 0.6525 | 0.646 | 0.444 | `t6_init_def` ep25 |

This is the same split and metric as [Face Segmentation on Real Parts](#face-segmentation-on-real-parts-defeature-5-class)
above, so the two tables compose: `trial1_finetune_ss` ep32 was already rank 1 of 26 there at
0.8886, and the sweep's best arm lands *below* it.

**No arm beat the checkpoint it was seeded from.** 0.8860 against 0.8886; 0.8454 against
0.8564. (Amended 2026-09-03: `t6_init_def_wf` ran on to epoch 67 and reaches 0.8579 against its
init's 0.8564 — **+0.0015**, a tie rather than a win, with its neighbouring epochs straddling the
init at 0.8559 and 0.8543. The best sweep checkpoint still loses to `trial1_finetune_ss` ep32 by
3.1 points.) Nine arms and ~29 GPU-hours produced no weights better than the two files the sweep
started from, and nothing in the per-arm logs could have revealed that, because the init was
never scored as a baseline on the arm's own split. **Score the initialization alongside the
arms, on the same split, always.**

With the confound removed the corpus question has a small answer: retention against each arm's
own init is **99.7%** for exact defeature GT and **98.7%** for trial6 masked pseudo-labels. The
0.8637-vs-0.6958 chasm was two different inits (3.2 points apart on their own) plus two
different val sets. Masked pseudo-labels are nearly harmless — and nearly useless, since
neither corpus gained. Loss weighting remains the biggest knob at **+7.9 points** here
(0.7669 → 0.8454), against the +12 its own val reported.

And `t6_init_def` **degrades monotonically the longer it trains** — 0.7669 → 0.6807 → 0.6525,
with `random` falling 0.639 → 0.410 — while its own val face F1 held in the 0.64–0.70 band.
That is overfitting to the masked-label distribution, and it is invisible from inside the run.
Put a real-GT split in the early-stopping monitor for any future run on pseudo-labels.

**The 10th arm settles the corpus question, and it is negative.** The sweep driver later started
`t6_init_ft` — trial6 masked labels from the *better* init (0.8886 rather than 0.8564). It lands at
**0.8426**, destroying 4.6 points. Read the two init arms together: they begin 3.2 points apart and
converge to within 1.5 points of each other (0.8579 and 0.8426), one pulled up 0.2, the other
dragged down 4.6. Both land near 0.85. **The trial6 masked labels set the ceiling, not the
initialization, and that ceiling is about 0.86 on real parts.** Training on them cannot lift a
checkpoint above their own quality; it regresses a better one toward them. So the lever is label
quality — `fuse_labels.py`, the learned edge prune, the `face_attr` layout — not another sweep over
training knobs.

**Final tally, all 11 arms (2026-09-04).** Two more arms ran from the better init:
`t6_init_ft` 0.8444 and `t6_init_ft_wf` 0.8392. Together with `t6_init_def_wf` 0.8579 and
`t6_init_def` that is four initialized arms landing in a **2-point band** (0.8376–0.8579) from
inits **3.2 points apart**, half of them with loss weighting and half without. **11 arms, ~72
GPU-hours, no checkpoint better than the weights file that already existed** — `trial1_finetune_ss`
ep32 still leads by 3.1 points.

This also reverses the loss-weighting result. From the weaker init weighting helps (0.7669 →
0.8579); from the better init it *hurts* (0.8444 → 0.8392). It was never worth "+7.9 points" — it
was pulling a lagging arm up to a ceiling the other arm reached without it. **A knob measured from
one initialization tells you nothing about the knob.**

One claim elsewhere in this file narrows as a result. The best checkpoint scores **0.874 on
`chamfer`** over 157 real parts; the 12% chamfer recall behind "geometry beats the network" was
measured on 59 **mftrcad** models remapped 27→5. Both numbers stand, but the conclusion is
"the network cannot do chamfer *on remapped mftrcad*", not in general.

### Rotation robustness: the tail, not the mean, and no augmentation exists

A different question from the one above: not *which knob helped* but *does the model depend on
the world frame*. Worth asking because FoV-Net reports UV-Net collapsing from **98.95 to 35.44**
per-face accuracy on MFCAD++ under random test rotation, and real parts arrive in whatever frame
their author used. `brepformer/so3_probe.py` rotates only the frame-dependent slots of a
preprocessed sample (face/edge grid points, normals, tangents, face centroid), leaving every
rotation invariant alone, so a drop cannot be blamed on a corrupted input.

| checkpoint | identity | mean | **worst frame** | SO(3) retention |
|---|---|---|---|---|
| `trial1_ss1500` ep57 | 0.8564 | 0.7534 | 0.5898 (so3-1) | 86.9% |
| `trial1_finetune_ss` ep32 | 0.8886 | 0.8696 | **0.8470** (so3-5) | **97.1%** |
| `t6_init_def_wf` ep29 | 0.8454 | 0.7386 | 0.5770 (**octahedral**-2) | 88.2% |

**The catastrophe does not reproduce** — worst-case retention is 67–69%, not 36% — so don't
reprioritise the `face_attr` layout fix behind rotation. **But the tail is heavy and it is not
about SO(3) specifically:** `t6_init_def_wf`'s worst frame is an *axis-aligned 90° rotation*,
worse than any of six random SO(3) draws. Report the worst frame; the mean hides it.

**`hole` collapses 0.971 → 0.120** in that frame, and the mechanism is the same one as the
layout bug. A hole is a cylinder distinguished from every other cylinder by its *axis
direction*, and after `normalize_geometry` the axis is not in `face_attr` at all — only one-hot
type, area and centroid. With the discriminative attribute withheld, the network substitutes a
frame-dependent proxy read off the UV-grid normals.

**There is no geometric augmentation anywhere in the training path.** Neither
`data/preprocessed_dataset.py` nor `train_preprocessed.py` contains any rotation or flip; the
only `rotat` match under `data/` is a docstring calling D2 rotation-invariant. So this is not
"we do 90° and need SO(3)" — it is that nothing is augmented. That makes it the cheapest fix
available: rotating a loaded sample is a tensor op `so3_probe.rotate_sample` already implements,
so it needs no re-preprocessing, just a flag in the dataset's `__getitem__`.

`trial1_finetune_ss` ep32 is both the best checkpoint on the common split and by far the most
frame-invariant, losing 4.7% in its worst frame. It is the production checkpoint on both counts —
and `defeature.py` / `defeature_v2.py` now default to it.

**The flag now exists.** `--rotate_aug none|octahedral|so3` on `train_preprocessed.py`, applied in
`PreprocessedDataset.__getitem__` on the **train split only**:

```bash
python brepformer/train_preprocessed.py ... --rotate_aug so3 --exp_name aug_so3_def
```

Three choices in it worth stating, because each is a way this is usually done wrong:

- It calls `so3_probe.rotate_sample` directly, so the frames trained on are generated by the same
  code that measures robustness — one definition of "what carries the frame", not two that drift.
- **Val and test stay unrotated.** Rotating val would change what `val/face_f1` means and break
  comparability with every run in the tables above; the early-stop monitor has to keep ranking the
  same quantity. Robustness is measured afterwards, by the probe.
- Each dataloader worker gets its own RNG stream. Sharing one seed across workers is the standard
  way augmentation silently ends up `num_workers`× less diverse than intended.

`rotate_sample` was also changed from a `deepcopy` to copying only the three arrays it rewrites.
The full copy duplicated `d2_distance` — (F+1, F+1, 64) float32, **256 MB on a 1000-face part** —
on every call, fine for a probe that runs once and fatal inside a training loop.

Verified before use: for both modes face labels are unchanged, normals stay unit length, the
invariant slots (`spatial_pos`, `in_degree`, `edge_attr`, `d2_distance`) are bit-identical, two
visits to one index give different frames, and the area column behaves as re-preprocessing would
(unchanged under axis permutation, 1.3675 → 1.0693 under SO(3), since the axis-aligned bbox the
normaliser measures grows when the part is turned inside it).

### The sweep driver now refuses to spend GPU-hours on the collapse (2026-09-04)

The finding above — every from-scratch arm collapses, and no arm beat its own initialisation — was
sitting in a table where the next person to run `trial6_sweep --run --family 27` would have paid
for it again. Five of the seven family-27 arms have no `--init_weights`, on ~12× the samples per
epoch. So three things moved out of the prose and into `brepformer/sweeps/trial6_sweep.py`:

**From-scratch arms are opt-in.** `select()` drops every arm marked `scratch=True` unless
`--include_scratch` is passed, which makes `--run --family 27` cost two arms instead of seven.
`--emit` still prints all of them: SCRIPTS.md documents what the sweep *is*, and a table row whose
command is missing is a row nobody can check.

**`--attr_layout` is always emitted, never defaulted.** The model's default decoding changed the
same day (see "The mismatch" above) and the same weights score 82.95% under one layout and 32.75%
under the other. A command in a results table that omits the flag stops reproducing its own number
the moment the default moves — so `command()` pins it, and the default here is `categorical`,
the layout the recorded numbers were produced under, not the one that is now correct.

**A re-sweep cannot overwrite the run it is being compared against.** `suffix()` derives
`t6_init_ft__onehot_rotso3` from the non-default flags, and `run_all` keys the results directory,
the log and the `status.json` entry on that suffixed name. Keying on the bare arm name — the
obvious implementation — has two distinct failure modes, and the second is silent: the re-sweep is
reported "already complete, skipped" against the *original* arm's status entry, and if it does run
it writes over the directory the tables cite.

```bash
python -m brepformer.sweeps.trial6_sweep --run --family 27              # 2 warm-start arms
python -m brepformer.sweeps.trial6_sweep --run --family 27 --include_scratch   # all 7
python -m brepformer.sweeps.trial6_sweep --run --family 5 \
    --attr_layout onehot --rotate_aug so3                               # -> t6_*__onehot_rotso3
```

`status.json` now also records `attr_layout`, `rotate_aug` and `scratch` per arm, so the
provenance of a number is readable off the status file rather than reconstructed from a log.

The runnable form of all of this — in dependency order, with the checkpoint-scoring,
D2-repair, auto-accept and rotation-probe commands that follow it — is the last section of
`SCRIPTS.md`, "Runnable scripts — the current queue".

## Where This Stands, and What Each Remaining Step Would Settle

Eight bugs have been filed. Some are fixed in code, one is deliberately left in place, and two are
open with arms designed but not run. Several have now been *measured*, and measuring them is what
changed the plan — a fix that is correct and a fix that helps are different claims, and this project
has one of each.

| # | bug | state | measured effect |
|---|---|---|---|
| 1 | `infer.py` collapsed 27→5 by argmax-then-map | fixed | marginalize-then-argmax; `balanced` mode for macro-F1 reporting |
| 2 | `geometric_rules.py` co-axiality tested direction only | fixed | part of the false-denial drop, 15.0% → 0.6% |
| 3 | `must_be_planar` denied every conical chamfer | fixed | same |
| 4 | face-attribute pathway read the wrong columns | fixed | **negative**: 0.8699 vs the incumbent's 0.8886 |
| 5 | `GraphAttnBias.edge_conv_layer` never called | folded into bug 7 | never runs, so no effect either way |
| 6 | RoPE table 512 < `max_faces` 1000 | fixed | latent (needs `use_rope`); now guarded with an actionable error |
| 7 | the whole multi-hop `edge_path` pathway is dead, in three places | **open**, arms designed | unmeasured; edge geometry reaches attention only between edge-sharing faces |
| 8 | UV validity mask dead on real parts, live in pretraining | 3 candidate fixes built (`--mftrcad_uv`, re-derive MFTRCAD, `--mask_policy`); L0–L4 ladder written | unmeasured; 28.5% of real faces fed off-face geometry as valid |

Bug 5 turned out not to be standalone: it is one step of bug 7's mechanism, and the earlier note that
removing it would break `load_from_checkpoint` was wrong — `edge_conv_layer` is already constructed
and serialized in every checkpoint, so wiring it into `forward` adds no parameters. It is tested as
part of bug 7, not separately.

### The two rules this program keeps re-learning

**Score the initialization as a baseline.** Nine family-5 arms failed to beat the checkpoint they
started from, and the sweep would have reported hyperparameter conclusions from that if the init had
not been scored. `trial5` had no `bestface-*` checkpoints at all — it predates face-F1 monitoring —
so `t6a_init_t5`'s 0.8592 was a number with nothing to compare against. Measured on the same 250
held-out models, the arm posts **0.8832 macro against the init's 0.8655**: the first arm in this
program to beat its own init. The interesting part is that weighted F1 moved only +0.2 while macro
moved +1.8, because the gains are on rare classes (`circular_blind_slot` n=19, 0.621 → 0.889;
`rectangular_through_slot` n=40, 0.731 → 0.860) and the small losses are on common ones
(`rectangular_passage` n=423, 0.891 → 0.864). That is what adding a corpus enriched in
under-represented features looks like — and it is exactly the result for which the `t6actl_base`
control is load-bearing, since without it "the data helped" and "27 more epochs helped" are one
number.

**A number measured on the selection split is an upper bound.** That +1.8 was measured on
`trial6a_processed/val`, which is the arm's own early-stop monitor, while `trial5` never saw it for
selection. It has to be repeated on `test` before it is quotable. The same caution applies to
`fix4_onehot_def`, which *beat* its twin on val (0.8714 vs 0.8637) and lost on test (0.8699 vs
0.8886) — had the bar not been pre-registered on test, the layout fix would have been recorded as a
win.

### Remaining work, and what each step decides

Full commands are in `SCRIPTS.md` §9, "What is still outstanding, in dependency order".

1. **Re-score the arm against its init on `test`.** Decides whether the rare-class gain is real or
   selection bias. Blocked only on the arm finishing (~14 h; plateaued since ep21).
2. **`t6actl_base`, the mftrcad-only control.** Decides whether the gain belongs to the pcq27 data or
   to the extra training. It is from-scratch, so expect collapse and read it as a floor.
3. **Repair `defeature_processed`, then one 5-class arm.** The only clean attribution of the D2 fix,
   because the 0.8886 baseline was measured on the unrepaired corpus. Hold `--attr_layout
   categorical` or it measures two fixes at once.
4. **The rotation arms.** `--rotate_aug` is built and verified label- and invariant-preserving, but
   has never trained a model, so its effect is unknown. Run `octahedral` alongside `so3`: if the
   axis-aligned frames recover `hole` and uniform SO(3) does not, the failure is grid interpolation,
   not the frame — different problem, different fix.
5. **27-class fusion and a 27-class accept/review table.** Both need 27-class posteriors from a
   finished arm. The 27-class thresholds are the more interesting ones: with 27 classes several will
   certify nothing at any confidence, which turns the review queue from confidence-targeted into
   class-targeted.

Two things are deliberately *not* queued. The `onehot` re-sweep, because the layout A/B was negative
and reopening it needs a bigger corpus or an init that never saw the broken layout — not another
arm. And bug 5, because it is an architecture change that must not ride along inside an experiment
about something else.

### The 27-Class Arm's Gain Is Real, and It Is Not Only the Rare Classes (2026-09-08)

`t6a_init_t5` finished at 100 epochs / 62 h, best 27-class val face F1 0.8676 at ep74 — it kept
improving for 44 epochs after a stretch that looked like a plateau, so a plateau should not be read
off two consecutive val checkpoints on this corpus.

Scored against its own `trial5` initialization on the full `trial6a_processed/test` split (2923
models, no subsampling): **macro F1 0.8062 → 0.8529 (+4.7), weighted 0.8721 → 0.9068 (+3.5),
min-class 0.5234 → 0.6528.** 23 of 27 classes improve.

> **RETRACTED 2026-09-17.** `trial6a_processed/test` is not a ground-truth split — it is a symlink farm
> of 2856 MFTRCAD GT models plus **81 pcq27 models labelled by the very teacher this arm was trained to
> imitate**. On the clean MFTRCAD-only split the arm gains **+0.0068** macro and the corpus control,
> which never saw pcq27, gains **+0.0070**. The +4.7 was contamination. Everything in the rest of this
> subsection is measured on the contaminated split; see "The corpus control retires the +4.7" below for
> the corrected numbers and the process changes.

Two corrections to what was recorded from the val measurement:

**The val number understated the win rather than inflating it.** +1.8 macro on 250 val models became
+4.7 on 2923 test models. The concern was that val is the arm's own early-stop monitor, so the figure
should have been an upper bound; instead it was a small-sample effect. Selection bias was the right
thing to worry about — `fix4_onehot_def` won on val and lost on test — it just was not what happened
here.

**The gain is not confined to rare classes.** Split at median support (n = 1493 faces), the 14 common
classes gain +3.5 mean F1 (12 up, 2 down) and the 13 rare ones +5.9 (11 up, 2 down). Rare classes gain
more per class, but the common half holds 87% of the faces, which is why weighted F1 moved +3.5 and
not the +0.2 seen on val. `stock` — 25,834 faces, the largest class — goes 0.963 → 0.983. `Oring`
(n=5532) goes 0.654 → 0.956. And the two classes this project cares about most, because their
confusion is its characteristic failure and its costliest auto-accept error, both improve: `chamfer`
0.828 → 0.912 and `round` (the 27-class fillet) 0.909 → 0.952.

These are held-out models, used for neither training nor checkpoint selection. The limit worth
stating: `test` is the same distribution as `train`, so this is generalization across models, not
across corpora — cross-corpus generalization is a separate measurement against `defeature_processed`
or the Fusion 360 set.

**The two real regressions are common classes, and they are one confusion.**
`rectangular_passage` 0.892 → 0.760 and `6sides_passage` 0.896 → 0.815. The arm over-predicts
`6sides_passage` (recall 0.918 at precision 0.734) and eats `rectangular_passage`: two classes
separated only by side count. That is a target for the next arm, not a reason to prefer an init that
is 4.7 macro points behind.

What this still does not establish is *why*. The comparison is arm-versus-init, so the pcq27 corpus
and 62 h of additional training are the same number until `t6actl_base` separates them. That control
is now the highest-value open item.

### Two Process Failures Worth More Than the Arms They Delayed (2026-09-13)

**A crashed run looks exactly like a running one.** `t6a_init_t5_wf` died at epoch 11 with
`RuntimeError: CUDA error: unknown error` — raised from inside `torchmetrics`, after ten clean epochs
on identical batch shapes, on a GPU that tests fine now. That is the WSL2 device-dropout symptom, not
a model or data fault, and it is not the expensive part. It was the last arm in the driver's
`subprocess.call` loop, so the loop exited and the GPU sat at 0% for **four days** while `results/`
still showed an arm directory with recent checkpoints in it.

The sweep driver now takes `--retries` (default 2) and relaunches with `--resume_from last.ckpt`.
Three details that matter more than the retry itself: retries *append* to the log so the original
traceback survives; the resume flag is mandatory, because a plain relaunch would restart from
`--init_weights` at epoch 0 and report a different experiment under the same arm name; and an arm that
dies before writing `last.ckpt` is deliberately not retried, since a first-epoch crash is far more
likely a config error than a dropped device. A crashed arm recorded in `status.json` is also picked
up rather than restarted, gated on the recorded return code so a *completed* arm is never silently
continued. The operational rule: `nvidia-smi` is part of answering "how is training going", because
an idle GPU and a training run are indistinguishable from the filesystem.

**The control that was going to attribute the +4.7 could not have.** `t6actl_base` was queued as the
arm that separates "the pcq27 corpus helped" from "62 more hours of training helped". It is
`scratch=True` — it removes the pcq27 data *and* throws away `--init_weights`. Had it collapsed, and
every from-scratch arm in this program has, the result would be explained equally well by "the pcq27
data was necessary" and by "no 27-class model trains from scratch on this hardware". It confounds the
exact variable it exists to isolate, which is the same mistake the sweep was designed to prevent and
which survived in it for a month.

`t6actl_init_t5` replaces it: `--init_weights` pinned to `trial5`, corpus switched from
`trial6a_processed` to `mftrcad_processed`, every other flag byte-identical to `t6a_init_t5`. One
variable moves, so the gap between the two on `trial6a_processed/test` *is* the pcq27 contribution.
(It is not, and finding out why is what the control was actually worth — `trial6a_processed/test` turned
out to contain 81 pcq27 pseudo-labelled models. The measurement had to be redone on
`mftrcad_processed/test`. See below.)
`t6actl_base` stays in the sweep behind `--include_scratch` as a floor, but nothing depends on it.

The general lesson, which the layout A/B taught in a different key: a control has to differ from its
treatment in **one** thing. A control that changes two is not a weak measurement, it is not a
measurement.

### The Corpus Control Retires the +4.7, and Finds a Contaminated Test Split (2026-09-17)

`t6actl_init_t5` finished: same init as the arm (`trial5` ep39), same 200-epoch schedule, same every
flag, **mftrcad only — no pcq27**. It is the one-variable control described above, and it does not
confirm the arm.

Scored the way every earlier 27-class number was scored, on `trial6a_processed/test`, it looks like an
even stronger confirmation: the control reaches **0.7841** macro against the arm's 0.8529 — it lands
2.2 points *below* the shared init, so the pcq27 corpus appears to be worth +6.9. That reading is
wrong, and the split is why.

**`trial6a_processed/test` is not a ground-truth split.** Resolving its symlinks: 2856 point into
`mftrcad_processed/test` (exact synthetic labels), and **81 point into
`suitable_samples_pcq27_processed/test`** — GrabCAD parts whose labels are pseudo-labels emitted by the
teacher `t6a_init_t5` was trained to imitate. `merge_suitable_samples.py` splits the merged pool
0.8/0.1/0.1 like any other corpus, which is right for training data and quietly self-serving for
evaluation. 2.8% of the models, and enough to move a 27-class macro by 5 points because those 81 carry
21 of the 27 classes at very low support.

On the uncontaminated `mftrcad_processed/test` (2856 models, exact labels):

| checkpoint | trained on | macro | weighted | min-class | vs init |
|---|---|---|---|---|---|
| **`t6actl_init_t5` ep61 (control)** | **mftrcad only** | **0.8781** | 0.9280 | 0.6492 | **+0.0070** |
| `t6a_init_t5` ep74 (the arm) | mftrcad + pcq27 | 0.8779 | 0.9269 | 0.6528 | +0.0068 |
| `t6a_init_t5_wf` ep85 | mftrcad + pcq27 | 0.8753 | 0.9246 | 0.6297 | +0.0042 |
| `trial5` ep39 (shared init) | mftrcad | 0.8711 | 0.9233 | 0.6556 | — |

**The control ties the arm to within 0.0002 macro.** On ground truth, the pcq27 corpus is worth +0.7
points; extra epochs on mftrcad alone are worth the same +0.7; and the difference between having the
corpus and not having it is two ten-thousandths of a point. The entire +4.7 lived in 81 test models.

Scoring those 81 alone says precisely what the arm learned:

| checkpoint | macro on the 81 pcq27 models |
|---|---|
| `t6a_init_t5` ep74 (arm) | **0.5132** |
| `t6a_init_t5_wf` ep85 | 0.4859 |
| `trial5` ep39 (init) | 0.3351 |
| `t6actl_init_t5` ep61 (control) | **0.2145** |

The arm beats its init by **+18 points** there; the control *loses* 12. So the pcq27 training worked,
at the thing it was actually optimising: reproducing its own teacher's labels on GrabCAD-shaped inputs.
That is not the same as being correct. The same checkpoint scores 0.4287 against the 157 **hand-labelled**
real parts versus its init's 0.5977 (next section). It agrees with the teacher more and with the truth
less — which is the failure mode self-distillation always has, and which a pseudo-labelled test split
cannot see by construction.

Three consequences, all process rather than another arm:

1. **`trial6a_processed` and `trial6_processed` are training corpora only.** Both are pcq-merged symlink
   farms and both leak pseudo-labels into `test`. Evaluate on `mftrcad_processed/test` (synthetic GT)
   and `defeature_processed/test` (157 real hand-labelled parts), and report both.
2. **A pseudo-label corpus must not contribute to its own test split.** The merge step needs to route
   pseudo-labelled models to `train` only.
3. **The control was worth its 41 GPU-hours because it came back negative.** Without it, "the corpus
   helped" and "the split was contaminated" produce the identical +4.7, and the contamination is
   invisible. It also caught a failure mode it was not designed for — which is the argument for running
   controls even when the treatment already looks won.

Commands and the symlink-resolution snippet are in `SCRIPTS.md` under "The corpus control retires the
+4.7".

### The 27-Class Gain Does Not Survive Contact With a Real Part (2026-09-14)

Everything the 27-class program measured was same-distribution: held-out mftrcad + pcq27 models. The
deployed use case is real designer parts, and the only real hand-labelled evidence in this repo is the
157 parts in `defeature_processed/test`. Scoring the 27-class winner there, marginalized into the 5
defeature classes, asks the question the synthetic numbers cannot: does +4.7 macro mean anything for
the thing that ships?

| checkpoint | head | macro | random | hole | chamfer | fillet | cut |
|---|---|---|---|---|---|---|---|
| `trial1_finetune_ss` ep32 (deployed) | 5 | **0.8886** | 0.820 | 0.954 | 0.874 | 0.887 | 0.908 |
| `trial5` ep39 (the 27-class init) | 27 | 0.5977 | 0.369 | 0.729 | 0.704 | 0.520 | 0.666 |
| `t6a_init_t5` ep74 (the +4.7 arm) | 27 | **0.4287** | 0.170 | 0.276 | 0.614 | 0.487 | 0.596 |

The arm is **17 points worse than its own initialization** on real parts while being 4.7 points better
on synthetic ones. The regression is worst on `hole` (0.729 → 0.276) and on `random` (0.369 → 0.170) —
`random` being where `stock` maps, the class that gained *most* on synthetic data (0.963 → 0.983). The
model learned the mftrcad/pcq27 notion of a stock face more sharply, and that notion is not the one a
real part has.

What this licenses and what it does not. The 27-class models were never trained on defeature labels
and the 27→5 map is lossy, so the *level* (0.43 vs 0.89) is partly a label-space penalty, not a fair
head-to-head. The *direction* is sound: `trial5` and `t6a_init_t5` are the same architecture scored the
same way through the same mapping, and more synthetic training moved that comparison 17 points the
wrong way. The +4.7 was correct on its own terms and *was* the first arm in the program to beat its
init — until the corpus control showed the split it was measured on contained the arm's own teacher's
labels; on clean ground truth it is +0.7, and so is the control's. It did not merely fail to generalize
off-corpus; on-corpus it was never there. Every future 27-class arm now reports both numbers,
because on the synthetic axis alone this looked like the program's best result.

Scoring a 27-class head in the 5-class space needs `--collapse marginalize`. The default,
`argmax_remap`, takes the 27-class argmax and maps it down — bug 1's pattern, which understates the
model because mass split across several ids mapping to one defeature class loses to a single id that
does not. It stays the default only so the 27-class-at-`--label_space 5` numbers already recorded stay
comparable; new claims must use `marginalize`, the rule `infer.py` deploys.

### Bug 7: the Multi-Hop Edge Pathway Is Dead in Three Places, and Bug 5 Is One of Them

Auditing bug 5 for its own arm showed that `edge_conv_layer` is not one unused module but the visible
end of an entirely absent mechanism — Graphormer's multi-hop *edge* encoding, the paper's central
device for getting edge features into attention:

1. **`edge_path` is never produced.** `compute_shortest_paths` ends `return spatial_pos, None`, its
   docstring saying "(optional, not implemented)". No pickle in any corpus carries the key.
2. **So section 5 of `GraphAttnBias.forward` never executes** — it is guarded by
   `if edge_path is not None`. `edge_dis_encoder` is therefore a second dead module, and
   `config.multi_hop_max_dist = 16` is a knob wired to nothing.
3. **Even if `edge_path` were supplied, section 5 would carry no edge information.** It embeds
   `torch.full_like(hop_edges, hop + 1)` — the hop *number* — and never gathers the edge features at
   the indices `hop_edges` holds. That is a learned per-hop constant applied wherever a path of that
   length exists: a function of `spatial_pos` alone, and thus redundant with the `spatial_pos_encoder`
   two sections above it.
4. **`edge_conv_layer`** is the aggregation step that would consume the gathered features.

The consequence: edge features reach attention through exactly one route, the direct-neighbour
`scatter_add` in section 4. Two faces that do not share an edge exchange only shortest-path distance
and the D2/angle descriptors — no edge geometry whatsoever. For fillet-versus-chamfer, where the
discriminator is the convexity of edges along a short path between faces, that is precisely the
pathway that should matter; BRepNet's ablation prices convexity removal at -4.6 IoU, its single
largest feature loss.

**The bug-5 entry was wrong in both directions.** It is not a standalone fix — it is one step of this
mechanism — and it does not break checkpoint loading. `edge_conv_layer` and `edge_dis_encoder` are
already constructed and already serialized in every checkpoint, carrying untrained weights. Wiring
them in adds no parameters, so every checkpoint stays loadable and a warm start works normally.

**The cost is in preprocessing, not the model.** `edge_path` is `(N+1, N+1, max_dist)`: at
`max_faces 1000` and `multi_hop_max_dist 16` that is 128 MB per sample on top of the 256 MB the D2
tensor already costs, and the collator pads to the batch maximum. Any implementation caps
`multi_hop_max_dist` at ~4 — the useful range for fillet/chamfer adjacency — and stores int16/int32,
or the arm OOMs before it trains. Path-tracking BFS is also materially slower than the current
distance-only BFS across 28.5k models.

**Two arms, and the control is not optional.** The treatment needs a re-preprocessed corpus, and
re-preprocessing also picks up the D2 repair — two changes, one number, the same mistake as
`t6actl_base`. So the D2 arm has to land first and become the baseline, and `fix7ctl_noedgepath` reads
the *same* new pickles while ignoring the `edge_path` key. The treatment-control gap is then the
mechanism alone. Commands and the three flags that must be built first are in `SCRIPTS.md`.

### Bug 8: the UV Validity Mask Is Dead on Real Parts and Alive in Pretraining (2026-09-15)

This one was not found by reading model code. It was found by asking what is actually *in* the inputs.

`face_grid` is seven channels: xyz, a unit normal, and a **mask**. The mask exists because a B-rep
face is a *trimmed region* of a surface, and its trimmed region is almost never a rectangle in
parameter space. Grid the face's UV box and some samples land outside the face — inside a
through-hole in a planar wall, off the corner of a triangular pocket floor, past the taper of a
narrowing fillet. Channel 6 is what tells the SurfaceEncoder which samples are real.

`step_to_graph.compute_face_uv_grid` never computed it. It gridded the UV box and wrote `mask = 1`
wherever the surface evaluated successfully. So on every corpus this repo derives from STEP —
`defeature_processed` and everything downstream of it — channel 6 is **identically 1**: a dead input,
and every off-face sample handed the underlying surface's xyz and normal and labelled valid.

MFTRCAD's published graphs are not like that. Over 3,798 faces of `mftrcad_processed/test`, 44.5% of
grid points are masked, every face has at least one masked point, and 21.6% of faces are more than
half masked. Over 11,544 faces of `defeature_processed/test`, the figure is 0.000.

That is the finding: **the SurfaceEncoder learns channel 6 as a validity gate on the pretraining
corpus and is handed a constant on the real parts it is deployed on.** It is an input-channel
distribution shift sitting precisely on the transfer this whole program depends on, and it is a
candidate mechanism for the anti-transfer measured a day earlier (the 27-class arm at 0.4287 macro on
real parts, 17 points *below* its own init, with `hole` falling 0.729 → 0.276 — and hole walls are
exactly the faces whose UV box overshoots the face most).

#### The convention is fitted against MFTRCAD, not guessed

The obvious fix is to classify every UV sample against the face's trim loops with
`BRepTopAdaptor_FClass2d`. That gets the direction right and the level wrong: run on MFTRCAD's *own*
STEP files and compared to their published masks on the same 410 faces, an unpadded trim test gives
0.196 against their 0.452. Correlation 0.896 — the same quantity, measured at less than half the
magnitude.

Sweeping a UV padding applied before sampling settles it. At pad 0.10 the reproduction is 0.463
against 0.452 with mean absolute per-face error 0.038; at 0.15 and above it overshoots to 0.68 and the
per-face error grows sevenfold. The competing hypothesis — that MFTRCAD grids the *underlying
surface's* full parameter range, a whole cylinder's 0..2π — is wrong, giving 0.866.

So MFTRCAD pads the face UV box by 10% of its extent on each side, then masks `TopAbs_OUT`. Scoring
per face rather than per corpus is what makes this a fit rather than a coincidence: a corpus mean is
easy to hit by accident, and two conventions with very different per-face behaviour can share one.

This also means the mismatch has **two halves, not one**. Our grids are taken over a UV domain 10%
narrower on each side than the pretraining corpus's, so even the xyz and normal channels are sampled
at different parameter locations — and then everything is declared valid. Both halves are one flag
(`--mftrcad_uv`), because padding without masking would be strictly worse than doing nothing: it
samples further off-face and still calls it valid.

#### What it costs, and one claim that did not survive measurement

On 622 faces of 12 real `defeature` parts: 28.5% of faces are fed at least one off-face sample as
valid, and 10.0% of faces are more than half off-face (mean 10.6% of grid points, median 0 — so it is
concentrated on the trimmed faces rather than smeared across all of them).

The claim I *expected* to matter more was that this also corrupts normalization: `normalize_geometry`
derives the bbox centre and scale from `mask > 0.5`, which should shift every normalized coordinate
and the area scaling. Measured on the same 12 parts, it does not — the scale is identical on 12/12 and
the centre moves by at most 0.6% of the half-extent. The part bbox is set by the extreme faces, whose
UV boxes are tight, and the overshoot is interior. Withdrawn. Bug 8 is an input-channel and
sampling-domain bug, not a normalization bug.

#### The cheap read comes before any GPU time

The fix is off by default: flipping it rewrites every pickle and would make new numbers incomparable
with every table recorded here. The first experiment needs **no training at all** — re-preprocess the
real corpus under `--mftrcad_uv` into a new tree and re-score the checkpoints that already exist. One
forward pass per model.

The asymmetry is the diagnostic, and stating it in advance is what makes it falsifiable:

- `trial5` and `t6a_init_t5` saw only MFTRCAD-convention grids in training. If bug 8 is what broke
  them on real parts, they should **improve** on the fixed split.
- `trial1_finetune_ss` was fine-tuned on the broken-mask corpus and therefore adapted to `mask = 1`.
  It should **degrade** — not a failure of the fix, but the fix removing an input the incumbent
  learned to lean on.
- If all three move together, the mask is not the mechanism, and bug 8 is a data-quality improvement
  with no bearing on the anti-transfer.

Only a confirmed asymmetry justifies the re-finetune arms (`fix8_uv_def` against `fix8ctl_legacy_def`
at identical hyperparameters). `mftrcad_processed` is deliberately left alone: it already has the
right convention, which is the entire point. Commands are in `SCRIPTS.md` under "Bug 8", and
`brepformer/data/uv_mask_audit.py` reproduces every number above in three modes (`disk`, `steps`,
`fit`).

#### What the audit ruled *out*, which is most of what it looked at

Worth recording, because the audit's first three candidates all dissolved and it would be easy to
re-derive them later as if they were new:

- **"The dropped `face_attr` columns starve the model."** `normalize_geometry` discards the face
  normal, bbox extent, loop count and orientation it computes, so four of the eight face-attribute
  encoders receive zeros. But per-point unit normals are in `face_grid` channels 3–5 and the bbox is
  derivable from channels 0–2, so the SurfaceEncoder already sees both. Largely redundant, not
  starved.
- **"The dropped dihedral angle is why chamfer/fillet confuse."** `edge_grid` is 12 channels: xyz,
  tangent, and the normals of *both* adjacent faces sampled along the curve. The dihedral is the angle
  between channels 6–8 and 9–11, so it is present per sample point and reaches the model through
  `curve_encoder`. Supplying the scalar explicitly would be a cheap inductive bias, not new
  information.
- **"Average-pooling the UV grid destroys curvature."** `AdaptiveAvgPool2d(1)` follows three conv
  layers, which can compute local normal differences before the pool. The signal survives in a
  learnable form.

What did *not* dissolve was the mask — and it is the one that differs between corpora rather than
being merely imperfect on both. That is the general lesson: a feature that is equally crude
everywhere costs accuracy, but a feature that means *different things* on either side of a transfer
costs the transfer.

#### Fixing it recursively: three candidate fixes and a five-rung ladder (2026-09-17)

One code change is not a fix. Bug 8 is *two* mechanisms plus a third quantity that moves with them, so
a single before/after number would be uninterpretable no matter which way it came out:

1. **Mask semantics** — channel 6 said "the surface evaluated here", not "this point is on the face".
2. **Sampling domain** — MFTRCAD pads the UV box 10% per side, so our xyz and normals are sampled over
   a 10%-narrower domain as well. Repairing only the mask leaves this half in place.
3. **The attention-bias descriptors, confounded with both.** `compute_d2_descriptor` and
   `compute_angle_descriptor` select points with `face_grid[i, 6] > 0.5`, so fixing the mask silently
   changes the Graphormer attention bias too. `preprocess.py --descriptors_ignore_mask` reproduces the
   legacy descriptor content on top of a fixed mask, which is the only way to tell the two apart.

And there are three *independent* fixes, so a null result on the first does not retire the hypothesis:
**(A)** make the real parts match MFTRCAD (`steps_to_graphs --mftrcad_uv`, ~1 h CPU); **(B)** make
MFTRCAD match the real parts by re-deriving its 28,820 STEPs under the legacy convention (~20 h CPU,
the same hypothesis from the other side); **(C)** delete the channel's information on *both* sides with
`--mask_policy const|zero`, which needs no preprocessing whatsoever and is the right fix if channel 6
carries little signal but a lot of shift. C is wired through `train_preprocessed.py`,
`test_preprocessed.py` and `scan_checkpoints.py` — the last patches the loaded pickle directly, because
it bypasses `PreprocessedDataset` and goes straight to `prepare_batch`. `const` and `zero` are
information-identical and differ only in what the first BatchNorm sees; a constant channel is not
automatically inert once it is normalised.

The rungs, cheapest first, each one able to kill the hypothesis before the next is paid for:

| Rung | What it costs | What it decides |
|---|---|---|
| **L0** assert the fix reached the pickles | seconds | Whether the flag threaded at all. It failed to twice — the `--single` worker and the `--tolerate_crashes` re-exec both dropped it, producing a corpus that *looks* converted |
| **L1** re-score existing checkpoints, legacy vs fixed | no GPU | The asymmetric prediction above. All three checkpoints moving the same way ends the climb |
| **L2** corpora C (`--descriptors_ignore_mask`) and D (`--uv_pad 0.0`) | ~1 h CPU each | Attribution: mask vs sampling domain vs attention bias |
| **L3** `fix8_uv_def` / `fix8ctl_legacy_def` | ~60 GPU-hours | Whether the repair is worth deploying |
| **L4** switch to approach B or C | 20 h CPU / free | Only if L1–L2 are flat. Do not re-sweep the padding — that is sweeping, not diagnosing |

Facet **F4 was cleared before any of this**: our face traversal count and order agree with MFTRCAD's
published graphs on 120/120 models, median per-face centroid distance 0.0008 (92.3% within 0.05). That
matters because the whole pad fit is scored per face, and a permuted traversal would have produced a
plausible corpus mean over meaningless pairings. Facet **F5** likewise: masked points keep their
surface geometry in MFTRCAD's graphs rather than being zeroed, and our fix matches.

If A, B and C all come back flat, the honest conclusion is that bug 8 is real as a data defect and
false as an explanation for the anti-transfer. The 28.5% of real faces being handed off-face geometry
as valid is still worth repairing on its own merits — it is just not the thing that cost the transfer.
The full command ladder is in `SCRIPTS.md` under "Bug 8: the recursive fix ladder".

#### L0 and L1 came back confirmed: the mask is worth about a third of the anti-transfer (2026-09-17)

**L0.** `defeature_uv_processed` — 1561 real parts re-derived under `--mftrcad_uv`, 0 failures — reads
`face_grid` channel 6 masked mean **0.400**, median 0.360, 100% of faces masked somewhere, 11.4% over
half. MFTRCAD reads **0.445**, median 0.360, 100%, 22.1%. Same quantity, same median, same shape; the
legacy corpus still reads 0.000. The flag threaded and the two corpora genuinely differ.

**L1, no training, no GPU.** Four checkpoints, the same 157 hand-labelled real parts, 5-class face F1
through `--collapse marginalize`, legacy grids vs fixed grids:

| checkpoint | trained on masks | legacy | fixed | delta |
|---|---|---|---|---|
| `t6a_init_t5` ep74 (worst anti-transfer) | MFTRCAD only | 0.4307 | **0.5064** | **+0.0757** |
| `t6actl_init_t5` ep61 (control) | MFTRCAD only | 0.6017 | **0.6304** | **+0.0287** |
| `trial5` ep39 (shared init) | MFTRCAD only | 0.5884 | **0.6069** | **+0.0185** |
| `trial1_finetune_ss` ep32 (deployed) | **broken (`mask = 1`)** | 0.8853 | 0.8700 | **-0.0153** |

The prediction was written down before the measurement: MFTRCAD-convention checkpoints improve, the
broken-mask incumbent degrades, and all four moving together would have killed the hypothesis. All four
did not move together. Every checkpoint that only saw MFTRCAD's convention gains; the one fine-tuned on
`mask = 1` loses, because the fix removes an input it learned to lean on.

The size is what makes this worth the GPU time. `t6a_init_t5` trails its own init by 0.158 macro on real
parts — the anti-transfer this whole thread is chasing. On the fixed grids it trails by 0.101. **Roughly
a third of the anti-transfer is this one input channel**, recovered with no retraining at all.

Per class it behaves the way the mechanism says it should. `hole` gains most on the worst-hit checkpoint
(**+0.096**), and `hole` is both the class the anti-transfer destroyed (0.729 → 0.276) and the class
whose faces overshoot their UV box hardest: a hole wall is a full-period cylinder trimmed to a partial
arc, so the legacy grid hands the encoder geometry from the far side of the hole and calls it valid.
`random` +0.216 and `fillet` +0.073 follow.

The honest exception is `chamfer`, which loses 0.018–0.042 on **all four** checkpoints, including the
three that gain overall. Chamfers are narrow bands: on a 10×10 UV grid there are few samples to spare,
and masking costs coverage faster than it buys correctness. That is a genuine cost of the fix rather
than noise, and it is the number L3 has to justify — chamfer is also the class the accept-queue always
reviews and the one the labelling workflow most needs.

L2 (attribution: `--descriptors_ignore_mask` vs `--uv_pad 0.0`) and L3 (`fix8_uv_def` against
`fix8ctl_legacy_def`) were launched on the strength of this, L2 on CPU and L3 on the GPU.
