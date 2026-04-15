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

# With 8 real machining feature categories (instead of 27)
python -m brepformer.infer \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output preds.seg --real_classes
```

### FreeCAD Colored Export

```bash
# Export colored STEP (per-face colors viewable in FreeCAD)
python -m brepformer.export_freecad --step model.step --seg preds.seg --output colored.step

# Export with 8 real class colors
python -m brepformer.export_freecad --step model.step --seg preds.seg --output colored.step --real_classes

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

# Viewer with 8 real class display
python -m brepformer.visualize_seg --step model.step --seg preds.seg --real_classes

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
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step

# With full analysis (requires preprocessed dataset)
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step \
    --data_dir brepformer/data/defeature_processed

# Headless (skip interactive viewers)
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step --skip_viewers
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
    --step brepformer/data/defeature/steps/some_model.step \
    --defeatured brepformer/defeatured_output/some_model_defeatured.step

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
