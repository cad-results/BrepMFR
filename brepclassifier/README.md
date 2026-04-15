# BrepClassifier: Pipe Fitting Classification with BrepFormer + GAT

Two-stage B-rep classification pipeline for 8 pipe fitting classes using a pretrained BrepFormer encoder and a GATv2 classification head.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Data Format](#data-format)
- [Scripts Reference](#scripts-reference)
- [Configuration](#configuration)
- [Pretrained Weights](#pretrained-weights)
- [Usage Examples](#usage-examples)

---

## Overview

BrepClassifier classifies entire B-rep CAD models into one of 8 pipe fitting categories:

| Class | Name |
|-------|------|
| 0 | Elbow - Weld Fitting |
| 1 | Elbow - Pipe End Fitting |
| 2 | Elbow - Socket Fitting |
| 3 | Tee - Weld Fitting |
| 4 | Tee - Pipe End Fitting |
| 5 | Tee - Socket Fitting |
| 6 | Elbow - Miscellaneous |
| 7 | Tee - Miscellaneous |

Key features:

- **Transfer learning**: Reuses pretrained BrepFormer encoder from MTFRCAD (27-class machining features)
- **GATv2 classification head**: Graph Attention Network v2 with global attention pooling
- **Handles class imbalance**: Inverse-frequency class weights with cap, stratified splitting
- **End-to-end pipeline**: STEP files -> graph JSON -> preprocessed pickle -> training -> analysis -> visualization

---

## Installation

### Prerequisites

```bash
conda activate brep_mfr

# Required: pythonocc-core for STEP conversion
conda install -c conda-forge pythonocc-core=7.9.0 -y

# Required: torch-geometric for GAT
pip install torch-geometric

# Other dependencies (likely already installed)
pip install pytorch-lightning torchmetrics tqdm scikit-learn
pip install matplotlib seaborn  # for visualization
```

---

## Quick Start

### 1. Convert STEP Files to Graph JSON

```bash
python brepclassifier/convert_steps.py \
    --data_dir brepclassifier/data/ssdata1 \
    --num_workers 4
```

### 2. Preprocess Data

```bash
python brepclassifier/preprocess.py \
    --data_dir brepclassifier/data/ssdata1 \
    --output_dir brepclassifier/data/ssdata1_processed
```

### 3. Train Model

```bash
# With main brepformer encoder (recommended)
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder "results/brepformer/best-epoch=99-val/f1=0.8466.ckpt" \
    --max_epochs 300

# Or with face seg heavy encoder
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder "results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt" \
    --max_epochs 300

# From scratch
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --max_epochs 300
```

### 4. Test Model

```bash
python brepclassifier/test.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --checkpoint results/pipe_classifier/best-epoch=64-val/f1=0.5517.ckpt
```

### 5. Analyze Results

```bash
python brepclassifier/analyze.py \
    --checkpoint results/pipe_classifier/best-epoch=64-val/f1=0.5517.ckpt \
    --data_dir brepclassifier/data/ssdata1_processed \
    --mode all \
    --output_dir analysis_results/pipe_classifier
```

---

## Project Structure

```
brepclassifier/
├── configs/
│   ├── __init__.py
│   └── config.py                  # PipeFittingConfig dataclass
│
├── models/
│   ├── __init__.py
│   ├── gat_head.py                # GATv2 classification head
│   └── pipe_classifier.py         # PL module: encoder + GAT
│
├── data/
│   ├── __init__.py
│   ├── step_to_graph.py           # STEP -> JSON graph converter
│   └── preprocessed_dataset.py    # Pickle dataset loader
│
├── data/ssdata1/                  # Raw dataset
│   ├── 1_elbow_wf/ ... 9_tee_misc/  (class folders with STEP files)
│   ├── graphs/                    (generated JSON graphs)
│   ├── steps/                     (copied STEP files)
│   ├── labels.json                (model_id -> class_idx)
│   └── metadata.json
│
├── data/ssdata1_processed/        # Preprocessed (after preprocess.py)
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── metadata.json              (includes class_weights)
│
├── convert_steps.py               # Batch STEP conversion
├── preprocess.py                  # Preprocessing + stratified split
├── train.py                       # Training script
├── test.py                        # Testing script
├── analyze.py                     # Analysis (6 modes)
├── visualize.py                   # Plot generation
├── viewer.py                      # Interactive 3D viewer
├── visualize_seg.py               # Qt + pythonOCC STEP viewer (whole-model)
├── visualize_face_seg.py          # Qt + pythonOCC per-face seg viewer
├── export_model.py                # Model export
├── export_freecad.py              # Export colored STEP for FreeCAD
├── run_viewer.sh                  # Shell wrapper for all viewers
├── README.md                      # This file
└── VIEWER.md                      # Viewer documentation
```

---

## Model Architecture

```
Input B-rep Graph (same format as brepformer)
       │
       ▼
┌─────────────────────────────────────┐
│ BrepEncoder (from brepformer)       │
│  - SurfaceEncoder: 2D CNN           │
│  - CurveEncoder: 1D CNN             │
│  - 8 Transformer layers             │
│  - [CLS] token + face embeddings    │
│  Output: [batch, N+1, 256]          │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│ GATClassificationHead               │
│  - Convert padded -> PyG Batch      │
│  - GATv2Conv layer 1: 256→256       │
│    (64×4 heads, BN, ELU, skip)      │
│  - GATv2Conv layer 2: 256→256       │
│  - GATv2Conv layer 3: 256→256       │
│  - GlobalAttention pooling → 256    │
│  - Dense(256→512) + BN + ReLU       │
│  - Dense(512→256) + BN + ReLU       │
│  - Dense(256→8) → logits            │
└─────────────────────────────────────┘
```

### Transfer Learning Strategy

- **Encoder**: Loaded from pretrained BrepFormer checkpoint, trained at 0.1x base LR
- **GAT Head**: Trained from scratch at full LR
- **Optional**: Freeze encoder entirely (`--freeze_encoder`)

---

## Dataset

**ssdata1**: 3,610 STEP files of pipe fittings in 8 classes.

| Class | Folder | Count | Description |
|-------|--------|-------|-------------|
| 0 | 1_elbow_wf | 849 | Elbow - Weld Fitting |
| 1 | 2_elbow_pef | 1,110 | Elbow - Pipe End Fitting |
| 2 | 3_elbow_sf | 306 | Elbow - Socket Fitting |
| 3 | 4_tee_wf | 374 | Tee - Weld Fitting |
| 4 | 5_tee_pef | 14 | Tee - Pipe End Fitting |
| 5 | 6_tee_sf | 42 | Tee - Socket Fitting |
| 6 | 8_elbow_misc | 706 | Elbow - Miscellaneous |
| 7 | 9_tee_misc | 209 | Tee - Miscellaneous |

**Note**: Severe class imbalance (14 to 1,110 samples). The pipeline uses:
- Stratified splitting to preserve class ratios in train/val/test
- Inverse-frequency class weights (capped at 10x) in CrossEntropyLoss

---

## Data Format

### STEP to Graph Conversion

Each STEP file is converted to the brepformer JSON format:

```json
["model_name", {
  "graph": {"edges": [[src...], [dst...]], "num_nodes": N},
  "graph_face_attr": [[14 float values per face], ...],
  "graph_face_grid": [[[7 channels, 10×10 UV grid]], ...],
  "graph_edge_attr": [[15 float values per edge], ...],
  "graph_edge_grid": [[[12 channels, 10 points]], ...]
}]
```

### Preprocessed Pickle

After preprocessing, each sample contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `face_grid` | (N, 7, 10, 10) | UV-sampled points + normals + mask |
| `face_attr` | (N, 14) | Face attributes |
| `edge_grid` | (E, 12, 10) | Curve-sampled points + tangents + normals |
| `edge_attr` | (E, 15) | Edge attributes |
| `spatial_pos` | (N+1, N+1) | Shortest path distances |
| `in_degree` | (N,) | Node in-degrees |
| `label` | scalar int64 | Class index (0-7) |

---

## Scripts Reference

### `convert_steps.py` — STEP to Graph Conversion

```bash
python brepclassifier/convert_steps.py \
    --data_dir brepclassifier/data/ssdata1 \
    --num_workers 4 \
    --limit 10  # for testing
```

### `preprocess.py` — Preprocessing + Stratified Split

```bash
python brepclassifier/preprocess.py \
    --data_dir brepclassifier/data/ssdata1 \
    --output_dir brepclassifier/data/ssdata1_processed \
    --split_ratio "0.8,0.1,0.1"
```

### `train.py` — Training

```bash
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder results/brepformer/best.ckpt \
    --max_epochs 300 \
    --batch_size 16 \
    --learning_rate 1e-4
```

### `test.py` — Testing

```bash
python brepclassifier/test.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --checkpoint results/pipe_classifier/best.ckpt
```

### `analyze.py` — Analysis (6 modes)

```bash
# All analyses
python brepclassifier/analyze.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --data_dir brepclassifier/data/ssdata1_processed \
    --mode all

# Specific modes: architecture, per_class, confusion_matrix, predictions, embeddings
```

### `visualize.py` — Plot Generation

```bash
python brepclassifier/visualize.py --mode all \
    --input_dir analysis_results/pipe_classifier \
    --output_dir plots/pipe_classifier
```

### `viewer.py` — Interactive 3D Viewer

```bash
python brepclassifier/viewer.py --mode browse --split test
python brepclassifier/viewer.py --mode predictions --sort worst
python brepclassifier/viewer.py --mode analysis --metrics --confusion
```

### `export_model.py` — Model Export

```bash
python brepclassifier/export_model.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --format state_dict \
    --output_path exported_pipe_classifier.pt
```

### `export_freecad.py` — Export Colored STEP for FreeCAD

Exports colored STEP files with two modes: whole-model (pipe classifier) or per-face (face segmentation).

```bash
# Pipe mode: all faces colored by pipe class
python brepclassifier/export_freecad.py \
    --step model.step \
    --pipe_checkpoint results/pipe_classifier/best-epoch=64-val/f1=0.5517.ckpt \
    --output colored.step

# Face mode: per-face coloring from face seg checkpoint
python brepclassifier/export_freecad.py \
    --step model.step \
    --face_checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output colored.step

# From labels.json
python brepclassifier/export_freecad.py \
    --step model.step \
    --pipe_labels_json brepclassifier/data/ssdata1/labels.json \
    --output colored.step
```

### `visualize_face_seg.py` — Per-Face Segmentation Viewer

Qt + pythonOCC viewer for per-face machining feature class visualization on pipe fitting STEP files.

```bash
# With face seg checkpoint
python brepclassifier/visualize_face_seg.py \
    --step model.step \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt

# With 8 real class remapping + pipe class in status
python brepclassifier/visualize_face_seg.py \
    --step model.step \
    --checkpoint results/face_seg_heavy/best.ckpt \
    --pipe_checkpoint results/pipe_classifier/best.ckpt \
    --real_classes
```

---

## Configuration

### PipeFittingConfig

```python
from brepclassifier.configs.config import PipeFittingConfig

config = PipeFittingConfig(
    # Encoder (same as BrepFormer)
    hidden_dim=256, num_layers=8, num_heads=32,

    # Classification
    num_classes=8, multi_label=False,

    # GAT head
    gat_num_layers=3, gat_heads=4, gat_hidden_dim=256,
    gat_v2=True, gat_dropout=0.3, gat_pooling="global_attention",
    dense_dims=[512, 256], dense_dropout=0.3,

    # Training
    learning_rate=1e-4, encoder_lr_factor=0.1,
    batch_size=16, warmup_steps=500, max_epochs=300,

    # Weights
    pretrained_encoder_ckpt="results/brepformer/best.ckpt",
    freeze_encoder=False,
    class_weights=[1.0, 0.8, 1.5, 1.2, 10.0, 5.0, 1.0, 2.0],
)
```

---

## Pretrained Weights

### Available Checkpoints

| Checkpoint | Path | Use |
|---|---|---|
| Main brepformer | `results/brepformer/best-epoch=99-val/f1=0.8466.ckpt` | Encoder pretraining (27-class model-only) |
| Face seg heavy | `results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt` | Encoder pretraining + per-face inference |
| Pipe classifier | `results/pipe_classifier/best-epoch=64-val/f1=0.5517.ckpt` | 8-class pipe fitting inference |

### Using BrepFormer Encoder Weights

The BrepFormer encoder was trained on the MTFRCAD dataset (27 machining feature classes). Two checkpoints are available:

```bash
# Main checkpoint (model-only classification)
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder "results/brepformer/best-epoch=99-val/f1=0.8466.ckpt"

# Face seg checkpoint (includes face segmentation head — encoder weights still reused)
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder "results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt"
```

Both checkpoints work with `_load_pretrained_encoder` which uses `strict=False` to load only matching encoder keys. The GAT head is initialized randomly.

---

## Usage Examples

### Complete Pipeline

```bash
# 1. Convert STEP files
python brepclassifier/convert_steps.py --data_dir brepclassifier/data/ssdata1

# 2. Preprocess
python brepclassifier/preprocess.py \
    --data_dir brepclassifier/data/ssdata1 \
    --output_dir brepclassifier/data/ssdata1_processed

# 3. Train with pretrained encoder
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder results/brepformer/best-epoch=97-val/f1=0.8482.ckpt \
    --max_epochs 300

# 4. Test
python brepclassifier/test.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --checkpoint results/pipe_classifier/best.ckpt

# 5. Analyze
python brepclassifier/analyze.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --data_dir brepclassifier/data/ssdata1_processed \
    --mode all

# 6. Visualize
python brepclassifier/visualize.py --mode all \
    --input_dir brepclassifier/analysis_results/pipe_classifier

# 7. Interactive viewer
python brepclassifier/viewer.py --mode browse --split test
```

### Frozen Encoder (Train Only GAT Head)

```bash
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder results/brepformer/best.ckpt \
    --freeze_encoder \
    --learning_rate 5e-4 \
    --max_epochs 100
```
