# brepclassifier — Script Reference

Pipe fitting classification pipeline using BrepFormer encoder + GAT head (8 classes).

---

## Overview

`brepclassifier` trains a whole-model classifier on STEP B-rep pipe fittings. It extracts
face-level graph features with the pretrained BrepFormer encoder, aggregates them with a
Graph Attention Network (GAT), and predicts one of 8 pipe fitting classes per model.

**8 classes:**

| ID | Name |
|----|------|
| 0 | Elbow - Weld Fitting |
| 1 | Elbow - Pipe End Fitting |
| 2 | Elbow - Socket Fitting |
| 3 | Tee - Weld Fitting |
| 4 | Tee - Pipe End Fitting |
| 5 | Tee - Socket Fitting |
| 6 | Elbow - Miscellaneous |
| 7 | Tee - Miscellaneous |

---

## Quick Start

```bash
# Convert STEP files to JSON geometry
python brepclassifier/convert_steps.py --data_dir brepclassifier/data/ssdata1

# Preprocess graph features + split
python brepclassifier/preprocess.py \
    --data_dir brepclassifier/data/ssdata1 \
    --output_dir brepclassifier/data/ssdata1_processed

# Train (with pretrained BrepFormer encoder)
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder results/brepformer/best.ckpt

# Evaluate
python brepclassifier/test.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --checkpoint results/pipe_classifier/best.ckpt

# Analyze
python brepclassifier/analyze.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --data_dir brepclassifier/data/ssdata1_processed \
    --mode all --output_dir analysis_results/pipe_classifier

# Open3D viewer (point cloud)
./brepclassifier/run_viewer.sh browse --split test
./brepclassifier/run_viewer.sh predictions --sort worst

# STEP geometry viewer (Qt + pythonOCC)
./brepclassifier/run_viewer.sh seg \
    --step_dir brepclassifier/data/ssdata1/steps/ \
    --labels_json brepclassifier/data/ssdata1/labels.json

# Per-face segmentation viewer
./brepclassifier/run_viewer.sh faceseg \
    --step_dir brepclassifier/data/ssdata1/steps/ \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt

# Export colored STEP for FreeCAD
./brepclassifier/run_viewer.sh freecad \
    --step brepclassifier/data/ssdata1/steps/model.step \
    --pipe_checkpoint results/pipe_classifier/best-epoch=64-val/f1=0.5517.ckpt \
    --output colored.step
```

---

## Pipeline

```
STEP files (raw)
    └── convert_steps.py        → JSON geometry per model
         └── preprocess.py      → pickle graph features + train/val/test split
              └── train.py      → train PipeFittingClassifier checkpoint
                   ├── test.py  → evaluate on test split
                   └── analyze.py → per-class metrics / embeddings / predictions
                        └── visualize.py  → publication plots
                             ├── viewer.py              → Open3D point cloud viewer
                             ├── visualize_seg.py       → Qt + pythonOCC STEP viewer (whole-model)
                             ├── visualize_face_seg.py  → Qt + pythonOCC per-face seg viewer
                             └── export_freecad.py      → colored STEP export for FreeCAD
```

---

## Script Reference

### `convert_steps.py` — STEP → JSON conversion

Converts raw STEP files to JSON geometry (face UV-grids, edge curves, topology).

```bash
python brepclassifier/convert_steps.py \
    --data_dir brepclassifier/data/ssdata1 \
    [--num_workers 8]
```

**Input:** `data_dir/` with class subdirectories containing STEP files
**Output:** JSON files alongside each STEP, plus `steps/` flat directory

---

### `preprocess.py` — Graph features + stratified split

Extracts BrepFormer-compatible graph features and creates stratified train/val/test splits.

```bash
python brepclassifier/preprocess.py \
    --data_dir brepclassifier/data/ssdata1 \
    --output_dir brepclassifier/data/ssdata1_processed \
    [--val_ratio 0.1] [--test_ratio 0.2] [--seed 42] [--num_workers 8]
```

**Input:** `data_dir/` with JSON files (from `convert_steps.py`)
**Output:** `output_dir/{train,val,test}/` with `.pkl` per model + `split_info.json`

---

### `train.py` — Training

Trains the PipeFittingClassifier (BrepFormer encoder + GAT head).

```bash
# Basic
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed

# With pretrained encoder
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder results/brepformer/best.ckpt \
    --max_epochs 300

# Frozen encoder (GAT head only)
python brepclassifier/train.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --pretrained_encoder results/brepformer/best.ckpt \
    --freeze_encoder --max_epochs 300
```

Key arguments: `--data_dir`, `--pretrained_encoder`, `--freeze_encoder`, `--max_epochs`,
`--lr`, `--batch_size`, `--output_dir`

---

### `test.py` — Evaluation

Evaluates a trained checkpoint on the test split and reports per-class metrics.

```bash
python brepclassifier/test.py \
    --data_dir brepclassifier/data/ssdata1_processed \
    --checkpoint results/pipe_classifier/best.ckpt \
    [--output_file results/test_results.json]
```

Outputs: overall accuracy/F1 + per-class precision/recall/F1/support table.

---

### `analyze.py` — 6 analysis modes

```bash
python brepclassifier/analyze.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --data_dir brepclassifier/data/ssdata1_processed \
    --mode all \
    --output_dir analysis_results/pipe_classifier
```

| Mode | Description |
|------|-------------|
| `architecture` | Model summary, parameter counts, module breakdown |
| `per_class` | Per-class precision/recall/F1/support |
| `confusion_matrix` | 8×8 confusion matrix → `.npy` |
| `predictions` | Per-sample predictions → `predictions.json` |
| `embeddings` | Encoder embeddings → PCA → t-SNE → `.npy` files |
| `all` | Run all modes |

`--data_dir` is optional for `architecture` mode only.

---

### `visualize.py` — Publication-quality plots

Generates matplotlib figures from analysis output.

```bash
python brepclassifier/visualize.py \
    --analysis_dir brepclassifer/analysis_results/pipe_classifier \
    --output_dir brepclassifer/plots/pipe_classifier \
    [--mode all]
```

---

### `export_model.py` — Model export

```bash
# State dict only
python brepclassifier/export_model.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --output_dir exported/ --format state_dict

# TorchScript
python brepclassifier/export_model.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --output_dir exported/ --format torchscript

# ONNX
python brepclassifier/export_model.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --output_dir exported/ --format onnx
```

---

### `viewer.py` — Open3D point cloud viewer

Visualizes preprocessed models as point clouds with class coloring.

```bash
# Browse test split
python brepclassifier/viewer.py --mode browse --split test

# Browse by sort order
python brepclassifier/viewer.py --mode predictions --sort worst
python brepclassifier/viewer.py --mode predictions --sort best

# Analysis plots (no 3D viewer)
python brepclassifier/viewer.py --mode analysis --metrics --confusion --embeddings
```

Or via the shell wrapper:
```bash
./brepclassifier/run_viewer.sh browse --split test
./brepclassifier/run_viewer.sh predictions --sort worst
./brepclassifier/run_viewer.sh analysis --metrics
```

**Requires:** `open3d`, `matplotlib`

---

### `visualize_seg.py` — Qt + pythonOCC STEP geometry viewer *(new)*

Displays raw STEP geometry with whole-model class coloring. All faces receive the same color
based on the model-level classification (GT or predicted class).

```bash
# Single file, specify GT class directly
python brepclassifier/visualize_seg.py \
    --step brepclassifier/data/ssdata1/steps/model.step \
    --gt_class 0

# Batch mode with labels JSON
python brepclassifier/visualize_seg.py \
    --step_dir brepclassifier/data/ssdata1/steps/ \
    --labels_json brepclassifier/data/ssdata1/labels.json

# With live inference
python brepclassifier/visualize_seg.py \
    --step_dir brepclassifier/data/ssdata1/steps/ \
    --labels_json brepclassifier/data/ssdata1/labels.json \
    --checkpoint results/pipe_classifier/best.ckpt
```

Or via the shell wrapper:
```bash
./brepclassifier/run_viewer.sh seg \
    --step_dir brepclassifier/data/ssdata1/steps/ \
    --labels_json brepclassifier/data/ssdata1/labels.json
```

**CLI arguments:**

| Argument | Description |
|----------|-------------|
| `--step PATH` | Single STEP file to display |
| `--gt_class INT` | Ground-truth class (0-7) for the given file |
| `--labels_json PATH` | `labels.json` with `model_id → class_idx` mapping |
| `--checkpoint PATH` | Checkpoint — enables "Run Inference" button |
| `--step_dir PATH` | Directory of STEP files for batch browsing |

**Display modes:**

| Mode | Description |
|------|-------------|
| GT | All faces colored by GT class |
| PRED | All faces colored by predicted class |
| COMPARE | GT color; status bar shows GT vs PRED with ✓/✗ |

**Requires:** `PyQt5` (or `PySide2`), `pythonocc-core`

---

### `run_viewer.sh` — Shell wrapper *(new)*

Sets WSL2-compatible display environment variables and dispatches to the correct viewer.

```bash
./brepclassifier/run_viewer.sh <command> [args...]
```

| Command | Dispatches to |
|---------|--------------|
| `browse` | `viewer.py --mode browse` |
| `predictions` | `viewer.py --mode predictions` |
| `analysis` | `viewer.py --mode analysis` |
| `seg` | `visualize_seg.py` |
| `faceseg` | `visualize_face_seg.py` |
| `freecad` | `export_freecad.py` |

Sets `LIBGL_ALWAYS_SOFTWARE=1` and `MESA_GL_VERSION_OVERRIDE=3.3` for WSL2 software rendering.

---

### `visualize_face_seg.py` — Qt + pythonOCC per-face segmentation viewer

Displays raw STEP geometry with per-face class coloring from BrepFormer face segmentation
(27-class MFTRCAD or 8 real categories). Optionally shows pipe class in the status bar.

```bash
# With face seg checkpoint
python brepclassifier/visualize_face_seg.py \
    --step model.step \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt

# Browse directory with seg files
python brepclassifier/visualize_face_seg.py \
    --step_dir brepclassifier/data/ssdata1/steps/ \
    --seg_dir predictions/

# With 8 real class remapping and pipe class
python brepclassifier/visualize_face_seg.py \
    --step model.step \
    --checkpoint results/face_seg_heavy/best.ckpt \
    --pipe_checkpoint results/pipe_classifier/best.ckpt \
    --real_classes
```

Or via the shell wrapper:
```bash
./brepclassifier/run_viewer.sh faceseg \
    --step model.step \
    --checkpoint results/face_seg_heavy/best.ckpt
```

**CLI arguments:**

| Argument | Description |
|----------|-------------|
| `--step PATH` | Single STEP file to display |
| `--seg PATH` | Path to .seg label file (one label per line) |
| `--labels_json PATH` | Path to face labels JSON file |
| `--labels_dir PATH` | Directory of face label JSON files |
| `--checkpoint PATH` | BrepFormer face seg checkpoint for inference |
| `--step_dir PATH` | Directory of STEP files for batch browsing |
| `--seg_dir PATH` | Directory of .seg prediction files |
| `--real_classes` | Remap 27 classes to 8 real categories |
| `--pipe_checkpoint PATH` | PipeFittingClassifier checkpoint (shows pipe class in status) |

**Requires:** `PyQt5` (or `PySide2`), `pythonocc-core`

---

### `export_freecad.py` — Export colored STEP for FreeCAD

Exports STEP files with per-face or whole-model colors using XCAF, readable by FreeCAD.

```bash
# Pipe mode: all faces colored by pipe classifier prediction
python brepclassifier/export_freecad.py \
    --step model.step \
    --pipe_checkpoint results/pipe_classifier/best.ckpt \
    --output colored.step

# Face mode: per-face coloring
python brepclassifier/export_freecad.py \
    --step model.step \
    --face_checkpoint results/face_seg_heavy/best.ckpt \
    --output colored.step

# From labels.json
python brepclassifier/export_freecad.py \
    --step model.step \
    --pipe_labels_json brepclassifier/data/ssdata1/labels.json \
    --output colored.step

# Batch mode
python brepclassifier/export_freecad.py \
    --step_dir steps/ --face_checkpoint results/face_seg_heavy/best.ckpt \
    --output_dir colored/
```

Or via the shell wrapper:
```bash
./brepclassifier/run_viewer.sh freecad \
    --step model.step \
    --pipe_checkpoint results/pipe_classifier/best.ckpt \
    --output colored.step
```

**Requires:** `pythonocc-core`

---

## Keyboard Controls

### `viewer.py` (Open3D)

| Key | Action |
|-----|--------|
| T / TAB | Cycle views: Plain → GT Class → Predicted → Comparison |
| D / RIGHT | Next model |
| A / LEFT | Previous model |
| 1 | Sort: worst first |
| 2 | Sort: best first |
| 3 | Sort: random shuffle |
| M | Per-class metrics bar chart |
| N | Confusion matrix heatmap |
| E | t-SNE embeddings scatter |
| I | Print model info to console |
| L | Toggle class legend |
| S | Save screenshot |
| R | Reset camera |
| H | Print help |
| ESC / Q | Exit |

### `visualize_seg.py` (Qt + pythonOCC)

| Key | Action |
|-----|--------|
| T | Cycle display: GT → PRED → COMPARE → FACE_SEG (if available) |
| D / RIGHT | Next model (batch mode) |
| A / LEFT | Previous model (batch mode) |
| I | Print model info to console |
| S | Summary dialog |
| ESC / Q | Quit |

### `visualize_face_seg.py` (Qt + pythonOCC)

| Key | Action |
|-----|--------|
| T | Cycle display: GT → PRED → COMPARE |
| D / RIGHT | Next model (batch mode) |
| A / LEFT | Previous model (batch mode) |
| I | Print face info to console |
| S | Summary dialog |
| ESC / Q | Quit |

---

## Troubleshooting

### WSL2 display issues

Use the `run_viewer.sh` wrapper — it sets `LIBGL_ALWAYS_SOFTWARE=1` automatically.

For manual setup:
```bash
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
python brepclassifier/visualize_seg.py --help
```

WSLg (Windows 11) provides X11/Wayland automatically. On WSL2 without WSLg, set `DISPLAY`:
```bash
export DISPLAY=:0
```

### NumPy 2.0 / pytorch_lightning incompatibility

`pytorch_lightning==1.7.1` uses `np.Inf` which was removed in NumPy 2.0. All scripts in
`brepclassifier/` include the compatibility shim:
```python
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
```

### Missing pythonOCC or PyQt5

`visualize_seg.py` requires:
```bash
conda install -c conda-forge pythonocc-core pyqt
```

### Missing Open3D

`viewer.py` requires:
```bash
pip install open3d
```

### Missing analysis data

Run `analyze.py --mode all` before using `viewer.py` predictions or analysis modes:
```bash
python brepclassifier/analyze.py \
    --checkpoint results/pipe_classifier/best.ckpt \
    --data_dir brepclassifier/data/ssdata1_processed \
    --mode all --output_dir analysis_results/pipe_classifier
```

### labels.json format

Expected by `visualize_seg.py --labels_json`:
```json
{
  "ELL_ROUND_BASE_ASME": 0,
  "Victaulic_Cu_Tee_Redc": 7,
  "DT15-T90": 3
}
```

Keys are STEP file stems (filename without `.step`). Values are class indices 0–7.
