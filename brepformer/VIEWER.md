# BrepMFR Interactive Viewer

Interactive 3D point cloud viewer for browsing MFTRCAD B-rep models with ground-truth labels, model predictions, and analysis visualizations.

## Overview

The viewer loads preprocessed MFTRCAD data (pickle files from `preprocess.py`), reconstructs point clouds from face UV-grids, and displays them with per-face class coloring. It integrates with analysis results from `analyze.py` to show model predictions, per-class metrics, confusion matrices, and t-SNE embeddings.

**What it visualizes:**
- Point clouds built from `face_grid` (N, 7, 10, 10): xyz channels 0-2, normals 3-5, mask channel 6
- ~100 points per face, ~3000 per model
- 27 MFTRCAD machining feature classes with a fixed color palette

---

## Modes

### Browse Mode

Browse preprocessed data split with GT labels and prediction overlays.

```bash
python viewer.py --mode browse --split test --sort index
python viewer.py --mode browse --split train
python viewer.py --mode browse --split val --sort worst
python viewer.py --mode browse --model_id 20240125_003844_7317
```

### Predictions Mode

Browse models sorted by prediction accuracy (Jaccard similarity). Automatically detects which split matches the predictions.

```bash
python viewer.py --mode predictions --sort worst
python viewer.py --mode predictions --sort best
python viewer.py --mode predictions --sort random
```

### Analysis Mode

Show analysis plots without launching the 3D viewer. Only requires matplotlib (no Open3D display).

```bash
python viewer.py --mode analysis --metrics
python viewer.py --mode analysis --confusion
python viewer.py --mode analysis --embeddings
python viewer.py --mode analysis  # shows all available
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| T / TAB | Cycle views: Plain -> GT Labels -> Predicted -> Comparison |
| D / RIGHT | Next model |
| A / LEFT | Previous model |
| 1 | Sort: worst accuracy first |
| 2 | Sort: best accuracy first |
| 3 | Sort: random shuffle |
| M | Per-class metrics bar chart popup |
| N | Confusion matrix heatmap popup |
| E | t-SNE embeddings scatter popup |
| I | Print detailed model info to console |
| L | Toggle class legend (prints to console) |
| S | Save screenshot to current directory |
| R | Reset camera view |
| F | Defeature current model (requires `--defeature` flag) |
| H | Print help to console |
| ESC / Q | Exit viewer |

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `browse` | Viewer mode: `browse`, `predictions`, `analysis` |
| `--split` | `test` | Data split: `train`, `val`, `test` |
| `--sort` | `index` | Sort order: `index`, `best`, `worst`, `random` |
| `--model_id` | None | View a specific model by ID |
| `--data_dir` | `brepformer/data/mftrcad` | Raw MFTRCAD data directory (for per-face GT labels) |
| `--processed_dir` | `brepformer/data/mftrcad_processed` | Preprocessed pickle data directory |
| `--analysis_dir` | `analysis_results` | Analysis results directory |
| `--metrics` | False | Show per-class metrics chart (analysis mode) |
| `--confusion` | False | Show confusion matrix heatmap (analysis mode) |
| `--embeddings` | False | Show t-SNE embeddings scatter (analysis mode) |

---

## Visualization Details

### Point Cloud Construction

Each B-rep model's faces are stored as UV-grids with shape `(N, 7, 10, 10)`:
- Channels 0-2: XYZ coordinates
- Channels 3-5: Surface normals
- Channel 6: Validity mask (>0.5 = valid point)

The viewer extracts valid points from each face's 10x10 grid, yielding ~100 points per face. With typical models having ~30 faces, each model produces ~3000 points displayed with `point_size = 5.0`.

### View Modes

1. **Plain**: Uniform gray point cloud showing model geometry
2. **GT Labels**: Each face colored by its per-face ground-truth class from `labels/{model_id}_result.json`
3. **Predicted**: Faces colored by model-level predicted classes (cycling through predicted class colors)
4. **Comparison**: GT (left) and Predicted (right) side-by-side with spatial offset

### Color Palette

Fixed 27-class palette using matplotlib `tab20` (classes 0-19) + `tab20b` (classes 20-26). Press L to see the full legend with RGB values.

### Real Classes Mode

The `visualize_seg.py` viewer supports `--real_classes` to remap the 27 MFTRCAD classes into 8 higher-level categories (other_surfaces, through_hole, blind_hole, chamfer, fillet, through_cut, blind_cut, extrude). When enabled, the legend, face colors, summary, and status bar all use the 8-class palette. See [SCRIPTS.md](SCRIPTS.md) for the full mapping table.

### Per-Face vs Model-Level Labels

- **GT labels** are per-face: each face has its own class from the label JSON files
- **Predicted labels** are model-level: the classifier predicts which classes exist in the entire model (multi-label), so all faces cycle through the predicted class colors

---

## Analysis Scripts Reference

### Generating Required Data

Run `analyze.py` to generate the data files the viewer reads:

```bash
# Generate all analysis outputs
python brepformer/analyze.py \
    --checkpoint results/brepformer/best-epoch=97-val/f1=0.8482.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode all \
    --output_dir analysis_results

# Or run specific analyses:
python brepformer/analyze.py --checkpoint CKPT --data_dir DATA --mode per_class
python brepformer/analyze.py --checkpoint CKPT --data_dir DATA --mode predictions
python brepformer/analyze.py --checkpoint CKPT --data_dir DATA --mode embeddings
python brepformer/analyze.py --checkpoint CKPT --data_dir DATA --mode confusion_matrix
```

### Output File Reference

| File | Description |
|------|-------------|
| `predictions.json` | Per-sample predictions with model_id, predicted_classes, target_classes, jaccard_similarity, class_probabilities, num_pred, num_target |
| `per_class_metrics.json` | Per-class precision, recall, F1, support |
| `confusion_matrix_multilabel.npy` | (27, 27) co-occurrence matrix: entry (i,j) = samples where class i is GT and class j is predicted |
| `per_class_confusion.npy` | (27, 2, 2) per-class confusion with [[TN, FP], [FN, TP]] |
| `confusion_matrix.npy` | (27, 27) standard confusion matrix (single-label mode) |
| `embeddings.npy` | Raw graph embeddings |
| `embeddings_pca50.npy` | PCA-reduced embeddings (50D) |
| `embeddings_tsne.npy` | t-SNE embeddings (2D) |
| `labels.npy` | Labels corresponding to embeddings |
| `architecture.json` | Model architecture details |

### predictions.json Format

```json
{
  "summary": {
    "total_samples": 2856,
    "correct": 1234,
    "accuracy": 0.4321
  },
  "predictions": [
    {
      "model_id": "20240125_003844_7317",
      "predicted_classes": [0, 9, 24],
      "target_classes": [0, 8, 9, 24],
      "num_pred": 3,
      "num_target": 4,
      "class_probabilities": {"0": 0.95, "1": 0.02, ...},
      "jaccard_similarity": 0.667,
      "correct": false
    }
  ]
}
```

---

## Pipeline

Complete workflow from raw data to interactive visualization:

```bash
# 1. Preprocess raw MFTRCAD data
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed

# 2. Train model
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 100

# 3. Test model
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt

# 4. Run analysis (generates viewer data)
python brepformer/analyze.py \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode all \
    --output_dir analysis_results

# 5. Launch viewer
./run_viewer.sh browse --split test
./run_viewer.sh predictions --sort worst
./run_viewer.sh analysis --metrics
```

---

## Troubleshooting

### WSL2 Display Issues

The viewer requires an X11 display server. WSLg (Windows 11) provides this automatically. For older Windows:

1. Install VcXsrv or X410 on Windows
2. Set `DISPLAY` in WSL: `export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0`
3. Run via `./run_viewer.sh` which handles display setup automatically

The `run_viewer.sh` script forces X11 mode (disables Wayland) and configures Mesa software rendering for compatibility.

### Analysis mode without display

If you cannot set up a display server, use analysis mode which only needs matplotlib:

```bash
python viewer.py --mode analysis --metrics --confusion --embeddings
```

### Missing Dependencies

Required packages:
- `open3d` - 3D visualization (not needed for analysis mode)
- `matplotlib` - Analysis plots and color palette
- `numpy` - Numerical operations

Install: `pip install open3d matplotlib numpy`

### Missing Analysis Data

If the viewer reports missing predictions or metrics, run `analyze.py` first:

```bash
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode all --output_dir analysis_results
```

### Per-Face GT Labels Not Showing

GT label coloring requires the raw MFTRCAD labels directory. Make sure `--data_dir` points to the directory containing `labels/` with `{model_id}_result.json` files. Default: `brepformer/data/mftrcad`.

### STEP Rendering

The viewer displays point clouds from preprocessed UV-grids, not STEP geometry. For STEP rendering, use pythonocc or FreeCAD separately. The STEP files are in `brepformer/data/mftrcad/steps/`.

---

## Defeature Dataset (5 classes)

The viewer and `visualize_seg.py` work with the defeature dataset (5 classes: random, hole, chamfer, fillet, cut) using the same scripts. Since the defeature model is trained with 5 classes natively, no `--real_classes` remapping is needed.

### Defeature Color Palette

| Class | Name | Color |
|-------|------|-------|
| 0 | random | gray (#7f7f7f) |
| 1 | hole | blue (#1f77b4) |
| 2 | chamfer | orange (#ff7f0e) |
| 3 | fillet | green (#2ca02c) |
| 4 | cut | red (#d62728) |

### Defeature Viewer Commands

```bash
# Browse defeature STEP files with GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/

# Browse with predictions from inference
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir inference/defeature/

# Single model with live inference
python -m brepformer.visualize_seg \
    --step brepformer/data/defeature/steps/some_model.step \
    --checkpoint results/defeature/best-*.ckpt
```

### Limited Data Subset Viewer

When training with `--limit_data`, the manifest records which model_ids were used. The viewer scripts work normally since they operate on STEP files and inference results; just ensure your inference was run with `--limit_data_manifest` so the results match the training subset.

```bash
# 1. Train on 1000-sample subset
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation --num_face_classes 5 \
    --output_dir results --exp_name defeature_1k \
    --limit_data 1000

# 2. Infer on the same subset's STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint results/defeature_1k/best-*.ckpt \
    --output_dir results/defeature_1k/inference/ \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

# 3. Browse the subset's predictions vs GT
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir results/defeature_1k/inference/
```

### Defeature Analysis Viewer

```bash
# Generate analysis data for the viewer
python brepformer/analyze.py \
    --checkpoint results/defeature/best-*.ckpt \
    --data_dir brepformer/data/defeature_processed \
    --mode all \
    --output_dir analysis_results/defeature

# View analysis (5-class confusion matrix, metrics)
python viewer.py --mode analysis \
    --processed_dir brepformer/data/defeature_processed \
    --analysis_dir analysis_results/defeature \
    --metrics --confusion
```

### Interactive Defeaturing (F key)

When launched with `--defeature`, pressing **F** runs `defeature.py` on the currently displayed model. The defeatured STEP file is saved to `brepformer/defeatured_output/` (or `--defeature_output_dir`).

```bash
# Browse defeature models with live defeaturing enabled
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --defeature

# Single model — view predictions, press F to defeature
python -m brepformer.visualize_seg \
    --step brepformer/data/defeature/steps/some_model.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --defeature
```

For batch (non-interactive) defeaturing, use `defeature.py` directly. See [SCRIPTS.md](SCRIPTS.md) for full CLI reference.

### Viewing Inference Pipeline Output

After running `inference_pipeline.sh`, all output is saved to `brepformer/pipeline_output/<filename>/`. You can re-open the viewers on this output at any time:

```bash
# Re-open segmentation viewer on pipeline output
python -m brepformer.visualize_seg \
    --step /path/to/original.step \
    --seg brepformer/pipeline_output/<filename>/preds.seg

# Re-open defeature comparison viewer (requires new_brepmfr env)
conda activate new_brepmfr
python -m brepformer.visualize_defeature \
    --step /path/to/original.step \
    --defeatured brepformer/pipeline_output/<filename>/<filename>_defeatured.step \
    --seg brepformer/pipeline_output/<filename>/preds.seg

# Open colored STEP in FreeCAD (no Python needed)
# Just open: brepformer/pipeline_output/<filename>/<filename>_colored.step
```

**Pipeline output folder layout:**
```
brepformer/pipeline_output/<filename>/
├── preds.json                    # Full predictions (loadable in Python)
├── preds.seg                     # One label per line (for viewer --seg)
├── <filename>_colored.step       # Open in FreeCAD for colored faces
├── analysis/                     # Analysis JSON/npy files
├── <filename>_defeatured.step    # Open in FreeCAD for clean stock shape
└── <filename>_report.json        # Defeaturing stats (removed/failed/valid)
```

The colored STEP file can be opened directly in FreeCAD — each face displays its predicted class color with no plugins required. Compare the colored and defeatured files side by side in FreeCAD for a quick visual check.

See [SCRIPTS.md](SCRIPTS.md) for the full `inference_pipeline.sh` CLI reference.

### Defeature Comparison Viewer

`visualize_defeature.py` is a dedicated side-by-side comparison viewer for original vs defeatured STEP models. While `visualize_seg.py` focuses on face segmentation labels with an optional defeature action (F key), `visualize_defeature.py` shows both models simultaneously for visual quality assessment.

```bash
# Browse defeatured results with GT labels on the original
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --labels_dir brepformer/data/defeature/labels/

# Single model comparison
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/some_model.step \
    --defeatured brepformer/defeatured_output/some_model_defeatured.step

# With predicted labels coloring the original
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --seg_dir inference/trial1_ss1500/

# With defeaturing reports (shows removed/failed/valid stats)
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --report_dir brepformer/defeatured_output/
```

**Display layout:**
- **Left panel**: Original STEP file colored by per-face class predictions (5-class defeature palette)
- **Right panel**: Defeatured STEP file shown in uniform gray (features removed)

**Keyboard controls** are the same as `visualize_seg.py` — D/RIGHT for next, A/LEFT for prev, T to toggle GT/Pred on the original, S for summary with defeaturing stats, I for info, Q to quit.

When `--report_dir` is provided, the status panel shows defeaturing statistics from the JSON report: faces removed, faces failed, shape validity.

See [SCRIPTS.md](SCRIPTS.md) for the full CLI reference and argument table.
