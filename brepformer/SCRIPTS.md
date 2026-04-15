# BrepFormer Scripts Reference

Complete reference for all scripts in the `brepformer/` directory with usage, CLI arguments, and examples. For an overview and quick start, see [README.md](README.md).

> Scripts marked ***recommended*** are the most capable variant when multiple scripts serve the same purpose.
> Examples marked ***best*** show the optimal flags for face segmentation quality.

---

## End-to-End Pipeline

### `run_pipeline.py` ***recommended — runs the full pipeline in one command***

Chains all stages — preprocess, train, test, analyze — with automatic checkpoint detection and sensible defaults. Use `--skip_*` flags to skip stages, or `--only` to run a single stage.

```bash
# *best* — full pipeline optimized for face segmentation quality
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/best_face_seg \
    --face_segmentation \
    --face_seg_weight 2.0 --model_cls_weight 0.5 \
    --compute_descriptors \
    --max_epochs 200 \
    --batch_size 32 --learning_rate 0.0016 --num_workers 4

# Full pipeline from raw data
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --max_epochs 200

# Full pipeline with face segmentation
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/face_run \
    --face_segmentation \
    --face_seg_weight 2.0 --model_cls_weight 0.5

# With D2/angle descriptors for best quality
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_desc \
    --face_segmentation \
    --compute_descriptors

# Skip preprocessing (data already prepared)
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --skip_preprocess

# Resume training from checkpoint
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --skip_preprocess \
    --resume_from results/full_run/train/last.ckpt

# Only test + analyze an existing checkpoint
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir results/full_run \
    --only test \
    --checkpoint results/full_run/train/best-epoch=50-val/f1=0.8800.ckpt

# Quick sanity check
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/sample \
    --output_dir /tmp/sanity \
    --fast_dev_run --num_workers 0 --batch_size 4 \
    --hidden_dim 64 --ffn_dim 128 --num_heads 8 --num_kv_heads 4 --num_layers 2
```

| Argument | Default | Description |
|----------|---------|-------------|
| **Stage control** | | |
| `--only` | None | Run only this stage: "preprocess", "train", "test", "analyze" |
| `--skip_preprocess` | False | Skip preprocessing stage |
| `--skip_train` | False | Skip training stage |
| `--skip_test` | False | Skip testing stage |
| `--skip_analyze` | False | Skip analysis stage |
| **Paths** | | |
| `--data_dir` | required | Raw data directory (graphs/ + labels/) or preprocessed dir |
| `--output_dir` | required | Root output directory |
| `--checkpoint` | auto | Checkpoint path (auto-detected from best training result) |
| `--resume_from` | None | Resume training from this checkpoint |
| **Preprocessing** | | |
| `--compute_descriptors` | False | Compute D2/angle multi-sample histograms (slower but improves quality) |
| `--split_ratio` | "0.8,0.1,0.1" | Train/val/test split ratios |
| `--seed` | 42 | Random seed |
| **Architecture** | | |
| `--hidden_dim` | 256 | Hidden dimension |
| `--ffn_dim` | 512 | FFN dimension |
| `--num_layers` | 8 | Transformer layers |
| `--num_heads` | 32 | Attention heads |
| `--num_kv_heads` | 8 | KV heads (GQA) |
| `--dropout` | 0.3 | Dropout probability |
| **Face segmentation** | | |
| `--face_segmentation` | False | Enable face-level segmentation head with automatic class weighting |
| `--face_seg_weight` | 1.0 | Loss weight for face segmentation |
| `--model_cls_weight` | 1.0 | Loss weight for model classification |
| **Training** | | |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 0.002 | Learning rate |
| `--max_epochs` | 200 | Maximum epochs |
| `--num_workers` | 4 | Data loader workers |
| `--devices` | 1 | Number of GPUs |
| `--precision` | 32 | Training precision (16 or 32) |
| `--fast_dev_run` | False | Quick 1-batch sanity check |
| `--limit_data` | None | Limit total dataset to N samples; manifest auto-propagated to test/analyze |
| **Test/Analysis** | | |
| `--real_classes` | False | Remap 27 classes to 8 real machining feature categories |
| `--analyze_modes` | "all" | Comma-separated analysis modes |

**Output structure:**
```
output_dir/
├── preprocessed/        # train.pkl, val.pkl, test.pkl, metadata.json
├── train/               # checkpoints (best-*.ckpt, last.ckpt)
├── test_results.json    # model-level metrics on test split (Stage 3)
├── train_results.json   # same metrics evaluated on train split (Stage 3b)
├── face_preds.json      # per-face predictions on test split (if --face_segmentation)
├── train_face_preds.json# per-face predictions on train split (if --face_segmentation)
└── analysis/            # per_class_metrics.json, embeddings, confusion matrices
```

> **Stage 3b** runs automatically alongside Stage 3 whenever the test stage is active. It evaluates the checkpoint on the **training split** using the same metrics as the test evaluation, producing `train_results.json` (and `train_face_preds.json`). Comparing these files against `test_results.json` is the primary way to detect overfitting.

#### Defeature Dataset (5 classes)

```bash
# Full pipeline for the navin_defeaturing dataset (5 classes)
# Note: run_pipeline.py does not pass --num_classes to preprocess.py,
# so run the individual stages instead (see each section below).
```

### `inference_pipeline.sh` ***recommended — single STEP file end-to-end pipeline***

Takes any STEP file and runs the full pipeline: inference → FreeCAD export → analysis → segmentation viewer → defeature v2 → defeature viewer. All output is saved to `brepformer/pipeline_output/<filename>/`. Automatically switches conda environments from `brepmfr` (steps 1-4) to `new_brepmfr` (steps 5-6) for defeature v2 compatibility (pythonocc >= 7.9).

```bash
# *best* — full pipeline with analysis data
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step \
    --data_dir brepformer/data/defeature_processed

# Basic: just run on a STEP file (architecture analysis only)
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step

# Headless (no interactive viewers — good for batch/CI)
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step --skip_viewers

# Custom checkpoint
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step \
    --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt

# Custom output directory
./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step \
    --output_dir /tmp/my_results/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `<step_file>` | required | Path to a STEP (.step/.stp) file |
| `--checkpoint` | `results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt` | Model checkpoint |
| `--data_dir` | None | Preprocessed data directory for full analysis (`--mode all`) |
| `--skip_viewers` | false | Skip interactive viewer steps (4 and 6) |
| `--output_dir` | `brepformer/pipeline_output/<filename>/` | Override output directory |

**Pipeline steps:**

| Step | Module | Conda Env | Description |
|------|--------|-----------|-------------|
| 1 | `brepformer.infer` | brepmfr | Predict per-face labels → `preds.json`, `preds.seg` |
| 2 | `brepformer.export_freecad` | brepmfr | Export colored STEP → `<name>_colored.step` |
| 3 | `brepformer.analyze` | brepmfr | Model analysis → `analysis/` |
| 4 | `brepformer.visualize_seg` | brepmfr | Interactive 3D segmentation viewer |
| 5 | `brepformer.defeature_v2` | new_brepmfr | Remove features → `<name>_defeatured.step`, `<name>_report.json` |
| 6 | `brepformer.visualize_defeature` | new_brepmfr | Original vs defeatured comparison viewer |

**Output structure:**
```
brepformer/pipeline_output/<filename>/
├── preds.json                    # Full predictions (per-face probs, class names)
├── preds.seg                     # Per-face labels (one label per line)
├── <filename>_colored.step       # Colored STEP for FreeCAD
├── analysis/                     # Analysis outputs
│   ├── architecture.json         # Model architecture (always present)
│   ├── per_class_metrics.json    # Per-class P/R/F1 (if --data_dir)
│   ├── embeddings.npy            # Embeddings (if --data_dir)
│   ├── face_seg_metrics.json     # Face segmentation metrics (if --data_dir)
│   └── ...
├── <filename>_defeatured.step    # Defeatured STEP (features removed)
├── <filename>_colored.step       # Colored predictions overlay
└── <filename>_report.json        # Defeaturing report (removed/failed/valid)
```

**Environment notes:**
- The script expects `brepmfr` to be the active conda environment when invoked.
- Steps 5-6 auto-switch to `new_brepmfr` (requires pythonocc >= 7.9 for `defeature_v2`).
- If `--data_dir` is not provided, step 3 runs `--mode architecture` only (no dataset needed).
- If `--data_dir` is provided, step 3 runs `--mode all` (per_class, embeddings, confusion_matrix, face_segmentation, etc.).

---

## Data Preprocessing

### `copy data command`
cp -r /mnt/c/projects/data/mftrcad /home/adminho/BrepMFR/brepformer/data/mftrcad

#### Defeature Dataset — Data Preparation

The defeature dataset uses STEP files with per-face text labels (one class per line). `prepare_defeature.py` copies STEP files, remaps the 7-class labels to 5 classes, and writes JSON labels in the format expected by `preprocess.py`. Then `step_to_graph.py` converts STEP geometry to graph JSONs.

| Original Class | Remapped Class | Name |
|----------------|----------------|------|
| 0 | 0 | Random (Other) |
| 1 | 1 | Hole |
| 2 | 1 | Hole |
| 3 | 2 | Chamfer |
| 4 | 3 | Fillet |
| 5 | 4 | Cut |
| 6 | 4 | Cut |

```bash
# Step 1: Copy STEP files and convert text labels to JSON (fast, no pythonocc needed)
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature

# Step 2: Convert STEP files to graph JSONs (requires pythonocc, slow — ~1-5 sec/model)
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature \
    --convert_graphs

# Or run just the graph conversion on already-copied data:
python -c "
import sys, json
sys.path.insert(0, '.')
from brepformer.data.step_to_graph import step_to_graph
from pathlib import Path
from tqdm import tqdm

steps_dir = Path('brepformer/data/defeature/steps')
out_dir = Path('brepformer/data/defeature/graphs')
out_dir.mkdir(parents=True, exist_ok=True)

for step_file in tqdm(sorted(steps_dir.glob('*.step'))):
    model_id = step_file.stem
    if (out_dir / f'{model_id}.json').exists():
        continue
    try:
        data = step_to_graph(str(step_file))
        if data is None:
            continue
        out = [model_id, {
            'graph': {'edges': data['edge_index'].tolist(), 'num_nodes': data['num_nodes']},
            'graph_face_attr': data['face_attr'].tolist(),
            'graph_face_grid': data['face_grid'].tolist(),
            'graph_edge_attr': data['edge_attr'].tolist(),
            'graph_edge_grid': data['edge_grid'].tolist(),
        }]
        with open(out_dir / f'{model_id}.json', 'w') as f:
            json.dump(out, f)
    except Exception as e:
        print(f'  Failed: {model_id}: {e}')
"
```

**Output structure:**
```
brepformer/data/defeature/
├── steps/          # 1561 sanitized STEP files
├── labels/         # 1561 JSON label files (remapped 5-class)
├── graphs/         # graph JSONs from step_to_graph (after conversion)
└── dataset_info.json  # metadata: class distribution, remap table
```

### `preprocess.py` ***recommended***

Converts raw JSON graph data to pickle format for efficient multi-worker loading. Reads graph files from `graphs/` and per-face labels from `labels/`, computes shortest paths and in-degree features, and writes train/val/test splits to pickle files.

The output includes `face_labels` (per-face class IDs, 0-26 or -1 for missing) alongside the multi-hot model labels, enabling face segmentation training without re-preprocessing.

```bash
# *best* — with multi-sample descriptors for highest model quality
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# Basic usage
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed

# Full configuration
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_processed \
    --num_classes 27 \
    --split_ratio "0.8,0.1,0.1" \
    --seed 42

# With D2/angle descriptors (slower, needed for descriptor-aware models)
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_desc \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# With external single-label file
python brepformer/preprocess.py \
    --data_dir brepformer/data/mftrcad \
    --output_dir brepformer/data/mftrcad_custom \
    --label_file path/to/my_labels.json \
    --num_classes 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Input directory with `graphs/` and `labels/` subdirectories |
| `--output_dir` | required | Output directory for pickle files |
| `--label_file` | None | External labels JSON for single-label mode (optional) |
| `--num_classes` | 27 | Number of classes |
| `--compute_descriptors` | False | Compute D2/angle descriptors using multi-sample histograms from UV-grids (slower preprocessing) |
| `--num_spatial` | 64 | Max shortest-path distance for spatial position |
| `--d2_bins` | 64 | D2 descriptor histogram bins |
| `--angle_bins` | 64 | Angle descriptor histogram bins |
| `--split_ratio` | "0.8,0.1,0.1" | Train/val/test split ratios |
| `--seed` | 42 | Random seed for split reproducibility |

**Output:**
- `train.pkl`, `val.pkl`, `test.pkl` — serialized sample lists
- `metadata.json` — dataset statistics (num_samples, num_classes, split sizes)

#### Defeature Dataset — Preprocessing

```bash
# *best* — preprocess defeature data with descriptors (5 classes)
python brepformer/preprocess.py \
    --data_dir brepformer/data/defeature \
    --output_dir brepformer/data/defeature_processed \
    --num_classes 5 \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# Basic preprocessing (faster, no descriptors)
python brepformer/preprocess.py \
    --data_dir brepformer/data/defeature \
    --output_dir brepformer/data/defeature_processed \
    --num_classes 5
```

---

## Training

### `train_preprocessed.py` ***recommended***

Training with preprocessed pickle data. Supports multi-worker data loading, optional face segmentation head (with automatic inverse-frequency class weighting), mixed precision, gradient accumulation, and checkpoint resumption.

```bash
# *best* — face segmentation with tuned weights and class balancing
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --face_segmentation \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --face_seg_hidden_dim 512 \
    --max_faces 500
    --max_epochs 300 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --num_workers 4 \
    --precision "32" \
    --output_dir results \
    --exp_name trial3

# Basic training (model-level classification only)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 100

# Full configuration
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0016 \
    --num_workers 4 \
    --hidden_dim 256 \
    --num_layers 8 \
    --num_heads 32 \
    --num_kv_heads 8 \
    --dropout 0.3 \
    --output_dir results \
    --exp_name trial2 \
    --devices 1 \
    --precision "32"

# With face segmentation head (multi-task training)
# Note: inverse-frequency class weights are computed automatically from
# the training data when --face_segmentation is enabled.
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --face_segmentation \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0016 \
    --precision "32"
    --exp_name face_seg

# Face segmentation with tuned loss weights
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --face_segmentation \
    --face_seg_weight 2.0 \
    --model_cls_weight 0.5 \
    --face_seg_hidden_dim 512 \
    --face_seg_dropout 0.3 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0016 \
    --precision "32"
    --exp_name face_seg_heavy

# Resume from checkpoint
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --resume_from results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --max_epochs 200

# Fast development test (1 batch train + 1 batch val)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --fast_dev_run
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Preprocessed data directory |
| `--num_classes` | auto | Number of classes (auto-detected from metadata.json) |
| `--multi_label` | True | Multi-label classification |
| `--face_segmentation` | False | Enable face-level segmentation head |
| `--face_seg_weight` | 1.0 | Loss weight for face segmentation |
| `--model_cls_weight` | 1.0 | Loss weight for model classification |
| `--num_face_classes` | 27 | Number of face-level classes |
| `--face_seg_hidden_dim` | 512 | Hidden dimension for face segmentation MLP |
| `--face_seg_dropout` | 0.3 | Dropout for face segmentation MLP |
| `--use_rope` | False | Enable Rotary Position Embeddings (disabled by default for graph nodes) |
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
| `--accumulate_grad_batches` | 1 | Gradient accumulation steps |
| `--output_dir` | "results" | Output directory |
| `--exp_name` | "brepformer" | Experiment name |
| `--seed` | 42 | Random seed |
| `--devices` | 1 | Number of GPUs |
| `--precision` | "32" | Training precision ("16" or "32") |
| `--fast_dev_run` | False | Quick test run |
| `--resume_from` | None | Checkpoint to resume from |
| `--limit_data` | None | Limit total dataset to N samples; saves manifest for test/analyze/infer |

**Callbacks:**
- `ModelCheckpoint`: saves top-3 models by `val/f1`
- `EarlyStopping`: patience=20 on `val/loss`
- `LearningRateMonitor`: logs LR to TensorBoard

#### Defeature Dataset — Training

```bash
# *best* — face segmentation training on defeature data (5 classes)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_face_classes 5 \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --face_seg_hidden_dim 512 \
    --max_faces 500 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_workers 0 \
    --precision "32" \
    --output_dir results \
    --exp_name trial1_ss

# Quick sanity check
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_face_classes 5 \
    --fast_dev_run \
    --num_workers 0 \
    --batch_size 4

# Model-level classification only (no face segmentation)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir results \
    --exp_name defeature_cls
```

#### Limited Data Training (Subset Testing)

Use `--limit_data N` to train on a random subset of the full preprocessed dataset. This is useful for quick experiments and debugging without re-preprocessing. The flag:

1. Randomly selects N samples proportionally across train/val/test splits (seeded by `--seed`)
2. Saves a `limit_data_manifest.json` in the experiment output directory
3. The manifest is reused by test, analyze, and infer scripts via `--limit_data_manifest`

```bash
# Train on 1000 samples (defeature: ~800 train, ~100 val, ~100 test)
python brepformer/train_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_face_classes 5 \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --max_faces 500 \
    --max_epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_workers 0 \
    --output_dir results \
    --exp_name defeature_1k \
    --limit_data 1000

# Test on the same 1000-sample subset
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/defeature_1k/best-epoch=XX-val/f1=X.XXXX.ckpt \
    --batch_size 32 \
    --num_workers 0 \
    --output_file results/defeature_1k/test_results.json \
    --output_face_preds results/defeature_1k/face_preds.json \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

# Analyze on the same subset
python brepformer/analyze.py \
    --checkpoint results/defeature_1k/best-epoch=XX-val/f1=X.XXXX.ckpt \
    --data_dir brepformer/data/defeature_processed \
    --mode all \
    --output_dir results/defeature_1k/analysis \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

# Batch inference on the same subset's STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint results/defeature_1k/best-epoch=XX-val/f1=X.XXXX.ckpt \
    --output_dir results/defeature_1k/inference/ \
    --limit_data_manifest results/defeature_1k/limit_data_manifest.json

# Or use run_pipeline.py which auto-propagates the manifest
python brepformer/run_pipeline.py \
    --data_dir brepformer/data/defeature \
    --output_dir results/defeature_1k_pipeline \
    --face_segmentation \
    --num_face_classes 5 \
    --skip_preprocess \
    --limit_data 1000
```

**Manifest format** (`limit_data_manifest.json`):
```json
{
  "limit_data": 1000,
  "seed": 42,
  "data_dir": "brepformer/data/defeature_processed",
  "splits": {
    "train": ["00001.pkl", "00005.pkl", "..."],
    "val": ["00002.pkl", "..."],
    "test": ["00003.pkl", "..."]
  },
  "model_ids": ["model_a", "model_b", "..."]
}
```

| Script | Flag | Description |
|--------|------|-------------|
| `train_preprocessed.py` | `--limit_data N` | Create manifest and train on N-sample subset |
| `run_pipeline.py` | `--limit_data N` | Same as above; manifest auto-propagated to test/analyze |
| `test_preprocessed.py` | `--limit_data_manifest PATH` | Evaluate on the manifest's subset |
| `analyze.py` | `--limit_data_manifest PATH` | Analyze on the manifest's subset |
| `infer.py` | `--limit_data_manifest PATH` | Filter batch STEP files to the manifest's model_ids |

### `train.py` (legacy — use `train_preprocessed.py` instead)

Training with raw JSON data. Single-worker only, no face segmentation, no class weighting, no `--use_rope`. Kept for backward compatibility with raw JSON workflows.

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

## Testing

### `test_preprocessed.py` ***recommended***

Evaluate a trained model on any dataset split. When the loaded model has `face_segmentation=True`, face metrics (`test/face_acc`, `test/face_f1`) are computed and logged automatically. Use `--output_face_preds` to write per-face predictions with aggregate metrics. Use `--split train` (or `val`) to evaluate on splits other than test.

```bash
# *best* — full face segmentation evaluation with per-face predictions (test split)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial5/best-epoch=39-val/f1=0.8832.ckpt \
    --batch_size 32 \
    --num_workers 4 \
    --output_file preds/trial5/test_face_results.json \
    --output_face_preds preds/trial5/face_preds.json

# Evaluate on the training split (to check for overfitting)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --batch_size 32 \
    --num_workers 4 \
    --split train \
    --output_file brepformer/preds/trial5/train_face_results.json \
    --output_face_preds brepformer/preds/trial5/train_face_preds.json

# Evaluate on the validation split
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial3/best-epoch=42-val/f1=0.8795.ckpt \
    --batch_size 32 \
    --num_workers 4 \
    --split val \
    --output_file preds/trial3/val_face_results.json

# Standard test (no face preds)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --batch_size 32 \
    --num_workers 4 \
    --output_file test_results.json

# With per-face predictions and metrics (face segmentation models)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/mftrcad_processed \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output_file test_face_results.json \
    --output_face_preds face_preds.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Preprocessed data directory |
| `--checkpoint` | required | Model checkpoint path |
| `--batch_size` | 32 | Batch size |
| `--num_workers` | 4 | Data loader workers |
| `--split` | `"test"` | Dataset split to evaluate: `train`, `val`, or `test` |
| `--output_file` | None | Output JSON for model-level results |
| `--output_face_preds` | None | Output JSON for per-face predictions |
| `--real_classes` | False | Remap 27 MFTRCAD classes to 8 real machining feature categories |
| `--limit_data_manifest` | None | Path to manifest JSON for reproducible dataset subsetting |

**`--output_face_preds` JSON structure:**
```json
{
  "metrics": {
    "face_accuracy": 0.85,
    "mean_iou": 0.72,
    "per_class_iou": [0.91, 0.78, 0.65, ...]
  },
  "predictions": [
    {
      "model_id": "00000001",
      "num_faces": 24,
      "face_preds": [24, 0, 0, 1, ...],
      "face_targets": [24, 0, 0, 1, ...],
      "face_probs": [[0.01, 0.02, ...], ...]
    }
  ]
}
```

#### Defeature Dataset — Testing

```bash
# *best* — full face segmentation evaluation on defeature test split
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --batch_size 4 \
    --num_workers 0 \
    --output_file brepformer/preds/trial1_ss/test_results.json \
    --output_face_preds brepformer/preds/trial1_ss/face_preds.json

# Train split evaluation (overfitting check)
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --batch_size 4 \
    --num_workers 0 \
    --split train \
    --output_file brepformer/preds/trial1_ss/train_results.json \
    --output_face_preds brepformer/preds/trial1_ss/train_face_preds.json

# Standard test
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint results/defeature/best-*.ckpt \
    --batch_size 32 \
    --num_workers 4
```

---

## Inference

### `infer.py` ***recommended***

End-to-end inference on raw STEP files. Converts STEP geometry to a graph representation using pythonOCC, runs it through the trained model, and outputs per-face predictions. Requires `pythonocc-core`.

```bash
# *best* — batch inference with face segmentation model
python -m brepformer.infer \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --output_dir inference/trial5/face_inference_results/

# Single file → JSON output (full predictions with probabilities)
python -m brepformer.infer \
    --step path/to/model.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output preds.json

# Single file → .seg output (one label per line)
python -m brepformer.infer \
    --step path/to/model.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output preds.seg

# Batch mode (writes per-model .json + .seg + combined all_predictions.json)
python -m brepformer.infer \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output_dir inference_results/

# Batch mode for face model (writes per-model .json + .seg + combined all_predictions.json)
python -m brepformer.infer \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output_dir face_inference_results/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--checkpoint` | required | Model checkpoint path |
| `--output` | None | Output file path (.json or .seg) |
| `--output_dir` | None | Output directory (batch mode) |
| `--real_classes` | False | Remap 27 MFTRCAD classes to 8 real machining feature categories |
| `--limit_data_manifest` | None | Path to manifest JSON to filter batch STEP files by model_id |

**Output formats:**
- **`.seg`**: One integer label per line. Line _i_ is the predicted class for face _i_ (TopExp_Explorer order). Directly loadable by `visualize_seg.py` and `export_freecad.py`.
- **`.json`**: Full predictions including model-level classes, per-face predictions, per-face class probabilities, and human-readable class names.

**Console output** includes a summary of predicted face counts per class:
```
Model: 00000001
Faces: 24
Model classes: ['chamfer', 'through_hole', 'stock']
Face predictions (24 faces):
  chamfer: 6
  through_hole: 4
  stock: 14
```

#### Defeature Dataset — Inference

```bash
# *best* — batch inference on defeature STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --output_dir inference/trial1_ss/

# Single defeature STEP file
python -m brepformer.infer \
    --step brepformer/data/defeature/steps/some_model.step \
    --checkpoint results/defeature/best-*.ckpt \
    --output preds.seg
```

---

## FreeCAD Export

### `export_freecad.py` ***recommended***

Writes colored STEP AP214 files using pythonOCC's XCAF framework (`STEPCAFControl_Writer` + `XCAFDoc_ColorSurf`). Each face is assigned a surface color matching its predicted or ground-truth class. FreeCAD reads these colors natively — no plugins required. Requires `pythonocc-core`.

```bash
# *best* — batch colored export from face segmentation inference results
python -m brepformer.export_freecad \
    --step_dir brepformer/data/mftrcad/steps/ \
    --seg_dir face_inference_results/ \
    --output_dir colored_steps/

# From a .seg label file
python -m brepformer.export_freecad \
    --step path/to/model.step \
    --seg preds.seg \
    --output colored.step

# From live inference (runs model on the fly)
python -m brepformer.export_freecad \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output colored.step

# From live inference for the face model
python -m brepformer.export_freecad \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --output colored.step
    
# From ground-truth labels JSON
python -m brepformer.export_freecad \
    --step path/to/model.step \
    --labels_json brepformer/data/mftrcad/labels/00000001_result.json \
    --output colored_gt.step

# Auto-match GT labels by model_id from labels directory
python -m brepformer.export_freecad \
    --step brepformer/data/mftrcad/steps/00000001.step \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --output colored_gt.step

# Batch mode with GT labels
python -m brepformer.export_freecad \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --output_dir colored_steps/

# Batch mode with inference
python -m brepformer.export_freecad \
    --step_dir brepformer/data/mftrcad/steps/ \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --output_dir colored_steps/

# From predicted .seg files in a directory (e.g. inference_results/)
python -m brepformer.export_freecad \
    --step brepformer/data/sample/steps/20240116_231044_0_result.step \
    --seg_dir inference_results/ \
    --output colored.step

# Batch mode with predicted .seg files from face inference
python -m brepformer.export_freecad \
    --step_dir brepformer/data/sample/steps/ \
    --seg_dir face_inference_results/ \
    --output_dir colored_steps/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--seg` | None | Path to .seg label file |
| `--seg_dir` | None | Directory of .seg prediction files (auto-matches by model_id) |
| `--checkpoint` | None | Model checkpoint (runs inference on the fly) |
| `--labels_json` | None | Path to GT labels JSON (`_result.json`) |
| `--labels_dir` | None | Directory of GT label JSON files (auto-matches by model_id) |
| `--output` | None | Output STEP file path (default: `{input}.colored.step`) |
| `--output_dir` | None | Output directory (batch mode, default: `{step_dir}/colored/`) |
| `--real_classes` | False | Remap 27 MFTRCAD classes to 8 real machining feature categories |

**Label source priority** (first match wins): `--seg` > `--seg_dir` > `--labels_json` > `--labels_dir` > `--checkpoint`.

**FreeCAD workflow:** Open the colored `.step` file in FreeCAD (File → Open). Each face displays its predicted class color. See [viewer.md](viewer.md) for the 27-class color reference table.

#### Defeature Dataset — FreeCAD Export

```bash
# Batch colored export from defeature inference results
python -m brepformer.export_freecad \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/defeature/ \
    --output_dir colored_defeature/

# Single model from GT labels
python -m brepformer.export_freecad \
    --step brepformer/data/defeature/steps/some_model.step \
    --labels_json brepformer/data/defeature/labels/some_model.json \
    --output colored_gt.step

# Live inference on a single defeature STEP file
python -m brepformer.export_freecad \
    --step brepformer/data/defeature/steps/some_model.step \
    --checkpoint results/defeature/best-*.ckpt \
    --output colored.step
```

---

## Automatic Defeaturing

Two defeaturing engines are available. Both have the same CLI interface and output format:

| Engine | Module | pythonocc | Key feature |
|---|---|---|---|
| **v2** (recommended) | `brepformer.defeature_v2` | >= 7.9 | Adaptive tolerance, history tracking, enhanced healing |
| v1 | `brepformer.defeature` | >= 7.5 | Original 5-phase progressive engine |

The `scripts/run_defeature.sh` wrapper selects the engine via `--engine v1|v2` (default: v2).

### `defeature_v2.py` ***recommended — modern engine with adaptive tolerance***

Drop-in replacement for `defeature.py` with six improvements that increase per-face removal success rate:

1. **Adaptive fuzzy tolerance** — retries with `SetFuzzyValue()` at 0 → 1e-5 → 5e-5 → 1e-4 → 5e-4 → 1e-3
2. **History-based face tracking** — uses `BRepTools_History` instead of `IndexedMap.Contains()` heuristics
3. **Pre-validation** — auto-repairs input shapes with `ShapeFix_Shape` + `ShapeFix_ShapeTolerance`
4. **Enhanced healing** — tolerance harmonization → shape fix → conditional sewing → face unification with angular/linear tolerance
5. **Area-based ordering** — removes smallest features first (higher success rate, unblocks larger features)
6. **Intermediate healing** — `UnifySameDomain` between single-face removals in Phase 5c

```bash
# *best* — v2 engine via script wrapper
./scripts/run_defeature.sh --step model.step --verbose

# Or directly
python -m brepformer.defeature_v2 --step model.step

# Batch mode
python -m brepformer.defeature_v2 --step_dir brepformer/data/defeature/steps/

# Higher tolerance for complex fillet models
python -m brepformer.defeature_v2 --step model.step --max_fuzzy 0.01 --verbose

# From pre-computed .seg predictions
python -m brepformer.defeature_v2 --step model.step --seg preds.seg

# Batch with pre-computed predictions + colored output
python -m brepformer.defeature_v2 \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output/trial1_ss1500 \
    --max_fuzzy 0.01
    --save_colored --verbose
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--checkpoint` | `results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt` | Model checkpoint (must have face segmentation head) |
| `--seg` | None | Path to .seg file with pre-computed predictions (single mode) |
| `--seg_dir` | None | Directory of .seg prediction files (batch mode) |
| `--output_dir` | `brepformer/defeatured_output` | Output directory for defeatured STEP files |
| `--save_colored` | False | Also save a colored STEP showing predictions before defeaturing |
| `--verbose` | False | Print detailed per-phase defeaturing progress |
| `--max_fuzzy` | `1e-3` | Maximum fuzzy tolerance for adaptive retry (v2 only) |

**Phase algorithm (v2):**
```
Phase 0 — Pre-validate input shape (auto-repair if needed)
Phase 1 — Try ALL features at once (adaptive tolerance)
Phase 2 — Progressive type-by-type: fillet → chamfer → hole → cut (adaptive)
Phase 3 — Connected components per failed type (area-sorted, adaptive)
Phase 4 — Retry on modified shape (history-based face tracking, adaptive)
Phase 5 — Last-ditch: alternate orders, single-face iterative (adaptive, intermediate healing)
Cleanup — Multi-stage healing pipeline
```

### `defeature.py` (v1 — original engine)

Original defeaturing pipeline with 5-phase progressive face removal. Same CLI as v2 except without `--max_fuzzy`.

```bash
# *best* -- defeature a single STEP file (uses default defeature checkpoint)
python -m brepformer.defeature \
    --step model.step

# Defeature a directory of STEP files
python -m brepformer.defeature \
    --step_dir brepformer/data/defeature/steps/

# Custom checkpoint and output directory
python -m brepformer.defeature \
    --step model.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output_dir my_defeatured/

# From pre-computed .seg predictions (skip inference)
python -m brepformer.defeature \
    --step model.step \
    --seg preds.seg

# Batch mode with pre-computed .seg files
python -m brepformer.defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/

# Also save colored STEP for visual comparison
python -m brepformer.defeature \
    --step model.step \
    --save_colored --verbose
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--step_dir` | None | Directory of STEP files (batch mode) |
| `--checkpoint` | `results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt` | Model checkpoint (must have face segmentation head) |
| `--seg` | None | Path to .seg file with pre-computed predictions (single mode) |
| `--seg_dir` | None | Directory of .seg prediction files (batch mode) |
| `--output_dir` | `brepformer/defeatured_output` | Output directory for defeatured STEP files |
| `--save_colored` | False | Also save a colored STEP showing predictions before defeaturing |
| `--verbose` | False | Print detailed defeaturing progress |

**Output structure:**
```
brepformer/defeatured_output/
├── model_id_defeatured.step    # Defeatured STEP file
├── model_id_colored.step       # Colored predictions (if --save_colored)
├── model_id_report.json        # Per-model defeaturing report
└── batch_report.json           # Batch summary (batch mode only)
```

**Report JSON fields:**
| Field | Description |
|-------|-------------|
| `status` | `"success"`, `"no_features"`, or `"error"` |
| `num_faces` | Total faces in input model |
| `kept` | Faces classified as random (kept) |
| `removed` | Feature faces successfully removed |
| `failed` | Feature faces that could not be removed |
| `valid` | Whether the output shape passes BRepCheck validation |

**Integration with other scripts:**

Both `export_freecad.py` and `visualize_seg.py` support a `--defeature` flag that delegates to this module:

```bash
# Export colored STEP + defeatured STEP in one command
python -m brepformer.export_freecad \
    --step model.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --defeature

# In the viewer, press F to defeature the current model
python -m brepformer.visualize_seg \
    --step model.step \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --defeature
```

#### Defeature Dataset -- Full Defeaturing Pipeline

```bash
# Step 1: Run inference on defeature STEP files
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output_dir inference/trial1_ss1500/

# Step 2: Defeature using inference results (v2 engine via script)
./scripts/run_defeature.sh \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output/ \
    --save_colored --verbose

# Or using v2 module directly
python -m brepformer.defeature_v2 \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output/ \
    --save_colored --verbose

# Or combine inference + defeaturing in one command
python -m brepformer.defeature_v2 \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
    --output_dir brepformer/defeatured_output/

# Using v1 engine for comparison
./scripts/run_defeature.sh --engine v1 \
    --step_dir brepformer/data/defeature/steps/ \
    --seg_dir inference/trial1_ss1500/ \
    --output_dir brepformer/defeatured_output_v1/ \
    --verbose
```

---

## Analysis

### `analyze.py` ***recommended***

Comprehensive model and prediction analysis with multiple modes. Includes a `face_segmentation` mode for per-face metrics (per-class IoU, mean IoU, per-face accuracy, 27x27 confusion matrix).

```bash
# step inference analysis comparing preprocessed vs STEP conversion paths

# *best* 
python brepformer/analyze.py     
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt"     
    --data_dir brepformer/data/mftrcad_processed     --step_dir brepformer/data/mftrcad/steps     --mode all     
    --output_dir brepformer/analysis_results/trial5

# Full analysis including face segmentation metrics
python brepformer/analyze.py \
    --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/mftrcad_processed \
    --step_dir brepformer/data/mftrcad/steps \
    --mode all \
    --output_dir analysis_results

# Architecture analysis (no data needed)
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --mode architecture \
    --output_dir analysis_results

# Per-class model-level performance
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

# Detailed per-sample predictions
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode predictions \
    --output_dir analysis_results

# Confusion matrix
python brepformer/analyze.py \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode confusion_matrix \
    --output_dir analysis_results

# Face segmentation metrics (requires face_segmentation model)
python brepformer/analyze.py \
    --checkpoint results/face_seg_heavy/best-epoch=46-val/f1=0.8608.ckpt \
    --data_dir brepformer/data/mftrcad_processed \
    --mode face_segmentation \
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
| `--split` | "test" | Data split to analyze ("train", "val", "test") |
| `--real_classes` | False | Remap 27 MFTRCAD classes to 8 real machining feature categories |
| `--step_dir` | None | Directory of STEP files (required for `step_inference` mode) |
| `--max_models` | 50 | Max models to evaluate in `step_inference` mode |
| `--limit_data_manifest` | None | Path to manifest JSON for reproducible dataset subsetting |

| Mode | Description | Output files |
|------|-------------|--------------|
| `architecture` | Model config, parameter counts, per-module breakdown, model size | `architecture.json` |
| `per_class` | Per-class precision, recall, F1, support; co-occurrence matrix | `per_class_metrics.json`, `co_occurrence_matrix.npy` |
| `embeddings` | Graph embeddings: raw (256D), PCA (50D), t-SNE (2D) | `embeddings.npy`, `embeddings_pca50.npy`, `embeddings_tsne.npy` |
| `predictions` | Per-sample predictions with probabilities, Jaccard similarity | `predictions.json` |
| `confusion_matrix` | Standard confusion matrix (single-label) or co-occurrence (multi-label) | `confusion_matrix.npy` or `co_occurrence_matrix.npy` |
| `face_segmentation` | Per-face accuracy, per-class IoU, mean IoU, 27x27 confusion matrix | `face_seg_metrics.json`, `face_seg_confusion.npy` |
| `step_inference` | Compares preprocessed pickle vs step_to_graph inference paths on test models. Reports face accuracy, F1, precision, recall for each path. Requires `--step_dir`. | `step_inference_results.json` |
| `all` | Runs all of the above (except `step_inference`) | All output files |

#### Defeature Dataset — Analysis

```bash
# *best* — full analysis on defeature model
python brepformer/analyze.py \
    --checkpoint results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt \
    --data_dir brepformer/data/defeature_processed \
    --mode all \
    --output_dir brepformer/analysis_results/defeature

# Face segmentation metrics (5-class confusion matrix)
python brepformer/analyze.py \
    --checkpoint results/defeature/best-*.ckpt \
    --data_dir brepformer/data/defeature_processed \
    --mode face_segmentation \
    --output_dir analysis_results/defeature

# Step inference comparison
python brepformer/analyze.py \
    --checkpoint results/defeature/best-*.ckpt \
    --data_dir brepformer/data/defeature_processed \
    --step_dir brepformer/data/defeature/steps \
    --mode step_inference \
    --max_models 50 \
    --output_dir analysis_results/defeature
```

**`face_seg_metrics.json` structure:**
```json
{
  "face_accuracy": 0.85,
  "mean_iou": 0.72,
  "per_class": [
    {
      "class_id": 0,
      "class_name": "chamfer",
      "precision": 0.91,
      "recall": 0.88,
      "f1": 0.89,
      "iou": 0.81,
      "support": 3420
    }
  ]
}
```

---

## Visualization

### `visualize.py`

Generate matplotlib plots from analysis results.

```bash
# *best* — all plots from face segmentation analysis
python brepformer/visualize.py \
    --mode all \
    --input_dir analysis_results \
    --log_dir results/face_seg_best \
    --output_dir plots \
    --format png --dpi 150

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
| `--mode` | "all" | Visualization mode ("embeddings", "metrics", "training", "all") |
| `--input_dir` | "analysis_results" | Analysis results directory |
| `--log_dir` | "results/brepformer" | TensorBoard logs directory |
| `--output_dir` | "plots" | Output directory for image files |
| `--format` | "png" | Image format ("png", "pdf", "svg") |
| `--dpi` | 150 | Image DPI |

### `visualize_seg.py`

Interactive Qt + pythonOCC 3D viewer for visualizing 27 MFTRCAD machining feature classes on STEP models. Displays actual B-rep geometry with per-face coloring. Requires `pythonocc-core` and `PyQt5`.

```bash
# *best* — browse dataset with face seg predictions vs ground truth
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --seg_dir inference/trial5/face_inference_results/

# View with predicted labels from .seg file
python -m brepformer.visualize_seg \
    --step path/to/model.step \
    --seg preds.seg

# View with ground-truth labels from JSON
python -m brepformer.visualize_seg \
    --step path/to/model.step \
    --labels_json brepformer/data/mftrcad/labels/00000001_result.json

# Run inference and display
python -m brepformer.visualize_seg \
    --step path/to/model.step \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt

# Browse a dataset directory with auto-loaded GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/

# Browse with live inference enabled
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps/ \
    --labels_dir brepformer/data/mftrcad/labels/ \
    --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt

# Browse with predicted .seg files from inference_results/
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --seg_dir brepformer/inference_results/

# Browse with predicted .seg files from face inference + GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --labels_dir brepformer/data/sample/labels/ \
    --seg_dir brepformer/face_inference_results/

# The same but for all data
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/mftrcad/steps \
    --labels_dir brepformer/data/sample/labels/ \
    --seg_dir brepformer/face_inference_results/


# Browse with predicted .seg files from face inference + GT labels + small classes 
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/sample/steps/ \
    --labels_dir brepformer/data/sample/labels/ \
    --seg_dir brepformer/face_inference_results/
    --real_classes
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single STEP file |
| `--seg` | None | Path to .seg label file (predicted labels) |
| `--seg_dir` | None | Directory of .seg prediction files (auto-matches by model_id) |
| `--labels_json` | None | Path to GT labels JSON file |
| `--labels_dir` | None | Directory of GT label JSON files (auto-matches by model_id) |
| `--checkpoint` | None | Model checkpoint (enables "Run Inference" button) |
| `--step_dir` | None | Directory of STEP files (enables batch prev/next navigation) |
| `--real_classes` | False | Remap 27 MFTRCAD classes to 8 real machining feature categories |

See [viewer.md](viewer.md) for keyboard shortcuts, display modes, and the 27-class color reference.

#### Defeature Dataset — Visualization

```bash
# Browse defeature dataset with GT labels
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/

# Browse with predicted .seg files from inference
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir inference/trial1_ss/

# Single defeature model with live inference
python -m brepformer.visualize_seg \
    --step brepformer/data/defeature/steps/some_model.step \
    --checkpoint results/defeature/best-*.ckpt
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir inference/defeature/

# Plots from defeature analysis
python brepformer/visualize.py \
    --mode all \
    --input_dir analysis_results/defeature \
    --log_dir results/defeature \
    --output_dir plots/defeature
```

### `visualize_defeature.py`

Interactive Qt + pythonOCC 3D viewer for comparing original STEP models side-by-side with their defeatured counterparts. Shows the original model with per-face class coloring (left) next to the defeatured clean stock shape (right). Requires `pythonocc-core` and `PyQt5`.

```bash
# *best* — browse defeatured results with GT labels
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/trial1_ss1500 \
    --labels_dir brepformer/data/defeature/labels/

# Single model comparison
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/some_model.step \
    --defeatured brepformer/defeatured_output/some_model_defeatured.step

# With predicted .seg labels on original
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/some_model.step \
    --defeatured brepformer/defeatured_output/some_model_defeatured.step \
    --seg inference/trial1_ss1500/some_model.seg

# Browse with predictions coloring the original
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --seg_dir inference/trial1_ss1500/

# With live inference on original model
python -m brepformer.visualize_defeature \
    --step brepformer/data/defeature/steps/some_model.step \
    --defeatured brepformer/defeatured_output/some_model_defeatured.step \
    --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt

# Browse with defeaturing reports (shows removed/failed stats)
python -m brepformer.visualize_defeature \
    --step_dir brepformer/data/defeature/steps/ \
    --defeatured_dir brepformer/defeatured_output/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --report_dir brepformer/defeatured_output/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | None | Path to a single original STEP file |
| `--step_dir` | None | Directory of original STEP files (batch navigation) |
| `--defeatured` | None | Path to a single defeatured STEP file |
| `--defeatured_dir` | None | Directory of defeatured STEP files (auto-matches `{model_id}_defeatured.step`) |
| `--seg` | None | Path to .seg label file for coloring original faces |
| `--seg_dir` | None | Directory of .seg prediction files (auto-matches by model_id) |
| `--labels_json` | None | Path to GT labels JSON file |
| `--labels_dir` | None | Directory of GT label JSON files (auto-matches by model_id) |
| `--checkpoint` | None | Model checkpoint (enables live inference on original) |
| `--report_dir` | None | Directory of `{model_id}_report.json` defeaturing reports |

**Keyboard controls:** Same as `visualize_seg.py` — D/RIGHT (next model), A/LEFT (prev model), T (toggle GT/Pred on original), I (face info), S (summary with defeaturing stats), Q/ESC (quit).

**Display layout:**
- **Left viewer**: Original STEP model with per-face class coloring (5-class defeature palette)
- **Right viewer**: Defeatured STEP model in uniform gray (all features removed)

---

## TensorBoard

### `tensorboard_server.py`

Launch TensorBoard for training monitoring. When training with `--face_segmentation`, additional face-level metrics (`face_acc`, `face_f1`) appear alongside model-level metrics.

```bash
# *best* — compare all experiments (face seg metrics visible when trained with --face_segmentation)
python brepformer/tensorboard_server.py --log_dir results --compare --face_segmentation

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

## Model Export

### `export_model.py`

Export trained models for deployment.

```bash
# *best* — export face segmentation model as state dict
python brepformer/export_model.py \
    --checkpoint results/face_seg_best/best-epoch=XX-val/f1=X.XXXX.ckpt \
    --format state_dict \
    --output_path face_seg_model.pt

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
| `--format` | "state_dict" | Export format ("state_dict", "torchscript", "onnx") |
| `--output_path` | required | Output file path |
| `--max_faces` | 100 | Max faces for tracing |
| `--max_edges` | 200 | Max edges for tracing |

---

## STEP to Graph Conversion

### `data/step_to_graph.py`

Converts STEP CAD files to BrepFormer graph format using pythonOCC. Used internally by `infer.py` and `export_freecad.py`. Can also be run standalone for debugging.

```bash
# Convert a STEP file and print summary
python -m brepformer.data.step_to_graph path/to/model.step
```

**Key functions:**
- `step_to_graph(step_path)` — returns raw graph data dict (face_grid, face_attr, edge_index, edge_attr, edge_grid)
- `step_to_preprocessed_sample(step_path)` — returns a dict matching `PreprocessedDataset` format (with spatial_pos, in_degree computed)

---

## Class Definitions

### `data/classes.py`

Single source of truth for the 27 MFTRCAD machining feature class definitions. All visualization, export, and analysis scripts import from here.

**Contents:**
- `CLASS_NAMES` — list of 27 class name strings
- `CLASS_COLORS_HEX` — list of 27 hex color strings (tab20 + tab20b palette)
- `NUM_CLASSES` — integer 27
- `COLOR_NAMES` — human-readable color names
- `UNLABELED_COLOR_HEX`, `HIGHLIGHT_COLOR_HEX`, `EDGE_COLOR_HEX` — special colors
- `hex_to_rgb01()`, `hex_to_rgb255()`, `get_class_color_rgb01()`, `get_class_name()` — utility functions

```python
from brepformer.data.classes import CLASS_NAMES, CLASS_COLORS_HEX, NUM_CLASSES

print(CLASS_NAMES[0])       # "chamfer"
print(CLASS_COLORS_HEX[0])  # "#1f77b4"
print(NUM_CLASSES)           # 27
```

**Real classes (8 grouped categories):**

The `--real_classes` flag (available on `infer.py`, `visualize_seg.py`, `export_freecad.py`, `test_preprocessed.py`, `analyze.py`) remaps the 27 fine-grained classes into 8 higher-level machining feature categories:

| Real ID | Name | 27-class IDs |
|---------|------|-------------|
| 0 | other_surfaces | stock(24), rectangular_passage_2(25) |
| 1 | through_hole | through_hole(1) |
| 2 | blind_hole | blind_hole(12) |
| 3 | chamfer | chamfer(0), chamfer_2(26) |
| 4 | fillet | round(23) |
| 5 | through_cut | triangular_passage(2), rectangular_passage(3), 6sides_passage(4), triangular_through_slot(5), rectangular_through_slot(6), circular_through_slot(7) |
| 6 | blind_cut | triangular_pocket(13), rectangular_pocket(14), 6sides_pocket(15), circular_end_pocket(16), rectangular_blind_slot(17), v_shaped_blind_slot(18), circular_blind_slot(19) |
| 7 | extrude | rectangular_through_step(8), 2sides_through_step(9), slanted_through_step(10), Oring(11), rectangular_blind_step(20), 2sides_blind_step(21), triangular_blind_step(22) |

```python
from brepformer.data.classes import REAL_CLASS_NAMES, REAL_NUM_CLASSES, map_labels_to_real

print(REAL_CLASS_NAMES)     # ['other_surfaces', 'through_hole', ..., 'extrude']
print(REAL_NUM_CLASSES)     # 8
print(map_labels_to_real([0, 1, 12, 24]))  # [3, 1, 2, 0]
```

**Defeature classes (5 categories):**

The defeature dataset (navin_defeaturing) uses 5 machining feature classes, remapped from the original 7-class scheme:

| Class ID | Name | Color | Original IDs |
|----------|------|-------|-------------|
| 0 | random | gray (#7f7f7f) | 0 |
| 1 | hole | blue (#1f77b4) | 1, 2 |
| 2 | chamfer | orange (#ff7f0e) | 3 |
| 3 | fillet | green (#2ca02c) | 4 |
| 4 | cut | red (#d62728) | 5, 6 |

```python
from brepformer.data.classes import DEFEATURE_CLASS_NAMES, DEFEATURE_NUM_CLASSES

print(DEFEATURE_CLASS_NAMES)  # ['random', 'hole', 'chamfer', 'fillet', 'cut']
print(DEFEATURE_NUM_CLASSES)  # 5
```

**Dataset statistics:**
- 1561 models, 115,766 total faces
- Class distribution: random 27.3%, hole 23.6%, chamfer 3.8%, fillet 14.9%, cut 30.5%

---

## Fine-Tuning Pipeline (MFTRCAD 27-class -> Defeature 5-class)

### Overview

Fine-tune a pre-trained MFTRCAD model on the defeature dataset by:
1. Loading encoder weights from a pre-trained 27-class checkpoint (trial5)
2. Replacing classifier heads with new 5-class heads, warm-started by averaging the 27-class weights according to the `CLASS_TO_DEFEATURE` mapping
3. Training with differential learning rates (lower for encoder, higher for new heads)
4. Optionally freezing the encoder for initial epochs

### `fine_tune.py` ***recommended***

Dedicated fine-tuning script that handles weight transfer, head remapping, and differential learning rates automatically.

#### MFTRCAD -> Defeature Class Mapping

| MFTRCAD 27-Class | ID | Defeature 5-Class | ID |
|---|---|---|---|
| chamfer, chamfer_2 | 0, 26 | chamfer | 2 |
| through_hole, blind_hole | 1, 12 | hole | 1 |
| round | 23 | fillet | 3 |
| stock | 24 | random | 0 |
| All passages, slots, steps, pockets, Oring, passage_2 | 2-11, 13-22, 25 | cut | 4 |

```python
from brepformer.data.classes import CLASS_TO_DEFEATURE, map_labels_to_defeature

print(CLASS_TO_DEFEATURE)  # [2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0, 4, 2]
print(map_labels_to_defeature([0, 1, 12, 23, 24]))  # [2, 1, 1, 3, 0]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained` | required | Path to pre-trained 27-class checkpoint |
| `--data_dir` | required | Preprocessed defeature data directory |
| `--num_classes` | 5 | Number of target classes |
| `--num_face_classes` | 5 | Number of target face classes |
| `--freeze_encoder_epochs` | 0 | Freeze encoder for this many initial epochs (0 = never) |
| `--encoder_lr_factor` | 0.1 | LR multiplier for encoder params (relative to head LR) |
| `--no_warm_start` | False | Don't warm-start heads from 27-class weights |
| `--face_segmentation` | True | Enable face segmentation (default for fine-tuning) |
| `--face_seg_weight` | 1.0 | Loss weight for face segmentation |
| `--model_cls_weight` | 1.0 | Loss weight for model classification |
| `--weighted_crossentropy` | False | Use inverse-frequency class weights for face seg |
| `--learning_rate` | 0.0005 | Learning rate for classifier heads |
| `--max_epochs` | 100 | Maximum fine-tuning epochs |
| `--warmup_steps` | 1000 | Warmup steps (lower than pre-training) |
| `--batch_size` | auto | Batch size (auto-detected from GPU memory) |
| `--num_workers` | 0 | Data loader workers |
| `--output_dir` | "results" | Output directory |
| `--exp_name` | "finetune" | Experiment name |

### Complete Fine-Tuning Pipeline

The full pipeline from raw defeature data to evaluated fine-tuned model:

```bash
# ========================================================
# Step 0: Ensure defeature data is prepared
# (skip if brepformer/data/defeature/ already has graphs/ and labels/)
# ========================================================
python brepformer/data/prepare_defeature.py \
    --source /mnt/c/projects/data/navin_defeaturing \
    --dest brepformer/data/defeature \
    --convert_graphs

# ========================================================
# Step 1: Preprocess defeature data (5 classes, with descriptors)
# ========================================================
python brepformer/preprocess.py \
    --data_dir brepformer/data/defeature \
    --output_dir brepformer/data/defeature_processed \
    --num_classes 5 \
    --compute_descriptors \
    --d2_bins 64 \
    --angle_bins 64

# ========================================================
# Step 2: Fine-tune from trial5 (best MFTRCAD weights)
# ========================================================
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --face_seg_weight 1.0 \
    --model_cls_weight 1.0 \
    --learning_rate 0.0005 \
    --encoder_lr_factor 0.1 \
    --max_epochs 100 \
    --warmup_steps 1000 \
    --num_workers 0 \
    --precision 32 \
    --output_dir results \
    --exp_name trial1_finetune_ss

# ========================================================
# Step 3: Test the fine-tuned model
# ========================================================
python brepformer/test_preprocessed.py \
    --data_dir brepformer/data/defeature_processed \
    --checkpoint "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --batch_size 1 \
    --num_workers 0 \
    --output_file brepformer/preds/trial1_finetune_ss/test_results.json \
    --output_face_preds brepformer/preds/trial1_finetune_ss/face_preds.json

# ========================================================
# Step 4: Analyze the fine-tuned model
# ========================================================
python brepformer/analyze.py \
    --checkpoint "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --mode all \
    --output_dir brepformer/analysis_results/trial1_finetune_ss

# ========================================================
# Step 5: Inference on STEP files
# ========================================================
python -m brepformer.infer \
    --step_dir brepformer/data/defeature/steps/ \
    --checkpoint "results/trial1_ss1500/last.ckpt" \
    --output_dir inference/trial1_1500_ss/

# ========================================================
# Step 6: Visualize results
# ========================================================
python -m brepformer.visualize_seg \
    --step_dir brepformer/data/defeature/steps/ \
    --labels_dir brepformer/data/defeature/labels/ \
    --seg_dir inference/trial1_finetune_ss/
```

#### Fine-Tuning Variants

```bash
# *best* — fine-tune with frozen encoder for 10 epochs, then unfreeze
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --freeze_encoder_epochs 10 \
    --encoder_lr_factor 0.1 \
    --learning_rate 0.0005 \
    --max_epochs 100 \
    --output_dir results \
    --exp_name finetune_frozen

# With class-weighted loss (for imbalanced defeature classes)
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --weighted_crossentropy \
    --learning_rate 0.0005 \
    --output_dir results \
    --exp_name finetune_weighted

# Without warm-start (random init for heads)
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --no_warm_start \
    --output_dir results \
    --exp_name finetune_cold

# Quick sanity check
python brepformer/fine_tune.py \
    --pretrained "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
    --data_dir brepformer/data/defeature_processed \
    --face_segmentation \
    --num_classes 5 \
    --num_face_classes 5 \
    --fast_dev_run \
    --num_workers 0 \
    --batch_size 4
```

---

## Flexible Defeaturing Commands

Both `defeature.py` and `defeature_v2.py` accept the same core CLI. Below are flexible Python commands for common workflows with variable input/output paths. The `scripts/run_defeature.sh` wrapper adds `--engine v1|v2` selection (default: v2).

### Variable Input/Output Paths

```bash
# Defeature a single file to a custom output directory
python -m brepformer.defeature_v2 \
    --step /path/to/my_model.step \
    --output_dir /path/to/output/

# Defeature a custom directory of STEP files
python -m brepformer.defeature_v2 \
    --step_dir /path/to/my_steps/ \
    --output_dir /path/to/my_output/ \
    --verbose

# v1 engine on a custom directory
python -m brepformer.defeature \
    --step_dir /path/to/my_steps/ \
    --output_dir /path/to/my_output/
```

### Using Pre-computed Predictions

```bash
# Single file with custom .seg file
python -m brepformer.defeature_v2 \
    --step /path/to/model.step \
    --seg /path/to/predictions.seg \
    --output_dir /path/to/output/

# Batch: match .seg files from one dir to STEP files from another
python -m brepformer.defeature_v2 \
    --step_dir /path/to/step_files/ \
    --seg_dir /path/to/seg_predictions/ \
    --output_dir /path/to/defeatured/ \
    --save_colored --verbose
```

### Custom Checkpoint

```bash
# Use a different model checkpoint
python -m brepformer.defeature_v2 \
    --step /path/to/model.step \
    --checkpoint /path/to/my_checkpoint.ckpt \
    --output_dir /path/to/output/

# Batch with custom checkpoint
python -m brepformer.defeature_v2 \
    --step_dir /path/to/step_files/ \
    --checkpoint results/my_experiment/best-epoch=50-val/f1=0.95.ckpt \
    --output_dir /path/to/defeatured/ \
    --verbose
```

### v2 Tolerance Control

```bash
# Higher tolerance for complex/filleted models
python -m brepformer.defeature_v2 \
    --step model.step \
    --max_fuzzy 0.01 \
    --verbose

# Lower tolerance for precision-critical models
python -m brepformer.defeature_v2 \
    --step model.step \
    --max_fuzzy 1e-4 \
    --verbose
```

### Shell Wrapper with Variable Paths

```bash
# Via run_defeature.sh with custom paths
./brepformer/scripts/run_defeature.sh \
    --step /path/to/model.step \
    --output_dir /path/to/output/ \
    --save_colored --verbose

# v1 engine via wrapper
./brepformer/scripts/run_defeature.sh --engine v1 \
    --step_dir /path/to/step_files/ \
    --output_dir /path/to/v1_output/ \
    --verbose

# v2 with high tolerance via wrapper
./brepformer/scripts/run_defeature.sh \
    --step model.step \
    --max_fuzzy 0.01 --verbose
```

### Combined Inference + Defeaturing (Variable Paths)

```bash
# Step 1: Run inference on custom STEP files
python -m brepformer.infer \
    --step_dir /path/to/my_steps/ \
    --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt \
    --output_dir /path/to/my_inference/

# Step 2: Defeature using those predictions
python -m brepformer.defeature_v2 \
    --step_dir /path/to/my_steps/ \
    --seg_dir /path/to/my_inference/ \
    --output_dir /path/to/my_defeatured/ \
    --save_colored --verbose

# Or combine inference + defeaturing in one command
python -m brepformer.defeature_v2 \
    --step_dir /path/to/my_steps/ \
    --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt \
    --output_dir /path/to/my_defeatured/ \
    --verbose
```
