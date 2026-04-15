#!/bin/bash
# ============================================================================
# BrepFormer Inference Pipeline
# ============================================================================
#
# End-to-end pipeline that takes any STEP file and runs:
#   1. Inference        — predict per-face labels (brepformer.infer)
#   2. FreeCAD export   — colored STEP file (brepformer.export_freecad)
#   3. Analysis         — model analysis (brepformer.analyze --mode all)
#   4. Visualize seg    — interactive 3D segmentation viewer (brepformer.visualize_seg)
#   --- conda env switch: brepmfr → new_brepmfr ---
#   5. Defeature v2     — remove predicted features (brepformer.defeature_v2)
#   6. Visualize defeature — original vs defeatured comparison (brepformer.visualize_defeature)
#
# All output is saved to: brepformer/pipeline_output/<filename>/
#
# Usage:
#   ./brepformer/scripts/inference_pipeline.sh /path/to/model.step
#   ./brepformer/scripts/inference_pipeline.sh /path/to/model.step --data_dir brepformer/data/defeature_processed
#   ./brepformer/scripts/inference_pipeline.sh /path/to/model.step --skip_viewers
#
# Prerequisites:
#   - conda env "brep_mfr" active (steps 1-4)
#   - conda env "new_brepmfr" available (steps 5-6, pythonocc >= 7.9)
#
# ============================================================================

set -eo pipefail

# ─── Initialize conda ──────────────────────────────────────────────────────
# conda activate/deactivate are shell functions that require initialization.
# Without this, they are unavailable in non-interactive bash scripts.

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook 2>/dev/null)" || {
        echo "ERROR: Cannot initialize conda. Make sure conda is installed and on PATH."
        exit 1
    }
fi

# ─── Configuration ──────────────────────────────────────────────────────────

CHECKPOINT="results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt"
CONDA_ENV_MAIN="brep_mfr"
CONDA_ENV_DEFEATURE="new_brepmfr"

# ─── Activate main conda env ───────────────────────────────────────────────

verify_conda_env() {
    local expected="$1"
    local actual="${CONDA_DEFAULT_ENV:-}"
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: Expected conda env '$expected' but got '${actual:-<none>}'."
        echo "       Active python: $(which python 2>/dev/null || echo 'not found')"
        echo "       Available envs:"
        conda env list
        exit 1
    fi
    echo "  Verified conda env: $actual (python: $(which python))"
}

conda activate "$CONDA_ENV_MAIN" || {
    echo "ERROR: Failed to activate conda env '$CONDA_ENV_MAIN'."
    echo "       Make sure it exists: conda env list"
    exit 1
}
verify_conda_env "$CONDA_ENV_MAIN"

# ─── Usage ──────────────────────────────────────────────────────────────────

show_usage() {
    cat <<'EOF'
BrepFormer Inference Pipeline

Usage:
  ./brepformer/scripts/inference_pipeline.sh <step_file> [options]

Arguments:
  <step_file>           Path to a STEP (.step/.stp) file

Options:
  --checkpoint PATH     Model checkpoint (default: results/trial1_ss/best-epoch=41-val/f1=0.9476.ckpt)
  --data_dir DIR        Preprocessed data directory for full analysis (enables all analyze modes)
  --skip_viewers        Skip interactive viewer steps (4 and 6) for headless/batch use
  --output_dir DIR      Override output directory (default: brepformer/pipeline_output/<filename>/)
  -h, --help            Show this help message

Steps:
  1. Inference         python -m brepformer.infer          → preds.json, preds.seg
  2. FreeCAD export    python -m brepformer.export_freecad → <name>_colored.step
  3. Analysis          python brepformer/analyze.py        → analysis/
  4. Visualize seg     python -m brepformer.visualize_seg  (interactive, skipped with --skip_viewers)
  --- conda switch: brepmfr → new_brepmfr ---
  5. Defeature v2      python -m brepformer.defeature_v2   → <name>_defeatured.step, <name>_report.json
  6. Visualize defeature  python -m brepformer.visualize_defeature (interactive, skipped with --skip_viewers)

Environment:
  Steps 1-4 run in conda env "brepmfr" (should be active when you run the script).
  Steps 5-6 run in conda env "new_brepmfr" (auto-switched by the script).

Examples:
  # Basic: run full pipeline on a single STEP file
  ./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step

  # With full analysis (requires preprocessed dataset)
  ./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step \
      --data_dir brepformer/data/defeature_processed

  # Headless (no interactive viewers)
  ./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step --skip_viewers

  # Custom checkpoint
  ./brepformer/scripts/inference_pipeline.sh /path/to/my_model.step \
      --checkpoint results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt

Output:
  brepformer/pipeline_output/<filename>/
  ├── preds.json                    # Full predictions (per-face probs, class names)
  ├── preds.seg                     # One label per line (for downstream tools)
  ├── <filename>_colored.step       # Colored STEP for FreeCAD
  ├── analysis/                     # Analysis outputs
  │   ├── architecture.json         # Model architecture details
  │   └── ...                       # (per_class_metrics, embeddings, etc. if --data_dir)
  ├── <filename>_defeatured.step    # Defeatured STEP (features removed)
  ├── <filename>_colored.step       # Colored STEP from defeature (predictions overlay)
  └── <filename>_report.json        # Defeaturing report (faces removed/failed/valid)
EOF
}

# ─── Parse arguments ────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    show_usage
    exit 1
fi

# First positional arg is the STEP file
STEP_FILE=""
DATA_DIR=""
SKIP_VIEWERS=false
OUTPUT_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --skip_viewers)
            SKIP_VIEWERS=true
            shift
            ;;
        --output_dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        -*)
            echo "ERROR: Unknown option '$1'"
            echo ""
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$STEP_FILE" ]]; then
                STEP_FILE="$1"
            else
                echo "ERROR: Unexpected argument '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$STEP_FILE" ]]; then
    echo "ERROR: No STEP file provided."
    echo ""
    show_usage
    exit 1
fi

if [[ ! -f "$STEP_FILE" ]]; then
    echo "ERROR: STEP file not found: $STEP_FILE"
    exit 1
fi

# ─── Resolve paths ──────────────────────────────────────────────────────────

# Get the project root (grandparent of brepformer/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Make STEP_FILE absolute
if [[ "$STEP_FILE" != /* ]]; then
    STEP_FILE="$(cd "$(dirname "$STEP_FILE")" && pwd)/$(basename "$STEP_FILE")"
fi

# Extract filename without extension
FILENAME=$(basename "$STEP_FILE")
FILENAME="${FILENAME%.*}"

# Output directory
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
    OUTPUT_DIR="$PROJECT_ROOT/brepformer/pipeline_output/${FILENAME}"
fi
mkdir -p "$OUTPUT_DIR"

# Make OUTPUT_DIR absolute
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

cd "$PROJECT_ROOT"

# Verify checkpoint exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "  (looked in $PROJECT_ROOT/$CHECKPOINT)"
    exit 1
fi

# ─── Banner ─────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  BrepFormer Inference Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo "  STEP file:    $STEP_FILE"
echo "  Filename:     $FILENAME"
echo "  Output:       $OUTPUT_DIR"
echo "  Checkpoint:   $CHECKPOINT"
echo "  Data dir:     ${DATA_DIR:-"(none — architecture analysis only)"}"
echo "  Skip viewers: $SKIP_VIEWERS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ─── Step 1: Inference ──────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/6] Inference (brepformer.infer)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  Generating predictions..."

# Save .json (full predictions with probabilities)
python -m brepformer.infer \
    --step "$STEP_FILE" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR/preds.json"

# Save .seg (one label per line, consumed by downstream tools)
python -m brepformer.infer \
    --step "$STEP_FILE" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR/preds.seg"

echo ""
echo "  ✓ $OUTPUT_DIR/preds.json"
echo "  ✓ $OUTPUT_DIR/preds.seg"
echo ""

# ─── Step 2: FreeCAD Export ─────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/6] FreeCAD Export (brepformer.export_freecad)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m brepformer.export_freecad \
    --step "$STEP_FILE" \
    --seg "$OUTPUT_DIR/preds.seg" \
    --output "$OUTPUT_DIR/${FILENAME}_colored.step"

echo ""
echo "  ✓ $OUTPUT_DIR/${FILENAME}_colored.step"
echo "  Open this file in FreeCAD to see per-face color predictions."
echo ""

# ─── Step 3: Analysis ──────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/6] Analysis (brepformer.analyze)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ -n "$DATA_DIR" ]]; then
    echo "  Running full analysis (--mode all) with data_dir=$DATA_DIR..."
    python brepformer/analyze.py \
        --checkpoint "$CHECKPOINT" \
        --data_dir "$DATA_DIR" \
        --mode all \
        --output_dir "$OUTPUT_DIR/analysis"
else
    echo "  Running architecture analysis (no --data_dir provided)..."
    echo "  Tip: pass --data_dir <preprocessed_dir> for full analysis"
    echo "       (per_class, embeddings, confusion_matrix, face_segmentation, etc.)"
    python brepformer/analyze.py \
        --checkpoint "$CHECKPOINT" \
        --mode architecture \
        --output_dir "$OUTPUT_DIR/analysis"
fi

echo ""
echo "  ✓ $OUTPUT_DIR/analysis/"
echo ""

# ─── Step 4: Visualize Segmentation ────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/6] Visualize Segmentation (brepformer.visualize_seg)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$SKIP_VIEWERS" == true ]]; then
    echo "  Skipped (--skip_viewers)"
else
    echo "  Launching interactive 3D viewer..."
    echo "  Controls: T=toggle mode, D/A=next/prev, I=info, S=summary, Q=quit"
    echo ""
    python -m brepformer.visualize_seg \
        --step "$STEP_FILE" \
        --seg "$OUTPUT_DIR/preds.seg"
fi

echo ""

# ─── Conda Environment Switch ──────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Switching conda environment: $CONDA_ENV_MAIN → $CONDA_ENV_DEFEATURE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# conda was already initialized at the top of the script
conda deactivate
conda activate "$CONDA_ENV_DEFEATURE" || {
    echo "ERROR: Failed to activate conda env '$CONDA_ENV_DEFEATURE'."
    echo "       Make sure it exists: conda env list"
    exit 1
}
verify_conda_env "$CONDA_ENV_DEFEATURE"

echo ""

# ─── Step 5: Defeature v2 ──────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/6] Defeature v2 (brepformer.defeature_v2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m brepformer.defeature_v2 \
    --step "$STEP_FILE" \
    --seg "$OUTPUT_DIR/preds.seg" \
    --output_dir "$OUTPUT_DIR" \
    --save_colored --verbose

echo ""
echo "  ✓ $OUTPUT_DIR/${FILENAME}_defeatured.step"
echo "  ✓ $OUTPUT_DIR/${FILENAME}_report.json"
echo ""

# ─── Step 6: Visualize Defeature ───────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [6/6] Visualize Defeature (brepformer.visualize_defeature)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$SKIP_VIEWERS" == true ]]; then
    echo "  Skipped (--skip_viewers)"
else
    echo "  Launching original vs defeatured comparison viewer..."
    echo "  Controls: T=toggle GT/Pred, D/A=next/prev, S=summary, Q=quit"
    echo ""
    python -m brepformer.visualize_defeature \
        --step "$STEP_FILE" \
        --defeatured "$OUTPUT_DIR/${FILENAME}_defeatured.step" \
        --seg "$OUTPUT_DIR/preds.seg"
fi

echo ""

# ─── Done ───────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  All output saved to:"
echo "    $OUTPUT_DIR/"
echo ""
echo "  Files:"
echo "    preds.json                     Full predictions (probs, class names)"
echo "    preds.seg                      Per-face labels (one per line)"
echo "    ${FILENAME}_colored.step       Colored STEP for FreeCAD"
echo "    analysis/                      Model analysis outputs"
echo "    ${FILENAME}_defeatured.step    Defeatured STEP (features removed)"
echo "    ${FILENAME}_report.json        Defeaturing report"
echo ""
echo "  To view in FreeCAD:"
echo "    Open ${FILENAME}_colored.step (original with predictions)"
echo "    Open ${FILENAME}_defeatured.step (clean stock shape)"
echo ""
echo "════════════════════════════════════════════════════════════════"
