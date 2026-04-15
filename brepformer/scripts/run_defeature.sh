#!/bin/bash
# Run the BrepFormer defeaturing pipeline (v1 or v2).
#
# Usage:
#   ./scripts/run_defeature.sh --step model.step
#   ./scripts/run_defeature.sh --step_dir steps/ --verbose
#   ./scripts/run_defeature.sh --step model.step --engine v2 --max_fuzzy 0.01
#   ./scripts/run_defeature.sh --step model.step --engine v1
#
# The --engine flag selects which defeaturing engine to use:
#   v2 (default) — Modern engine with adaptive tolerance, history-based face
#                  tracking, pre-validation, enhanced healing, and area-based
#                  ordering. Requires pythonocc >= 7.9.
#   v1           — Original 5-phase progressive engine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_usage() {
    cat <<'EOF'
BrepFormer Defeaturing Pipeline

Usage:
  ./scripts/run_defeature.sh [--engine v1|v2] [defeature args...]

Engine selection:
  --engine v2    (default) Modern engine — adaptive fuzzy tolerance,
                 history-based face tracking, pre-validation, enhanced
                 healing, area-based ordering.
  --engine v1    Original 5-phase progressive engine.

Examples:

  # Single file (v2 engine, default checkpoint)
  ./scripts/run_defeature.sh --step model.step

  # Single file, verbose output
  ./scripts/run_defeature.sh --step model.step --verbose

  # Batch directory
  ./scripts/run_defeature.sh --step_dir steps/

  # Custom checkpoint and output
  ./scripts/run_defeature.sh --step model.step \
      --checkpoint results/my_model/best.ckpt \
      --output_dir my_output/

  # From pre-computed .seg predictions (skip inference)
  ./scripts/run_defeature.sh --step model.step --seg preds.seg

  # Batch with pre-computed predictions
  ./scripts/run_defeature.sh --step_dir steps/ --seg_dir inference/trial1/

  # Save colored STEP for visual comparison
  ./scripts/run_defeature.sh --step model.step --save_colored --verbose

  # Use v1 engine explicitly
  ./scripts/run_defeature.sh --engine v1 --step model.step

  # v2 with higher fuzzy tolerance for difficult models
  ./scripts/run_defeature.sh --step model.step --max_fuzzy 0.01 --verbose

Defeature arguments (passed through to the engine):
  --step FILE          Single STEP file to defeature
  --step_dir DIR       Directory of STEP files (batch mode)
  --checkpoint PATH    Model checkpoint (default: auto-detected)
  --seg FILE           Pre-computed .seg predictions (single mode)
  --seg_dir DIR        Pre-computed .seg directory (batch mode)
  --output_dir DIR     Output directory (default: brepformer/defeatured_output)
  --save_colored       Also save colored STEP with predictions
  --verbose            Detailed phase-by-phase progress

v2-only arguments:
  --max_fuzzy FLOAT    Max fuzzy tolerance for adaptive retry (default: 1e-3)

Output:
  {output_dir}/{model}_defeatured.step   Defeatured STEP file
  {output_dir}/{model}_colored.step      Colored predictions (if --save_colored)
  {output_dir}/{model}_report.json       Per-model report
  {output_dir}/batch_report.json         Batch summary (batch mode)
EOF
}

# ─── Parse --engine flag (extract it, pass rest through) ─────────────────────

ENGINE="v2"
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --engine)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --engine requires a value (v1 or v2)"
                exit 1
            fi
            ENGINE="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate engine choice
case "$ENGINE" in
    v1|v2) ;;
    *)
        echo "ERROR: Unknown engine '$ENGINE'. Use v1 or v2."
        exit 1
        ;;
esac

# Check we have something to do
if [[ ${#PASSTHROUGH_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: No arguments provided. Use --step or --step_dir."
    echo ""
    show_usage
    exit 1
fi

# ─── Select module ───────────────────────────────────────────────────────────

if [[ "$ENGINE" == "v2" ]]; then
    MODULE="brepformer.defeature_v2"
    echo "Engine: v2 (modern — adaptive tolerance, history tracking, enhanced healing)"
else
    MODULE="brepformer.defeature"
    echo "Engine: v1 (original — 5-phase progressive)"
fi

echo "─────────────────────────────────────────"
echo ""

# ─── Run ─────────────────────────────────────────────────────────────────────

cd "$PROJECT_ROOT"
exec python -m "$MODULE" "${PASSTHROUGH_ARGS[@]}"
