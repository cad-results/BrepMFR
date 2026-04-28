#!/bin/bash
# Run the BrepFormer primitive fitting pipeline.
#
# Two modes:
#   Direct:   Fit primitives to an already-defeatured STEP file.
#   Pipeline: Defeature a raw STEP file first, then fit primitives.
#
# Usage:
#   # Direct mode — input is already defeatured
#   ./scripts/run_primitives.sh --step defeatured_model.step
#
#   # Pipeline mode — defeature first, then fit primitives
#   ./scripts/run_primitives.sh --step raw_model.step --pipeline
#
#   # Pipeline with pre-computed predictions
#   ./scripts/run_primitives.sh --step raw_model.step --pipeline --seg preds.seg
#
#   # Pipeline with custom checkpoint and verbose output
#   ./scripts/run_primitives.sh --step raw_model.step --pipeline \
#       --checkpoint results/my_model/best.ckpt --verbose
#
#   # Save all intermediate outputs
#   ./scripts/run_primitives.sh --step raw_model.step --pipeline \
#       --save_intermediate --save_colored --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_usage() {
    cat <<'EOF'
BrepFormer Primitive Fitting Pipeline

Usage:
  ./scripts/run_primitives.sh [options]

Modes:
  Direct (default):  --step takes a defeatured STEP file, fits primitives.
  Pipeline:          --step takes a raw STEP file; defeaturing runs first.
                     Enabled with --pipeline flag.

Options:
  --step FILE            Input STEP file (required)
  --output FILE          Output STEP path (direct mode)
  --output_dir DIR       Output directory (pipeline mode, default: brepformer/primitives_output)
  --pipeline             Run full defeature → primitives pipeline
  --checkpoint PATH      Model checkpoint (pipeline mode)
  --seg FILE             Pre-computed .seg predictions (pipeline mode)
  --tol FLOAT            Geometric tolerance for face grouping (default: 0.1)
  --max_fuzzy FLOAT      Max fuzzy tolerance for defeaturing (default: 1e-3)
  --lenient              Tessellate BSpline faces into small prisms (for originals)
  --save_colored         Save colored STEP with predictions (pipeline mode)
  --save_intermediate    Keep defeatured STEP (pipeline mode)
  --verbose              Detailed progress output

Output:
  {stem}_primitives.step        Primitive-fitted STEP file
  {stem}_defeatured.step        Defeatured STEP (--pipeline --save_intermediate)
  {stem}_colored.step           Colored predictions (--pipeline --save_colored)
  {stem}_primitives_report.json Combined report (--pipeline)
EOF
}

# Check for help flag
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            show_usage
            exit 0
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "ERROR: No arguments provided."
    echo ""
    show_usage
    exit 1
fi

cd "$PROJECT_ROOT"
exec python -m brepformer.primitives "$@"
