#!/bin/bash
# Run the BrepSimplifier primitive fitting pipeline.
#
# Two modes:
#   Direct:   Fit primitives to an already-defeatured STEP file.
#   Pipeline: Defeature a raw STEP file first, then fit primitives.
#
# Usage:
#   # Direct mode — input is already defeatured
#   ./brepsimplifier/run_simplifier.sh --step defeatured_model.step
#
#   # Pipeline mode — defeature first, then fit primitives
#   ./brepsimplifier/run_simplifier.sh --step raw_model.step --pipeline
#
#   # Pipeline with verbose output
#   ./brepsimplifier/run_simplifier.sh --step raw_model.step --pipeline --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

show_usage() {
    cat <<'EOF'
BrepSimplifier — Primitive Fitting Pipeline

Usage:
  ./brepsimplifier/run_simplifier.sh [options]

Modes:
  Direct (default):  --step takes a defeatured STEP file, fits primitives.
  Pipeline:          --step takes a raw STEP file; defeaturing runs first.
                     Enabled with --pipeline flag.

Options:
  --step FILE            Input STEP file (required)
  --output FILE          Output STEP path (direct mode)
  --output_dir DIR       Output directory (pipeline mode)
  --pipeline             Run full defeature → primitives pipeline
  --checkpoint PATH      Model checkpoint (pipeline mode)
  --seg FILE             Pre-computed .seg predictions (pipeline mode)
  --tol FLOAT            Geometric tolerance for face grouping (default: 0.1)
  --max_fuzzy FLOAT      Max fuzzy tolerance for defeaturing (default: 1e-3)
  --save_colored         Save colored STEP with predictions (pipeline mode)
  --save_intermediate    Keep defeatured STEP (pipeline mode)
  --verbose              Detailed progress output
EOF
}

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
exec python -m brepformer.brepsimplifier.primitives "$@"
