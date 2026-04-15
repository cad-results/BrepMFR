#!/usr/bin/env bash
# run_viewer.sh — launch brepclassifier viewer scripts with WSL2-compatible display.
#
# Usage:
#   ./brepclassifier/run_viewer.sh browse [--split test] [--sort worst] [...]
#   ./brepclassifier/run_viewer.sh predictions [--sort worst] [...]
#   ./brepclassifier/run_viewer.sh analysis [--metrics] [--confusion] [...]
#   ./brepclassifier/run_viewer.sh seg --step model.step --gt_class 0
#   ./brepclassifier/run_viewer.sh seg --step_dir brepclassifier/data/ssdata1/steps/ \
#       --labels_json brepclassifier/data/ssdata1/labels.json

set -euo pipefail

# ---- WSL2 software rendering ----
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# Resolve script location so this works regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  browse       Open3D point cloud viewer (browse mode)"
    echo "  predictions  Open3D point cloud viewer (predictions mode)"
    echo "  analysis     Open3D point cloud viewer (analysis mode)"
    echo "  seg          Qt + pythonOCC STEP geometry viewer (whole-model)"
    echo "  faceseg      Qt + pythonOCC per-face segmentation viewer"
    echo "  freecad      Export colored STEP files for FreeCAD"
    echo ""
    echo "Examples:"
    echo "  $0 browse --split test"
    echo "  $0 predictions --sort worst"
    echo "  $0 analysis --metrics --confusion"
    echo "  $0 seg --step model.step --gt_class 0"
    echo "  $0 seg --step_dir brepclassifier/data/ssdata1/steps/ \\"
    echo "         --labels_json brepclassifier/data/ssdata1/labels.json"
    echo "  $0 faceseg --step model.step \\"
    echo "         --checkpoint results/face_seg_heavy/best.ckpt"
    echo "  $0 freecad --step model.step \\"
    echo "         --pipe_checkpoint results/pipe_classifier/best.ckpt \\"
    echo "         --output colored.step"
    exit 1
fi

COMMAND="$1"
shift

case "${COMMAND}" in
    browse)
        exec python brepclassifier/viewer.py --mode browse "$@"
        ;;
    predictions)
        exec python brepclassifier/viewer.py --mode predictions "$@"
        ;;
    analysis)
        exec python brepclassifier/viewer.py --mode analysis "$@"
        ;;
    seg)
        exec python brepclassifier/visualize_seg.py "$@"
        ;;
    faceseg)
        exec python brepclassifier/visualize_face_seg.py "$@"
        ;;
    freecad)
        exec python brepclassifier/export_freecad.py "$@"
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        echo "Valid commands: browse, predictions, analysis, seg, faceseg, freecad"
        exit 1
        ;;
esac
