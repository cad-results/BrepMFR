#!/bin/bash
# View STEP files using the pythonOCC Qt viewer.
#
# Compares original / defeatured / primitive-fitted STEP files side-by-side.
# Uses software rendering for WSL2 compatibility.
#
# Usage:
#   # View a single STEP file
#   ./brepformer/brepsimplifier/view_step.sh model.step
#
#   # Compare original vs. defeatured vs. primitives
#   ./brepformer/brepsimplifier/view_step.sh --original raw.step --defeatured defeatured.step --primitives primitives.step
#
#   # Batch view all results in test_data/results
#   ./brepformer/brepsimplifier/view_step.sh --dir brepformer/brepsimplifier/test_data/results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# --- WSL2 / display detection ---

is_wsl() {
    [ -f /proc/version ] && grep -qi microsoft /proc/version
}

find_working_display() {
    local displays=(":0" ":0.0" "localhost:0" "localhost:0.0")
    if command -v ip &>/dev/null; then
        local host_ip=$(ip route show | grep -i default | awk '{ print $3 }')
        [ -n "$host_ip" ] && displays+=("${host_ip}:0" "${host_ip}:0.0")
    fi
    if [ -f /etc/resolv.conf ]; then
        local ns=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
        [ -n "$ns" ] && displays+=("${ns}:0" "${ns}:0.0")
    fi
    for d in "${displays[@]}"; do
        DISPLAY=$d timeout 1 xset -q &>/dev/null && echo "$d" && return 0
    done
    return 1
}

echo "BrepSimplifier STEP Viewer — initializing display..."

if is_wsl; then
    if [ -d "/mnt/wslg" ]; then
        export DISPLAY=:0
        unset WAYLAND_DISPLAY; export WAYLAND_DISPLAY=
        export GDK_BACKEND=x11 QT_QPA_PLATFORM=xcb SDL_VIDEODRIVER=x11 PYOPENGL_PLATFORM=x11
    elif [ -z "${DISPLAY:-}" ]; then
        wd=$(find_working_display) && export DISPLAY="$wd" || {
            echo "ERROR: No X server. Install WSLg or VcXsrv."; exit 1; }
    fi
fi

export LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330 GALLIUM_DRIVER=llvmpipe
is_wsl && {
    export LIBGL_ALWAYS_INDIRECT=0 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe __GLX_VENDOR_LIBRARY_NAME=mesa
    export QT_QPA_PLATFORM=xcb GDK_BACKEND=x11 LIBGL_DRI3_DISABLE=1
    [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/mnt/wslg/runtime-dir" ] && export XDG_RUNTIME_DIR="/mnt/wslg/runtime-dir"
}

# --- Dependency check ---

if ! python3 -c "from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox" 2>/dev/null; then
    echo "ERROR: pythonocc-core required.  conda install -c conda-forge pythonocc-core"
    exit 1
fi

# --- Dispatch ---

show_usage() {
    cat <<'EOF'
BrepSimplifier STEP Viewer

Usage:
  view_step.sh <file.step>                           View a single STEP file
  view_step.sh --original O --defeatured D           Compare two stages
  view_step.sh --original O --defeatured D --prim P  Compare three stages
  view_step.sh --dir <results_dir>                   Browse a results directory

The viewer uses the brepformer.visualize_defeature Qt/pythonOCC viewer.
EOF
}

# Simple single-file mode
if [[ $# -eq 1 && ! "$1" =~ ^-- ]]; then
    echo "Viewing $1 ..."
    cd "$PROJECT_ROOT"
    exec python3 -m brepformer.visualize_defeature --step "$1"
fi

# Parse arguments
ORIGINAL="" DEFEATURED="" PRIMITIVES="" DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --original|-o)  ORIGINAL="$2"; shift 2 ;;
        --defeatured|-d) DEFEATURED="$2"; shift 2 ;;
        --primitives|--prim|-p) PRIMITIVES="$2"; shift 2 ;;
        --dir)          DIR="$2"; shift 2 ;;
        -h|--help)      show_usage; exit 0 ;;
        *)              echo "Unknown arg: $1"; show_usage; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

if [ -n "$DIR" ]; then
    echo "Browsing STEP files in $DIR ..."
    exec python3 -m brepformer.visualize_defeature --step_dir "$DIR"
fi

if [ -n "$ORIGINAL" ] && [ -n "$DEFEATURED" ]; then
    args=(--step "$ORIGINAL" --defeatured "$DEFEATURED")
    [ -n "$PRIMITIVES" ] && echo "Note: primitives file ignored for dual-pane view."
    echo "Comparing $ORIGINAL ↔ $DEFEATURED ..."
    exec python3 -m brepformer.visualize_defeature "${args[@]}"
fi

if [ -n "$ORIGINAL" ]; then
    echo "Viewing $ORIGINAL ..."
    exec python3 -m brepformer.visualize_defeature --step "$ORIGINAL"
fi

show_usage
exit 1
