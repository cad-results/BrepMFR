#!/bin/bash
# BrepMFR Viewer - Launch script with WSL2/X11 display setup
#
# Usage:
#   ./run_viewer.sh browse [--split test] [--sort index]
#   ./run_viewer.sh predictions [--sort worst]
#   ./run_viewer.sh analysis [--metrics] [--confusion] [--embeddings]
#   ./run_viewer.sh [any viewer.py args...]

set -euo pipefail

show_usage() {
    cat <<'EOF'
BrepMFR Viewer - Interactive MTFRCAD B-rep visualization

Usage:
  ./run_viewer.sh <mode> [options]

Modes:
  browse        Browse preprocessed data with GT labels
  predictions   Browse models sorted by prediction accuracy (Jaccard)
  analysis      Show analysis plots (no 3D viewer needed)

Browse options:
  --split train|val|test   Data split to browse (default: test)
  --sort index|best|worst|random  Sort order (default: index)
  --model_id ID            View a specific model

Predictions options:
  --sort index|best|worst|random  Sort order (default: index)

Analysis options:
  --metrics       Per-class metrics bar chart
  --confusion     Confusion matrix heatmap
  --embeddings    t-SNE embeddings scatter

Common options:
  --data_dir DIR        Raw MTFRCAD data directory (default: brepformer/data/mtfrcad)
  --processed_dir DIR   Preprocessed data directory (default: brepformer/data/mtfrcad_processed)
  --analysis_dir DIR    Analysis results directory (default: analysis_results)

Examples:
  ./run_viewer.sh browse --split test --sort index
  ./run_viewer.sh browse --split train
  ./run_viewer.sh predictions --sort worst
  ./run_viewer.sh predictions --sort best
  ./run_viewer.sh predictions --sort random
  ./run_viewer.sh analysis --metrics
  ./run_viewer.sh analysis --confusion
  ./run_viewer.sh analysis --embeddings

Keyboard controls (browse/predictions modes):
  T/TAB       Cycle: Plain -> GT -> Predicted -> Comparison
  D/RIGHT     Next model
  A/LEFT      Previous model
  1           Sort: worst accuracy first
  2           Sort: best accuracy first
  3           Sort: random shuffle
  M           Metrics chart popup
  N           Confusion matrix popup
  E           Embeddings plot popup
  I           Print model info
  L           Toggle class legend (3D swatches + sidebar panel)
  F           Toggle face labels (predicted feature name at each face)
  S           Screenshot
  R           Reset camera
  H           Help
  ESC/Q       Exit
EOF
}

# ─── Functions ───────────────────────────────────────────────────────────────

is_wsl() {
    [ -f /proc/version ] && grep -qi microsoft /proc/version
}

check_x_server() {
    xset -q &>/dev/null
    return $?
}

test_display() {
    local display=$1
    DISPLAY=$display timeout 1 xset -q &>/dev/null
    return $?
}

find_working_display() {
    local displays=(":0" ":0.0" "localhost:0" "localhost:0.0")

    if command -v ip &>/dev/null; then
        local host_ip=$(ip route show | grep -i default | awk '{ print $3 }')
        if [ -n "$host_ip" ]; then
            displays+=("${host_ip}:0" "${host_ip}:0.0")
        fi
    fi

    if [ -f /etc/resolv.conf ]; then
        local nameserver=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
        if [ -n "$nameserver" ]; then
            displays+=("${nameserver}:0" "${nameserver}:0.0")
        fi
    fi

    for display in "${displays[@]}"; do
        if test_display "$display"; then
            echo "$display"
            return 0
        fi
    done

    return 1
}

check_dependencies() {
    local missing=()

    if ! python3 -c "import open3d" 2>/dev/null; then
        missing+=("open3d")
    fi

    if ! python3 -c "import matplotlib" 2>/dev/null; then
        missing+=("matplotlib")
    fi

    if ! python3 -c "import numpy" 2>/dev/null; then
        missing+=("numpy")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: Missing Python packages: ${missing[*]}"
        echo "Install with: pip install ${missing[*]}"
        exit 1
    fi
}

setup_display() {
    # Determine if we need full display setup or just matplotlib backend
    local need_3d="$1"

    if [ "$need_3d" = "false" ]; then
        # Analysis-only mode: just configure matplotlib backend
        export MPLBACKEND=TkAgg
        return 0
    fi

    echo "Configuring display environment..."

    if is_wsl; then
        echo "Running in WSL2 environment"

        if [ -d "/mnt/wslg" ]; then
            echo "WSLg detected - forcing X11 mode for Open3D/GLFW"
            export DISPLAY=:0
            unset WAYLAND_DISPLAY
            export WAYLAND_DISPLAY=
            export GDK_BACKEND=x11
            export QT_QPA_PLATFORM=xcb
            export SDL_VIDEODRIVER=x11
            export PYOPENGL_PLATFORM=x11
            export MPLBACKEND=TkAgg
        else
            if [ -z "${DISPLAY:-}" ]; then
                echo "DISPLAY not set, searching for X server..."
                working_display=$(find_working_display)
                if [ -n "$working_display" ]; then
                    export DISPLAY="$working_display"
                    echo "Found working display: $DISPLAY"
                else
                    echo ""
                    echo "ERROR: No X server detected!"
                    echo ""
                    echo "For WSL2, you need one of:"
                    echo "  1. Windows 11 with WSLg (wsl --update)"
                    echo "  2. VcXsrv / X410 on Windows"
                    echo ""
                    echo "Or use analysis mode (no display needed):"
                    echo "  ./run_viewer.sh analysis --metrics"
                    exit 1
                fi
            fi
        fi

        # OpenGL software rendering
        export LIBGL_ALWAYS_SOFTWARE=1
        export MESA_GL_VERSION_OVERRIDE=3.3
        export MESA_GLSL_VERSION_OVERRIDE=330
        export GALLIUM_DRIVER=llvmpipe
        export LIBGL_ALWAYS_INDIRECT=0
        export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
        export __GLX_VENDOR_LIBRARY_NAME=mesa
        export LIBGL_DRI3_DISABLE=1
        export GLFW_IM_MODULE=
        export DISABLE_WAYLAND=1

        if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/mnt/wslg/runtime-dir" ]; then
            export XDG_RUNTIME_DIR="/mnt/wslg/runtime-dir"
        fi

    else
        # Native Linux
        if [ -z "${DISPLAY:-}" ]; then
            if [ -n "${WAYLAND_DISPLAY:-}" ]; then
                export DISPLAY=:0
            else
                export DISPLAY=:0
            fi
        fi
    fi

    echo "  DISPLAY=$DISPLAY"
    echo "  OpenGL: Software rendering (Mesa llvmpipe)"
}

# ─── Main ────────────────────────────────────────────────────────────────────

# Show help if no args or -h/--help
if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_usage
    exit 0
fi

# Determine mode for display setup
MODE="${1:-browse}"
NEED_3D="true"

case "$MODE" in
    browse|predictions)
        NEED_3D="true"
        # Convert mode arg to --mode flag
        shift
        VIEWER_ARGS="--mode $MODE $*"
        ;;
    analysis)
        NEED_3D="false"
        shift
        VIEWER_ARGS="--mode analysis $*"
        ;;
    *)
        # Pass everything through as-is (user used --mode directly)
        VIEWER_ARGS="$*"
        # Check if --mode analysis is in the args
        if echo "$VIEWER_ARGS" | grep -q -- "--mode analysis"; then
            NEED_3D="false"
        fi
        ;;
esac

echo "BrepMFR Viewer"
echo "----------------------------------------"

check_dependencies
setup_display "$NEED_3D"

echo ""
echo "Starting viewer..."
echo "----------------------------------------"
echo ""

# Run the viewer
exec python3 viewer.py $VIEWER_ARGS
