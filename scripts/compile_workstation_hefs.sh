#!/bin/bash
# Workstation HEF compilation driver.
#
# Loops the standard "gap models" the Hailo Model Zoo does not publish
# prebuilt HEFs for, and compiles each via `python -m benchmark compile`
# for the selected target architecture(s). Continue-on-failure: a single
# broken model does not abort the rest of the sweep, and the summary at
# the end shows pass/fail per model.
#
# Designed for x86_64 Linux workstations with the Hailo SDK installed
# (hailo_dataflow_compiler-*.whl + hailo_model_zoo-*.whl). The Pi cannot
# run this script — the Dataflow Compiler is x86_64 only.
#
# Usage:
#   scripts/compile_workstation_hefs.sh [--arch hailo8|hailo8l|hailo10h|both]
#                                       [--include-detection]
#                                       [--models a.pt,b.pt,...]
#                                       [--input-resolution N]
#                                       [--calibration-set-size N]
#                                       [--output-dir resources/hefs]
#                                       [--force-recompile]
#
# Examples:
#   # Default: compile the seven gap models for hailo8 (AI HAT / AI HAT+ 26 TOPS)
#   scripts/compile_workstation_hefs.sh --arch hailo8
#
#   # Compile gap models for both AI HAT+ and AI HAT+ 2 in one sweep
#   scripts/compile_workstation_hefs.sh --arch both
#
#   # Also compile detection at sizes n/s (useful for AI HAT+ 2 since no
#   # hailo10h detection HEFs ship in the Zoo today)
#   scripts/compile_workstation_hefs.sh --arch hailo10h --include-detection
#
#   # Override the model list entirely
#   scripts/compile_workstation_hefs.sh --arch hailo10h \
#       --models yolov8n-seg.pt,yolo11n-pose.pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# --- Parse args -----------------------------------------------------------
ARCH="hailo8"
INCLUDE_DETECTION=0
MODELS_OVERRIDE=""
INPUT_RESOLUTION=640
CALIBRATION_SET_SIZE=100
OUTPUT_DIR="$REPO_ROOT/resources/hefs"
FORCE_RECOMPILE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)
            ARCH="$2"; shift 2 ;;
        --include-detection)
            INCLUDE_DETECTION=1; shift ;;
        --models)
            MODELS_OVERRIDE="$2"; shift 2 ;;
        --input-resolution)
            INPUT_RESOLUTION="$2"; shift 2 ;;
        --calibration-set-size)
            CALIBRATION_SET_SIZE="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --force-recompile)
            FORCE_RECOMPILE=1; shift ;;
        -h|--help)
            sed -n '2,32p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            error "Unknown argument: $1"
            error "Use --help to see options."
            exit 2 ;;
    esac
done

case "$ARCH" in
    hailo8|hailo8l|hailo10h|both) ;;
    *)
        error "Invalid --arch '$ARCH'. Choose hailo8, hailo8l, hailo10h, or both."
        exit 2 ;;
esac

# --- Pre-flight checks ----------------------------------------------------
if ! is_x86_64; then
    error "HEF compilation requires x86_64 Linux. Detected: $(uname -m)."
    error "The Hailo Dataflow Compiler does not run on aarch64."
    exit 3
fi

VENV_PY="$REPO_ROOT/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    error "Project venv not found at $REPO_ROOT/venv/."
    error "Run 'python3 -m venv venv && source venv/bin/activate && pip install -e .[dev]' first."
    exit 4
fi

info "Probing for hailo_sdk_client..."
if ! "$VENV_PY" -c "import hailo_sdk_client" 2>/dev/null; then
    error "hailo_sdk_client is not installed in the venv."
    error "Install the Hailo Dataflow Compiler wheel (.whl) from the"
    error "Hailo Developer Zone (https://hailo.ai/developer-zone/) into"
    error "$REPO_ROOT/venv/. See docs/hef_compilation.md."
    exit 5
fi
success "hailo_sdk_client is importable."

# --- Model selection ------------------------------------------------------
# The seven "gap models" the Hailo Model Zoo does not publish prebuilt
# HEFs for, ordered by likelihood of clean compile per
# docs/hef_compilation.md.
GAP_MODELS=(
    "yolo11n-seg.pt"
    "yolo11n-pose.pt"
    "yolov8n-obb.pt"
    "yolo11n-obb.pt"
    "yolo26n-obb.pt"
    "yolo26n-seg.pt"
    "yolo26n-pose.pt"
)

# Detection at sizes n/s — only relevant for hailo10h today (Zoo has
# detection prebuilts for hailo8/8l). Toggled by --include-detection.
DETECTION_MODELS=(
    "yolov8n.pt"
    "yolov8s.pt"
    "yolo11n.pt"
    "yolo26n.pt"
)

declare -a MODELS
if [[ -n "$MODELS_OVERRIDE" ]]; then
    IFS=',' read -ra MODELS <<< "$MODELS_OVERRIDE"
else
    MODELS=("${GAP_MODELS[@]}")
    if [[ "$INCLUDE_DETECTION" -eq 1 ]]; then
        MODELS+=("${DETECTION_MODELS[@]}")
    fi
fi

if [[ "$ARCH" == "both" ]]; then
    ARCHES=("hailo8" "hailo10h")
else
    ARCHES=("$ARCH")
fi

# --- Compile loop ---------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

declare -a PASS_LIST
declare -a FAIL_LIST

run_compile() {
    local target="$1"
    local model="$2"
    local extra_flags=""
    if [[ "$FORCE_RECOMPILE" -eq 1 ]]; then
        extra_flags="--force-recompile"
    fi

    info "Compiling $model -> $target"
    if "$VENV_PY" -m benchmark compile \
        --hw-arch "$target" \
        --model "$model" \
        --input-resolution "$INPUT_RESOLUTION" \
        --calibration-set-size "$CALIBRATION_SET_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        $extra_flags; then
        success "  PASS  $target / $model"
        PASS_LIST+=("$target / $model")
    else
        warn "  FAIL  $target / $model (exit $?)"
        FAIL_LIST+=("$target / $model")
    fi
}

START_TS=$(date +%s)
info "Sweep starting: ${#MODELS[@]} model(s) x ${#ARCHES[@]} arch(es)"
info "Output: $OUTPUT_DIR"
echo ""

for target in "${ARCHES[@]}"; do
    for model in "${MODELS[@]}"; do
        run_compile "$target" "$model" || true
    done
done

END_TS=$(date +%s)
ELAPSED_S=$((END_TS - START_TS))

# --- Summary --------------------------------------------------------------
echo ""
echo "================================================================"
echo "Workstation compile summary  (elapsed: ${ELAPSED_S}s)"
echo "================================================================"
echo "Passed (${#PASS_LIST[@]}):"
for m in "${PASS_LIST[@]:-}"; do
    [[ -n "$m" ]] && echo "  PASS  $m"
done
echo ""
echo "Failed (${#FAIL_LIST[@]}):"
for m in "${FAIL_LIST[@]:-}"; do
    [[ -n "$m" ]] && echo "  FAIL  $m"
done
echo ""
info "Staged HEFs are in: $OUTPUT_DIR"
info "Commit them and pull on the Pi to run verify with the new coverage."

if [[ "${#FAIL_LIST[@]}" -gt 0 ]]; then
    exit 1
fi
exit 0
