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
#                                       [--venv PATH | $BENCHY_VENV]
#                                       [--force-recompile]
#
# The Hailo Dataflow Compiler wheels target Python 3.10 / 3.11 only, so
# the compile venv often needs to be separate from the project's main
# venv (which may be a newer Python). Pass --venv venv-compile (or set
# BENCHY_VENV=venv-compile in the environment) to point the script at
# whichever venv has hailo_sdk_client installed.
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
#
#   # Use a separate compile venv (Python 3.11 with the Hailo SDK)
#   scripts/compile_workstation_hefs.sh --venv venv-compile --arch both
#   # Or via env var
#   BENCHY_VENV=venv-compile scripts/compile_workstation_hefs.sh --arch both
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
# Hailo's INT8 bias-correction passes (Bias Correction / Adaround /
# Finetune encoding) require >= 1024 calibration samples; below that
# the optimizer drops to level 0 and biases stay 16-bit, which fails
# chip mapping on Hailo-8 for some seg/pose models with
# "DW resources calculation failed for 16bit L2 biases / 16x4 not
# supported in activation2". Mirror the CLI default added in cli.py.
CALIBRATION_SET_SIZE=1024
OUTPUT_DIR="$REPO_ROOT/resources/hefs"
FORCE_RECOMPILE=0
VENV_DIR="${BENCHY_VENV:-$REPO_ROOT/venv}"

# Helper: assert a value-taking flag actually has a following value.
# Without this, `set -u` turns "--output-dir" with nothing after it
# into an opaque "$2: unbound variable" — easy to hit when a fish-shell
# line-continuation backslash is missing.
require_value() {
    local flag="$1"
    local count="$2"
    if [[ "$count" -lt 2 ]]; then
        error "Missing value for $flag"
        error "Use --help to see options."
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)
            require_value "$1" "$#"; ARCH="$2"; shift 2 ;;
        --include-detection)
            INCLUDE_DETECTION=1; shift ;;
        --models)
            require_value "$1" "$#"; MODELS_OVERRIDE="$2"; shift 2 ;;
        --input-resolution)
            require_value "$1" "$#"; INPUT_RESOLUTION="$2"; shift 2 ;;
        --calibration-set-size)
            require_value "$1" "$#"; CALIBRATION_SET_SIZE="$2"; shift 2 ;;
        --output-dir)
            require_value "$1" "$#"; OUTPUT_DIR="$2"; shift 2 ;;
        --venv)
            require_value "$1" "$#"; VENV_DIR="$2"; shift 2 ;;
        --force-recompile)
            FORCE_RECOMPILE=1; shift ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            error "Unknown argument: $1"
            error "Use --help to see options."
            exit 2 ;;
    esac
done

# If VENV_DIR is relative, resolve it against the repo root so the
# script doesn't depend on the caller's CWD.
case "$VENV_DIR" in
    /*) ;;
    *)  VENV_DIR="$REPO_ROOT/$VENV_DIR" ;;
esac

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

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    error "Compile venv not found at $VENV_DIR/."
    error "Create it with one of:"
    error "  python3 -m venv venv && source venv/bin/activate && pip install -e .[dev]"
    error "  python3.11 -m venv venv-compile && --venv venv-compile (Python 3.10/3.11 needed for Hailo SDK)"
    error "Or set BENCHY_VENV=<path> to point at an existing venv."
    exit 4
fi

info "Using compile venv: $VENV_DIR"
info "  Python: $("$VENV_PY" --version 2>&1)"
info "Probing for hailo_sdk_client..."
if ! "$VENV_PY" -c "import hailo_sdk_client" 2>/dev/null; then
    error "hailo_sdk_client is not installed in $VENV_DIR/."
    error "Install the Hailo Dataflow Compiler wheel (.whl) from the"
    error "Hailo Developer Zone (https://hailo.ai/developer-zone/):"
    error "  $VENV_PY -m pip install /path/to/hailo_dataflow_compiler-*.whl"
    error "  $VENV_PY -m pip install /path/to/hailo_model_zoo-*.whl"
    error "See docs/hef_compilation.md."
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

# Initialize as empty arrays (not just declared) so `${#PASS_LIST[@]}`
# and `${#FAIL_LIST[@]}` work cleanly under `set -u`. On bash without
# this, accessing an empty array element trips "unbound variable".
declare -a PASS_LIST=()
declare -a FAIL_LIST=()

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
