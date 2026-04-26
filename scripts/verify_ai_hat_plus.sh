#!/bin/bash
# Hardware verification sweep for Pi 5 + AI HAT+ (Hailo-8 or Hailo-8L).
#
# Vision-only — Hailo-8/8L cannot host LLMs (no onboard SDRAM and the
# silicon isn't a transformer accelerator). For the LLM-on-NPU sweep,
# use verify_ai_hat_plus_2.sh on the Hailo-10H board.
#
# Run once per Pi after `setup_rpi_ai_hat_plus.sh` has completed and the
# venv is activated:
#
#     source venv/bin/activate
#     ./scripts/verify_ai_hat_plus.sh
#
# Output lands in `results/hw_verify_<timestamp>/` with per-step logs,
# bench_*.json artefacts, and a final pass/fail summary. Exit code is 0
# when all blocking steps passed (experimental v26 failures don't gate).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hw_verify_common.sh
source "$SCRIPT_DIR/hw_verify_common.sh"

# 1 (sanity) + 9 (Phase 3 a/b/c × v8/v11/v26) + 1 (pytest) = 11 steps.
HW_TOTAL_STEPS=11

hw_init
hw_preflight_rpi_ai_hat_plus

# Mock-only smoke suite first. If pytest is broken, every later step
# is suspect; better to know immediately.
hw_run_step "pytest" "pytest tests/ -q"

# Detection sanity baseline (already verified pre-Phase-3, but run it
# anyway so the report includes a known-good comparison row).
hw_run_step "yolo-v8-detection" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolov8n.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task detection --backend hailo

# Phase 3a — OBB. Drone-priority since DOTA is aerial imagery.
hw_run_step "yolo-v8-obb" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolov8n-obb.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task obb --backend hailo
hw_run_step "yolo-v11-obb" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo11n-obb.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task obb --backend hailo
hw_run_step "yolo-v26-obb [experimental]" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo26n-obb.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task obb --backend hailo

# Phase 3b — Segmentation.
hw_run_step "yolo-v8-seg" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolov8n-seg.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task segmentation --backend hailo
hw_run_step "yolo-v11-seg" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo11n-seg.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task segmentation --backend hailo
hw_run_step "yolo-v26-seg [experimental]" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo26n-seg.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task segmentation --backend hailo

# Phase 3c — Pose.
hw_run_step "yolo-v8-pose" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolov8n-pose.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task pose --backend hailo
hw_run_step "yolo-v11-pose" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo11n-pose.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task pose --backend hailo
hw_run_step "yolo-v26-pose [experimental]" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo26n-pose.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task pose --backend hailo

hw_summary
