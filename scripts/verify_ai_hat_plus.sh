#!/bin/bash
# Hardware verification sweep for Pi 5 + AI HAT+ (Hailo-8 or Hailo-8L).
#
# Mirrors the AI HAT+ 2 runner shape so the two boards produce
# directly-comparable result bundles. Vision sweep is identical (10
# steps including pytest). The LLM-on-NPU step is included and *will*
# fail on Hailo-8/8L — no onboard SDRAM, not a transformer accelerator
# — but the Python runner emits a zero-valued unsupported-on-this-hw
# stub so the cross-platform dashboard gets an explicit "tried, 0 TPS"
# row instead of a missing one. The LLM-on-CPU comparison row works
# identically on both Pi 5 boards (same Cortex-A76 SoC).
#
# Run once per Pi after `setup_rpi_ai_hat_plus.sh` has completed and the
# venv is activated:
#
#     source venv/bin/activate
#     ./scripts/verify_ai_hat_plus.sh
#
# Output lands in `results/hw_verify_<timestamp>/` with per-step logs,
# bench_*.json artefacts, an auto-generated dashboard, and a final
# pass/fail summary. Exit code is 0 when all blocking steps passed —
# `[experimental]` and `[unsupported-on-this-hw]` failures don't gate.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hw_verify_common.sh
source "$SCRIPT_DIR/hw_verify_common.sh"

# 1 (sanity) + 10 (Phase 3 a/b/c × v8/v11/v26 + v8-pose-n) + 1 (pytest)
# + 1 (LLM-on-NPU stub) + 1 (LLM-on-CPU comparison row) = 14 steps.
HW_TOTAL_STEPS=14

hw_init rpi_ai_hat_plus
hw_ensure_python_deps
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

# Phase 3c — Pose. v8 runs both n (added in hefs-v3 — closes the
# previous Hailo Model Zoo gap on hailo8) and s (the historical step,
# kept so the multi-size v8-pose family produced in hefs-v3 has a
# smoke test beyond just the nano size). v11 / v26 stay at 'n' for
# symmetry with the AI HAT+ 2 script.
hw_run_step "yolo-v8-pose-n" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolov8n-pose.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task pose --backend hailo
hw_run_step "yolo-v8-pose" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolov8s-pose.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task pose --backend hailo
hw_run_step "yolo-v11-pose" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo11n-pose.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task pose --backend hailo
hw_run_step "yolo-v26-pose [experimental]" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolo26n-pose.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task pose --backend hailo

# LLM-on-NPU. Hailo-8/8L can't host LLMs, so this step always fails on
# AI HAT+ — by design. The Python runner emits an unsupported-on-this-hw
# stub LLMResult (backend=hailo-10h, tps=0) so the dashboard renders an
# explicit zero bar for cross-platform comparison rather than a missing
# row. We deliberately don't gate on a curl preflight to :8000 here —
# we want the stub written regardless of hailo-ollama's reachability.
hw_run_step "llm-npu-llama3.2:1b [unsupported-on-this-hw]" \
    "python -m benchmark run llm --profile npu --output $HW_RESULTS_DIR" \
    --workload llm --backend hailo-10h --require-npu-metrics

# CPU-side comparison row at the same 1B model size as the AI HAT+ 2 Pi's
# `npu` step + step 12's stub above — so the two boards' "CPU LLM" rows
# in the cross-platform dashboard are at the same model size and directly
# comparable. Drone profile (llama2:7b) was tried 2026-05-01: per-request
# wall time ran ~40 min at max_tokens=256, projecting ~26 hours for the
# full 5-prompt × 13-run sweep. See
# resources/session_notes_2026-05-02_llm_drone_profile_unworkable.md.
hw_run_step "llm-cpu-llama3.2:1b (compare profile)" \
    "python -m benchmark run llm --profile compare --output $HW_RESULTS_DIR" \
    --workload llm --backend ollama-cpu

hw_finalize_with_report
hw_summary
