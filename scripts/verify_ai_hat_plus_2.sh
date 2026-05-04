#!/bin/bash
# Hardware verification sweep for Pi 5 + AI HAT+ 2 (Hailo-10H, HailoRT 5.x).
#
# Same vision sweep as verify_ai_hat_plus.sh PLUS the LLM-on-NPU run via
# HailoRT GenAI (`hailo-ollama` REST endpoint at :8000). Auto-triggers
# `python -m benchmark report` at the end so the cross-backend dashboard
# is ready immediately — the AI HAT+ 2 is the comparison-friendly board.
#
# Prerequisites (run once before this script):
#
#   1. setup_rpi_ai_hat_plus_2.sh --with-genai (drops the EULA-gated
#      hailo_gen_ai_model_zoo_<ver>_arm64.deb at the repo root first)
#   2. source .cache/hailo-apps/setup_env.sh
#   3. hailo-ollama &      # GenAI REST server bound to :8000
#   4. curl -sS http://localhost:8000/api/pull \
#        -H 'Content-Type: application/json' \
#        -d '{"model":"llama3.2:1b","stream":true}'
#   5. source venv/bin/activate
#
# Then:
#
#   ./scripts/verify_ai_hat_plus_2.sh
#
# Output lands in `results/hw_verify_<timestamp>/` plus a generated
# dashboard in the same directory. Exit code is 0 when all blocking
# steps passed (experimental v26 failures don't gate).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hw_verify_common.sh
source "$SCRIPT_DIR/hw_verify_common.sh"

# 1 (sanity) + 10 (Phase 3 a/b/c × v8/v11/v26 + v8-pose-n) + 1 (pytest)
# + 1 (LLM-on-NPU) + 1 (LLM-on-CPU comparison row) = 14 steps.
HW_TOTAL_STEPS=14

hw_init rpi_ai_hat_plus_2
hw_ensure_python_deps
hw_preflight_rpi_ai_hat_plus_2

# Mock-only smoke suite first.
hw_run_step "pytest" "pytest tests/ -q"

# YOLO sweep — same shape as the AI HAT+ runner. The Hailo-10H has
# different SDK + HEFs but the postprocessor + whitelist contracts
# are identical from the runner's perspective.
hw_run_step "yolo-v8-detection" \
    "python -m benchmark run yolo --backend hailo --yolo-model yolov8n.pt --output $HW_RESULTS_DIR" \
    --workload yolo --task detection --backend hailo

# Phase 3a — OBB.
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
# previous Hailo Model Zoo gap) and s (the historical step, kept so
# the multi-size v8-pose family produced in hefs-v3 has a smoke test
# beyond just the nano size). v11 / v26 stay at 'n' for symmetry.
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

# Phase 2 — LLM-on-NPU. Skip with a clear note if hailo-ollama isn't up;
# the runner's platform precondition would error-but-exit-0 anyway, but
# a friendly skip is more helpful.
if curl -sS --max-time 3 http://localhost:8000/api/tags >/dev/null 2>&1; then
    hw_run_step "llm-npu-llama3.2:1b" \
        "python -m benchmark run llm --profile npu --output $HW_RESULTS_DIR" \
        --workload llm --backend hailo-10h --require-npu-metrics
else
    hw_skip "llm-npu-llama3.2:1b" "hailo-ollama not reachable on :8000"
fi

# CPU-side comparison row so the dashboard has something to split on.
# Uses the `compare` profile (llama3.2:1b on Ollama CPU + drone prompts)
# to mirror the npu profile exactly, giving a true apples-to-apples
# 1B-vs-1B cross-backend comparison row. The standalone `drone` profile
# uses llama2:7b which needs ~5.5 GB RAM at runtime and won't fit on a
# 4 GB Pi 5; `compare` is the RAM-safe alternative.
hw_run_step "llm-cpu-llama3.2:1b (drone prompts)" \
    "python -m benchmark run llm --profile compare --output $HW_RESULTS_DIR" \
    --workload llm --backend ollama-cpu

hw_finalize_with_report
hw_summary
