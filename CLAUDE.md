# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchy is an edge AI benchmarking suite that evaluates YOLO (computer vision) and LLM (via Ollama) inference across three hardware platforms: NVIDIA Jetson Orin Nano (GPU), Raspberry Pi + AI HAT+ (Hailo-8L 13 TOPS or Hailo-8 26 TOPS, both HailoRT 4.x), and Raspberry Pi + AI HAT+ 2 (Hailo-10H NPU, HailoRT 5.x). It collects latency, throughput, accuracy, resource utilization, and power metrics.

## Running Benchmarks

Activate the project venv (created by the setup scripts in `scripts/`) before running anything: `source venv/bin/activate` from the repo root. Both `python -m benchmark <cmd>` and the installed `edge-bench <cmd>` console script (from `setup.py` entry point) invoke the same CLI.

```bash
# Quick benchmark (yolov8n detection + llama2:7b)
python -m benchmark run all

# Full profile (all YOLO versions/tasks/sizes + all LLM model groups)
python -m benchmark run all --profile full

# Single workload
python -m benchmark run yolo
python -m benchmark run llm

# Specific YOLO model or version
python -m benchmark run yolo --yolo-model yolo26n.pt
python -m benchmark run yolo --yolo-version v11 --backend hailo

# Force Hailo model recompile (ignores ~/.cache/benchy/hailo/)
python -m benchmark run yolo --backend hailo --force-recompile

# Drone-full sweep (det+obb+seg+pose at 1280, sizes n/s — Hailo only)
python -m benchmark run yolo --profile drone_full

# LLM only — llama-only policy (1B / 3B / 7B on CPU; 1B-only on NPU).
# Default runs llama2:7b (bare tag, no quant sweep); full runs
# llama3.2:1b + llama3.2:3b + llama2:7b across the 1B/3B/7B groups in
# configs/llm_benchmark.yaml; drone runs llama2:7b on the curated drone
# prompt set; npu runs llama3.2:1b on the Hailo-10H via hailo-ollama
# (only llama with a prebuilt HEF in the HailoRT 5.3.0 GenAI Model Zoo).
python -m benchmark run llm
python -m benchmark run llm --profile full
python -m benchmark run llm --profile drone

# LLM on Hailo-10H NPU (AI HAT+ 2 only; needs hailo-ollama on :8000)
python -m benchmark run llm --profile npu

# Diagnostic subcommands
python -m benchmark info            # detected platform + system info
python -m benchmark list-models     # YOLO models supported per backend
python -m benchmark backends        # available inference backends

# Override platform autodetect (e.g. on dev hosts where hailortcli probing fails)
python -m benchmark run all --platform rpi_ai_hat_plus_2

# Reporting pipeline
python -m benchmark aggregate
python -m benchmark dashboard
python -m benchmark report          # aggregate + dashboard combined
python -m benchmark verify results/bench_*.json  # cross-platform comparison of per-run JSONs
```

`--skip-validation` disables YOLO mAP/precision/recall validation (faster runs).

Per-run JSON/CSV land in `results/`; HW-verify bundles land in `results/hw_verify_<timestamp>/` with `report/index.html` for the dashboard.

## Development Commands

Activate the project venv first (`source venv/bin/activate`); on Raspberry Pi OS Bookworm system-wide pip is blocked by PEP 668, so the venv is mandatory there. Python >= 3.10 is required (`setup.py:python_requires`).

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Linting / typing (configured in setup.py extras_require)
black benchmark/
mypy benchmark/

# Tests — pytest smoke suite under tests/. Covers schemas, platform
# detection, profile parsing, all five YOLO postprocessors (det / cls /
# obb / seg / pose), rotated NMS / mask blender / pose decoder math,
# dashboard backend axis, calibration defaults, the HW-verify validator,
# LLM quant tag templating, NPU profile gating, the LLM-side NPU
# collector, and YOLO model-name parsing. Mocks platform probes so it
# runs on dev hosts too.
pytest tests/
pytest tests/test_platform_detection.py                                         # single file
pytest tests/test_platform_detection.py::test_returns_platform_enum_on_real_host  # single test
pytest tests/ -k "obb or seg or pose"                                           # rotated/seg/pose postprocessor tests
pytest tests/test_dashboard_backend_filter.py                                   # backend axis filter
```

## Architecture

**Entry point**: `benchmark/__main__.py` → `benchmark/cli.py:main()` (argparse-based CLI with subcommands: run, info, aggregate, dashboard, report, list-models, backends, verify).

**Data flow**: CLI → workload runners → metrics collectors → result writers → aggregation → dashboard HTML.

**Platform detection**: `benchmark/metrics/collectors.py:detect_platform()` identifies the platform by checking `/etc/nv_tegra_release` (Jetson), `/proc/device-tree/model` (RPi), and `hailortcli fw-control identify` output (Hailo-8L vs Hailo-10H/8 → distinguishes AI HAT+ from AI HAT+ 2). Override with `--platform`.

**YOLO model name parsing**: `benchmark/cli.py:_infer_yolo_model_info()` infers version and task from the filename — pattern `yolov8` / `yolo11` / `yolo26` + optional suffix (`-seg`, `-pose`, `-obb`, `-cls`; no suffix = detection). This means `--yolo-model` accepts any standard Ultralytics filename without extra metadata.

Key modules:

- **`benchmark/schemas.py`** — All dataclasses and enums (Platform, YOLOTask, Backend, SystemInfo, BenchmarkRun, result types). `LLMResult` carries the backend axis (`backend`, `npu_utilization_percent`, `npu_power_watts`, `hailort_version`).
- **`benchmark/cli.py`** — CLI routing and orchestration. `cmd_benchmark()` loads YAML configs from `configs/`, dispatches to workload runners, writes results. `run_llm_benchmark()` reads profile-level `api_base` / `backend` / `npu_metrics` and gates the `npu` profile on `Platform.RPI_AI_HAT_PLUS_2`.
- **`benchmark/workloads/yolo/runner.py`** — YOLO benchmark runner using Ultralytics. `YOLO_MODELS` dict maps version→task→model files; `DEFAULT_DATASETS` carries the runtime validation YAML per task.
- **`benchmark/workloads/yolo/backends/`** — Backend abstraction: `base.py` (interface), `pytorch.py` (CPU/GPU), `hailo.py` (NPU; `SUPPORTED_TASKS` lists all five YOLO tasks), `registry.py` (auto-selection based on detected hardware).
- **`benchmark/workloads/yolo/conversion/`** — Hailo model pipeline: `.pt → .onnx → .har → .hef`. `pipeline.py` orchestrates the stages (`onnx_export.py` → `har_generator.py` → `hef_compiler.py`); `cache.py` manages `~/.cache/benchy/hailo/` and `validation.py` sanity-checks the produced artefacts. `calibration.py:DEFAULT_DATASETS` swaps to full DOTAv1 / coco-pose for OBB / pose calibration.
- **`benchmark/workloads/yolo/execution.py`** — Hailo execution enforcement and supported configuration validation. `HAILO_SUPPORTED_TASKS` and `HAILO_OPTIMIZED_MODELS` whitelist all five tasks across v8/v11/v26.
- **`benchmark/workloads/yolo/postprocessing.py`** — Per-task post-processing for Hailo outputs: `Detection` / `_process_detection` (axis-aligned NMS), `OrientedBox` / `_process_obb` / `_rotated_iou` / `_rotated_nms` (Sutherland-Hodgman polygon clipping), `SegmentationResult` / `_process_segmentation` / `_generate_seg_masks` (sigmoid mask blender), `PoseResult` / `_process_pose` (17-keypoint COCO-Pose decoder), `ClassificationResult`. `DOTA_CLASSES` and `COCO_POSE_KEYPOINTS` constants.
- **`benchmark/workloads/llm/runner.py`** — LLM benchmark runner. Calls Ollama-compatible HTTP API (default `localhost:11434`; `localhost:8000` when `--profile npu`). `LLMBenchmarkConfig.backend` / `.npu_metrics` flags drive backend tagging and the NPU collector. `PromptSet` carries DRONE_PROMPTS plus the legacy / lightweight sets.
- **`benchmark/workloads/{yolo,llm}/hailo_metrics.py`** — NPU metric collection on a background thread (`hailortcli`-derived utilisation + power); mirrors `ResourceMonitor`'s start/stop/snapshot API. The YOLO copy is wired into the inference loop; the LLM copy (`HailoLLMMetricsCollector`) is gated by the `npu_metrics` profile flag.
- **`benchmark/backends/hailo_utils.py`** — Shared Hailo probes used by both YOLO and LLM paths: `get_hailort_version`, `get_sdk_family`, `get_power_watts`, `get_npu_utilization_percent` (currently a documented stub).
- **`benchmark/metrics/collectors.py`** — Platform detection, CPU/GPU/NPU utilization, memory, power monitoring.
- **`benchmark/results/writers.py`** — JSON and CSV output writers; CSV columns include the backend axis.
- **`benchmark/aggregation/aggregator.py`** — Multi-run result aggregation with group-safe comparisons. `LLMAggregatedMetrics` carries the backend axis; default `aggregate_llm_results` group_by includes `backend` so CPU and NPU runs of the same model+prompt stay separate rows. `aggregation/csv_writer.py` writes the aggregated CSV (separate from the per-run JSON/CSV in `results/writers.py`).
- **`benchmark/reporting/dashboard.py`** — Generates self-contained HTML dashboard with charts and filters. Includes a `Backend` filter chip + column with friendly labels (`Ollama (CPU)`, `Hailo-10H (NPU)`).
- **`benchmark/verification.py`** — Cross-platform result comparison and fairness validation.

## Configuration

YAML configs in `configs/` define profiles, model lists, inference params, and prompt sets. Profile fields: `input_resolution`, per-task `datasets:`, `prompt_set`, `quants` + `quant_tag_template`, and (LLM-only) `api_base`, `backend`, `npu_metrics`. The LLM backend axis (`ollama-cpu` / `ollama-cuda` / `hailo-10h`) is recorded on every `LLMResult` and surfaced as a dashboard filter chip + column. Built-in profiles (`default` / `full` / `drone` / `drone_full` / `npu` / `compare`) are summarised in the README; see `docs/workloads.md` and `docs/output.md` for field-level reference, and `docs/hailo.md` for the NPU metric path (`benchmark/workloads/{yolo,llm}/hailo_metrics.py` + `benchmark/backends/hailo_utils.py`).

### LLM model policy

Llama-only, one model per size group, since 2026-04-27 (see `resources/session_notes_2026-04-27.md` for the rationale — chat-quant sweep filled the SD card mid-pull). The CPU and NPU sides are constrained differently — CPU is curated to fit the SD card; NPU is constrained by Hailo's prebuilt HEF catalogue:

| Group | Tag | CPU (Ollama) | NPU (Hailo-10H HEF) |
|---|---|---|---|
| 1B | `llama3.2:1b` | ✓ | ✓ (only llama HEF in HailoRT 5.3.0 zoo; `npu` profile target as of 2026-04-28) |
| 3B | `llama3.2:3b` | ✓ | — (had a HEF in 5.1.1; Hailo dropped it in 5.3.0) |
| 7B | `llama2:7b` | ✓ | — (no 7B HEF has ever shipped in any HailoRT GenAI release) |

**CPU side:** Do not add qwen / mistral / gemma / dolphin / olmo / granite / starcoder / sailor models or 1.5B / 8B / 9B groups to `configs/llm_benchmark.yaml` without explicit user direction; the CPU side is intentionally narrow to fit an SD-card-backed Pi 5. Quant sweeps are off by default; re-add `quants:` + `quant_tag_template:` to a profile to opt back in (`docs/workloads.md` has the example). Runtime metadata for the dropped families (e.g. MoE/Code badges in `benchmark/reporting/dashboard.py`, `MODEL_METADATA` in `benchmark/workloads/llm/runner.py`) is preserved in code, not removed — re-introducing those models doesn't require re-deriving the metadata.

**NPU side:** Constrained by what Hailo prebuilds in the HailoRT GenAI Model Zoo (currently 5.3.0); see `tests/test_llm_npu_profile.py:HAILO_GENAI_PREBUILT_HEFS` for the verified-against-server inventory. The 5.3.0 zoo ships exactly one llama (`llama3.2:1b`) plus four qwen variants and one deepseek, all in the 1B-1.7B range. The npu profile sticks to the llama for cross-backend symmetry with the CPU 1B group; non-llama HEFs (qwen, deepseek) are inventoried in the test whitelist but not exercised by the npu profile. Cross-backend comparison is therefore only at the 1B level — the 3B and 7B llama groups are CPU-only by virtue of Hailo not shipping HEFs at those sizes.

## Hardware Verification

`scripts/verify_ai_hat_plus.sh` and `scripts/verify_ai_hat_plus_2.sh` are smart runners that sweep every Hailo-supported YOLO task on the respective board. Both source `scripts/hw_verify_common.sh` (built on top of `scripts/common.sh` for log primitives), call `python -m benchmark run …` per step, and validate the produced `bench_*.json` against per-task contracts via `scripts/hw_verify_validators.py`. Continue-on-failure semantics; the final exit code is non-zero only on blocking failures — anything tagged `[experimental]` (v26-{obb,seg,pose}) or `[unsupported-on-this-hw]` (NPU LLM on Hailo-8/8L) is treated as advisory and recorded but doesn't gate exit. Both runners produce identical 13-step bundles (vision sweep + LLM-on-NPU + LLM-on-CPU comparison row) plus an auto-generated dashboard at `results/hw_verify_<timestamp>/report/` via `hw_finalize_with_report`, so the two boards' `results/hw_verify_<timestamp>/` directories are directly diff-able. The validator is the test surface (`tests/test_hw_verify_validators.py` + `tests/test_llm_npu_unsupported_stub.py`) since bash orchestration is hard to unit-test on dev.

When the `npu` profile runs on a non-`rpi_ai_hat_plus_2` platform, `cli.py:_build_unsupported_npu_stubs()` emits zero-valued `LLMResult` rows tagged `backend="hailo-10h"` and `prompt_id="unsupported-on-this-hardware"` instead of returning an empty list. This guarantees the cross-platform dashboard renders an explicit "tried, 0 TPS" bar on the NPU axis for the AI HAT+ Pi, matching the row count of the AI HAT+ 2 Pi for chart comparison.

## Hardware-Specific Notes

- Hailo backend supports all five YOLO tasks: detection, classification, OBB, segmentation, pose. OBB uses `postprocessing.py:_rotated_nms` / `_process_obb`; segmentation uses `_process_segmentation` / `_generate_seg_masks`; pose uses `_process_pose` with a 17-keypoint COCO-Pose decoder. v8/v11 are verified across all tasks; v26-{obb,seg,pose} are marked experimental until hardware confirmation.
- Hailo models use INT8 quantization; first compilation per model takes 5-30 minutes.
- **Benchmark integrity rule**: CPU inference on Hailo-equipped platforms is NOT a valid configuration. The Hailo backend deliberately does NOT fall back to CPU — do not add such a fallback, and do not suggest running YOLO without `--backend hailo` (or auto-selection) on RPi + AI HAT+. See `docs/hailo.md`.
- LLM benchmarks require an Ollama-compatible server: `ollama serve` on `:11434` for CPU profiles (`default` / `full` / `drone` / `compare`), or `hailo-ollama` on `:8000` for the `npu` profile (Hailo-10H only). The `api_base` profile field selects which. The `drone` profile uses `llama2:7b` and requires an 8 GB Pi 5; on 4 GB Pi 5s use the `compare` profile (1B equivalent) instead — it's also what `verify_ai_hat_plus_2.sh` uses for the CPU comparison row to mirror the `npu` profile.
- `resources/hailo-8/` and `resources/hailo-10H/` bundle HailoRT `.deb` installers + user guides for the two AI HAT+ generations. The 4.x / 4.23.0 stack for AI HAT+ (Hailo-8/8L) is the apt-installed default that `scripts/setup_rpi_ai_hat_plus.sh` configures. The 5.2.0 `.deb` for AI HAT+ 2 (Hailo-10H) is **userspace only** — verified 2026-04-28 that `scripts/setup_rpi_ai_hat_plus_2.sh` cannot complete the 5.x install on Pi OS Bookworm because the RPi apt repo has no `hailo-h10-all` package and the script never falls back to the bundled `.deb`. The full HailoRT 5.x install (driver + firmware + userspace + Python wheel + GenAI model zoo) must be downloaded manually from the Hailo Developer Zone — see `docs/hailo.md` § "LLM on Hailo-10H → Setup (high level)" for the procedure. `scripts/setup_jetson_orin_nano.sh` is the symmetric setup script for the Jetson platform.
- Setup scripts emit `*.log` files (e.g. `hailort.log`, `setup_rpi_ai_hat_plus_2.log`) to the repo root. They are not in `.gitignore` but are runtime artefacts, not sources — safe to delete.
- `resources/` (checked in) ships with the product and is consumed by setup scripts; `documents/` (gitignored) holds internal PRDs and phase-scoping notes that are not part of the runtime — don't treat anything under `documents/` as authoritative for code behaviour, and don't recreate it if it's missing locally.

## Additional Documentation

`docs/` contains the authoritative reference for CLI flags (`cli.md`), workload metrics and model groups (`workloads.md`), Hailo integration and limits (`hailo.md`), benchmark methodology and reproducibility verification (`methodology.md`), output file layout and dashboard filters (`output.md`), and common failures (`troubleshooting.md`). Consult these before duplicating information here.

## Git Conventions

- Never add Co-Authored-By, Signed-off-by, or any Claude attribution to commits.
