# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchy is an edge AI benchmarking suite that evaluates YOLO (computer vision) and LLM (via Ollama) inference across three hardware platforms: NVIDIA Jetson Orin Nano (GPU), Raspberry Pi + AI HAT+ (Hailo-8L 13 TOPS or Hailo-8 26 TOPS, both HailoRT 4.x), and Raspberry Pi + AI HAT+ 2 (Hailo-10H NPU, HailoRT 5.x).

It collects latency, throughput, accuracy, resource utilization, and power metrics.

## Running Benchmarks

Platform setup is one of three scripts in `scripts/`: `setup_jetson_orin_nano.sh`, `setup_rpi_ai_hat_plus.sh` (Hailo-8 / 8L, HailoRT 4.x), or `setup_rpi_ai_hat_plus_2.sh` (Hailo-10H, HailoRT 5.x). Each installs platform-specific system packages and creates the project venv. Activate it before running anything: `source venv/bin/activate` from the repo root. Both `python -m benchmark <cmd>` and the installed `edge-bench <cmd>` console script (from `setup.py` entry point) invoke the same CLI.

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

# Drone-focused YOLO profiles: `drone` is detection at 1280 across
# v8/v11/v26 sizes n/s/m on VisDrone; `drone_full` extends to det + OBB
# + seg + pose at 1280 (YOLO-only — no LLM step).
python -m benchmark run yolo --profile drone
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

# Diagnostic subcommands — run `info` first on a new host to confirm
# platform detection without invoking a benchmark.
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

# Workstation HEF compilation (x86_64 only) — bypasses the runtime
# backend so a box without HailoRT or a Hailo device can still produce
# HEFs and stage them into resources/hefs/ with canonical naming. Gap
# models (seg/pose/OBB) require an NVIDIA GPU on the compile box —
# AMD/CPU-only fails chip mapping with 16-bit-bias errors at the seg
# head. See docs/compilation/nvidia_workstation_setup.md for the
# full bring-up checklist.
python -m benchmark compile --hw-arch hailo10h --model yolov8n-seg.pt
python -m benchmark compile --hw-arch hailo8 --models yolov8n-obb.pt,yolo11n-seg.pt
# --calibration-data-path stages val2017 (1 GB) instead of triggering
# Ultralytics' full coco auto-download (~27 GB). The CLI default is
# now --calibration-set-size 1024 to enable Hailo's bias-correction
# passes; below 1024 the optimizer drops to level 0 and biases stay
# 16-bit, which fails chip mapping on Hailo-8 for seg/pose/OBB.
python -m benchmark compile --hw-arch hailo8 --model yolo11n-seg.pt \
    --calibration-data-path ~/Documents/datasets/coco-val/images/val2017
scripts/compile_workstation_hefs.sh --arch both           # gap-model sweep
scripts/compile_workstation_hefs.sh --arch hailo10h --include-detection
scripts/fetch_prebuilt_hefs.py --arch both --dry-run      # Hailo Model Zoo S3 fetcher
```

`--skip-validation` disables YOLO mAP/precision/recall validation (faster runs).

Per-run JSON/CSV land in `results/`; HW-verify bundles land in `results/<platform>/hw_verify_<timestamp>/` (e.g. `results/rpi_ai_hat_plus/hw_verify_<ts>/`, `results/rpi_ai_hat_plus_2/hw_verify_<ts>/`) with `report/*_dashboard.html` for the dashboard. Per-board scoping keeps the two Pis' bundles from intermingling when both push to the same repo — see `scripts/hw_verify_common.sh:hw_init` for how the platform name is threaded in.

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

**Entry point**: `benchmark/__main__.py` → `benchmark/cli.py:main()` (argparse-based CLI with subcommands: run, info, aggregate, dashboard, report, list-models, backends, verify, compile).

**Data flow**: CLI → workload runners → metrics collectors → result writers → aggregation → dashboard HTML.

**Platform detection**: `benchmark/metrics/collectors.py:detect_platform()` identifies the platform by checking `/etc/nv_tegra_release` (Jetson), `/proc/device-tree/model` (RPi), and `hailortcli fw-control identify` output. Hailo-8L (13 TOPS) and Hailo-8 (26 TOPS) both map to `Platform.RPI_AI_HAT_PLUS`; only Hailo-10H maps to `Platform.RPI_AI_HAT_PLUS_2`. Override with `--platform`.

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

**CPU side:** Do not add qwen / mistral / gemma / dolphin / olmo / granite / starcoder / sailor models or 1.5B / 8B / 9B groups to `configs/llm_benchmark.yaml` without explicit user direction; the CPU side is intentionally narrow to fit an SD-card-backed Pi 5. Quant sweeps are off by default; re-add `quants:` + `quant_tag_template:` to a profile to opt back in (`docs/workloads.md` has the example). Runtime metadata for the dropped families (e.g. MoE/Code badges in `benchmark/reporting/dashboard.py`, `MODEL_METADATA` in `benchmark/workloads/llm/runner.py`) is preserved in code, not removed — re-introducing those models doesn't require re-deriving the metadata. Adding any non-llama model is a cross-cutting change: update Issue 7 in `resources/session_issues_2026-04-27.md`, the YAML, the runner constants, and the docs together.

**NPU side:** Constrained by what Hailo prebuilds in the HailoRT GenAI Model Zoo (currently 5.3.0); see `tests/test_llm_npu_profile.py:HAILO_GENAI_PREBUILT_HEFS` for the verified-against-server inventory. The 5.3.0 zoo ships exactly one llama (`llama3.2:1b`) plus four qwen variants and one deepseek, all in the 1B-1.7B range. The npu profile sticks to the llama for cross-backend symmetry with the CPU 1B group; non-llama HEFs (qwen, deepseek) are inventoried in the test whitelist but not exercised by the npu profile. Cross-backend comparison is therefore only at the 1B level — the 3B and 7B llama groups are CPU-only by virtue of Hailo not shipping HEFs at those sizes.

## Hardware Verification

`scripts/verify_ai_hat_plus.sh` and `scripts/verify_ai_hat_plus_2.sh` are smart runners that sweep every Hailo-supported YOLO task on the respective board. Both source `scripts/hw_verify_common.sh` (built on top of `scripts/common.sh` for log primitives), call `python -m benchmark run …` per step, and validate the produced `bench_*.json` against per-task contracts via `scripts/hw_verify_validators.py`. Continue-on-failure semantics; the final exit code is non-zero only on blocking failures — anything tagged `[experimental]` (v26-{obb,seg,pose}) or `[unsupported-on-this-hw]` (NPU LLM on Hailo-8/8L) is treated as advisory and recorded but doesn't gate exit. Both runners produce 13-step bundles with the same structural shape (vision sweep + LLM-on-NPU + LLM-on-CPU comparison row) plus an auto-generated dashboard at `results/<platform>/hw_verify_<timestamp>/report/` via `hw_finalize_with_report`, so the two boards' bundle dirs (under `results/rpi_ai_hat_plus/` vs `results/rpi_ai_hat_plus_2/`) are directly diff-able. Both runners use `--profile compare` (llama3.2:1b) for the CPU LLM step so the two boards' "CPU LLM" rows are at the same 1B size as the AI HAT+ 2 Pi's `npu` step (and as this Pi's `[unsupported-on-this-hw]` stub) for true cross-backend comparison. The drone profile (llama2:7b) was tried on the AI HAT+ Pi 2026-05-01 and projected ~26h wall time at `max_tokens=256` — see `resources/session_notes_2026-05-02_llm_drone_profile_unworkable.md`. The validator is the test surface (`tests/test_hw_verify_validators.py` + `tests/test_llm_npu_unsupported_stub.py`) since bash orchestration is hard to unit-test on dev.

`hw_verify_common.sh:hw_ensure_python_deps` is a verify-time self-heal step: it `import`-probes `onnx` and `onnxruntime` against the active venv and pip-installs anything missing before the sweep starts. Both verify entrypoints call it right after `hw_init`. This means an existing venv that pre-dates a dep change doesn't need a sudo-driven setup re-run — just running the verify script fixes itself. New deps that fall into the same trap (silently swallowed `ModuleNotFoundError` deep in the runner) should be added to the `pairs=()` array in this function — see Issue 8 in `resources/session_issues_2026-04-27.md` for the failure mode that motivated it.

When the `npu` profile runs on a non-`rpi_ai_hat_plus_2` platform, `cli.py:_build_unsupported_npu_stubs()` emits zero-valued `LLMResult` rows tagged `backend="hailo-10h"` and `prompt_id="unsupported-on-this-hardware"` instead of returning an empty list. This guarantees the cross-platform dashboard renders an explicit "tried, 0 TPS" bar on the NPU axis for the AI HAT+ Pi, matching the row count of the AI HAT+ 2 Pi for chart comparison.

## Critical Rules

**Benchmark integrity — no CPU fallback on Hailo platforms.** CPU inference on Hailo-equipped platforms is NOT a valid configuration. The Hailo backend deliberately does NOT fall back to CPU — do not add such a fallback, and do not suggest running YOLO without `--backend hailo` (or auto-selection) on RPi + AI HAT+. See `docs/hailo.md`.

## Hardware-Specific Notes

- Hailo backend supports all five YOLO tasks: detection, classification, OBB, segmentation, pose. OBB uses `postprocessing.py:_rotated_nms` / `_process_obb`; segmentation uses `_process_segmentation` / `_generate_seg_masks`; pose uses `_process_pose` with a 17-keypoint COCO-Pose decoder. v8/v11 are verified across all tasks; v26-{obb,seg,pose} are marked experimental until hardware confirmation.
- Hailo models use INT8 quantization; first compilation per model takes 5-30 minutes.
- LLM benchmarks require an Ollama-compatible server: `ollama serve` on `:11434` for CPU profiles (`default` / `full` / `drone` / `compare`), or `hailo-ollama` on `:8000` for the `npu` profile (Hailo-10H only). The `api_base` profile field selects which. The `drone` profile uses `llama2:7b` and requires an 8 GB Pi 5; on 4 GB Pi 5s use the `compare` profile (1B equivalent) instead — it's also what `verify_ai_hat_plus_2.sh` uses for the CPU comparison row to mirror the `npu` profile.
- The Hailo conversion pipeline (`.pt → .onnx → .har → .hef`) requires `onnx` + `onnxruntime`. Both are pinned in `setup.py:install_requires` and explicitly listed in the platform setup scripts; `hw_verify_common.sh:hw_ensure_python_deps` self-heals existing venvs that pre-date the pin. Don't move them out of `install_requires` — silently shipping a venv where `onnx_export.py` can't import is exactly Issue 8.
- **HEF compilation is x86_64-only** (Hailo Dataflow Compiler / `hailo_sdk_client` does not run on aarch64). On the Pi the backend therefore looks for prebuilt HEFs in two locations before attempting compilation: `resources/hefs/` (mix of Hailo Model Zoo v2.16.0 prebuilts and workstation-compiled HEFs, named `<version>_<task>_<size>_<arch>.hef` per `resources/hefs/NAMING.txt`; 35 currently staged via `hefs-v3` — 20 hailo8 covering v8 det n/s/m/l/x, v11 det n/s/m, v8 seg n/s/m, v8 pose n/s/m, v11 pose n, v8 obb n, v11 obb n, v26 det n, v26 obb n, v26 pose n; and 15 hailo10h covering v8 det n/s, v11 det/seg/pose at n, v8 pose n/s/m/l, v8 obb n, v11 obb n, v26 obb/seg/pose at n) and `/usr/share/hailo-models/` (the curated subset shipped by `rpicam-apps-hailo-postprocess`). See `benchmark/workloads/yolo/conversion/hef_source.py` and Issue 11 in `resources/session_issues_2026-04-27.md`. To extend coverage of the system-package map, only add entries that physically exist on disk — `tests/test_hef_source.py:test_system_package_map_only_contains_observed_filenames` will fail otherwise.
- **Workstation compile flow.** `python -m benchmark compile --hw-arch <hailo8|hailo8l|hailo10h> --model <X.pt>` (or `--models a.pt,b.pt,...`) runs the conversion pipeline directly via `ModelConversionPipeline` and stages the result into `resources/hefs/` with the canonical filename. It bypasses the runtime Hailo backend (no HailoRT / Hailo device required), which the legacy `run yolo --backend hailo --skip-validation` recipe does not — that path is gated by `HailoBackend.is_available()` and never actually worked on a typical compile box. `scripts/compile_workstation_hefs.sh` is the batch driver for the seven gap models per arch (or both archs). `scripts/fetch_prebuilt_hefs.py` pulls what the Hailo Model Zoo publishes (detection / seg / pose) directly from the S3 catalogue documented in `resources/hefs/NAMING.txt` — defensive against 404s, supports `--arch both` and `--dry-run`. See `docs/hef_compilation.md` and `tests/{test_compile_cmd.py,test_fetch_prebuilt_hefs.py}`.
- **Per-arch compile venvs.** DFC 3.33.x (Hailo-8 / 8L) and DFC 5.3.x (Hailo-10H) both ship the top-level `hailo_sdk_client` package, so they cannot coexist in one venv — installing both clobbers whichever line was installed first and you'd silently produce HEFs the other Pi can't load. The pattern is one venv per arch: `venv-compile-h8` for the 4.x line, `venv-compile-h10h` for the 5.x line. `scripts/compile_workstation_hefs.sh` defaults to `./venv` but accepts `--venv <path>` or `BENCHY_VENV=<path>` to point at the right one (`BENCHY_VENV=venv-compile-h8 scripts/compile_workstation_hefs.sh --arch hailo8`). The EULA-gated DFC + Model Zoo wheels live in `resources/hailo-sdk/` (gitignored as `*.whl` since each developer pulls their own copy under their Hailo Developer Zone entitlement); `resources/hailo-sdk/README.md` is the per-arch bring-up checklist.
- **Workstation compile cache layout.** `models/hailo/<arch>/<version>/<task>/<model>/{model.onnx,model.har,model.hef,metadata.json}`. `<arch>` is the load-bearing first level — same source `.pt` produces different bytes per arch (hailo8 vs hailo10h), so the cache must isolate them. Every `ModelCache` method (`get_*_path`, `has_*`, `get_metadata`, `save_metadata`, `clear_cache`, `has_valid_cache`) takes `target_device` as a required keyword-only arg; `tests/test_cache_arch_isolation.py` is the regression. Pre-2026-04-30 caches under `models/hailo/<version>/...` are silently dead (cache miss → recompile).
- **Workstation compile path internals.** End-node truncation is mandatory for YOLO heads — Hailo's parser can't ingest the DFL/Reshape/Cos tail, so `benchmark/workloads/yolo/conversion/har_generator.py:END_NODE_TABLE` keys `(yolo_version, YOLOTask)` to the per-task end-node lists. **The cut must be at the raw `cv*.X.X.2/Conv` outputs, matching `hailo_model_zoo/cfg/networks/<name>.yaml` `parser.nodes`** — not at the deeper post-processing layers (Sigmoid / Concat / Mul). Cutting deeper passes HAR generation but pulls high-precision-bias activations onto the chip subgraph, which fails Hailo-8 mapping with `16x4 not supported in activation*`. For OBB, truncate before `/model.X/Cos` and `/model.X/Sin` (the angle decoders). For v26, head conv names are prefixed `one2one_cv*` (one-to-one matching head); v26 pose specifically uses a flatter `one2one_cv4_kpts.X/Conv` for the keypoint branch. Verified end-to-end on the 2026-04-29 NVIDIA bring-up + 2026-04-30 retry sweep; see `resources/session_notes_2026-04-29_nvidia_workstation.md` Issue 6 + "Retry sweep". Currently populated (11 of 15 HAILO_SUPPORTED combos): v8/v11/v26 det + obb, v8/v11/v26 seg, v11/v26 pose. Remaining gaps (`v8 pose` + classification across all versions) rely on the `parse_end_node_hint` fallback, which often suggests the wrong (deep) cut — adding new entries by hand is preferred. `pipeline.py:_run_har_generation` populates `HARGeneratorConfig.end_nodes` from the table. Calibration data is stacked to `(N, H, W, C)` numpy before `runner.optimize` (a list of `(H, W, C)` arrays trips Hailo's type detector with "Couldn't detect CalibrationDataType"). The compile CLI exposes `--calibration-set-size` (default 1024 — Hailo's threshold for Bias Correction passes), `--calibration-data-path` (stage val2017 / ~1 GB instead of letting the loader auto-download all of coco / ~27 GB), and `--compression-level` (default 1 — 8-bit biases via Bias Correction; 0 fails seg/pose/OBB chip mapping; 2 adds Adaround + Finetune at the cost of compile time). `hef_compiler.py` emits an ALLS model script (`model_optimization_flavor(...)` + `post_quantization_optimization(bias_correction, policy=enabled)`) before `runner.optimize` — the explicit bias_correction force is necessary because the SDK's optimization-level cases are mutually exclusive (level 2 picks Finetune but skips Bias Correction without it). Per-(version, task) ALLS overrides are appended via `MODEL_SCRIPT_OVERRIDES` (also in `hef_compiler.py`) — currently used to apply the official `yolo26n.alls` per-layer `precision_mode=a16_w16` overrides for v26 detection on Hailo-8 (`dw1..8`, `conv61/77/91/64/80/94`, `output_layer1..6`); the table is keyed on `(version, task)` because the overrides are usually no-ops on the chip that doesn't need them. **Gap-model compiles (seg/pose/OBB) require an NVIDIA GPU** AND the matching `tensorflow[and-cuda]` / `jax[cuda12]` extras — the DFC wheel does not bundle a complete CUDA runtime; without those extras the optimizer logs `[warning] no available GPU` even on an NVIDIA box and chip mapping fails the same way it does on AMD/CPU-only. The CUDA-extras install also breaks torch's NCCL ABI; restore via `pip install --force-reinstall --no-deps nvidia-nccl-cu13`. See `docs/compilation/nvidia_workstation_setup.md` for the validated bring-up procedure (steps 5b/5c) and `docs/compilation/pitfalls.md` § 10-11 for end-node-depth rules and the `yolo11n-seg / hailo8` capacity miss.
- LLM cold-load can exceed the per-request HTTP timeout on slow storage. `LLMBenchmarkRunner._prewarm_model()` issues a single `keep_alive=-1` generate before the timed loop with a long timeout (`prewarm_timeout_seconds`, default 1800s in `configs/llm_benchmark.yaml`); the timed loop then runs against the resident model with the tighter `http_timeout_seconds` (default 600s). Both timeouts are YAML fields, not source constants — change the config, not the code. See Issue 12.
- `resources/hailo-8/` and `resources/hailo-10H/` bundle HailoRT `.deb` installers + user guides for the two AI HAT+ generations. The 4.x / 4.23.0 stack for AI HAT+ (Hailo-8/8L) is the apt-installed default that `scripts/setup_rpi_ai_hat_plus.sh` configures. The 5.2.0 `.deb` for AI HAT+ 2 (Hailo-10H) is **userspace only** — verified 2026-04-28 that `scripts/setup_rpi_ai_hat_plus_2.sh` cannot complete the 5.x install on Pi OS Bookworm because the RPi apt repo has no `hailo-h10-all` package and the script never falls back to the bundled `.deb`. The full HailoRT 5.x install (driver + firmware + userspace + Python wheel + GenAI model zoo) must be downloaded manually from the Hailo Developer Zone — see `docs/hailo.md` § "LLM on Hailo-10H → Setup (high level)" for the procedure. `scripts/setup_jetson_orin_nano.sh` is the symmetric setup script for the Jetson platform.
- Setup scripts emit `*.log` files (e.g. `hailort.log`, `setup_rpi_ai_hat_plus_2.log`, `setup_rpi_ai_hat_plus.log`) to the repo root. They are not in `.gitignore` but are runtime artefacts, not sources — safe to delete and leave them out of commits.
- `resources/` (checked in) ships with the product and is consumed by setup scripts; `documents/` (gitignored) holds internal PRDs and phase-scoping notes that are not part of the runtime — don't treat anything under `documents/` as authoritative for code behaviour, and don't recreate it if it's missing locally.

## Additional Documentation

`docs/` contains the authoritative reference for CLI flags (`cli.md`), workload metrics and model groups (`workloads.md`), Hailo integration and limits (`hailo.md`), benchmark methodology and reproducibility verification (`methodology.md`), output file layout and dashboard filters (`output.md`), common failures (`troubleshooting.md`), the workstation HEF compilation procedure (`hef_compilation.md`), and per-arch compilation workflow guides (`compilation/` — `hailo8/` and `hailo10h/` per-chip workflows plus shared `setup.md` / `tools.md` / `pitfalls.md`). Consult these before duplicating information here.

`resources/` contains:
- `hailo-8/` and `hailo-10H/` — bundled HailoRT 4.x and 5.x `.deb` installers + user guides consumed by the platform setup scripts.
- `hailo-sdk/` — workstation-side EULA-gated Hailo Dataflow Compiler + Model Zoo wheels (gitignored as `*.whl`; the README is the per-arch venv bring-up procedure). Required only on x86_64 compile boxes, not on the Pi.
- `hefs/` — staged prebuilt HEFs for the Pi-side runtime (canonical naming per `NAMING.txt`); `benchmark/workloads/yolo/conversion/hef_source.py` resolves these before falling back to compile or `/usr/share/hailo-models/`.
- `session_issues_<YYYY-MM-DD>.md` and `session_notes_<YYYY-MM-DD>_*.md` — dated incident logs from setup-and-verify sessions on real hardware. The 2026-04-27 entry documents twelve issues fixed during the AI HAT+ Pi bring-up (setup hygiene, LLM model consolidation, dep gaps, prebuilt-HEF source layer). The 2026-04-29 NVIDIA workstation notes document eleven issues fixed during the gap-model compile validation: missing CUDA extras under DFC, NCCL cu12/cu13 collision breaking torch, dead `optimization_level` config fields → ALLS model script, mutually-exclusive optimization flavors → forced `bias_correction`, and the load-bearing finding that `END_NODE_TABLE` must cut at raw Conv outputs (not deep post-processing layers) plus the `yolo11n-seg / hailo8` capacity miss. The arch-blind cache-key bug (Issue 9) bit twice during the sweeps — fixed in the AMD-side cache refactor (target_device is now a required keyword-only arg on every cache method; on-disk layout is `models/hailo/<arch>/<version>/<task>/<model>/`; regression covered by `tests/test_cache_arch_isolation.py`). The 2026-05-02 entry documents the AI HAT+ 2 NPU bring-up: HailoRT 4.x → 5.x runtime migration in code (deprecated `InferVStreams` → `InferModel`/`bindings`), a class-count heuristic fix for truncated-head outputs, the truncated-head assembler that reassembles per-stride box+cls+extras into the combined layout the existing decoders expect (DFL for 64-ch v8/v11 box, direct read for 4-ch v26 box), and the `hailo-ollama.service` systemd unit that finally activates the NPU LLM step (was silently SKIP'd on every prior verify). Headline result: NPU at 10.29 TPS vs CPU at 6.89 TPS on `llama3.2:1b` / drone prompts — the cross-backend comparison the AI HAT+ 2 verify exists to produce. Read the relevant file before debugging Hailo-stack, venv, LLM-config, compile-pipeline, or report-pipeline failures — most failure modes are already catalogued there with root cause and verified fix.

## Git Conventions

- Never add Co-Authored-By, Signed-off-by, or any Claude attribution to commits.
