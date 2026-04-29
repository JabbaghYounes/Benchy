# CLI Reference

All commands are run via `python -m benchmark <command>`.

## Run Benchmarks

```bash
# Run YOLO benchmarks only
python -m benchmark run yolo

# Run LLM benchmarks only
python -m benchmark run llm

# Run all benchmarks
python -m benchmark run all

# Run with full profile
python -m benchmark run all --profile full

# Drone profile: small-object aerial detection at 1280 input, VisDrone
# dataset on the YOLO side; on the LLM side, drone-relevant prompts
# (scene description, target identification, mission preflight, telemetry,
# hazard reasoning).
python -m benchmark run all --profile drone

# Drone-full profile: extends drone with the Phase-3-unlocked tasks —
# OBB on DOTA (the drone-priority addition), plus segmentation (COCO128)
# and pose (COCO8) for completeness. Sizes drop to n/s only since five
# tasks at 1280 grow runtime fast. Use this when you want the broadest
# drone-relevant Hailo coverage in a single YOLO command.
python -m benchmark run yolo --profile drone_full

# NPU profile (Pi 5 + AI HAT+ 2 only): runs the LLM workload through
# HailoRT GenAI's Ollama-compatible REST endpoint on the Hailo-10H NPU.
# Each LLMResult is tagged with backend="hailo-10h" and gets NPU-side
# metrics (power, HailoRT version) alongside the host-side ResourceMonitor
# readings. Uses llama3.2:1b — the only llama-family model with a
# prebuilt HEF in the HailoRT 5.3.0 GenAI Model Zoo (5.1.1 had llama3.2:3b
# but Hailo dropped it; no 7B HEFs ship in any release). See docs/hailo.md
# for the full prebuilt HEF list.
python -m benchmark run llm --profile npu

# Smart hardware-verification runners (Hailo boards). Each sweeps every
# Phase 2 / 3 task with progress + timing + JSON validation + a final
# pass/fail summary. Continue-on-failure semantics; v26 entries are
# tagged [experimental] and counted separately. Output lands in
# results/hw_verify_<timestamp>/.
./scripts/verify_ai_hat_plus.sh         # Pi 5 + AI HAT+ (Hailo-8 / 8L)
./scripts/verify_ai_hat_plus_2.sh       # Pi 5 + AI HAT+ 2 (Hailo-10H, also LLM-on-NPU)

# Specify output directory
python -m benchmark run all --output ./my_results

# Override platform detection
python -m benchmark run all --platform jetson_orin_nano

# Use specific backend (Hailo NPU)
python -m benchmark run yolo --backend hailo

# Force recompilation of Hailo models
python -m benchmark run yolo --backend hailo --force-recompile

# Run specific YOLO version only
python -m benchmark run yolo --yolo-version v26

# Run a specific YOLO model
python -m benchmark run yolo --yolo-model yolo26n.pt
python -m benchmark run yolo --yolo-model yolov8s-seg.pt

# Combine version and backend
python -m benchmark run yolo --yolo-version v11 --backend hailo
```

### Llama-only LLM groups (1B / 3B / 7B)

The CLI's `--profile` choices are `default`, `full`, `drone`, `drone_full`,
`npu`, `compare`. There is no per-model or per-group flag — model
selection is profile-driven via `configs/llm_benchmark.yaml`. Under the
project's llama-only policy:

```bash
# 7B llama (llama2:7b) on five legacy prompts — quick CPU smoke test
# (requires 8 GB Pi 5; llama2:7b needs ~5.5 GB RAM at runtime)
python -m benchmark run llm

# All three llama sizes (llama3.2:1b + llama3.2:3b + llama2:7b)
# (requires 8 GB Pi 5 for the 7B step)
python -m benchmark run llm --profile full

# 7B llama on the curated drone prompt set (8 GB Pi 5 only)
python -m benchmark run llm --profile drone

# 1B llama on the Hailo-10H NPU (AI HAT+ 2 only)
python -m benchmark run llm --profile npu

# 1B llama on Ollama CPU + drone prompts — RAM-safe mirror of the npu
# profile; gives a true 1B-vs-1B cross-backend comparison row when
# combined with --profile npu. Used by verify_ai_hat_plus_2.sh.
python -m benchmark run llm --profile compare
```

To swap or add models, edit the `models:` block in
`configs/llm_benchmark.yaml`. Adding non-llama tags is outside the
project's current policy (see `resources/session_notes_2026-04-27.md`).

## Show System Information

```bash
python -m benchmark info
```

## Aggregate Results

```bash
# Aggregate results from default directory
python -m benchmark aggregate

# Specify input/output directories
python -m benchmark aggregate --input ./results --output ./aggregated
```

## Generate Dashboard

```bash
# Generate HTML dashboard
python -m benchmark dashboard

# Custom title and paths
python -m benchmark dashboard --input ./results --output ./dashboard.html --title "My Benchmark"
```

## Generate Full Report

```bash
# Generate aggregated results + dashboard
python -m benchmark report

# Custom paths
python -m benchmark report --input ./results --output ./report
```

## List Supported Models

```bash
# List all supported models
python -m benchmark list-models

# List Hailo-supported models only
python -m benchmark list-models --backend hailo

# Output as JSON
python -m benchmark list-models --format json
```

## Check Available Backends

```bash
python -m benchmark backends
```

## Cross-Platform Verification

```bash
# Compare results from two platforms
python -m benchmark verify results/jetson_run.json results/rpi_hailo_run.json

# Save verification report
python -m benchmark verify results/jetson_run.json results/rpi_hailo_run.json --output verification.json
```
