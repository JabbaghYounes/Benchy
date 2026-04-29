# Edge AI Benchmark Suite

A standardized, automated benchmarking framework to evaluate and compare AI inference capabilities across popular edge AI platforms.

## Overview

This benchmark suite provides comprehensive performance evaluation for:

- **Computer Vision**: YOLO inference (v8, v11, v26) across all five tasks —
  detection, classification, OBB, segmentation, and pose — on Hailo NPU,
  Jetson GPU, or PyTorch CPU.
- **Local LLM Inference**: Llama-only policy. CPU side (1B / 3B / 7B):
  `llama3.2:1b`, `llama3.2:3b`, `llama2:7b` via Ollama. NPU side (1B
  only): `llama3.2:1b` — the only llama-family prebuilt HEF in the
  HailoRT 5.3.0 GenAI Model Zoo — via HailoRT GenAI's Ollama-compatible
  REST endpoint. Cross-backend comparison is only at the 1B level (no
  3B or 7B HEFs ship in the zoo).
- **Backend axis**: Every `LLMResult` is tagged with a backend label
  (`ollama-cpu` / `ollama-cuda` / `hailo-10h`) and the dashboard splits
  CPU and NPU runs into separate filterable rows.

### Hailo task coverage

| YOLO Version | Detection | Classification | OBB | Segmentation | Pose |
|---|---|---|---|---|---|
| v8 | ✓ | ✓ | ✓ (Phase 3a) | ✓ (Phase 3b) | ✓ (Phase 3c) |
| v11 | ✓ | ✓ | ✓ (Phase 3a) | ✓ (Phase 3b) | ✓ (Phase 3c) |
| v26 | ✓ | ✓ | experimental | experimental | experimental |

v26 entries clear the conversion + postprocessor pipeline but lack public
Hailo Model Zoo backing; hardware verification (the HW-verify runners
below) is what moves them from experimental to verified.

### Supported Platforms

| Platform | Accelerator | NPU TOPS | Host RAM | NPU RAM |
|----------|-------------|----------|----------|---------|
| NVIDIA Jetson Orin Nano Developer Kit | Ampere GPU | — | 8 GB | shared |
| Raspberry Pi 5 + AI HAT+ (13 TOPS) | Hailo-8L NPU | 13 (INT8) | 8 GB | uses host |
| Raspberry Pi 5 + AI HAT+ (26 TOPS) | Hailo-8 NPU  | 26 (INT8) | 8 GB | uses host |
| Raspberry Pi 5 + AI HAT+ 2            | Hailo-10H NPU | 40 (INT4) | 8 GB | 8 GB onboard SDRAM |

Both AI HAT+ variants and the AI HAT+ 2 connect to the Raspberry Pi 5 via PCIe (the 26 TOPS HAT auto-switches the link to PCIe Gen 3 for full bandwidth) and conform to the Raspberry Pi HAT+ spec. Hailo-10H is the only variant with onboard SDRAM and is the only one that can run local LLMs / VLMs (~6B params) on the accelerator itself.

References: [AI HAT+ product brief](https://datasheets.raspberrypi.com/ai-hat/ai-hat-plus-product-brief.pdf), [Raspberry Pi AI HAT documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html), [Hailo-8 overview](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/).

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/JabbaghYounes/Benchy.git
cd Benchy
```

### 2. Platform Setup

Run the appropriate setup script for your hardware. **`sudo` is required**
(the script installs apt packages, udev rules, and the kernel
HailoRT/PCIe driver). Pass `--pull-models` to also pre-pull the three
llama LLM models (`llama3.2:1b` / `llama3.2:3b` / `llama2:7b`, ~7 GB
total) so the LLM benchmarks can run without an extra `ollama pull`
step:

```bash
# NVIDIA Jetson Orin Nano
sudo ./scripts/setup_jetson_orin_nano.sh --pull-models

# Raspberry Pi with AI HAT+
sudo ./scripts/setup_rpi_ai_hat_plus.sh --pull-models

# Raspberry Pi with AI HAT+ 2
sudo ./scripts/setup_rpi_ai_hat_plus_2.sh --pull-models
```

> **AI HAT+ 2 caveat (verified 2026-04-28).** On Pi OS Bookworm the setup script cannot complete the HailoRT 5.x install — Raspberry Pi's apt repo has no `hailo-h10-all` package and caps at HailoRT 4.20.0, which doesn't recognise the Hailo-10H. The script reports SUCCESS but leaves you on 4.x with the chip invisible (`/dev/hailo0` missing, `hailortcli scan` empty). The HailoRT 5.x driver/userspace/firmware/Python-wheel and the GenAI model-zoo `.deb` must currently be downloaded manually from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free account). See `docs/hailo.md` § "LLM on Hailo-10H → Setup (high level)" for the full step-by-step procedure.

Drop `--pull-models` if you only need YOLO benchmarks or want a leaner
install (~7 GB smaller). The setup script also installs the project's
`[dev]` extras (pytest / black / mypy) into the venv so the verify
runners' first pytest step works out of the box.

### 3. Activate the Virtual Environment

The setup script creates a virtual environment at `./venv`. Activate it before running benchmarks:

```bash
source venv/bin/activate
```

**Note:** On Raspberry Pi OS Bookworm and newer, system-wide pip installs are blocked (PEP 668). Always use the virtual environment created by the setup script.

### 4. Run Benchmarks

```bash
# Run default profile (quick benchmark)
python -m benchmark run all

# Run full profile (comprehensive benchmark)
python -m benchmark run all --profile full
```

### 5. Hardware verification (Hailo boards)

Smart runners that sweep every Phase 2 / 3 task in one command, with
per-step progress, timing, log capture, JSON validation, and a final
pass/fail summary. Continue-on-failure: a single broken model doesn't
abort the rest of the sweep, and the exit code reflects only blocking
(non-experimental) failures.

Both scripts produce identical 13-step bundles (vision sweep
det/obb/seg/pose × v8/v11/v26 + LLM-on-NPU + LLM-on-CPU comparison row
+ auto-generated dashboard) so the two boards' result directories are
directly diff-able. The LLM-on-NPU step writes a documented
`[unsupported-on-this-hw]` stub on AI HAT+ (no onboard SDRAM, can't
host LLMs) and a real run on AI HAT+ 2.

```bash
# Pi 5 + AI HAT+ (Hailo-8 / 8L)
./scripts/verify_ai_hat_plus.sh

# Pi 5 + AI HAT+ 2 (Hailo-10H)
./scripts/verify_ai_hat_plus_2.sh
```

Output lands in `results/hw_verify_<timestamp>/` (per-step `.log`s,
`bench_*.json` artefacts, and on AI HAT+ 2 a generated `report/`
dashboard). v26 entries are tagged `[experimental]` and counted
separately at the end so a v26 failure doesn't read as a project
regression.

## Benchmark Profiles

| Profile | YOLO | LLM | Use Case |
|---------|------|-----|----------|
| **default** | v8 detection, nano size | llama2:7b (bare tag, no quant sweep) | Quick CPU smoke test |
| **full** | All versions, all tasks, all sizes | llama3.2:1b + llama3.2:3b + llama2:7b (1B / 3B / 7B llama groups) | Thorough evaluation |
| **drone** | v8/v11/v26 detection at 1280, sizes n/s/m, VisDrone dataset | llama2:7b on the curated drone prompt set (scene / target / mission / telemetry / hazard) | Realistic small-object aerial detection |
| **drone_full** | Detection + OBB + seg + pose at 1280, sizes n/s — broadest drone-relevant Hailo sweep | _(YOLO-only)_ | Exercises every Phase 3 task at altitude-realistic resolution |
| **npu** | _(LLM-only)_ | llama3.2:1b on the Hailo-10H NPU via HailoRT GenAI on `:8000`, drone prompt set (only llama HEF in HailoRT 5.3.0 zoo; no 3B/7B HEFs ship) | LLM-on-NPU comparison row (AI HAT+ 2 only) |
| **compare** | _(LLM-only)_ | llama3.2:1b on Ollama CPU + drone prompt set — RAM-safe CPU mirror of the `npu` profile | True 1B-vs-1B cross-backend (NPU vs CPU) comparison row; used by `verify_ai_hat_plus_2.sh`. Fits any Pi 5 (4 GB or 8 GB), unlike `drone` which needs 8 GB for llama2:7b |

Profiles are configured in `configs/yolo_benchmark.yaml` and
`configs/llm_benchmark.yaml`. They can declare `input_resolution`,
per-task `datasets:`, `prompt_set`, `quants` + `quant_tag_template`, and
(LLM-only) `api_base`, `backend`, `npu_metrics`.

## Key Assumptions

1. **Native installation** - Benchmarks run natively on target hardware, not in containers
2. **Single device** - One benchmark instance per device at a time
3. **Stable power** - Consistent power supply during benchmarking
4. **Thermal stability** - Allow device to reach thermal equilibrium before full runs
5. **LLM server** - CPU-side LLM benchmarks require Ollama on `localhost:11434`. The `npu` profile additionally requires the HailoRT GenAI server (`hailo-ollama`) reachable at `localhost:8000`; see `docs/hailo.md` for setup.
6. **Network isolation** - No network-dependent operations during benchmarks

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli.md) | Full command reference with examples |
| [Workloads](docs/workloads.md) | YOLO and LLM benchmark details, metrics, and model groups |
| [Hailo NPU](docs/hailo.md) | Hailo-8 / 8L / 10H integration, model conversion, and limitations |
| [Methodology](docs/methodology.md) | Benchmark methodology and reproducibility |
| [Output & Configuration](docs/output.md) | Result formats, dashboard, and YAML configuration |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |

## Dependencies

### Core
- psutil >= 5.9.0
- requests >= 2.28.0
- pyyaml >= 6.0
- numpy >= 1.21.0
- ultralytics >= 8.0.0
- onnx >= 1.14.0, onnxruntime >= 1.15.0 (drive the `.pt → .onnx → .har → .hef` Hailo conversion pipeline; pinned in `setup.py:install_requires`)

### Hailo NPU (Raspberry Pi only)
- hailo-platform >= 4.17.0 (HailoRT SDK)
- hailo-dataflow-compiler >= 3.26.0 (for model compilation)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Validate locally with `black benchmark/`, `mypy benchmark/`, and `pytest tests/` (after `pip install -e ".[dev]"`)
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
