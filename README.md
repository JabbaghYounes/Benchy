# Edge AI Benchmark Suite

A standardized, automated benchmarking framework to evaluate and compare AI inference capabilities across popular edge AI platforms.

## Overview

This benchmark suite provides comprehensive performance evaluation for:

- **Computer Vision**: YOLO inference (v8, v11, v26) across all five tasks —
  detection, classification, OBB, segmentation, and pose — on Hailo NPU,
  Jetson GPU, or PyTorch CPU.
- **Local LLM Inference**: Ollama-based models (1.5B, 7B, 8B, 9B groups) on
  CPU; or via HailoRT GenAI's Ollama-compatible REST endpoint on the
  Hailo-10H NPU (qwen2:1.5b, qwen2.5-{instruct,coder}:1.5b,
  deepseek_r1_distill_qwen:1.5b, llama3.2:3b).
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

Run the appropriate setup script for your hardware:

```bash
# NVIDIA Jetson Orin Nano
./scripts/setup_jetson_orin_nano.sh

# Raspberry Pi with AI HAT+
./scripts/setup_rpi_ai_hat_plus.sh

# Raspberry Pi with AI HAT+ 2
./scripts/setup_rpi_ai_hat_plus_2.sh
```

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

```bash
# Pi 5 + AI HAT+ (Hailo-8 / 8L) — vision sweep (det/obb/seg/pose × v8/v11/v26)
./scripts/verify_ai_hat_plus.sh

# Pi 5 + AI HAT+ 2 (Hailo-10H) — same vision sweep + LLM-on-NPU + auto-dashboard
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
| **default** | v8 detection, nano size | llama2:7b across q4_K_M / q5_K_M / q8_0 (quant sweep) | Quick validation + INT-quant baseline |
| **full** | All versions, all tasks, all sizes | All CPU model groups (7B / 8B / 9B) | Thorough evaluation |
| **drone** | v8/v11/v26 detection at 1280, sizes n/s/m, VisDrone dataset | llama2:7b on the curated drone prompt set (scene / target / mission / telemetry / hazard) | Realistic small-object aerial detection |
| **drone_full** | Detection + OBB + seg + pose at 1280, sizes n/s — broadest drone-relevant Hailo sweep | _(YOLO-only)_ | Exercises every Phase 3 task at altitude-realistic resolution |
| **npu** | _(LLM-only)_ | qwen2:1.5b on the Hailo-10H NPU via HailoRT GenAI on `:8000`, drone prompt set | LLM-on-NPU comparison row (AI HAT+ 2 only) |

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

### Hailo NPU (Raspberry Pi only)
- hailo-platform >= 4.17.0 (HailoRT SDK)
- hailo-dataflow-compiler >= 3.26.0 (for model compilation)
- onnx >= 1.14.0, onnxruntime >= 1.15.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Validate locally with `black benchmark/`, `mypy benchmark/`, and `pytest tests/` (after `pip install -e ".[dev]"`)
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
