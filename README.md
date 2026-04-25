# Edge AI Benchmark Suite

A standardized, automated benchmarking framework to evaluate and compare AI inference capabilities across popular edge AI platforms.

## Overview

This benchmark suite provides comprehensive performance evaluation for:

- **Computer Vision**: YOLO inference benchmarks (v8, v11, v26)
- **Local LLM Inference**: Ollama-based models (7B, 8B, 9B parameter groups)

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

## Benchmark Profiles

| Profile | YOLO | LLM | Use Case |
|---------|------|-----|----------|
| **default** | v8 detection, nano size | llama2:7b only | Quick validation |
| **full** | All versions, all tasks, all sizes | All model groups (7B-9B) | Thorough evaluation |

## Key Assumptions

1. **Native installation** - Benchmarks run natively on target hardware, not in containers
2. **Single device** - One benchmark instance per device at a time
3. **Stable power** - Consistent power supply during benchmarking
4. **Thermal stability** - Allow device to reach thermal equilibrium before full runs
5. **Ollama server** - LLM benchmarks require Ollama running on localhost:11434
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
4. Validate locally with `black benchmark/` and `mypy benchmark/` (no test suite yet)
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
