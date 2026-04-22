# Edge AI Benchmark Suite

A standardized, automated benchmarking framework to evaluate and compare AI inference capabilities across popular edge AI platforms.

## Overview

This benchmark suite provides comprehensive performance evaluation for:

- **Computer Vision**: YOLO inference benchmarks (v8, v11, v26)
- **Local LLM Inference**: Ollama-based models (1B, 3B, 7B, 8B, 9B parameter sizes)

### Supported Platforms

| Platform | Accelerator | RAM |
|----------|-------------|-----|
| NVIDIA Jetson Orin Nano Developer Kit | Ampere GPU | 8GB |
| Raspberry Pi + AI HAT+ | Hailo-8L NPU | 8GB |
| Raspberry Pi + AI HAT+ 2 | Hailo-10H NPU | 8GB |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/edge-ai-benchmark.git
cd edge-ai-benchmark
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
cd ~/Benchy  # or your project directory
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
| [Hailo NPU](docs/hailo.md) | Hailo-8L/10H integration, model conversion, and limitations |
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
4. Run tests and validation
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
