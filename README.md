# Edge AI Benchmark Suite

A standardized, automated benchmarking framework to evaluate and compare AI inference capabilities across popular edge AI platforms.

## Overview

This benchmark suite provides comprehensive performance evaluation for:

- **Computer Vision**: YOLO inference benchmarks (v8, v11, v26)
- **Local LLM Inference**: Ollama-based models (7B, 8B, 9B parameter sizes)

### Supported Platforms

| Platform | Accelerator | RAM |
|----------|-------------|-----|
| NVIDIA Jetson Nano Developer Kit | Tegra X1 GPU | 4GB |
| Raspberry Pi + AI HAT+ | Hailo-8L NPU | 8GB |
| Raspberry Pi + AI HAT+ 2 | Hailo-8 NPU | 8GB |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/edge-ai-benchmark.git
cd edge-ai-benchmark
```

### 2. Platform Setup

Run the appropriate setup script for your hardware:

**NVIDIA Jetson Nano:**
```bash
./scripts/setup_jetson_nano.sh
```

**Raspberry Pi with AI HAT+:**
```bash
./scripts/setup_rpi_ai_hat_plus.sh
```

**Raspberry Pi with AI HAT+ 2:**
```bash
./scripts/setup_rpi_ai_hat_plus_2.sh
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Benchmarks

```bash
# Run default profile (quick benchmark)
python -m benchmark run all

# Run full profile (comprehensive benchmark)
python -m benchmark run all --profile full
```

## Usage

### CLI Commands

The benchmark suite provides several CLI commands:

#### Run Benchmarks

```bash
# Run YOLO benchmarks only
python -m benchmark run yolo

# Run LLM benchmarks only
python -m benchmark run llm

# Run all benchmarks
python -m benchmark run all

# Run with full profile
python -m benchmark run all --profile full

# Specify output directory
python -m benchmark run all --output ./my_results

# Override platform detection
python -m benchmark run all --platform jetson_nano
```

#### Show System Information

```bash
python -m benchmark info
```

#### Aggregate Results

```bash
# Aggregate results from default directory
python -m benchmark aggregate

# Specify input/output directories
python -m benchmark aggregate --input ./results --output ./aggregated
```

#### Generate Dashboard

```bash
# Generate HTML dashboard
python -m benchmark dashboard

# Custom title and paths
python -m benchmark dashboard --input ./results --output ./dashboard.html --title "My Benchmark"
```

#### Generate Full Report

```bash
# Generate aggregated results + dashboard
python -m benchmark report

# Custom paths
python -m benchmark report --input ./results --output ./report
```

### Benchmark Profiles

The suite supports two benchmark profiles:

#### Default Profile (Quick)
- **YOLO**: v8 detection with nano (n) model size
- **LLM**: llama2:7b model only
- Suitable for quick validation and testing

#### Full Profile (Comprehensive)
- **YOLO**: All versions (v8, v11, v26), all tasks, all model sizes (n, s, m, l, x)
- **LLM**: All model groups (7B, 8B, 9B) with multiple models per group
- Suitable for thorough performance evaluation

## Workloads

### YOLO Benchmarks

**Versions:** v8, v11, v26

**Tasks:**
- Detection
- Segmentation
- Pose estimation
- Oriented Bounding Box (OBB)
- Classification

**Model Sizes:** nano (n), small (s), medium (m), large (l), extra-large (x)

**Metrics Collected:**
| Metric | Description |
|--------|-------------|
| Latency (mean, std, min, max, p50, p95, p99) | Inference time in milliseconds |
| Throughput | Frames per second (FPS) |
| mAP | Mean Average Precision |
| Precision | Detection precision |
| Recall | Detection recall |
| CPU % | CPU utilization |
| Accelerator % | GPU/NPU utilization |
| Memory (MB) | Memory usage |
| Power (W) | Power consumption |

### LLM Benchmarks

**Model Groups:**
- **7B**: llama2:7b, mistral:7b, olmo2:7b
- **8B**: llama3.1:8b, dolphin3:8b, dolphin-llama3:8b
- **9B**: gemma2:9b

**Prompts:** Fixed set of 5 deterministic prompts:
- simple_qa
- reasoning
- code_generation
- summarization
- creative

**Metrics Collected:**
| Metric | Description |
|--------|-------------|
| TTFT (ms) | Time to First Token |
| Tokens/sec | Generation speed |
| Total Latency (ms) | End-to-end response time |
| Prompt Tokens | Input token count |
| Output Tokens | Generated token count |
| CPU % | CPU utilization |
| Accelerator % | GPU/NPU utilization |
| Memory (MB) | Memory usage |
| Power (W) | Power consumption |

## Output Files

### Raw Results (per run)

```
results/
├── bench_YYYYMMDD_HHMMSS_XXXXXXXX.json     # Complete benchmark run
├── bench_YYYYMMDD_HHMMSS_XXXXXXXX_yolo.csv # YOLO results table
└── bench_YYYYMMDD_HHMMSS_XXXXXXXX_llm.csv  # LLM results table
```

### Aggregated Results

```
results/aggregated/
├── agg_YYYYMMDD_HHMMSS.json           # Complete aggregated results
├── agg_YYYYMMDD_HHMMSS_yolo.csv       # Aggregated YOLO metrics
├── agg_YYYYMMDD_HHMMSS_llm.csv        # Aggregated LLM metrics
└── agg_YYYYMMDD_HHMMSS_platforms.csv  # Platform summaries
```

### Dashboard

The HTML dashboard includes:
- System overview with platform comparison
- YOLO performance charts (latency, FPS, accuracy, power)
- YOLO scaling analysis (throughput vs model size, latency vs accuracy trade-off)
- LLM performance charts (tokens/sec, TTFT)
- LLM efficiency charts (memory usage, TPS vs memory trade-off)
- Stability/variance analysis
- Raw data tables
- Data download links

Open in browser:
```bash
# After generating dashboard
firefox results/dashboard.html
# or
google-chrome results/dashboard.html
```

## Configuration

Configuration files are located in `configs/`:

### YOLO Configuration (`configs/yolo_benchmark.yaml`)

```yaml
benchmark:
  warmup_runs: 3
  measured_runs: 10
  input_resolution: 640

inference:
  device: "0"
  conf_threshold: 0.25
  iou_threshold: 0.45

default:
  yolo_versions: ["v8"]
  tasks: ["detection"]
  model_sizes: ["n"]

full:
  yolo_versions: ["v8", "v11", "v26"]
  tasks: ["detection", "segmentation", "pose", "obb", "classification"]
  model_sizes: ["n", "s", "m", "l", "x"]
```

### LLM Configuration (`configs/llm_benchmark.yaml`)

```yaml
benchmark:
  warmup_runs: 3
  measured_runs: 10

generation:
  temperature: 0.0
  top_p: 1.0
  top_k: 1
  seed: 42
  max_tokens: 256

default:
  model_groups: ["7B"]
  models: ["llama2:7b"]

full:
  model_groups: ["7B", "8B", "9B"]
```

## Benchmark Methodology

### Warmup and Measured Runs

Each benchmark executes:
1. **3 warmup runs** - Not recorded, allows system to reach steady state
2. **10 measured runs** - Recorded for statistical analysis

### Deterministic LLM Evaluation

LLM benchmarks use fixed parameters:
- `temperature: 0.0`
- `seed: 42`
- `top_p: 1.0`
- `top_k: 1`

This ensures reproducible results across runs.

### YOLO Accuracy Validation

YOLO benchmarks can optionally run validation:
- Uses model's built-in validation method
- Reports mAP, precision, and recall
- Skip with `--skip-validation` flag

## Reproducibility

To ensure reproducible results:

1. **Use consistent power state**
   - Jetson: Set power mode with `nvpmodel`
   - Ensure consistent cooling

2. **Minimize background processes**
   ```bash
   # Check running processes
   htop
   ```

3. **Use fixed random seeds**
   - LLM benchmarks use `seed=42`

4. **Record environment**
   - System info is captured automatically
   - Check with `python -m benchmark info`

5. **Multiple runs**
   - Run benchmarks multiple times
   - Use aggregation to analyze variance

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama server
ollama serve

# Check status
curl http://localhost:11434/api/version
```

### Model Not Found

```bash
# Pull required model
ollama pull llama2:7b
```

### CUDA/GPU Issues (Jetson)

```bash
# Check GPU status
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Hailo NPU Issues (Raspberry Pi)

```bash
# Check Hailo status
hailortcli fw-control identify
```

### Insufficient Memory

- Use smaller model sizes (n, s)
- Run workloads separately instead of `all`
- Close other applications

## Project Structure

```
edge-ai-benchmark/
├── benchmark/
│   ├── __init__.py
│   ├── __main__.py           # Entry point
│   ├── cli.py                # CLI interface
│   ├── schemas.py            # Data models
│   ├── workloads/
│   │   ├── yolo/
│   │   │   └── runner.py     # YOLO benchmark runner
│   │   └── llm/
│   │       └── runner.py     # LLM benchmark runner
│   ├── metrics/
│   │   └── collectors.py     # System info & resource monitoring
│   ├── results/
│   │   └── writers.py        # JSON/CSV result writers
│   ├── aggregation/
│   │   ├── aggregator.py     # Results aggregation
│   │   └── csv_writer.py     # Aggregated CSV output
│   └── reporting/
│       └── dashboard.py      # HTML dashboard generation
├── configs/
│   ├── yolo_benchmark.yaml   # YOLO configuration
│   └── llm_benchmark.yaml    # LLM configuration
├── scripts/
│   ├── common.sh             # Common setup utilities
│   ├── setup_jetson_nano.sh  # Jetson Nano setup
│   ├── setup_rpi_ai_hat_plus.sh    # RPi AI HAT+ setup
│   └── setup_rpi_ai_hat_plus_2.sh  # RPi AI HAT+ 2 setup
├── results/                  # Benchmark output directory
├── requirements.txt          # Python dependencies
└── README.md
```

## Dependencies

### Core
- psutil >= 5.9.0
- requests >= 2.28.0
- pyyaml >= 6.0
- numpy >= 1.21.0
- ultralytics >= 8.0.0

### Optional
- tqdm >= 4.64.0 (progress display)

## Key Assumptions

1. **Native installation** - Benchmarks run natively on target hardware, not in containers
2. **Single device** - One benchmark instance per device at a time
3. **Stable power** - Consistent power supply during benchmarking
4. **Thermal stability** - Allow device to reach thermal equilibrium before full runs
5. **Ollama server** - LLM benchmarks require Ollama running on localhost:11434
6. **Network isolation** - No network-dependent operations during benchmarks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and validation
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
