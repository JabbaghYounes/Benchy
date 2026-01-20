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

# Use specific backend (Hailo NPU)
python -m benchmark run yolo --backend hailo

# Force recompilation of Hailo models
python -m benchmark run yolo --backend hailo --force-recompile
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

#### List Supported Models

```bash
# List all supported models
python -m benchmark list-models

# List Hailo-supported models only
python -m benchmark list-models --backend hailo

# Output as JSON
python -m benchmark list-models --format json
```

#### Check Available Backends

```bash
# Show available inference backends
python -m benchmark backends
```

#### Cross-Platform Verification

```bash
# Compare results from two platforms
python -m benchmark verify results/jetson_run.json results/rpi_hailo_run.json

# Save verification report
python -m benchmark verify results/jetson_run.json results/rpi_hailo_run.json --output verification.json
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

| Group | Models | Architecture | Specialization |
|-------|--------|--------------|----------------|
| **1B** | llama3.2:1b, granite3.1-moe:1b, sailor2:1b | Dense/MoE | General |
| **3B** | llama3.2:3b, granite3.1-moe:3b, starcoder2:3b | Dense/MoE | General/Code |
| **7B** | llama2:7b, mistral:7b, olmo2:7b | Dense | General |
| **8B** | llama3.1:8b, dolphin3:8b, dolphin-llama3:8b | Dense | General |
| **9B** | gemma2:9b | Dense | General |

**Important Constraints:**
- Models are **only compared within the same parameter group**
- **MoE models** (granite3.1-moe) may show different performance characteristics than dense models
- **Code-specialized models** (starcoder2:3b) are evaluated with both general and code prompts

**Prompt Sets:**

*Legacy Prompts (7B+):*
- simple_qa, reasoning, code_generation, summarization, creative

*Lightweight Model Prompts (1B/3B):*
- **General Reasoning**: summarization, instruction_following, short_reasoning
- **Code Generation**: function_generation, code_completion, syntax_validation

**Benchmark Parameters (1B/3B):**
- Warmup runs: 2
- Measured runs: 10
- Temperature: 0.2
- top_p: 0.95
- max_tokens: 256
- Streaming: disabled
- Prompt batching: 3 prompts per batch

**Metrics Collected:**
| Metric | Description |
|--------|-------------|
| TTFT (ms) | Time to First Token (mean, median, min, max) |
| Tokens/sec | Generation speed (mean, median, min, max) |
| Total Latency (ms) | End-to-end response time |
| Peak Memory (MB) | Maximum memory usage during inference |
| Prompt Tokens | Input token count |
| Output Tokens | Generated token count |
| Truncation Rate | Percentage of responses truncated at max_tokens |
| CPU % | CPU utilization |
| Accelerator % | GPU/NPU utilization |
| Memory (MB) | Memory usage |
| Power (W) | Power consumption |

**Memory Requirements:**
| Group | Minimum Available RAM |
|-------|----------------------|
| 1B | 2 GB |
| 3B | 4 GB |
| 7B | 8 GB |
| 8B | 10 GB |
| 9B | 12 GB |

The benchmark performs memory preflight checks before loading models. If insufficient memory is detected or swap usage is required, the benchmark will abort with an error.

## Hailo NPU Integration

The benchmark suite includes full support for Hailo-8 and Hailo-8L NPUs on Raspberry Pi platforms.

### Supported Configurations

| YOLO Version | Detection | Classification | Segmentation | Pose | OBB |
|--------------|-----------|----------------|--------------|------|-----|
| v8 | Yes | Yes | No | No | No |
| v11 | Yes | Yes | No | No | No |
| v26 | Yes | Yes | No | No | No |

**Optimized Models:**
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` (Detection)
- `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt` (Classification)
- Similar patterns for v11 and v26

### Model Conversion Pipeline

Hailo requires model conversion from PyTorch to HEF format:

```
.pt (PyTorch) → .onnx (ONNX) → .har (Hailo Archive) → .hef (Hailo Executable)
```

**Compilation is automatic** - models are compiled on first use and cached for subsequent runs.

### Known Limitations

1. **Supported Tasks Only**: Segmentation, pose estimation, and OBB tasks are NOT supported on Hailo NPU due to architectural constraints.

2. **INT8 Quantization**: All Hailo models use INT8 quantization. Minor accuracy differences compared to FP32/FP16 models are expected.

3. **Model Size**: Larger models (l, x variants) may have longer compilation times and higher memory requirements.

4. **No CPU Fallback**: When using `--backend hailo`, the benchmark will NOT fall back to CPU if Hailo is unavailable. This ensures benchmark integrity.

### Important: CPU Inference is Invalid

**CPU inference on Hailo-equipped platforms is NOT a valid benchmark configuration.**

When benchmarking Raspberry Pi with Hailo:
- Always use `--backend hailo` explicitly, OR
- Let auto-detection select Hailo (default on RPi + AI HAT+)
- CPU fallback is disabled by default to prevent misleading results

### Compilation Requirements

First-time model compilation requires:
- **HailoRT SDK** installed (version 4.17+ recommended)
- **Hailo Dataflow Compiler** for .har → .hef conversion
- **Calibration data**: Uses 100 images from COCO validation set
- **Disk space**: ~500MB per compiled model
- **Time**: 5-30 minutes per model depending on size

### Cache Management

Compiled models are cached in `~/.cache/benchy/hailo/`:
```
~/.cache/benchy/hailo/
├── yolov8n_detection_640/
│   ├── model.onnx
│   ├── model.har
│   ├── model.hef
│   └── metadata.json
└── ...
```

To force recompilation:
```bash
python -m benchmark run yolo --backend hailo --force-recompile
```

### Cross-Platform Comparison

To compare Jetson (GPU) vs Raspberry Pi + Hailo (NPU):

```bash
# Run on Jetson
python -m benchmark run yolo --output results/jetson/

# Run on RPi + Hailo
python -m benchmark run yolo --backend hailo --output results/rpi_hailo/

# Generate comparison report
python -m benchmark verify results/jetson/bench_*.json results/rpi_hailo/bench_*.json
```

The verification report shows:
- Performance deltas (FPS, latency)
- Validation of fair comparison criteria
- Warnings for potentially misleading comparisons

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
  warmup_runs: 3        # 2 for lightweight models (1B/3B)
  measured_runs: 10

generation:
  temperature: 0.0      # 0.2 for lightweight models
  top_p: 1.0            # 0.95 for lightweight models
  top_k: 1
  seed: 42
  max_tokens: 256

default:
  model_groups: ["7B"]
  models: ["llama2:7b"]

full:
  model_groups: ["1B", "3B", "7B", "8B", "9B"]

lightweight:            # Profile for 1B/3B models
  model_groups: ["1B", "3B"]
  warmup_runs: 2
  temperature: 0.2
  top_p: 0.95
  prompt_batch_size: 3  # Batch prompts for timer accuracy
```

## Benchmark Methodology

### Warmup and Measured Runs

Each benchmark executes:
1. **Warmup runs** - Not recorded, allows system to reach steady state
   - 3 warmup runs for 7B+ models
   - 2 warmup runs for 1B/3B models
2. **10 measured runs** - Recorded for statistical analysis

### Deterministic LLM Evaluation

LLM benchmarks use fixed parameters:

**Standard Models (7B+):**
- `temperature: 0.0`
- `seed: 42`
- `top_p: 1.0`
- `top_k: 1`

**Lightweight Models (1B/3B):**
- `temperature: 0.2`
- `seed: 42`
- `top_p: 0.95`
- `max_tokens: 256`
- `streaming: disabled`

This ensures reproducible results across runs.

### Group-Safe Aggregation

Results are aggregated only within the same parameter group:
- 1B models are compared only with other 1B models
- Cross-group comparisons are visualized separately in the dashboard
- MoE and code-specialized models are clearly labeled

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
# Check Hailo device status
hailortcli fw-control identify

# List available backends
python -m benchmark backends

# List Hailo-supported models
python -m benchmark list-models --backend hailo

# Check HailoRT version
hailortcli --version

# View Hailo device info
hailortcli scan
```

**Common Hailo Issues:**

| Issue | Solution |
|-------|----------|
| "Hailo device not found" | Check PCIe connection, run `lspci \| grep Hailo` |
| "HailoRT not installed" | Install HailoRT SDK from Hailo Developer Zone |
| "Model compilation failed" | Check disk space, ensure Dataflow Compiler is installed |
| "Unsupported task" | Use detection or classification only (see Supported Configurations) |
| "CPU fallback error" | This is expected - Hailo backend requires NPU hardware |

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
│   ├── verification.py       # Cross-platform verification (Phase 7)
│   ├── workloads/
│   │   ├── yolo/
│   │   │   ├── runner.py     # YOLO benchmark runner
│   │   │   ├── execution.py  # Phase 5 execution enforcement
│   │   │   ├── postprocessing.py  # YOLO output processing (NMS)
│   │   │   ├── hailo_metrics.py   # Hailo-specific metrics
│   │   │   ├── backends/
│   │   │   │   ├── base.py       # Backend interface
│   │   │   │   ├── pytorch.py    # PyTorch backend
│   │   │   │   ├── hailo.py      # Hailo NPU backend
│   │   │   │   └── registry.py   # Backend auto-selection
│   │   │   └── conversion/       # Hailo model conversion
│   │   │       ├── pipeline.py   # Full conversion pipeline
│   │   │       ├── onnx_export.py
│   │   │       ├── har_generator.py
│   │   │       ├── hef_compiler.py
│   │   │       ├── calibration.py  # Phase 3 calibration
│   │   │       ├── validation.py   # Phase 3 validation
│   │   │       └── cache.py        # Model caching
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
├── Hailo-Prd-Tasks.txt       # Hailo integration PRD
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

### Hailo NPU (Raspberry Pi only)
- hailo-platform >= 4.17.0 (HailoRT SDK)
- hailo-dataflow-compiler >= 3.26.0 (for model compilation)
- onnx >= 1.14.0 (ONNX export)
- onnxruntime >= 1.15.0 (ONNX validation)

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
