# Output Files

## Raw Results (per run)

```
results/
├── bench_YYYYMMDD_HHMMSS_XXXXXXXX.json     # Complete benchmark run
├── bench_YYYYMMDD_HHMMSS_XXXXXXXX_yolo.csv # YOLO results table
└── bench_YYYYMMDD_HHMMSS_XXXXXXXX_llm.csv  # LLM results table
```

## Aggregated Results

```
results/aggregated/
├── agg_YYYYMMDD_HHMMSS.json           # Complete aggregated results
├── agg_YYYYMMDD_HHMMSS_yolo.csv       # Aggregated YOLO metrics
├── agg_YYYYMMDD_HHMMSS_llm.csv        # Aggregated LLM metrics
└── agg_YYYYMMDD_HHMMSS_platforms.csv  # Platform summaries
```

## Dashboard

The HTML dashboard includes:
- System overview with platform comparison
- YOLO performance charts (latency, FPS, accuracy, power)
- YOLO scaling analysis (throughput vs model size, latency vs accuracy trade-off)
- LLM performance charts (tokens/sec, TTFT)
- LLM efficiency charts (memory usage, TPS vs memory trade-off)
- Stability/variance analysis
- Raw data tables with model metadata badges
- Data download links

**Dashboard Filters:**

| Filter | Options | Description |
|--------|---------|-------------|
| Platform | All, Jetson Orin Nano, RPi AI HAT+, etc. | Filter by hardware platform |
| YOLO Version | All, v8, v11, v26 | Filter YOLO results by version |
| Task | All, Detection, Classification, etc. | Filter by YOLO task type |
| LLM Size | All, 1B, 3B, 7B, 8B, 9B | Filter by model size |
| Parameter Group | All, 1B, 3B, 7B, 8B, 9B | Filter by parameter group (same as size) |
| Architecture | All, Dense, MoE | Filter by model architecture |
| Specialization | All, General, Code | Filter by model specialization |

**Visual Indicators:**
- **MoE Badge** (purple): Indicates Mixture of Experts models (e.g., granite3.1-moe)
- **Code Badge** (orange): Indicates code-specialized models (e.g., starcoder2)
- **Parameter Group Badge**: Color-coded by size (green=1B, blue=3B/7B/8B, red=9B)

Open in browser:
```bash
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
