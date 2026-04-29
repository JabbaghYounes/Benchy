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
| LLM Size | All, 1B, 3B, 7B | Filter by model size (filter is dynamically populated from the data; under the llama-only policy only these three groups appear) |
| Parameter Group | All, 1B, 3B, 7B | Filter by parameter group (same as size) |
| Architecture | All, Dense, MoE | Filter by model architecture (only `Dense` appears under the llama-only policy; the `MoE` option is reserved for non-llama profiles like granite3.1-moe) |
| Specialization | All, General, Code | Filter by model specialization (only `General` appears under the llama-only policy; the `Code` option is reserved for code-specialised models like starcoder2) |

**Visual Indicators:**
- **MoE Badge** (purple): Indicates Mixture of Experts models. Not shown under the current llama-only policy; preserved in the dashboard code so non-llama models can be re-introduced without re-deriving the badge logic.
- **Code Badge** (orange): Indicates code-specialized models. Same caveat as MoE — preserved but not used by the shipped profiles.
- **Parameter Group Badge**: Color-coded by size (green=1B, blue=3B/7B).

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

# Drone profile: detection-only at 1280 with VisDrone validation. `tasks`
# stays Hailo-compatible across every listed YOLO version.
drone:
  yolo_versions: ["v8", "v11", "v26"]
  tasks: ["detection"]
  model_sizes: ["n", "s", "m"]
  input_resolution: 1280
  datasets:
    detection: VisDrone.yaml

# Drone-full profile: exercises every Phase-3-unlocked task on the Hailo
# path. Detection on VisDrone, OBB on DOTA, plus segmentation and pose on
# COCO defaults. Sizes drop to n/s only because at 1280 with five tasks
# the total runtime grows fast.
drone_full:
  yolo_versions: ["v8", "v11", "v26"]
  tasks: ["detection", "obb", "segmentation", "pose"]
  model_sizes: ["n", "s"]
  input_resolution: 1280
  datasets:
    detection: VisDrone.yaml
    obb: DOTAv1.yaml
    segmentation: coco128-seg.yaml
    pose: coco8-pose.yaml
```

### LLM Configuration (`configs/llm_benchmark.yaml`)

```yaml
benchmark:
  warmup_runs: 3        # The runner auto-overrides to 2 for 1B/3B models
                        # via LLMBenchmarkConfig.for_lightweight_model
  measured_runs: 10

generation:
  temperature: 0.0      # Auto-overridden to 0.2 for 1B/3B models
  top_p: 1.0            # Auto-overridden to 0.95 for 1B/3B models
  top_k: 1
  seed: 42
  max_tokens: 256

default:
  model_groups: ["7B"]
  models: ["llama2:7b"]
  # Quantization sweep: each base model × each quant becomes one Ollama tag.
  # Default template is `{base}-{quant}`; use `{base}-chat-{quant}` for
  # llama2 tags or `{base}-instruct-{quant}` for instruct variants.
  # The shipped default omits these to keep the SD-card-backed Pi run lean;
  # add them back to opt back in to a quant sweep.
  # quants: ["q4_K_M", "q5_K_M", "q8_0"]
  # quant_tag_template: "{base}-chat-{quant}"

full:
  model_groups: ["1B", "3B", "7B"]

# Drone profile: drone-use-case prompts (scene description, target ID,
# mission preflight, telemetry, hazard reasoning). `prompt_set: drone`
# tells the runner to ignore the top-level `prompts:` block and pull from
# the curated DRONE_PROMPTS set in benchmark/workloads/llm/runner.py.
drone:
  model_groups: ["7B"]
  models: ["llama2:7b"]
  prompt_set: drone

# NPU profile (Hailo-10H only): runs the LLM workload through HailoRT
# GenAI's REST endpoint instead of Ollama. Uses the only llama with a
# prebuilt HEF in the HailoRT 5.3.0 GenAI Model Zoo.
npu:
  api_base: "http://localhost:8000"
  backend: "hailo-10h"
  npu_metrics: true
  model_groups: ["1B"]
  models: ["llama3.2:1b"]
  prompt_set: drone
```

There is no `lightweight:` profile in the shipped configs — the CLI only
accepts `default / full / drone / drone_full / npu / compare`. The 1B/3B-friendly
generation settings (warmup=2, temperature=0.2, top_p=0.95, prompt
batching) are applied automatically by
`LLMBenchmarkConfig.for_lightweight_model` whenever the runner sees a 1B
or 3B model in any profile.

The `quantization` column on `*_llm.csv` carries the actual level reported
by Ollama's `/api/show` (e.g. `Q4_K_M`, `Q5_K_M`, `Q8_0`), making quant
sweeps groupable in the dashboard without further config.
