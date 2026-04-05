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

### Lightweight LLM Benchmarks (1B/3B Models)

```bash
# Run benchmarks for all lightweight models (1B and 3B)
python -m benchmark run llm --profile lightweight

# Run benchmarks for specific parameter group
python -m benchmark run llm --model-group 1B
python -m benchmark run llm --model-group 3B

# Run a specific lightweight model
python -m benchmark run llm --model llama3.2:1b

# Run with custom output directory
python -m benchmark run llm --profile lightweight --output ./lightweight_results
```

**Note:** Lightweight models (1B/3B) use different benchmark parameters than standard models:
- 2 warmup runs (vs 3 for 7B+)
- Non-streaming inference
- Temperature 0.2, top_p 0.95
- Prompt batching (3 prompts per batch)

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
