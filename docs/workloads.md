# Workloads

## YOLO Benchmarks

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

## LLM Benchmarks

### Model Groups

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

### Prompt Sets

*Legacy Prompts (7B+):*
- simple_qa, reasoning, code_generation, summarization, creative

*Lightweight Model Prompts (1B/3B):*
- **General Reasoning**: summarization, instruction_following, short_reasoning
- **Code Generation**: function_generation, code_completion, syntax_validation

### Benchmark Parameters (1B/3B)

- Warmup runs: 2
- Measured runs: 10
- Temperature: 0.2
- top_p: 0.95
- max_tokens: 256
- Streaming: disabled
- Prompt batching: 3 prompts per batch

### Metrics Collected

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

### Memory Requirements

| Group | Minimum Available RAM |
|-------|----------------------|
| 1B | 2 GB |
| 3B | 4 GB |
| 7B | 8 GB |
| 8B | 10 GB |
| 9B | 12 GB |

The benchmark performs memory preflight checks before loading models. If insufficient memory is detected or swap usage is required, the benchmark will abort with an error.
