# Workloads

## YOLO Benchmarks

**Versions:** v8, v11, v26

**Tasks:**
- Detection
- Segmentation — Hailo-supported on v8/v11 (Phase 3b); v26 experimental.
- Pose estimation — Hailo-supported on v8/v11 (Phase 3c); v26 experimental.
- Oriented Bounding Box (OBB) — Hailo-supported on v8/v11 (Phase 3a); v26 experimental.
- Classification

**OBB output format.** OBB benchmarks return `OrientedBox` objects
(`benchmark/workloads/yolo/postprocessing.py:OrientedBox`) with fields
`cx, cy, w, h, angle_rad, confidence, class_id, class_name`. Angle is
canonical: `angle_rad ∈ [-π/2, π/2]` regardless of which Ultralytics
checkpoint emitted it (the head's native convention varies by version).
Default class names come from DOTA-15 (`plane`, `ship`, `large vehicle`,
…) and the dataset is `dota8.yaml` for both calibration and validation.
For production runs at scale, swap to a larger DOTA val subset rather
than the bundled 8-image sample.

**Segmentation output format.** Seg benchmarks return `SegmentationResult`
objects (`postprocessing.py:SegmentationResult`) with fields `bbox,
confidence, class_id, class_name, mask`. The mask is a boolean numpy
array cropped to the bbox at the prototype's native resolution
(~input/4). It is **not** serialised through `to_dict()` — to keep
result JSON files reasonable in size, only `has_mask` and a
`mask_pixel_count` summary are emitted. Use the in-process `mask`
attribute for mAP-with-masks validation or visualisation. Default class
names come from COCO-80 (the same set yolov8-seg / yolo11-seg are
trained on); calibration and validation use `coco128-seg.yaml` per the
existing dataset map in `benchmark/workloads/yolo/runner.py`.

**Pose output format.** Pose benchmarks return `PoseResult` objects
(`postprocessing.py:PoseResult`) with fields `bbox, confidence,
class_id, class_name, keypoints`. `keypoints` is a `(K, 3)` numpy
array — for COCO-Pose, K = 17 with rows
`(x, y, visibility)` in the order
`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`.
Visibility is the post-sigmoid score in [0, 1]; values above ~0.5 are
conventionally treated as "visible". Unlike segmentation masks,
keypoints **are** serialised through `to_dict()` (17 × 3 floats per
detection is small). The default model is `yolov8n-pose.pt` (or
v11/v26 equivalents), trained on COCO-Pose with a single
"person" class. Calibration and validation use `coco8-pose.yaml`.

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

| Group | Model | Architecture | Backend |
|-------|-------|--------------|---------|
| **1B** | `llama3.2:1b` | Dense | Ollama (CPU); also the `npu` profile target (HEF in HailoRT 5.3.0 GenAI Model Zoo — only llama HEF that ships) |
| **3B** | `llama3.2:3b` | Dense | Ollama (CPU only — had a HEF in HailoRT 5.1.1 zoo, dropped in 5.3.0) |
| **7B** | `llama2:7b` | Dense | Ollama (CPU only — no 7B HEF ships in any HailoRT GenAI release) |

The project follows a **llama-only policy**: one model per size group, all from the llama family. Other families (qwen, mistral, gemma, dolphin, olmo, granite, starcoder, sailor) and other size groups (1.5B, 8B, 9B) were dropped on 2026-04-27 to fit an SD-card-backed Pi 5; see `resources/session_notes_2026-04-27.md` for the rationale.

**Important Constraints:**
- Models are **only compared within the same parameter group**.
- The `1B` group is the only one with both a CPU (Ollama) and NPU (Hailo-10H HEF) backend, so it is the natural axis for cross-backend comparisons in the dashboard. The 3B and 7B groups are CPU-only because Hailo doesn't ship HEFs at those sizes (verified against the HailoRT 5.3.0 GenAI Model Zoo on 2026-04-28).

### Prompt Sets

*Legacy Prompts (7B+):*
- simple_qa, reasoning, code_generation, summarization, creative

*Lightweight Model Prompts (1B/3B):*
- **General Reasoning**: summarization, instruction_following, short_reasoning
- **Code Generation**: function_generation, code_completion, syntax_validation

*Drone Prompts (drone profile):*
- **scene_description** — describe contents of an aerial frame
- **target_identification** — distinguish vehicle classes from altitude
- **mission_preflight** — preflight checks for a delivery route
- **telemetry_interpretation** — return-to-base reasoning over live telemetry
- **hazard_reasoning** — hazard call-out and recommended action

Selected via `prompt_set: drone` on a profile (the shipped `drone` profile in
`configs/llm_benchmark.yaml` does this). The runner ignores the top-level
`prompts:` block in the YAML when a profile sets `prompt_set`.

### Quantization Sweep

Profiles can declare `quants:` and a `quant_tag_template` to expand
`models × quants` into Ollama tags at run time. The shipped `default`
profile no longer enables a quant sweep (it pulls a single chat-quant set
totalling ~16 GB, which doesn't fit on an SD-card-backed Pi); to opt in,
add the fields back to a profile:

```yaml
# Example — re-enable a quant sweep on the default profile
default:
  model_groups: ["7B"]
  models: ["llama2:7b"]
  quants: ["q4_K_M", "q5_K_M", "q8_0"]
  quant_tag_template: "{base}-chat-{quant}"  # llama2 chat tags
```

The default template is `{base}-{quant}` (works for most modern Ollama tags
where the variant infix is already in `models`). Use `{base}-chat-{quant}`
for llama2-style tags where the chat infix is implicit. The runner records
the actual quantization level reported by Ollama's `/api/show` into
`LLMResult.quantization`, so the column in the CSV reflects what was
loaded — not just the requested label.

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
| `backend` | `ollama-cpu` / `ollama-cuda` / `hailo-10h` — dashboard grouping key for split-by views |
| `npu_utilization_percent` | NPU utilization over the measured loop (currently `None` on HailoRT 5.x — see `docs/hailo.md`) |
| `npu_power_watts` | AI HAT+ subsystem power during the NPU measured loop |
| `hailort_version` | HailoRT version label, captured once per LLMResult on `hailo-10h` runs |

### Memory Requirements

| Group | Minimum Available RAM |
|-------|----------------------|
| 1B | 2 GB |
| 3B | 4 GB (Ollama-CPU); on Hailo-10H the model resides in the AI HAT+ 2's onboard 8 GB SDRAM and host RAM is mostly free |
| 7B | 8 GB |

The benchmark performs memory preflight checks before loading models. If insufficient memory is detected or swap usage is required, the benchmark will abort with an error.
