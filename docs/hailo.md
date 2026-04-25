# Hailo NPU Integration

The benchmark suite supports the Hailo NPUs shipped on Raspberry Pi AI HATs. The AI HAT+ ships in two variants (which Benchy treats as the same `RPI_AI_HAT_PLUS` platform — both use the same HailoRT 4.x driver, packages, and HEF format):

| Board | NPU | TOPS (peak) | Quantization | HailoRT | Onboard memory | Host |
|-------|-----|-------------|--------------|---------|----------------|------|
| Raspberry Pi AI HAT+ (13 TOPS) | Hailo-8L | 13 | INT8 | 4.x | none (uses host) | Pi 5 |
| Raspberry Pi AI HAT+ (26 TOPS) | Hailo-8  | 26 | INT8 | 4.x | none (uses host) | Pi 5 |
| Raspberry Pi AI HAT+ 2         | Hailo-10H | 40 | INT4 / INT8 | 5.x | 8 GB SDRAM | Pi 5 |

Notes:

- The 26 TOPS AI HAT+ automatically configures the Pi 5's PCIe link to Gen 3 to expose the full bandwidth.
- The Hailo-8 silicon has a typical power consumption of ~2.5 W; the AI HAT+ 2 ships with a heatsink for the Hailo-10H. All three boards conform to the Raspberry Pi HAT+ specification.
- Only the AI HAT+ 2 has onboard SDRAM, so it is the only HAT that can host local LLMs / VLMs (Raspberry Pi rates it for models up to ~6 B parameters); the AI HAT+ variants are vision-only.

HEF files compiled for the Hailo-8 family (8 / 8L) are not compatible with the Hailo-10H, and vice versa — the backend maintains separate caches per family.

References: [AI HAT+ product brief (PDF)](https://datasheets.raspberrypi.com/ai-hat/ai-hat-plus-product-brief.pdf), [Raspberry Pi AI HAT documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html), [Hailo-8 accelerator overview](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/).

## Supported Configurations

| YOLO Version | Detection | Classification | Segmentation | Pose | OBB |
|--------------|-----------|----------------|--------------|------|-----|
| v8 | Yes | Yes | No | No | No |
| v11 | Yes | Yes | No | No | No |
| v26 | Yes | Yes | No | No | No |

**Optimized Models:**
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` (Detection)
- `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt` (Classification)
- Similar patterns for v11 and v26

## Model Conversion Pipeline

Hailo requires model conversion from PyTorch to HEF format:

```
.pt (PyTorch) -> .onnx (ONNX) -> .har (Hailo Archive) -> .hef (Hailo Executable)
```

**Compilation is automatic** - models are compiled on first use and cached for subsequent runs.

## Known Limitations

1. **Supported Tasks Only**: Segmentation, pose estimation, and OBB tasks are NOT supported on Hailo NPU due to architectural constraints.
2. **INT8 Quantization**: All Hailo models use INT8 quantization. Minor accuracy differences compared to FP32/FP16 models are expected.
3. **Model Size**: Larger models (l, x variants) may have longer compilation times and higher memory requirements.
4. **No CPU Fallback**: When using `--backend hailo`, the benchmark will NOT fall back to CPU if Hailo is unavailable. This ensures benchmark integrity.

## Important: CPU Inference is Invalid

**CPU inference on Hailo-equipped platforms is NOT a valid benchmark configuration.**

When benchmarking Raspberry Pi with Hailo:
- Always use `--backend hailo` explicitly, OR
- Let auto-detection select Hailo (default on RPi + AI HAT+)
- CPU fallback is disabled by default to prevent misleading results

## Compilation Requirements

First-time model compilation requires:
- **HailoRT SDK** installed (version 4.17+ recommended)
- **Hailo Dataflow Compiler** for .har -> .hef conversion
- **Calibration data**: Uses 100 images from COCO validation set
- **Disk space**: ~500MB per compiled model
- **Time**: 5-30 minutes per model depending on size

## Cache Management

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

## LLM on Hailo-10H

Only the AI HAT+ 2 (Hailo-10H, 8 GB onboard SDRAM) can host LLMs on the
NPU itself. The AI HAT+ variants (Hailo-8 / 8L) are vision-only and have
no onboard memory, so LLMs always run on the Pi 5 CPU on those boards.

### Runtime

Benchy targets HailoRT GenAI's **Ollama-compatible REST API**, exposed by
the [`hailo-ollama`](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/gen_ai_apps/hailo_ollama/README.md)
binary that ships in the [`hailo-apps`](https://github.com/hailo-ai/hailo-apps)
repo. The runner keeps using the existing `OllamaClient` and just points
`api_base` at the GenAI server (default port **8000** per the hailo-ollama
README; adjust if your install binds elsewhere). TTFT, tokens/sec,
prompt/eval token counts, and truncation detection all work without
client changes — `hailo-ollama` mirrors Ollama's `/api/generate` streaming
shape including `eval_count` and `prompt_eval_count`.

### Setup (high level)

1. `sudo apt install dkms hailo-h10-all` — Hailo-10H driver + HailoRT
   (already covered by `scripts/setup_rpi_ai_hat_plus_2.sh`).
2. `git clone https://github.com/hailo-ai/hailo-apps.git && cd hailo-apps && sudo ./install.sh && source setup_env.sh`
3. `sudo dpkg -i hailo_gen_ai_model_zoo_<version>_arm64.deb` (the
   hailo-ollama README pins this to `5.1.1` at the time of writing — adjust
   to whatever your HailoRT 5.x install ships with).
4. Run `hailo-ollama` to start the server. Config at
   `~/.config/hailo-ollama/hailo-ollama.json`; HEF cache at
   `~/.local/share/hailo-ollama/models/`.
5. Pull a model on first use:
   ```bash
   curl --silent http://localhost:8000/api/pull \
     -H 'Content-Type: application/json' \
     -d '{ "model": "qwen2:1.5b", "stream": true }'
   ```

### Prebuilt HEFs

The Hailo Model Zoo GenAI 5.1.1 catalogue ships precompiled HEFs for:

| Tag (Ollama-compat) | Params |
|---|---|
| `qwen2:1.5b`, `qwen2.5-instruct:1.5b`, `qwen2.5-coder:1.5b` | 1.5B |
| `deepseek_r1_distill_qwen:1.5b` | 1.5B |
| `llama3.2:3b` | 3B |

`tests/test_llm_npu_profile.py:HAILO_GENAI_PREBUILT_HEFS` is the canonical
whitelist; profiles that list anything outside it fail the test suite.

### Running

```bash
# On a Pi 5 + AI HAT+ 2 with the hailo-ollama server reachable at
# http://localhost:8000 (the default in configs/llm_benchmark.yaml).
python -m benchmark run llm --profile npu
```

The `npu` profile starts with the smallest HEF (`qwen2:1.5b`) so the
pipeline is validated end-to-end on a fast model before scaling up. Add
larger tags to `configs/llm_benchmark.yaml` once a smaller one has
published clean numbers.

### Output

Every `LLMResult` from the `npu` profile is tagged with:

- `backend = "hailo-10h"` (`ollama-cpu` for the regular Ollama path)
- `hailort_version` — recorded once per measured loop via `hailortcli --version`
- `npu_power_watts` — read from `/sys/class/hwmon/*/power1_input`; on a
  Pi 5 + AI HAT+ 2 during NPU inference this is approximately the AI HAT+
  subsystem power, not a chip-level NPU reading
- `npu_utilization_percent` — currently always `None` on HailoRT 5.x.
  HailoRT exposes utilization only via the interactive `hailortcli monitor`
  TUI; no scriptable probe exists yet. `benchmark/backends/hailo_utils.py:get_npu_utilization_percent`
  is the single place to wire one in when it lands.

### Platform gating

The runner aborts a `--profile npu` run on anything other than
`Platform.RPI_AI_HAT_PLUS_2` so you don't silently fall back to Ollama-CPU
under the wrong backend label. To override (e.g. when developing against
a remote HailoRT GenAI server), pass `--platform rpi_ai_hat_plus_2`
explicitly.

## Cross-Platform Comparison

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
