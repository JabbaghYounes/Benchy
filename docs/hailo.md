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
| v8 | Yes | Yes | Yes | Yes | Yes |
| v11 | Yes | Yes | Yes | Yes | Yes |
| v26 | Yes | Yes | Experimental | Experimental | Experimental |

**Optimized Models:**
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` (Detection)
- `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt` (Classification)
- `yolov8n-obb.pt`, `yolov8s-obb.pt`, `yolov8m-obb.pt` (OBB, Phase 3a)
- `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt` (Segmentation, Phase 3b)
- `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt` (Pose, Phase 3c)
- Similar patterns for v11 and v26.

**OBB note (Phase 3a).** v11-obb has official Hailo Model Zoo backing;
v8-obb is community-supported and well-documented. v26-obb is listed for
whitelist symmetry but is **experimental** — no public confirmation that
the Ultralytics → ONNX → HAR → HEF pipeline produces clean weights, and
hardware verification (Slice 6 of Phase 3a) is what will move it from
experimental to either Yes or No. The `_process_obb` postprocessor and
custom rotated NMS in `benchmark/workloads/yolo/postprocessing.py` work
identically across versions; the risk is at the conversion stage.

**Segmentation note (Phase 3b).** Instance segmentation on Hailo uses
the standard Ultralytics two-output head (detections + 32 mask
prototypes). Mask blending happens host-side in
`YOLOPostProcessor._process_segmentation`: the kept detections'
32-coefficient vectors are sigmoid-blended with the prototype tensor,
binarised at 0.5, and cropped to bbox at the prototype's native
resolution (~input/4). Masks are intentionally NOT serialised through
`SegmentationResult.to_dict()` — they would inflate the output JSON by
orders of magnitude. The in-process arrays are available for
mAP-with-masks validation. v26-seg is **experimental** for the same
reason as v26-obb.

**Pose note (Phase 3c).** Pose estimation on Hailo uses a single
detection-style head extended with 17 keypoints × 3 channels
`(x, y, visibility)` per anchor, totalling 56 channels per anchor for
COCO-Pose (1 class). `YOLOPostProcessor._process_pose` runs standard
axis-aligned NMS on the bbox component, then sigmoid-applies the
visibility logit so consumers see scores in [0, 1] (conventional
threshold ~0.5). Keypoint coords scale with `original_width` /
`original_height` if provided. Unlike segmentation masks, pose
keypoints are small enough (17 × 3 floats per detection) to ship in
`PoseResult.to_dict()`. v26-pose is **experimental** for the same
reason as v26-obb / v26-seg.

## Model Conversion Pipeline

Hailo requires model conversion from PyTorch to HEF format:

```
.pt (PyTorch) -> .onnx (ONNX) -> .har (Hailo Archive) -> .hef (Hailo Executable)
```

**Compilation is automatic** - models are compiled on first use and cached for subsequent runs.

## Known Limitations

1. **Supported Tasks**: All five YOLO tasks (detection, classification, OBB, segmentation, pose) clear the Hailo conversion + postprocessing pipeline. OBB shipped in Phase 3a with a custom rotated-NMS path; segmentation shipped in Phase 3b with a host-side mask-prototype blender; pose shipped in Phase 3c with a 17-keypoint decoder. v26 variants of OBB / segmentation / pose are marked experimental until hardware verification.
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
- **Calibration data**: 100 images sampled from a task-appropriate Ultralytics
  dataset:
  - Detection: `coco128` (~7 MB)
  - Segmentation: `coco128-seg` (~10 MB)
  - OBB: `DOTAv1` (~10 GB on first download — was `dota8` (8 imgs) before
    Polish 4; the small sample produced poor INT8 calibration)
  - Pose: `coco-pose` (~20 GB on first download — was `coco8-pose` (8 imgs)
    before Polish 4)
  - Classification: `imagenet10` (~50 MB)
  Override per-task via `CalibrationConfig.dataset_path` if you have a
  pre-curated subset and want to skip the full download.
- **Disk space**: ~500MB per compiled model
- **Time**: 5-30 minutes per model depending on size

**Compilation must run on x86_64 Linux.** The Hailo Dataflow Compiler
(provider of `hailo_sdk_client`) is not available on aarch64, so a
Raspberry Pi cannot compile its own HEFs. The standard workflow is to
compile on a workstation (Hailo Developer Zone licence required) and
copy the resulting `.hef` to the Pi — see the prebuilt HEF source layer
below.

## Prebuilt HEF source layer

To make the verify suite work on a Pi without a separate workstation
round-trip, the Hailo backend looks for prebuilt HEFs in two
project-controlled locations before trying to compile:

1. **`resources/hefs/`** — drop HEFs here using the convention
   `<yolo_version>_<task>_<model_size>_<arch>.hef`
   (e.g. `v8_detection_n_hailo8.hef`, `v11_pose_s_hailo10h.hef`).
   This is the canonical landing spot for HEFs compiled on a
   workstation. See `resources/hefs/NAMING.txt` for the full naming
   convention.
2. **`/usr/share/hailo-models/`** — the
   `rpicam-apps-hailo-postprocess` Debian package ships a curated
   subset of Hailo Model Zoo HEFs vetted by Raspberry Pi. The mapping
   from our `(yolo_version, task, size, arch)` tuple to its filename
   convention lives in `benchmark/workloads/yolo/conversion/hef_source.py:SYSTEM_PACKAGE_MAP`.
   On Pi OS Bookworm 2026-04 this covers `yolov8s` detection and
   `yolov8s` pose (both Hailo-8 and Hailo-8L variants); other tasks
   and sizes need a workstation compile.

If neither location has a match, the backend falls through to the
in-tree compile path, which fails fast on aarch64 with a clear error
message pointing at `resources/hefs/`. See Issue 11 in
`resources/session_issues_2026-04-27.md`.

### Sourcing HEFs from the Hailo Model Zoo

The Hailo Model Zoo's public S3 catalogue serves prebuilt HEFs over
plain HTTPS — no auth, no SDK install required. URL pattern:

```
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<version>/<arch>/<name>.hef
```

`<version>` is the Model Zoo release the HEF was compiled against
(`v2.16.0` pairs with HailoRT 4.20.x, `v2.18.0` with HailoRT 4.22+).
`<arch>` is `hailo8` or `hailo8l`. `<name>` is the Zoo's own filename
convention (`yolov8n` for YOLOv8 nano detection, `yolov8n_seg` for
the segmentation variant, `yolov8s_pose` for the pose variant — note
`s` not `n`). Browse the available list at
[hailo-ai/hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo)
under `docs/public_models/HAILO8/HAILO8_*.rst`.

After download, rename to our `<version>_<task>_<size>_<arch>.hef`
convention and drop into `resources/hefs/`. Example:

```
curl -sSL -o resources/hefs/v8_detection_n_hailo8.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/yolov8n.hef
```

**Coverage gaps in the Model Zoo (as of 2026-04-28):** no OBB HEFs
exist for any YOLO version; pose is published at sizes s and m only
(not n); segmentation / pose / OBB are not published for v11 or v26.
Those tasks require workstation compilation from `.pt`.

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
5. Pull the model on first use:
   ```bash
   curl --silent http://localhost:8000/api/pull \
     -H 'Content-Type: application/json' \
     -d '{ "model": "llama3.2:3b", "stream": true }'
   ```

### Prebuilt HEFs

The Hailo Model Zoo GenAI 5.1.1 catalogue ships precompiled HEFs for
several model families. The benchmark consumes only the llama-family
entry from this set (Issue 7 in
`resources/session_issues_2026-04-27.md` — llama-only consolidation):

| Tag (Ollama-compat) | Params | Used by benchmark? |
|---|---|---|
| `llama3.2:3b` | 3B | ✅ — `npu` profile |
| `qwen2:1.5b`, `qwen2.5-instruct:1.5b`, `qwen2.5-coder:1.5b` | 1.5B | ❌ out of scope |
| `deepseek_r1_distill_qwen:1.5b` | 1.5B | ❌ out of scope |

`tests/test_llm_npu_profile.py:HAILO_GENAI_PREBUILT_HEFS` is the canonical
whitelist of in-scope HEFs; profiles that list anything outside it fail
the test suite.

### Running

```bash
# On a Pi 5 + AI HAT+ 2 with the hailo-ollama server reachable at
# http://localhost:8000 (the default in configs/llm_benchmark.yaml).
python -m benchmark run llm --profile npu
```

The `npu` profile uses `llama3.2:3b` so the cross-platform dashboard
gets a clean "same model, two backends" comparison row when paired
with the CPU-side 3B run from `--profile drone` or `--profile full`.

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
