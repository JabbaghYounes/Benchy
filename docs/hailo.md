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

### HEF availability in `hefs-v2`

Backend-supported (table above) is a separate axis from "is there a
prebuilt HEF you can actually run." The Benchy GitHub Release
[`hefs-v2`](https://github.com/JabbaghYounes/Benchy/releases/tag/hefs-v2)
ships **32 HEFs** that the fetcher
(`scripts/fetch_prebuilt_hefs.py`) stages into `resources/hefs/` on
first setup:

- **Hailo-8 (19):** v8 det n/s/m/l/x, v11 det n/s/m, v26 det n,
  v8 seg n/s/m, v8 pose s/m, v11 pose n, v8 obb n, v11 obb n,
  v26 obb n, v26 pose n.
- **Hailo-10H (13):** v8 det n/s, v11 det n, v8 seg n, v11 seg n,
  v8 pose n/s, v11 pose n, v8 obb n, v11 obb n, v26 obb n,
  v26 seg n, v26 pose n.

`hefs-v2` adds two HEFs over `hefs-v1` (`v8 pose n_hailo10h` and
`v8 pose s_hailo10h`) — both unblocked by the
`("v8", YOLOTask.POSE)` `END_NODE_TABLE` entry +
`MODEL_SCRIPT_OVERRIDES` ALLS recipe added in the 2026-05-03
workstation session (commits in
`har_generator.py` / `hef_compiler.py`; full session log in
`resources/session_notes_2026-05-03_hefs-v2_release_prep.md`).

#### Known gaps (each understood)

| Missing HEF | Cause | Status |
|---|---|---|
| `v8 pose n_hailo8` | Hailo Model Zoo publishes pose at sizes s/m only for Hailo-8 | Use `s` instead |
| `v11 seg_hailo8` | Chip-side FPS budget overflow — compiles fine on Hailo-10H. See `docs/compilation/pitfalls.md` § 11 | h10h-only |
| `v26 seg_hailo8` | Hardware-unfittable — v26 head attention's multi-output `matmul1` cannot be ingested by Hailo-8 in any supported precision mode (six override variants attempted). See `pitfalls.md` § 12 | h10h-only |
| `v26 det_hailo10h` | Not supported by Hailo's own `yolo26n.yaml` (`supported_hw_arch: [hailo8, hailo8l]` only). See `pitfalls.md` § 13 | h8-only |
| `v8 pose m/l_hailo10h`, `v11 pose s/m/l_hailo10h` | Not yet attempted — the v8 pose `END_NODE_TABLE` patch should unblock them but they're not in the compile matrix yet | Future work |
| Classification (all three YOLO versions, both arches) | No postprocessor priority yet — backend supports it but no HEFs compiled | Future work |

See `resources/hefs/NAMING.txt` for the full inventory and
[`docs/compilation/pitfalls.md`](compilation/pitfalls.md) §§ 10-13 for
the gating issues.

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
Those tasks require workstation compilation from `.pt` — see
[docs/hef_compilation.md](hef_compilation.md) for the end-to-end
workflow, or pull the workstation-compiled HEFs from the Benchy
GitHub Release (next subsection).

### Sourcing HEFs from the Benchy GitHub Release

The gap-filler HEFs the Model Zoo doesn't ship (every Hailo-10H HEF,
every OBB HEF, every v26 HEF, v8/v11 segmentation/pose at size `n`)
are published as a curated GitHub Release. The fetcher falls back to
the release automatically when the Zoo returns 404/403:

```bash
scripts/fetch_prebuilt_hefs.py --arch hailo10h
# Tries Zoo S3 first per the existing path; for HEFs the Zoo doesn't
# publish, falls back to the release pinned by HEFS_RELEASE_TAG (in
# scripts/fetch_prebuilt_hefs.py — currently "hefs-v2"). Verifies
# SHA-256 against the release's manifest.json before staging into
# resources/hefs/. Mismatch is a hard fail; partial files are deleted.
```

The release is the canonical home for the workstation-compiled HEFs
listed in `resources/session_notes_2026-04-29_nvidia_workstation.md`
and `resources/session_notes_2026-05-03_hefs-v2_release_prep.md`. To
pin a different release version (reproducing an older verify run),
pass `--release-tag hefs-vN`. The full release catalogue, per-HEF
provenance, and compile-environment metadata (DFC version, calibration
setup, wheel SHA-256s) live on the
[release page](https://github.com/JabbaghYounes/Benchy/releases).

`scripts/fetch_prebuilt_hefs.py --source release` skips the Zoo
entirely and pulls only from the release — useful if the Zoo bucket
is rate-limiting or the release covers everything you need.
`--source zoo` is the legacy path (Zoo only). The default `--source
both` covers everything available across either source.

The setup scripts (`setup_rpi_ai_hat_plus.sh`,
`setup_rpi_ai_hat_plus_2.sh`) call the fetcher with the right
`--arch` for the detected board at the end of installation — fresh Pi
clones get HEFs staged automatically. The HW-verify runners
(`verify_ai_hat_plus[_2].sh`) gate on HEFs being present in
`resources/hefs/` before starting the sweep, so a setup that ran with
no network gets a clear remediation message ("run the fetcher first")
instead of opaque per-step compile failures.

### Restricted-egress / air-gapped setups

If the Pi can't reach `github.com` (corporate firewall, isolated
network, deliberate offline), HEFs can be staged manually instead of
running the fetcher:

1. On any machine with network access, download the HEFs you need
   from the [release page](https://github.com/JabbaghYounes/Benchy/releases)
   using the canonical `<version>_<task>_<size>_<arch>.hef` naming.
2. Transfer them to the Pi (scp / USB / etc.) into `resources/hefs/`.
3. Run `verify_ai_hat_plus[_2].sh` as normal. The runtime resolver
   (`benchmark/workloads/yolo/conversion/hef_source.py`) checks
   `resources/hefs/` first; the fetcher is convenience, not a hard
   dependency.

The same workflow applies to manually-compiled HEFs from a workstation
that aren't yet in the release — drop them into `resources/hefs/`
with canonical names and the runtime picks them up unchanged.

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

> **Important — verified 2026-04-28 on Pi OS Bookworm.** `scripts/setup_rpi_ai_hat_plus_2.sh` is apt-only and Raspberry Pi's apt repo currently caps at **HailoRT 4.20.0** (no `hailo-h10-all`, no 5.x packages). Running the setup script will report "SUCCESS" but leave you on 4.x with the chip invisible (`/dev/hailo0` missing, `hailortcli scan` empty). The HailoRT 5.x install for the Hailo-10H must currently be done manually from the Hailo Developer Zone. The bundled `resources/hailo-10H/hailort_5.2.0_arm64.deb` is **userspace only** — it covers the `hailort` package, but the kernel driver and Hailo-10 firmware (now bundled together inside `hailort-pcie-driver_5.x.x_all.deb`) and the Python wheel must be downloaded separately.

1. **Download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) (free account)**, picking HailoRT 5.x for arm64 / Hailo-10:
   - `hailort_<ver>_arm64.deb` — userspace lib + `hailortcli`
   - `hailort-pcie-driver_<ver>_all.deb` — DKMS kernel driver **and** Hailo-10H firmware (bundled into `/lib/firmware/hailo/hailo10h/`; no separate `hailofw` package in 5.x)
   - `hailort-<ver>-cp311-cp311-linux_aarch64.whl` — Python bindings; pick the wheel whose `cpXY` matches the venv's Python version (`venv/bin/python --version`)
   - `hailo_gen_ai_model_zoo_<ver>_arm64.deb` — `hailo-ollama` server + LLM HEF manifests; only needed for the `npu` LLM profile
2. **Remove the old 4.x stack first** — otherwise both `hailo_pci 4.x` and `hailo1x_pci 5.x` modules co-exist on boot and the old one may claim PCIe device `1e60:45c4` first:
   ```bash
   sudo apt-get remove hailo-all hailo-dkms hailofw python3-hailort
   ```
3. **Install in dependency order** — driver first because the userspace and model-zoo .debs both depend on `hailort`:
   ```bash
   sudo dpkg -i hailort-pcie-driver_<ver>_all.deb       # DKMS compiles against running kernel
   sudo dpkg -i hailort_<ver>_arm64.deb                  # userspace
   sudo dpkg -i hailo_gen_ai_model_zoo_<ver>_arm64.deb  # optional, for npu profile
   ```
4. **Reboot.** The 4.x kernel module stays resident in memory until reboot; the chip won't enumerate under the new driver until then.
5. **Verify post-reboot:**
   ```bash
   lsmod | grep hailo                  # expect ONLY hailo1x_pci (NOT hailo_pci)
   ls /dev/h1x-*                       # expect /dev/h1x-0 — HailoRT 5.x renamed
                                       # the device node from /dev/hailo* to /dev/h1x-*
   hailortcli scan                     # expect a Hailo device line (pci/0001:01:00.0)
   hailortcli fw-control identify      # expect non-empty body, "Device Architecture: HAILO10H",
                                       # "Firmware Version: 5.x.x (release,app)"
   ```
6. **Install the Python wheel into the project venv** (replaces the `python3-hailort` removed in step 2):
   ```bash
   source venv/bin/activate
   pip install hailort-<ver>-cpXY-cpXY-linux_aarch64.whl
   python -c "import hailort; print(hailort.__version__)"
   ```
7. **Start hailo-ollama** for the `npu` LLM profile: `hailo-ollama serve` (binds `:8000`). Config at `~/.config/hailo-ollama/hailo-ollama.json`; HEF cache at `~/.local/share/hailo-ollama/models/`.
8. **Pull a model on first use** (note the field is `model`, not `name` — `name` triggers a 500 null-pointer error from oatpp):
   ```bash
   curl --silent http://localhost:8000/api/pull \
     -H 'Content-Type: application/json' \
     -d '{ "model": "llama3.2:1b", "stream": true }'
   ```

### Prebuilt HEFs

The HailoRT 5.3.0 GenAI Model Zoo catalogue ships precompiled HEFs for the following — **verified 2026-04-28** by hitting `/api/tags` on the running hailo-ollama server after installing `hailo_gen_ai_model_zoo_5.3.0_arm64.deb`:

| Tag (Ollama-compat) | Params | Family | Used by |
|---|---|---|---|
| `llama3.2:1b` | 1B | llama | **`npu` profile** |
| `qwen2:1.5b` | 1.5B | qwen | not used (non-llama; outside project policy) |
| `qwen2.5:1.5b` | 1.5B | qwen | not used (non-llama; outside project policy) |
| `qwen2.5-coder:1.5b` | 1.5B | qwen | not used (non-llama; outside project policy) |
| `qwen3:1.7b` | 1.7B | qwen | not used (non-llama; outside project policy) |
| `deepseek_r1:1.5b` | 1.5B | deepseek | not used (non-llama; outside project policy) |

**Catalogue history:** The 5.1.1 zoo had `llama3.2:3b`; Hailo dropped it in 5.3.0 and added `llama3.2:1b` instead. No 7B HEFs have ever shipped in any HailoRT GenAI release — Hailo positions the Hailo-10H for ~1-2B edge inference, not 7B-class workloads. This is why cross-backend (CPU vs NPU) comparison is only available at the **1B** size group; 3B and 7B llamas remain CPU-only by virtue of Hailo not shipping HEFs at those sizes.

The project's llama-only policy means only `llama3.2:1b` is exercised on the Hailo-10H. The other prebuilt HEFs remain documented because they are in the Model Zoo catalogue, but they are not part of the verify sweep.

`tests/test_llm_npu_profile.py:HAILO_GENAI_PREBUILT_HEFS` is the canonical whitelist; profiles that list anything outside it fail the test suite.

### Running

```bash
# On a Pi 5 + AI HAT+ 2 with the hailo-ollama server reachable at
# http://localhost:8000 (the default in configs/llm_benchmark.yaml).
python -m benchmark run llm --profile npu
```

The `npu` profile uses `llama3.2:1b` — the only llama-family prebuilt HEF in the HailoRT 5.3.0 GenAI Model Zoo catalogue, in line with the project's llama-only policy. To exercise other catalogue HEFs (qwen, deepseek), add them under a separate profile in `configs/llm_benchmark.yaml`; they are intentionally not part of the verify sweep.

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
