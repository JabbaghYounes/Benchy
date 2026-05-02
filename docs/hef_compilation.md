# Compiling HEFs on an x86_64 workstation

The Hailo Dataflow Compiler is x86_64 Linux only — the Pi cannot
compile its own HEFs. To unblock verify-suite steps that the Hailo
Model Zoo doesn't publish HEFs for (OBB, v11 segmentation/pose, v26
non-detection variants), compile on a workstation and stage the
resulting `.hef` files in `resources/hefs/` on the Pi.

This document covers the workflow end-to-end. Read [docs/hailo.md](hailo.md)
"Prebuilt HEF source layer" first for the runtime side.

## What you need

| Requirement | Notes |
|---|---|
| **x86_64 Linux** | Ubuntu 20.04 or 22.04 LTS strongly preferred — Hailo's SDK is fussy about distros. CentOS / RHEL 8 also supported. |
| **Hailo AI Software Suite** | Or the standalone `hailo_dataflow_compiler-*.whl` + `hailo_model_zoo-*.whl`. Both EULA-gated downloads from [Hailo Developer Zone](https://hailo.ai/developer-zone/). |
| **Python 3.10 or 3.11** | DFC wheels are version-specific. Match the venv to the wheel filename (`cp310` / `cp311`). |
| **~50 GB free disk** | Calibration datasets for OBB (DOTAv1, ~10 GB) and pose (coco-pose, ~20 GB) are large. Cached at `~/.cache/benchy/hailo/`. |
| **NVIDIA GPU + CUDA** | **Required** for seg / pose / OBB. Without CUDA the DFC drops to optimization level 0, biases stay 16-bit, and chip mapping fails on Hailo-8 (`DW resources calculation failed: more than 1 subclusters needed for 16bit L2 biases at activation2`). Plain detection compiles work CPU-only. AMD GPUs do not work — Hailo doesn't support ROCm. See [docs/compilation/nvidia_workstation_setup.md](compilation/nvidia_workstation_setup.md) for the full bring-up. |
| **Network** | First compile per task family pulls calibration data from Ultralytics. |

## What we need to compile

Seven HEFs the Hailo Model Zoo does not publish, ranked by likelihood
of a clean compile:

| YOLO version | Task | Source `.pt` | Compile confidence |
|---|---|---|---|
| v11 | segmentation | `yolo11n-seg.pt` | **High** — v11 head ≈ v8, well-supported |
| v11 | pose | `yolo11n-pose.pt` | **High** — same reasoning |
| v8 | obb | `yolov8n-obb.pt` | **Medium-high** — OBB heads supported by SDK; minor head-tweaks may be needed |
| v11 | obb | `yolo11n-obb.pt` | **Medium-high** — same |
| v26 | obb | `yolo26n-obb.pt` | **Medium** — yolo26 backbone works (detection is in the Zoo); non-detection heads on v26 unproven, tagged `[experimental]` in the verify suite |
| v26 | seg | `yolo26n-seg.pt` | **Medium** — same |
| v26 | pose | `yolo26n-pose.pt` | **Medium** — same |

If a v26 compile fails, leave the `[experimental]` tag in the verify
suite — it's not a regression, just an unproven combination.

## Step-by-step

### 1. Install the SDK

On the x86_64 workstation, in a fresh venv:

```bash
python -m venv ~/venv-hailo
source ~/venv-hailo/bin/activate
pip install /path/to/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
pip install /path/to/hailo_model_zoo-2.18.0-py3-none-any.whl
python -c "import hailo_sdk_client; print(hailo_sdk_client.__version__)"
```

Or run the AI Software Suite installer (`./hailo8_ai_sw_suite_2025-10.run`)
which sets up the SDK + dependencies in one shot. Either path works.

### 2. Clone Benchy and install in editable mode

```bash
git clone <your Benchy fork>
cd Benchy
pip install -e ".[dev]"
```

The conversion pipeline lives in
`benchmark/workloads/yolo/conversion/` and orchestrates `.pt → .onnx →
.har → .hef`. Running `python -m benchmark run yolo` on the workstation
will execute the conversion stages but **fail at the inference stage**
because there's no Hailo device on the workstation — that's expected
and fine. The HEF lands in `~/.cache/benchy/hailo/` either way.

### 3. Compile each missing HEF

The recommended workstation entry point is the dedicated
`compile` subcommand (added on the `hef` branch). Unlike the
benchmark `run` flow, it bypasses the runtime Hailo backend, so it
works on a workstation that has the Dataflow Compiler but no HailoRT
or Hailo device. After each compile it stages the HEF directly into
`resources/hefs/` with the canonical
`<version>_<task>_<size>_<arch>.hef` filename the Pi-side runtime
expects:

```bash
# Single model
python -m benchmark compile --hw-arch hailo8 --model yolov8n-obb.pt

# Single model with a pre-staged val2017 directory (recommended) —
# avoids triggering Ultralytics' full coco auto-download (~27 GB) when
# only val2017 (~1 GB, ~5000 images) is consumed for calibration.
python -m benchmark compile --hw-arch hailo8 --model yolo11n-seg.pt \
    --calibration-data-path ~/Documents/datasets/coco-val/images/val2017

# Batch (continues on failure; per-model summary at the end)
python -m benchmark compile --hw-arch hailo8 \
  --models yolov8n-obb.pt,yolo11n-obb.pt,yolo11n-seg.pt,yolo11n-pose.pt,\
yolo26n-obb.pt,yolo26n-seg.pt,yolo26n-pose.pt

# Or use the driver script that batches both archs in one sweep
scripts/compile_workstation_hefs.sh --arch both
scripts/compile_workstation_hefs.sh --arch hailo10h --include-detection \
    --calibration-data-path ~/Documents/datasets/coco-val/images/val2017
```

Per-model time: 5–30 minutes. Total: roughly 1–3.5 hours.

The compile CLI's `--calibration-set-size` defaults to **1024** so
Hailo's bias-correction passes (Bias Correction / Adaround / Finetune
encoding) actually run. Below 1024 the optimizer drops to level 0 and
biases stay at 16-bit — which then fails chip mapping on Hailo-8 for
seg / pose / OBB heads. Override to a smaller value only when
iterating fast on a known-good model.

The compile CLI's `--compression-level` defaults to **1** (8-bit
biases via Bias Correction). Level 0 is the SDK default but leaves
biases at 16-bit and fails chip mapping for seg/pose/OBB heads on
Hailo-8 with `16x4 not supported in activation*`. Level 2 enables
Adaround + Finetune on top, at the cost of longer compile time.
The pipeline emits an ALLS model script
(`model_optimization_flavor(...)` +
`post_quantization_optimization(bias_correction, policy=enabled)`)
to force bias_correction on regardless of the flavor's default —
necessary because Hailo's optimization-level cases are mutually
exclusive (level 2 picks Finetune but skips Bias Correction without
the explicit ALLS command).

**First-time only:** without `--calibration-data-path`, Ultralytics'
auto-download pulls `coco.yaml` (~27 GB: train2017 19 GB + val2017
1 GB + test2017 7 GB) for SEG, plus DOTAv1 (~10 GB) for OBB and
coco-pose (~27 GB) for POSE. The `--calibration-data-path` escape
hatch points the loader at any directory of images on disk and skips
the auto-download entirely. The cache key embeds the dataset
identity, so different paths get different `.npz` caches (no stale
hits). See `benchmark/workloads/yolo/conversion/calibration.py`.

> **Legacy fallback.** Older docs recommended
> `python -m benchmark run yolo --backend hailo --skip-validation || true`
> as a workaround. That path is gated by the runtime backend's
> `is_available()` check (which fails without HailoRT / a Hailo device)
> and never actually worked on a typical compile box. Use `compile`
> instead.

### 4. Stage HEFs to the repo

`compile` already stages the HEF — there's no separate copy step.
The default `--output-dir` is `resources/hefs/`. Override only if
you need to stash builds elsewhere (e.g. `--output-dir ~/staging` for
manual review before committing).

Naming convention reference: `resources/hefs/NAMING.txt`. The
filename pattern is `<yolo_version>_<task>_<size>_<arch>.hef` —
the source layer (`benchmark/workloads/yolo/conversion/hef_source.py`)
matches against this exactly.

### 5. Commit, push, pull on the Pi

```bash
# On workstation
git add resources/hefs/*.hef
git commit -m "hefs: stage workstation-compiled HEFs for OBB / v11 / v26"
git push

# On Pi
git pull
ls -lh resources/hefs/
```

The verify script picks them up automatically — no code changes needed.

### 6. Run verify on the Pi

```bash
source venv/bin/activate
./scripts/verify_ai_hat_plus.sh
```

If a HEF was compiled with a Dataflow Compiler version that targets a
newer HailoRT than what's installed on the Pi, the load will fail with
an explicit version-mismatch error. If that happens, install the
bundled HailoRT 4.23 upgrade pack from `resources/hailo-8/`:

```bash
sudo dpkg -i resources/hailo-8/hailort_4.23.0_arm64.deb \
             resources/hailo-8/hailort-pcie-driver_4.23.0_all.deb
sudo systemctl restart hailort.service  # if applicable
```

## Caveats and gotchas

- **Repo size growth.** Each HEF is ~4–55 MB. The staged set is
  currently 19 HEFs across hailo8 + hailo10h (sum ~290 MB). Still
  manageable as plain blobs; if the staged set grows past ~30 HEFs
  or any single HEF exceeds GitHub's 100 MB single-file ceiling,
  migrate to git-LFS.
- **HailoRT version pairing.** DFC v3.33.1 → HEFs target HailoRT
  4.22+. The Pi runs 4.20.0 by default. Either upgrade the Pi runtime
  (bundled .deb in `resources/hailo-8/`) or use an older DFC matched
  to 4.20 if you specifically need to keep the Pi pinned.
- **`--force-recompile` is required** if the cache already has a HEF
  for the same `(version, task, size)` triple. Without it, the
  conversion pipeline short-circuits.
- **`--skip-validation` is required.** Validation runs inference,
  which fails on a workstation with no Hailo device. The HEF is still
  produced and cached.
- **v26 anything is unproven.** Compile may fail with a head-related
  error. If so: don't fight it, leave the `[experimental]` tag, the
  verify suite already treats those as advisory and they don't gate
  exit code.

## Alternatives if you don't have a workstation

- **Fetch what the Hailo Model Zoo publishes.**
  `scripts/fetch_prebuilt_hefs.py --arch both` downloads detection /
  segmentation / pose HEFs the Zoo publishes for hailo8 and hailo10h
  and stages them with canonical naming directly into
  `resources/hefs/`. No license, no SDK, no x86 box needed —
  network only. Doesn't help with OBB or v26 non-detection (Zoo
  doesn't publish those), but covers the common cases for free.
- **Borrow / rent x86_64 Linux.** AWS / GCP / Lambda Labs all rent
  Ubuntu boxes by the hour. The compile session takes 1–4 hours. No
  GPU needed for the SDK.
- **Hailo Compilation Service.** Hailo offers a hosted compile API for
  customers — check the Developer Zone for current availability.
- **Skip the unsupported tasks.** The verify suite's continue-on-failure
  semantics mean unblocked steps still run. Two of the six remaining
  blockers are OBB on v8/v11 (drone-relevant); the rest are
  v11-non-detection (more easily backfilled with v8 equivalents).

## Reference: Hailo SDK CLI for ad-hoc compiles

If you'd rather drive the SDK directly without going through Benchy's
conversion pipeline:

```bash
# 1. Export Ultralytics .pt to ONNX
yolo export model=yolov8n-obb.pt format=onnx imgsz=640 opset=11

# 2. Parse ONNX → HAR
hailo parser onnx yolov8n-obb.onnx --net-name yolov8n_obb \
  --hw-arch hailo8 --output-dir ./har/

# 3. Optimise (INT8 quantisation against calibration dataset)
hailo optimize ./har/yolov8n_obb.har \
  --hw-arch hailo8 \
  --calib-set-path /path/to/dotav1/calibration/ \
  --output-dir ./har/

# 4. Compile HAR → HEF
hailo compiler ./har/yolov8n_obb_optimized.har \
  --hw-arch hailo8 \
  --output-dir ./hef/
```

Benchy's `benchmark/workloads/yolo/conversion/{onnx_export,har_generator,hef_compiler}.py`
modules wrap exactly these calls with the calibration-dataset
plumbing already wired up — using the wrapper is usually less
error-prone than the manual flow.

## When you're done

Update [resources/session_issues_2026-04-27.md](../resources/session_issues_2026-04-27.md)
"Issue 11 — Don't reintroduce this" with which HEFs you successfully
compiled, which v26 entries (if any) failed, and the DFC + HailoRT
versions used. That's the trail future-you will look at the next time
the verify suite has a missing-HEF gap.
