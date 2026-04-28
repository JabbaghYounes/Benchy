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
| **GPU (optional)** | Calibration / quantisation runs ~10× faster with CUDA. CPU-only works but adds ~minutes per model. |
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

```bash
for model in yolov8n-obb yolo11n-obb yolo11n-seg yolo11n-pose \
             yolo26n-obb yolo26n-seg yolo26n-pose; do
  echo "=== Compiling ${model} ==="
  python -m benchmark run yolo --backend hailo \
    --yolo-model "${model}.pt" --force-recompile --skip-validation \
    || true   # ignore the post-compile inference failure
done
```

Per-model time: 5–30 minutes. Total: roughly 1–3.5 hours.

**First-time only:** the OBB compiles trigger a DOTAv1 download (~10
GB), and the pose compiles trigger coco-pose (~20 GB). Subsequent
compiles reuse the cached datasets.

### 4. Stage HEFs to the repo

```bash
mkdir -p resources/hefs

# v8 / v11 — straight rename
cp ~/.cache/benchy/hailo/hailo8/v8/obb/yolov8n-obb/model.hef         resources/hefs/v8_obb_n_hailo8.hef
cp ~/.cache/benchy/hailo/hailo8/v11/obb/yolo11n-obb/model.hef        resources/hefs/v11_obb_n_hailo8.hef
cp ~/.cache/benchy/hailo/hailo8/v11/segmentation/yolo11n-seg/model.hef  resources/hefs/v11_segmentation_n_hailo8.hef
cp ~/.cache/benchy/hailo/hailo8/v11/pose/yolo11n-pose/model.hef      resources/hefs/v11_pose_n_hailo8.hef

# v26 — same pattern, only stage the ones that compiled cleanly
cp ~/.cache/benchy/hailo/hailo8/v26/obb/yolo26n-obb/model.hef        resources/hefs/v26_obb_n_hailo8.hef
cp ~/.cache/benchy/hailo/hailo8/v26/segmentation/yolo26n-seg/model.hef  resources/hefs/v26_segmentation_n_hailo8.hef
cp ~/.cache/benchy/hailo/hailo8/v26/pose/yolo26n-pose/model.hef      resources/hefs/v26_pose_n_hailo8.hef
```

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

- **Repo size growth.** Each HEF is ~5–15 MB. Seven new HEFs add
  ~50–100 MB on top of the existing 29 MB in `resources/hefs/`. Still
  manageable as plain blobs; if the staged set grows past ~15 HEFs,
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
