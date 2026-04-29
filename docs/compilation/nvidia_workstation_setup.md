# NVIDIA workstation bring-up for HEF compilation

End-to-end checklist for setting up a fresh NVIDIA-equipped Linux box
to compile HEFs for the AI HAT+ (Hailo-8 / 8L, DFC 3.33.1) and
AI HAT+ 2 (Hailo-10H, DFC 5.3.0) Pis.

## Why NVIDIA, not AMD or CPU-only

Hailo's Dataflow Compiler (DFC) needs CUDA to run optimization
level ≥ 1 — that's where Bias Correction, Adaround, and Finetune
encoding compress weights and biases to 8-bit. On a CPU-only or
AMD/ROCm box, the DFC drops to optimization level 0, leaves biases
at 16-bit, and the seg / pose / OBB heads fail chip mapping on
Hailo-8 with errors like:

```
DW resources calculation failed: more than 1 subclusters are needed
for 16bit L2 biases and contexts at activation2
activation2 failed on kernel validation: 16x4 is not supported
Agent infeasible
```

Confirmed on the AMD workstation in the 2026-04-29 session — see the
end of that session's commit chain (`5aac3ec` and earlier).

NVIDIA + CUDA unblocks Bias Correction → 8-bit biases → mapping
succeeds.

## Prerequisites

- Linux x86_64 (Ubuntu 22.04 or similar)
- NVIDIA GPU with a recent driver (any CUDA 11.x or 12.x DFC supports)
- Python 3.10 (the only version Hailo officially supports; install
  via system packages or `uv` if your distro defaults to a newer one)
- ~30 GB free disk (1 GB val2017 + ~5 GB venvs + ~1 GB wheel cache +
  HEF outputs)
- SSH access to a workstation that already has the four Hailo wheels
  downloaded — saves a re-pull from the EULA-gated Developer Zone

## Step 1 — System packages

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3.10-dev \
    python3-pip git build-essential \
    graphviz graphviz-dev \
    unzip curl
```

If your distro doesn't ship Python 3.10, install via `uv`:

```bash
uv python install 3.10
ln -sf ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 ~/.local/bin/
```

## Step 2 — Verify NVIDIA driver + CUDA

```bash
nvidia-smi
```

Should print a table with the GPU model and driver version. If it
fails, install the NVIDIA proprietary driver per your distro's
instructions before continuing.

CUDA toolkit isn't strictly required at the system level — Hailo's
DFC bundles its own CUDA runtime — but the kernel driver must be
present and `nvidia-smi` must work.

## Step 3 — Clone the repo

```bash
git clone https://github.com/JabbaghYounes/Benchy.git
cd Benchy
git checkout hef
```

## Step 4 — Transfer the Hailo wheels (one-time)

The four Hailo wheels are EULA-gated and **never committed to the
repo** (gitignored under `resources/hailo-sdk/*.whl`). Transfer them
once from a workstation that already has them.

From the workstation that already has the wheels:

```bash
scp resources/hailo-sdk/*.whl jt@<nvidia-host>:~/Benchy/resources/hailo-sdk/
```

Or via USB stick if the boxes aren't on the same network:

```bash
# Source workstation:
cp resources/hailo-sdk/*.whl /run/media/jt/<usbstick>/

# NVIDIA box:
cp /run/media/jt/<usbstick>/*.whl ~/Benchy/resources/hailo-sdk/
```

Verify on the NVIDIA box:

```bash
ls -lh ~/Benchy/resources/hailo-sdk/
```

Expect four files:

```
hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl  (~489 MB)
hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl   (~499 MB)
hailo_model_zoo-2.18.0-py3-none-any.whl                   (~750 KB)
hailo_model_zoo-5.3.0-py3-none-any.whl                    (~890 KB)
```

If the wheels aren't transferable from another workstation, fall
back to a direct download from <https://hailo.ai/developer-zone/>
under your own Hailo account.

## Step 5 — Build the AI HAT+ compile venv (Hailo-8 / 8L)

```bash
cd ~/Benchy

python3.10 -m venv venv-compile-h8
./venv-compile-h8/bin/pip install --upgrade pip wheel setuptools
./venv-compile-h8/bin/pip install -e ".[dev]"
./venv-compile-h8/bin/pip install \
    resources/hailo-sdk/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
./venv-compile-h8/bin/pip install \
    resources/hailo-sdk/hailo_model_zoo-2.18.0-py3-none-any.whl
```

### Common fixes during venv build

**pygraphviz fails on graphviz 14+ / GCC 15+** with `incompatible
pointer types`:

```bash
CFLAGS="-Wno-error=incompatible-pointer-types" \
    ./venv-compile-h8/bin/pip install pygraphviz
```

**opencv-python conflicts** with DFC's `numpy<2` pin:

```bash
./venv-compile-h8/bin/pip install "opencv-python<4.10"
```

### Sanity-check the venv

```bash
./venv-compile-h8/bin/python -c "import hailo_sdk_client; print(hailo_sdk_client.__version__)"
./venv-compile-h8/bin/python -m pytest tests/ -q
```

The pytest run should be 277/277 passing.

## Step 6 — Build the AI HAT+ 2 compile venv (Hailo-10H, optional)

Only needed if you'll compile for the AI HAT+ 2 Pi. The 4.x and 5.x
Hailo SDK lines share the top-level `hailo_sdk_client` package so
they cannot coexist in one venv — hence two separate venvs.

```bash
python3.10 -m venv venv-compile-h10h
./venv-compile-h10h/bin/pip install --upgrade pip wheel setuptools
./venv-compile-h10h/bin/pip install -e ".[dev]"
./venv-compile-h10h/bin/pip install \
    resources/hailo-sdk/hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl
./venv-compile-h10h/bin/pip install \
    resources/hailo-sdk/hailo_model_zoo-5.3.0-py3-none-any.whl
```

Same pygraphviz / opencv workarounds apply if needed.

## Step 7 — Stage val2017 calibration images

The compile path expects ≥ 1024 calibration images for Hailo's
Bias Correction passes to actually run. The compile CLI's
`--calibration-data-path` flag lets you point at a directory of
images instead of triggering Ultralytics' ~27 GB COCO auto-download
(of which only val2017 — 1 GB — is actually used).

```bash
mkdir -p ~/Documents/datasets/coco-val/images
curl -o /tmp/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip /tmp/val2017.zip -d ~/Documents/datasets/coco-val/images/
ls ~/Documents/datasets/coco-val/images/val2017 | wc -l
# Expect: 5000
```

HTTP not HTTPS — COCO's image CDN doesn't support HTTPS for these
specific URLs (cert mismatch). Acceptable on a trusted network; see
the 2026-04-29 session log for the prior discussion.

## Step 8 — Canary compile

Pick a known-failing-on-AMD model and verify it now compiles:

```bash
BENCHY_VENV=venv-compile-h8 scripts/compile_workstation_hefs.sh \
    --arch hailo8 \
    --models yolo11n-seg.pt \
    --calibration-data-path ~/Documents/datasets/coco-val/images/val2017 \
    --output-dir /tmp/benchy-hef-canary
```

What to look for in the output:

- `[info] Found GPU` or absence of `[warning] no available GPU`
  near "Starting Model Optimization".
- `[info] Bias Correction` runs (the line should NOT say "skipped").
- `[info] Adaround` runs (NOT skipped).
- `[info] Finetune encoding` runs (NOT skipped).
- `[info] Mapping succeeded` or no `[error] Mapping Failed`.
- File on disk: `/tmp/benchy-hef-canary/v11_segmentation_n_hailo8.hef`,
  size > 1 MB.

If `Mapping Failed` still appears with the same `16x4 not supported
in activation2` errors as on AMD, GPU isn't being picked up by the
DFC — check `nvidia-smi` from inside the venv and verify the DFC
log says it found a GPU during optimization.

## Step 9 — Full sweep

Once the canary works, compile the gap models for both architectures:

### Hailo-8 / 8L (AI HAT+ 26 / 13 TOPS)

```bash
BENCHY_VENV=venv-compile-h8 scripts/compile_workstation_hefs.sh \
    --arch hailo8 \
    --calibration-data-path ~/Documents/datasets/coco-val/images/val2017 \
    --output-dir resources/hefs/
```

Compiles the 7 default gap models:
`yolo11n-seg`, `yolo11n-pose`, `yolov8n-obb`, `yolo11n-obb`,
`yolo26n-obb`, `yolo26n-seg`, `yolo26n-pose`.

Each model takes ~5–30 min on CPU-fallback, faster with CUDA.

### Hailo-10H (AI HAT+ 2)

```bash
BENCHY_VENV=venv-compile-h10h scripts/compile_workstation_hefs.sh \
    --arch hailo10h \
    --include-detection \
    --calibration-data-path ~/Documents/datasets/coco-val/images/val2017 \
    --output-dir resources/hefs/
```

`--include-detection` adds detection at sizes n/s on top of the
seven gap models — Hailo's S3 publishes detection prebuilts for
hailo8 but not hailo10h, so that gap also needs filling on the 10H
side.

## Step 10 — Commit and push the HEFs

The compiled artefacts land in `resources/hefs/` with the canonical
filename pattern `<version>_<task>_<size>_<arch>.hef`. They're
small enough to commit (each typically 5–50 MB).

```bash
git add resources/hefs/v*.hef
git status
git commit -m "hefs: workstation compile of gap models for hailo8 and hailo10h"
git push origin hef
```

After pushing, pull on each Pi and run `verify_ai_hat_plus.sh` /
`verify_ai_hat_plus_2.sh` — the runtime will pick up the new HEFs
automatically via `benchmark/workloads/yolo/conversion/hef_source.py`.

## Reference

- `resources/hailo-sdk/README.md` — wheel-management details.
- `docs/compilation/setup.md` — AMD workstation status from the
  2026-04-29 session (most steps mirror, except the GPU-required
  bits).
- `docs/compilation/pitfalls.md` — known compilation failures.
- `docs/compilation/end_node_truncation_plan.md` — rationale for
  the END_NODE_TABLE in `har_generator.py`.
- `resources/session_issues_2026-04-27.md` — bring-up issue
  catalogue from the original AI HAT+ Pi session.

## What gets transferred manually vs. via git

| Artifact | How | Why |
|---|---|---|
| Hailo `.whl` files (~1 GB total) | scp / rsync / USB | EULA, plus exceeds GitHub's 100 MB single-file limit |
| `coco/val2017/*.jpg` (~1 GB) | curl from `images.cocodataset.org` | not redistributable in repo |
| Source code | `git clone` | tracked normally |
| Compiled `*.hef` files | `git push` after compile | small, repo-tracked, downstream Pi pulls them via plain git |
