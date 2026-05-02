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
succeeds. Bring-up validated end-to-end on an RTX 2080 Ti node the
night of 2026-04-29 → 04-30; six gap-model HEFs landed (one Hailo-8,
five Hailo-10H). Full session log:
`resources/session_notes_2026-04-29_nvidia_workstation.md`.

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

**Ubuntu 24.04+ note.** 24.04 ships Python 3.12, not 3.10, and the
above `apt install python3.10*` will fail with "Unable to locate
package". Add the deadsnakes PPA first:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
```

Or, if you don't want to add the PPA, install via `uv`:

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

CUDA toolkit isn't strictly required at the system level — but the
kernel driver must be present and `nvidia-smi` must work. The DFC
wheel does **not** bundle a complete CUDA runtime in practice; the
TF/JAX-side CUDA libs come from the `tensorflow[and-cuda]` /
`jax[cuda12]` extras you'll install in step 5b below. Without those
extras, `tf.config.list_physical_devices("GPU")` is empty, the
optimizer logs `[warning] no available GPU`, opt level drops to 0,
and mapping fails the same way it does on AMD/CPU-only.

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

### Step 5b — Install matching CUDA-enabled TensorFlow + JAX

The DFC wheel pins TF/JAX versions but installs them in the *CPU-only*
form. The optimizer needs the GPU form, which lives in PyPI extras.
Pick the versions the DFC wheel pinned (look at `pip show tensorflow`
output for the version) and reinstall with the CUDA extras:

```bash
./venv-compile-h8/bin/pip install \
    "tensorflow[and-cuda]==<pinned-tf-version>" \
    "jax[cuda12]==<pinned-jax-version>"
```

Verify GPU is now visible from inside the venv:

```bash
./venv-compile-h8/bin/python -c \
    "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
./venv-compile-h8/bin/python -c \
    "import jax; print(jax.default_backend())"
```

Expect non-empty `[PhysicalDevice(name='/physical_device:GPU:0', ...)]`
for TF and `gpu` for JAX.

### Step 5c — Restore torch's NCCL ABI

`tensorflow[and-cuda]` pulls in `nvidia-nccl-cu12`, which installs
`libnccl.so.2` to `nvidia/nccl/lib/` — the same path torch's
`nvidia-nccl-cu13` already populates. The cu12 install overwrites
cu13's `libnccl.so.2`, breaking torch's `libtorch_cuda.so` (it was
linked against the cu13 ABI symbol `ncclCommWindowDeregister` which
cu12-2.21.5 lacks). Symptom: `import torch` works but the ONNX
exporter that sits behind stage 1 of the compile pipeline fails to
load with `undefined symbol: ncclCommWindowDeregister`, which the
runner reports as "Ultralytics not available for ONNX export".

Restore the cu13 NCCL at the shared path:

```bash
./venv-compile-h8/bin/pip install --force-reinstall --no-deps nvidia-nccl-cu13
```

Single-GPU TF doesn't invoke NCCL collective ops, so it remains
happy with the cu13 lib at runtime.

### Sanity-check the venv

```bash
./venv-compile-h8/bin/python -c "import hailo_sdk_client; print(hailo_sdk_client.__version__)"
./venv-compile-h8/bin/python -m pytest tests/ -q
```

Expect 277/277 once the Ultralytics weight cache is warm. **First
run on a cold machine returns 276/277** — `test_compile_cmd.py::test_python_m_benchmark_propagates_exit_code` invokes a
real `python -m benchmark compile yolov8n.pt` which Ultralytics-
downloads `yolov8n.pt` on first call and trips the 30s timeout.
Re-run `pytest tests/test_compile_cmd.py -q` once the cache is warm
and it passes.

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
./venv-compile-h10h/bin/pip install \
    "tensorflow[and-cuda]==<pinned-tf-version>" \
    "jax[cuda12]==<pinned-jax-version>"
./venv-compile-h10h/bin/pip install --force-reinstall --no-deps nvidia-nccl-cu13
```

Same pygraphviz / opencv workarounds apply if needed. The CUDA-extras
+ NCCL-restore steps from 5b/5c are required here too.

## Step 7 — Stage val2017 calibration images

The compile path expects ≥ 1024 calibration images for Hailo's
Bias Correction passes to actually run. The compile CLI's
`--calibration-data-path` flag lets you point at a directory of
images instead of triggering Ultralytics' ~27 GB COCO auto-download
(of which only val2017 — 1 GB — is actually used).

```bash
mkdir -p ~/Documents/datasets/coco-val/images
curl --retry 3 --retry-delay 5 -o /tmp/val2017.zip \
    http://images.cocodataset.org/zips/val2017.zip
unzip /tmp/val2017.zip -d ~/Documents/datasets/coco-val/images/
ls ~/Documents/datasets/coco-val/images/val2017 | wc -l
# Expect: 5000
```

`--retry 3 --retry-delay 5` tolerates the transient DNS / connection
hiccups the COCO mirror occasionally throws (seen on the 2026-04-29
NVIDIA bring-up).

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
- `[info] Loading model script:` followed by two ALLS commands:
  `model_optimization_flavor(optimization_level=2, compression_level=1)`
  and `post_quantization_optimization(bias_correction, policy=enabled)`.
  The pipeline emits these automatically (see `hef_compiler.py`); if
  they're missing, `runner.load_model_script` was not called and the
  SDK silently uses its defaults.
- `[info] Bias Correction is done` runs to completion (NOT
  `Bias Correction skipped`). At `optimization_level=2` the SDK's
  flavor would *not* enable bias_correction by itself — Hailo's
  `mo_config.py` uses an if/elif chain where each level enables one
  pass, so level 2 picks Finetune but skips Bias Correction.
  `post_quantization_optimization(bias_correction, policy=enabled)`
  forces it back on, which is what makes the seg/pose/OBB heads
  end up with 8-bit biases.
- `[info] Mapping succeeded` or no `[error] Mapping Failed`.
- File on disk: `/tmp/benchy-hef-canary/v11_segmentation_n_hailo8.hef`,
  size > 1 MB.

If `Mapping Failed` still appears with the same `16x4 not supported
in activation2` errors as on AMD, two possible causes — try them in
this order:

1. **GPU not visible to the DFC.** Check `nvidia-smi` from inside the
   venv; verify step 5b's GPU-visibility one-liners print a real
   device. Without GPU the optimizer drops opt level to 0 and biases
   stay 16-bit.
2. **End-node truncation list is too deep.** The `END_NODE_TABLE` in
   `benchmark/workloads/yolo/conversion/har_generator.py` must cut at
   the raw Conv outputs that the host postprocessor expects, not at
   the deep post-processing layers (Sigmoid / Concat / Mul). Compare
   the `(version, task)` entry against
   `venv-compile-*/lib/python3.10/site-packages/hailo_model_zoo/cfg/networks/<name>.yaml`
   `parser.nodes` for any Hailo-published model. If the SDK's own
   "use these end node names" hint suggests deep layers, **don't
   trust it** — that's the documented failure mode (Issue 6 in the
   2026-04-29 NVIDIA session notes). Cutting at the deep layers pulls
   the high-precision-bias activations onto the chip subgraph, which
   doesn't fit Hailo-8 L2.

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

`--compression-level` defaults to 1 (8-bit biases via Bias
Correction). Override to 0 only for debugging — level 0 leaves
biases at 16-bit and fails Hailo-8 chip mapping for seg/pose/OBB.
Level 2 enables Adaround + Finetune on top, at the cost of longer
compile time.

**Expect a partial-success sweep until OBB and v26 entries are
added to `END_NODE_TABLE`.** As of 2026-04-30 the table covers
v8/v11 detection + segmentation and v11 pose; OBB across all three
versions and every v26 task fail at HAR generation with "no entry
in END_NODE_TABLE for (..., ...)" or at mapping with
`concat14/18/23 errors` from the parse-end-node-hint fallback. See
the 2026-04-29 NVIDIA session notes (Issue 10) for the procedure
to derive the missing entries from ONNX inspection. Also note:
`yolo11n-seg` does not fit Hailo-8's compute budget regardless of
end-nodes — it's a chip-side capacity miss, hailo10h-only.

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
- `resources/session_issues_2026-04-27.md` — bring-up issue
  catalogue from the original AI HAT+ Pi session.
- `resources/session_notes_2026-04-29_nvidia_workstation.md` —
  full account of the NVIDIA bring-up that validated this doc;
  the 11 documented issues there are why most of the "common
  fixes" callouts in this doc exist.

## What gets transferred manually vs. via git

| Artifact | How | Why |
|---|---|---|
| Hailo `.whl` files (~1 GB total) | scp / rsync / USB | EULA, plus exceeds GitHub's 100 MB single-file limit |
| `coco/val2017/*.jpg` (~1 GB) | curl from `images.cocodataset.org` | not redistributable in repo |
| Source code | `git clone` | tracked normally |
| Compiled `*.hef` files | `git push` after compile | small, repo-tracked, downstream Pi pulls them via plain git |
