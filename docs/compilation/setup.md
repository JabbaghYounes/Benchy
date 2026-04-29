# Workstation Setup Status

What's already in place on this machine and what's still required to
compile HEFs.

> **Note (2026-04-29):** This doc captures the AMD/CPU workstation
> setup. After hitting Hailo's CUDA requirement for seg / pose / OBB
> compilation (see [pitfalls.md § 8](pitfalls.md)), the live compile
> path moved to an NVIDIA-equipped box — see
> [nvidia_workstation_setup.md](nvidia_workstation_setup.md) for the
> full bring-up checklist. The steps below still apply to detection
> compiles and to the wheel-installation parts; they just don't
> unblock the gap models on AMD hardware.

## ✅ Already done

- **Benchy cloned** at `/home/jt/Documents/benchy-hef/Benchy/`
- **Python venv created** at `Benchy/venv/` (Python 3.14.4)
- **Benchy installed** in editable dev mode (`pip install -e ".[dev]"`)
  — pulls in ultralytics, torch, torchvision, opencv-python, etc.
- **Test suite passing** — 196/196 in `pytest tests/`
- **Pi-side HailoRT installers bundled in repo:**
  - `Benchy/resources/hailo-8/hailort_4.23.0_arm64.deb` (for AI HAT / AI Kit)
  - `Benchy/resources/hailo-10H/hailort_5.2.0_arm64.deb` (for AI HAT+ 2)
  - These run on the Pi, not the workstation

## ❌ Still required (user must do)

### 1. Hailo Dataflow Compiler + Model Zoo (x86 wheels)

EULA-gated download from the **Hailo Developer Zone**
(<https://hailo.ai/developer-zone/>). You need:

- `hailo_dataflow_compiler-*-py3-none-linux_x86_64.whl`
- `hailo_model_zoo-*-py3-none-any.whl`

For the Hailo-10H target you need a DFC build new enough to expose
`--hw-arch hailo10h` and emit HailoRT-5 HEFs. Confirm with
`hailo compiler --help | grep hailo10h` after install.

Install into the existing venv:

```bash
source /home/jt/Documents/benchy-hef/Benchy/venv/bin/activate
pip install /path/to/hailo_dataflow_compiler-*.whl
pip install /path/to/hailo_model_zoo-*.whl
```

### 2. OS compatibility caveat

Hailo officially supports **Ubuntu 20.04 / 22.04** for the SDK. This
machine is CachyOS (Arch-based). The wheels may install but undefined
behaviour is on you. Two safer options:

- **Docker** — Hailo ships an Ubuntu-based DFC container; pull it and
  bind-mount `Benchy/` + a calibration dir
- **Ubuntu VM / second machine** — clone the venv setup there

### 3. Calibration data

Benchy's `benchmark/workloads/yolo/conversion/calibration.py` expects per-task
datasets (coco128, coco128-seg, DOTAv1, coco-pose, imagenet10) and will
download them on first compile. Pre-stage them or override
`CalibrationConfig.dataset_path` per task to point at a curated subset
(~100–1000 representative images is sufficient for INT8 calibration —
no need for full DOTAv1 / coco-pose).

### 4. (Optional) ONNX from the Pi

Per `Benchy/docs/hailo.md`, compilation is automatic — Benchy runs
`.pt → .onnx → .har → .hef` itself, so you don't strictly need to scp
ONNX files off the Pi. Only do this if you want to skip the export step
or guarantee identical ONNX inputs.

## How Benchy compiles, in practice

Once the SDK is installed, the upstream-supported way to compile is:

```bash
source Benchy/venv/bin/activate
python -m benchmark run yolo --backend hailo \
  --yolo-model yolov8n-seg.pt \
  --force-recompile --skip-validation
```

This wraps the manual `hailo parser → optimize → compiler` chain
documented in [hailo8/workflow.md](hailo8/workflow.md) and
[hailo10h/workflow.md](hailo10h/workflow.md). HEFs land in
`~/.cache/benchy/hailo/<arch>/...` and can then be moved into the repo
at the destination paths in each pipeline's `models.md`.

`--force-recompile` ignores any cached HEF; omit it once a model has
compiled cleanly.

## Recommended first command after SDK install

Smoke-test with the easiest-to-compile model before batching all 7:

```bash
python -m benchmark run yolo --backend hailo \
  --yolo-model yolov8n-seg.pt \
  --force-recompile --skip-validation
```

If that produces a `.hef` and the output tensor shapes match Benchy's
decoder, the chain is good — loop the rest.
