# Hailo-10H Compilation Workflow

Run on an x86_64 Ubuntu 20.04 / 22.04 workstation. Targets a Pi with a
Hailo-10H module (Raspberry Pi AI HAT+).

> The DFC build must include `hailo10h` as a supported `--hw-arch`. Older DFC
> versions do not. Check `hailo --version` and `hailo compiler --help` before
> starting.

## Step-by-step

### 1. Export to ONNX

Hailo does not ingest PyTorch directly.

```bash
python export.py --weights model.pt --include onnx
```

Bake these in at export time so they match what Benchy expects:

- input `640×640`
- `batch=1`
- `opset 11`

### 2. Parse → HAR

```bash
hailo parser onnx model.onnx \
  --start-node-names <input_node> \
  --end-node-names <output_node> \
  --hw-arch hailo10h
```

Output: `model.har`. Set `--end-node-names` explicitly for OBB/pose heads —
see [../pitfalls.md](../pitfalls.md).

### 3. Optimize (INT8 quantization)

```bash
hailo optimize model.har \
  --hw-arch hailo10h \
  --calib-set-path ./calibration_dataset \
  --model-script model.alls
```

Output: `model_optimized.har`. ~100–1000 representative images is enough;
full COCO/DOTA is overkill.

### 4. Compile → HEF

```bash
hailo compiler model_optimized.har \
  --hw-arch hailo10h \
  --performance
```

Output: `model.hef`. This is what the Pi consumes.

## Shortcut: `hailomz compile`

For Zoo-registered models:

```bash
hailomz compile <name> \
  --ckpt model.onnx \
  --hw-arch hailo10h \
  --yaml config.yaml
```

Wraps parse + optimize + compile in one call. Falls back to raw DFC for
unregistered models — see [../tools.md](../tools.md).

> Some Zoo YAMLs were authored before 10H was a target. If `hailomz compile`
> rejects `--hw-arch hailo10h`, drop to the raw DFC commands above.

## Benchy wrapper pipeline

The Benchy `compile` subcommand wraps the same DFC calls, runs the
full `.pt → .onnx → .har → .hef` pipeline, and stages the result
into `resources/hefs/` with canonical naming in one step:

```bash
git clone <your Benchy fork>
cd Benchy
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pip install /path/to/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
pip install /path/to/hailo_model_zoo-2.18.0-py3-none-any.whl

# Single model
python -m benchmark compile --hw-arch hailo10h --model yolov8n-seg.pt

# Batch — gap models + detection (no hailo10h detection HEFs ship in
# the Zoo today, so AI HAT+ 2 needs them compiled locally)
scripts/compile_workstation_hefs.sh --arch hailo10h --include-detection

# Or compile for both AI HAT+ chips in one sweep
scripts/compile_workstation_hefs.sh --arch both
```

Commit `resources/hefs/*.hef`, push, pull on the Pi, run verify.

## Deploy to the Pi

```bash
scp model.hef pi@device:/home/pi/models/
```

Run via HailoRT (what Benchy wraps):

```bash
hailortcli run model.hef
```

## Verify before committing

Before checking a `.hef` into the repo, run it once with `hailortcli run` and
confirm:

- output tensor count matches the ONNX export
- output shapes match what Benchy's decoder expects
- no NMS / mask-blending / pose-decoding ops were baked in
  (Benchy does all postprocessing host-side)

If shapes don't match, fix at compile time — not by patching
`benchmark/workloads/yolo/postprocessing.py`. See [../pitfalls.md](../pitfalls.md).

## Time and disk budget

- First compile per model: 5–30 minutes (10H builds can be slightly faster
  on the optimize step thanks to wider activation handling)
- All 7 missing models: roughly 1–3.5 hours
- Calibration data: ~30 GB if pulling full datasets — use small curated
  subsets (~100–1000 representative images) to avoid this
- Repo growth: ~50–100 MB after adding 7 HEFs. Combined with the Hailo-8
  set, plan for ~150–200 MB total. Consider `git-lfs` once the combined set
  passes ~30 HEFs
