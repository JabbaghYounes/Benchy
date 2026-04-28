# Hailo-8 / 8L Compilation Workflow

Run on an x86_64 Ubuntu 20.04 / 22.04 workstation. Targets a Pi with a
Hailo-8 or Hailo-8L module.

> Substitute `hailo8` ↔ `hailo8l` in every `--hw-arch` flag below to match the
> chip on the Pi. Don't mix them — the binaries are not interchangeable.

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
  --hw-arch hailo8
```

Output: `model.har`. Set `--end-node-names` explicitly for OBB/pose heads —
see [../pitfalls.md](../pitfalls.md).

### 3. Optimize (INT8 quantization)

```bash
hailo optimize model.har \
  --hw-arch hailo8 \
  --calib-set-path ./calibration_dataset \
  --model-script model.alls
```

Output: `model_optimized.har`. Calibration quality drives final accuracy —
~100–1000 representative images is enough; full COCO/DOTA is overkill.

### 4. Compile → HEF

```bash
hailo compiler model_optimized.har \
  --hw-arch hailo8 \
  --performance
```

Output: `model.hef`. This is what the Pi consumes.

## Shortcut: `hailomz compile`

For models registered in the Model Zoo YAML catalog (YOLOv5, YOLOv8
detection, ResNet, etc.):

```bash
hailomz compile <name> \
  --ckpt model.onnx \
  --hw-arch hailo8 \
  --yaml config.yaml
```

Wraps parse + optimize + compile in one call. Falls back to raw DFC for
unregistered models — see [../tools.md](../tools.md).

## Benchy wrapper pipeline

The Benchy CLI wraps the same DFC calls and writes to
`~/.cache/benchy/hailo/`:

```bash
git clone <your Benchy fork>
cd Benchy
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pip install /path/to/hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
pip install /path/to/hailo_model_zoo-2.18.0-py3-none-any.whl

# Compile each model — pipeline runs .pt → .onnx → .har → .hef.
# Not benchmark runs (no Hailo device on workstation); pipeline stops
# after HEF generation.
for model in yolov8n-obb yolo11n-obb yolo11n-seg yolo11n-pose \
             yolo26n-obb yolo26n-seg yolo26n-pose; do
  python -m benchmark run yolo --backend hailo \
    --hw-arch hailo8 \
    --yolo-model "${model}.pt" --force-recompile --skip-validation || true
done
```

Then copy HEFs into the repo with the destination paths from
[models.md](models.md), commit, push, pull on the Pi, run verify.

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

- First compile per model: 5–30 minutes
- All 7 missing models: roughly 1–3.5 hours
- Calibration data: ~30 GB if pulling full datasets — use small curated
  subsets (~100–1000 representative images) to avoid this
- Repo growth: ~50–100 MB after adding 7 HEFs. Git-manageable for now;
  consider `git-lfs` once the set passes ~15 HEFs
