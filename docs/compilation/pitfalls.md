# Pitfalls

The plan and tooling are correct. Whether a given model compiles cleanly
depends almost entirely on these.

## 1. Wrong `--hw-arch`

Most common silent failure. The `.hef` is built for a specific Hailo chip
family — compile for the wrong one and it won't run on the device at all.

- Hailo-8 → `--hw-arch hailo8`
- Hailo-8L → `--hw-arch hailo8l`
- Hailo-10H → `--hw-arch hailo10h`
- Hailo-15 → `--hw-arch hailo15`

Confirm the chip in your Pi before committing to a target. Don't trust a
default value in someone else's command snippet.

## 2. Postprocessing baked into the HEF

Benchy expects raw feature-map outputs. All of NMS, mask blending, pose
decoding, and box decoding happen host-side in
`benchmark/workloads/yolo/postprocessing.py`.

If your `.alls` script or `hailomz` flags apply `nms_postprocess`,
`change_output_activation`, or any end-of-network postprocess op:

- the HEF emits decoded detections instead of raw tensors
- Benchy's decoders silently choke or produce garbage

**Fix:** use the no-postprocess compile path. With `hailomz`, avoid the
`--end-postprocess` flag (or whatever the equivalent is in your DFC version).
Treat raw output tensors as the contract.

## 3. Wrong input/output node names

The parser auto-picking output nodes is responsible for ~90% of beginner
pain. For OBB and pose heads especially, explicitly set them:

```bash
hailo parser onnx model.onnx \
  --start-node-names <input> \
  --end-node-names <last_conv_layers>
```

Even if compile succeeds with auto-picked nodes, output tensor count and
ordering may not match what Benchy expects.

## 4. Unsupported ONNX ops

For newer YOLO variants (v11, v26) you may hit:

- unsupported layers
- dynamic shapes
- custom activations

The compiler doesn't fail gracefully — it refuses to compile. Fix by:

- editing the ONNX graph (drop offending tail ops)
- modifying export settings
- trimming end nodes earlier in the graph

Just because a model exports clean ONNX and runs in ONNX Runtime does **not**
mean it will compile in Hailo. The ONNX file is "ready to attempt compile,"
not "guaranteed to compile."

## 5. Calibration dataset overkill

You do **not** need full COCO / DOTA / coco-pose. ~100–1000 representative
images per task is enough. A small curated subset:

```
calibration/
  det/   (200 images)
  seg/   (200 images)
  pose/  (200 images)
  obb/   (200 images)
```

saves tens of GB and hours of preprocessing. Bad calibration → bad accuracy
or FPS, but "bad" comes from non-representative images, not insufficient
volume.

## 6. DFC ↔ HailoRT version mismatch

Different DFC versions emit HEFs targeting different minimum HailoRT
versions. Mismatch symptoms:

- `.hef` won't load
- runtime crashes on inference

Pair the workstation DFC version with the Pi's HailoRT version. Easiest:
upgrade the Pi to match the compiler output (the bundled
`hailort_*_arm64.deb` makes it a one-liner).

## 7. OBB heads need attention

OBB adds angle regression and custom decoding. Hailo often requires:

- modifying output nodes
- removing post-processing layers from the export

Even when compile succeeds, output ordering may differ from the ONNX export.
Verify tensor shapes against `postprocessing.py` expectations before
committing the HEF.

## 8. v26 is genuinely unproven

YOLOv26 detection is in the Zoo, so the backbone is known to compile.
Non-detection v26 heads (OBB, seg, pose) have no public confirmation that
the Ultralytics → Hailo pipeline produces clean weights.

Keep v26 variants tagged `[experimental]`. Failure ≠ regression.

## Verify-before-commit checklist

Before adding any `.hef` to the repo:

- [ ] Ran `hailortcli run model.hef` on the Pi
- [ ] Output tensor count matches the ONNX export
- [ ] Output shapes match Benchy's decoder expectations
- [ ] No NMS / mask / pose decoding baked in
- [ ] Compiled for the correct `--hw-arch`
- [ ] HailoRT version on the Pi is ≥ the DFC's minimum
