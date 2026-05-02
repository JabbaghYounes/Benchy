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

## 8. CPU-only / AMD compile box can't produce gap-model HEFs

Hailo's DFC (3.33.x and 5.3.x both) needs **CUDA** to run optimization
level ≥ 1 — that's where the Bias Correction / Adaround / Finetune
encoding passes compress weights and biases to 8-bit. Without a
working CUDA GPU, the DFC drops to optimization level 0, those passes
are skipped, and biases stay at 16-bit. On Hailo-8 the chip allocator
then can't fit the seg / pose / OBB head's depthwise conv with 16-bit
biases, and you see:

```
[warning] Reducing optimization level to 0 ... because there's no
          available GPU
[info]    Bias Correction skipped
[info]    Adaround skipped
[info]    Finetune encoding skipped
...
[error]   Mapping Failed (allocation time: Xs)
No successful assignments: concat18 / concat20 / activation2 errors:
    Agent infeasible
    DW resources calculation failed: more than 1 subclusters are
    needed for 16bit L2 biases and contexts at activation2
    activation2 failed on kernel validation: 16x4 is not supported in
    activation2
```

This is a hardware requirement, not a calibration / model-script /
end-node-truncation issue. More calibration data, smaller input
resolution, or different end-nodes won't fix it. AMD GPUs with ROCm
also do not work — Hailo's DFC binds to the CUDA runtime explicitly.

**Fix:** move compilation to an NVIDIA-equipped Linux box. See
[nvidia_workstation_setup.md](nvidia_workstation_setup.md) for the
full bring-up. Plain detection compiles work fine CPU-only — it's
specifically the seg / pose / OBB head that needs the bias-correction
passes to fit Hailo-8 silicon.

The fetcher (`scripts/fetch_prebuilt_hefs.py`) already covers the
detection HEFs the Hailo Model Zoo publishes, so a CPU-only box can
still ship the full detection coverage without a CUDA dependency —
the gap is exactly the seg / pose / OBB models the Zoo doesn't
prebuild.

## 9. v26 is genuinely unproven

YOLOv26 detection is in the Zoo, so the backbone is known to compile.
Non-detection v26 heads (OBB, seg, pose) have no public confirmation that
the Ultralytics → Hailo pipeline produces clean weights.

Keep v26 variants tagged `[experimental]`. Failure ≠ regression.

## 10. End-node truncation depth — cut at raw Conv, not post-processing

This is the actual blocker behind most `16x4 not supported in
activation*` mapping failures (see also #4 and #8). The
`END_NODE_TABLE` in
`benchmark/workloads/yolo/conversion/har_generator.py` must list
the raw `cv*.X.X.2/Conv` outputs of each YOLO head, not the deeper
post-processing layers (`Sigmoid`, `Concat`, `Sigmoid_1`, `Mul`,
`Mul_3`).

The deep-layer cut compiles fine through HAR generation but pulls
the high-precision-bias activations onto the chip subgraph; mapping
then fails on Hailo-8 with:

```
activation1/activation2 failed on kernel validation: 16x4 is not supported
DW resources calculation failed: more than 1 subclusters are needed for
    16bit L2 biases
Agent infeasible (× hundreds)
```

**Fix:** verify the `END_NODE_TABLE` entry against the corresponding
Hailo Model Zoo YAML at
`venv-compile-*/lib/python3.10/site-packages/hailo_model_zoo/cfg/networks/<name>.yaml`
under `parser.nodes`. For gap models with no published YAML (v11
seg / pose; all OBB; v26 non-detection), derive by analogy with the
nearest published version + verify in ONNX:

```python
import onnx
m = onnx.load("yolov8n-seg.onnx")
print([n.name for n in m.graph.node if "/cv" in n.name])
```

**Don't trust the SDK parser's own "use these end node names"
hint** — when it does extract one, it tends to suggest the deep
post-processing layers, which is the wrong cut. Cross-check against
the host-side decoder in `benchmark/workloads/yolo/postprocessing.py`
to confirm the Conv outputs match what the decoder expects.

## 11. `yolo11n-seg` does not fit Hailo-8 (chip-side capacity)

Verified 2026-04-29 NVIDIA bring-up: even with the full bias-
correction pipeline, the right end-nodes, and CUDA-enabled
optimization, `yolo11n-seg` fails Hailo-8 mapping after ~6m 29s
allocation:

```
Mapping Failed
Compilation failed: Failed to reach required FPS on the following
    layers: …
```

The model exists but doesn't fit the chip's performance budget.
Consistent with Hailo not publishing a `v11_seg_n_hailo8` prebuilt
in their Model Zoo (they ship `v8_seg_n_hailo8` and v11 detection,
but not v11 seg). The same model compiles cleanly on Hailo-10H
(~22 min, 4.8 MB HEF).

This is a hardware capability mismatch, not a tooling bug. Don't
spend more than one ~6-minute mapping attempt rediscovering it; if
you need v11 segmentation on the AI HAT+ Pi (Hailo-8/8L), the
honest answer is "use v8 segmentation instead, or upgrade to AI HAT+
2 with Hailo-10H." Lowering the SDK's FPS target via
`model_optimization_config(...)` is theoretically possible but
unlikely to recover enough budget — left as a speculative experiment.

## 12. `yolo26n-seg / hailo8` is hardware-unfittable (confirmed)

**Status:** unfixable on Hailo-8 silicon. Use Hailo-10H instead.

Initial 2026-04-30 retry sweep showed v26-seg failing on h8 with
`concat23 errors: format_conversion13_sd48 has 2 APUs but max
allowed is 1`. Looked like the same per-layer-precision case that
yolo26n.alls solves for v26 detection. Step (b) of the session
notes investigated and built the override infrastructure
(`MODEL_SCRIPT_OVERRIDES` in `hef_compiler.py`) that successfully
unblocked v26 detection. Six variants were attempted for v26-seg:

| Attempt | Override | Outcome |
|---|---|---|
| 1 | `pre_quantization_optimization(matmul_decomposition, [matmul1, matmul2], precision_mode=a16_w8)` | SDK `KeyError: 'meta'` after 33 s |
| 2 | `pre_quantization_optimization(matmul_decomposition, [matmul1..4])` (no precision_mode) | same `KeyError: 'meta'` |
| 3 | `quantization_param([matmul1..4], precision_mode=a16_w16)` | `Unsupported value [<PrecisionMode.a16_w16>]` at script load |
| 4 | `quantization_param([matmul1..4], precision_mode=a16_w8)` | same `Unsupported value` |
| 5 | `quantization_param([matmul1..4], precision_mode=a8_w8_a16)` | optimizer ran 16 min; mapper rejected with `precision mode is not accurate` |
| 6 | `quantization_param([matmul1..4], precision_mode=a8_w8_a8)` | optimizer ran 19.7 min; mapper rejected with the original `More than one output is not supported for layer matmul1` after 3m 39s |

Per `hailo_model_optimization/acceleras/hailo_layers/hailo_matmul.py`,
`HailoMatmul.SUPPORTED_PRECISION_MODE = {a8_w8, a8_w8_a8, a8_w8_a16}`.
All three were exercised; none accept v26-seg's matmul1 multi-output
structure on Hailo-8. `matmul_decomposition` (the official Hailo
workaround for multi-output matmuls) has its own SDK bug on this
specific network (`KeyError: 'meta'`).

**Conclusion:** v26-seg's head attention block produces a multi-
output matmul1 that Hailo-8 cannot ingest in any supported
precision mode. This is a hardware capability gap, not a tooling
gap. Consistent with Hailo not publishing a
`v26_segmentation_*_hailo8.hef` in their Model Zoo either.

If you need v26 segmentation, use the Hailo-10H Pi (AI HAT+ 2);
the more capable chip handles the multi-output matmul natively
and v26-seg compiles cleanly there (~30 min, 4.7 MB HEF). On
Hailo-8/8L, fall back to v8 or v11 segmentation.

The `MODEL_SCRIPT_OVERRIDES` infrastructure is committed in
`hef_compiler.py` regardless — it's load-bearing for v26 detection
on Hailo-8 (per Hailo's official yolo26n.alls) and serves as
scaffolding for any future per-(version, task) ALLS workarounds
that *do* turn out to be tractable.

## 13. `yolo26n` detection on Hailo-10H is unsupported by Hailo's own YAML

`hailo_model_zoo/cfg/networks/yolo26n.yaml` (the only v26 task
with a published Hailo Model Zoo YAML) declares:

```yaml
supported_hw_arch: [hailo8, hailo8l]
```

Hailo's own definition excludes Hailo-10H. Compiling
`--hw-arch hailo10h --model yolo26n.pt` will fail at mapping with
something like `format_conversion13 errors`; this is intended by
Hailo, not a tooling bug. v26 seg/pose/obb on Hailo-10H work fine
(verified 2026-04-30) — only the plain detection variant is
out-of-policy. If you need v26 detection on the AI HAT+ 2 Pi:
either wait for Hailo to broaden support (unlikely soon), use v11
detection instead (similar accuracy in the n size), or fall back
to the Hailo-8 / 8L Pi for v26 detection specifically.

## Verify-before-commit checklist

Before adding any `.hef` to the repo:

- [ ] Ran `hailortcli run model.hef` on the Pi
- [ ] Output tensor count matches the ONNX export
- [ ] Output shapes match Benchy's decoder expectations
- [ ] No NMS / mask / pose decoding baked in
- [ ] Compiled for the correct `--hw-arch`
- [ ] HailoRT version on the Pi is ≥ the DFC's minimum
