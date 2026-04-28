# Hailo-8 / 8L Model List

The 10 YOLO models targeted for Benchy's Hailo-8 family HEF set.

ONNX exports are produced on the Pi (`640×640`, opset 11, batch=1) and `scp`'d
to the workstation for compilation.

## Compile parameters

- Input: `640×640`, `batch=1`, opset 11
- Target: `--hw-arch hailo8` **or** `--hw-arch hailo8l` — pick the chip on
  the Pi (binaries are not interchangeable)
- Quantization: INT8
- Calibration data per task:
  - detection → coco128
  - segmentation → coco128-seg
  - OBB → DOTAv1 (subset is fine)
  - pose → coco-pose (subset is fine)
  - classification → imagenet10
- **No baked postprocessing** — see [../pitfalls.md](../pitfalls.md)

## Models — Hailo-8 destination paths

| # | Model `.pt` | Source ONNX (on Pi) | HEF destination |
|---|---|---|---|
| 1 | `yolov8n.pt` | `models/hailo/v8/detection/yolov8n/model.onnx` | `models/hailo/hailo8/v8/detection/yolov8n/model.hef` |
| 2 | `yolov8n-obb.pt` | `models/hailo/v8/obb/yolov8n-obb/model.onnx` | `models/hailo/hailo8/v8/obb/yolov8n-obb/model.hef` |
| 3 | `yolov8n-seg.pt` | `models/hailo/v8/segmentation/yolov8n-seg/model.onnx` | `models/hailo/hailo8/v8/segmentation/yolov8n-seg/model.hef` |
| 4 | `yolov8n-pose.pt` | `models/hailo/v8/pose/yolov8n-pose/model.onnx` | `models/hailo/hailo8/v8/pose/yolov8n-pose/model.hef` |
| 5 | `yolo11n-obb.pt` | `models/hailo/v11/obb/yolo11n-obb/model.onnx` | `models/hailo/hailo8/v11/obb/yolo11n-obb/model.hef` |
| 6 | `yolo11n-seg.pt` | `models/hailo/v11/segmentation/yolo11n-seg/model.onnx` | `models/hailo/hailo8/v11/segmentation/yolo11n-seg/model.hef` |
| 7 | `yolo11n-pose.pt` | `models/hailo/v11/pose/yolo11n-pose/model.onnx` | `models/hailo/hailo8/v11/pose/yolo11n-pose/model.hef` |
| 8 | `yolo26n-obb.pt` *(experimental)* | `models/hailo/v26/obb/yolo26n-obb/model.onnx` | `models/hailo/hailo8/v26/obb/yolo26n-obb/model.hef` |
| 9 | `yolo26n-seg.pt` *(experimental)* | `models/hailo/v26/segmentation/yolo26n-seg/model.onnx` | `models/hailo/hailo8/v26/segmentation/yolo26n-seg/model.hef` |
| 10 | `yolo26n-pose.pt` *(experimental)* | `models/hailo/v26/pose/yolo26n-pose/model.onnx` | `models/hailo/hailo8/v26/pose/yolo26n-pose/model.hef` |

For Hailo-8L builds, swap `hailo8` → `hailo8l` in both the destination path
and the `--hw-arch` flag.

## Expected success rates

| Tier | Models | Confidence | Notes |
|------|--------|------------|-------|
| ✅ Smooth | v8 detection / seg / pose | High | Standard heads, well-supported |
| ✅ Likely | v11 seg, v11 pose | High | v11 ≈ v8 architecture |
| ⚠️ Moderate | v8 OBB | Medium-high | OBB heads newer; may need head tweaks |
| ⚠️ Higher effort | v11 OBB | Medium | Same risk, less validated |
| ❌ Experimental | v26 OBB / seg / pose | Low | v26 detection is in the Zoo; non-detection heads are unproven |

Keep v26 entries tagged `[experimental]` in the verify suite — failure isn't
a project regression.

## Suggested phasing

1. **Phase 1 — quick wins.** `yolov8n-seg`, `yolov8n-pose`. Should compile
   cleanly with stock Zoo configs.
2. **Phase 2 — moderate.** `yolo11n-seg`, `yolo11n-pose`. Likely doable, may
   need minor fixes.
3. **Phase 3 — OBB.** `yolov8n-obb`, `yolo11n-obb`. Expect graph tweaks;
   possible failure on v11.
4. **Phase 4 — experimental.** `yolo26n-*`. Don't rely on these landing.

## Hailo-8 vs 8L performance note

The 8L is half the TOPS of the 8 (13 vs 26). Same model on 8L will:

- compile with a slightly different memory layout
- run noticeably slower
- consume less power

Benchy's verify suite should record both numbers if you're publishing
benchmark results.
