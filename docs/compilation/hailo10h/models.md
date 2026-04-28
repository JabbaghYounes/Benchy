# Hailo-10H Model List

The 10 YOLO models targeted for Benchy's Hailo-10H HEF set.

ONNX exports are produced on the Pi (`640×640`, opset 11, batch=1) and `scp`'d
to the workstation for compilation.

## Compile parameters

- Input: `640×640`, `batch=1`, opset 11
- Target: `--hw-arch hailo10h`
- Quantization: INT8
- Calibration data per task:
  - detection → coco128
  - segmentation → coco128-seg
  - OBB → DOTAv1 (subset is fine)
  - pose → coco-pose (subset is fine)
  - classification → imagenet10
- **No baked postprocessing** — see [../pitfalls.md](../pitfalls.md)

## Models — Hailo-10H destination paths

| # | Model `.pt` | Source ONNX (on Pi) | HEF destination |
|---|---|---|---|
| 1 | `yolov8n.pt` | `models/hailo/v8/detection/yolov8n/model.onnx` | `models/hailo/hailo10h/v8/detection/yolov8n/model.hef` |
| 2 | `yolov8n-obb.pt` | `models/hailo/v8/obb/yolov8n-obb/model.onnx` | `models/hailo/hailo10h/v8/obb/yolov8n-obb/model.hef` |
| 3 | `yolov8n-seg.pt` | `models/hailo/v8/segmentation/yolov8n-seg/model.onnx` | `models/hailo/hailo10h/v8/segmentation/yolov8n-seg/model.hef` |
| 4 | `yolov8n-pose.pt` | `models/hailo/v8/pose/yolov8n-pose/model.onnx` | `models/hailo/hailo10h/v8/pose/yolov8n-pose/model.hef` |
| 5 | `yolo11n-obb.pt` | `models/hailo/v11/obb/yolo11n-obb/model.onnx` | `models/hailo/hailo10h/v11/obb/yolo11n-obb/model.hef` |
| 6 | `yolo11n-seg.pt` | `models/hailo/v11/segmentation/yolo11n-seg/model.onnx` | `models/hailo/hailo10h/v11/segmentation/yolo11n-seg/model.hef` |
| 7 | `yolo11n-pose.pt` | `models/hailo/v11/pose/yolo11n-pose/model.onnx` | `models/hailo/hailo10h/v11/pose/yolo11n-pose/model.hef` |
| 8 | `yolo26n-obb.pt` *(experimental)* | `models/hailo/v26/obb/yolo26n-obb/model.onnx` | `models/hailo/hailo10h/v26/obb/yolo26n-obb/model.hef` |
| 9 | `yolo26n-seg.pt` *(experimental)* | `models/hailo/v26/segmentation/yolo26n-seg/model.onnx` | `models/hailo/hailo10h/v26/segmentation/yolo26n-seg/model.hef` |
| 10 | `yolo26n-pose.pt` *(experimental)* | `models/hailo/v26/pose/yolo26n-pose/model.onnx` | `models/hailo/hailo10h/v26/pose/yolo26n-pose/model.hef` |

## Expected success rates

| Tier | Models | Confidence | Notes |
|------|--------|------------|-------|
| ✅ Smooth | v8 detection / seg / pose | High | Standard heads, well-supported |
| ✅ Likely | v11 seg, v11 pose | High | v11 ≈ v8 architecture |
| ⚠️ Moderate | v8 OBB | Medium-high | OBB heads newer; may need head tweaks |
| ⚠️ Moderate | v11 OBB | Medium-high | 10H's newer op coverage helps a bit here |
| ❌ Experimental | v26 OBB / seg / pose | Low | v26 detection is in the Zoo; non-detection heads are unproven |

The 10H's wider op coverage gives slightly better odds on v11 OBB compared
to Hailo-8. v26 risk is unchanged — backbone works, non-detection heads
have no public confirmation.

Keep v26 entries tagged `[experimental]` in the verify suite — failure isn't
a project regression.

## Suggested phasing

1. **Phase 1 — quick wins.** `yolov8n-seg`, `yolov8n-pose`. Should compile
   cleanly with stock Zoo configs.
2. **Phase 2 — moderate.** `yolo11n-seg`, `yolo11n-pose`. Likely doable, may
   need minor fixes.
3. **Phase 3 — OBB.** `yolov8n-obb`, `yolo11n-obb`. Expect graph tweaks.
4. **Phase 4 — experimental.** `yolo26n-*`. Don't rely on these landing.

## Hailo-10H performance note

The 10H is ~1.5× the TOPS of Hailo-8 and ~3× Hailo-8L. Same model, same
INT8 quant, you should expect:

- noticeably higher FPS
- room to either increase batch size or move to larger backbones (`s` /
  `m` variants) without regressing latency

If you're publishing benchmark numbers, record both 10H and 8/8L results
and label clearly — they aren't the same artifact.
