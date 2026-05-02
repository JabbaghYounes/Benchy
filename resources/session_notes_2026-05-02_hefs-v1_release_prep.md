# Session Notes — 2026-05-02 — `hefs-v1` Release Prep

## Scope

Capture compile-environment metadata, sha256 the in-tree HEFs, and (best
effort) compile the two `hailo10h` HEFs missing from the 2026-05-02
AI HAT+ 2 verify run (11/13). Outputs feed the `hefs-v1` GitHub Release
being published from a separate machine.

Workstation: node01 (cluster relay) — same machine that ran the
2026-04-29/30 sweeps. Local repo at `~/Documents/Benchy` was switched
to the `hef` branch and source files synced to node01 before any new
compile, since the compile pipeline (`pipeline.py`,
`hef_compiler.py`, `har_generator.py`, plus several callers) had
diverged from the snapshot zipped to node01 on 2026-04-30.

## Task C — two missing `hailo10h` HEFs

Both targets driven through `scripts/compile_workstation_hefs.sh
--arch hailo10h --calibration-data-path
/home/ks222-he/Documents/datasets/coco-val/images/val2017
--output-dir resources/hefs`. Calibration path made explicit because
the default code path triggers Ultralytics' COCO auto-download
(val2017 + train2017 + test2017 ≈ 25 GB) when `coco.yaml` cannot be
located via `DATASETS_DIR`. With the explicit path, calibration
loads directly from the staged 5000-image val2017 subset.

### `v8_segmentation_n_hailo10h.hef` — PASS

- File: `resources/hefs/v8_segmentation_n_hailo10h.hef`
- Size: 5,468,160 bytes (5.21 MB)
- sha256: `e7ee7e9165c38fddb4f5e77b9f057fcf1b9ef0b2247d3e4f0459ec54a861f46d`
- DFC: 5.3.0 (`venv-compile-h10h`, Python 3.10.20)
- Compression level: 1, optimization level: 2 (defaults)
- Calibration: 1024 images sampled from val2017 (seed 42)
- Output SNRs: 21.2–37.7 dB across 10 output layers (healthy)
- Final partition: 4 contexts, multi-context flow (single-context
  presolve failed `lcus=(110/80)`, expected for seg-class models)

### `v8_pose_s_hailo10h.hef` — FAIL

- Stage: `har_generation` (HAR generator, before any quantization)
- Cause: no `("v8", YOLOTask.POSE)` entry in
  `benchmark/workloads/yolo/conversion/har_generator.py:END_NODE_TABLE`.
  With no explicit end-nodes, the SDK auto-detected
  `/model.22/dfl/Reshape, /model.22/Concat_4, /model.22/Sigmoid`,
  which raises:
  - `UnsupportedShuffleLayerError in op /model.22/dfl/Transpose`
  - `UnsupportedShuffleLayerError in op /model.22/Reshape_10`
  - `UnsupportedModelError in op /model.22/Sub` — constant shape
    `(1, 2, 8400)` not broadcastable to `[1, 8400, 2]`
  - `UnsupportedModelError in op /model.22/Add_1` — same
  Diagnostic-hint retry to those same end-nodes also fails with
  `cannot reshape array of size 8400 into shape (1,1,17,1)`.
- Note: `v8_pose_m_hailo8.hef` and `v8_pose_s_hailo8.hef` are already
  in the tree (compiled in the 04-29/30 sweep). Either DFC 3.33.1 was
  more permissive about auto-end-nodes for v8 pose than 5.3.0 is, or
  those files used a path the current har_generator no longer takes.
- Fix (deferred — not applied this session): add a `("v8", YOLOTask.POSE)`
  entry to `END_NODE_TABLE` analogous to `("v11", YOLOTask.POSE)`
  but with `/model.22/...` paths:

  ```python
  ("v8", YOLOTask.POSE): [
      "/model.22/cv2.0/cv2.0.2/Conv",
      "/model.22/cv3.0/cv3.0.2/Conv",
      "/model.22/cv4.0/cv4.0.2/Conv",
      "/model.22/cv2.1/cv2.1.2/Conv",
      "/model.22/cv3.1/cv3.1.2/Conv",
      "/model.22/cv4.1/cv4.1.2/Conv",
      "/model.22/cv2.2/cv2.2.2/Conv",
      "/model.22/cv3.2/cv3.2.2/Conv",
      "/model.22/cv4.2/cv4.2.2/Conv",
  ],
  ```

  Verify against `models/hailo/hailo10h/v8/pose/yolov8s-pose/model.onnx`
  before merging. This should also let `yolov8n-pose` and
  `yolov8m/l-pose` compile cleanly on h10h, expanding the AI HAT+ 2
  matrix beyond v11 pose.

## Decision for `hefs-v1`

Ship with **30 HEFs** (29 already in tree + new
`v8_segmentation_n_hailo10h.hef`). Do not block the release on
`v8_pose_s_hailo10h.hef`; track the END_NODE_TABLE fix as a follow-up
for `hefs-v2`.

## Other observations

- `onnxsim` not installed in `venv-compile-h10h` — every parse logs
  `[Errno 2] No such file or directory: 'onnxsim'`. Non-fatal (the
  pipeline falls back to direct SDK parse) but adding `onnxsim` to
  the compile-venv requirements would make logs less noisy and might
  unblock a few edge-case ONNX inputs.
- Calibration warning: `The calibration set seems to not be normalized,
  because the values range is [(0.0, 1.0), ...]`. Same as prior
  sweeps; preprocessing scales to `[0,1]` while the SDK's normalization
  layer expects `[0,255]` to keep normalization on the neural core.
  Cosmetic at compile time; runtime quantizes on host CPU instead.
