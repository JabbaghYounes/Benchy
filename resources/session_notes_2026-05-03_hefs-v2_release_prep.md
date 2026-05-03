# Session Notes — 2026-05-03 — `hefs-v2` Release Prep

## Scope

Compile the two `hailo10h` pose HEFs that block AI HAT+ 2 verify
parity:
- `v8_pose_s_hailo10h.hef` (failed in hefs-v1 prep at HAR generation)
- bonus: `v8_pose_n_hailo10h.hef` (not previously attempted)

Workstation: node01 (cluster relay), `venv-compile-h10h` with DFC
5.3.0 / Python 3.10.20. Branch `hef-v2` cut from `main`.

## Code changes (local branch `hef-v2`, not yet committed)

Two files touched, both fall out of the v8-pose investigation:

### `benchmark/workloads/yolo/conversion/har_generator.py`

Added `("v8", YOLOTask.POSE)` entry to `END_NODE_TABLE`. Without it
the SDK auto-detects `/model.22/dfl/Reshape, /model.22/Concat_4,
/model.22/Sigmoid` and trips:

  - `UnsupportedShuffleLayerError` on `/model.22/dfl/Transpose`
  - `UnsupportedShuffleLayerError` on `/model.22/Reshape_10`
  - `UnsupportedModelError` on `/model.22/Sub` and `/model.22/Add_1`
    (constant shape (1,2,8400) not broadcastable to [1,8400,2])

The new entry mirrors the existing `("v11", YOLOTask.POSE)` shape but
points at `/model.22/cv{2,3,4}.{0,1,2}.{0,1,2}.2/Conv` (v8's head
module sits at `/model.22`, v11's at `/model.23`). Verified against
`models/hailo/hailo10h/v8/pose/yolov8s-pose/model.onnx` — all 9 nodes
present.

### `benchmark/workloads/yolo/conversion/hef_compiler.py`

Added `("v8", YOLOTask.POSE)` entry to `MODEL_SCRIPT_OVERRIDES`,
derived from `hailo_model_zoo/cfg/alls/generic/yolov8s_pose.alls`
(the official Hailo Model Zoo recipe). Headline knob:

```
allocator_param(automatic_reshapes=disabled)
```

The END_NODE_TABLE patch alone unblocks HAR generation but v8s-pose
then fails at resource allocation: with auto-reshape placement
enabled the allocator's search space explodes and v8s-pose times out
the per-context 1h watchdog on hailo10h (5 contexts, all 1h timeouts;
total partition+allocation 1h11m before SDK gives up). Disabling
auto-reshape forces a simpler placement the allocator finds in
seconds. v8n-pose maps fine without this override (3.3M params,
4 contexts) but it's harmless to apply uniformly.

The other ALLS lines copy the rest of the official recipe:
disabled equalization (improves pose accuracy), 16-bit precision on
`output_layer3/6/9` (the cv4.* keypoint outputs — 8-bit quant noise
hurts pose more than box/class), and the canonical finetune learning
rate (0.00015).

## Compile results

### `v8_pose_s_hailo10h.hef` — PASS (with both patches)

- Path: `resources/hefs/v8_pose_s_hailo10h.hef`
- Size: 13,574,144 bytes (12.95 MB)
- sha256: `b9d67432e82867484466e5ee7a521d4aab9c2dcacd3e9bd195ae099924beae5d`
- DFC: 5.3.0 (`venv-compile-h10h`, Python 3.10.20)
- Compression level: 1, optimization level: 2 (defaults)
- Calibration: 1024 images sampled from val2017 (seed 42)
- Output SNRs: 18.82–32.11 dB across 9 output layers
- Final partition: 5 contexts (multi-context flow)
- First attempt (only the END_NODE_TABLE patch, no ALLS override):
  HAR + quant passed cleanly, then failed at resource allocation
  with the watchdog timeout described above.
- Second attempt (END_NODE_TABLE + ALLS override): allocation
  completed in seconds; total compile ≈ 24min.

### `v8_pose_n_hailo10h.hef` — PASS

- Path: `resources/hefs/v8_pose_n_hailo10h.hef`
- Size: 4,820,992 bytes (4.60 MB)
- sha256: `16a4d2a173b479a8dd8ae1474a8910b1ca41fb855a887ec349f34e3637cb0d00`
- DFC: 5.3.0 (`venv-compile-h10h`, Python 3.10.20)
- Compression level: 1, optimization level: 2 (defaults)
- Calibration: 1024 images sampled from val2017 (seed 42)
- Output SNRs: 18.0–31.83 dB across 9 output layers
- Final partition: 4 contexts
- Compiled cleanly with only the END_NODE_TABLE patch (no ALLS
  override needed). The ALLS override was added afterwards for v8s
  but is keyed to `(v8, POSE)` in the table, so any future v8n-pose
  rebuild will pick it up too (harmless).

## What this changes for hefs-v2

- AI HAT+ 2 verify gap: hefs-v1 = 12/13 (v8_pose_s_hailo10h missing).
  hefs-v2 = expected 13/13 + bonus v8_pose_n_hailo10h.
- HEF count: hefs-v1 = 30 in tree. hefs-v2 = 32 in tree (adds
  v8_pose_n_hailo10h.hef + v8_pose_s_hailo10h.hef).
- The END_NODE_TABLE patch should also unblock yolov8m-pose and
  yolov8l-pose for hailo10h whenever they're added to the compile
  matrix — both are bigger than v8s and will likely need the same
  ALLS override to map within the watchdog budget.

## Suggested follow-ups (deferred from this session)

- Re-export and rebuild v8m-pose and v8l-pose for hailo10h to round
  out the AI HAT+ 2 pose family. These weren't in scope for hefs-v2.
- Consider rebuilding `v8_pose_m_hailo8.hef` and `v8_pose_s_hailo8.hef`
  (already in tree from the 2026-04-29/30 sweeps) under the new
  ALLS override and comparing accuracy — the official Hailo recipe
  may produce slightly better quality than the auto-detected
  end-nodes-without-overrides path used during those sweeps.
- `onnxsim` still missing from `venv-compile-h10h`. Same observation
  as the 2026-05-02 session notes; non-fatal.
