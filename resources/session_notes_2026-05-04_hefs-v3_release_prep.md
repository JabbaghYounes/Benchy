# Session Notes — 2026-05-04 — `hefs-v3` Release Prep

## Scope

Compile two `hailo10h` pose HEFs to broaden the v8 pose family on
AI HAT+ 2, plus the optional `hailo8` v8n-pose to close the AI HAT+
gap (`hefs-v2` shipped only `s` and `m` for hailo8 — yolov8n-pose
isn't published by Hailo Model Zoo for hailo8 either, so this is
gap-closing on both ends).

Workstation: node01 (RTX 2080 Ti, 11 GB VRAM), `venv-compile-h10h`
and `venv-compile-h8`, DFC 5.3.0 / Python 3.10.20. No git on the
workstation copy of the repo — patches applied locally; publishing
machine handles the upstream commit.

## Code changes (local-only on node01, not yet on `main`)

The v2 patches to `END_NODE_TABLE` and `MODEL_SCRIPT_OVERRIDES` for
`("v8", YOLOTask.POSE)` are already on `main` and were sufficient for
v8n / v8m on hailo10h. v8l on the 2080 Ti needed one extra knob:

### `benchmark/workloads/yolo/conversion/hef_compiler.py`

Added `batch_size=4` to the v8-pose `finetune` line:

```diff
-        "post_quantization_optimization("
-        "finetune, policy=enabled, learning_rate=0.00015)",
+        "post_quantization_optimization("
+        "finetune, policy=enabled, learning_rate=0.00015, batch_size=4)",
```

**Why:** v8l-pose's first attempt failed mid-Quantization-Aware
Fine-Tuning with `GPU memory has been exhausted. Please try to use
Quantization-Aware Fine-Tuning with lower batch size or run on CPU.`
The SDK default finetune batch size is 8; on the 2080 Ti's 11 GB
VRAM, v8l-pose's QAT step exceeds it. `batch_size=4` fits with
margin.

The override is keyed on `(v8, POSE)` so it also applies to
v8n / v8s / v8m pose. This is harmless: the smaller variants already
fit at `batch_size=8`, and a halved batch only slows finetune (it
doesn't change the loss landscape meaningfully at this scale). Future
v8 pose rebuilds will pick this up automatically.

## Compile results

### `v8_pose_m_hailo10h.hef` — PASS

- Path: `resources/hefs/v8_pose_m_hailo10h.hef`
- Size: 19,726,336 bytes (18.81 MB)
- sha256: `13c8667305111785b71c97902776fd183f1d52bc68ce0a32a1c49c946b254443`
- DFC: 5.3.0 (`venv-compile-h10h`, Python 3.10.20)
- Compression level: 1, optimization level: 2 (defaults)
- Calibration: 1024 images sampled from val2017 (seed 42)
- SDK Successful Compilation: 23 m 17 s
- Wall (real): 64 m 39 s — Layer Noise CPU fallback dominates
- Final partition: 5 contexts (multi-context flow; single-context
  presolve failed with `lcus=(132/80)`)
- Layer Noise Analysis hit the **same GPU OOM as `hefs-v2`** at 50 %
  iteration — SDK falls back to CPU and continues, but **does not
  emit per-layer SNR statistics** in this run. All 9 output layers
  finish "Pass" at the compile-stage gate.
- No allocator timeouts; SDK accepted the partition cleanly.

### `v8_pose_l_hailo10h.hef` — PASS (with `batch_size=4` finetune patch)

- Path: `resources/hefs/v8_pose_l_hailo10h.hef`
- Size: 32,276,480 bytes (30.78 MB)
- sha256: `ef4667184a150d67d3a4b1ec7ad7a91c2e1d217348e2c5a5a7007002a86cfe26`
- DFC: 5.3.0 (`venv-compile-h10h`, Python 3.10.20)
- Compression level: 1, optimization level: 2 (defaults)
- Calibration: 1024 images sampled from val2017 (seed 42)
- SDK Successful Compilation: 31 m 51 s
- Wall (real): 118 m 56 s — Layer Noise CPU fallback alone is
  59 m 37 s; QAT 19 m 56 s; Bias Correction 5 m 05 s
- Final partition: 7 contexts (single-context presolve failed with
  `lcus=(166/80)`; multi-context allocator searched 271 iterations)
- First attempt aborted at QAT with GPU OOM. Retry succeeded after
  the `batch_size=4` finetune override was added.
- Layer Noise SNR stats again absent (same GPU OOM at 50 %
  Full Quant Analysis pattern as v8m). Output-layer compile gates
  all "Pass".

### `v8_pose_n_hailo8.hef` — PASS (optional bonus)

- Path: `resources/hefs/v8_pose_n_hailo8.hef`
- Size: 6,909,188 bytes (6.59 MB)
- sha256: `ddfbf2712788ff1fad6bdf8f0febd64591a1e287c51d81c2a8dc18bb464103a6`
- DFC: 5.3.0 (`venv-compile-h8`, Python 3.10.20)
- Compression level: 1, optimization level: 2 (defaults)
- Calibration: 1024 images sampled from val2017 (seed 42)
- SDK Successful Compilation: 7 s
- Wall (real): 15 m 45 s (Bias Correction 1 m 58 s, QAT 4 m 51 s,
  Layer Noise Analysis 1 m 03 s — small enough to fit on the 2080 Ti)
- Final partition: 2 contexts on hailo8 (single-context too tight)
- The v8-pose end-node + ALLS patches inherited from `hefs-v2` are
  the same path used here. No extra hailo8-specific work was needed
  beyond the v3 finetune `batch_size=4` override (which v8n-pose
  doesn't strictly need but inherits via the `(v8, POSE)` key).
- Layer Noise SNR stats are present (model fits in GPU at 50 %
  Full Quant Analysis), but raw values weren't extracted into the
  log at INFO level — output-layer gates all "Pass".

## What this changes for hefs-v3

- HEF count: hefs-v2 = 32 in tree → hefs-v3 = 35 in tree (adds
  `v8_pose_n_hailo8.hef`, `v8_pose_m_hailo10h.hef`,
  `v8_pose_l_hailo10h.hef`).
- AI HAT+ 2 (hailo10h) v8 pose family: was n/s only after v2, now
  n/s/m/l — full sweep complete short of x.
- AI HAT+ (hailo8) v8 pose family: was s/m, now n/s/m. The "no
  hailo8 v8n-pose Zoo prebuilt" gap noted in `docs/hailo.md` is
  now closed; verify scripts can run a symmetric `yolo-v8-pose-n`
  step on both Pis.

## Suggested follow-ups (deferred from this session)

- v8x-pose (extra-large) for both architectures. Not in any verify
  matrix today; bigger than v8l so will likely need an even smaller
  finetune batch (or QAT disabled) plus more allocator latitude.
- v11 pose s/m/l for hailo10h — same shape as v8 pose; the existing
  `(v11, POSE)` END_NODE_TABLE entry should cover HAR generation,
  but ALLS overrides analogous to the v8 pose recipe are likely
  needed for the bigger sizes. Not attempted this session.
- Layer Noise SNR statistics: the 2080 Ti's 11 GB VRAM is too small
  to run `Full Quant Analysis` for v8m/v8l-pose at default batch
  size, so SNR numbers aren't being captured. Adding
  `model_optimization_config(checker_cfg, batch_size=1)` to the
  `(v8, POSE)` override (or `policy=disabled` and accepting we
  don't get SNRs at all) would be cleaner than the silent
  GPU→CPU fallback we get now. Doesn't affect HEF correctness.
- Consider rebuilding `v8_pose_n_hailo10h.hef` and
  `v8_pose_s_hailo10h.hef` from `hefs-v2` under the new finetune
  `batch_size=4` override and comparing accuracy. Any delta should
  be small (smaller batch, longer finetune at same step count) but
  worth a smoke test if accuracy regressions show up downstream.
