# Session Notes — 2026-05-02

End-to-end AI HAT+ 2 (Raspberry Pi 5 + Hailo-10H) bring-up: HailoRT 5.x
runtime migration in code, truncated-head decoder rewrite, NPU LLM
service installation, and the first valid cross-backend LLM comparison
this benchmark suite has produced.

## Headline result

`./scripts/verify_ai_hat_plus_2.sh` final state: **11/13 pass · 0 skip ·
2 fail** (35–41 min wall, depending on whether NPU LLM is exercised).
The two failures are missing `hailo10h` HEFs (`v8_segmentation_n` and
`v8_pose_s`) — out of scope for the Pi, need the x86 compile box.

Cross-backend LLM, `llama3.2:1b`, drone prompt set, identical request
shape:

| Backend                     | Avg TPS | Avg TTFT | Speedup |
|-----------------------------|--------:|---------:|--------:|
| Hailo-10H NPU (hailo-ollama)|  10.29  |  455 ms  |  1.49×  |
| Pi 5 CPU (ollama)           |   6.89  |  451 ms  |  baseline |

TTFT is essentially identical because prefill is the same workload on
both backends; the speedup comes from decode TPS. Verify bundle:
`results/hw_verify_20260502_094155/`. Dashboard:
`results/hw_verify_20260502_094155/report/`.

## Verify journey

The verify went `2/12 → 8/12 → 10/12 → 11/13` across this session as
each blocker fell. The final number is 11/13 because the NPU LLM step
joined the count (was a SKIP, now a PASS).

## Issue 1 — HailoRT 4.x → 5.x runtime migration

**Symptom.** All 10 YOLO steps failed in <15 s each. First failure
mode: `'ConfigureParams' has no attribute 'default_interface'`. After
patching that attribute, the next call hit `HAILO_NOT_IMPLEMENTED`
(`libhailort` error 7) on `vdevice.configure(hef, params)` itself.

**Root cause.** `hailo.py` and `validation.py` were using the deprecated
HailoRT 4.x `InferVStreams` API. HailoRT 5.x on Hailo-10H replaced it
wholesale with the `InferModel` / `bindings` pattern; the old surface
returns "not implemented" on the new firmware.

**Fix (commit `b1ee4ab`).** Migrate both call sites to the 5.x lifecycle:

```python
vdevice = VDevice()
infer_model = vdevice.create_infer_model(str(hef_path))
infer_model.input().set_format_type(FormatType.FLOAT32)
for o in infer_model.outputs:
    o.set_format_type(FormatType.FLOAT32)
configured = infer_model.configure()
# ...
bindings = configured.create_bindings(
    input_buffers={inp_name: np.ascontiguousarray(preprocessed)},
    output_buffers={out.name: np.empty(out.shape, np.float32)
                    for out in infer_model.outputs},
)
configured.run([bindings], timeout=10000)
```

**Cleanup ordering matters.** `ConfiguredInferModel.shutdown()` →
drop `infer_model` → `vdevice.release()`. Doing the release first
logs `Lost communication with the server` because `InferModel` keeps
the `VDevice` alive (per HailoRT 5.x docs). The migration commit
encodes this in a comment in `cleanup()`.

## Issue 2 — `_detect_class_count` lied on truncated heads

**Symptom.** Every Hailo-10H HEF logged
`Class count mismatch: expected 80, detected 60` in validation.

**Root cause.** `_detect_class_count` assumed the combined-head layout
`(N, anchors, 4 + nc)` and returned `last_dim - 4` of the first output
with channels > 4. Truncated heads (per `END_NODE_TABLE` in
`har_generator.py`) emit per-stride **box (64 ch) + cls (nc ch)**
branches as separate tensors. The heuristic picked the box branch and
returned 64 - 4 = 60.

**Fix (commit `a128e1d`).** Task-aware filter that excludes box
channels (4 / 64) and task-specific extras:

```python
excluded = {4, 64}
if task == YOLOTask.OBB:        excluded |= {1}     # angle branch
elif task == YOLOTask.POSE:     excluded |= {51}    # 17 kpts × 3
elif task == YOLOTask.SEGMENTATION: excluded |= {32}  # mask coeffs
candidates = [d for d in last_dims if d not in excluded]
```

If multiple stride branches survive the filter, take the modal value
(class branches replicate across strides; legitimate combined-head
classifications occupy one tensor only).

## Issue 3 — Pose decoder crashed on truncated heads

**Symptom.** v11-pose / v26-pose: `cannot reshape array of size 2891
into shape (17, 3)` in warmup run 1. Whole step failed in 4 s.

**Root cause.** `_get_detection_output()` picked one per-stride branch
arbitrarily (v11 picked the 64-ch box branch, v26 picked the 51-ch
kpts branch). `_process_pose` then ran an unconditional
`kpt_block.reshape(-1, 17, 3)` which crashed when channel count didn't
divide cleanly.

**Surgical fix (commit `b6ae9c7`).** Return `[]` early when the
last-dim mismatch warning would fire — symmetric with seg/OBB which
already silently returned 0 detections via best-effort. Verify steps
pass; pose results are empty (placeholder until proper decode).

## Issue 4 — Truncated-head outputs returned 0 decoded detections

**Symptom.** After the surgical pose fix, verify "passed" 10/10 on
YOLO but the dashboard rows for pose / seg / OBB carried 0 detections.
Only detection produced real boxes (via the best-effort fallback in
`_decode_detection_output`).

**Root cause.** None of the decoders understood the multi-output
truncated-head layout. The HEFs we ship in `resources/hefs/` are
compiled with end-node truncation per `END_NODE_TABLE`, so the
runtime returns 6/9/9/10 per-stride tensors (det / obb / pose / seg)
instead of a single combined-head tensor that the decoders expect.

**Proper fix (commit `8ba8821`).** Add `_assemble_truncated_head` that
reassembles per-stride branches into the combined-head layout:

- Group outputs by feature-map shape `(H, W)`; identify branches by
  channel count (64 = DFL box, `nc` = cls, 51 = kpts, 1 = OBB angle,
  32 = seg coeffs, 32 at input/4 = seg prototype).
- DFL-decode the 64-ch box branch (4×16 softmax → distance expectation
  → ltrb in feature-map units → xywh in input pixels via anchor
  center + stride).
- Sigmoid-decode class scores.
- Pose kpt xy via Ultralytics formula `(2v + (anchor − 0.5)) × stride`;
  visibility left raw because `_process_pose` sigmoids it later.
- OBB angle via `(sigmoid(v) − 0.25) × π`.
- Returns `None` for combined-head HEFs (single tensor) or
  single-stride inputs, so the existing best-effort paths stay intact.

Wired into `_process_detection`, `_process_obb`, `_process_segmentation`,
and `_process_pose` at the top of each decoder.

**Hardware-confirmed** on `bus.jpg` via direct HailoRT probe:

| HEF                       | Decoded results                            |
|---------------------------|--------------------------------------------|
| `v11_detection_n`         | 5 results: bus (89%) + 2× person          |
| `v11_pose_n`              | 4 person detections, ~89% conf            |
| `v11_segmentation_n`      | 5 results, real masks (bus mask = 7997 px) |
| `v11_obb_n`               | 0 (correct — no DOTA classes in image)    |

## Issue 5 — v26 used 4-ch direct distances, not 64-ch DFL

**Symptom.** After Issue 4 fix, v26-pose / v26-seg still returned 0
detections (v11 worked). v26-pose log:
`Pose output last-dim 51, expected 56; truncated-head layout is not
yet decoded — returning no detections.` (Graceful empty fired,
assembler did not.)

**Root cause.** v26 dropped DFL. The box branch is **4 channels** of
raw `(l, t, r, b)` in feature-map units, not 64 = 4 × 16 softmax bins.
The assembler only matched `c == 64` for box.

**Fix (commit `01eda07`).** `_decode_box_branch` dispatches by channel
count: 64 → DFL softmax + expectation, 4 → direct read. Both produce
ltrb distances in feature-map units, which a shared `_ltrb_to_xywh`
helper converts to xywh in input pixels.

**Hardware-confirmed** on `bus.jpg`:

| HEF                       | Before this commit | After |
|---------------------------|--------------------|-------|
| `v26_pose_n`              | 0 detections       | 4 person detections |
| `v26_segmentation_n`      | 0 detections       | 5 results with masks |
| `v26_obb_n`               | 0 detections       | 0 (correct, matches v11) |

## Issue 6 — NPU LLM step skipped every verify (the headline gap)

**Symptom.** Every verify run reported
`[12/13] llm-npu-llama3.2:1b — SKIP (hailo-ollama not reachable on :8000)`.
Only CPU LLM numbers ever made it into the dashboard, defeating the
entire reason `verify_ai_hat_plus_2.sh` exists. The `hw_verify` CPU LLM
step was being mistaken for the headline result for several sessions.

**Root cause.** `hailo-ollama` ships with the
`hailo-gen-ai-model-zoo` deb (already installed at `/usr/bin/hailo-ollama`
on this Pi as of HailoRT 5.3.0) but no systemd unit is shipped. Manual
`hailo-ollama serve` works but doesn't survive reboot, so unless
someone remembers to start it, the verify silently degrades to
CPU-only LLM.

**Fix.** Install `/etc/systemd/system/hailo-ollama.service`:

```ini
[Unit]
Description=Hailo-Ollama (HailoRT GenAI) LLM server on Hailo-10H NPU
After=network.target hailort.service
Wants=hailort.service

[Service]
Type=simple
User=vpn
ExecStart=/usr/bin/hailo-ollama serve
Restart=on-failure
RestartSec=5
Environment=OLLAMA_HOST=0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now hailo-ollama` makes it permanent. The
service is ordered after `hailort.service` so the firmware is loaded
before the daemon tries to claim the chip. The model cache stays at
`~/.local/share/hailo-ollama/` (under `vpn`'s home), consistent with
where `llama3.2:1b` was already pulled — running as root would
re-download under `/root/`.

**Cold-load quirks** in the runner output that are *not* failures:

- First `/api/show` returns 500 (model metadata endpoint not implemented
  the same way as upstream Ollama).
- First prewarm `/api/generate` returns 500 (the daemon's first request
  carries the cold-load).

`LLMBenchmarkRunner` already handles both — it logs the warning and
proceeds; the timed loop's first request absorbs the cold-load cost.

## Test deltas

- `tests/test_truncated_head_assembler.py` (new, 9 tests): synthetic
  per-stride fixtures for det / pose / OBB / seg covering both DFL and
  v26 direct paths; round-trip checks for box xywh, kpt xy/visibility,
  OBB angle decode; fallthrough cases for combined-head, single-stride,
  and missing-branch inputs.
- Suite total: **284 → 292 pass** across this session.

## Open items

1. **Compile the 2 missing HEFs** on the x86 box. From `venv-compile-h10h`:

   ```bash
   BENCHY_VENV=venv-compile-h10h scripts/compile_workstation_hefs.sh \
       --arch hailo10h --include-detection
   ```

   Stages `v8_segmentation_n_hailo10h.hef` and `v8_pose_s_hailo10h.hef`
   into `resources/hefs/` with canonical naming. Closes the verify
   to 13/13.

2. **Mirror the NPU LLM service install on the Hailo-8/8L Pi** if/when
   that board ever gets a HailoRT GenAI port. Today the AI HAT+ (not
   AI HAT+ 2) is vision-only and has no onboard SDRAM, so this is
   forward-looking only — `verify_ai_hat_plus.sh` correctly tags
   `llm-npu` as `[unsupported-on-this-hw]` there.

## Reference

- Verify bundle: `results/hw_verify_20260502_094155/`
- Commits this session (in order): `b1ee4ab`, `a128e1d`, `b6ae9c7`,
  `8ba8821`, `01eda07` plus the systemd unit at
  `/etc/systemd/system/hailo-ollama.service` (system state, not in
  the repo).
- Existing references that this note builds on:
  `resources/session_notes_2026-04-27.md` (pre-flight audit,
  bring-up checklist), `resources/session_notes_2026-04-29_nvidia_workstation.md`
  (the workstation compile flow that produced the HEFs this Pi is now
  successfully decoding).
