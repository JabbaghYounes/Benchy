# Session Notes — 2026-05-02 — AI HAT+ (Hailo-8) bring-up companion

Companion to `resources/session_notes_2026-05-02.md` (the AI HAT+ 2 /
Hailo-10H side). End-to-end fix-up of the AI HAT+ Pi 5 (Hailo-8 26 TOPS,
HailoRT 4.23, hostname `raspberrypi`) so the board produces a valid
verify bundle off the same `hef` branch the AI HAT+ 2 Pi runs on. Three
HailoRT 4.x quirks chained — each one masked the next — and one verify-
script methodology change.

## Headline result

`./scripts/verify_ai_hat_plus.sh` final state: **10/13 pass · 0 skip ·
3 fail** (327 m wall, dominated by step 13 LLM CPU sweep at ~5h). The
three failures:

| Step | Tag | Why |
|---|---|---|
| `yolo-v11-seg` | (blocking) | Missing prebuilt `v11_segmentation_n_hailo8.hef`. Needs the x86 compile box. |
| `yolo-v26-seg` | `[experimental]` | Same missing HEF (`v26_segmentation_n_hailo8.hef`). Advisory. |
| `llm-npu-llama3.2:1b` | `[unsupported-on-this-hw]` | Deliberate stub. Hailo-8 has no onboard SDRAM, no `hailo-ollama`. |

**Cross-backend LLM headline** (now valid for the first time on this
board), `llama3.2:1b`, drone prompt set, identical request shape:

| Board | Backend | TPS |
|---|---|---|
| AI HAT+ 2 Pi (`wiregaurd`) | Hailo-10H NPU | **10.61** |
| AI HAT+ Pi (`raspberrypi`) | Pi 5 CPU (Cortex-A76) | **0.62** |

≈ **17× NPU advantage** at the same model size on equivalent SoC.
Bundle: `results/hw_verify_20260502_093819/report/report_20260502_150548_dashboard.html`.

## Issue 1 — numpy 2.x ABI silently broke `set_buffer`

**Symptom.** Every YOLO step: `[HailoRT] [error] CHECK failed - Input
buffer size 0 is different than expected 4915200 for input
'yolov8n/input_layer1'`. Same shape on both old `InferVStreams` and the
new `InferModel` API — pybind buffer never reached C++.

**Root cause.** The HailoRT 4.23 wheel (`hailort-4.23.0-cp311-cp311-linux_aarch64.whl`)
declares `Requires-Dist: numpy<2` in METADATA. The bundle install in
`upgrade_hailo_runtime_from_bundle()` uses `--no-deps`, so that
constraint isn't enforced — numpy 2.4.1 stayed in the venv. With the
1→2 ABI break, pybind's numpy buffer protocol leaves the C++ side
reading `size==0` while Python `set_buffer` returns success.

**Fix (commit `b0174b0`).** Pin `numpy<2` explicitly in
`scripts/setup_rpi_ai_hat_plus.sh:upgrade_hailo_runtime_from_bundle`
right after the wheel install (root cause), and add a self-heal guard
in `scripts/hw_verify_common.sh:hw_ensure_python_deps` for existing
venvs that pre-date the fix:

```bash
if "$py" -c "import hailo_platform; ... .startswith('4.')"; then
    if "$py" -c "import numpy; ... >= 2"; then
        "$py" -m pip install --quiet 'numpy<2'
    fi
fi
```

Gated on HailoRT 4.x so the AI HAT+ 2 Pi (HailoRT 5.x, numpy-2-friendly)
is untouched.

## Issue 2 — on-chip NMS output not dispatched

**Symptom.** After the numpy fix, validation passed (2.26 ms test
inference) but warmup run 1 failed with `Inference failed: too many
indices for array: array is 1-dimensional, but 2 were indexed` in
`postprocessing.py:_process_detection`.

**Root cause.** Hailo Model Zoo's prebuilt v8/v11 detection HEFs for
Hailo-8 bake the NMS layer into the chip and emit a single
`HAILO_NMS_BY_CLASS` output (vstream shape `(80, 5, 100)` = num_classes
× 5 box params × max_bboxes), packed into a flat 1D float32 buffer at
the runtime layer. Hailo-10H detection HEFs emit raw per-stride FPN
tensors instead — that's why the migration's smoke test on Hailo-10H
worked but this Pi crashed. The dispatch must therefore branch on
`infer_model.output(name).is_nms`, not on chip arch.

Confirmed by HEF metadata inspection:

```
v8_detection_n_hailo10h.hef → 6 outputs, NHWC, is_nms=False each
v8_detection_n_hailo8.hef   → 1 output, HAILO_NMS_BY_CLASS, is_nms=True
```

**Fix (commit `17844d3`, first half).** Add
`HailoBackend._decode_on_chip_nms` that calls
`bindings.output(name).get_buffer(tf_format=False)` to unpack the
packed buffer into a `list[num_classes]` of `(n_dets, 5)` arrays per
class, then converts directly to `Detection` objects. Coordinates are
normalized [0, 1] so we rescale to input-frame pixels here so
downstream consumers see the same units the raw-tensor postprocessor
produces. `run_inference` branches on `any(o.is_nms for o in
infer_model.outputs)` and skips `postprocessor.process()` for the
on-chip-NMS case.

## Issue 3 — HailoRT 4.x InferModel auto-deactivates after one run

**Symptom.** After issue 2's fix, warmup run 1 succeeded but every
subsequent run failed with `HAILO_STREAM_NOT_ACTIVATED(72)`. Initial
suspicion was buffer-lifetime / cleanup ordering, but a minimal probe
showed the failure on a fresh `ConfiguredInferModel` doing
back-to-back `cim.run([bindings], 10000)` calls.

**Root cause.** HailoRT 4.x: `ConfiguredInferModel` requires explicit
`activate()` to keep the pipeline live across multiple runs. The pyhailort
docs hint at "use inside a context manager" but `__enter__`/`__exit__`
only handle cleanup, not activation. HailoRT 5.x manages activation via
the scheduler and rejects `cim.activate()` outright, so the call must
be version-gated.

**Fix (commit `17844d3`, second half).** In `_load_hef`, after
`infer_model.configure()`, version-check and conditionally activate:

```python
hailort_major = int(hailo_platform.__version__.split(".", 1)[0])
if hailort_major < 5:
    self._configured_infer_model.activate()
    self._cim_activated = True
```

Pair with `deactivate()` in `cleanup()` guarded by
`self._cim_activated` so the AI HAT+ 2 path stays untouched.

End-to-end after all three fixes: `yolov8n.pt --backend hailo` on this
Pi → **93.16 FPS / 10.73 ms** over 3 warmup + 10 measured runs.

## Issue 4 — `verify_ai_hat_plus.sh` step 13 was unworkable

**Symptom.** `verify_ai_hat_plus.sh` hardcoded `--profile drone`
(llama2:7b). Cancelled twice in a row at step 13 — 2026-04-27 first
session, 2026-05-01 / 02 again with this Pi's bundle
`hw_verify_20260501_221138/`. ~26 h projected wall time on the 5-prompt
× 13-run sweep at observed ~40 min/request, dominating the rest of the
verify by an order of magnitude.

**Fix (commit `1130a41`).** Switched to `--profile compare`
(`llama3.2:1b`) so the AI HAT+ Pi's CPU LLM row is at the same model
size as the AI HAT+ 2 Pi's `npu` step + this Pi's
`[unsupported-on-this-hw]` stub. Cross-backend rows now directly
comparable. Standalone `--profile drone` recipe stays for anyone who
wants the 7B CPU number.

Real wall time of the new step 13 was ~5 h (1B at `max_tokens=256`
still hits the cap on 4 of 5 prompts) — slower than the bookmark's
"~45 min" guess but completes cleanly. Full timing breakdown in
`resources/session_notes_2026-05-02_llm_drone_profile_unworkable.md`.

## Test deltas

No new tests this session — all four fixes are physical-hardware
(setup script, runtime backend) and not unit-testable on dev hosts
that lack the HailoRT 4.x wheel + numpy ABI surface. Coverage today:

- `pytest tests/`: **284 → 293 pass** across the rebase that pulled in
  the AI HAT+ 2 Pi's truncated-head decoder tests (commits `8ba8821`
  / `01eda07`).
- Validation regression: the failure modes for issues 1-3 all surface
  as runtime hardware errors that `tests/` can't reach. The verify
  bundle is the regression evidence.

## Open items

1. **Compile the 2 missing Hailo-8 HEFs** on the x86 box. From
   `venv-compile-h8`:

   ```bash
   BENCHY_VENV=venv-compile-h8 scripts/compile_workstation_hefs.sh \
       --arch hailo8 --include-detection
   ```

   Stages `v11_segmentation_n_hailo8.hef` and `v26_segmentation_n_hailo8.hef`
   into `resources/hefs/` with canonical naming. Closes the verify
   to 13/13 (the v26 one is currently `[experimental]`-tagged and
   advisory; v11-seg is the only blocking failure).

2. **Class-count heuristic returns 100 for the on-chip-NMS HEFs** —
   it reads `max_boxes_per_class` (the trailing dim of the packed
   shape `(80, 5, 100)`) instead of `num_classes`. Cosmetic warning
   only; iterating the per-class list in `_decode_on_chip_nms` uses
   the actual list length, so detections are correct. Could be tidied
   later by adding an NMS-aware code path to `_detect_class_count`.

## Reference

- Verify bundle: `results/hw_verify_20260502_093819/`
- Cancelled drone-7B bundle (12 valid YOLO + step-12 stub):
  `results/hw_verify_20260501_221138/`
- Commits this session (in order): `b0174b0`, `17844d3`, `1130a41`,
  `a481e46` (the bundle stage).
- Companion AI HAT+ 2 session note: `resources/session_notes_2026-05-02.md`
- Methodology rationale for the verify-script switch:
  `resources/session_notes_2026-05-02_llm_drone_profile_unworkable.md`
- Memory bookmark: `~/.claude/projects/-home-snpi-Documents-Benchy/memory/wip_infermodel_broken_on_4_23.md`
  (records the diagnostic order in case the same pattern shows up
  elsewhere — fix 1 unblocked fix 2 unblocked fix 3).
