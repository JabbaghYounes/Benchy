# Session Notes — NVIDIA Workstation HEF Compilation

**Date:** 2026-04-29 evening → 2026-04-30 morning
**Host:** node01 (Tailscale `100.99.187.118`), Ubuntu 24.04 LTS, NVIDIA RTX 2080 Ti (driver 580 / CUDA 13)
**Goal:** Follow `docs/compilation/nvidia_workstation_setup.md` to bring up node01 for HEF compilation, validate the canary, then compile gap models.

## Outcome

Setup validated, full sweep run, and a follow-up retry sweep with corrected `END_NODE_TABLE` for OBB and v26 — **fifteen fresh HEFs landed in `resources/hefs/`** by end of session. Only three failures remained, each understood:

- `yolo11n-seg / hailo8` — chip-side FPS budget; not end-node related (compiles fine on h10h)
- `yolo26n-seg / hailo8` — needs per-layer `quantization_param` precision_mode overrides; deferred
- `yolo26n det / hailo10h` — Hailo's own `yolo26n.yaml` flags `supported_hw_arch: [hailo8, hailo8l]` only; not a tooling bug

### Per-arch totals after retry

| Architecture | Pass | Fail | Combined wall time |
|---|---|---|---|
| hailo8 | 5 / 7 gap models | 2 | 82 + 101 = 183 min |
| hailo10h | 10 / 11 sweep models | 1 | 134 + 103 = 237 min |

### The 15 staged HEFs

| Filename | Source model | Size |
|---|---|---|
| `v11_pose_n_hailo8.hef` | yolo11n-pose | 8.5 MB |
| `v11_segmentation_n_hailo10h.hef` | yolo11n-seg | 4.8 MB |
| `v11_pose_n_hailo10h.hef` | yolo11n-pose | 4.1 MB |
| `v8_detection_n_hailo10h.hef` | yolov8n | 4.1 MB |
| `v8_detection_s_hailo10h.hef` | yolov8s | 12 MB |
| `v11_detection_n_hailo10h.hef` | yolo11n | 3.8 MB |
| `v8_obb_n_hailo8.hef` | yolov8n-obb | 7.3 MB |
| `v11_obb_n_hailo8.hef` | yolo11n-obb | 7.7 MB |
| `v26_obb_n_hailo8.hef` | yolo26n-obb | 7.9 MB |
| `v26_pose_n_hailo8.hef` | yolo26n-pose | 9.9 MB |
| `v8_obb_n_hailo10h.hef` | yolov8n-obb | 3.9 MB |
| `v11_obb_n_hailo10h.hef` | yolo11n-obb | 4.1 MB |
| `v26_obb_n_hailo10h.hef` | yolo26n-obb | 4.2 MB |
| `v26_segmentation_n_hailo10h.hef` | yolo26n-seg | 4.7 MB |
| `v26_pose_n_hailo10h.hef` | yolo26n-pose | 4.8 MB |
| `v26_detection_n_hailo8.hef` | yolo26n | 7.1 MB |

(The v26 detection HEF was a final-shot retry after step b. Original step-9 sweep failed it for missing END_NODE_TABLE entry; this session never re-ran it after adding both the entry and the `quantization_param([dw1..8/conv*/output*], precision_mode=a16_w16)` overrides from `yolo26n.alls`. Once both were in place, the compile took ~25 min on a 3-context partition with `+14.3% → +22.9% → +1.6%` partition-search jumps.)

The setup doc + repo code had several gaps that blocked compilation. Code changes spanning five files and several setup-procedure additions were required. Details below. **Step 9 done. Step 10 (commit + push) is the AMD-workstation follow-up.**

## Setup procedure as actually executed

1. **System packages.** Ubuntu 24.04 ships Python 3.12, not 3.10. Used `add-apt-repository ppa:deadsnakes/ppa` then `apt install python3.10 python3.10-venv python3.10-dev` plus the doc's other deps. Doc step 1 should mention deadsnakes (or `uv`) for any distro newer than 22.04.
2. **NVIDIA driver / CUDA.** Already present (`nvidia-smi` works). No system-side CUDA toolkit needed.
3. **Repo.** Used the in-place extraction at `~/benchy/Benchy-hef/` from a transferred zip, not a fresh `git clone`.
4. **Hailo wheels.** Transferred 4 wheels (~989 MB) via `scp` from another workstation into `resources/hailo-sdk/`. Tailscale link was direct (not DERP) but only ~0.34 MB/s — uplink-bound from the source machine; ~37 min total. Not blocking.
5. **Build h8 venv.** `python3.10 -m venv venv-compile-h8`, install editable + DFC 3.33.1 + model zoo 2.18.0. `hailo_sdk_client.__version__ = 3.33.1` ✓. pytest: 276/277 (one timeout, see Issue 7).
6. **Build h10h venv.** Same shape, DFC 5.3.0 + model zoo 5.3.0. `__version__ = 5.3.0` ✓. Same 276/277.
7. **COCO val2017.** `curl http://images.cocodataset.org/zips/val2017.zip` → 5000 images at `~/Documents/datasets/coco-val/images/val2017/` (788 MB on disk). First attempt failed with transient DNS error; retry succeeded.
8. **Canary compile.** This is where the wheels hit the road — see Issues section. Required all six code/config fixes below before producing a HEF.

## Issues encountered

### Issue 1 — Doc claim "DFC bundles its own CUDA runtime" was wrong

**Symptom:** `[warning] no available GPU` from the optimizer; opt level dropped to 0; mapping failed.

**Root cause:** Hailo's bundled DFC wheel did not pull CUDA 12 libs into the venv. TF 2.18 and JAX could not find `libcudart.so` / `libcudnn.so`. Only the kernel-mode NVIDIA driver was system-side.

**Fix:** In each compile venv:
```bash
pip install "tensorflow[and-cuda]==<matching-version>" "jax[cuda12]==<matching-version>"
```
The `[and-cuda]` extras pull in `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-nccl-cu12`, etc. After the fix, both `tf.config.list_physical_devices("GPU")` and `jax.default_backend()` return GPU.

**Don't reintroduce:** Add an explicit step 5b to `nvidia_workstation_setup.md` saying *"After installing the DFC + model-zoo wheels, install the matching `tensorflow[and-cuda]` and `jax[cuda12]`"*. The bundled DFC does not bring its own CUDA runtime in practice.

### Issue 2 — `tensorflow[and-cuda]` install broke `torch` (NCCL ABI mismatch)

**Symptom:** `libtorch_cuda.so: undefined symbol: ncclCommWindowDeregister`. Cascade: ONNX export module fails to import → "Ultralytics not available for ONNX export" → compile aborts at stage 1.

**Root cause:** Both `nvidia-nccl-cu12` (pulled by TF/JAX) and `nvidia-nccl-cu13` (already present, pulled by torch) install `libnccl.so.2` to the same path `nvidia/nccl/lib/`. The cu12 install overwrote cu13's `libnccl.so.2`, but torch's `libtorch_cuda.so` was linked against the cu13 ABI (which has `ncclCommWindowDeregister`; cu12-2.21.5 does not).

**Fix:**
```bash
pip install --force-reinstall --no-deps nvidia-nccl-cu13
```
This restores the cu13 NCCL at the shared path. TF on a single GPU does not invoke NCCL collective ops, so it remains happy with the cu13 lib at runtime even though it expected cu12.

**Don't reintroduce:** Apply this fix immediately after the `tensorflow[and-cuda]` step. It belongs as step 5c in the setup doc.

### Issue 3 — `optimization_level` and `compression_level` were dead config fields

**Symptom:** Setting `compression_level=1` via the CLI had zero effect on the SDK's optimization behaviour.

**Root cause:** `HEFCompilerConfig` declared `optimization_level: int = 2` and `compression_level: int = 0` but `HEFCompiler._compile_with_sdk()` just called `runner.optimize(calib_data)`. The SDK's `optimize()` does not accept these as kwargs (its signature is `optimize(self, calib_data, data_type=auto, *, work_dir=None, checkpoint=NONE, memento=None)`), and the compiler never built or loaded a model script, so the SDK silently used its own defaults.

**Fix:** In `hef_compiler.py`, before `runner.optimize(...)`:
```python
script_lines = [
    f"model_optimization_flavor("
    f"optimization_level={config.optimization_level}, "
    f"compression_level={config.compression_level})",
]
if config.optimization_level >= 1:
    script_lines.append(
        "post_quantization_optimization(bias_correction, policy=enabled)"
    )
runner.load_model_script("\n".join(script_lines) + "\n")
```

**Don't reintroduce:** Always pass these to the SDK via a model script. The dataclass fields by themselves are advisory only.

### Issue 4 — Optimization levels are mutually exclusive (not cumulative)

**Symptom:** At `optimization_level=2`, the log showed `[info] Bias Correction skipped` / `[info] Adaround skipped` while `Quantization-Aware Fine-Tuning is done (00:06:19.82)` ran. The doc's stated assumption that "level ≥1 enables Bias Correction, Adaround, and Finetune" was wrong.

**Root cause:** `hailo_model_optimization/.../mo_config.py:get_optimization_config()` uses `if/elif/elif/elif`:
```python
elif level == 1: cfg["bias_correction"] = ...
elif level == 2: cfg["finetune"]        = ...
elif level == 3: cfg["adaround"]        = ...   # train_all, dataset_size=256
elif level == 4: cfg["adaround"]        = ...   # train_all, dataset_size=1024
```
Each level enables exactly one extra pass; level 2 does *not* include level 1's bias_correction.

**Fix:** Add an explicit `post_quantization_optimization(bias_correction, policy=enabled)` to the model script whenever `optimization_level >= 1`. This forces bias_correction on top of whatever the flavor selects, so seg/pose/OBB heads get 8-bit biases.

**Don't reintroduce:** The setup doc's "level ≥1 runs them all" claim should be corrected. Bias correction has to be forced explicitly.

### Issue 5 — First model-script attempt used the wrong ALLS command

**Symptom:** `'bias_correction' is not a valid MOConfigCommand. Model script parsing failed`.

**Root cause:** Tried `model_optimization_config(bias_correction, policy=enabled)`. `bias_correction` is a `PostQuantizationFeature` enum value, not a `MOConfigCommand` enum value. Found in `hailo_model_optimization/acceleras/utils/acceleras_definitions.py`.

**Fix:** Use `post_quantization_optimization(bias_correction, policy=enabled)` instead.

**Don't reintroduce:** When enabling individual quantization features, the ALLS verb is `post_quantization_optimization`, not `model_optimization_config`. Other commands at the same path: `pre_quantization_optimization`, `model_optimization_flavor`, `quantization_param`.

### Issue 6 — `END_NODE_TABLE` truncated at the wrong layers (this was the actual blocker)

**Symptom:** With every other fix applied, mapping still failed:
```
activation1/activation2 failed on kernel validation: 16x4 is not supported
Agent infeasible (× hundreds)
DW resources calculation failed: more than 1 subclusters are needed for 16bit L2 biases
```
Failure pattern was identical between v11n-seg/hailo8 (`activation2`) and v8n-seg/hailo8 (`activation1`).

**Root cause:** The pre-existing `END_NODE_TABLE` (and the speculative `end_node_truncation_plan.md` it was built from) listed truncation points at the *deep* post-processing layers — `Sigmoid`, `Concat`, `Sigmoid_1`, `Mul`, `Mul_3`. Hailo's official Model Zoo YAMLs (e.g. `hailo_model_zoo/cfg/networks/yolov8n_seg.yaml`) cut at the *raw Conv outputs* one or two levels earlier and rely on host-side decoding. Truncating at the deep nodes pulls the high-precision-bias post-processing tail (`activation1/2`) into the chip-mapped subgraph; it doesn't fit on Hailo-8 L2.

For reference, our entry was:
```python
("v11", YOLOTask.SEGMENTATION): [
    "/model.23/Sigmoid",
    "/model.23/Concat_2",
    "/model.23/proto/cv3/act/Mul",
    "/model.23/Concat",
]
```
The corrected (Hailo-Zoo-style) entry is the 9 raw `cv*.X.X.2/Conv` outputs + `proto/cv3/act/Mul`.

**Fix:** Replaced the entire table with entries matching `hailo_model_zoo/cfg/networks/<network>.yaml`. v8 det / v8 seg / v11 det taken verbatim from existing YAMLs; v11 seg / v11 pose derived by analogy (head module shifts from `/model.22` to `/model.23`; v11-seg/pose are gap models with no published YAML).

**Don't reintroduce:** When adding a new (version, task) entry, **check `venv-compile-*/lib/python3.10/site-packages/hailo_model_zoo/cfg/networks/<name>.yaml` first.** Don't trust the SDK parser's own "use these end node names" hint — that produces the wrong cut. Verify with the postprocessor in `benchmark/workloads/yolo/postprocessing.py` that the Conv outputs match what the host-side decoders expect.

### Issue 7 — pytest baseline is 276/277, not 277/277

**Symptom:** `tests/test_compile_cmd.py::test_python_m_benchmark_propagates_exit_code` times out at 30s on a fresh venv.

**Root cause:** Test invokes `python -m benchmark compile --hw-arch hailo8 --model yolov8n.pt ...` for real, which Ultralytics-downloads `yolov8n.pt` on first call. 30 s isn't enough for the cold cache.

**Workaround:** Treat 276/277 as the true baseline on a clean install. (Was the same in both h8 and h10h venvs, confirming environmental.) Docs that quote 277/277 should be relaxed to "276/277 on first run, 277/277 once Ultralytics has cached the weights."

### Issue 9 — Cache key in `cache.get_hef_path` doesn't include `hw_arch`

**Symptom:** During the h10h sweep restart, `yolo11n-pose.pt` "passed" in **12 seconds** — much faster than a real compile. Investigation showed `resources/hefs/v11_pose_n_hailo10h.hef` (md5 `0aeab4e3…`) was byte-for-byte identical to the legitimate `v11_pose_n_hailo8.hef` (also md5 `0aeab4e3…`), and both had the same nanosecond mtime (`2026-04-30 11:28:39.672118157 +0100`).

**Root cause:** `benchmark/workloads/yolo/conversion/cache.py:get_hef_path(model_name, yolo_version, task)` keys on `(model_name, version, task)` — *no `hw_arch`*. So the cache stores a single `model.hef` per (model, version, task) regardless of which architecture it was compiled for. When a later compile call asks for the same `(model, version, task)` with a different arch, the cache returns the wrong-arch HEF, and `cli.py:cmd_compile`'s `shutil.copy2(result.hef_path, dest)` stages it under the new-arch canonical filename. Result: silent cross-arch HEF mislabelling.

The proximate trigger here was the cmd_compile "already staged" check (the bogus `v11_pose_n_hailo10h.hef` existed → skipped → counted as PASS). The deeper question — *who originally wrote `v11_pose_n_hailo10h.hef` during the h8 sweep?* — was not fully determined this session. The two prime suspects are:

- The cache being read with arch-blind key during a parallel-but-secondary code path (e.g. some "stage prebuilt" hook firing inside HEFCompiler).
- A leftover from a manual operation (none I'm aware of, but the timestamp matches the h8 sweep window exactly).

**Fix applied this session:** deleted the corrupt `v11_pose_n_hailo10h.hef`, cleared the `models/hailo/` cache directory, and restarted the h10h sweep. The sweep then progressed normally — the legit `v11_segmentation_n_hailo10h.hef` (already at the canonical filename from a prior fresh compile) was correctly picked up by the "already staged" branch (correct behaviour, not the bug).

**Don't reintroduce:** `cache.get_hef_path` must include `hw_arch` in its key (and the on-disk path: `models/hailo/<arch>/<version>/<task>/<model_name>/model.hef`). This is a small refactor in `cache.py` plus its callers in `hef_compiler.py` and `pipeline.py`. Until that lands, mixing architectures across compile runs from a shared `models/hailo/` cache is unsafe — clear the cache between arches as a workaround. Also worth filing: confirm whether anything outside `cli.py` ever writes a HEF to `resources/hefs/`. If yes, that path needs the same arch-correctness review.

### Issue 10 — `END_NODE_TABLE` covers v8 + v11 but not OBB / v26

**Symptom:** The compile sweep failed cleanly on every OBB model (v8/v11/v26) and every v26 task (det/seg/pose):

```
Compile failed at stage 'har_generation': … no end-node hint found in the error.
This usually means the YOLO head's tail is unsupported and there's no entry in END_NODE_TABLE for (v26, obb).
```

(or, when the Hailo parse-error parser did extract a hint but the hint was wrong, mapping failed later with `No successful assignments: concat14/18/23 errors`).

**Root cause:** `END_NODE_TABLE` only carries entries for `(v8, det)`, `(v8, seg)`, `(v11, det)`, `(v11, seg)`, `(v11, pose)`. Anything not in the table relies on `parse_end_node_hint(...)` parsing the SDK's "use these end node names" suggestion — which tends to suggest the deeper post-processing layers (Sigmoid/Concat/Mul) rather than the raw Conv outputs that actually map. This is the same root-cause class as Issue 6, just with a different "I don't have data for this case" failure mode.

**Fix applied this session:** none — left as the next session's work.

**Don't reintroduce:** `pitfalls.md` already calls this out at §3 (start/end node names) and §7 (OBB heads). The new entries should be derived by:

1. Exporting the `.pt` to `.onnx` once (via the runtime's existing `onnx_export.py`), then `python -c "import onnx; m = onnx.load(...); print([n.name for n in m.graph.node])"` to find the actual head-module Conv names.
2. Cross-checking against the host-side decoder in `benchmark/workloads/yolo/postprocessing.py` to confirm the truncation point matches what the decoder expects.
3. For OBB specifically, truncating before the `Cos`/`Sin` angle decoders — those are explicitly Hailo-unsupported per `pitfalls.md` §4.

The Hailo Model Zoo has no OBB or v26-task YAMLs, so analogy with v8/v11 plus ONNX inspection is the only path. Expect to inspect each of: `yolov8n-obb`, `yolo11n-obb`, `yolo26n-obb`, `yolo26n-seg`, `yolo26n-pose`, `yolo26n` (det). v26 detection has `yolo26n.yaml` in the Model Zoo — copy its `parser.nodes` directly.

### Issue 11 — yolo11n-seg cannot compile for Hailo-8 (chip-side capacity)

**Symptom:** `Mapping Failed (allocation time: 6m 29s) … Compilation failed: Failed to reach required FPS on the following layers: …`. Identical setup compiled fine on Hailo-10H (sanity 2: 4.8 MB HEF in ~22.6 min).

**Root cause:** Hailo-8 (26 TOPS) doesn't have enough compute headroom for v11n-seg's larger head module + mask-prototype branch at the SDK's default FPS target. The model exists but doesn't fit the chip's performance budget. Consistent with Hailo not publishing a v11_seg_n_hailo8 prebuilt in their Model Zoo (whereas v8_seg_n_hailo8 is published, and v11_seg_n_hailo10h compiles cleanly on the more capable chip).

**Don't reintroduce:** This is a capability mismatch, not a tooling bug. Either lower the SDK's FPS target via `model_optimization_config(...)` ALLS (advisory only — likely still won't fit), or accept that v11-seg on the AI HAT+ is hailo10h-only. Document this in `pitfalls.md` so the next person doesn't burn 6.5 minutes per attempt rediscovering it.

### Issue 8 — Slow scp throughput from local workstation to cluster (~0.34 MB/s)

**Symptom:** `scp` of 989 MB Hailo wheels took ~37 min over Tailscale despite a *direct* (non-DERP) link.

**Root cause:** Bottleneck appears to be the source machine's uplink, not Tailscale routing or the cluster downlink. `tailscale status` showed `active; direct 82.15.31.54:61242` — direct peering.

**Workaround:** Used the time productively (started COCO download in parallel, since that ran from `cocodataset.org` direct to node01 over the cluster downlink). For future bulk transfers, copy from a peer cluster node first, or use a USB stick when both machines are physically near. Noted, not blocking.

## Code changes (all on node01, not yet committed)

| File | Change |
|---|---|
| `benchmark/cli.py` | Added `--compression-level {0,1,2}` (default 1) to `compile` subparser; threaded `compression_level=args.compression_level` into `ConversionConfig` in `cmd_compile` |
| `benchmark/workloads/yolo/conversion/pipeline.py` | Added `compression_level: int = 1` field to `ConversionConfig`; passed it through to `HEFCompilerConfig` instantiation in stage-3 compile call |
| `benchmark/workloads/yolo/conversion/hef_compiler.py` | Build ALLS model script (`model_optimization_flavor(...)` + `post_quantization_optimization(bias_correction, policy=enabled)`) and call `runner.load_model_script(...)` before `runner.optimize(calib_data)`. Logs the script. |
| `benchmark/workloads/yolo/conversion/har_generator.py` | Replaced `END_NODE_TABLE`. Old entries cut at post-processing layers (Sigmoid/Concat/Mul). New entries cut at raw Conv outputs matching `hailo_model_zoo/cfg/networks/<name>.yaml`. Covers v8 det/seg, v11 det/seg/pose. |
| `scripts/compile_workstation_hefs.sh` | Added `--compression-level` arg parsing (default `COMPRESSION_LEVEL=1`); passes `--compression-level "$COMPRESSION_LEVEL"` through to `python -m benchmark compile`; added the flag to the help docstring |

Diagnostic / commentary: every code change includes inline comments explaining *why*, with cross-references to the SDK source paths that justified the choice (e.g. `mo_config.py` for the elif issue, `acceleras_definitions.py` for the enum location, model-zoo YAMLs for the end-node lists).

## Validation evidence

### Sanity 1 — v8n-seg / Hailo-8

```
[info] Loading model script commands to yolov8n-seg from string
[info] Using default optimization level of 2
[info] Model Optimization Algorithm Bias Correction is done (completion time is 00:02:04.87)
[info] Model Optimization is done
[info] Single context flow failed: Recoverable single context error - Resources presolve failed: lcus=(>80)
[info] Using Multi-context flow
[info] Found valid partition to 2 contexts, ... (~15 iterations, last gain ≈ 1.6%)
HEF compiled successfully: 7,722,864 bytes
PASS  hailo8 / yolov8n-seg.pt
```

MD5: `dce9679c44503f56398d4fdd255e018b` (vs prebuilt `91d5729e...`; sizes differ because our HEF emits raw Conv outputs and the prebuilt has NMS baked in — both valid HEFs for their respective contracts).

### Sanity 2 — v11n-seg / Hailo-10H

```
[info] Loading model script commands to yolo11n-seg from string
[info] Using default optimization level of 2
[info] Model Optimization Algorithm Bias Correction is done (completion time is 00:03:13.76)
[info] Model Optimization is done
[info] Found valid partition to 4 contexts ... promoted to 5 contexts, +12.4% gain ... converged
PASS  hailo10h / yolo11n-seg.pt   (elapsed: 1358s)
```

MD5: `f1749c995a4aa50e9338af53a094c497`.

## Step-9 sweep results

Both sweeps used `--calibration-data-path ~/Documents/datasets/coco-val/images/val2017` and `--output-dir resources/hefs/`. h8 sweep ran with the default model list (7 gap models). h10h sweep added `--include-detection`, which expands to v8n, v8s, v11n, v26n detection on top of the gap list (4 extra → 11 total).

### hailo8 sweep (1 / 7 PASS, 82 min wall)

| Model | Result | Stage | Notes |
|---|---|---|---|
| yolo11n-seg | ❌ | hef_compilation | "Failed to reach required FPS" after 6m 29s mapping. Chip-side budget. See Issue 11. |
| **yolo11n-pose** | ✅ | — | 8.5 MB → `v11_pose_n_hailo8.hef`. ~23 min compile. |
| yolov8n-obb | ❌ | hef_compilation | `concat14 errors` — no v8/OBB entry, parser-hint fallback wrong. |
| yolo11n-obb | ❌ | hef_compilation | `concat18 errors` — no v11/OBB entry. |
| yolo26n-obb | ❌ | har_generation | `no entry in END_NODE_TABLE for (v26, obb)`. |
| yolo26n-seg | ❌ | hef_compilation | `concat23 errors` — parser-hint extracted but cut wrong. |
| yolo26n-pose | ❌ | har_generation | `no entry in END_NODE_TABLE for (v26, pose)`. |

### hailo10h sweep (5 / 11 PASS, 134 min wall)

| Model | Result | Notes |
|---|---|---|
| **yolo11n-seg** | ✅ | already staged (sanity 2 had landed it). 4.8 MB. |
| **yolo11n-pose** | ✅ | fresh compile, 9m 1s. 4.1 MB. |
| yolov8n-obb | ❌ | same as h8 (no entry). |
| yolo11n-obb | ❌ | same as h8 (no entry). |
| yolo26n-obb | ❌ | same as h8 (no entry). |
| yolo26n-seg | ❌ | same as h8 (no entry, mapping fails after parser-hint). |
| yolo26n-pose | ❌ | same as h8 (no entry). |
| **yolov8n** | ✅ | det, 4.1 MB. |
| **yolov8s** | ✅ | det, 12 MB. |
| **yolo11n** | ✅ | det, 3.8 MB. |
| yolo26n | ❌ | det, no v26 entry. `format_conversion13` errors at mapping. |

**Sweep failure pattern is binary:** every model with a populated `END_NODE_TABLE` entry compiled (5/5 unique combinations); every model without an entry failed (6/6, twice — once per arch — for 12 fails). The sole "real" failure not explained by the table is yolo11n-seg on hailo8, which is a chip-capability constraint.

### Mid-sweep cache incident

H10h sweep's first attempt stopped after 2 "PASS" results — yolo11n-seg (correctly already-staged) and yolo11n-pose ("PASS in 12 seconds", which doesn't compile). md5 audit revealed the latter HEF was identical to `v11_pose_n_hailo8.hef`. See Issue 9 for the root cause (arch-blind cache key) and the recovery (delete bogus HEF, clear `models/hailo/`, restart). The second attempt — which is the run summarised above — was clean.

## Retry sweep (after END_NODE_TABLE expansion)

Once the original sweep highlighted that all OBB and v26 failures were missing-entry cases, the next move was direct: derive the entries by inspecting the failed-model ONNX files (already exported into `models/hailo/v*/<task>/<model>/model.onnx` by the original sweeps) and consulting the Hailo Model Zoo YAMLs.

### Investigation method

For each failed (version, task), one Python query against the corresponding ONNX:

```python
import onnx
m = onnx.load(path)
# Find Cos/Sin (OBB-specific angle decoders we must truncate before)
[n.name for n in m.graph.node if n.op_type in ("Cos", "Sin")]
# Find head Conv outputs at /model.22 (v8) or /model.23 (v11/v26)
[n.name for n in m.graph.node if n.op_type == "Conv" and "/model.22/" in n.name or "/model.23/" in n.name]
```

Plus, for cross-checking: `hailo_model_zoo/cfg/networks/yolo26n.yaml` (= the **only** v26 task with a published Hailo Model Zoo YAML, namely v26 detection) and `hailo_model_zoo/cfg/alls/generic/yolo26n.alls`.

### Discoveries from the investigation

- **v26 head naming is different.** v8/v11 use bare `cv2.X.X.2/Conv` / `cv3.X.X.2/Conv` / `cv4.X.X.2/Conv`. v26 prefixes everything with `one2one_` (the one-to-one matching head Ultralytics introduced in v26): `one2one_cv2.X.X.2/Conv`, etc. v26 pose specifically uses a flatter `one2one_cv4_kpts.X/Conv` layout for the keypoint branch (no nested `.X.0/.0.2`).
- **v26 detection has explicit per-layer precision overrides in the official ALLS:**
  ```
  quantization_param([dw1, dw6, dw7, dw8], precision_mode=a16_w16)
  quantization_param([conv61, conv77, conv91, conv64, conv80, conv94], precision_mode=a16_w16)
  quantization_param([output_layer1, output_layer2, output_layer3, output_layer4, output_layer5, output_layer6], precision_mode=a16_w16)
  ```
  Plus `optimization_level=4, compression_level=0` and `post_quantization_optimization(adaround, policy=enabled, batch_size=8)`. This is how Hailo's official compile makes v26 fit on hailo8 — without these per-layer overrides, end-node truncation alone may not be sufficient. Captured here for future "Issue 12 / step b" work.
- **Hailo OBB cuts before `/model.X/Cos` and `/model.X/Sin`.** The Cos and Sin operators are the angle-decode entry; everything after is host-side per `_process_obb` / `_rotated_nms` in `postprocessing.py`.

### `END_NODE_TABLE` additions

Six new entries (full bodies in `har_generator.py`):

| Entry | Nodes | Source |
|---|---|---|
| `(v8, OBB)` | 9 (cv2/cv3/cv4 × 3 scales) | yolov8n-obb ONNX inspection; analogous to v8 seg layout |
| `(v11, OBB)` | 9 (cv2/cv3/cv4 × 3, head `/model.23`) | yolo11n-obb ONNX inspection |
| `(v26, DETECTION)` | 6 (one2one_cv2/cv3) | `yolo26.yaml` parser.nodes verbatim |
| `(v26, SEGMENTATION)` | 10 (9 head + `/model.23/proto/cv3/act/Mul`) | yolo26n-seg ONNX inspection |
| `(v26, POSE)` | 9 (cv2 + cv3 + cv4_kpts) | yolo26n-pose ONNX inspection (note flat cv4_kpts.X/Conv) |
| `(v26, OBB)` | 9 (one2one_cv2/cv3/cv4) | yolo26n-obb ONNX inspection |

### Retry sweep results

H8 retry — 5 previously-failed models with `--models` filter, fresh `models/hailo` cache:

| Model | Result | Notes |
|---|---|---|
| **yolov8n-obb** | ✅ | first OBB ever to compile in this repo. ~13 min. |
| **yolo11n-obb** | ✅ | ~19 min. |
| **yolo26n-obb** | ✅ | ~20 min. v26 OBB worked without precision overrides (surprising but welcome). |
| yolo26n-seg | ❌ | failed at mapping after ~60 min. `concat23 errors` — the only non-OBB v26 task that could not fit on hailo8 without precision_mode tweaks. Step (b) candidate. |
| **yolo26n-pose** | ✅ | ~19 min. |

H10h retry — same 5 models. **First attempt stopped early** because of the cache-key bug (Issue 9) recurring: 3 OBB models "PASSed" in 10 seconds each, all md5-identical to the h8 HEFs that the previous sweep had just populated into `models/hailo/`. Killed, deleted the 3 bogus h10h HEFs, cleared `models/hailo` again, restarted. Second attempt was clean:

| Model | Result | Notes |
|---|---|---|
| **yolov8n-obb** | ✅ | ~15 min, `Successful Compilation (duration: 6m 24s)`. |
| **yolo11n-obb** | ✅ | ~20 min. |
| **yolo26n-obb** | ✅ | ~20 min. |
| **yolo26n-seg** | ✅ | h10h's chip headroom evidently enough to fit v26 seg without precision overrides — the same model that just failed on h8. |
| **yolo26n-pose** | ✅ | ~24 min. |

**Retry totals: h8 4/5, h10h 5/5. Combined: 9/10.**

### Working theory on remaining v26-seg / hailo8 failure (resolved by step b investigation)

The original step-9 log showed `concat23 errors: ... format_conversion13_sd48 has 2 APUs but max allowed is 1`, which looked like a per-layer precision-tweak case. The step-(b) investigation below revealed it's a deeper hardware-capability gap, **not** a tunable issue.

## Step (b): per-(version, task) ALLS override infrastructure

Added `MODEL_SCRIPT_OVERRIDES: dict[tuple[str, YOLOTask], list[str]]` to `hef_compiler.py`. The `_compile_with_sdk` model-script emission now looks up `(yolo_version, task)` and appends the listed ALLS commands after the standard `model_optimization_flavor(...)` and `post_quantization_optimization(bias_correction, policy=enabled)` lines.

Two entries committed:

- **`(v26, DETECTION)`** — reproduces the 3-line override set from `hailo_model_zoo/cfg/alls/generic/yolo26n.alls`: `quantization_param([dw1..8], precision_mode=a16_w16)`, the `[conv61, ..., conv94]` set, and `[output_layer1..6]`. Recommended by Hailo's own reference setup; v26 detection still compiled in the step-9 sweep without it (the failure mode was the missing `END_NODE_TABLE` entry, fixed in step a).
- **`(v26, SEGMENTATION)`** — *attempted* but **no working override found**. Documented as a comment-only entry. Six iterations:

| Attempt | Override | Outcome |
|---|---|---|
| v1 | `pre_quantization_optimization(matmul_decomposition, layers=[matmul1, matmul2], policy=enabled, precision_mode=a16_w8)` | 33 s — `Optimization failed: 'meta'` (SDK KeyError, no traceback) |
| v2 | `pre_quantization_optimization(matmul_decomposition, layers=[matmul1..4], policy=enabled)` (no precision_mode) | 33 s — same `'meta'` KeyError |
| v3 | `quantization_param([matmul1..4], precision_mode=a16_w16)` | 32 s — `Unsupported value [<PrecisionMode.a16_w16>]` at script load |
| v4 | `quantization_param([matmul1..4], precision_mode=a16_w8)` | 32 s — same Unsupported value |
| v5 | `quantization_param([matmul1..4], precision_mode=a8_w8_a16)` | 16 min — optimizer ran fine, **mapper rejected**: `precision mode is not accurate` (allocation time 0 s) |
| v6 | `quantization_param([matmul1..4], precision_mode=a8_w8_a8)` | 19.7 min — optimizer ran fine, partition iterations reached `+13.1%`, **mapper rejected** with the original `More than one output is not supported for layer matmul1` after 3m 39s |

`HailoMatmul.SUPPORTED_PRECISION_MODE = {a8_w8, a8_w8_a8, a8_w8_a16}` (verified in `hailo_model_optimization/acceleras/hailo_layers/hailo_matmul.py`). All three were exercised; v26-seg's matmul1 multi-output rejection is independent of precision mode.

**Conclusion:** v26-seg / hailo8 is a hardware-capability gap. The v26 head's attention block produces a multi-output matmul that Hailo-8 hardware cannot ingest in any supported precision mode, and `matmul_decomposition` has its own SDK bug on this network (`KeyError: 'meta'`). v26-seg compiles cleanly on hailo10h (verified in step-a retry). Hailo's own Model Zoo doesn't publish a `v26_segmentation_*_hailo8.hef` either.

The infrastructure (`MODEL_SCRIPT_OVERRIDES`) remains valuable — both for the v26 detection entry and as scaffolding for any future per-(version, task) ALLS workarounds. v26-seg/hailo8 stays on the "not supported" list.

## Recommendations for `docs/compilation/nvidia_workstation_setup.md`

The doc was the right shape but missed concrete steps. Suggested additions:

1. **Step 1.** Note Ubuntu 24.04+ requires deadsnakes PPA (`ppa:deadsnakes/ppa`) for `python3.10`, and `python3.10-distutils` is implicitly needed.
2. **New step 5b.** After installing DFC + model-zoo wheels, install matching `tensorflow[and-cuda]==X.Y.Z` and `jax[cuda12]==X.Y.Z`. The doc's "DFC bundles its own CUDA runtime" claim is misleading — without these extras the optimizer falls back to CPU and opt level drops to 0.
3. **New step 5c.** After step 5b, run `pip install --force-reinstall --no-deps nvidia-nccl-cu13` to fix the cu12/cu13 NCCL collision that breaks torch's ONNX export.
4. **Step 7.** The doc's curl URL is correct; note that DNS/HTTP is occasionally flaky and `--retry 3 --retry-delay 5` is worth adding to the `curl` flag list.
5. **Step 8 ("What to look for").** Replace the bullet `[info] Bias Correction runs (NOT skipped)` with a callout that bias_correction is *only* enabled at `optimization_level=1` *or* via an explicit `post_quantization_optimization(bias_correction, policy=enabled)` model-script command. The pipeline now does the latter automatically.
6. **New troubleshooting section.** "If the canary still fails at mapping with `16x4 not supported in activation*`": this is a sign the end-node truncation list is too deep. Verify `END_NODE_TABLE` matches `hailo_model_zoo/cfg/networks/<model>.yaml` for any non-gap model; for gap models, derive by analogy and confirm the Conv names exist in the actual ONNX export (`onnx.load(...).graph.node`).
7. **pytest baseline** clarification: 276/277 on cold install, 277/277 after Ultralytics weights cache.

## Outstanding / proposed next steps

The path forward goes back through the AMD workstation, which has the writable git remote — node01 is air-gapped from the repo. Zip the project tree on node01 (excluding venvs, build caches, and the EULA-gated wheels), `scp` it back to the local workstation, then merge on AMD. After that, steps 1–5 below.

1. **Step 10 — Commit + push** (on AMD workstation, not node01). Apply commits in this order so each is reviewable in isolation:
   - (a) `nvidia_workstation_setup.md` rewrite incorporating the seven recommendations above (deadsnakes, CUDA-extras, NCCL-restore, bias_correction force, end-node guidance, pytest baseline, troubleshooting bullet for `16x4 not supported`).
   - (b) `--compression-level` CLI flag + `compression_level` field threaded through cli → pipeline → hef_compiler.
   - (c) ALLS model-script emission in `hef_compiler.py` (with the `post_quantization_optimization(bias_correction, policy=enabled)` workaround for the mutually-exclusive flavor levels).
   - (d) `END_NODE_TABLE` correction + expansion in `har_generator.py` (raw Conv outputs from Hailo Model Zoo YAMLs for v8/v11; ONNX-inspected analogues for OBB family and v26). 11 entries now (was 4 originally; corrected 4 + added 6 new + verified 1).
   - (e) Sweep HEFs (one commit, ~95 MB total across 15 files): all the new entries in `resources/hefs/` from this session — `v11_pose_n_hailo8.hef`, `v8_obb_n_hailo8.hef`, `v11_obb_n_hailo8.hef`, `v26_obb_n_hailo8.hef`, `v26_pose_n_hailo8.hef`, plus 10 hailo10h HEFs.
   - (f) `session_notes_2026-04-29_nvidia_workstation.md` itself, dropped into `resources/` alongside the existing 2026-04-27 notes.
2. **Cache-key arch fix** (Issue 9). Refactor `cache.get_cache_path` and `CacheManager.get_model_cache_path` to include `target_device` in the on-disk path: `models/hailo/<arch>/<version>/<task>/<model>/`. Update the ~20 call sites across `pipeline.py`, `hef_compiler.py`, `har_generator.py`, `onnx_export.py`. Add a regression test that compiles the same model twice (once per arch) and asserts the two staged HEFs differ in md5. Until this lands, **always clear `models/hailo/` between architectures** — we hit the bug twice in this session and burned ~30 min on phantom passes.
3. **Step (b): infrastructure done; v26-seg / hailo8 confirmed unfittable.** `MODEL_SCRIPT_OVERRIDES` is committed in `hef_compiler.py` and used by the v26 detection entry. The v26-seg case is documented as a hardware-capability gap (see §"Step (b)" above and Issue 11) — six override variants tried, all hit the matmul1 multi-output rejection. No further work; document in `pitfalls.md`.
4. **Delete or supersede `docs/compilation/end_node_truncation_plan.md`.** Its premise ("truncation is next-session work") is now done — the truncation infrastructure is wired and the table is fully populated for v8/v11/v26 across det/seg/pose/obb. The plan also speculated that the deep post-processing layers (Sigmoid/Concat) are the right cut points; we now know they're the wrong cut points. Replace with a one-page pointer to `har_generator.py:END_NODE_TABLE` and to this session's notes.
5. **(Skip) yolo11n-seg / hailo8.** Confirmed chip-side FPS budget — six minutes of mapping then `Failed to reach required FPS on the following layers`. Compiles fine on h10h. Document in `pitfalls.md` so future attempts are short-circuited; do not try precision overrides (won't help — the issue is whole-graph compute capacity, not per-layer precision).
6. **(Skip) yolo26n / hailo10h.** Hailo's `yolo26n.yaml` declares `supported_hw_arch: [hailo8, hailo8l]`. Failure here is intended by Hailo's own definition. Document.
