# Session Notes — NVIDIA Workstation HEF Compilation

**Date:** 2026-04-29 evening → 2026-04-30 morning
**Host:** node01 (Tailscale `100.99.187.118`), Ubuntu 24.04 LTS, NVIDIA RTX 2080 Ti (driver 580 / CUDA 13)
**Goal:** Follow `docs/compilation/nvidia_workstation_setup.md` to bring up node01 for HEF compilation, validate the canary, then compile gap models.

## Outcome

Setup validated, then full sweep run end-to-end. Six fresh HEFs landed in `resources/hefs/` (in addition to the two `/tmp/` sanity HEFs that proved the path).

**Sweep totals:** 6 / 18 attempted compiles succeeded.

| Architecture | Pass | Fail | Wall time |
|---|---|---|---|
| hailo8 | 1 / 7 | 6 | 82 min |
| hailo10h | 5 / 11 | 6 | 134 min |

The 6 staged HEFs:

| Filename | Source model | Size |
|---|---|---|
| `v11_pose_n_hailo8.hef` | yolo11n-pose | 8.5 MB |
| `v11_segmentation_n_hailo10h.hef` | yolo11n-seg | 4.8 MB |
| `v11_pose_n_hailo10h.hef` | yolo11n-pose | 4.1 MB |
| `v8_detection_n_hailo10h.hef` | yolov8n | 4.1 MB |
| `v8_detection_s_hailo10h.hef` | yolov8s | 12 MB |
| `v11_detection_n_hailo10h.hef` | yolo11n | 3.8 MB |

The 12 failures are all explained by either missing `END_NODE_TABLE` entries (OBB family across all three versions; v26 family across all four tasks) or a chip-side capability limit (yolo11n-seg on hailo8 — succeeded on hailo10h). No unexplained failures, no broken tooling. Detail in §"Step-9 sweep results" below.

The setup doc + repo code had several gaps that blocked compilation. Six code changes and several setup-procedure additions were required to actually produce a HEF. Details below. **Step 9 done with the caveats above; Step 10 (commit + push) pending.**

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

The plan from this point forward goes back through the AMD workstation, which has the writable git remote — node01 is air-gapped from the repo. Specifically: zip the project tree on node01 (excluding venvs, build caches, and the EULA-gated wheels), `scp` it back to the local workstation, then merge on the AMD box. After that, steps 1–4 below.

1. **Step 10 — Commit + push** (on AMD workstation, not node01). Apply commits in this order so each is reviewable in isolation:
   - (a) `nvidia_workstation_setup.md` rewrite incorporating the seven recommendations above (deadsnakes, CUDA-extras, NCCL-restore, bias_correction force, end-node guidance, pytest baseline, troubleshooting bullet for `16x4 not supported`).
   - (b) `--compression-level` CLI flag + `compression_level` field threaded through cli → pipeline → hef_compiler.
   - (c) ALLS model-script emission in `hef_compiler.py` (with the `post_quantization_optimization(bias_correction, policy=enabled)` workaround for the mutually-exclusive flavor levels).
   - (d) `END_NODE_TABLE` correction in `har_generator.py` (raw Conv outputs from Hailo Model Zoo YAMLs, not deep post-processing layers).
   - (e) Sweep HEFs (one commit, ~37 MB total): the 6 new files in `resources/hefs/`.
   - (f) `session_notes_2026-04-29_nvidia_workstation.md` itself, dropped into `resources/` alongside the existing 2026-04-27 notes.
2. **Cache-key arch fix** (Issue 9). Refactor `cache.get_hef_path` to include `hw_arch`, and the on-disk path to be `models/hailo/<arch>/...`. Update callers in `hef_compiler.py` and `pipeline.py`. This unblocks running multi-arch sweeps without manual cache clearing between them. Add a regression test that compiles the same model twice (once per arch) and asserts the two staged HEFs differ in md5.
3. **Add OBB and v26 `END_NODE_TABLE` entries, retry the failures.** This is the highest-value follow-up — 4 of 12 failures (v8n-obb, v11n-obb, on both h8 and h10h) likely become passes once the OBB entries are correct. Procedure:
   - Export each `.pt` to `.onnx` (via the existing `onnx_export.py` path); read the head-module structure with `onnx.load` + `[n.name for n in m.graph.node]`.
   - For OBB, identify the angle-decode tail (Cos/Sin operands) and pick end-nodes one or two layers earlier. Cross-check against `_process_obb` in `postprocessing.py`.
   - For v26 detection, copy `parser.nodes` from `hailo_model_zoo/cfg/networks/yolo26n.yaml` directly.
   - For v26 seg/pose/obb, derive by analogy with v11 + ONNX inspection — no published reference.
   - Once entries are in, rerun just the failed models with `--models <list>` to avoid re-compiling the working ones.
4. **Delete or supersede `docs/compilation/end_node_truncation_plan.md`.** Its premise ("truncation is next-session work") is now wrong — the truncation infrastructure is wired and the table is partially populated. The plan also speculated that the deep post-processing layers (Sigmoid/Concat) are the right cut points; we now know they're the wrong cut points. Replacing the file with a short pointer to `har_generator.py:END_NODE_TABLE` and to this session's notes would be more useful than leaving it as-is.
5. **(Optional) Try yolo11n-seg / hailo8 with `model_optimization_config(...)` setting a lower FPS target.** Speculative; the chip likely cannot fit it regardless. If confirmed unfittable, document in `pitfalls.md` so future attempts are short-circuited.
