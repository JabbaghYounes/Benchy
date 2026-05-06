# Session Notes — 2026-05-05 — hefs-v3 Cycle Closed, Deferred Follow-ups

The hefs-v3 release cycle is closed (release published 2026-05-04, both
Pis re-verified at 14/14 on 2026-05-05, showcase + README + docs all
refreshed). The items below are non-blocking deferrals — repo is in a
stable state without them.

## 1. LLM-CPU anomaly on AI HAT+ (0.62 TPS)

**State:** Persistent across all post-merge runs. AI HAT+ Pi runs
`llama3.2:1b` on Ollama CPU at ~0.62 TPS vs the AI HAT+ 2 Pi at
~6.85 TPS for the same model. ~11× slower on the same Cortex-A76 SoC.

**Suspected root cause:** Different default quantization being pulled —
per-run CSVs show AI HAT+ pulled `Q8_0` explicitly while AI HAT+ 2 used
the `Q4_0` default. Could also be thermal/governor differences (AI HAT+
Pi has more thermal headroom from 16 GB RAM cooling but might have a
different power-mode default). TTFT is ~4× higher too (~1922 ms vs
~454 ms), which fits a quant-size hypothesis (bigger weights → slower
prefill).

**Why it's deferred:** Doesn't affect the cross-platform prescription —
inter-board LLM-CPU comparison is already invalid (different RAM
configurations, different ollama state). Caveat #2 in
`docs/showcase.md` already calls this out.

**To investigate:** SSH into the AI HAT+ Pi, run `ollama show
llama3.2:1b` to see actual quantization, compare to AI HAT+ 2 Pi.
If quant differs, force both to the same tag (`llama3.2:1b-instruct-q4_0`
or similar) in `configs/llm_benchmark.yaml` and re-verify.

## 2. More pose HEFs for a future `hefs-v4`

**State:** Documented gaps in `docs/hailo.md` § "Known gaps":

| Missing HEF | Notes |
|---|---|
| `v8 pose x_hailo10h` | Bigger than v8l; will need even smaller finetune batch (`batch_size=2`?) or QAT disabled. Not in current verify matrix. |
| `v11 pose s_hailo10h`, `v11 pose m_hailo10h`, `v11 pose l_hailo10h` | The existing `(v11, POSE)` `END_NODE_TABLE` entry should cover HAR generation, but a `MODEL_SCRIPT_OVERRIDES` entry analogous to `(v8, POSE)` is likely needed for the bigger sizes. Not attempted. |

**Why it's deferred:** Nothing in the current verify scripts exercises
these. The 14-step verify is comprehensive enough for the cross-platform
showcase the project exists to produce.

**To do:** When ready, write a workstation prompt in the same shape as
`/tmp/benchy-hefs-v3-instructions.md` — point at `node01`, the
`venv-compile-h10h` venv, DFC 5.3.0, and the matching session-notes
expectation. Then publish hefs-v4 with the same flow used for v3.

## 3. Rebuild `v8_pose_n_hailo10h` and `v8_pose_s_hailo10h` under `batch_size=4`

**State:** Both were compiled in hefs-v2 under the SDK default
`batch_size=8` finetune. The hefs-v3 patch added `batch_size=4` for
`(v8, POSE)` to fit `v8l-pose` QAT in 11 GB VRAM. Smaller variants
fit at batch=8 fine, but a future rebuild would put the entire v8
pose family under the same finetune regime.

**Why it's deferred:** Workstation session notes flagged it as a
"smoke test if accuracy regressions show up downstream." None have.
Cross-Pi `yolov8s-pose` numbers (87.56 FPS on Hailo-8, 14.82 FPS on
Hailo-10H) are stable and predictable.

**To do:** Only worth doing if Phase 4 (or whatever's next) needs the
v8 pose family for accuracy-sensitive work. Until then, leave the
existing HEFs.

## 4. Layer Noise SNR statistics for v8m/l-pose compiles

**State:** The DFC's `Full Quant Analysis` step OOMs at 50% iteration
on the workstation 2080 Ti for `v8m-pose` and `v8l-pose` (GPU has 11 GB
VRAM, the analysis pass needs more). SDK silently falls back to CPU
and continues, but does not emit per-layer SNR numbers in this mode.
HEF compile-stage gates all "Pass" — correctness is unaffected.

**Why it's deferred:** Diagnostic-only; doesn't change what the chip
actually runs. Output-layer compile gates are the load-bearing check
and they're all green.

**To do:** Add `model_optimization_config(checker_cfg, batch_size=1)`
to the `(v8, POSE)` entry in `MODEL_SCRIPT_OVERRIDES`
(`benchmark/workloads/yolo/conversion/hef_compiler.py:126`) — would
keep the analysis on GPU at 1/8 the working set. Or
`policy=disabled` to skip Full Quant Analysis entirely if the SNR
numbers aren't going to be acted on. Either is a one-line change to
the ALLS recipe.

## 5. Stabilise the `yolov8n-pose` head-to-head row

**State:** As of 2026-05-05, both Pis have run `yolo-v8-pose-n` exactly
once. The numbers (65.96 FPS on Hailo-8 vs 15.61 FPS on Hailo-10H,
4.2× lead) are in `docs/showcase.md` without std bands.

**Why it's deferred:** It's just verify-cadence — every subsequent
sweep on either Pi tightens the std. No action needed unless the row
gets weird.

**To do:** Just run more verify sweeps. Re-run
`scripts/aggregate_by_platform.py` → `generate_showcase_charts.py`
→ `regenerate_showcase_dashboard.py` after pushes to refresh the
showcase numbers with std bands.

## 6. Close the Hailo-10H size-ladder gap (`hefs-v4` scope)

**State:** Hailo-8 has the full v8 detection ladder (n/s/m/l/x) and
v8-seg up to `m` staged in hefs-v3. Hailo-10H tops out at v8 det `s`
and has no v8-seg HEFs at sizes above `n`. This is what blocks the
head-to-head chart in `docs/showcase.md` from showing scaling beyond
nano — `scripts/generate_showcase_charts.py:yolo_throughput_chart`
filters to the intersection of models present on both Pis (line 51),
so any model only on one chip is dropped.

| Missing HEF | Path |
|---|---|
| `v8 det m/l/x_hailo10h` | Likely Hailo Model Zoo S3 prebuilts (detection sizes are usually published). Try `scripts/fetch_prebuilt_hefs.py --arch hailo10h --dry-run` first to see what's available without re-compiling. |
| `v8 seg s/m_hailo10h` | Workstation gap-model compile, same path as the hefs-v3 sweep on `node01`. `(v8, SEGMENTATION)` `END_NODE_TABLE` entry already exists. v8m-seg likely needs `batch_size=4` in `MODEL_SCRIPT_OVERRIDES` (same VRAM bottleneck that bit v8l-pose). |

**Why it's deferred:** Verify scripts only exercise nano sizes
today, so even closing the gap doesn't immediately surface in any
chart — must be paired with item 7 to be visible. Pure HEF coverage
is otherwise non-blocking.

**To do:** Pull what the Model Zoo publishes via the fetcher,
write a workstation prompt for the seg gap analogous to
`/tmp/benchy-hefs-v3-instructions.md`, stage into `resources/hefs/`,
regenerate manifest, publish `hefs-v4`.

## 7. Per-platform charts in the showcase (chart code landed; verify expansion pending)

**State (updated 2026-05-06):** Chart code shipped —
`scripts/generate_showcase_charts.py:yolo_per_platform_charts()`
produces `yolo_per_platform_hailo8.png` (9 models) and
`yolo_per_platform_hailo10h.png` (11 models, gains
`yolo11n-seg` + `yolo26n-seg`); both are embedded in
`docs/showcase.md` under "Per-platform breakdown" and in
`README.md`. The remaining work is **populating bigger sizes** —
verify scripts still only exercise `n` (and `yolov8s-pose`), so
the per-platform charts max out at the same nano models as the
head-to-head plus the two Hailo-10H-only seg rows.

**Why it's deferred:** Adding verify steps for `m / l / x` sizes
adds wall time per sweep — needs deliberate scheduling, not just
a code change.

**To do:**

1. Expand `scripts/verify_ai_hat_plus.sh` with size-ladder steps
   that match the staged Hailo-8 HEFs: `yolo-v8-detection-{s,m,l,x}`,
   `yolo-v8-seg-{s,m}`, `yolo-v8-pose-m`. Re-run on the AI HAT+ Pi.
2. Expand `scripts/verify_ai_hat_plus_2.sh` with whatever ladder
   Hailo-10H has after item 6 lands (today: `s` for det, `s/m/l`
   for pose).
3. Re-run `aggregate_by_platform.py` →
   `generate_showcase_charts.py` → `regenerate_showcase_dashboard.py`.
   Existing per-platform chart code will pick up the new rows
   automatically — no chart edits needed unless we want size-axis
   grouping (e.g. n / s / m / l / x as a secondary x-axis).

## 8. Jetson Orin Nano: GPU not selected during third-party run

**State:** A CSV from a Jetson Orin Nano run was added at
`results/jetson_orin_nano/jetson-run.csv` on 2026-05-06
(downloaded from a third-party — not run locally). The numbers
strongly suggest the GPU was never engaged:

| Model | FPS | Latency |
|---|---:|---:|
| `yolov8n` | 1.85 | 541.8 ms |
| `yolo26n` | 2.17 | 460.9 ms |

A Jetson Orin Nano on the Ampere GPU should hit ~50+ FPS on
yolov8n at 640×640. 1.85 FPS is in PyTorch-CPU territory.
Corroborating signs in the CSV columns:

- `accelerator_percent_mean` is empty (GPU collector never reported)
- `power_watts_mean` is empty
- `cpu_percent_mean` is 33-41% (load is on the CPU)
- `num_runs=1` (no std bands, single-sample)

**Why it's deferred:** No local Jetson hardware available. Without
hands on the device it's not possible to debug backend selection
or repair the run. **Do not embed these numbers in the showcase or
README** — they would mislead viewers about Jetson Orin Nano's
actual performance ceiling.

**To investigate when a Jetson is in hand:**

1. Confirm `python -m benchmark info` reports
   `Platform.JETSON_ORIN_NANO` (otherwise `detect_platform()` in
   `benchmark/metrics/collectors.py` is failing on this kit, and
   the platform-aware backend selection won't trigger).
2. Confirm `python -m benchmark backends` lists a CUDA-capable
   backend as available; verify with
   `python -c "import torch; print(torch.cuda.is_available())"`.
   On JetPack, torch must be the NVIDIA-published wheel — the PyPI
   wheel is CPU-only.
3. Inspect `benchmark/workloads/yolo/backends/registry.py` and
   `pytorch.py` — if Jetson auto-selects pytorch but pytorch falls
   back to CPU silently because `torch.cuda.is_available() is
   False`, that's the smoking gun.
4. Re-run with `python -m benchmark run yolo` (auto-select) and
   confirm `accelerator_percent_mean` / `power_watts_mean` are
   populated in the resulting `bench_*.json`. A correct GPU run
   should also drop CPU usage well below the 33-41% range from
   the existing CSV.
5. Once verified `bench_*.json` files exist under
   `results/jetson_orin_nano/`,
   `scripts/aggregate_by_platform.py` will pick the platform up
   automatically (no code changes needed). The chart generator's
   intersection-filter (`yolo_throughput_chart`) currently
   hardcodes the two Hailo platforms — `yolo_per_platform_charts`
   does not, so Jetson would appear as a third per-platform figure
   with zero edits. Adding Jetson to the head-to-head chart needs
   a small rework of the platform pair logic in
   `scripts/generate_showcase_charts.py`.
