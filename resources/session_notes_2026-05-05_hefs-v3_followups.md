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
