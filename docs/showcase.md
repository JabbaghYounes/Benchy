# Benchy Cross-Platform Showcase

> Aggregated results across **2 platforms × 8+ verify runs** of the Benchy
> hardware-verification suite, run on the actual edge AI hardware. Most
> recent runs (both Pis, 2026-05-05) exercise the `hefs-v3` release
> end-to-end — adds `v8_pose_n_hailo8.hef` (closes the AI HAT+ pose gap)
> plus `v8_pose_m_hailo10h.hef` and `v8_pose_l_hailo10h.hef` (extends
> the AI HAT+ 2 v8 pose family to n/s/m/l). Source bundles in
> [`results/rpi_ai_hat_plus/`](../results/rpi_ai_hat_plus/) and
> [`results/rpi_ai_hat_plus_2/`](../results/rpi_ai_hat_plus_2/);
> aggregated CSVs in this directory; full interactive dashboard at
> [`docs/showcase/dashboard.html`](showcase/dashboard.html).

## Headline prescription

| Use case | Winner | Margin |
|---|---|---|
| **Pure vision** (detection / OBB / seg / pose) | **AI HAT+ Hailo-8 (26 TOPS)** | 2.6× – 6.8× faster than AI HAT+ 2 across every comparable model |
| **LLM on NPU** | **AI HAT+ 2 Hailo-10H** | Only choice — Hailo-8 has no onboard SDRAM. **1.49× decode speedup** vs Pi 5 CPU on the same hardware |
| **Both vision AND LLM-on-NPU** | **AI HAT+ 2 Hailo-10H** | Pay 2.6× – 6.8× vision throughput penalty for the LLM capability |

The Hailo-8 (26 TOPS) is **vision-dedicated silicon** with INT8 throughput
tuned for convnets — narrower, faster, simpler. The Hailo-10H trades raw
vision throughput for **40 TOPS at INT4 + 8 GB onboard SDRAM**, which is
what enables LLM hosting at all. You can't have both. The data reflects
exactly that engineering trade.

## Hardware

| | AI HAT+ | AI HAT+ 2 |
|---|---|---|
| Host | Raspberry Pi 5 | Raspberry Pi 5 |
| Accelerator | Hailo-8 NPU (this unit: 26 TOPS) | Hailo-10H NPU |
| Max throughput | 26 TOPS INT8 | 40 TOPS INT4 |
| Onboard SDRAM | None (uses host RAM) | 8 GB onboard |
| LLM-on-NPU capable | No | **Yes** (~6B params via HailoRT GenAI) |
| HailoRT version | 4.x | 5.x |
| HEF supported tasks | det / cls / OBB / seg / pose | det / cls / OBB / seg / pose |
| Verify-suite count | 14 steps (12 active + 2 `[unsupported-on-this-hw]` advisory: v11n-seg, v26n-seg) | 14 steps (all active) |

Both Pis are the same Pi 5 form factor connected over PCIe per the HAT+
spec. AI HAT+ ships in 13 TOPS (Hailo-8L) and 26 TOPS (Hailo-8) variants;
the data here is from the **26 TOPS variant**. AI HAT+ 2 ships in one
configuration only.

## YOLO inference — head to head

All YOLO models tested at **640×640**, mean ± std across multiple runs
per `(platform, model, task)` combination. Hailo backend (INT8 on
Hailo-8, INT4 on Hailo-10H), no CPU fallback.

![YOLO throughput comparison](showcase/charts/yolo_throughput_comparison.png)

| Model | Task | AI HAT+ throughput | AI HAT+ 2 throughput | AI HAT+ wins |
|---|---|---:|---:|---:|
| `yolov8n` | detection | **101.36 ± 1.36 FPS** | 14.91 ± 0.08 FPS | **6.8×** |
| `yolov8s-pose` | pose | **87.56 ± 1.42 FPS** | 14.82 ± 0.08 FPS | **5.9×** |
| `yolov8n-obb` | obb | **71.97 ± 0.91 FPS** | 16.09 ± 0.04 FPS | **4.5×** |
| `yolov8n-pose` | pose | **65.96 FPS¹** | 15.61 FPS¹ | **4.2×** |
| `yolov8n-seg` | segmentation | **63.66 ± 1.89 FPS** | 12.12 ± 0.31 FPS | **5.3×** |
| `yolo26n-obb` | obb | **60.36 ± 1.41 FPS** | 18.30 ± 0.10 FPS | **3.3×** |
| `yolo11n-obb` | obb | **58.17 ± 0.99 FPS** | 15.91 ± 0.10 FPS | **3.7×** |
| `yolo11n-pose` | pose | **54.24 ± 0.56 FPS** | 15.41 ± 0.19 FPS | **3.5×** |
| `yolo26n-pose` | pose | **43.11 ± 0.55 FPS** | 16.40 ± 0.06 FPS | **2.6×** |

¹ `yolov8n-pose` has a single verify run on each Pi so far — both HEFs
(`v8_pose_n_hailo8` and the existing `v8_pose_n_hailo10h`) only became
mutually present in `hefs-v3` (2026-05-04), and the `yolo-v8-pose-n`
verify step was added in the same release. Stds will tighten with
subsequent sweeps.

AI HAT+ 2 also has segmentation models the AI HAT+ chip can't physically
run (Hailo-8 chip-incompatible per `docs/compilation/pitfalls.md` §§
11-12 — both are tagged `[unsupported-on-this-hw]` in the AI HAT+ verify
script and produce empty bench rows there):

| Model | Task | AI HAT+ 2 throughput |
|---|---|---:|
| `yolo11n-seg` | segmentation | 12.14 ± 0.12 FPS |
| `yolo26n-seg` | segmentation | 12.31 ± 0.17 FPS |

### Latency (lower is better)

![YOLO latency comparison](showcase/charts/yolo_latency_comparison.png)

| Model | AI HAT+ latency | AI HAT+ 2 latency |
|---|---:|---:|
| `yolov8n` det | **9.87 ms** | 67.05 ms |
| `yolov8s-pose` | **11.42 ms** | 67.49 ms |
| `yolov8n-obb` | **13.90 ms** | 62.15 ms |
| `yolov8n-pose` | **15.16 ms** | 64.07 ms |
| `yolov8n-seg` | **15.72 ms** | 82.57 ms |
| `yolo26n-obb` | **16.57 ms** | 54.65 ms |
| `yolo11n-obb` | **17.20 ms** | 62.85 ms |
| `yolo11n-pose` | **18.44 ms** | 64.90 ms |
| `yolo26n-pose` | **23.20 ms** | 60.98 ms |

For real-time vision pipelines (>30 FPS), the AI HAT+ comfortably
clears that bar across every standard model. The AI HAT+ 2 is in the
12-18 FPS range — fine for periodic inference (drone snapshots,
surveillance triggers) but not for high-rate video.

### Per-platform breakdown — every model each chip can run

The head-to-head charts above filter to models present on **both**
boards. The two charts below drop that filter — each Pi shows its
full benchmarked-model set, so the AI HAT+ 2's segmentation coverage
(`yolo11n-seg`, `yolo26n-seg` — both chip-incompatible on Hailo-8) is
visible. Bars are coloured by task and sorted descending.

![AI HAT+ — Hailo-8 per-platform throughput](showcase/charts/yolo_per_platform_hailo8.png)

![AI HAT+ 2 — Hailo-10H per-platform throughput](showcase/charts/yolo_per_platform_hailo10h.png)

Hailo-8 lists 9 models; Hailo-10H lists 11 (gains the two Hailo-10H-
only seg HEFs). Bigger size variants (m / l / x) exist as HEFs on
both chips for several families but aren't exercised by the verify
sweep yet — see follow-ups 6 and 7 in
[`resources/session_notes_2026-05-05_hefs-v3_followups.md`](../resources/session_notes_2026-05-05_hefs-v3_followups.md)
for the path to populate them.

## LLM inference — `llama3.2:1b`, drone prompt set

The **AI HAT+ 2 with HailoRT GenAI is the only platform** in this
benchmark that can host LLMs on the NPU. Hailo-8 lacks onboard SDRAM
required for transformer weights.

The comparison below runs both backends on the **same Pi** (the AI HAT+
2 Pi), so it's a fair NPU-vs-CPU comparison without confounds from
different hardware.

![LLM NPU vs CPU](showcase/charts/llm_npu_vs_cpu.png)

| Backend | Decode TPS (mean ± std) | Speedup vs CPU |
|---|---:|---:|
| **Hailo-10H NPU** (`hailo-ollama` GenAI) | **10.09 – 10.48 TPS** (9 runs/prompt) | **1.45× – 1.53×** |
| Pi 5 CPU (`ollama`) | 6.83 – 6.99 TPS (11 runs/prompt) | baseline |

Per-prompt breakdown (5 drone prompt categories), aggregated across
all post-2026-05-02 verify runs:

| Prompt category | NPU TPS | CPU TPS | Speedup |
|---|---:|---:|---:|
| `scene_description` | 10.48 ± 0.07 | 6.98 ± 0.09 | 1.50× |
| `target_identification` | 10.18 ± 0.12 | 6.83 ± 0.17 | 1.49× |
| `mission_preflight` | 10.09 ± 0.04 | 6.86 ± 0.11 | 1.47× |
| `telemetry_interpretation` | 10.11 ± 0.07 | 6.99 ± 0.08 | 1.45× |
| `hazard_reasoning` | 10.43 ± 0.15 | 6.85 ± 0.11 | 1.53× |

### TTFT (prefill latency) — essentially identical

![LLM TTFT comparison](showcase/charts/llm_ttft.png)

Both backends share the same prefill workload, so first-token latency is
near-identical (~454 ms). The NPU's advantage is **decode** (the
per-token cost after the first one) — which is where the 1.49× average
speedup lives, and where it matters for long-form generation.

## Methodology

- **Test rig**: each Pi runs the same `verify_ai_hat_plus*.sh` script —
  14 steps (vision sweep + `yolo-v8-pose-n` added in `hefs-v3` +
  LLM-on-CPU + LLM-on-NPU) — multiple times across multiple days. All
  bundles live under `results/<platform>/`. On AI HAT+, two seg steps
  are tagged `[unsupported-on-this-hw]` because Hailo-8 silicon can't
  fit `yolo11n-seg` (chip-side FPS budget overflow) or `yolo26n-seg`
  (matmul1 unfittable) — they record the failure mode for cross-platform
  comparison without gating exit code.
- **Aggregation**: results grouped by `(platform, model_name, task)`
  for YOLO and `(platform, backend, prompt_id)` for LLM. Mean / std /
  min / max computed per group across all runs. Stub rows
  (`unsupported-on-this-hardware`) are filtered out.
- **YOLO inference**: 3 warmup runs + 10 measured runs per HEF, on the
  Hailo backend (INT8 / INT4 quantization, NPU-resident). No CPU
  fallback — the Hailo backend deliberately doesn't fall back to CPU.
- **LLM inference**: 3 warmup runs + 10 measured runs per prompt,
  pre-warmed (`6f721a3`) before the timed loop so cold-load doesn't
  poison the TPS measurement. NPU path goes through HailoRT GenAI's
  Ollama-compatible REST endpoint on `:8000`; CPU path goes through
  vanilla `ollama` on `:11434`. Same model (`llama3.2:1b`), identical
  request shape.
- **No mAP scoring** in these particular bundles — `--skip-validation`
  is the verify-suite default for speed. Accuracy comparison would need
  separate full-validation runs; see `docs/methodology.md`.

For the full methodology — including validation rigor, calibration,
and the cross-platform fairness rules — see
[`docs/methodology.md`](methodology.md).

## Caveats

1. **Different Pi RAM between the two boards.** AI HAT+ Pi has 16 GB
   RAM; AI HAT+ 2 Pi has 4 GB. This **invalidates inter-board LLM-on-CPU
   comparison** — only trust the same-Pi NPU-vs-CPU comparison from the
   AI HAT+ 2 Pi (where both backends run on identical hardware).
2. **AI HAT+ LLM-CPU number is anomalously low** (~0.62 TPS vs the
   AI HAT+ 2 Pi's CPU at ~6.88 TPS for the same model). Persists across
   10 measurements on the post-merge run, so it's not a fluke or a
   pre-polish artifact — likely root cause is the AI HAT+ Pi pulling a
   different `llama3.2:1b` quantization (Q8_0 explicit per the per-run
   CSV vs Q4 default), or something thermal/governor-related.
   Investigation pending; doesn't affect the prescription since
   inter-board CPU comparison is invalid anyway (different RAM, different
   ollama state).
3. **Power consumption not captured** in these bundles
   (`power_watts: null`) — the prescription is throughput-only, not
   perf/watt. Intuitively the Hailo-10H draws more (higher clock + onboard
   SDRAM) but it's unmeasured here.
4. **Only `n` and `s` size models tested.** For larger sizes (m / l / x)
   memory bandwidth becomes more of a bottleneck on Hailo-8; the gap may
   shrink. Early signal from AI HAT+'s `yolov8s-pose` (87.96 FPS) and
   `yolov8n-seg` (63.45 FPS) suggests the lead holds at small-medium sizes.
5. **HEF coverage gaps closed incrementally across releases** —
   `v8 seg n_hailo10h` closed in `hefs-v1`; `v8 pose n/s_hailo10h`
   closed in `hefs-v2`; `v8 pose m/l_hailo10h` and `v8 pose n_hailo8`
   closed in `hefs-v3` (2026-05-04). The `yolov8n-pose` head-to-head
   row in this showcase is the first cross-platform datapoint enabled
   by hefs-v3 — both Pis ran it for the first time on 2026-05-05; std
   bands will appear once subsequent verify sweeps land. Remaining
   gaps (v8 pose x_hailo10h; v11 seg/pose s/m/l_hailo10h;
   classification across the board) are documented in
   [`docs/hailo.md`](hailo.md#hef-availability-in-hefs-v3).
6. **Both Pis last ran 2026-05-05** (AI HAT+ commit `07bd2a3`,
   AI HAT+ 2 commit `27ce056`) — first synchronous sweep after the
   hefs-v3 release. AI HAT+ numbers are highly repeatable (stds
   <2 FPS); AI HAT+ 2 numbers come from a deeper history (10-11 runs
   per cell) since that board has been the active development target.
7. **Both Pis run the same verify code** off the same `main` branch
   commits. Differences are hardware, not software.

## Reproduce

Per-Pi data collection:

```bash
cd ~/Documents/Benchy
git pull origin main
./scripts/setup_rpi_ai_hat_plus[_2].sh   # auto-fetches HEFs
./scripts/verify_ai_hat_plus[_2].sh      # full sweep, ~30-40 min wall
git add results/<platform>/hw_verify_<ts>/ && git commit -m '...' && git push
```

Cross-platform showcase regeneration (any machine with both Pis'
bundles available — typically the dev box pulling from main):

```bash
# 1. Per-platform aggregated CSVs (powers the showcase tables)
python3 scripts/aggregate_by_platform.py

# 2. Static PNG charts (embedded in this page)
python3 scripts/generate_showcase_charts.py

# 3. Interactive dashboard + machine-readable aggregated JSON
python3 scripts/regenerate_showcase_dashboard.py
```

All three scripts are repo-relative — they derive paths from their own
location and write under `docs/showcase/`. None of them mutate
`results/` or any other tracked source.

`scripts/aggregate_by_platform.py` is a thin wrapper over
`benchmark.aggregation.ResultsAggregator` that adds **platform** to the
group-by key. The default aggregator (used by `python -m benchmark
report`) groups YOLO by `(model, task)` only, so AI HAT+ and AI HAT+ 2
results for the same model collapse into one row with a meaningless
std — fine for single-platform reports, wrong for cross-platform
comparison.

## Files in this directory

| File | What |
|---|---|
| [`dashboard.html`](showcase/dashboard.html) | Self-contained interactive dashboard with platform / backend / version / task filter chips. Open in any browser. |
| [`aggregated.json`](showcase/aggregated.json) | Full aggregated result set in machine-readable form |
| [`yolo_by_platform.csv`](showcase/yolo_by_platform.csv) | Per-`(platform, model, task)` aggregated YOLO metrics |
| [`llm_by_platform.csv`](showcase/llm_by_platform.csv) | Per-`(platform, backend, prompt)` aggregated LLM metrics |
| [`system_info_by_platform.csv`](showcase/system_info_by_platform.csv) | Hardware + OS info per platform |
| [`charts/`](showcase/charts/) | Static PNG charts embedded above |
