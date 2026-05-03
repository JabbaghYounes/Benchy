# Benchy Cross-Platform Showcase

> Aggregated results across **2 platforms × 6+ verify runs** of the Benchy
> hardware-verification suite, run on the actual edge AI hardware (most
> recent runs include the post-`hef→main` migration verify on both Pis,
> which exercised the new `hefs-v1` GitHub-Release HEF-fetch flow
> end-to-end). Source bundles in
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
| Verify-suite count | 13 steps | 13 steps |

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
| `yolov8n` | detection | **100.75 ± 0.75 FPS** | 14.91 ± 0.09 FPS | **6.8×** |
| `yolov8s-pose` | pose | **87.33 ± 1.64 FPS** | _(no h10h HEF — see gaps)_ | — |
| `yolov8n-obb` | obb | **71.67 ± 0.85 FPS** | 16.09 ± 0.04 FPS | **4.5×** |
| `yolov8n-seg` | segmentation | **62.84 ± 1.16 FPS** | 12.30 FPS¹ | **5.1×** |
| `yolo26n-obb` | obb | **60.10 ± 1.61 FPS** | 18.31 ± 0.10 FPS | **3.3×** |
| `yolo11n-obb` | obb | **58.02 ± 1.16 FPS** | 15.93 ± 0.08 FPS | **3.6×** |
| `yolo11n-pose` | pose | **54.49 ± 0.29 FPS** | 15.40 ± 0.21 FPS | **3.5×** |
| `yolo26n-pose` | pose | **43.04 ± 0.65 FPS** | 16.39 ± 0.07 FPS | **2.6×** |

¹ `v8 seg n hailo10h` has only one verify run so far — the HEF was
added in `hefs-v1` (commit `4ca0efb`) and only the post-merge run on
2026-05-03 picked it up. Std will tighten with subsequent verify runs.

AI HAT+ 2 also has segmentation models AI HAT+ doesn't yet (different
verify HEF coverage):

| Model | Task | AI HAT+ 2 throughput |
|---|---|---:|
| `yolo11n-seg` | segmentation | 12.14 ± 0.07 FPS |
| `yolo26n-seg` | segmentation | 12.35 ± 0.10 FPS |

### Latency (lower is better)

![YOLO latency comparison](showcase/charts/yolo_latency_comparison.png)

| Model | AI HAT+ latency | AI HAT+ 2 latency |
|---|---:|---:|
| `yolov8n` det | **9.93 ms** | 67.06 ms |
| `yolov8s-pose` | **11.45 ms** | _(no h10h HEF)_ |
| `yolov8n-obb` | **13.95 ms** | 62.14 ms |
| `yolov8n-seg` | **15.92 ms** | 81.31 ms |
| `yolo26n-obb` | **16.65 ms** | 54.61 ms |
| `yolo11n-obb` | **17.24 ms** | 62.77 ms |
| `yolo11n-pose` | **18.35 ms** | 64.96 ms |
| `yolo26n-pose` | **23.24 ms** | 61.00 ms |

For real-time vision pipelines (>30 FPS), the AI HAT+ comfortably
clears that bar across every standard model. The AI HAT+ 2 is in the
12-18 FPS range — fine for periodic inference (drone snapshots,
surveillance triggers) but not for high-rate video.

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
| **Hailo-10H NPU** (`hailo-ollama` GenAI) | **10.08 – 10.52 TPS** | **1.45× – 1.55×** |
| Pi 5 CPU (`ollama`) | 6.81 – 6.98 TPS | baseline |

Per-prompt breakdown (5 drone prompt categories), aggregated across
all post-2026-05-02 verify runs:

| Prompt category | NPU TPS | CPU TPS | Speedup |
|---|---:|---:|---:|
| `scene_description` | 10.52 ± 0.03 | 6.98 ± 0.10 | 1.51× |
| `target_identification` | 10.15 ± 0.09 | 6.81 ± 0.17 | 1.49× |
| `mission_preflight` | 10.08 ± 0.04 | 6.84 ± 0.10 | 1.47× |
| `telemetry_interpretation` | 10.09 ± 0.04 | 6.97 ± 0.07 | 1.45× |
| `hazard_reasoning` | 10.43 ± 0.17 | 6.82 ± 0.10 | 1.53× |

### TTFT (prefill latency) — essentially identical

![LLM TTFT comparison](showcase/charts/llm_ttft.png)

Both backends share the same prefill workload, so first-token latency is
near-identical (~454 ms). The NPU's advantage is **decode** (the
per-token cost after the first one) — which is where the 1.49× average
speedup lives, and where it matters for long-form generation.

## Methodology

- **Test rig**: each Pi runs the same `verify_ai_hat_plus*.sh` script —
  13 steps (vision sweep + LLM-on-CPU + LLM-on-NPU) — multiple times
  across multiple days. All bundles live under `results/<platform>/`.
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
5. **HEF coverage gaps mean some cells are blank** — `v8 seg n` for
   `hailo10h` was missing from the verify until release `hefs-v1`;
   `v8 pose s` for `hailo10h` is still missing (deferred to `hefs-v2`).
   Re-running verify after each release batch fills more cells.
6. **Both Pis run the same verify code** off the same `main` branch
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
