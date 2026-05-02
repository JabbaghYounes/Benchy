# Session Notes — 2026-05-02 — LLM drone profile unworkable on Hailo-8 Pi

The 2026-05-01 verify run on the Pi 5 + AI HAT+ (Hailo-8, HailoRT 4.23,
hostname `raspberrypi`) had to be cancelled at step 13 of 13 because the
LLM CPU step (`llm-cpu-llama2:7b drone prompts`) projected to take
**~26 hours of wall time** on its own. The 12 prior YOLO steps and the
LLM-on-NPU stub completed normally; only step 13 was pulled. This note
records the timing data so future bring-up sessions can short-circuit
the same wait, and proposes the methodology change.

## What we observed

Verify bundle: `results/hw_verify_20260501_221138/`. Step 13 invocation
was the verify script's hardcoded `python -m benchmark run llm
--profile drone`, which runs `llama2:7b` against the five `DRONE_PROMPTS`
in `benchmark/workloads/llm/runner.py:219`. The Ollama server
(`ollama serve` on `:11434`) ran throughout — Pi 5 CPU only, no NPU
involved (Hailo-8 has no on-board SDRAM and `hailo-ollama` is
Hailo-10H only).

Per-request timing observed in `journalctl -u ollama` covering 2026-05-01
22:15 → 2026-05-02 09:00:

| Prompt | `expected_tokens` | Per-request median | Per-request range | Likely cause |
|---|---|---|---|---|
| `scene_description` | 120 | ~17 min | 13-22 min | Model hits a stop token before `max_tokens=256` |
| `target_identification` | 100 | **~40 min** | 40m05s-47m02s | Generation hits `max_tokens=256`; hard cap dominates |

`max_tokens: 256` is the YAML default in `configs/llm_benchmark.yaml`.
Per-request wall time on prompts that fill it out is **~40 minutes** at
roughly 6.4 tokens/min on llama2:7b q4 on a Pi 5 CPU (4 cores pegged at
398 % CPU, 5.9 GB RSS — the runner is healthy, the work is just
genuinely this slow).

## Why the projected total is ~26 hours

A single prompt's measurement burst is **3 warmup + 10 measured = 13
runs**. With ~40 min per run on a `max_tokens`-bound prompt:

```
13 runs × 40 min ≈ 520 min ≈ 8h 40m per prompt
```

`DRONE_PROMPTS` has 5 entries; conservatively assume four of them fill
`max_tokens` (only `scene_description` ran short of it):

```
1 short prompt  : 13 × 17 min ≈ 3h 40m   (scene_description, observed)
4 long prompts  : 4 × 8h 40m  ≈ 34h 40m  (worst-case projection)
prewarm + I/O  ≈ 5 min
                                 ≈ 38 hours total
```

Even the optimistic case — every prompt as short as `scene_description`
— is 5 × 3h 40m ≈ **18 hours**. The realistic mid-point sits at
**~24-28 hours**, dominating the rest of the verify sweep by an order
of magnitude.

## Why we are not skipping LLM on this board

Skipping LLM on the AI HAT+ Pi would break the cross-platform dashboard
contract. The LLM panel is the row-for-row comparison between the two
RPi boards (`results/hw_verify_<ts>/report/index.html`); without a CPU
LLM result on the Hailo-8 Pi the dashboard can't:

- show the CPU baseline that the AI HAT+ 2 Pi's `npu` row is being
  compared against (the only reason `verify_ai_hat_plus_2.sh` runs
  `--profile compare` for its CPU step is so the two boards' CPU rows
  are at the same model size);
- demonstrate the deliberate `[unsupported-on-this-hw]` stub from step
  12 alongside an actual CPU result, which is the whole point of the
  stub (`benchmark/cli.py:_build_unsupported_npu_stubs`);
- give a real-world Pi 5 CPU LLM datapoint for the README's edge-AI
  positioning (covered today by step 13 only).

So: the LLM step stays in the sweep. What we change is *which* LLM
profile the AI HAT+ verify script runs.

## Proposed change

`scripts/verify_ai_hat_plus.sh` currently hardcodes `--profile drone`
(llama2:7b). For the AI HAT+ board this is impractical because:

1. llama2:7b is the wrong model size for cross-platform comparison —
   the AI HAT+ 2 Pi runs `npu` (`llama3.2:1b`) on the NPU, so its CPU
   comparison row also needs to be at 1B (`compare` profile,
   `llama3.2:1b`). With the drone-profile 7B model on this Pi, the two
   boards' "CPU LLM" rows are not at the same model size and aren't
   directly comparable.
2. ~26-hour runtime makes the verify script unusable as a routine
   smoke check. Last time (2026-05-01) we cancelled at step 13 for the
   same reason — see the bookmark in
   `~/.claude/projects/-home-snpi-Documents-Benchy/memory/wip_infermodel_broken_on_4_23.md`.

**Switch step 13 of `verify_ai_hat_plus.sh` to `--profile compare`**
(matching `verify_ai_hat_plus_2.sh`). Estimated runtime: ~45 minutes.
The cross-platform dashboard then has same-size CPU LLM rows on both
boards plus the `npu` stub on this Pi for full row-count parity.

The drone-profile run remains available for anyone who explicitly
wants the 7B CPU number on the Pi 5 — it's just no longer baked into
the verify sweep.

## Status of the cancelled bundle

`results/hw_verify_20260501_221138/` contains:

- 12 valid YOLO step results (the InferModel + numpy fixes from
  commits `b0174b0` and `17844d3` were exercised end-to-end here).
- Step 12 LLM-NPU stub: emits the documented `[unsupported-on-this-hw]`
  row, expected behaviour on Hailo-8.
- Step 13 LLM-CPU drone: partial — prompt 1 (`scene_description`) plus
  10 of 13 runs of prompt 2 (`target_identification`) completed. Bench
  JSON likely truncated; treat as discarded.

Remediation: re-run just the LLM step on this Pi with
`python -m benchmark run llm --profile compare --output
results/hw_verify_20260501_221138/llm_compare/` (~45 min) to fill in a
clean LLM row, then regenerate the bundle's dashboard.
