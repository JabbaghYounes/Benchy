# Benchmark Methodology

## Warmup and Measured Runs

Each benchmark executes:
1. **Warmup runs** - Not recorded, allows system to reach steady state
   - 3 warmup runs for 7B+ models
   - 2 warmup runs for 1B/3B models
2. **10 measured runs** - Recorded for statistical analysis

## Deterministic LLM Evaluation

LLM benchmarks use fixed parameters:

**Standard Models (7B+):**
- `temperature: 0.0`
- `seed: 42`
- `top_p: 1.0`
- `top_k: 1`

**Lightweight Models (1B/3B):**
- `temperature: 0.2`
- `seed: 42`
- `top_p: 0.95`
- `max_tokens: 256`
- `streaming: disabled`

This ensures reproducible results across runs.

## Group-Safe Aggregation

Results are aggregated only within the same parameter group:
- 1B models are compared only with other 1B models
- 3B models likewise; the `npu` profile reuses the 3B group
  (`llama3.2:3b` has a published Hailo HEF), so its NPU rows aggregate
  alongside the CPU-side 3B rows after the backend axis split below
- 7B models likewise
- Cross-group comparisons are visualized separately in the dashboard

The benchmark surface is llama-family only (one model per group). See
Issue 7 of `resources/session_issues_2026-04-27.md` for the
consolidation rationale.

**Backend axis (Phase 7).** `aggregate_llm_results` additionally groups by
`backend` so that an Ollama-CPU run of `llama3.2:3b` and a Hailo-10H run of
the same model on the same prompt do not collapse into one aggregated row.
The dashboard exposes a `Backend` filter chip + table column so you can
view "all backends", "Ollama CPU only", "Hailo-10H only", or "legacy
(pre-Phase-7)" data.

## YOLO Task Coverage on Hailo

Phase 3 unblocked all five YOLO tasks on the Hailo NPU through bespoke
post-processors in `benchmark/workloads/yolo/postprocessing.py`:

| Task | Hailo support | Postprocessor |
|---|---|---|
| Detection | shipped pre-Phase 3 | `_process_detection` (axis-aligned NMS) |
| Classification | shipped pre-Phase 3 | `_process_classification` |
| OBB | Phase 3a | `_process_obb` + `_rotated_nms` (Sutherland-Hodgman polygon clipping in pure numpy) |
| Segmentation | Phase 3b | `_process_segmentation` + `_generate_seg_masks` (sigmoid mask blender, threshold 0.5, bbox-cropped at proto resolution) |
| Pose | Phase 3c | `_process_pose` (17-keypoint COCO-Pose decoder, sigmoid visibility) |

v8 and v11 are verified candidates across all five; v26 entries are
whitelisted for symmetry but tagged **experimental** until hardware
verification (the HW-verify runners) clears them.

## YOLO Accuracy Validation

YOLO benchmarks can optionally run validation:
- Uses model's built-in validation method
- Reports mAP, precision, and recall
- Skip with `--skip-validation` flag

## Reproducibility

To ensure reproducible results:

1. **Use consistent power state**
   - Jetson: Set power mode with `nvpmodel`
   - Ensure consistent cooling

2. **Minimize background processes**
   ```bash
   htop
   ```

3. **Use fixed random seeds** - LLM benchmarks use `seed=42`

4. **Record environment**
   - System info is captured automatically
   - Check with `python -m benchmark info`

5. **Multiple runs** - Run benchmarks multiple times, use aggregation to analyze variance

### Reproducibility Verification

The benchmark suite includes built-in reproducibility verification to ensure metrics variance is within acceptable bounds.

**Programmatic Verification:**

```python
from benchmark.workloads.llm.runner import (
    verify_reproducibility,
    verify_parameter_group_reproducibility,
)

# Verify a specific model
result = verify_reproducibility(
    model_name="llama3.2:1b",
    model_size="1B",
    num_runs=3,
    variance_threshold=0.15,  # 15% max coefficient of variation
)
print(f"Passed: {result['passed']}")
print(f"TTFT CV: {result['metrics']['ttft']['cv']:.2%}")
print(f"TPS CV: {result['metrics']['tps']['cv']:.2%}")

# Verify one model per parameter group
result = verify_parameter_group_reproducibility(
    parameter_group="1B",
    variance_threshold=0.15,
)
```

**Acceptance Criteria:**
- Coefficient of Variation (CV) for TTFT and TPS must be <= 15%
- CV = standard deviation / mean
- Lower CV indicates more consistent results

**Factors Affecting Reproducibility:**
- Thermal throttling (allow cooldown between runs)
- Background processes (minimize system load)
- Power state (use consistent power mode)
- Model warm-up (use adequate warmup runs)
