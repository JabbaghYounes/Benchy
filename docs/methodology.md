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
- Cross-group comparisons are visualized separately in the dashboard
- MoE and code-specialized models are clearly labeled

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
