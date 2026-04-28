# Troubleshooting

## Ollama Not Running

```bash
# Start Ollama server
ollama serve

# Check status
curl http://localhost:11434/api/version
```

## Model Not Found

```bash
# Pull required model
ollama pull llama2:7b
```

## CUDA/GPU Issues (Jetson)

```bash
# Check GPU status
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Hailo NPU Issues (Raspberry Pi)

```bash
# Check Hailo device status
hailortcli fw-control identify

# List available backends
python -m benchmark backends

# List Hailo-supported models
python -m benchmark list-models --backend hailo

# Check HailoRT version
hailortcli --version

# View Hailo device info
hailortcli scan
```

**Common Hailo Issues:**

| Issue | Solution |
|-------|----------|
| "Hailo device not found" | Check PCIe connection, run `lspci \| grep Hailo` |
| "HailoRT not installed" | Install HailoRT SDK from Hailo Developer Zone |
| "Model compilation failed" | Check disk space, ensure Dataflow Compiler is installed |
| "Unsupported task" for pose | Pose is the only YOLO task still blocked at the conversion gate. Detection/classification/OBB/segmentation all work post-Phase-3. |
| OBB / seg / pose fails on v26 specifically | v26 entries are tagged experimental in the whitelist; if conversion fails, drop `YOLOTask.{OBB,SEGMENTATION,POSE}` from `HAILO_SUPPORTED_TASKS["v26"]` in `benchmark/workloads/yolo/execution.py` and update the `tests/test_drone_profile.py::test_*_in_hailo_supported_tasks[v26]` test. |
| "CPU fallback error" | This is expected — Hailo backend requires NPU hardware |
| Quantised mAP looks suspicious | Calibration defaults bumped from 8-image samples to full DOTAv1 / coco-pose in Polish 4. First-run download is heavier (~10 GB / ~20 GB). Override via `CalibrationConfig.dataset_path` if you have a pre-curated subset. |

## HailoRT GenAI / hailo-ollama (LLM-on-NPU, AI HAT+ 2 only)

```bash
# Confirm the GenAI server is reachable
curl -sS http://localhost:8000/api/tags
```

| Issue | Solution |
|-------|----------|
| "profile 'npu' requires Platform.RPI_AI_HAT_PLUS_2" | The `npu` LLM profile gates on the Hailo-10H (only board that can host LLMs on the NPU). Run on AI HAT+ 2 or use `--profile drone` for the CPU-side path. |
| `:8000` not responding | Source `setup_env.sh` and start the server: `source .cache/hailo-apps/setup_env.sh && hailo-ollama &` |
| Model not in `/api/tags` | Pull the llama HEF: `curl -sS http://localhost:8000/api/pull -H 'Content-Type: application/json' -d '{"model":"llama3.2:3b","stream":true}'` |
| Model Zoo GenAI `.deb` not installed | Download `hailo_gen_ai_model_zoo_<ver>_arm64.deb` from Hailo Developer Zone (EULA-gated), drop it at the repo root, then re-run `sudo ./scripts/setup_rpi_ai_hat_plus_2.sh --with-genai` |

## Hardware verification runner failures

The smart HW-verify scripts (`scripts/verify_ai_hat_plus.sh`,
`scripts/verify_ai_hat_plus_2.sh`) use continue-on-failure: a single
broken model doesn't abort the rest of the sweep. Each step writes a
`.log` under `results/hw_verify_<timestamp>/logs/` — that's the first
place to look when the summary reports a FAIL.

| Issue | Solution |
|-------|----------|
| "Platform mismatch" at preflight | Run the matching script for your board (`verify_ai_hat_plus.sh` on Hailo-8/8L, `verify_ai_hat_plus_2.sh` on Hailo-10H), or override `python -m benchmark info` detection with `--platform`. |
| Validator reports `backend=pytorch, expected hailo` | The runner fell back to CPU silently (rare — this is what `enforce_no_fallback` in execution.py is supposed to prevent). Re-run with `python -m benchmark backends` to confirm Hailo is detected. |
| `llm-npu-llama3.2:3b` reports SKIP rather than FAIL | hailo-ollama wasn't reachable on `:8000` at preflight. See the GenAI section above. |
| All v26 steps FAIL but v8/v11 PASS | Expected during the experimental-v26 phase. The summary tags those failures as experimental and the script still exits 0. To park v26, drop the failing tasks from `HAILO_SUPPORTED_TASKS["v26"]`. |
| All YOLO steps PASS-but-empty (`yolo_results: []`); validator reports "the runner produced no rows" | The Hailo conversion pipeline failed at `.pt → .onnx`. Almost always a missing `onnx` / `onnxruntime` in the venv. The verify script's `hw_ensure_python_deps` step now self-heals this on the next run; if you've patched it out, do `pip install onnx onnxruntime` in the venv. See Issue 8 in `resources/session_issues_2026-04-27.md`. |
| `llm-cpu-llama2:7b` step times out with `Read timed out (read timeout=300)` | Old runner timeout. Pull the latest `benchmark/workloads/llm/runner.py` — both `generate` and `generate_stream` use 600s now. The 300s window was too tight for an Ollama cold-load of 7B weights from SD storage on a Pi 5. See Issue 9 in the same log. |
| `cmd_report` crashes with `dict contains fields not in fieldnames` | `LLM_AGGREGATED_COLUMNS` in `benchmark/aggregation/csv_writer.py` has drifted from `LLMAggregatedMetrics.to_dict()`. The fix is to add the missing fields to the columns list — they must stay in lockstep. See Issue 10 in the same log. |
| YOLO step fails with "ONNX file generated but Hailo SDK not available" | The Pi cannot compile HEFs (Hailo Dataflow Compiler is x86_64-only). Drop a prebuilt HEF named per the convention into `resources/hefs/` (see `resources/hefs/NAMING.txt`) — the backend looks there and at `/usr/share/hailo-models/` (the rpicam-apps-hailo-postprocess package's curated subset) before attempting compilation. See Issue 11 in `resources/session_issues_2026-04-27.md` and "Prebuilt HEF source layer" in `docs/hailo.md`. |
| `llm-cpu-llama2:7b` still times out with `Read timed out (read timeout=600)` | First measurement hit cold-load. The runner now pre-warms with a long timeout (`prewarm_timeout_seconds`, default 1800s) before the timed loop — pull the latest `benchmark/workloads/llm/runner.py`. To tune further, raise `benchmark.prewarm_timeout_seconds` (or `benchmark.http_timeout_seconds` for the timed loop) in `configs/llm_benchmark.yaml`. See Issue 12 in the same log. |

## Insufficient Memory

- Use smaller model sizes (n, s)
- Run workloads separately instead of `all`
- Close other applications
- For LLM-on-NPU runs, the `npu` profile uses `llama3.2:3b` (~2 GB HEF) — well within the AI HAT+ 2's 8 GB onboard SDRAM, so memory should rarely be the bottleneck
