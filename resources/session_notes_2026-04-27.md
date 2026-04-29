# Session Notes — 2026-04-27

Pre-flight audit of `./scripts/verify_ai_hat_plus_2.sh` on the local Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H), plus six narrow improvements to `CLAUDE.md`.

## Hardware identification

- **Board**: Raspberry Pi 5 Model B Rev 1.0 (hostname `wiregaurd`)
- **Accelerator**: Hailo-10H confirmed via PCIe device ID `45c4` (`lspci -d 1e60:` → `0001:01:00.0 Co-processor: Hailo Technologies Ltd. Device 45c4 (rev 01)`)
- **OS**: Debian GNU/Linux 12 (bookworm), kernel `6.12.75+rpt-rpi-2712`
- `python -m benchmark info` reports `platform: rpi_ai_hat_plus_2`. Detection succeeded via the PCIe-ID fallback path in `benchmark/metrics/collectors.py:detect_platform()` (line 68: `if "45c4" in result.stdout: return Platform.RPI_AI_HAT_PLUS_2`).

## Findings

### Blocker — wrong HailoRT major version installed

| Item | Observed | Required for Hailo-10H |
|---|---|---|
| `hailort` package | 4.20.0-1 | 5.x (5.2.0 bundled at `resources/hailo-10H/hailort_5.2.0_arm64.deb`) |
| `hailo-dkms` | 4.20.0-1 | 5.x driver |
| `/dev/hailo*` node | absent | `/dev/hailo0` |
| `hailortcli scan` | `Hailo devices not found` | identifies the chip |
| `hailortcli fw-control identify` | rc=0 but empty body | firmware metadata |
| `python -m benchmark info` → `accelerator` | `None` | populated |

The Hailo-10H is on the PCIe bus but unreachable because the HailoRT 4.x driver does not bind to device ID `45c4`. The first YOLO step of the verify sweep will fail until 5.x is installed.

### Other prerequisites not met

1. `hailort.log` at the repo root is owned by `root:root` and read-only for `vpn` — blocks `hailortcli` from writing its log file (warning seen on every invocation: `Cannot create log file hailort.log! Please check the file ./hailort.log write permissions.`).
2. Ollama is up on `:11434` but `models: []` — the CPU comparison step (`llm-cpu-llama2:7b (drone prompts)`) needs only the bare `llama2:7b` tag pulled (~3.8 GB). The drone profile does **not** sweep quants (`configs/llm_benchmark.yaml` defines `quants` only on the `default` profile, and `cli.py:_expand_quant_sweep` returns the model list unchanged when `quants` is empty). Chat-quant variants are not required for the verify sweep.
3. `hailo-ollama` is not installed and `:8000` is unreachable. Without it the LLM-on-NPU step (`llm-npu-llama3.2:3b`) will skip via `hw_skip` (graceful, but the dashboard's NPU axis goes empty — which defeats the cross-backend comparison this board exists for).
4. `~/.cache/benchy/hailo/` does not exist — first run will compile one `.hef` per YOLO model (5–30 min each per `docs/hailo.md`; nine YOLO steps in the sweep, so budget hours).
5. `.cache/hailo-apps/setup_env.sh` does not exist — `setup_rpi_ai_hat_plus_2.sh --with-genai` has not been run.

## Pre-flight checklist (six steps)

Steps 1–4 are blocking. Step 5 is optional but enables LLM-on-NPU. Step 6 runs the sweep.

```bash
# 1. Install HailoRT 5.x for the Hailo-10H. Reboot afterwards if DKMS prompts you.
sudo ./scripts/setup_rpi_ai_hat_plus_2.sh

# 2. Remove the root-owned stale log so hailortcli can write its own.
sudo rm hailort.log

# 3. Activate the project venv (PEP 668 requires this on Bookworm).
source venv/bin/activate

# 4. [DONE 2026-04-27] Pull the three CPU LLM models. The project's
#    llama-only policy is one model per size group (1B / 3B / 7B), all
#    from the llama family (see configs/llm_benchmark.yaml). The verify
#    sweep's CPU step uses --profile drone (llama2:7b only, ~3.8 GB); the
#    1B and 3B models are used by --profile full and by the npu profile
#    (3B) when LLM-on-NPU is enabled in step 5. Total disk: ~7.1 GB.
ollama pull llama3.2:1b   # ~1.3 GB ✓ pulled
ollama pull llama3.2:3b   # ~2.0 GB ✓ pulled
ollama pull llama2:7b     # ~3.8 GB ✓ pulled (interrupted by network drop mid-pull;
                          #          resumed cleanly via second `ollama pull` —
                          #          daemon kept the partial blob server-side
                          #          even though `~/.ollama/models/blobs/` is
                          #          empty for the user, since Ollama runs as a
                          #          system service under /usr/share/ollama/)

# 5. (Optional) Enable LLM-on-NPU. Requires the EULA-gated
#    hailo_gen_ai_model_zoo_<ver>_arm64.deb at the repo root first
#    (download from the Hailo Developer Zone). The npu profile pulls
#    llama3.2:3b — the only llama with a prebuilt HEF in the GenAI
#    Model Zoo.
sudo ./scripts/setup_rpi_ai_hat_plus_2.sh --with-genai
source .cache/hailo-apps/setup_env.sh
hailo-ollama &
curl -sS http://localhost:8000/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","stream":true}'

# 6. Run the sweep.
./scripts/verify_ai_hat_plus_2.sh
```

Output lands in `results/hw_verify_<timestamp>/` (per-step `.log`s + `bench_*.json` + an auto-generated dashboard under `report/`). Exit code is 0 when all blocking steps pass; v26-{obb, seg, pose} are tagged `[experimental]` and don't gate exit. `hw_skip` is the expected (non-failing) outcome for the LLM-on-NPU step if `hailo-ollama` is not running.

## Mid-session pivot — storage exhaustion + llama-only policy

While following step 4 of the original checklist, the SD card filled to 100% during Ollama pulls (the `default` profile's chat-quant sweep — `llama2:7b-chat-{q4_K_M, q5_K_M, q8_0}` — is ~16 GB combined and the bash tool itself started failing with no output, the symptom of a fully-saturated `/`). Recovery: cleared `~/.ollama/models/blobs/` orphans, ran `apt-get clean` and `journalctl --vacuum-size=100M`, and removed all pulled tags. Disk dropped to 51% used (14 GB free).

In response we adopted a **llama-only policy** for the LLM workload, with one model per size group:

| Group | Model | Size | Used by |
|---|---|---|---|
| 1B | `llama3.2:1b` | ~1.3 GB | `full` profile |
| 3B | `llama3.2:3b` | ~2.0 GB | `full` profile + `npu` profile (only llama with a prebuilt HEF in the Hailo GenAI Model Zoo) |
| 7B | `llama2:7b` | ~3.8 GB | `default` / `drone` / `full` profiles |

Other model families (qwen, mistral, gemma, dolphin, olmo) and other size groups (1.5B, 8B, 9B) were dropped from `configs/llm_benchmark.yaml`. The `default` profile's quant sweep (`q4_K_M / q5_K_M / q8_0`) was also removed — it was the disk-bomb that triggered the exhaustion, and on an SD-card-backed Pi it is not reproducible anyway. To restore the quant sweep later, re-add `quants:` and `quant_tag_template:` to the `default` profile in `configs/llm_benchmark.yaml` (commit history preserves the original).

## Changes this session

### `CLAUDE.md`

Six narrow improvements applied early in the session:

1. **Python version** — added `Python >= 3.10 is required (setup.py:python_requires)` to the Development Commands intro.
2. **Test coverage list** — expanded the test-coverage sentence to include LLM quant tag templating, NPU profile gating, the LLM-side NPU collector, and YOLO model-name parsing (test files: `test_llm_quant.py`, `test_llm_npu_profile.py`, `test_hailo_llm_metrics.py`, `test_yolo_model_info.py`).
3. **Conversion sub-modules** — named the files inside `benchmark/workloads/yolo/conversion/`: `pipeline.py` orchestrates the stages (`onnx_export.py` → `har_generator.py` → `hef_compiler.py`); `cache.py` manages `~/.cache/benchy/hailo/`; `validation.py` sanity-checks artefacts.
4. **Aggregation CSV writer** — distinguished `benchmark/aggregation/csv_writer.py` (aggregated CSV) from `benchmark/results/writers.py` (per-run JSON/CSV) in the aggregator bullet.
5. **Jetson setup script** — added a one-liner under Hardware-Specific Notes about `scripts/setup_jetson_orin_nano.sh` next to the Pi setup-script entry, so all three platforms are mentioned symmetrically.
6. **Untracked `*.log` files** — noted that setup scripts emit `*.log` to the repo root (`hailort.log`, `setup_rpi_ai_hat_plus_2.log`), they are not in `.gitignore`, and they are runtime artefacts safe to delete.

A late-session edit also rewrote the LLM-profile summary in the Running Benchmarks section to reflect the new llama-only policy (1B / 3B / 7B llamas across `default` / `full` / `drone` / `npu`).

### `configs/llm_benchmark.yaml`

- `default` profile: dropped `quants` + `quant_tag_template`; bare `llama2:7b` only.
- `full` profile: `model_groups` changed from `["7B", "8B", "9B"]` → `["1B", "3B", "7B"]`.
- `npu` profile: model changed from `qwen2:1.5b` → `llama3.2:3b`; `model_groups: ["3B"]`.
- `models:` map: `1.5B / 7B / 8B / 9B` groups (with qwen / mistral / gemma / dolphin / olmo entries) replaced by `1B / 3B / 7B` groups, each with one llama tag.
- `drone` profile unchanged (was already `llama2:7b`).

### `scripts/verify_ai_hat_plus_2.sh`

- LLM-on-NPU step label renamed from `llm-npu-qwen2:1.5b` → `llm-npu-llama3.2:3b` (both the run-step and the skip-step labels). Affects `hw_summary` output and per-step log filenames.

### `README.md`

- "Local LLM Inference" bullet updated to reflect the llama-only policy.
- "Benchmark Profiles" table rows for `default`, `full`, and `npu` updated to match the new config.

### `docs/`

- `docs/workloads.md` — model-groups table reduced to 1B/3B/7B llama-only; memory requirements table trimmed to match; quantization-sweep example reframed as opt-in (the shipped default no longer enables it).
- `docs/hailo.md` — npu-pull curl example updated to `llama3.2:3b`; "starts smallest" wording replaced with the llama-only justification; prebuilt-HEFs table annotated with which entries the project actually exercises.
- `docs/methodology.md` — Group-Safe Aggregation section rewritten around the llama-only group set; backend-axis example now uses `llama3.2:3b` (the only group that exists on both CPU and NPU).
- `docs/output.md` — config-example yaml updated: full profile drops 8B/9B; default profile shows the quant-sweep fields commented-out with an explainer.
- `docs/troubleshooting.md` — three references to `qwen2:1.5b` / `llm-npu-qwen2:1.5b` updated to the llama3.2:3b equivalents.

### `tests/test_llm_quant.py`

- Two tests inverted to match the new policy:
  - `test_default_profile_declares_quant_sweep` → `test_default_profile_does_not_declare_quant_sweep` (now asserts the default profile does NOT declare `quants` / `quant_tag_template`).
  - `test_default_profile_expands_to_three_tags` → `test_default_profile_expands_to_single_tag` (asserts `_expand_quant_sweep` on the default profile returns just `["llama2:7b"]`).
- Other tests in the file (which use arbitrary model names like `llama3.1:8b`, `mistral:7b` as test data for `_expand_quant_sweep`) are unchanged — they verify the function's logic, not policy.

Full pytest suite (`pytest tests/ -q`) is green: **196 passed**.

## Storage budget (SD-card-backed Pi)

| Item | Approx size | Required? |
|---|---|---|
| `llama3.2:1b` (Ollama) | ~1.3 GB | yes — `full` profile |
| `llama3.2:3b` (Ollama) | ~2.0 GB | yes — `full` profile + `npu` HEF target |
| `llama2:7b` (Ollama) | ~3.8 GB | yes — `default` / `drone` / `full` profiles |
| HailoRT 5.x `.deb` install | <500 MB | yes |
| `hailo-apps` + `hailo-ollama` (`--with-genai`) | ~6 GB | optional (Step 5) |
| Hailo Model Zoo GenAI HEF (`llama3.2:3b`) | ~1–3 GB | optional (Step 5, only if pulled) |
| `~/.cache/benchy/hailo/` (per-model `.hef` cache) | ~50–500 MB per model × 9 in the sweep | yes (built on first run) |

Recovery if a pull fails mid-stream:

- `ollama list` then `ollama rm <tag>` for any partial Ollama tags.
- Wipe orphaned blobs (aborted pulls leave `~/.ollama/models/blobs/sha256-*` files that `ollama list` doesn't show): `rm -rf ~/.ollama/models/blobs/sha256-*` and `rm -rf ~/.ollama/models/manifests`.
- `sudo apt-get clean && sudo journalctl --vacuum-size=100M` for system-side reclamation.
- Relocate Ollama's model store off the SD card by setting `OLLAMA_MODELS=/path/to/external` (in `systemd` drop-in or env before `ollama serve`) before pulling.
- Defer Step 5 (`--with-genai`) — the verify script's `hw_skip` falls back gracefully when `hailo-ollama` is not on `:8000`, and the rest of the sweep still runs.

## Notes for future sessions

- Subsequent runs of `verify_ai_hat_plus_2.sh` hit `~/.cache/benchy/hailo/` and skip recompilation — the multi-hour first-run cost is paid once per `(model, input_resolution)` variant.
- The platform detection's PCIe-ID fallback (`benchmark/metrics/collectors.py:68`) lets `python -m benchmark info` report `rpi_ai_hat_plus_2` even when HailoRT can't open the device. Treat that as a hardware-presence signal, not a "ready to run" signal — confirm `/dev/hailo0` exists and `hailortcli scan` finds the chip before launching the sweep.
- If Step 5 is skipped, the LLM-on-NPU row in the dashboard will be absent (not zero-valued). The zero-valued `unsupported-on-this-hardware` stub is only emitted for the `npu` profile on **Hailo-8/8L** boards (`cli.py:_build_unsupported_npu_stubs()`), not when `hailo-ollama` is simply not running on a Hailo-10H board.
- The `drone` profile is intentionally a single-tag run (`llama2:7b`, no quant sweep). For the 1B / 3B llama coverage on the CPU axis, run `python -m benchmark run llm --profile full` separately — but that is not part of the HW-verify sweep.
- The bash tool inside this session went fully unresponsive (all commands exit-1, including builtins like `:` and `true`) when `/` hit 100% — useful diagnostic if it recurs. Read tool kept working since it doesn't write.
- Ollama pull is robust to network drops: when DNS or connectivity vanishes mid-pull, the client errors out (`dial tcp: lookup … connection refused`) but the **daemon retains the partial blob** under `/usr/share/ollama/.ollama/models/blobs/`. Re-running the same `ollama pull <tag>` resumes from the byte offset, even though the user's `~/.ollama/models/blobs/` directory does not exist (Ollama runs as a system service on this host). Confirmed during the 7B pull: dropped at ~7.5 MB / 3.8 GB, resumed and reached 47 % within minutes at restored line speed (~6 MB/s).

## Pre-flight progress as of end of 2026-04-27

| Step | Status |
|---|---|
| 1. HailoRT 5.x install (`setup_rpi_ai_hat_plus_2.sh`) | not started — still on 4.20.0; the Hailo-10H is unreachable until this runs |
| 2. `sudo rm hailort.log` | not started |
| 3. `source venv/bin/activate` | per-terminal, no persistent state |
| 4. Pull llamas (1B / 3B / 7B) | **done** — all three present in `ollama list` |
| 5. (Optional) `--with-genai` + `hailo-ollama` + curl-pull `llama3.2:3b` HEF | not started |
| 6. `./scripts/verify_ai_hat_plus_2.sh` | not started |

Disk state at end of session: 20 GB used / 7.1 GB free on `/dev/mmcblk0p2` (74 %). The remaining 7.1 GB is enough for HailoRT 5.x install + the per-model `.hef` cache built during the verify sweep, but step 5 (GenAI / `hailo-ollama` + HEF pull, ~7–9 GB combined) would push the card uncomfortably close to full. Decide step 5 vs not before kicking off the verify sweep.

Desktop copy of the remaining commands: `~/Desktop/benchy_preflight_commands.txt` (generated 2026-04-27).
