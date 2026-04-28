# Session Issues — 2026-04-27

Issues raised during a benchmarking session on the Raspberry Pi 5 + AI HAT+
(`hostname: raspberrypi`), root cause analysis, and applied fixes. Recorded
so future sessions on this board can recognise the same failure modes.

## Hardware identified

| Field | Value |
|---|---|
| Pi model | Raspberry Pi 5 Model B Rev 1.1 |
| Accelerator board | AI HAT+ (not AI HAT+ 2) |
| Hailo chip | **Hailo-8 (26 TOPS)** — confirmed by `hailortcli fw-control identify` after the fix below. `Device Architecture: HAILO8`, `Board Name: Hailo-8`, FW 4.20.0. PCI ID `1e60:2864`. |
| Kernel | `6.12.75+rpt-rpi-2712` |
| Detector says | `Platform.RPI_AI_HAT_PLUS` |

## Issue 1 — `HAILO_DRIVER_INVALID_IOCTL(86)` on every `hailortcli` call

### Symptom

```
$ hailortcli fw-control identify
HailoRT warning: Cannot create log file hailort.log! ...
[HailoRT] [error] Ioctl HAILO_QUERY_DRIVER_INFO failed due to inappropriate ioctl for device (can happen due to version mismatch or unsupported feature)
[HailoRT] [error] Failed to query driver info with HAILO_DRIVER_INVALID_IOCTL(86)
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_INVALID_IOCTL(86)
[HailoRT CLI] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_INVALID_IOCTL(86)
```

`dmesg` corroborates the userspace/kernel mismatch:

```
hailo 0001:01:00.0: Invalid general ioctl code 0x400c6701 (nr: 1)
```

### Root cause — two HailoRT userspace versions installed side-by-side

| Path | Source | Version | Used by |
|---|---|---|---|
| `/usr/bin/hailortcli` | apt `hailort` package | 4.20.0 | (correct, but second on `PATH`) |
| `/lib/libhailort.so.4.20.0` | apt `hailort` package | 4.20.0 | resolved by `/usr/bin/hailortcli` via RPATH |
| `/usr/local/bin/hailortcli` | **manual install** | 5.2.0 | first on `PATH`, so this is what runs |
| `/usr/local/bin/hailort_service` | **manual install** | 5.2.0 | — |
| `/usr/local/lib/libhailort.so.5.2.0` | **manual install** | 5.2.0 | loaded by `/usr/local/bin/hailortcli` |
| `/usr/local/lib/libhailort.so` → `libhailort.so.5.2.0` | **manual install** | 5.2.0 | symlink |
| `/usr/local/include/hailo/` | **manual install** | 5.2.0 headers | — |
| `hailo_pci` kernel module | DKMS, apt `hailo-dkms` | **4.20.0** | (correct for this Hailo-8 board) |
| `python3-hailort` | apt | 4.20.0 | Python `hailo_platform` module |

The HailoRT 5.x userspace at `/usr/local/` is the AI HAT+ 2 (Hailo-10H)
stack — it should not be installed on this AI HAT+ board. Because
`/usr/local/bin` precedes `/usr/bin` on `PATH`, every `hailortcli`
invocation runs the 5.2.0 binary, which speaks the new IOCTL ABI to the
4.20.0 driver and gets rejected. The .deb shipped at
`resources/hailo-10H/hailort_5.2.0_arm64.deb` installs to `/usr/`, not
`/usr/local/`, so this stray install was almost certainly produced by
extracting an upstream HailoRT 5.x tarball (or running a `cmake … && make
install` build) on this machine at some prior point.

### Fix applied

```bash
sudo rm -f \
  /usr/local/bin/hailortcli \
  /usr/local/bin/hailort_service \
  /usr/local/lib/libhailort.so \
  /usr/local/lib/libhailort.so.5.2.0
sudo rm -rf /usr/local/include/hailo
sudo ldconfig
hash -r            # drop bash's PATH cache for hailortcli
```

After that, `which hailortcli` resolves to `/usr/bin/hailortcli` (4.20.0)
and `ldd` resolves `libhailort.so` to `/lib/libhailort.so.4.20.0`,
matching the 4.20.0 driver. `hailortcli fw-control identify` then
succeeds and reports the actual Hailo-8 / 8L variant.

### Don't reintroduce this

- The setup script for this board is `scripts/setup_rpi_ai_hat_plus.sh`,
  which uses the **apt** `hailo-all` metapackage (currently 4.20.0+1).
  Do not run `setup_rpi_ai_hat_plus_2.sh` on this Pi — that script
  targets HailoRT 5.x for the AI HAT+ 2 board, and reinstalling the 5.x
  userspace will reproduce this exact mismatch.
- If you need to evaluate HailoRT 5.x manually, do it inside a chroot or
  on the AI HAT+ 2 Pi, never on top of an apt 4.x install.

## Issue 2 — `Cannot create log file hailort.log!`

### Symptom

```
HailoRT warning: Cannot create log file hailort.log! Please check the file ./hailort.log write permissions.
```

Printed by every `hailortcli` / `hailo_platform` invocation when run from
the repo root or `scripts/` directory.

### Root cause

HailoRT writes its log to `./hailort.log` in the current working
directory. Two pre-existing files exist, both root-owned from a prior
sudo-driven session:

```
-rw-r--r-- 1 root root    0 Feb  5 22:49 /home/snpi/Documents/Benchy/hailort.log
-rw-r--r-- 1 root root 2084 Jan 22 11:26 /home/snpi/Documents/Benchy/scripts/hailort.log
```

The unprivileged `snpi` user can read but not write them, so HailoRT
can't open the log for append and emits the warning. (The benchmark
itself still runs — only logging is blocked.)

### Fix applied

```bash
sudo rm -f /home/snpi/Documents/Benchy/hailort.log /home/snpi/Documents/Benchy/scripts/hailort.log
```

Removing them lets HailoRT recreate them owned by the running user. They
are also covered by `.gitignore` (the repo's untracked-files list shows
`hailort.log` only because it was created before the ignore rule), so
deletion has no source-control impact.

## Issue 3 — `ModuleNotFoundError: No module named 'hailo_platform'` inside the venv

### Symptom

```
$ source venv/bin/activate
$ python -c "import hailo_platform"
ModuleNotFoundError: No module named 'hailo_platform'
```

The same import works under the system Python (`/usr/bin/python3`) — the
problem is venv-local. This blocks `verify_ai_hat_plus.sh`, since the
project's `benchmark.workloads.yolo.backends.hailo.HailoBackend` imports
`hailo_platform` at module load time.

### Root cause

`hailo_platform` is shipped as a Debian package (`python3-hailort` apt)
into `/usr/lib/python3/dist-packages/hailo_platform/`. The project venv
at `/home/snpi/Documents/Benchy/venv` was created without
`--system-site-packages` (`pyvenv.cfg` shows
`include-system-site-packages = false`), so it cannot see anything in
`dist-packages/`.

`scripts/setup_rpi_ai_hat_plus.sh` line 313 *attempts* to bridge this
gap with a symlink:

```bash
ln -sf /usr/lib/python3/dist-packages/hailo_platform \
    "$VENV_DIR/lib/python3.*/site-packages/" 2>/dev/null || true
```

That command is buggy: `ln` does **not** expand globs, so the literal
path `…/lib/python3.*/site-packages/` is passed as the destination and
fails because no such directory exists. The trailing
`2>/dev/null || true` silently masks the failure, so setup reports
success while leaving the symlink absent. Result: a venv that looks
healthy until the first `from hailo_platform import VDevice` blows up.

### Fix applied — two parts

**(a) Symlink the package into the existing venv on this machine** (gets
`verify_ai_hat_plus.sh` unblocked immediately):

```bash
ln -sfn /usr/lib/python3/dist-packages/hailo_platform \
    /home/snpi/Documents/Benchy/venv/lib/python3.11/site-packages/hailo_platform
```

Resolved post-fix: `python -c "import hailo_platform as h;
print(h.__version__)"` → `4.20.0`, `from hailo_platform import VDevice`
→ ok, and `from benchmark.workloads.yolo.backends.hailo import
HailoBackend` → ok.

**(b) Patch `scripts/setup_rpi_ai_hat_plus.sh:311-316`** so a fresh
install on a future Pi doesn't hit the same trap. Diff (mirrors the
path-resolution style already used in
`setup_rpi_ai_hat_plus_2.sh:382-383`):

```diff
     if [[ -d /usr/lib/python3/dist-packages/hailo_platform ]]; then
         info "Linking system Hailo packages to venv..."
-        ln -sf /usr/lib/python3/dist-packages/hailo_platform "$VENV_DIR/lib/python3.*/site-packages/" 2>/dev/null || true
+        local venv_site
+        venv_site=$("$VENV_DIR/bin/python" -c "import site; print(site.getsitepackages()[0])")
+        ln -sfn /usr/lib/python3/dist-packages/hailo_platform "$venv_site/hailo_platform"
     fi
```

Three things changed:
1. Resolve the venv site-packages path via the venv's own Python
   instead of relying on `ln` to expand `python3.*` (which it never
   does).
2. `ln -sfn` instead of `ln -sf` — the `-n` flag makes the link replace
   atomically when the destination already exists, avoiding the
   "link-created-inside-existing-dir" footgun.
3. Drop the `2>/dev/null || true`. Under `set -euo pipefail` (in effect
   for this script), a real failure now halts setup loudly, which is
   what we want — silently shipping a broken venv is what hid this for
   so long.

### Don't reintroduce this

- If the venv is ever recreated from scratch on this Pi outside the
  setup script, redo step (a) (or pass `--system-site-packages` to
  `python3 -m venv` and skip the symlink entirely).
- `setup_rpi_ai_hat_plus_2.sh:384-385` resolves the path correctly but
  still uses the `2>/dev/null || true` mask. Not patched here (out of
  scope — that script targets a different Pi), but worth tightening if
  someone touches it.

## Issue 4 — `pytest` missing from the venv (verify step 1 would fail)

### Symptom

After fixing Issues 1-3, an audit of `verify_ai_hat_plus.sh`
prerequisites surfaced:

```
$ ./venv/bin/python -c "import pytest"
ModuleNotFoundError: No module named 'pytest'
```

Step 1 of the verify script is `pytest tests/ -q` (`verify_ai_hat_plus.sh:38`).
With pytest absent from the venv, that step fails before any vision or
LLM workload runs.

### Root cause

`scripts/setup_rpi_ai_hat_plus.sh:326-328` installs the project editable
without dev extras:

```bash
pip install -e "$PROJECT_ROOT"
```

That pulls only `setup.py`'s `install_requires` (psutil, requests,
pyyaml, numpy, ultralytics). The `extras_require['dev']` block —
`pytest`, `pytest-cov`, `black`, `mypy` — is never installed by the
setup script, even though the very next-step recommendation
(`./scripts/verify_ai_hat_plus.sh`) immediately depends on `pytest`.

### Fix applied

```bash
./venv/bin/pip install -e "/home/snpi/Documents/Benchy[dev]"
```

Resolved post-fix: pytest 9.0.3, pytest-cov 7.1.0, black 26.3.1,
mypy 1.20.2 installed. Smoke suite — `./venv/bin/pytest tests/ -q` —
**196 passed in 1.66s**.

### Setup script patch

**Now applied** to all three platform setup scripts so a fresh-install
Pi (or Jetson) clones the repo and runs verify cleanly without manual
intervention:

- `scripts/setup_rpi_ai_hat_plus.sh:install_python_deps` →
  `pip install -e "$PROJECT_ROOT[dev]"`
- `scripts/setup_rpi_ai_hat_plus_2.sh:install_python_deps` → same
- `scripts/setup_jetson_orin_nano.sh:install_python_deps` → same

The cost is small (a few MB for pytest / pytest-cov / black / mypy);
the benefit is that step 1 of `verify_*.sh` works out of the box. A
setup user who genuinely wants runtime-only can still drop `[dev]`
manually, but the default is now verify-ready.

### Don't reintroduce this

- If the venv is ever recreated on this Pi outside the setup script,
  redo `pip install -e ".[dev]"`.

## Issue 5 — `edge_ai_benchmark.egg-info/` was root-owned (blocked pip rebuild)

### Symptom

Applying the Issue 4 fix initially failed during pip's editable
rebuild:

```
error: Cannot update time stamp of directory 'edge_ai_benchmark.egg-info'
ERROR: Failed to build 'file:///home/snpi/Documents/Benchy' when getting requirements to build editable
```

### Root cause

```
drwxr-xr-x 2 root root 4096 Feb  5 22:50 edge_ai_benchmark.egg-info
```

A leftover from a prior `sudo`-driven install (the platform setup
script is invoked with `sudo` per its docstring, line 21:
`sudo ./setup_rpi_ai_hat_plus.sh [--pull-models]`). Pip needs to
re-stamp `egg-info/PKG-INFO` and the directory mtime, but the
unprivileged `snpi` user can't.

This is the same shape as Issue 2 (root-owned `hailort.log`) — sudo
runs leave root-owned artefacts in the project tree that block later
unprivileged operations.

### Fix applied

```bash
sudo chown -R snpi:snpi /home/snpi/Documents/Benchy/edge_ai_benchmark.egg-info
```

After that, the Issue 4 dev install completed cleanly.

### Setup script patch

**Now applied** to all three platform setup scripts. The existing
`chown -R "$actual_user:$actual_user" "$VENV_DIR"` in
`install_python_deps` is now followed by:

```bash
if [[ -d "$PROJECT_ROOT/edge_ai_benchmark.egg-info" ]]; then
    chown -R "$actual_user:$actual_user" "$PROJECT_ROOT/edge_ai_benchmark.egg-info"
fi
```

so the editable install's egg-info directory ends up user-owned,
matching the venv. A fresh-install Pi running setup with sudo no
longer leaves a root-owned project artefact behind to break the next
unprivileged pip operation.

The same shape was added to the Jetson setup script, which previously
had no chown block at all in `install_python_deps`.

### Don't reintroduce this

- Repo-wide hygiene check worth running once after any sudo-touched
  setup: `find /home/snpi/Documents/Benchy -maxdepth 3 -uid 0` — flags
  any root-owned files in the project that will cause similar trouble.
  If the find returns anything, that's a candidate for an additional
  chown line in `install_python_deps`.

## Issue 6 — LLM model groups: docs/config drift, consolidated to 1B / 3B / 7B

### Symptom

While auditing what `verify_ai_hat_plus.sh`'s LLM step would actually
exercise, the active model surface didn't match the documented one.

`docs/workloads.md` documented six groups (1B / 1.5B / 3B / 7B / 8B / 9B)
with specific model names per group. `configs/llm_benchmark.yaml`
defined only four (1.5B / 7B / 8B / 9B) — **the 1B and 3B groups had
no `models:` entries**, so even though `benchmark/workloads/llm/runner.py`
already had complete `LLM_MODELS`, `MODEL_METADATA`, and memory-preflight
support for 1B and 3B, no profile could actually invoke them. The
runner code was ready; the YAML wiring was the missing piece.

The smallest model any built-in profile could exercise was qwen2:1.5b
on the `npu` profile (Hailo-10H prebuilt HEFs only), and that
profile writes the unsupported-on-this-hw stub on this AI HAT+ board.
On any CPU profile (default / full / drone / drone_full) the
floor was **7B**.

### Root cause

Three layers were out of sync:

1. **YAML `models:` block** (`configs/llm_benchmark.yaml`) — the
   group→model map. Source of truth at runtime: `cli.py:382` reads
   `config.get("models", {}).get(group, [])`.
2. **Runner constants** (`benchmark/workloads/llm/runner.py:LLM_MODELS`,
   `MODEL_METADATA`, memory minimums) — fully populated for 1B–9B.
3. **Docs** (`docs/workloads.md`, `docs/output.md`, `README.md`) —
   documented 1B–9B in tables and filter chips.

Layer 1 was missing 1B and 3B. Layers 2 and 3 had them. Result: a
benchmark that *claims* to span 1B–9B but practically tests 7B+ on CPU
and 1.5B on NPU.

### Decision

Settle on **three canonical CPU groups: 1B, 3B, 7B**. Drop 8B and 9B
entirely from the benchmark surface. Keep 1.5B as a separate NPU-only
group (Hailo Model Zoo prebuilt HEFs are 1.5B and that's a hardware
constraint, not a benchmark design choice).

Rationale for this consolidation rather than going wider:
- 1B / 3B / 7B is the practical range an edge device cares about. Above
  7B, latency/memory on a Pi 5 (Cortex-A76, 8 GB shared RAM) makes the
  result more about swap thrashing than model capability.
- Three CPU groups is enough to plot a clean param-vs-throughput curve
  per platform without bloating the dashboard.
- 8B / 9B were already absent in practice (no one had pulled the
  models, no Hailo HEFs exist for them); removing them from the
  schema removes drift, not capability.

### Fix applied

Five files touched to bring all three layers in sync:

1. **`configs/llm_benchmark.yaml`** — added `1B:` and `3B:` entries to
   the `models:` block; removed `8B:` and `9B:` entries; updated
   `full` profile from `model_groups: ["7B", "8B", "9B"]` to
   `["1B", "3B", "7B"]`. The `1.5B` block stays, with its existing
   "NPU-only" comment reinforced.
2. **`benchmark/workloads/llm/runner.py`** — dropped `"8B"` and
   `"9B"` keys from `LLM_MODELS` (line ~96), `MODEL_METADATA`
   (lines ~127-132), and the memory-preflight `min_memory_gb` table
   (line ~761). Updated the `parameter_group` docstring on
   `ModelMetadata`.
3. **`docs/workloads.md`** — removed the 8B and 9B rows from the
   Model Groups table and the Memory Requirements table; swapped the
   `llama3.1:8b-instruct` quant-template example to
   `llama3.2:3b-instruct`.
4. **`docs/output.md`** — updated the dashboard filter-chip option
   lists from `1B, 3B, 7B, 8B, 9B` to `1B, 1.5B, 3B, 7B` (1.5B is
   surfaced because the NPU profile produces 1.5B rows); updated the
   Parameter Group colour legend; updated the example
   `model_groups: ["1B", "3B", "7B", "8B", "9B"]` snippet.
5. **`README.md`** — line 12 LLM summary now says "three CPU groups
   (1B, 3B, 7B)"; the `full` profile row in the profiles table now
   reads "(1B / 3B / 7B)".

### Verification

- `pytest tests/ -q` → all 196 tests still pass (no test pinned 8B or
  9B; the 1.5B-using NPU stub test is unaffected).
- `python -m benchmark info` loads the new YAML cleanly.
- `python -c "from benchmark.workloads.llm.runner import LLM_MODELS; print(sorted(LLM_MODELS.keys()))"` →
  `['1B', '3B', '7B']`.

### Don't reintroduce this

- **The YAML `models:` block is the source of truth.** Adding a model
  group requires entries in three places: (a) `configs/llm_benchmark.yaml`
  `models:` block, (b) `runner.py:LLM_MODELS` + `MODEL_METADATA` +
  memory-preflight table, (c) docs (`docs/workloads.md` table,
  `docs/output.md` filter chips, `README.md`). If they drift again, the
  benchmark surface and the docs will disagree silently — exactly the
  state this issue uncovered.
- **Pulling the 1B / 3B Ollama models is a separate operator action.**
  Wiring them into the YAML doesn't pull them; `ollama pull` is still
  required before `--profile full` can actually run those rows. On
  this Pi only `llama2:7b` is currently pulled — the 1B / 3B groups
  will need `ollama pull llama3.2:1b granite3.1-moe:1b sailor2:1b
  llama3.2:3b granite3.1-moe:3b starcoder2:3b` before they're usable.
  This is intentional: pulling 6 more LLMs ahead of an actual `--profile
  full` run is a network/disk decision the operator should make.

## Issue 7 — Llama-only consolidation: one model per group, NPU pivots to llama3.2:3b

### Symptom

Issue 6 wired up 1B/3B groups but kept three models per group across
multiple model families: `granite3.1-moe`, `sailor2`, `starcoder2`,
`mistral`, `olmo2`, `qwen2`/`deepseek` (the latter as the NPU-only 1.5B
group). That design produced a sprawl that was actively hostile to a
defensible benchmark:

- **Provenance / licensing inconsistency.** Each model family carries
  its own training-data corpus and license terms. Comparing throughput
  across families muddies what's being measured (the model? the runtime?
  the quantization recipe?).
- **Architecture inconsistency.** `granite3.1-moe` is a Mixture-of-Experts
  model; `starcoder2` is code-specialized; the rest are dense / general.
  The dashboard had to carry MoE / Code badges to flag these — a code
  smell that the surface was wider than the analysis could honestly
  support within a parameter group.
- **NPU profile drift.** The `npu` profile pinned `qwen2:1.5b` because
  it was the only prebuilt HEF Hailo Model Zoo GenAI shipped at the
  time. That made the only NPU comparison row a non-llama, non-CPU
  model — the cross-platform CPU-vs-NPU plot couldn't show "same model,
  different backend" without bridging two model families.

### Decision

Consolidate to **llama-family only, one model per group**:

| Group | Model | Notes |
|---|---|---|
| 1B | `llama3.2:1b` | CPU sweep |
| 3B | `llama3.2:3b` | CPU sweep; **also** the NPU profile model (published Hailo HEF per `docs/hailo.md`) |
| 7B | `llama2:7b` | CPU sweep |

The 1.5B group is removed entirely (no llama 1.5B exists, and the
qwen2/deepseek 1.5B HEFs are out of scope under the llama-only rule).
The NPU profile pivots from `qwen2:1.5b` (1.5B group) to `llama3.2:3b`
(3B group), which gives the dashboard a clean "same model, two
backends" comparison row on the AI HAT+ 2 Pi.

Rationale:
- **Within-group fairness is now real.** All three groups share the
  same model family (Meta's Llama lineage), the same provenance, the
  same dense-decoder architecture, the same general-purpose
  specialization. Differences in throughput across groups can be
  attributed to parameter count alone — which is the comparison the
  benchmark was built to make.
- **Cross-backend (CPU vs Hailo-10H NPU) comparison is now llama-aligned.**
  The 3B row can be plotted side by side: Ollama-CPU `llama3.2:3b` vs
  hailo-ollama `llama3.2:3b`. Same prompts, same model, different
  silicon — the cleanest possible attribution of any throughput delta
  to the accelerator.
- **Local disk discipline.** With one model per group, an operator
  pulling everything needs ~7 GB total (1.3 + 2.0 + 3.8 GB). Down from
  the multi-family sweep's ~20+ GB.

### Risk

`llama3.2:3b` on Hailo-10H assumes hailo-ollama serves it. Per
`docs/hailo.md` and the YAML comment retained from the previous
revision (`"…ramp to … llama3.2:3b as each is validated on hardware"`),
this was already a planned ramp target. If the AI HAT+ 2 Pi's
hailo-ollama install doesn't serve `llama3.2:3b` (e.g., older Hailo
Model Zoo GenAI catalogue), the verify_ai_hat_plus_2.sh `llm-npu-…`
step will fail at `/api/pull`. Mitigation: pull the HEF manually via
the curl command in `docs/troubleshooting.md`, or temporarily revert
the NPU profile's `models:` to `qwen2:1.5b` until the Model Zoo update
is installed.

### Fix applied

Touched files (this session):

| Layer | File | Change |
|---|---|---|
| Config | `configs/llm_benchmark.yaml` | `models:` block reduced to llama-only one-per-group; 1.5B group removed; `npu` profile pivots to `model_groups: ["3B"]` / `models: ["llama3.2:3b"]`; comments updated |
| Runner | `benchmark/workloads/llm/runner.py` | `LLM_MODELS` and `MODEL_METADATA` reduced to the three llama models |
| Tests | `tests/test_llm_npu_profile.py` | `HAILO_GENAI_PREBUILT_HEFS` and `SMALL` sets reduced to `{"llama3.2:3b"}`; `test_npu_profile_starts_smallest` renamed to `test_npu_profile_stays_at_or_below_3b` |
| Tests | `tests/test_llm_npu_unsupported_stub.py` | All `qwen2:1.5b` / `1.5B` fixtures → `llama3.2:3b` / `3B` |
| Tests | `tests/test_llm_quant.py` | `mistral:7b` and `llama3.1:8b` examples → `llama3.2:3b` / `llama3.2:1b` |
| Tests | `tests/test_dashboard_backend_filter.py` | `qwen2:1.5b` → `llama3.2:3b` |
| Tests | `tests/test_hw_verify_validators.py` | default `model_name` → `llama3.2:3b` |
| Setup | `scripts/setup_rpi_ai_hat_plus.sh` | `pull_models()` model list → `(llama3.2:1b, llama3.2:3b, llama2:7b)` |
| Setup | `scripts/setup_jetson_orin_nano.sh` | same |
| Setup | `scripts/setup_rpi_ai_hat_plus_2.sh` | same; `--with-genai` curl example now pulls `llama3.2:3b` |
| Verify | `scripts/verify_ai_hat_plus.sh` | step renamed to `llm-npu-llama3.2:3b [unsupported-on-this-hw]` |
| Verify | `scripts/verify_ai_hat_plus_2.sh` | step renamed to `llm-npu-llama3.2:3b`; preflight curl example updated |
| Docs | `README.md` | Overview LLM bullet + `npu` row in profiles table |
| Docs | `docs/workloads.md` | Model Groups table reduced to 3 llama rows; MoE/Code constraints removed; Memory Requirements table reduced |
| Docs | `docs/output.md` | Filter chip lists reduced; MoE/Code badges + Architecture/Specialization filters retired (schema fields kept for forward compat) |
| Docs | `docs/methodology.md` | Group-Safe Aggregation now describes the 1B/3B/7B llama-only design |
| Docs | `docs/troubleshooting.md` | All `qwen2:1.5b` references in the GenAI section → `llama3.2:3b` |
| Docs | `docs/hailo.md` | Prebuilt HEF table now distinguishes "in scope" (llama3.2:3b) from "available but out of scope" (qwen2 / deepseek); curl-pull example and "npu profile starts with…" prose updated |

Local Ollama state was also pruned: `granite3.1-moe:1b`, `sailor2:1b`,
`granite3.1-moe:3b`, `starcoder2:3b` removed via `ollama rm`. Final
list is `llama3.2:1b` / `llama3.2:3b` / `llama2:7b`.

### Verification

- `pytest tests/ -q` → all tests still pass.
- `python -c "from benchmark.workloads.llm.runner import LLM_MODELS; print(LLM_MODELS)"`
  → `{'1B': ['llama3.2:1b'], '3B': ['llama3.2:3b'], '7B': ['llama2:7b']}`.
- `python -c "import yaml; print(yaml.safe_load(open('configs/llm_benchmark.yaml'))['npu'])"`
  → `{'api_base': 'http://localhost:8000', 'backend': 'hailo-10h', 'npu_metrics': True, 'model_groups': ['3B'], 'models': ['llama3.2:3b'], 'prompt_set': 'drone'}`.
- `ollama list` → 3 rows, all llama.

### Don't reintroduce this

- Adding a non-llama model to `configs/llm_benchmark.yaml` should fail
  in code review on the grounds that it violates the llama-only rule.
  If a future PR genuinely needs to extend the surface (e.g., to add
  Mistral for a comparative study), it must update Issue 7 here, the
  Model Groups table, and the `MOE Badge`/`Code Badge` retirement note
  in `docs/output.md`.
- The `Architecture` and `Specialization` fields on `LLMResult` are
  intentionally retained on the schema. Don't strip them — they are
  the forward-compat hook for any future widening, and removing them
  would force a schema migration when re-introduced.

## Pre-launch environment audit (non-blocking expectations)

After Issues 1-5 are resolved, an audit of `verify_ai_hat_plus.sh`'s
prerequisites against the live machine produced this matrix. None of
the gaps below are blockers — they are designed-in failure modes the
script handles, or expected one-time costs. Documented so the run
output isn't mistaken for a regression.

| Prerequisite | Status |
|---|---|
| Venv exists & functional | ✅ `/home/snpi/Documents/Benchy/venv` (Python 3.11.2) |
| Venv must be activated by the launching shell | ⚠ Operator action — the verify script preflights for `python` on `PATH` and aborts if missing; it does **not** auto-activate the venv |
| `pytest` in venv | ✅ 9.0.3 (Issue 4 fix) |
| `ultralytics` in venv | ✅ 8.4.7 |
| `hailo_platform` in venv | ✅ 4.20.0 (Issue 3 fix) |
| `yolov8n.pt` in repo root | ✅ Present (downloaded by setup `pull_models`) |
| `yolov8n-{obb,seg,pose}.pt`, `yolo11n-{obb,seg,pose}.pt` | ⚠ Absent — Ultralytics auto-downloads on first use from public CDN; adds a few MB and ~30 s per model to the first sweep |
| `yolo26n-{obb,seg,pose}.pt` | ❌ **404 on Ultralytics releases** — no public weights. These three steps fail at download, are tagged `[experimental]`, and do not gate the script's exit code |
| Hailo `.hef` cache `~/.cache/benchy/hailo/` | ❌ Empty — first run of each of the 6 supported v8/v11 OBB/seg/pose models triggers an ONNX → HAR → HEF compile. Per `docs/hailo.md`, **5-30 min per model**, so ~1-5 hours of one-off compile time before benchmark numbers land. Cache is persistent — subsequent sweeps are fast |
| Ollama daemon on `:11434` | ✅ Up |
| `llama2:7b` (Ollama) | ✅ Pulled (Q4_0, 3.8 GB) — sufficient for the `drone` LLM profile, which uses `llama2:7b` without a quant_tag_template (`configs/llm_benchmark.yaml:32-35`) |
| `hailo-ollama` on `:8000` | n/a — LLM-NPU step is gated by `Platform.RPI_AI_HAT_PLUS_2` in `cli.py`. On this Hailo-8 board it writes a zero-valued `[unsupported-on-this-hw]` stub regardless of `:8000` reachability, by design |
| `/dev/hailo0` + `hailo_pci` 4.20.0 | ✅ Live (Issue 1 fix verified) |
| Disk free | ✅ 73 GB on `/` |
| RAM | ✅ 13 GB free of 15 GB |

### Why three v26 steps fail by design

Concrete answer (sourced from
`benchmark/workloads/yolo/execution.py:130-132` and `docs/hailo.md:37-68`):

1. **No public weights.** Ultralytics has not released
   `yolo26n-{obb,seg,pose}.pt`. The Ultralytics GitHub release URL
   returns HTTP 404 — confirmed during this session. v26 detection and
   classification weights *are* published; the three rotated/dense-output
   task heads aren't.
2. **No public Hailo Model Zoo backing.** Even with weights, the
   support tier diverges: v11-obb has an official Model Zoo entry
   (known-good INT8 calibration recipe), v8-obb is community-supported,
   v26-obb has neither. The conversion `.pt → .onnx → .har → .hef`
   needs a careful per-task INT8 recipe — for non-standard heads
   (rotated regression for OBB, 32-coefficient prototype masks for seg,
   56-channel keypoint regression for pose) without a published recipe,
   conversion is a guess.
3. **Risk sits at conversion, not postprocessing.** The host-side
   decoders in `benchmark/workloads/yolo/postprocessing.py`
   (`_process_obb` + Sutherland-Hodgman rotated NMS,
   `_process_segmentation` + mask blender, `_process_pose` + 17-keypoint
   COCO decoder) are version-agnostic — they consume task-specific
   tensor shapes that v8/v11/v26 all emit identically. pytest already
   covers them. What pytest can't prove is that an INT8-quantized v26
   head, after passing through the Hailo Dataflow Compiler, still emits
   tensors that decode to sensible boxes / masks / keypoints. Only
   hardware can confirm that.

So `[experimental]` is a **verification status, not a code defect**.
The whitelist in `execution.py` allows the attempt for symmetry; the
HW-verify runners are explicitly designed as the test surface that
promotes a v26-* task from experimental to verified (or kicks it back
as broken) once a real Pi compiles and runs the HEF.

### Why the LLM-NPU step "fails" by design

`verify_ai_hat_plus.sh:81-83` runs `python -m benchmark run llm
--profile npu` on this AI HAT+ board. Hailo-8 has no onboard SDRAM and
isn't a transformer accelerator, so it can't host an LLM. Rather than
emit no row at all, `cli.py:_build_unsupported_npu_stubs()` writes a
zero-valued `LLMResult` tagged `backend="hailo-10h"`,
`prompt_id="unsupported-on-this-hardware"`. That guarantees the
cross-platform dashboard renders an explicit "tried, 0 TPS" bar on the
NPU axis for this Pi, matching the AI HAT+ 2 Pi's row count for chart
comparison. The step is tagged `[unsupported-on-this-hw]` and does not
gate exit code.

## Verification before running `verify_ai_hat_plus.sh`

After the two fixes above, this set of probes should all succeed without
error output:

```bash
which hailortcli              # expect /usr/bin/hailortcli
hailortcli --version          # expect HailoRT-CLI version 4.20.0
hailortcli fw-control identify # expect Board Name: Hailo-8 or Hailo-8L, FW 4.20.0
ldd /usr/bin/hailortcli | grep hailort   # expect /lib/libhailort.so.4.20.0

# Inside the venv (Issue 3 — must work after the symlink fix):
source venv/bin/activate
python -c "import hailo_platform as h; print(h.__version__)"  # expect 4.20.0
python -c "from hailo_platform import VDevice; print('ok')"
python -c "from benchmark.workloads.yolo.backends.hailo import HailoBackend; print('ok')"

# Issue 4 — dev extras must be installed in the venv:
pytest --version              # expect pytest 9.x
pytest tests/ -q              # expect 196 passed (~2s on this Pi)

# Issue 5 — confirm no other root-owned artefacts in the project root:
find /home/snpi/Documents/Benchy -maxdepth 3 -uid 0   # expect empty
```

Once those pass, the canonical full sweep is:

```bash
cd /home/snpi/Documents/Benchy
source venv/bin/activate
./scripts/verify_ai_hat_plus.sh
```

## Other observations (non-blocking)

- DKMS has `hailo_pci/4.20.0` built for six different kernels (two
  Debian 6.1 stems, the 6.12.62 Pi kernels, and the current 6.12.75
  Pi kernels). That's harmless — DKMS keeps a build per installed
  kernel — but if disk space ever gets tight, `dkms remove` of the
  unused 6.1.x builds is safe.
- Boot config has both `dtparam=pciex1_gen=3` and `dtparam=pciex1_gen=2`
  in `/boot/firmware/config.txt`. The later line wins, so the link is
  capped at Gen 2. **This board is the 26 TOPS Hailo-8** (confirmed by
  identify), which is the variant that actually saturates Gen 3 — so
  the Gen 2 line should be removed (or commented out) and the Pi
  rebooted to recover full PCIe bandwidth before benchmark numbers are
  treated as final. Not blocking for `verify_ai_hat_plus.sh`, but it
  will skew throughput-bound results downward.

## Issue 8 — Hailo conversion pipeline aborted: `ModuleNotFoundError: No module named 'onnx'` on every YOLO step

### Symptom

The first end-to-end `verify_ai_hat_plus.sh` run after Issues 1-7 were
resolved finished in **6 minutes** — far too fast. Every YOLO step in
the 13-step bundle produced an empty `yolo_results: []` row. The
per-step log told the story:

```
2026-04-27 17:58:25 - benchmark.workloads.yolo.backends.hailo - INFO - HEF not found at models/hailo/hailo8/v8/detection/yolov8n/model.hef, starting conversion pipeline...
2026-04-27 17:58:29 - benchmark.workloads.yolo.conversion.onnx_export - WARNING - ONNX export not available: No module named 'onnx'
2026-04-27 17:58:31 - benchmark.cli - ERROR -   Failed: ONNX export not available. Install ultralytics: pip install ultralytics onnx
```

All 9 supported v8/v11 OBB/seg/pose/det HEFs hit step 1 of the
conversion pipeline (`.pt → .onnx`), failed identically at the missing
`onnx` import, exited the pipeline, and wrote a bench file with empty
`yolo_results`. `scripts/hw_verify_validators.py` then flagged each
one as "yolo_results is empty — the runner produced no rows".

### Root cause

`onnx` and `onnxruntime` were not in `setup.py:install_requires`. They
were listed in `README.md` under "Hailo NPU (Raspberry Pi only)" but
never actually pinned anywhere — so `pip install -e ".[dev]"` (which
the platform setup scripts run) silently shipped a venv that *looked*
complete but had a missing piece of the YOLO Hailo pipeline.

`benchmark/workloads/yolo/conversion/onnx_export.py` imports `onnx` at
module load time. When that import fails, the conversion pipeline
emits a polite warning and refuses to compile — but the runner doesn't
propagate the error as a hard failure. Instead it returns an empty
result list, the bench file gets written with `yolo_results: []`, and
the user sees "Benchmark Complete" in their terminal. Silent partial
success — the worst kind.

This is Issue 4's failure shape repeating for a different dep: a
runtime-required Python package missing because nothing in
`setup.py:install_requires` forces it to be installed.

### Fix applied — three layers

**(a) Pin in `setup.py:install_requires`.** Future `pip install -e .`
or `pip install -e ".[dev]"` invocations now pull the deps:

```python
install_requires=[
    "psutil>=5.9.0",
    "requests>=2.28.0",
    "pyyaml>=6.0",
    "numpy>=1.21.0",
    "ultralytics>=8.0.0",
    # Required by the Hailo conversion pipeline (.pt → .onnx → .har → .hef).
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
],
```

Mirrored in `requirements.txt` for non-editable installs.

**(b) Pin explicitly in setup scripts.**
`setup_rpi_ai_hat_plus.sh:install_python_deps` and
`setup_rpi_ai_hat_plus_2.sh:install_python_deps` both append `onnx
onnxruntime` to the explicit pip block that runs before
`pip install -e ".[dev]"`. Redundant with (a) but defensive.

**(c) Verify-time self-heal in `scripts/hw_verify_common.sh`.** New
`hw_ensure_python_deps` function called from both verify entrypoints
right after `hw_init`:

```bash
hw_ensure_python_deps() {
    local py
    py="$(command -v python || command -v python3)"
    local pairs=("onnx:onnx" "onnxruntime:onnxruntime")
    local missing=()
    for pair in "${pairs[@]}"; do
        local module="${pair%%:*}"
        local pkg="${pair##*:}"
        if ! "$py" -c "import ${module}" >/dev/null 2>&1; then
            missing+=("$pkg")
        fi
    done
    if (( ${#missing[@]} > 0 )); then
        info "Self-heal: pip-installing missing deps (${missing[*]})..."
        "$py" -m pip install --quiet "${missing[@]}" || { error "..."; exit 2; }
    fi
}
```

This unblocks an existing venv that pre-dates the (a)/(b) pins without
forcing the user to re-run the sudo setup script. The cost is one
`import` probe per dep at script start — sub-second, no network unless
something is actually missing.

### Verification

- `pytest tests/ -q` → 196 passed in 1.33s after the fix.
- Self-heal dry-run on the affected venv reported `WOULD-INSTALL: onnx
  onnxruntime` (correct — both were missing).

### Don't reintroduce this

- **Any new runtime-required Python dep MUST go in
  `setup.py:install_requires`.** Listing it only in README dependency
  tables doesn't install it. If the dep is platform-specific (e.g.,
  `hailo-platform` is RPi-only), gate it via `extras_require` so a
  clear `pip install -e ".[hailo]"` invocation pulls it.
- The setup scripts' explicit pip block is defense in depth, not the
  primary install mechanism.
- If a YOLO step ever produces empty `yolo_results: []` while reporting
  "Benchmark Complete", check the per-step log under
  `results/hw_verify_<ts>/logs/` for a swallowed `ModuleNotFoundError`.
  The runner currently swallows them as warnings; the validator is
  what catches the empty-rows case downstream.

## Issue 9 — LLM-on-CPU step times out: `Read timed out (read timeout=300)` on `llama2:7b`

### Symptom

The `llm-cpu-llama2:7b (drone prompts)` step in the verify bundle ran
for exactly 5 minutes and then errored:

```
2026-04-27 17:59:05 - benchmark.workloads.llm.runner - INFO - Benchmarking prompt: scene_description
2026-04-27 17:59:05 - benchmark.workloads.llm.runner - INFO -   Warmup runs: 3, Measured runs: 10
2026-04-27 18:04:05 - benchmark.workloads.llm.runner - ERROR - Generation failed: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=300)
```

`scene_description` is the first drone prompt, run as warmup #1. The
HTTP `POST /api/generate` request hit the 300-second read timeout
before Ollama returned the streamed completion.

### Root cause

`benchmark/workloads/llm/runner.py` had a hardcoded `timeout=300` on
both `generate_stream` (line 522) and `generate` (line 564). That was
tight even for a warm `llama2:7b`: on Cortex-A76, Q4-quantized 7B
weights generate ~3-5 tokens/sec, and a 256-token completion plus
30-60s of TTFT lands around 100-150s when the model is already loaded.

**But Ollama unloads idle models.** When the warmup HTTP request hits
a daemon that hasn't run `llama2:7b` recently, the daemon transparently
re-loads ~3.8 GB of weights from disk first. On an SD-card-backed Pi,
that load step alone runs 60-180s before generation starts. Add the
100-150s of generation, and a single warmup request can consume
200-330s — straddling the 300s boundary.

The 300s constant was originally chosen for Jetson Orin Nano ("a
256-token completion in <5 min") and never re-evaluated when the
verify suite started exercising the same code on a Pi 5.

### Fix applied

Bumped the timeout from 300 to 600 in both call sites:

```python
# benchmark/workloads/llm/runner.py:522 (generate_stream)
response = self.session.post(
    f"{self.base_url}/api/generate",
    json=payload,
    stream=True,
    timeout=600,
)

# benchmark/workloads/llm/runner.py:564 (generate)
response = self.session.post(
    f"{self.base_url}/api/generate",
    json=payload,
    timeout=600,
)
```

`replace_all=True` was safe because the two were the only `timeout=300`
literals in the file (the others are 5/10/3600 for short health
checks and the long model-pull endpoint — verified pre-edit).

600s gives ~3 min for cold load on SD storage + ~2.5 min for
generation + margin. Subsequent warmup requests reuse the loaded
model and finish in 60-120s, well inside the window.

### Why not a smaller default model

A reasonable alternative would be to switch the verify LLM-CPU step
from `llama2:7b` to `llama3.2:3b` to shave ~15 minutes off every
verify run. Decided against:

1. The LLM-CPU step exists as the **cross-board comparison row**. Both
   AI HAT+ and AI HAT+ 2 Pis run identical Cortex-A76 SoCs; running
   the same model on both gives an apples-to-apples baseline. Changing
   the model in only one runner would break that.
2. A 7B model is more representative of "what realistic edge users
   actually want to run" than 3B. Benchmarking the smaller model is
   fine but loses the headline number.
3. The timeout was wrong — fixing the timeout is the surgical fix.

If the Pi 5 ever proves too slow even for 600s, the right next move is
to make the timeout configurable via `configs/llm_benchmark.yaml`
(`generation.request_timeout_seconds`), not to swap models.

### Don't reintroduce this

- HTTP timeout values that hardcode "<5 min should be enough" are a
  red flag on Cortex-A76 — first-request cold loads are
  model-size-dominant on SD storage. Default to ≥600s for any model
  ≥7B.
- The `llama2:7b` choice for the drone profile is intentional (Issue 7
  consolidation) — don't switch it to a smaller model just to make
  verify faster. Bump the timeout if cold loads exceed 600s.

## Issue 10 — `cmd_report` crashed with `dict contains fields not in fieldnames` on the verify dashboard step

### Symptom

After all 13 verify steps completed (most as failures from Issue 8,
but the bench files existed), `hw_finalize_with_report` invoked
`python -m benchmark report --input results/hw_verify_<ts>/`. The
report step crashed in the LLM CSV writer:

```
File "/home/snpi/Documents/Benchy/benchmark/aggregation/csv_writer.py", line 131, in write_llm_aggregated
    writer.writerow(metric.to_dict())
File "/usr/lib/python3.11/csv.py", line 149, in _dict_to_list
    raise ValueError("dict contains fields not in fieldnames: ")
ValueError: dict contains fields not in fieldnames: 'ttft_median_ms', 'tps_median',
  'npu_power_watts_mean', 'hailort_version', 'truncation_rate', 'peak_memory_mb_mean',
  'backend', 'prompt_category', 'specialization', 'parameter_group',
  'npu_utilization_percent_mean', 'architecture'
```

12 fields in the dict were not declared in `fieldnames`. `csv.DictWriter`
raises `ValueError` rather than dropping unknown fields — strict mode
by default.

### Root cause

`benchmark/aggregation/csv_writer.py:LLM_AGGREGATED_COLUMNS` and
`benchmark/aggregation/aggregator.py:LLMAggregatedMetrics.to_dict()`
had drifted. The dataclass had grown four batches of fields across
recent phases:

| Phase | Fields added to dataclass | Fields added to columns list |
|---|---|---|
| Initial | `model_name`, `prompt_id`, `ttft_*`, `tps_*`, `latency_*`, `prompt_tokens_mean`, `output_tokens_mean`, `cpu_percent_mean`, `accelerator_percent_mean`, `memory_used_mb_mean`, `power_watts_mean` | (matched) |
| Model Expansion Phase 1 | `parameter_group`, `architecture`, `specialization`, `prompt_category` | (not added) |
| Model Expansion Phase 5 | `peak_memory_mb_mean`, `truncation_rate` | (not added) |
| Stats robustness | `ttft_median_ms`, `tps_median` | (not added) |
| Phase 7 backend axis | `backend`, `npu_utilization_percent_mean`, `npu_power_watts_mean`, `hailort_version` | (not added) |

The dataclass + JSON path tested fine because JSON is dict-shaped and
unknown keys are harmless. The CSV path was untested in CI —
`tests/test_dashboard_backend_filter.py` exercises the aggregator and
the dashboard renderer, but no test exercises
`AggregatedCSVWriter.write_llm_aggregated` end-to-end with a populated
`LLMAggregatedMetrics`. So the drift went unnoticed until `cmd_report`
ran on a real bench bundle.

The first LLM row to trigger the crash on this verify run was the
unsupported-on-this-hw stub — it carries `backend="hailo-10h"`, which
is not in the column list. So even though the LLM-CPU step had failed
to produce rows (Issue 9), the LLM-NPU stub row alone was enough to
crash the report.

### Fix applied

Replaced `LLM_AGGREGATED_COLUMNS` with the full 34-field list matching
`LLMAggregatedMetrics.to_dict()` exactly, in the same order, with
section comments mirroring the dataclass field groups (initial /
Model Expansion / resource / Phase 7 backend axis). Added a comment at
the top of the constant to make the lockstep requirement explicit:

```python
# Column definitions for aggregated LLM results.
# Must stay in lockstep with LLMAggregatedMetrics.to_dict() in
# benchmark/aggregation/aggregator.py — csv.DictWriter raises ValueError
# if the dataclass emits a field not listed here.
LLM_AGGREGATED_COLUMNS = [...]
```

### Verification

```python
>>> from benchmark.aggregation.csv_writer import LLM_AGGREGATED_COLUMNS
>>> from benchmark.aggregation.aggregator import LLMAggregatedMetrics
>>> m = LLMAggregatedMetrics(model_name='x', model_size='3B', prompt_id='p')
>>> set(m.to_dict().keys()) == set(LLM_AGGREGATED_COLUMNS)
True
>>> set(m.to_dict().keys()) - set(LLM_AGGREGATED_COLUMNS)
set()
>>> set(LLM_AGGREGATED_COLUMNS) - set(m.to_dict().keys())
set()
```

`pytest tests/ -q` → 196 passed.

### Don't reintroduce this

- **The columns list and `to_dict()` must stay in lockstep.** Any
  addition to `LLMAggregatedMetrics` (or `YOLOAggregatedMetrics`, or
  `PlatformSummary`) that flows through `to_dict()` must also be added
  to the matching `*_AGGREGATED_COLUMNS` constant in `csv_writer.py`.
  The comment on the constant (added in this fix) flags this
  requirement.
- Worth adding an end-to-end test: instantiate a populated
  `LLMAggregatedMetrics`, hand it to
  `AggregatedCSVWriter.write_llm_aggregated`, and assert the round-trip
  succeeds. Would have caught this on the first commit that added the
  backend axis. (Not done in this fix — minimal scope.)
- The same drift can hit the YOLO and platform writers. Spot-checked
  during this fix; both currently match. If a future change touches
  `YOLOAggregatedMetrics.to_dict()` or `PlatformSummary.to_dict()`, do
  the same alignment check.

---

## Issue 11 — Hailo Dataflow Compiler is x86_64-only; Pi cannot compile HEFs

**Surfaced:** 2026-04-28, second `./scripts/verify_ai_hat_plus.sh` run on the Pi 5 + AI HAT+ (26 TOPS) after Issues 8/9/10 had been fixed. With `onnx`/`onnxruntime` self-heal in place, ONNX export now succeeds — but the next pipeline stage requests the Hailo SDK and finds nothing.

### Symptom

All 10 YOLO Hailo steps fail at the same place. Per-step log
(`results/hw_verify_20260428_091256/logs/yolo-v8-detection.log`):

```
WARNING - Hailo SDK not available for HAR generation. Only ONNX export will be performed.
WARNING - Hailo Dataflow Compiler not available for HEF compilation. Pipeline will stop at HAR.
INFO - Stage 1/3: ONNX Export
INFO - ONNX exported successfully: models/hailo/v8/detection/yolov8n/model.onnx
ERROR - Failed: ONNX file generated but Hailo SDK not available. ONNX file at: models/hailo/v8/detection/yolov8n/model.onnx. Please complete conversion using Hailo SDK.
Validation FAILED for ...: yolo_results is empty — the runner produced no rows
```

`pip install hailo-dataflow-compiler` does not work on the Pi:

```
$ python -c "import hailo_sdk_client"
ModuleNotFoundError: No module named 'hailo_sdk_client'
```

### Root cause

The Hailo Dataflow Compiler (and its `hailo_sdk_client` Python module) is **x86_64 Linux only**. There is no aarch64 build, official or community. This is documented in `benchmark/workloads/yolo/conversion/hef_compiler.py:73-78`:

> *"This requires the Hailo Dataflow Compiler which is part of the Hailo SDK and requires a license from Hailo Developer Zone. Note: Compilation is typically performed on an x86_64 development machine. The resulting HEF file is then deployed to the target Raspberry Pi with Hailo HAT."*

In other words, the standard Hailo workflow assumes a two-machine setup: workstation compiles HEFs, Pi runs them. The benchmark suite's in-tree `.pt → .onnx → .har → .hef` pipeline was written for the workstation case but called from `prepare_model()` on the Pi without a guard. ONNX export works on aarch64 (Ultralytics handles that natively), so the failure surfaces only at HAR generation time.

### Fix (this session)

Added a **prebuilt HEF source layer** at `benchmark/workloads/yolo/conversion/hef_source.py`. Before falling through to the compile path, `HailoBackend.prepare_model()` now consults two locations in order:

1. **`resources/hefs/`** in the repo. Users drop HEFs here using the convention `<yolo_version>_<task>_<model_size>_<arch>.hef` (e.g. `v8_detection_n_hailo8.hef`, `v11_pose_s_hailo10h.hef`). This is the canonical landing spot for HEFs compiled on a workstation. `resources/hefs/NAMING.txt` documents the convention.
2. **`/usr/share/hailo-models/`**. The `rpicam-apps-hailo-postprocess` Debian package ships with a curated subset of Hailo Model Zoo HEFs vetted by Raspberry Pi. The mapping from our `(yolo_version, task, size, arch)` tuple to its filenames lives in `hef_source.py:SYSTEM_PACKAGE_MAP`. As of Pi OS Bookworm 2026-04, the system package covers `yolov8s` detection and `yolov8s` pose only (both `_h8` and `_h8l` variants).

If neither source matches, the backend falls through to the compile pipeline, which then fails on aarch64 with its existing clear error. The new `find_prebuilt_hef()` log message also tells the user the exact filename to drop into `resources/hefs/`.

Files changed:

- **`benchmark/workloads/yolo/conversion/hef_source.py`** (new) — `find_prebuilt_hef()` + `SYSTEM_PACKAGE_MAP` + naming-convention helpers.
- **`benchmark/workloads/yolo/backends/hailo.py`** — `prepare_model()` consults the source layer between the cache check and the compile-pipeline fallback. If a prebuilt HEF is found, it's `shutil.copy2`'d into the runtime cache so the existing validation/load path picks it up unchanged.
- **`resources/hefs/.gitkeep`** + **`resources/hefs/NAMING.txt`** — directory scaffolding and naming convention.
- **`tests/test_hef_source.py`** (new) — covers naming-convention parsing, repo-vs-system precedence, the unhappy-no-match path, and locks the `SYSTEM_PACKAGE_MAP` to filenames actually shipped by the system package (so future edits can't silently invent paths that don't exist).
- **`docs/hailo.md`** — added "Prebuilt HEF source layer" section under Compilation Requirements with the lookup order, naming convention, and the workstation-compile note.
- **`docs/troubleshooting.md`** — new row in the HW-verify table for the "ONNX file generated but Hailo SDK not available" failure mode.

### Verification

```
$ pytest tests/test_hef_source.py -q
13 passed in 0.27s
$ pytest tests/ -q
209 passed in 1.39s
$ python -c "
from benchmark.workloads.yolo.conversion.hef_source import find_prebuilt_hef
from benchmark.schemas import YOLOTask
print(find_prebuilt_hef('yolov8s.pt', 'v8', YOLOTask.DETECTION, 'hailo8'))
print(find_prebuilt_hef('yolov8s-pose.pt', 'v8', YOLOTask.POSE, 'hailo8'))
print(find_prebuilt_hef('yolov8n.pt', 'v8', YOLOTask.DETECTION, 'hailo8'))
"
/usr/share/hailo-models/yolov8s_h8.hef
/usr/share/hailo-models/yolov8s_pose_h8.hef
None
```

### Don't reintroduce this

- **Don't try to install `hailo-dataflow-compiler` on the Pi.** It will not work, and the error path is more confusing than the prebuilt-HEF flow.
- **Extend `SYSTEM_PACKAGE_MAP` only with files that physically exist in `/usr/share/hailo-models/`.** The locking test in `tests/test_hef_source.py` asserts the value set, but the *list* of observed filenames in that test should be kept honest. If a new system package version adds HEFs, run `ls /usr/share/hailo-models/` and update both the map and the test fixture in the same commit.
- The verify suite still requests `yolov8n` (and its `-seg`/`-pose`/`-obb` variants) plus all v11/v26 variants. Most are not in the system package, so a clean verify run on the Pi requires the user to either (a) compile those HEFs on an x86_64 workstation and stage them in `resources/hefs/`, or (b) accept that those steps will fail until they do. The verify script's continue-on-failure semantics means the rest of the sweep still runs.
- For boards we want to ship "Just Works", consider committing a curated set of HEFs to `resources/hefs/` (with appropriate licence checks). Out of scope for this session.

---

## Issue 12 — LLM cold-load exceeds 600s timeout; pre-warm step + configurable timeout

**Surfaced:** 2026-04-28, same verify run as Issue 11. The LLM-on-CPU comparison row (llama2:7b on the drone prompt set) timed out *again* even after Issue 9's bump from 300s → 600s.

### Symptom

`results/hw_verify_20260428_091256/logs/llm-cpu-llama2_7b_drone_prompts.log`:

```
2026-04-28 09:14:37 - Benchmarking llama2:7b
2026-04-28 09:14:37 - Benchmarking prompt: scene_description
2026-04-28 09:14:37 -   Warmup runs: 3, Measured runs: 10
2026-04-28 09:24:37 - Generation failed: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=600)
Validation FAILED: llm_results is empty — the runner produced no rows
```

Exactly 600s elapsed. The first warmup iteration's `/api/generate` call sat blocked on the Ollama cold-load and never returned within the window. Disk-side, llama2:7b is 3.8 GB of weights on SD storage. On a Pi 5 under typical ambient with no thermal preconditioning, that load can exceed 10 minutes.

### Root cause

Issue 9 raised the per-request timeout to 600s, which covers all the *measurement-time* cases (warmup + measured runs against a model that's already resident). It did not cover the **cold-load case** where the model has to be paged in from disk before the first response token is produced. That latency is one-time per benchmark invocation (subsequent calls hit the loaded model) but its variance is huge — 60s to 1200s+ depending on storage health, thermal headroom, and what else the Pi is doing.

Bumping the timeout further (e.g. to 1800s) covers the cold case but contaminates the timed loop: a hung request would now wait up to 30 min before failing. The right fix is to **separate cold-load cost from measurement** and not gate them on the same timeout.

### Fix (this session)

Added a `_prewarm_model()` step to `LLMBenchmarkRunner` that runs before the warmup-iterations loop:

- Issues a single `/api/generate` with `prompt=" "`, `max_tokens=1`, and `keep_alive="-1"` (Ollama keeps the model resident for the rest of the run).
- Uses the new `prewarm_timeout_seconds` config (default **1800s**), independent of the per-request timeout used in the timed loop.
- Logs the elapsed time but does not raise on `RequestException` — if the prewarm fails, the first measurement attempt will surface a more useful error.

Made both timeouts YAML-configurable so future bring-ups on slower / faster hardware don't need source edits:

- New `LLMBenchmarkConfig` fields: `http_timeout_seconds: int = 600` and `prewarm_timeout_seconds: int = 1800`.
- `OllamaClient.__init__` now takes `request_timeout` and uses it as the default for `generate` / `generate_stream`. Both methods accept per-call `timeout` and `keep_alive` overrides.
- `cli.py:run_llm_benchmark()` reads `benchmark.http_timeout_seconds` and `benchmark.prewarm_timeout_seconds` from the YAML.
- `configs/llm_benchmark.yaml` documents both fields under `benchmark:` with the default values and rationale.

Files changed:

- **`benchmark/workloads/llm/runner.py`** — added timeout fields to `LLMBenchmarkConfig`; reworked `OllamaClient` to honour a configurable timeout and accept per-call overrides; new `LLMBenchmarkRunner._prewarm_model()` plus a call site in `run()` between `_ensure_model_available()` and the prompts loop.
- **`benchmark/cli.py`** — plumbed both new YAML fields into the config constructor.
- **`configs/llm_benchmark.yaml`** — documented the new fields under `benchmark:`.
- **`tests/test_llm_prewarm.py`** (new) — covers the field defaults, the OllamaClient timeout wiring, the prewarm call's contract (one-shot, `keep_alive=-1`, long timeout), the no-raise-on-failure invariant, and the config-to-client plumbing.
- **`docs/troubleshooting.md`** — new row for the "still times out at 600s" failure mode.
- **`CLAUDE.md`** — added a Hardware-Specific Notes bullet describing the prewarm + timeout split and pointing at the YAML fields.

### Verification

```
$ pytest tests/test_llm_prewarm.py -q
6 passed in 0.04s
$ pytest tests/ -q
215 passed in 1.39s
```

### Don't reintroduce this

- **Don't hardcode timeouts in `runner.py` again.** Both the per-request timeout and the prewarm timeout are now YAML fields; if a hardware bring-up needs a longer (or shorter) timeout, change `configs/llm_benchmark.yaml`, not the source. The fact that we hit Issue 9's 300s, then Issue 12's 600s, on the same model in successive runs is the smoking gun for "magic timeout constants get stale".
- **Keep the prewarm separate from the warmup runs.** The N warmup runs in the config (default 3) measure first-pass-after-load behaviour for repeatability; they're not for absorbing the cold-load. If you fold the prewarm into the warmup loop you lose that distinction *and* you re-introduce the timeout-on-cold-load failure mode.
- **`keep_alive: -1` is load-bearing.** Without it the prewarm's effect expires with Ollama's default keep-alive (5 min), and a long prompt set could see the model paged out mid-run. If you ever need to relax this, make it a config field rather than re-hardcoding.
