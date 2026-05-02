# Hailo-10H Pipeline

Compile YOLO models for the **Hailo-10H** — the 40 TOPS chip on the
**Raspberry Pi AI HAT+**.

## Contents

- [workflow.md](workflow.md) — parse → optimize → compile commands
- [models.md](models.md) — 10-model authoritative list with `hailo10h` paths

## Hardware

| Chip | Flag | TOPS | Module |
|------|------|------|--------|
| Hailo-10H | `--hw-arch hailo10h` | 40 | M.2 / AI HAT+ 2 |

The 10H supports both **INT8 and INT4** quantization (the Hailo-8 family
is INT8 only). Benchy's YOLO pipeline uses INT8; INT4 is currently
exercised on the LLM-on-NPU path.

Confirm what the Pi actually has:

```bash
hailortcli fw-control identify
```

A `.hef` compiled for `hailo10h` will not run on Hailo-8 or 8L hardware,
and vice versa. Maintain the Hailo-10H artifacts as a separate set under
`models/hailo/hailo10h/`.

## HailoRT pairing

The 10H is on a separate runtime track from the Hailo-8 family:

- **Hailo-10H requires HailoRT 5.x** (Hailo-8/8L use 4.x — they are not
  interchangeable)
- The repo bundles `Benchy/resources/hailo-10H/hailort_5.2.0_arm64.deb` for
  the Pi side
- Use a DFC build that targets HailoRT 5.x to keep the artifact loadable

Mismatch symptoms: the `.hef` won't load, or it crashes on first inference.

If the Pi is on an older HailoRT 5.x, install the bundled `.deb` to bring
it up to match. Do **not** install the `hailo-8` 4.23.0 `.deb` on a 10H
board — the runtimes are version-locked to the silicon family.

## Workstation requirements

- Ubuntu 20.04 or 22.04
- Hailo Dataflow Compiler **with hailo10h support** + Model Zoo
  (EULA-gated, from Hailo Developer Zone — older DFC builds may not include
  the 10H target)
- Optional CUDA GPU (speeds up the optimize/calibration step)
- ~50 GB free disk if pulling full calibration datasets — far less with
  curated subsets (recommended)

## Repo layout

HEFs land under:

```
models/hailo/hailo10h/<version>/<task>/<model>/model.hef
```

See [models.md](models.md) for the full destination table.

## When to prefer the 10H pipeline

- Higher throughput on the same model (40 vs 26 TOPS vs 8/8L)
- Headroom for larger / less-aggressively-quantized models
- Newer ops are more likely to be supported in the 10H DFC target

If your deployment fleet has both AI HAT and AI HAT+ devices, build both
sets — see [../hailo8/](../hailo8/).
