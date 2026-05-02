# Hailo-8 / Hailo-8L Pipeline

Compile YOLO models for the Hailo-8 family — the chips found on the
**Raspberry Pi AI HAT** (Hailo-8, 26 TOPS) and **Raspberry Pi AI Kit**
(Hailo-8L, 13 TOPS).

## Contents

- [workflow.md](workflow.md) — parse → optimize → compile commands
- [models.md](models.md) — 10-model authoritative list with `hailo8` paths

## Hardware

| Chip | Flag | TOPS | Module |
|------|------|------|--------|
| Hailo-8 | `--hw-arch hailo8` | 26 | M.2 / AI HAT |
| Hailo-8L | `--hw-arch hailo8l` | 13 | M.2 / AI Kit |

The two chips share the toolchain but **not** binaries — a `.hef` compiled
for `hailo8` will not run on `hailo8l` and vice versa. Maintain separate
`.hef` artifacts per chip.

Confirm what the Pi actually has:

```bash
hailortcli fw-control identify
```

## HailoRT pairing

The Hailo-8 family runs on the **HailoRT 4.x** track (Hailo-10H uses 5.x —
not interchangeable). DFC version determines the minimum HailoRT the `.hef`
will load on:

- DFC 3.33.1 → HailoRT 4.22+
- DFC 3.31.x → HailoRT 4.20+

Mismatch symptoms: the `.hef` won't load, or it crashes on first inference.

If the Pi is on an older HailoRT, install the bundled arm64 `.deb` —
`Benchy/resources/hailo-8/hailort_4.23.0_arm64.deb` — to bring it up to
match. (This `.deb` is the Pi-side runtime, **not** the workstation
compiler.)

## Workstation requirements

- Ubuntu 20.04 or 22.04
- Hailo Dataflow Compiler + Model Zoo (EULA-gated, from Hailo Developer Zone)
- Optional CUDA GPU (speeds up the optimize/calibration step)
- ~50 GB free disk if pulling full calibration datasets — far less with
  curated subsets (recommended)

## Repo layout

HEFs land under:

```
models/hailo/hailo8/<version>/<task>/<model>/model.hef
models/hailo/hailo8l/<version>/<task>/<model>/model.hef
```

See [models.md](models.md) for the full destination table.
