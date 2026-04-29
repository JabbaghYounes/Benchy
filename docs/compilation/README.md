# Benchy HEF Compilation Docs

How to pre-compile YOLO models to `.hef` on an x86 workstation so Raspberry Pi
devices running Benchy can consume them directly.

Benchy itself does **not** compile models — it expects `.hef` files produced by
Hailo's toolchain (Dataflow Compiler / Model Zoo) on a more powerful machine.

There are two parallel pipelines, one per Hailo accelerator family. Pick the
folder that matches your hardware — a `.hef` compiled for one family will not
run on the other.

## Pipelines

- **[hailo8/](hailo8/)** — Hailo-8 (26 TOPS) and Hailo-8L (13 TOPS).
  Raspberry Pi AI HAT and AI Kit.
- **[hailo10h/](hailo10h/)** — Hailo-10H (40 TOPS). Raspberry Pi AI HAT+.

Each folder contains:

- `README.md` — hardware overview + HailoRT pairing
- `workflow.md` — parse → optimize → compile commands for that arch
- `models.md` — 10-model authoritative list with destination paths

## Shared docs

- **[setup.md](setup.md)** — current workstation setup state + remaining blockers
- **[tools.md](tools.md)** — Hailo Model Zoo vs Dataflow Compiler
- **[pitfalls.md](pitfalls.md)** — gotchas common to both pipelines

> Benchy automates the full `.pt → .onnx → .har → .hef` chain via
> the `compile` subcommand, which bypasses the runtime backend so
> a workstation without HailoRT or a Hailo device can still produce
> HEFs:
>
> ```bash
> python -m benchmark compile --hw-arch hailo10h --model yolov8n-seg.pt
> scripts/compile_workstation_hefs.sh --arch both
> scripts/fetch_prebuilt_hefs.py --arch both --dry-run
> ```
>
> The arch-specific workflow docs below are still useful when you need
> to drop to raw `hailo parser` / `hailomz` calls (e.g. to debug a
> failed compile or set custom end-nodes).

## TL;DR pipeline

```
Train (PyTorch / TF)
   ↓
Export → ONNX (640×640, batch=1, opset 11)
   ↓                                      ←── runs on x86 workstation
Parse → Optimize (INT8) → Compile         ←──   (--hw-arch flag picks family)
   ↓
.hef
   ↓
scp to Pi → run via HailoRT / Benchy
```

Everything up to `.hef` happens off-device. The Pi only consumes the artifact.

## Choosing the target

| Chip | Flag | TOPS | Typical Pi accessory |
|------|------|------|----------------------|
| Hailo-8 | `--hw-arch hailo8` | 26 | AI HAT |
| Hailo-8L | `--hw-arch hailo8l` | 13 | AI Kit |
| Hailo-10H | `--hw-arch hailo10h` | 40 | AI HAT+ |

Confirm the chip on the Pi (`hailortcli fw-control identify`) before
committing to a target. Wrong arch → unusable HEF.
