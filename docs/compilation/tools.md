# Tools: Model Zoo vs Dataflow Compiler

There is no universal one-shot `onnx2hef` converter. The official toolchain is
split into two pieces.

## Hailo Model Zoo (`hailomz`)

The closest thing to "model → HEF" in one command.

- Wraps the lower-level compiler
- Ships prebuilt `.hef` files for supported models — sometimes you can skip
  compilation entirely
- Ships per-model YAML + `.alls` configs so standard architectures "just work"
- Limitation: only models present in the Zoo's YAML catalog are supported by
  `hailomz compile <name>`

```bash
hailomz compile <name> --ckpt model.onnx --hw-arch hailo8
```

Works cleanly for: YOLOv5, YOLOv8 detection, ResNet, and other zoo-registered
architectures.

Does **not** work directly for: YOLOv11, YOLOv26, custom heads — you fall back
to DFC for those.

## Hailo Dataflow Compiler (DFC)

The actual compiler engine. Model Zoo calls it under the hood. Use directly
when:

- The model isn't in the Zoo
- You need to specify custom start/end nodes
- You need a custom `.alls` script
- The Zoo's default config doesn't fit

Steps: parse → optimize → compile (see [01-workflow.md](01-workflow.md)).

## Why no universal converter exists

Hailo compilation needs:

- Model-specific graph info (input/output node names)
- A calibration dataset
- Optimization scripts (`.alls`)
- Sometimes manual graph trimming for unsupported ops

These can't be inferred from an ONNX file alone, which is why a generic
`onnx → hef` CLI doesn't exist.

## Practical recommendation

Three levels of effort:

| Level | Tool | When |
|-------|------|------|
| Almost automatic | `hailomz compile <name>` | Standard arch in the Zoo |
| Semi-manual | `hailomz` + custom YAML / calibration | Zoo-adjacent arch |
| Fully manual | DFC directly + custom `.alls` | Custom / new architectures |

Start with Model Zoo. Drop to DFC only when it fails.

## Benchy's role

Benchy is a runtime / demo framework. It:

- Consumes `.hef`
- Provides input pipelines (camera, etc.)
- Does **not** compile models

Benchy may ship a wrapper around the same DFC calls (e.g. a
`benchmark run yolo --backend hailo --force-recompile` pipeline that does
`.pt → .onnx → .har → .hef`). That wrapper inherits all the underlying
compiler limitations — it isn't a magical converter.
