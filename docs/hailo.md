# Hailo NPU Integration

The benchmark suite supports the Hailo NPUs shipped on Raspberry Pi AI HATs. The AI HAT+ ships in two variants (which Benchy treats as the same `RPI_AI_HAT_PLUS` platform — both use the same HailoRT 4.x driver, packages, and HEF format):

| Board | NPU | TOPS (peak) | Quantization | HailoRT | Onboard memory | Host |
|-------|-----|-------------|--------------|---------|----------------|------|
| Raspberry Pi AI HAT+ (13 TOPS) | Hailo-8L | 13 | INT8 | 4.x | none (uses host) | Pi 5 |
| Raspberry Pi AI HAT+ (26 TOPS) | Hailo-8  | 26 | INT8 | 4.x | none (uses host) | Pi 5 |
| Raspberry Pi AI HAT+ 2         | Hailo-10H | 40 | INT4 / INT8 | 5.x | 8 GB SDRAM | Pi 5 |

Notes:

- The 26 TOPS AI HAT+ automatically configures the Pi 5's PCIe link to Gen 3 to expose the full bandwidth.
- The Hailo-8 silicon has a typical power consumption of ~2.5 W; the AI HAT+ 2 ships with a heatsink for the Hailo-10H. All three boards conform to the Raspberry Pi HAT+ specification.
- Only the AI HAT+ 2 has onboard SDRAM, so it is the only HAT that can host local LLMs / VLMs (Raspberry Pi rates it for models up to ~6 B parameters); the AI HAT+ variants are vision-only.

HEF files compiled for the Hailo-8 family (8 / 8L) are not compatible with the Hailo-10H, and vice versa — the backend maintains separate caches per family.

References: [AI HAT+ product brief (PDF)](https://datasheets.raspberrypi.com/ai-hat/ai-hat-plus-product-brief.pdf), [Raspberry Pi AI HAT documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html), [Hailo-8 accelerator overview](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/).

## Supported Configurations

| YOLO Version | Detection | Classification | Segmentation | Pose | OBB |
|--------------|-----------|----------------|--------------|------|-----|
| v8 | Yes | Yes | No | No | No |
| v11 | Yes | Yes | No | No | No |
| v26 | Yes | Yes | No | No | No |

**Optimized Models:**
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` (Detection)
- `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt` (Classification)
- Similar patterns for v11 and v26

## Model Conversion Pipeline

Hailo requires model conversion from PyTorch to HEF format:

```
.pt (PyTorch) -> .onnx (ONNX) -> .har (Hailo Archive) -> .hef (Hailo Executable)
```

**Compilation is automatic** - models are compiled on first use and cached for subsequent runs.

## Known Limitations

1. **Supported Tasks Only**: Segmentation, pose estimation, and OBB tasks are NOT supported on Hailo NPU due to architectural constraints.
2. **INT8 Quantization**: All Hailo models use INT8 quantization. Minor accuracy differences compared to FP32/FP16 models are expected.
3. **Model Size**: Larger models (l, x variants) may have longer compilation times and higher memory requirements.
4. **No CPU Fallback**: When using `--backend hailo`, the benchmark will NOT fall back to CPU if Hailo is unavailable. This ensures benchmark integrity.

## Important: CPU Inference is Invalid

**CPU inference on Hailo-equipped platforms is NOT a valid benchmark configuration.**

When benchmarking Raspberry Pi with Hailo:
- Always use `--backend hailo` explicitly, OR
- Let auto-detection select Hailo (default on RPi + AI HAT+)
- CPU fallback is disabled by default to prevent misleading results

## Compilation Requirements

First-time model compilation requires:
- **HailoRT SDK** installed (version 4.17+ recommended)
- **Hailo Dataflow Compiler** for .har -> .hef conversion
- **Calibration data**: Uses 100 images from COCO validation set
- **Disk space**: ~500MB per compiled model
- **Time**: 5-30 minutes per model depending on size

## Cache Management

Compiled models are cached in `~/.cache/benchy/hailo/`:
```
~/.cache/benchy/hailo/
├── yolov8n_detection_640/
│   ├── model.onnx
│   ├── model.har
│   ├── model.hef
│   └── metadata.json
└── ...
```

To force recompilation:
```bash
python -m benchmark run yolo --backend hailo --force-recompile
```

## Cross-Platform Comparison

To compare Jetson (GPU) vs Raspberry Pi + Hailo (NPU):

```bash
# Run on Jetson
python -m benchmark run yolo --output results/jetson/

# Run on RPi + Hailo
python -m benchmark run yolo --backend hailo --output results/rpi_hailo/

# Generate comparison report
python -m benchmark verify results/jetson/bench_*.json results/rpi_hailo/bench_*.json
```

The verification report shows:
- Performance deltas (FPS, latency)
- Validation of fair comparison criteria
- Warnings for potentially misleading comparisons
