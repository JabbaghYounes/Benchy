# YOLO Model Conversion Pipeline for Hailo NPU
#
# This module handles the conversion of YOLO models from PyTorch format
# to Hailo-optimized HEF format through the following pipeline:
#
#   .pt (PyTorch) → .onnx (ONNX) → .har (Hailo Archive) → .hef (Hailo Executable)
#
# The pipeline supports:
# - YOLOv8, YOLOv11, YOLOv26
# - Detection and Classification tasks
# - Hailo-8 and Hailo-8L target devices
# - Deterministic caching with version-based invalidation
#
# Phase 3 additions:
# - Calibration dataset handling (deterministic Ultralytics dataset loading)
# - Model validation (sanity checks before benchmarking)

from benchmark.workloads.yolo.conversion.pipeline import (
    ModelConversionPipeline,
    ConversionConfig,
    ConversionResult,
    ConversionError,
)
from benchmark.workloads.yolo.conversion.onnx_export import (
    ONNXExporter,
    ONNXExportConfig,
)
from benchmark.workloads.yolo.conversion.har_generator import (
    HARGenerator,
    HARGeneratorConfig,
)
from benchmark.workloads.yolo.conversion.hef_compiler import (
    HEFCompiler,
    HEFCompilerConfig,
)
from benchmark.workloads.yolo.conversion.cache import (
    ModelCache,
    CacheMetadata,
    get_cache_path,
)
# Phase 3: Calibration
from benchmark.workloads.yolo.conversion.calibration import (
    CalibrationDatasetLoader,
    CalibrationConfig,
    CalibrationDataset,
    get_calibration_dataset,
)
# Phase 3: Validation
from benchmark.workloads.yolo.conversion.validation import (
    ModelValidator,
    ValidationConfig,
    ValidationResult,
    validate_hef_model,
    quick_sanity_check,
)

__all__ = [
    # Pipeline
    "ModelConversionPipeline",
    "ConversionConfig",
    "ConversionResult",
    "ConversionError",
    # ONNX Export
    "ONNXExporter",
    "ONNXExportConfig",
    # HAR Generation
    "HARGenerator",
    "HARGeneratorConfig",
    # HEF Compilation
    "HEFCompiler",
    "HEFCompilerConfig",
    # Caching
    "ModelCache",
    "CacheMetadata",
    "get_cache_path",
    # Phase 3: Calibration
    "CalibrationDatasetLoader",
    "CalibrationConfig",
    "CalibrationDataset",
    "get_calibration_dataset",
    # Phase 3: Validation
    "ModelValidator",
    "ValidationConfig",
    "ValidationResult",
    "validate_hef_model",
    "quick_sanity_check",
]
