# Model Conversion Pipeline for Hailo NPU
#
# Orchestrates the full conversion pipeline:
#   .pt (PyTorch) → .onnx (ONNX) → .har (Hailo Archive) → .hef (Hailo Executable)
#
# Phase 3: Includes calibration dataset handling and model validation
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.cache import (
    ModelCache,
    CacheMetadata,
    get_cache_path,
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
from benchmark.workloads.yolo.conversion.calibration import (
    CalibrationDatasetLoader,
    CalibrationConfig,
    CalibrationDataset,
)
from benchmark.workloads.yolo.conversion.validation import (
    ModelValidator,
    ValidationConfig,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """Exception raised during model conversion."""

    def __init__(self, message: str, stage: str, details: Optional[str] = None):
        self.stage = stage
        self.details = details
        super().__init__(f"[{stage}] {message}")


@dataclass
class ConversionConfig:
    """Configuration for the full conversion pipeline."""

    # Target device
    target_device: str = "hailo8l"  # hailo8 or hailo8l

    # Model settings
    input_resolution: int = 640
    batch_size: int = 1

    # ONNX export settings
    onnx_opset: int = 11
    onnx_simplify: bool = True

    # Calibration settings (Phase 3)
    calibration_data_path: Optional[Path] = None
    calibration_set_size: int = 100  # Default to 100 images per Phase 3 PRD
    use_ultralytics_calibration: bool = True  # Use Ultralytics datasets
    calibration_seed: int = 42  # Seed for deterministic ordering

    # Compilation settings
    optimization_level: int = 2

    # Cache settings
    cache_dir: Optional[Path] = None
    force_recompile: bool = False

    # Pipeline control
    skip_onnx: bool = False  # Skip ONNX export if already exists
    skip_har: bool = False  # Skip HAR generation if already exists
    stop_at_onnx: bool = False  # Stop after ONNX export
    stop_at_har: bool = False  # Stop after HAR generation

    # Validation settings (Phase 3)
    validate_after_compile: bool = True  # Run sanity check after compilation
    skip_inference_validation: bool = False  # Skip test inference during validation


@dataclass
class ConversionResult:
    """Result of a model conversion."""

    success: bool
    model_name: str
    yolo_version: str
    task: str
    target_device: str

    # Output paths
    onnx_path: Optional[Path] = None
    har_path: Optional[Path] = None
    hef_path: Optional[Path] = None

    # Timing
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    onnx_time_seconds: Optional[float] = None
    har_time_seconds: Optional[float] = None
    hef_time_seconds: Optional[float] = None
    total_time_seconds: Optional[float] = None

    # Error info
    error: Optional[str] = None
    error_stage: Optional[str] = None

    # Metadata
    metadata: Optional[CacheMetadata] = None

    # Phase 3: Calibration info
    calibration_images: int = 0
    calibration_hash: Optional[str] = None
    calibration_seed: int = 42

    # Phase 3: Validation info
    validation_passed: bool = False
    validation_result: Optional[ValidationResult] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "model_name": self.model_name,
            "yolo_version": self.yolo_version,
            "task": self.task,
            "target_device": self.target_device,
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
            "har_path": str(self.har_path) if self.har_path else None,
            "hef_path": str(self.hef_path) if self.hef_path else None,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "onnx_time_seconds": self.onnx_time_seconds,
            "har_time_seconds": self.har_time_seconds,
            "hef_time_seconds": self.hef_time_seconds,
            "total_time_seconds": self.total_time_seconds,
            "error": self.error,
            "error_stage": self.error_stage,
            # Phase 3 fields
            "calibration_images": self.calibration_images,
            "calibration_hash": self.calibration_hash,
            "calibration_seed": self.calibration_seed,
            "validation_passed": self.validation_passed,
        }


class ModelConversionPipeline:
    """Orchestrates the full model conversion pipeline.

    This class manages the conversion of YOLO models from PyTorch format
    to Hailo HEF format through the following stages:

    1. ONNX Export (.pt → .onnx)
       - Uses Ultralytics export
       - Enforces static shapes and compatible opset

    2. HAR Generation (.onnx → .har)
       - Uses Hailo SDK to parse ONNX
       - Creates Hailo Archive

    3. HEF Compilation (.har → .hef)
       - Uses Hailo Dataflow Compiler
       - Requires calibration data for quantization
       - Targets specific Hailo device (8 or 8L)

    All intermediate artifacts are cached for reuse.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the conversion pipeline.

        Args:
            cache_dir: Directory for caching artifacts
        """
        self.cache = ModelCache(cache_dir)
        self.onnx_exporter = ONNXExporter(self.cache)
        self.har_generator = HARGenerator(self.cache)
        self.hef_compiler = HEFCompiler(self.cache)

        # Phase 3: Calibration and validation components
        self.calibration_loader = CalibrationDatasetLoader()
        self.validator = ModelValidator()

    def check_requirements(self) -> dict:
        """Check if all requirements are available.

        Returns:
            Dictionary with availability status for each stage
        """
        return {
            "onnx_export": self.onnx_exporter.is_available(),
            "har_generation": self.har_generator.is_available(),
            "hef_compilation": self.hef_compiler.is_available(),
            "model_zoo": self.har_generator.is_model_zoo_available(),
            # Phase 3
            "calibration_datasets": self.calibration_loader.is_available(),
            "model_validation": self.validator.is_available(),
        }

    def convert(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: Optional[ConversionConfig] = None,
    ) -> ConversionResult:
        """Run the full conversion pipeline.

        Args:
            model_name: YOLO model name (e.g., "yolov8n.pt")
            yolo_version: YOLO version (e.g., "v8")
            task: Task type
            config: Conversion configuration

        Returns:
            ConversionResult with paths and status
        """
        import time

        config = config or ConversionConfig()
        start_time = time.time()

        result = ConversionResult(
            success=False,
            model_name=model_name,
            yolo_version=yolo_version,
            task=task.value,
            target_device=config.target_device,
        )

        logger.info(f"Starting conversion pipeline for {model_name}")
        logger.info(f"  YOLO version: {yolo_version}")
        logger.info(f"  Task: {task.value}")
        logger.info(f"  Target: {config.target_device}")

        # Check for valid cache first
        if not config.force_recompile:
            if self.cache.has_valid_cache(
                model_name, yolo_version, task,
                target_device=config.target_device,
                input_resolution=config.input_resolution,
            ):
                hef_path = self.cache.get_hef_path(model_name, yolo_version, task)
                result.success = True
                result.hef_path = hef_path
                result.onnx_path = self.cache.get_onnx_path(model_name, yolo_version, task)
                result.har_path = self.cache.get_har_path(model_name, yolo_version, task)
                result.completed_at = datetime.now().isoformat()
                result.total_time_seconds = time.time() - start_time
                result.metadata = self.cache.get_metadata(model_name, yolo_version, task)
                logger.info("Using cached HEF file")
                return result

        try:
            # Stage 1: ONNX Export
            onnx_start = time.time()
            result.onnx_path = self._run_onnx_export(
                model_name, yolo_version, task, config
            )
            result.onnx_time_seconds = time.time() - onnx_start

            if config.stop_at_onnx:
                result.success = True
                result.completed_at = datetime.now().isoformat()
                result.total_time_seconds = time.time() - start_time
                return result

            # Stage 2: HAR Generation
            har_start = time.time()
            result.har_path = self._run_har_generation(
                result.onnx_path, model_name, yolo_version, task, config
            )
            result.har_time_seconds = time.time() - har_start

            if config.stop_at_har:
                result.success = True
                result.completed_at = datetime.now().isoformat()
                result.total_time_seconds = time.time() - start_time
                return result

            # Stage 3: HEF Compilation
            hef_start = time.time()
            result.hef_path = self._run_hef_compilation(
                result.har_path, model_name, yolo_version, task, config
            )
            result.hef_time_seconds = time.time() - hef_start

            # Phase 3: Model validation (sanity check)
            if config.validate_after_compile:
                logger.info("Phase 3: Running model validation...")
                validation_result = self._run_validation(
                    result.hef_path, task, config
                )
                result.validation_result = validation_result
                result.validation_passed = validation_result.valid

                if not validation_result.valid:
                    raise ConversionError(
                        f"Model validation failed: {'; '.join(validation_result.errors)}",
                        stage="validation",
                    )

                logger.info("Model validation PASSED")
            else:
                result.validation_passed = True  # Assume valid if skipped

            # Store calibration info in result
            result.calibration_images = config.calibration_set_size
            result.calibration_seed = config.calibration_seed

            # Success
            result.success = True
            result.completed_at = datetime.now().isoformat()
            result.total_time_seconds = time.time() - start_time
            result.metadata = self.cache.get_metadata(model_name, yolo_version, task)

            logger.info(f"Conversion completed successfully in {result.total_time_seconds:.1f}s")
            logger.info(f"  ONNX: {result.onnx_time_seconds:.1f}s")
            logger.info(f"  HAR:  {result.har_time_seconds:.1f}s")
            logger.info(f"  HEF:  {result.hef_time_seconds:.1f}s")

            return result

        except ConversionError as e:
            result.error = str(e)
            result.error_stage = e.stage
            result.completed_at = datetime.now().isoformat()
            result.total_time_seconds = time.time() - start_time
            logger.error(f"Conversion failed at {e.stage}: {e}")
            return result

        except Exception as e:
            result.error = str(e)
            result.error_stage = "unknown"
            result.completed_at = datetime.now().isoformat()
            result.total_time_seconds = time.time() - start_time
            logger.error(f"Conversion failed: {e}")
            return result

    def _run_onnx_export(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: ConversionConfig,
    ) -> Path:
        """Run ONNX export stage.

        Args:
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            config: Pipeline config

        Returns:
            Path to ONNX file
        """
        logger.info("Stage 1/3: ONNX Export")

        if not self.onnx_exporter.is_available():
            raise ConversionError(
                "Ultralytics not available for ONNX export",
                stage="onnx_export",
                details="Install with: pip install ultralytics onnx",
            )

        # Check compatibility
        compat = self.onnx_exporter.estimate_onnx_compatibility(model_name, task)
        if not compat["likely_compatible"]:
            raise ConversionError(
                f"Model/task not compatible with Hailo: {compat['errors']}",
                stage="onnx_export",
            )

        for warning in compat.get("warnings", []):
            logger.warning(f"ONNX compatibility warning: {warning}")

        # Configure export
        onnx_config = ONNXExportConfig(
            input_resolution=config.input_resolution,
            batch_size=config.batch_size,
            opset_version=config.onnx_opset,
            simplify=config.onnx_simplify,
        )

        # Export
        try:
            onnx_path = self.onnx_exporter.export(
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                config=onnx_config,
                force=config.force_recompile and not config.skip_onnx,
            )
            logger.info(f"  ONNX exported: {onnx_path}")
            return onnx_path

        except Exception as e:
            raise ConversionError(
                f"ONNX export failed: {e}",
                stage="onnx_export",
            ) from e

    def _run_har_generation(
        self,
        onnx_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: ConversionConfig,
    ) -> Path:
        """Run HAR generation stage.

        Args:
            onnx_path: Path to ONNX file
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            config: Pipeline config

        Returns:
            Path to HAR file
        """
        logger.info("Stage 2/3: HAR Generation")

        if not self.har_generator.is_available():
            raise ConversionError(
                "Hailo SDK not available for HAR generation",
                stage="har_generation",
                details="Install the Hailo SDK from https://hailo.ai/developer-zone/",
            )

        # Check ONNX compatibility
        compat = self.har_generator.check_onnx_compatibility(onnx_path)
        if not compat["compatible"]:
            raise ConversionError(
                f"ONNX not compatible with Hailo parser: {compat['errors']}",
                stage="har_generation",
            )

        for warning in compat.get("warnings", []):
            logger.warning(f"HAR compatibility warning: {warning}")

        # Configure generator
        har_config = HARGeneratorConfig(
            target_device=config.target_device,
            input_resolution=config.input_resolution,
            batch_size=config.batch_size,
        )

        # Generate
        try:
            har_path = self.har_generator.generate(
                onnx_path=onnx_path,
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                config=har_config,
                force=config.force_recompile and not config.skip_har,
            )
            logger.info(f"  HAR generated: {har_path}")
            return har_path

        except Exception as e:
            raise ConversionError(
                f"HAR generation failed: {e}",
                stage="har_generation",
            ) from e

    def _run_hef_compilation(
        self,
        har_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: ConversionConfig,
    ) -> Path:
        """Run HEF compilation stage.

        Args:
            har_path: Path to HAR file
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            config: Pipeline config

        Returns:
            Path to HEF file
        """
        logger.info("Stage 3/3: HEF Compilation")

        if not self.hef_compiler.is_available():
            raise ConversionError(
                "Hailo Dataflow Compiler not available",
                stage="hef_compilation",
                details="Install the Hailo SDK from https://hailo.ai/developer-zone/",
            )

        # Configure compiler with Phase 3 calibration settings
        hef_config = HEFCompilerConfig(
            target_device=config.target_device,
            optimization_level=config.optimization_level,
            calibration_data_path=config.calibration_data_path,
            calibration_set_size=config.calibration_set_size,
            use_ultralytics_dataset=config.use_ultralytics_calibration,
            calibration_seed=config.calibration_seed,
        )

        # Compile
        try:
            hef_path = self.hef_compiler.compile(
                har_path=har_path,
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                config=hef_config,
                force=config.force_recompile,
            )
            logger.info(f"  HEF compiled: {hef_path}")
            return hef_path

        except Exception as e:
            raise ConversionError(
                f"HEF compilation failed: {e}",
                stage="hef_compilation",
            ) from e

    def _run_validation(
        self,
        hef_path: Path,
        task: YOLOTask,
        config: ConversionConfig,
    ) -> ValidationResult:
        """Run Phase 3 model validation (sanity check).

        This validates the compiled HEF model before it's used for benchmarking:
        - Verifies the HEF file is loadable
        - Validates input tensor shapes match expected resolution
        - Validates output tensor shapes for the task type
        - Checks class count compatibility
        - Optionally runs a single test inference

        Args:
            hef_path: Path to the compiled HEF file
            task: YOLO task type
            config: Pipeline configuration

        Returns:
            ValidationResult with detailed check results
        """
        logger.info("Phase 3: Model Validation (Sanity Check)")

        validation_config = ValidationConfig(
            input_resolution=config.input_resolution,
            skip_inference_check=config.skip_inference_validation,
        )

        result = self.validator.validate(hef_path, task, validation_config)

        # Log validation results
        if result.valid:
            logger.info("  Validation PASSED")
            logger.info(f"    Input layers: {len(result.input_layers)}")
            logger.info(f"    Output layers: {len(result.output_layers)}")
            if result.detected_class_count is not None:
                logger.info(f"    Detected classes: {result.detected_class_count}")
            if result.inference_latency_ms is not None:
                logger.info(f"    Test inference: {result.inference_latency_ms:.2f}ms")
        else:
            logger.error("  Validation FAILED")
            for error in result.errors:
                logger.error(f"    Error: {error}")

        for warning in result.warnings:
            logger.warning(f"    Warning: {warning}")

        return result

    def get_cached_hef(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        target_device: str = "hailo8l",
        input_resolution: int = 640,
    ) -> Optional[Path]:
        """Get cached HEF if available and valid.

        Args:
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            target_device: Target device
            input_resolution: Input resolution

        Returns:
            Path to HEF file if valid cache exists, None otherwise
        """
        if self.cache.has_valid_cache(
            model_name, yolo_version, task,
            target_device=target_device,
            input_resolution=input_resolution,
        ):
            return self.cache.get_hef_path(model_name, yolo_version, task)
        return None

    def clear_cache(
        self,
        model_name: Optional[str] = None,
        yolo_version: Optional[str] = None,
        task: Optional[YOLOTask] = None,
    ) -> None:
        """Clear cached artifacts.

        Args:
            model_name: Model name (None to clear all)
            yolo_version: YOLO version
            task: Task type
        """
        if model_name and yolo_version and task:
            self.cache.clear_cache(model_name, yolo_version, task)
        else:
            self.cache.clear_all()

    def list_cached_models(self) -> list:
        """List all cached models.

        Returns:
            List of cached model info
        """
        return self.cache.list_cached_models()
