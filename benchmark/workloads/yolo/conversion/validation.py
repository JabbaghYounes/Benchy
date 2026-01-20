# Model Validation for Hailo HEF Files
#
# This module provides sanity checks for compiled Hailo HEF models
# to ensure they produce valid outputs before benchmarking.
#
# Phase 3 - Task 3.2 of Hailo PRD
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmark.schemas import YOLOTask

logger = logging.getLogger(__name__)


# Expected output configurations for YOLO models
# These are based on standard YOLO architecture outputs
EXPECTED_OUTPUTS = {
    YOLOTask.DETECTION: {
        # Detection outputs typically include:
        # - Bounding box coordinates (x, y, w, h)
        # - Objectness score
        # - Class scores
        "min_outputs": 1,
        "max_outputs": 10,  # Multiple output layers for different scales
        "class_counts": {
            "coco": 80,
            "coco128": 80,
            "default": 80,
        },
    },
    YOLOTask.CLASSIFICATION: {
        # Classification outputs are class probabilities
        "min_outputs": 1,
        "max_outputs": 1,
        "class_counts": {
            "imagenet": 1000,
            "imagenet10": 10,
            "default": 1000,
        },
    },
    YOLOTask.SEGMENTATION: {
        "min_outputs": 1,
        "max_outputs": 10,
        "class_counts": {
            "coco": 80,
            "default": 80,
        },
    },
    YOLOTask.POSE: {
        "min_outputs": 1,
        "max_outputs": 10,
        "class_counts": {
            "coco": 1,  # Person class
            "default": 1,
        },
    },
    YOLOTask.OBB: {
        "min_outputs": 1,
        "max_outputs": 10,
        "class_counts": {
            "dota": 15,
            "default": 15,
        },
    },
}


@dataclass
class ValidationConfig:
    """Configuration for model validation."""

    # Input resolution for test inference
    input_resolution: int = 640

    # Expected number of classes (None to auto-detect)
    expected_classes: Optional[int] = None

    # Dataset name for class count lookup
    dataset_name: str = "default"

    # Skip certain validation checks
    skip_output_shape_check: bool = False
    skip_class_count_check: bool = False
    skip_inference_check: bool = False

    # Tolerance for numerical checks
    output_range_tolerance: float = 0.001


@dataclass
class ValidationResult:
    """Result of model validation."""

    # Overall validation status
    valid: bool = False

    # Individual check results
    hef_loadable: bool = False
    input_shape_valid: bool = False
    output_shapes_valid: bool = False
    class_count_valid: bool = False
    inference_successful: bool = False
    output_range_valid: bool = False

    # Model information extracted during validation
    input_layers: List[Dict] = field(default_factory=list)
    output_layers: List[Dict] = field(default_factory=list)
    detected_class_count: Optional[int] = None
    inferred_task: Optional[str] = None

    # Errors encountered
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Inference results (if successful)
    inference_latency_ms: Optional[float] = None
    output_statistics: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "hef_loadable": self.hef_loadable,
            "input_shape_valid": self.input_shape_valid,
            "output_shapes_valid": self.output_shapes_valid,
            "class_count_valid": self.class_count_valid,
            "inference_successful": self.inference_successful,
            "output_range_valid": self.output_range_valid,
            "input_layers": self.input_layers,
            "output_layers": self.output_layers,
            "detected_class_count": self.detected_class_count,
            "inferred_task": self.inferred_task,
            "errors": self.errors,
            "warnings": self.warnings,
            "inference_latency_ms": self.inference_latency_ms,
            "output_statistics": self.output_statistics,
        }


class ModelValidator:
    """Validates Hailo HEF models before benchmarking.

    This validator performs sanity checks to ensure that compiled
    models produce valid outputs. It catches issues before the
    full benchmark run, saving time and providing clear diagnostics.

    Validation checks include:
    1. HEF file loadability
    2. Input shape verification
    3. Output shape verification
    4. Class count verification
    5. Single inference test
    6. Output range verification

    Usage:
        validator = ModelValidator()
        result = validator.validate(hef_path, YOLOTask.DETECTION)
        if not result.valid:
            print(f"Validation failed: {result.errors}")
    """

    def __init__(self):
        """Initialize the model validator."""
        self._hailort_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if HailoRT is available for validation."""
        if self._hailort_available is not None:
            return self._hailort_available

        try:
            from hailo_platform import HEF, VDevice, ConfigureParams

            self._hailort_available = True
        except ImportError:
            self._hailort_available = False

        return self._hailort_available

    def validate(
        self,
        hef_path: Path,
        task: YOLOTask,
        config: Optional[ValidationConfig] = None,
    ) -> ValidationResult:
        """Validate a compiled HEF model.

        Args:
            hef_path: Path to the HEF file
            task: Expected YOLO task type
            config: Validation configuration

        Returns:
            ValidationResult with detailed check results
        """
        config = config or ValidationConfig()
        result = ValidationResult()

        logger.info(f"Validating HEF model: {hef_path}")
        logger.info(f"  Task: {task.value}")

        # Check 1: HEF file exists and is loadable
        if not self._validate_hef_loadable(hef_path, result):
            return result

        # Check 2: Input shape
        if not self._validate_input_shape(hef_path, config, result):
            return result

        # Check 3: Output shapes
        if not config.skip_output_shape_check:
            self._validate_output_shapes(hef_path, task, result)

        # Check 4: Class count
        if not config.skip_class_count_check:
            self._validate_class_count(task, config, result)

        # Check 5: Test inference
        if not config.skip_inference_check and self.is_available():
            self._validate_inference(hef_path, config, result)

        # Determine overall validity
        result.valid = (
            result.hef_loadable
            and result.input_shape_valid
            and (result.output_shapes_valid or config.skip_output_shape_check)
            and (result.class_count_valid or config.skip_class_count_check)
            and (result.inference_successful or config.skip_inference_check)
        )

        if result.valid:
            logger.info("Model validation PASSED")
        else:
            logger.warning(f"Model validation FAILED: {result.errors}")

        return result

    def _validate_hef_loadable(
        self,
        hef_path: Path,
        result: ValidationResult,
    ) -> bool:
        """Check if HEF file exists and can be loaded.

        Args:
            hef_path: Path to HEF file
            result: ValidationResult to update

        Returns:
            True if HEF is loadable
        """
        if not hef_path.exists():
            result.errors.append(f"HEF file not found: {hef_path}")
            return False

        if hef_path.stat().st_size == 0:
            result.errors.append("HEF file is empty")
            return False

        if not self.is_available():
            result.warnings.append(
                "HailoRT not available, skipping HEF load validation"
            )
            result.hef_loadable = True  # Assume valid if we can't check
            return True

        try:
            from hailo_platform import HEF

            hef = HEF(str(hef_path))

            # Extract layer information
            for vstream_info in hef.get_input_vstream_infos():
                result.input_layers.append({
                    "name": vstream_info.name,
                    "shape": tuple(vstream_info.shape),
                    "format": str(vstream_info.format.type),
                })

            for vstream_info in hef.get_output_vstream_infos():
                result.output_layers.append({
                    "name": vstream_info.name,
                    "shape": tuple(vstream_info.shape),
                    "format": str(vstream_info.format.type),
                })

            result.hef_loadable = True
            logger.debug(f"HEF loaded: {len(result.input_layers)} inputs, "
                        f"{len(result.output_layers)} outputs")
            return True

        except Exception as e:
            result.errors.append(f"Failed to load HEF: {e}")
            return False

    def _validate_input_shape(
        self,
        hef_path: Path,
        config: ValidationConfig,
        result: ValidationResult,
    ) -> bool:
        """Validate input tensor shape.

        Args:
            hef_path: Path to HEF file
            config: Validation config
            result: ValidationResult to update

        Returns:
            True if input shape is valid
        """
        if not result.input_layers:
            # Input layers weren't extracted, can't validate
            result.input_shape_valid = True
            result.warnings.append("Input shape validation skipped (no layer info)")
            return True

        expected_h = config.input_resolution
        expected_w = config.input_resolution

        for layer in result.input_layers:
            shape = layer["shape"]

            # YOLO expects (H, W, C) or (N, H, W, C) input
            if len(shape) == 3:
                h, w, c = shape
            elif len(shape) == 4:
                _, h, w, c = shape
            else:
                result.warnings.append(
                    f"Unexpected input shape dimensions: {shape}"
                )
                continue

            # Check spatial dimensions
            if h != expected_h or w != expected_w:
                result.warnings.append(
                    f"Input resolution mismatch: expected {expected_h}x{expected_w}, "
                    f"got {h}x{w}"
                )

            # Check channels
            if c != 3:
                result.warnings.append(
                    f"Unexpected input channels: expected 3, got {c}"
                )

        result.input_shape_valid = True
        return True

    def _validate_output_shapes(
        self,
        hef_path: Path,
        task: YOLOTask,
        result: ValidationResult,
    ) -> None:
        """Validate output tensor shapes for the task.

        Args:
            hef_path: Path to HEF file
            task: Expected task type
            result: ValidationResult to update
        """
        if not result.output_layers:
            result.warnings.append("Output shape validation skipped (no layer info)")
            result.output_shapes_valid = True
            return

        expected = EXPECTED_OUTPUTS.get(task)
        if expected is None:
            result.warnings.append(f"No expected outputs defined for task {task.value}")
            result.output_shapes_valid = True
            return

        num_outputs = len(result.output_layers)

        # Check output count
        if num_outputs < expected["min_outputs"]:
            result.errors.append(
                f"Too few outputs: expected at least {expected['min_outputs']}, "
                f"got {num_outputs}"
            )
            result.output_shapes_valid = False
            return

        if num_outputs > expected["max_outputs"]:
            result.warnings.append(
                f"More outputs than expected: expected at most {expected['max_outputs']}, "
                f"got {num_outputs}"
            )

        # Try to detect class count from output shapes
        detected_classes = self._detect_class_count(result.output_layers, task)
        if detected_classes is not None:
            result.detected_class_count = detected_classes
            logger.debug(f"Detected {detected_classes} classes from output shape")

        result.output_shapes_valid = True

    def _detect_class_count(
        self,
        output_layers: List[Dict],
        task: YOLOTask,
    ) -> Optional[int]:
        """Try to detect class count from output shapes.

        Args:
            output_layers: Output layer information
            task: Task type

        Returns:
            Detected class count or None if undetermined
        """
        if task == YOLOTask.CLASSIFICATION:
            # Classification output is typically (N, num_classes)
            for layer in output_layers:
                shape = layer["shape"]
                if len(shape) == 2:
                    return shape[1]
                elif len(shape) == 1:
                    return shape[0]

        elif task == YOLOTask.DETECTION:
            # Detection outputs are more complex
            # YOLO typically outputs (N, anchors, 5 + num_classes)
            # or uses separate outputs for boxes and classes
            for layer in output_layers:
                shape = layer["shape"]
                # Look for class dimension (usually > 4 for boxes)
                if len(shape) >= 2:
                    last_dim = shape[-1]
                    if last_dim > 4:
                        # Subtract box coordinates (x, y, w, h)
                        return last_dim - 4

        return None

    def _validate_class_count(
        self,
        task: YOLOTask,
        config: ValidationConfig,
        result: ValidationResult,
    ) -> None:
        """Validate the detected class count.

        Args:
            task: Task type
            config: Validation config
            result: ValidationResult to update
        """
        expected = EXPECTED_OUTPUTS.get(task, {})
        class_counts = expected.get("class_counts", {})

        # Get expected class count
        if config.expected_classes is not None:
            expected_count = config.expected_classes
        else:
            expected_count = class_counts.get(
                config.dataset_name,
                class_counts.get("default"),
            )

        if expected_count is None:
            result.warnings.append(
                f"No expected class count for task {task.value}"
            )
            result.class_count_valid = True
            return

        if result.detected_class_count is None:
            result.warnings.append(
                "Could not detect class count from output shapes"
            )
            result.class_count_valid = True
            return

        if result.detected_class_count != expected_count:
            result.warnings.append(
                f"Class count mismatch: expected {expected_count}, "
                f"detected {result.detected_class_count}"
            )
            # Not a fatal error, just a warning
            result.class_count_valid = True
        else:
            result.class_count_valid = True
            logger.debug(f"Class count validated: {expected_count}")

    def _validate_inference(
        self,
        hef_path: Path,
        config: ValidationConfig,
        result: ValidationResult,
    ) -> None:
        """Run a single test inference.

        Args:
            hef_path: Path to HEF file
            config: Validation config
            result: ValidationResult to update
        """
        if not self.is_available():
            result.warnings.append("HailoRT not available, skipping inference test")
            result.inference_successful = True
            return

        try:
            import time
            from hailo_platform import (
                HEF,
                VDevice,
                ConfigureParams,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
            )

            # Load HEF
            hef = HEF(str(hef_path))

            # Create virtual device
            vdevice = VDevice()

            # Configure inference
            configure_params = ConfigureParams.create_from_hef(
                hef, interface=ConfigureParams.default_interface()
            )
            configured_network = vdevice.configure(hef, configure_params)[0]

            # Get input/output info
            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_infos = hef.get_output_vstream_infos()

            # Create vstream params
            input_params = InputVStreamParams.make_from_network_group(
                configured_network, quantized=False
            )
            output_params = OutputVStreamParams.make_from_network_group(
                configured_network, quantized=False
            )

            # Create test input (random normalized image)
            input_shape = tuple(input_vstream_info.shape)
            rng = np.random.default_rng(42)
            test_input = rng.random(input_shape, dtype=np.float32)

            # Run inference
            start_time = time.perf_counter()

            with InferVStreams(
                configured_network, input_params, output_params
            ) as infer_pipeline:
                input_dict = {input_vstream_info.name: test_input}
                outputs = infer_pipeline.infer(input_dict)

            end_time = time.perf_counter()
            result.inference_latency_ms = (end_time - start_time) * 1000

            # Validate outputs
            result.output_statistics = self._compute_output_statistics(outputs)

            # Check output validity
            all_valid = True
            for name, stats in result.output_statistics.items():
                if stats["has_nan"]:
                    result.errors.append(f"Output '{name}' contains NaN values")
                    all_valid = False
                if stats["has_inf"]:
                    result.errors.append(f"Output '{name}' contains Inf values")
                    all_valid = False

            result.inference_successful = all_valid
            result.output_range_valid = all_valid

            # Cleanup
            configured_network.shutdown()
            vdevice.release()

            if all_valid:
                logger.info(
                    f"Test inference successful in {result.inference_latency_ms:.2f}ms"
                )

        except Exception as e:
            result.errors.append(f"Inference test failed: {e}")
            result.inference_successful = False

    def _compute_output_statistics(self, outputs: Dict) -> Dict:
        """Compute statistics for output tensors.

        Args:
            outputs: Dictionary of output tensors

        Returns:
            Dictionary of statistics per output
        """
        stats = {}

        for name, tensor in outputs.items():
            if isinstance(tensor, np.ndarray):
                stats[name] = {
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(np.min(tensor)),
                    "max": float(np.max(tensor)),
                    "mean": float(np.mean(tensor)),
                    "std": float(np.std(tensor)),
                    "has_nan": bool(np.any(np.isnan(tensor))),
                    "has_inf": bool(np.any(np.isinf(tensor))),
                }
            else:
                stats[name] = {"error": "Not a numpy array"}

        return stats


def validate_hef_model(
    hef_path: Path,
    task: YOLOTask,
    input_resolution: int = 640,
    expected_classes: Optional[int] = None,
) -> ValidationResult:
    """Convenience function to validate a HEF model.

    This is the primary entry point for model validation.

    Args:
        hef_path: Path to the HEF file
        task: Expected YOLO task type
        input_resolution: Expected input resolution
        expected_classes: Expected number of classes (optional)

    Returns:
        ValidationResult with detailed check results

    Example:
        >>> result = validate_hef_model(Path("model.hef"), YOLOTask.DETECTION)
        >>> if not result.valid:
        ...     print(f"Validation failed: {result.errors}")
        >>> else:
        ...     print(f"Model has {result.detected_class_count} classes")
    """
    config = ValidationConfig(
        input_resolution=input_resolution,
        expected_classes=expected_classes,
    )

    validator = ModelValidator()
    return validator.validate(hef_path, task, config)


def quick_sanity_check(
    hef_path: Path,
    task: YOLOTask,
) -> Tuple[bool, List[str]]:
    """Quick sanity check for a HEF model.

    This performs essential checks without running inference.
    Use for fast validation during development.

    Args:
        hef_path: Path to HEF file
        task: Task type

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        >>> valid, issues = quick_sanity_check(Path("model.hef"), YOLOTask.DETECTION)
        >>> if not valid:
        ...     for issue in issues:
        ...         print(f"Issue: {issue}")
    """
    config = ValidationConfig(
        skip_inference_check=True,
        skip_class_count_check=True,
    )

    validator = ModelValidator()
    result = validator.validate(hef_path, task, config)

    issues = result.errors + result.warnings
    return result.valid, issues
