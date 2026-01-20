# Hailo NPU backend for YOLO inference
#
# Phase 3: Uses the validation module for model sanity checks
# Phase 4: Full HailoRT inference runner with proper post-processing and metrics
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

import numpy as np

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.backends.base import (
    YOLOBackend,
    BackendType,
    BackendCapabilities,
    ModelInfo,
    InferenceResult,
)
# Phase 3: Validation
from benchmark.workloads.yolo.conversion.validation import (
    ModelValidator,
    ValidationConfig,
    ValidationResult,
    validate_hef_model,
)
# Phase 4: Post-processing and metrics
from benchmark.workloads.yolo.postprocessing import (
    YOLOPostProcessor,
    PostProcessConfig,
    Detection,
    ClassificationResult,
    decode_yolo_output,
)
from benchmark.workloads.yolo.hailo_metrics import (
    HailoMetrics,
    HailoMetricsCollector,
    InferenceTimer,
)

logger = logging.getLogger(__name__)


class HailoDevice:
    """Enumeration of Hailo device types."""

    HAILO8 = "hailo8"
    HAILO8L = "hailo8l"


@dataclass
class HailoDeviceInfo:
    """Information about a detected Hailo device."""

    device_type: str  # "hailo8" or "hailo8l"
    device_id: str
    firmware_version: Optional[str] = None
    driver_version: Optional[str] = None
    device_path: Optional[str] = None


class HailoBackend(YOLOBackend):
    """Hailo NPU backend for YOLO inference.

    This backend uses the Hailo Runtime (HailoRT) to execute
    pre-compiled HEF models on Hailo-8 or Hailo-8L NPUs.

    The model preparation pipeline is:
    1. .pt (PyTorch) -> .onnx (ONNX export via Ultralytics)
    2. .onnx -> .har (Hailo Archive via Hailo Model Zoo)
    3. .har -> .hef (Hailo Executable via Dataflow Compiler)

    Only step 4 (runtime execution) happens in run_inference().
    Steps 1-3 happen in prepare_model() if no cached HEF exists.
    """

    # Supported YOLO versions and tasks for Hailo
    # Detection is the primary supported task in v1
    SUPPORTED_TASKS = [YOLOTask.DETECTION, YOLOTask.CLASSIFICATION]
    SUPPORTED_VERSIONS = ["v8", "v11", "v26"]

    # Model cache directory structure:
    # models/hailo/{yolo_version}/{task}/{model_name}/
    DEFAULT_CACHE_DIR = Path("models/hailo")

    def __init__(self, device: str = "0", cache_dir: Optional[Path] = None):
        """Initialize Hailo backend.

        Args:
            device: Device identifier (typically "0" for first Hailo device)
            cache_dir: Directory for caching compiled models
        """
        super().__init__(device)
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self._device_info: Optional[HailoDeviceInfo] = None
        self._hef_path: Optional[Path] = None
        self._hailo_available: Optional[bool] = None

        # HailoRT objects (will be initialized when model is loaded)
        self._hef = None
        self._vdevice = None
        self._infer_model = None
        self._configured_infer_model = None

        # Phase 3: Model validator for sanity checks
        self._validator = ModelValidator()
        self._last_validation_result: Optional[ValidationResult] = None

        # Phase 4: Post-processor and metrics collector
        self._postprocessor: Optional[YOLOPostProcessor] = None
        self._metrics_collector = HailoMetricsCollector()
        self._last_metrics: Optional[HailoMetrics] = None
        self._current_task: Optional[YOLOTask] = None
        self._input_resolution: int = 640

    @property
    def backend_type(self) -> BackendType:
        return BackendType.HAILO

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_tasks=self.SUPPORTED_TASKS,
            supported_yolo_versions=self.SUPPORTED_VERSIONS,
            supports_gpu=False,
            supports_npu=True,
            supports_quantization=True,  # Hailo uses INT8 quantization
            requires_compilation=True,
            max_batch_size=1,  # Hailo typically uses batch size 1
        )

    def is_available(self) -> bool:
        """Check if Hailo runtime is available."""
        if self._hailo_available is not None:
            return self._hailo_available

        # Check for HailoRT Python bindings
        try:
            from hailo_platform import HEF, VDevice, ConfigureParams
            self._hailo_available = True
            logger.debug("HailoRT Python bindings available")
        except ImportError:
            logger.debug("HailoRT Python bindings not found")
            self._hailo_available = False
            return False

        # Check for actual Hailo device
        if not self._detect_device():
            logger.warning("No Hailo device detected")
            self._hailo_available = False
            return False

        return self._hailo_available

    def _detect_device(self) -> bool:
        """Detect Hailo device and populate device info.

        Returns:
            True if a Hailo device was found
        """
        # Check for /dev/hailo* devices
        hailo_devices = list(Path("/dev").glob("hailo*"))
        if not hailo_devices:
            return False

        # Try to get device info via hailortcli
        try:
            result = subprocess.run(
                ["hailortcli", "fw-control", "identify"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout.lower()

                # Determine device type
                if "hailo8l" in output or "hailo-8l" in output:
                    device_type = HailoDevice.HAILO8L
                elif "hailo8" in output or "hailo-8" in output:
                    device_type = HailoDevice.HAILO8
                else:
                    device_type = HailoDevice.HAILO8L  # Default to 8L

                self._device_info = HailoDeviceInfo(
                    device_type=device_type,
                    device_id=str(hailo_devices[0]),
                    device_path=str(hailo_devices[0]),
                )

                # Try to extract firmware version
                for line in result.stdout.split("\n"):
                    if "firmware" in line.lower():
                        self._device_info.firmware_version = line.strip()
                        break

                logger.info(f"Detected Hailo device: {device_type}")
                return True

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback: assume Hailo-8L if device node exists
        self._device_info = HailoDeviceInfo(
            device_type=HailoDevice.HAILO8L,
            device_id=str(hailo_devices[0]),
            device_path=str(hailo_devices[0]),
        )
        return True

    def get_device_info(self) -> Optional[HailoDeviceInfo]:
        """Get information about the detected Hailo device."""
        return self._device_info

    def get_target_device(self) -> str:
        """Get the target device string for compilation.

        Returns:
            "hailo8" or "hailo8l"
        """
        if self._device_info:
            return self._device_info.device_type
        return HailoDevice.HAILO8L

    def _get_cache_path(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Path:
        """Get the cache directory path for a model.

        Args:
            model_name: Model name (e.g., "yolov8n")
            yolo_version: YOLO version
            task: Task type

        Returns:
            Path to the cache directory
        """
        # Remove file extension from model name
        base_name = Path(model_name).stem

        return self.cache_dir / yolo_version / task.value / base_name

    def _get_hef_path(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Path:
        """Get the expected HEF file path."""
        cache_path = self._get_cache_path(model_name, yolo_version, task)
        return cache_path / "model.hef"

    def prepare_model(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        input_resolution: int = 640,
        force_recompile: bool = False,
    ) -> ModelInfo:
        """Prepare a YOLO model for Hailo inference.

        This method will:
        1. Check for a cached HEF file
        2. If not found (or force_recompile), run the conversion pipeline
        3. Load the HEF into the Hailo runtime

        Args:
            model_name: YOLO model name
            yolo_version: YOLO version
            task: Task type
            input_resolution: Input resolution
            force_recompile: Force recompilation

        Returns:
            ModelInfo describing the prepared model
        """
        if not self.is_available():
            raise RuntimeError(
                "Hailo backend is not available. "
                "Ensure HailoRT is installed and a Hailo device is connected."
            )

        # Validate task support
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Task {task.value} is not supported by Hailo backend. "
                f"Supported tasks: {[t.value for t in self.SUPPORTED_TASKS]}"
            )

        # Check for cached HEF
        hef_path = self._get_hef_path(model_name, yolo_version, task)

        if hef_path.exists() and not force_recompile:
            logger.info(f"Using cached HEF: {hef_path}")
        else:
            # Run the conversion pipeline
            logger.info(f"HEF not found at {hef_path}, starting conversion pipeline...")
            hef_path = self._convert_model(
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                input_resolution=input_resolution,
            )

        # Phase 3: Run model sanity check before loading
        logger.info("Running model sanity check (Phase 3)...")
        validation_result = self._validate_model(hef_path, task, input_resolution)

        if not validation_result.valid:
            error_msg = "; ".join(validation_result.errors)
            raise RuntimeError(
                f"Model validation failed: {error_msg}. "
                f"HEF file may be corrupted or incompatible."
            )

        # Store validation result for later inspection
        self._last_validation_result = validation_result

        # Load the HEF
        self._load_hef(hef_path)
        self._hef_path = hef_path

        # Phase 4: Initialize post-processor for this task
        self._current_task = task
        self._input_resolution = input_resolution

        num_classes = validation_result.detected_class_count or 80
        postprocess_config = PostProcessConfig(
            input_width=input_resolution,
            input_height=input_resolution,
            num_classes=num_classes,
        )
        self._postprocessor = YOLOPostProcessor(task, postprocess_config)
        logger.info(f"Initialized post-processor for {task.value} with {num_classes} classes")

        # Create model info with validation details
        self._model_info = ModelInfo(
            name=model_name,
            version=yolo_version,
            task=task,
            input_resolution=input_resolution,
            backend=BackendType.HAILO,
            model_path=Path(model_name),
            compiled_path=hef_path,
            quantization="int8",
            metadata={
                "target_device": self.get_target_device(),
                "hef_path": str(hef_path),
                "hailort_version": self._get_hailort_version(),
                "validation_passed": True,
                "detected_class_count": validation_result.detected_class_count,
                "input_layers": len(validation_result.input_layers),
                "output_layers": len(validation_result.output_layers),
            },
        )

        return self._model_info

    def _convert_model(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        input_resolution: int,
    ) -> Path:
        """Run the full model conversion pipeline.

        Pipeline: .pt -> .onnx -> .har -> .hef

        Args:
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            input_resolution: Input resolution

        Returns:
            Path to the generated HEF file

        Raises:
            RuntimeError: If conversion fails
        """
        from benchmark.workloads.yolo.conversion import (
            ModelConversionPipeline,
            ConversionConfig,
            ConversionError,
        )

        logger.info(f"Starting model conversion for {model_name}")

        # Create pipeline
        pipeline = ModelConversionPipeline(cache_dir=self.cache_dir)

        # Check requirements
        requirements = pipeline.check_requirements()
        if not requirements["onnx_export"]:
            raise RuntimeError(
                "ONNX export not available. Install ultralytics: pip install ultralytics onnx"
            )

        # Configure conversion
        config = ConversionConfig(
            target_device=self.get_target_device(),
            input_resolution=input_resolution,
            batch_size=1,
            force_recompile=False,
        )

        # Check what stages are available
        if not requirements["har_generation"]:
            logger.warning(
                "Hailo SDK not available for HAR generation. "
                "Only ONNX export will be performed."
            )
            config.stop_at_onnx = True

        if not requirements["hef_compilation"]:
            logger.warning(
                "Hailo Dataflow Compiler not available for HEF compilation. "
                "Pipeline will stop at HAR."
            )
            if not config.stop_at_onnx:
                config.stop_at_har = True

        # Run conversion
        result = pipeline.convert(
            model_name=model_name,
            yolo_version=yolo_version,
            task=task,
            config=config,
        )

        if not result.success:
            raise RuntimeError(
                f"Model conversion failed at stage '{result.error_stage}': {result.error}"
            )

        # Check what we got
        if result.hef_path and result.hef_path.exists():
            logger.info(f"HEF file ready: {result.hef_path}")
            return result.hef_path
        elif result.har_path and result.har_path.exists():
            raise RuntimeError(
                f"HAR file generated but HEF compilation not available. "
                f"HAR file at: {result.har_path}. "
                f"Please compile manually using Hailo Dataflow Compiler."
            )
        elif result.onnx_path and result.onnx_path.exists():
            raise RuntimeError(
                f"ONNX file generated but Hailo SDK not available. "
                f"ONNX file at: {result.onnx_path}. "
                f"Please complete conversion using Hailo SDK."
            )
        else:
            raise RuntimeError("Conversion produced no output files")

    def _load_hef(self, hef_path: Path) -> None:
        """Load a HEF file into the Hailo runtime.

        Args:
            hef_path: Path to the HEF file
        """
        if not hef_path.exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")

        try:
            from hailo_platform import HEF, VDevice, ConfigureParams, FormatType

            logger.info(f"Loading HEF: {hef_path}")

            # Load the HEF
            self._hef = HEF(str(hef_path))

            # Create virtual device
            self._vdevice = VDevice()

            # Configure the inference model
            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=ConfigureParams.default_interface()
            )
            self._configured_infer_model = self._vdevice.configure(
                self._hef, configure_params
            )[0]

            logger.info("HEF loaded successfully")

        except ImportError as e:
            raise RuntimeError(f"HailoRT not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load HEF: {e}")

    def _validate_model(
        self,
        hef_path: Path,
        task: YOLOTask,
        input_resolution: int,
    ) -> ValidationResult:
        """Run Phase 3 model sanity check.

        This validates the HEF model before loading it for inference:
        - Verifies the HEF file is loadable
        - Validates input tensor shapes
        - Validates output tensor shapes
        - Checks class count compatibility
        - Runs a single test inference

        Args:
            hef_path: Path to the HEF file
            task: Expected YOLO task type
            input_resolution: Expected input resolution

        Returns:
            ValidationResult with detailed check results
        """
        config = ValidationConfig(
            input_resolution=input_resolution,
            skip_inference_check=False,  # We want to test inference
        )

        result = self._validator.validate(hef_path, task, config)

        # Log validation summary
        if result.valid:
            logger.info("Model validation PASSED")
            logger.info(f"  Input layers: {len(result.input_layers)}")
            logger.info(f"  Output layers: {len(result.output_layers)}")
            if result.detected_class_count is not None:
                logger.info(f"  Detected classes: {result.detected_class_count}")
            if result.inference_latency_ms is not None:
                logger.info(f"  Test inference: {result.inference_latency_ms:.2f}ms")
        else:
            logger.warning("Model validation FAILED")
            for error in result.errors:
                logger.error(f"  Error: {error}")

        for warning in result.warnings:
            logger.warning(f"  Warning: {warning}")

        return result

    def get_validation_result(self) -> Optional[ValidationResult]:
        """Get the last validation result.

        Returns:
            ValidationResult from the last prepare_model call, or None
        """
        return self._last_validation_result

    def run_inference(
        self,
        input_data: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> InferenceResult:
        """Run inference on the Hailo NPU.

        Phase 4 implementation:
        - Executes inference fully on Hailo NPU
        - Applies proper post-processing (NMS for detection)
        - Collects comprehensive metrics

        Args:
            input_data: Input image as numpy array (H, W, C) or (B, H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Returns:
            InferenceResult with timing, outputs, and detections

        Raises:
            RuntimeError: If model is not prepared or inference fails
        """
        if self._configured_infer_model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        if self._postprocessor is None:
            raise RuntimeError("Post-processor not initialized. Call prepare_model() first.")

        try:
            from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

            # Phase 4: Preprocessing with timing
            preprocess_start = time.perf_counter()
            preprocessed = self._preprocess(input_data)
            preprocess_end = time.perf_counter()
            preprocess_ms = (preprocess_end - preprocess_start) * 1000

            # Get input/output stream info
            input_vstream_info = self._hef.get_input_vstream_infos()[0]
            output_vstream_infos = self._hef.get_output_vstream_infos()

            # Create vstream params
            input_params = InputVStreamParams.make_from_network_group(
                self._configured_infer_model, quantized=False
            )
            output_params = OutputVStreamParams.make_from_network_group(
                self._configured_infer_model, quantized=False
            )

            # Phase 4: Run inference on NPU with precise timing
            inference_start = time.perf_counter()

            with InferVStreams(
                self._configured_infer_model, input_params, output_params
            ) as infer_pipeline:
                input_dict = {input_vstream_info.name: preprocessed}
                raw_outputs = infer_pipeline.infer(input_dict)

            inference_end = time.perf_counter()
            inference_ms = (inference_end - inference_start) * 1000

            # Phase 4: Post-processing with timing
            postprocess_start = time.perf_counter()

            # Update post-processor config with thresholds
            postprocess_config = PostProcessConfig(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                input_width=self._input_resolution,
                input_height=self._input_resolution,
                num_classes=self._postprocessor.config.num_classes,
            )

            # Process outputs using the post-processor
            results = self._postprocessor.process(raw_outputs, postprocess_config)

            postprocess_end = time.perf_counter()
            postprocess_ms = (postprocess_end - postprocess_start) * 1000

            # Total latency (inference only, excluding pre/post processing per PRD)
            # Note: We report inference_ms as the main latency metric
            total_ms = preprocess_ms + inference_ms + postprocess_ms

            # Convert results to list of dicts for InferenceResult
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], Detection):
                    detections = [d.to_dict() for d in results]
                elif isinstance(results[0], ClassificationResult):
                    detections = [c.to_dict() for c in results]
                else:
                    detections = results
            else:
                detections = []

            return InferenceResult(
                latency_ms=inference_ms,  # NPU inference time only (per PRD)
                outputs=raw_outputs,
                detections=detections,
                metadata={
                    "preprocess_ms": preprocess_ms,
                    "inference_ms": inference_ms,
                    "postprocess_ms": postprocess_ms,
                    "total_ms": total_ms,
                    "num_detections": len(detections),
                    "backend": "hailo",
                    "device": self.get_target_device(),
                },
            )

        except ImportError:
            raise RuntimeError(
                "HailoRT not available. Ensure hailo_platform is installed."
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")

    def _preprocess(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess input for Hailo inference.

        Args:
            input_data: Raw input image

        Returns:
            Preprocessed input ready for inference
        """
        # Ensure correct shape and dtype
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dim

        # Normalize to [0, 1] if needed
        if input_data.dtype == np.uint8:
            input_data = input_data.astype(np.float32) / 255.0

        return input_data

    def collect_metrics(self) -> dict:
        """Collect comprehensive Hailo-specific metrics.

        Phase 4 implementation collects:
        - Device information
        - Memory usage (host and device)
        - Power consumption (if available)
        - NPU utilization (if available)
        - Last inference metrics

        Returns:
            Dictionary with all available metrics
        """
        import psutil

        metrics = {
            "backend": "hailo",
            "device": self.device,
        }

        # Device information
        if self._device_info:
            metrics["device_type"] = self._device_info.device_type
            metrics["firmware_version"] = self._device_info.firmware_version
            metrics["device_path"] = self._device_info.device_path

        # Host memory usage
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics["host_memory_used_mb"] = memory_info.rss / (1024 ** 2)
            metrics["host_memory_percent"] = process.memory_percent()

            # System-wide memory
            virtual_mem = psutil.virtual_memory()
            metrics["system_memory_used_mb"] = virtual_mem.used / (1024 ** 2)
            metrics["system_memory_total_mb"] = virtual_mem.total / (1024 ** 2)
            metrics["system_memory_percent"] = virtual_mem.percent
        except Exception as e:
            logger.debug(f"Failed to collect memory metrics: {e}")

        # CPU usage
        try:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # Power consumption
        power = self._get_power_consumption()
        if power is not None:
            metrics["power_watts"] = power

        # Include last metrics if available
        if self._last_metrics:
            metrics["last_inference_latency_ms"] = self._last_metrics.inference_latency_ms
            metrics["last_fps"] = self._last_metrics.fps
            if self._last_metrics.power_watts:
                metrics["inference_power_watts"] = self._last_metrics.power_watts

        # Model information
        if self._model_info:
            metrics["model_name"] = self._model_info.name
            metrics["model_version"] = self._model_info.version
            metrics["task"] = self._model_info.task.value
            metrics["input_resolution"] = self._model_info.input_resolution

        # HailoRT version
        metrics["hailort_version"] = self._get_hailort_version()

        return metrics

    def _get_power_consumption(self) -> Optional[float]:
        """Get current power consumption if available.

        Attempts to read power from:
        1. Raspberry Pi power sensors
        2. Hailo CLI tools

        Returns:
            Power in watts or None
        """
        # Try Raspberry Pi power sensors
        from glob import glob

        power_paths = [
            "/sys/class/hwmon/hwmon*/power1_input",
            "/sys/class/hwmon/hwmon*/curr1_input",
        ]

        for pattern in power_paths:
            matches = glob(pattern)
            for path in matches:
                try:
                    with open(path, "r") as f:
                        value = float(f.read().strip())
                        if "power" in path:
                            return value / 1_000_000.0  # microwatts to watts
                        elif "curr" in path:
                            return (value / 1000.0) * 5.0  # milliamps * 5V
                except (IOError, ValueError):
                    continue

        # Try hailortcli
        try:
            result = subprocess.run(
                ["hailortcli", "measure-power"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                import re
                for line in result.stdout.split("\n"):
                    if "power" in line.lower():
                        match = re.search(r"(\d+\.?\d*)\s*[wW]", line)
                        if match:
                            return float(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def start_metrics_collection(self) -> None:
        """Start background metrics collection.

        Call this before running inference loops to collect
        continuous metrics during the benchmark.
        """
        self._metrics_collector.measure_idle_power(duration=1.0)
        self._metrics_collector.start_monitoring()

    def stop_metrics_collection(self) -> HailoMetrics:
        """Stop metrics collection and return aggregated results.

        Returns:
            HailoMetrics with all collected data
        """
        self._last_metrics = self._metrics_collector.stop_monitoring()
        return self._last_metrics

    def record_inference_metrics(
        self,
        inference_ms: float,
        preprocess_ms: float = 0.0,
        postprocess_ms: float = 0.0,
    ) -> None:
        """Record metrics for a single inference run.

        Args:
            inference_ms: NPU inference latency
            preprocess_ms: Preprocessing latency
            postprocess_ms: Postprocessing latency
        """
        self._metrics_collector.record_inference(
            inference_ms, preprocess_ms, postprocess_ms
        )

    def get_last_metrics(self) -> Optional[HailoMetrics]:
        """Get the last collected metrics.

        Returns:
            HailoMetrics from the last collection period or None
        """
        return self._last_metrics

    def cleanup(self) -> None:
        """Release Hailo resources."""
        if self._configured_infer_model is not None:
            try:
                self._configured_infer_model.shutdown()
            except Exception:
                pass
            self._configured_infer_model = None

        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception:
                pass
            self._vdevice = None

        self._hef = None
        self._hef_path = None
        self._model_info = None

        # Phase 4: Clear post-processor and metrics
        self._postprocessor = None
        self._last_metrics = None
        self._current_task = None

        logger.debug("Hailo backend cleaned up")

    def run_benchmark(
        self,
        input_data: np.ndarray,
        num_runs: int = 10,
        warmup_runs: int = 3,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Tuple[List[InferenceResult], HailoMetrics]:
        """Run a complete benchmark with warmup and metrics collection.

        Phase 4 implementation provides:
        - Warmup runs (excluded from metrics)
        - Measured runs with full metrics collection
        - Comprehensive HailoMetrics

        Args:
            input_data: Input image as numpy array
            num_runs: Number of measured runs
            warmup_runs: Number of warmup runs
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            Tuple of (list of InferenceResults, HailoMetrics)
        """
        results = []

        # Warmup runs (not recorded)
        logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            self.run_inference(input_data, conf_threshold, iou_threshold)

        # Start metrics collection
        self.start_metrics_collection()

        # Measured runs
        logger.info(f"Running {num_runs} measured iterations...")
        for i in range(num_runs):
            result = self.run_inference(input_data, conf_threshold, iou_threshold)
            results.append(result)

            # Record metrics
            self.record_inference_metrics(
                result.latency_ms,
                result.metadata.get("preprocess_ms", 0),
                result.metadata.get("postprocess_ms", 0),
            )

        # Stop metrics collection
        metrics = self.stop_metrics_collection()

        logger.info(f"Benchmark complete: mean latency {metrics.latency_mean_ms:.2f}ms, FPS {metrics.fps:.1f}")

        return results, metrics

    def get_version_info(self) -> dict:
        """Get Hailo version information."""
        info = super().get_version_info()
        info["hailort"] = self._get_hailort_version()
        if self._device_info:
            info["device_type"] = self._device_info.device_type
            info["firmware"] = self._device_info.firmware_version
        return info

    def _get_hailort_version(self) -> str:
        """Get HailoRT version."""
        try:
            result = subprocess.run(
                ["hailortcli", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            from hailo_platform import __version__
            return __version__
        except (ImportError, AttributeError):
            pass

        return "unknown"
