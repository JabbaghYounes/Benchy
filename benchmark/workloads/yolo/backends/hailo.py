# Hailo NPU backend for YOLO inference
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.backends.base import (
    YOLOBackend,
    BackendType,
    BackendCapabilities,
    ModelInfo,
    InferenceResult,
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

        # Load the HEF
        self._load_hef(hef_path)
        self._hef_path = hef_path

        # Create model info
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

    def run_inference(
        self,
        input_data: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> InferenceResult:
        """Run inference on the Hailo NPU.

        Args:
            input_data: Input image as numpy array
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            InferenceResult with timing and outputs
        """
        if self._configured_infer_model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        # This will be fully implemented in Phase 4
        # For now, provide a basic implementation structure
        try:
            from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams

            # Preprocess input
            preprocessed = self._preprocess(input_data)

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

            # Run inference
            start_time = time.perf_counter()

            with InferVStreams(
                self._configured_infer_model, input_params, output_params
            ) as infer_pipeline:
                input_dict = {input_vstream_info.name: preprocessed}
                raw_outputs = infer_pipeline.infer(input_dict)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Post-process outputs (NMS, etc.)
            detections = self._postprocess(
                raw_outputs, conf_threshold, iou_threshold
            )

            return InferenceResult(
                latency_ms=latency_ms,
                outputs=raw_outputs,
                detections=detections,
            )

        except ImportError:
            raise RuntimeError("HailoRT not available")
        except Exception as e:
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

    def _postprocess(
        self,
        raw_outputs: dict,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list:
        """Post-process raw model outputs.

        This includes:
        - Decoding bounding boxes
        - Applying confidence threshold
        - Running NMS

        Args:
            raw_outputs: Raw outputs from the model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detections
        """
        # Post-processing will be implemented in Phase 4
        # For now, return raw outputs
        return []

    def collect_metrics(self) -> dict:
        """Collect Hailo-specific metrics."""
        metrics = {
            "backend": "hailo",
            "device": self.device,
        }

        if self._device_info:
            metrics["device_type"] = self._device_info.device_type
            metrics["firmware_version"] = self._device_info.firmware_version

        # Try to get power consumption
        power = self._get_power_consumption()
        if power is not None:
            metrics["power_watts"] = power

        return metrics

    def _get_power_consumption(self) -> Optional[float]:
        """Get current power consumption if available."""
        # Hailo doesn't expose power directly, but we can try system-level power
        # This will be platform-specific
        return None

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
        logger.debug("Hailo backend cleaned up")

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
