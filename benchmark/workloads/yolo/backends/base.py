# Abstract base class for YOLO inference backends
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any

import numpy as np

from benchmark.schemas import YOLOTask, LatencyMetrics, ResourceUtilization


class BackendType(Enum):
    """Supported inference backends."""

    PYTORCH = "pytorch"
    HAILO = "hailo"


@dataclass
class BackendCapabilities:
    """Describes what a backend can do."""

    supported_tasks: list[YOLOTask]
    supported_yolo_versions: list[str]
    supports_gpu: bool = False
    supports_npu: bool = False
    supports_quantization: bool = False
    requires_compilation: bool = False
    max_batch_size: int = 1


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    version: str
    task: YOLOTask
    input_resolution: int
    backend: BackendType
    model_path: Path
    compiled_path: Optional[Path] = None
    quantization: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from a single inference run."""

    latency_ms: float
    outputs: Any  # Raw model outputs
    detections: Optional[list] = None  # Post-processed detections
    metadata: dict = field(default_factory=dict)


class YOLOBackend(ABC):
    """Abstract base class for YOLO inference backends.

    All backends must implement these methods to be compatible with
    the benchmark suite. This allows swapping between PyTorch, Hailo,
    and potentially other backends without changing benchmark logic.
    """

    def __init__(self, device: str = "0"):
        """Initialize the backend.

        Args:
            device: Device identifier (e.g., "0" for GPU/NPU, "cpu" for CPU)
        """
        self.device = device
        self._model_info: Optional[ModelInfo] = None

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the backend capabilities."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        Returns:
            True if the backend can be used, False otherwise
        """
        pass

    @abstractmethod
    def prepare_model(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        input_resolution: int = 640,
        force_recompile: bool = False,
    ) -> ModelInfo:
        """Prepare a model for inference.

        This may involve:
        - Downloading the model weights
        - Converting to backend-specific format
        - Compiling for the target hardware
        - Loading into memory

        Args:
            model_name: Name of the YOLO model (e.g., "yolov8n.pt")
            yolo_version: YOLO version (e.g., "v8", "v11")
            task: YOLO task type
            input_resolution: Input image resolution
            force_recompile: Force recompilation even if cached

        Returns:
            ModelInfo with details about the prepared model

        Raises:
            ValueError: If model/task combination is not supported
            RuntimeError: If model preparation fails
        """
        pass

    @abstractmethod
    def run_inference(
        self,
        input_data: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> InferenceResult:
        """Run inference on input data.

        Args:
            input_data: Input image as numpy array (H, W, C) or (B, H, W, C)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Returns:
            InferenceResult with latency and outputs

        Raises:
            RuntimeError: If model is not prepared or inference fails
        """
        pass

    @abstractmethod
    def collect_metrics(self) -> dict:
        """Collect backend-specific metrics.

        Returns:
            Dictionary with metrics like:
            - device_utilization
            - memory_usage
            - power_consumption (if available)
            - backend-specific stats
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources.

        Called when done with the model. Should release:
        - GPU/NPU memory
        - File handles
        - Any other resources
        """
        pass

    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the currently loaded model."""
        return self._model_info

    def supports_model(self, model_name: str, task: YOLOTask) -> bool:
        """Check if this backend supports a specific model/task combination.

        Args:
            model_name: Model name
            task: Task type

        Returns:
            True if supported, False otherwise
        """
        return task in self.capabilities.supported_tasks

    def get_version_info(self) -> dict:
        """Get version information for the backend and its dependencies.

        Returns:
            Dictionary with version strings
        """
        return {
            "backend": self.backend_type.value,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
        return False
