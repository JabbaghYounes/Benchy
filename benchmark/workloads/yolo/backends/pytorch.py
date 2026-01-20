# PyTorch/Ultralytics backend for YOLO inference
import logging
import time
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


class PyTorchBackend(YOLOBackend):
    """PyTorch/Ultralytics backend for YOLO inference.

    This backend uses the Ultralytics library for inference,
    which supports PyTorch models running on CPU or CUDA GPU.
    """

    def __init__(self, device: str = "0"):
        """Initialize PyTorch backend.

        Args:
            device: Device identifier ("0" for GPU, "cpu" for CPU)
        """
        super().__init__(device)
        self._model = None
        self._ultralytics_available: Optional[bool] = None
        self._torch_available: Optional[bool] = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PYTORCH

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_tasks=[
                YOLOTask.DETECTION,
                YOLOTask.SEGMENTATION,
                YOLOTask.POSE,
                YOLOTask.OBB,
                YOLOTask.CLASSIFICATION,
            ],
            supported_yolo_versions=["v8", "v11", "v26"],
            supports_gpu=self._has_cuda(),
            supports_npu=False,
            supports_quantization=False,
            requires_compilation=False,
            max_batch_size=32,
        )

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def is_available(self) -> bool:
        """Check if PyTorch/Ultralytics is available."""
        if self._ultralytics_available is None:
            try:
                import ultralytics
                import torch
                self._ultralytics_available = True
                self._torch_available = True
            except ImportError:
                self._ultralytics_available = False
                self._torch_available = False
        return self._ultralytics_available

    def prepare_model(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        input_resolution: int = 640,
        force_recompile: bool = False,
    ) -> ModelInfo:
        """Load a YOLO model using Ultralytics.

        Args:
            model_name: Model file name (e.g., "yolov8n.pt")
            yolo_version: YOLO version string
            task: Task type
            input_resolution: Input resolution
            force_recompile: Ignored for PyTorch backend

        Returns:
            ModelInfo describing the loaded model
        """
        if not self.is_available():
            raise RuntimeError("Ultralytics/PyTorch is not available")

        from ultralytics import YOLO

        logger.info(f"Loading PyTorch model: {model_name}")

        # Load the model
        self._model = YOLO(model_name)

        # Move to appropriate device
        if self.device != "cpu" and self._has_cuda():
            self._model.to(f"cuda:{self.device}")
            device_str = f"cuda:{self.device}"
        else:
            device_str = "cpu"

        logger.info(f"Model loaded on device: {device_str}")

        # Create model info
        self._model_info = ModelInfo(
            name=model_name,
            version=yolo_version,
            task=task,
            input_resolution=input_resolution,
            backend=BackendType.PYTORCH,
            model_path=Path(model_name),
            metadata={
                "device": device_str,
                "ultralytics_version": self._get_ultralytics_version(),
                "torch_version": self._get_torch_version(),
            },
        )

        return self._model_info

    def run_inference(
        self,
        input_data: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> InferenceResult:
        """Run inference using Ultralytics.

        Args:
            input_data: Input image as numpy array
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            InferenceResult with timing and outputs
        """
        if self._model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        # Run inference with timing
        start_time = time.perf_counter()
        results = self._model(
            input_data,
            imgsz=self._model_info.input_resolution,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        # Extract detections if available
        detections = None
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None:
                detections = result.boxes.data.cpu().numpy().tolist()

        return InferenceResult(
            latency_ms=latency_ms,
            outputs=results,
            detections=detections,
        )

    def run_validation(
        self,
        data: str,
        input_resolution: int = 640,
    ) -> dict:
        """Run validation to get accuracy metrics.

        Args:
            data: Path to validation dataset YAML
            input_resolution: Input resolution

        Returns:
            Dictionary with accuracy metrics
        """
        if self._model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        try:
            metrics = self._model.val(
                data=data,
                imgsz=input_resolution,
                verbose=False,
            )

            # Extract metrics based on task type
            task = self._model_info.task

            if task == YOLOTask.DETECTION:
                return {
                    "map_score": float(metrics.box.map) if hasattr(metrics, "box") else None,
                    "map50": float(metrics.box.map50) if hasattr(metrics, "box") else None,
                    "precision": float(metrics.box.mp) if hasattr(metrics, "box") else None,
                    "recall": float(metrics.box.mr) if hasattr(metrics, "box") else None,
                }
            elif task == YOLOTask.SEGMENTATION:
                return {
                    "map_score": float(metrics.seg.map) if hasattr(metrics, "seg") else None,
                    "precision": float(metrics.seg.mp) if hasattr(metrics, "seg") else None,
                    "recall": float(metrics.seg.mr) if hasattr(metrics, "seg") else None,
                }
            elif task == YOLOTask.POSE:
                return {
                    "map_score": float(metrics.pose.map) if hasattr(metrics, "pose") else None,
                    "precision": float(metrics.pose.mp) if hasattr(metrics, "pose") else None,
                    "recall": float(metrics.pose.mr) if hasattr(metrics, "pose") else None,
                }
            elif task == YOLOTask.CLASSIFICATION:
                return {
                    "top1_accuracy": float(metrics.top1) if hasattr(metrics, "top1") else None,
                    "top5_accuracy": float(metrics.top5) if hasattr(metrics, "top5") else None,
                }
            else:
                return {
                    "map_score": float(metrics.box.map) if hasattr(metrics, "box") else None,
                }
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {}

    def collect_metrics(self) -> dict:
        """Collect PyTorch-specific metrics."""
        metrics = {
            "backend": "pytorch",
            "device": self.device,
        }

        # Try to get GPU memory usage
        if self._has_cuda():
            try:
                import torch
                if torch.cuda.is_available():
                    metrics["gpu_memory_allocated_mb"] = (
                        torch.cuda.memory_allocated() / (1024 ** 2)
                    )
                    metrics["gpu_memory_reserved_mb"] = (
                        torch.cuda.memory_reserved() / (1024 ** 2)
                    )
            except Exception:
                pass

        return metrics

    def cleanup(self) -> None:
        """Release model and GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        # Clear CUDA cache if available
        if self._has_cuda():
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._model_info = None
        logger.debug("PyTorch backend cleaned up")

    def get_version_info(self) -> dict:
        """Get version information."""
        info = super().get_version_info()
        info["ultralytics"] = self._get_ultralytics_version()
        info["torch"] = self._get_torch_version()
        if self._has_cuda():
            info["cuda"] = self._get_cuda_version()
        return info

    def _get_ultralytics_version(self) -> str:
        """Get Ultralytics version."""
        try:
            import ultralytics
            return ultralytics.__version__
        except (ImportError, AttributeError):
            return "unknown"

    def _get_torch_version(self) -> str:
        """Get PyTorch version."""
        try:
            import torch
            return torch.__version__
        except (ImportError, AttributeError):
            return "unknown"

    def _get_cuda_version(self) -> str:
        """Get CUDA version."""
        try:
            import torch
            return torch.version.cuda or "not available"
        except (ImportError, AttributeError):
            return "not available"
