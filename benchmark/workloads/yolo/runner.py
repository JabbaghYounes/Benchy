# YOLO Benchmark Runner
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.schemas import (
    YOLOResult,
    YOLOTask,
    LatencyMetrics,
    SystemInfo,
)
from benchmark.metrics import ResourceMonitor, detect_platform

logger = logging.getLogger(__name__)


@dataclass
class YOLOBenchmarkConfig:
    """Configuration for YOLO benchmark runs."""

    model_name: str = "yolov8n.pt"
    yolo_version: str = "v8"
    task: YOLOTask = YOLOTask.DETECTION
    input_resolution: int = 640
    warmup_runs: int = 3
    measured_runs: int = 10
    device: str = "0"  # GPU device ID or "cpu"
    dataset: Optional[str] = None  # Path to dataset, None uses default
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    verbose: bool = False
    backend: Optional[str] = None  # "pytorch", "hailo", or None for auto
    force_recompile: bool = False  # Force recompilation of Hailo models


# Model definitions per YOLO version and task
YOLO_MODELS = {
    "v8": {
        YOLOTask.DETECTION: ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        YOLOTask.SEGMENTATION: ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
        YOLOTask.POSE: ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt", "yolov8x-pose-p6.pt"],
        YOLOTask.OBB: ["yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt"],
        YOLOTask.CLASSIFICATION: ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"],
    },
    "v11": {
        YOLOTask.DETECTION: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        YOLOTask.SEGMENTATION: ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"],
        YOLOTask.POSE: ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"],
        YOLOTask.OBB: ["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt"],
        YOLOTask.CLASSIFICATION: ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt"],
    },
    "v26": {
        YOLOTask.DETECTION: ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        YOLOTask.SEGMENTATION: ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"],
        YOLOTask.POSE: ["yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt", "yolo26x-pose.pt"],
        YOLOTask.OBB: ["yolo26n-obb.pt", "yolo26s-obb.pt", "yolo26m-obb.pt", "yolo26l-obb.pt", "yolo26x-obb.pt"],
        YOLOTask.CLASSIFICATION: ["yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt", "yolo26l-cls.pt", "yolo26x-cls.pt"],
    },
}

# Default datasets per task type
DEFAULT_DATASETS = {
    YOLOTask.DETECTION: "coco128.yaml",
    YOLOTask.SEGMENTATION: "coco128-seg.yaml",
    YOLOTask.POSE: "coco8-pose.yaml",
    YOLOTask.OBB: "dota8.yaml",
    YOLOTask.CLASSIFICATION: "imagenet10",
}


class YOLOBenchmarkRunner:
    """Runner for YOLO inference benchmarks.

    This runner supports multiple backends (PyTorch, Hailo) and will
    automatically select the appropriate backend based on the platform
    unless a specific backend is requested.
    """

    def __init__(self, config: YOLOBenchmarkConfig):
        """Initialize the YOLO benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._backend = None
        self._resource_monitor = ResourceMonitor(sample_interval=0.1)

    def _select_backend(self):
        """Select and initialize the appropriate backend."""
        from benchmark.workloads.yolo.backends import (
            BackendType,
            auto_select_backend,
            get_backend,
        )

        # If backend is specified, use it
        if self.config.backend:
            if self.config.backend == "auto":
                # Explicit auto-selection
                self._backend = auto_select_backend(
                    device=self.config.device,
                    allow_fallback=False,
                )
            else:
                # Specific backend requested
                backend_type = BackendType(self.config.backend)
                self._backend = get_backend(backend_type, device=self.config.device)
        else:
            # Auto-select based on platform
            self._backend = auto_select_backend(
                device=self.config.device,
                allow_fallback=True,  # Allow fallback to PyTorch
            )

        logger.info(f"Using backend: {self._backend.backend_type.value}")

    def _prepare_model(self):
        """Prepare the model using the selected backend."""
        if self._backend is None:
            self._select_backend()

        logger.info(f"Preparing model: {self.config.model_name}")
        self._backend.prepare_model(
            model_name=self.config.model_name,
            yolo_version=self.config.yolo_version,
            task=self.config.task,
            input_resolution=self.config.input_resolution,
            force_recompile=self.config.force_recompile,
        )

    def _get_validation_source(self) -> str:
        """Get the validation data source."""
        if self.config.dataset:
            return self.config.dataset
        return DEFAULT_DATASETS.get(self.config.task, "coco128.yaml")

    def _run_single_inference(self, source: np.ndarray) -> float:
        """Run a single inference and return latency.

        Args:
            source: Input image as numpy array

        Returns:
            Latency in milliseconds
        """
        result = self._backend.run_inference(
            source,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
        )
        return result.latency_ms

    def _run_validation(self) -> dict:
        """Run validation to get accuracy metrics.

        Returns:
            Dictionary with mAP, precision, recall metrics
        """
        from benchmark.workloads.yolo.backends import BackendType

        # Validation is only supported for PyTorch backend currently
        if self._backend.backend_type == BackendType.PYTORCH:
            try:
                return self._backend.run_validation(
                    data=self._get_validation_source(),
                    input_resolution=self.config.input_resolution,
                )
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                return {}
        else:
            # Hailo validation not yet implemented
            logger.info("Validation not available for Hailo backend")
            return {}

    def _calculate_latency_metrics(
        self, latencies: list[float], first_latency: float
    ) -> LatencyMetrics:
        """Calculate latency statistics from measured runs.

        Args:
            latencies: List of latency measurements in ms
            first_latency: First inference latency in ms

        Returns:
            LatencyMetrics dataclass
        """
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return LatencyMetrics(
            first_inference_ms=round(first_latency, 3),
            mean_ms=round(statistics.mean(latencies), 3),
            std_ms=round(statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 3),
            min_ms=round(min(latencies), 3),
            max_ms=round(max(latencies), 3),
            p50_ms=round(sorted_latencies[min(p50_idx, n - 1)], 3),
            p95_ms=round(sorted_latencies[min(p95_idx, n - 1)], 3),
            p99_ms=round(sorted_latencies[min(p99_idx, n - 1)], 3),
        )

    def run(self, test_image: Optional[str] = None) -> YOLOResult:
        """Run the complete benchmark.

        Args:
            test_image: Optional path to test image. If None, uses a synthetic image.

        Returns:
            YOLOResult with all benchmark metrics
        """
        # Prepare model using the backend
        self._prepare_model()

        # Create test input
        if test_image:
            import cv2
            source = cv2.imread(test_image)
            if source is None:
                raise ValueError(f"Could not load test image: {test_image}")
            # Resize to expected resolution
            source = cv2.resize(
                source,
                (self.config.input_resolution, self.config.input_resolution)
            )
        else:
            # Create a synthetic image for consistent latency measurement
            source = np.random.randint(
                0, 255,
                (self.config.input_resolution, self.config.input_resolution, 3),
                dtype=np.uint8
            )

        logger.info(
            f"Starting benchmark: {self.config.warmup_runs} warmup, "
            f"{self.config.measured_runs} measured runs"
        )

        # Warmup runs (not recorded)
        logger.info("Running warmup iterations...")
        for i in range(self.config.warmup_runs):
            self._run_single_inference(source)
            logger.debug(f"Warmup run {i + 1}/{self.config.warmup_runs} complete")

        # Measured runs with resource monitoring
        logger.info("Running measured iterations...")
        self._resource_monitor.start()

        latencies = []
        first_latency = None

        for i in range(self.config.measured_runs):
            latency_ms = self._run_single_inference(source)
            latencies.append(latency_ms)
            if first_latency is None:
                first_latency = latency_ms
            logger.debug(
                f"Measured run {i + 1}/{self.config.measured_runs}: {latency_ms:.2f}ms"
            )

        resource_utilization = self._resource_monitor.stop()

        # Calculate metrics
        latency_metrics = self._calculate_latency_metrics(latencies, first_latency)
        throughput_fps = (
            1000.0 / latency_metrics.mean_ms if latency_metrics.mean_ms > 0 else 0.0
        )

        # Run validation for accuracy metrics (optional, can be slow)
        accuracy_metrics = {}
        try:
            logger.info("Running validation for accuracy metrics...")
            accuracy_metrics = self._run_validation()
        except Exception as e:
            logger.warning(f"Skipping accuracy validation: {e}")

        # Get backend type for result
        backend_name = self._backend.backend_type.value

        # Cleanup backend resources
        self._backend.cleanup()

        return YOLOResult(
            model_name=self.config.model_name,
            yolo_version=self.config.yolo_version,
            task=self.config.task.value,
            input_resolution=f"{self.config.input_resolution}x{self.config.input_resolution}",
            latency=latency_metrics,
            throughput_fps=round(throughput_fps, 2),
            backend=backend_name,
            map_score=accuracy_metrics.get("map_score"),
            precision=accuracy_metrics.get("precision"),
            recall=accuracy_metrics.get("recall"),
            resource_utilization=resource_utilization,
            warmup_runs=self.config.warmup_runs,
            measured_runs=self.config.measured_runs,
        )


def run_yolo_benchmark(
    model_name: str = "yolov8n.pt",
    yolo_version: str = "v8",
    task: YOLOTask = YOLOTask.DETECTION,
    input_resolution: int = 640,
    warmup_runs: int = 3,
    measured_runs: int = 10,
    device: str = "0",
    test_image: Optional[str] = None,
    skip_validation: bool = False,
    backend: Optional[str] = None,
) -> YOLOResult:
    """Convenience function to run a YOLO benchmark.

    Args:
        model_name: Name of the YOLO model file
        yolo_version: YOLO version (v8, v11, v26)
        task: YOLO task type
        input_resolution: Input image resolution
        warmup_runs: Number of warmup iterations
        measured_runs: Number of measured iterations
        device: Device ID for inference
        test_image: Optional test image path
        skip_validation: Skip accuracy validation
        backend: Backend to use (pytorch, hailo, or None for auto)

    Returns:
        YOLOResult with benchmark metrics
    """
    config = YOLOBenchmarkConfig(
        model_name=model_name,
        yolo_version=yolo_version,
        task=task,
        input_resolution=input_resolution,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        device=device,
        backend=backend,
    )

    runner = YOLOBenchmarkRunner(config)
    return runner.run(test_image=test_image)


def get_available_models(yolo_version: str, task: YOLOTask) -> list[str]:
    """Get list of available models for a YOLO version and task.

    Args:
        yolo_version: YOLO version (v8, v11, v26)
        task: YOLO task type

    Returns:
        List of model file names
    """
    return YOLO_MODELS.get(yolo_version, {}).get(task, [])
