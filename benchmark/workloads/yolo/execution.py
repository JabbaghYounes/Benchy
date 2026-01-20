# Phase 5 - Benchmark Execution Logic
#
# This module enforces strict benchmark execution rules:
# - Task 5.1: Warmup (3) & measured runs (10) with consistent order
# - Task 5.2: Error & fallback enforcement (NO CPU fallback for Hailo)
#
# All benchmark runs must go through this module to ensure consistency.

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from benchmark.schemas import YOLOTask

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Base exception for benchmark execution errors."""
    pass


class UnsupportedModelError(BenchmarkError):
    """Raised when a model is not supported on the target backend."""
    pass


class UnsupportedTaskError(BenchmarkError):
    """Raised when a task type is not supported on the target backend."""
    pass


class HailoFallbackError(BenchmarkError):
    """Raised when Hailo execution fails and fallback is not allowed."""
    pass


class ModelCompatibilityError(BenchmarkError):
    """Raised when model/task/backend combination is incompatible."""
    pass


class ExecutionPhase(Enum):
    """Benchmark execution phases for tracking."""
    INITIALIZATION = "initialization"
    WARMUP = "warmup"
    MEASUREMENT = "measurement"
    AGGREGATION = "aggregation"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ExecutionConfig:
    """Configuration for benchmark execution.

    Phase 5 enforces specific defaults:
    - warmup_runs: 3 (excluded from metrics)
    - measured_runs: 10 (aggregated for final metrics)
    - allow_cpu_fallback: False for Hailo backends
    """
    warmup_runs: int = 3
    measured_runs: int = 10
    allow_cpu_fallback: bool = False  # Phase 5.2: NO CPU fallback by default
    strict_mode: bool = True  # Abort on any error
    log_each_run: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.measured_runs < 1:
            raise ValueError("measured_runs must be at least 1")


@dataclass
class ExecutionResult:
    """Result of a benchmark execution run."""
    success: bool
    phase: ExecutionPhase
    warmup_latencies: List[float] = field(default_factory=list)
    measured_latencies: List[float] = field(default_factory=list)
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if execution completed successfully."""
        return self.success and self.phase == ExecutionPhase.COMPLETE

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics from measured runs only.

        Returns:
            Dictionary with mean, std, min, max, p50, p95, p99 latencies
        """
        if not self.measured_latencies:
            return {}

        latencies = np.array(self.measured_latencies)
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "fps": float(1000.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0.0,
        }


# Hailo-supported models and tasks
# Phase 5.2: Clear definition of what is supported
HAILO_SUPPORTED_VERSIONS = ["v8", "v11", "v26"]

HAILO_SUPPORTED_TASKS = {
    "v8": [YOLOTask.DETECTION, YOLOTask.CLASSIFICATION],
    "v11": [YOLOTask.DETECTION, YOLOTask.CLASSIFICATION],
    "v26": [YOLOTask.DETECTION, YOLOTask.CLASSIFICATION],
}

# Models known to have Hailo support (nano and small variants work best)
HAILO_OPTIMIZED_MODELS = {
    "v8": {
        YOLOTask.DETECTION: ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        YOLOTask.CLASSIFICATION: ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt"],
    },
    "v11": {
        YOLOTask.DETECTION: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        YOLOTask.CLASSIFICATION: ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt"],
    },
    "v26": {
        YOLOTask.DETECTION: ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"],
        YOLOTask.CLASSIFICATION: ["yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt"],
    },
}


def check_hailo_compatibility(
    model_name: str,
    yolo_version: str,
    task: YOLOTask,
) -> Tuple[bool, str]:
    """Check if a model/task combination is compatible with Hailo.

    Phase 5.2: Provides clear error messages for unsupported configurations.

    Args:
        model_name: Name of the YOLO model (e.g., "yolov8n.pt")
        yolo_version: YOLO version (v8, v11, v26)
        task: YOLO task type

    Returns:
        Tuple of (is_compatible, reason_message)
    """
    # Check YOLO version support
    if yolo_version not in HAILO_SUPPORTED_VERSIONS:
        return False, (
            f"YOLO version '{yolo_version}' is not supported on Hailo. "
            f"Supported versions: {', '.join(HAILO_SUPPORTED_VERSIONS)}"
        )

    # Check task support for this version
    supported_tasks = HAILO_SUPPORTED_TASKS.get(yolo_version, [])
    if task not in supported_tasks:
        task_names = [t.value for t in supported_tasks]
        return False, (
            f"Task '{task.value}' is not supported for YOLO {yolo_version} on Hailo. "
            f"Supported tasks: {', '.join(task_names)}. "
            f"Tasks like segmentation, pose, and OBB require architectures not yet "
            f"optimized for Hailo NPU."
        )

    # Check model availability (warning level, not blocking)
    optimized_models = HAILO_OPTIMIZED_MODELS.get(yolo_version, {}).get(task, [])
    if model_name not in optimized_models and optimized_models:
        # Model may still work, just not optimized
        logger.warning(
            f"Model '{model_name}' is not in the list of Hailo-optimized models. "
            f"Recommended models for {yolo_version} {task.value}: "
            f"{', '.join(optimized_models)}"
        )

    return True, "Model/task combination is compatible with Hailo"


def validate_execution_preconditions(
    backend_type: str,
    model_name: str,
    yolo_version: str,
    task: YOLOTask,
    allow_fallback: bool = False,
) -> None:
    """Validate all preconditions before benchmark execution.

    Phase 5.2: Abort run and log reason if unsupported.

    Args:
        backend_type: Backend being used ("hailo", "pytorch")
        model_name: Name of the YOLO model
        yolo_version: YOLO version string
        task: YOLO task type
        allow_fallback: Whether CPU fallback is allowed

    Raises:
        UnsupportedModelError: If model is not supported
        UnsupportedTaskError: If task is not supported
        HailoFallbackError: If Hailo fails and fallback not allowed
        ModelCompatibilityError: If combination is invalid
    """
    if backend_type == "hailo":
        is_compatible, reason = check_hailo_compatibility(
            model_name, yolo_version, task
        )

        if not is_compatible:
            if allow_fallback:
                # This should not happen in Phase 5 - fallback is disabled
                logger.warning(
                    f"Hailo incompatibility detected but fallback requested: {reason}"
                )
                raise HailoFallbackError(
                    f"CPU fallback is disabled (Phase 5.2 requirement). "
                    f"Original error: {reason}"
                )
            else:
                # Phase 5.2: Abort with clear message
                logger.error(f"Hailo compatibility check failed: {reason}")
                raise ModelCompatibilityError(reason)


class BenchmarkExecutor:
    """Executes benchmarks with strict Phase 5 enforcement.

    This executor ensures:
    - Warmup runs are excluded from metrics
    - Measured runs follow consistent execution order
    - No silent degradation or CPU fallback for Hailo
    - Clear failure messages on any error
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize the executor.

        Args:
            config: Execution configuration. Uses Phase 5 defaults if None.
        """
        self.config = config or ExecutionConfig()
        self._current_phase = ExecutionPhase.INITIALIZATION
        self._warmup_latencies: List[float] = []
        self._measured_latencies: List[float] = []

    def reset(self) -> None:
        """Reset executor state for a new benchmark run."""
        self._current_phase = ExecutionPhase.INITIALIZATION
        self._warmup_latencies = []
        self._measured_latencies = []

    def execute(
        self,
        inference_fn: Callable[[], float],
        backend_type: str,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> ExecutionResult:
        """Execute a complete benchmark with Phase 5 enforcement.

        Args:
            inference_fn: Function that runs one inference, returns latency in ms
            backend_type: Backend being used
            model_name: Model name for compatibility checking
            yolo_version: YOLO version
            task: YOLO task type

        Returns:
            ExecutionResult with all metrics and status
        """
        self.reset()

        try:
            # Phase 5.2: Validate preconditions
            validate_execution_preconditions(
                backend_type=backend_type,
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                allow_fallback=self.config.allow_cpu_fallback,
            )

            # Phase 5.1: Execute warmup runs
            self._current_phase = ExecutionPhase.WARMUP
            logger.info(
                f"Starting warmup phase: {self.config.warmup_runs} runs "
                f"(excluded from metrics)"
            )

            for i in range(self.config.warmup_runs):
                try:
                    latency = inference_fn()
                    self._warmup_latencies.append(latency)
                    if self.config.log_each_run:
                        logger.debug(
                            f"Warmup {i + 1}/{self.config.warmup_runs}: "
                            f"{latency:.2f}ms"
                        )
                except Exception as e:
                    if self.config.strict_mode:
                        raise BenchmarkError(
                            f"Warmup run {i + 1} failed: {e}"
                        ) from e
                    logger.warning(f"Warmup run {i + 1} failed: {e}")

            # Phase 5.1: Execute measured runs
            self._current_phase = ExecutionPhase.MEASUREMENT
            logger.info(
                f"Starting measurement phase: {self.config.measured_runs} runs"
            )

            for i in range(self.config.measured_runs):
                try:
                    latency = inference_fn()
                    self._measured_latencies.append(latency)
                    if self.config.log_each_run:
                        logger.debug(
                            f"Measured {i + 1}/{self.config.measured_runs}: "
                            f"{latency:.2f}ms"
                        )
                except Exception as e:
                    if self.config.strict_mode:
                        raise BenchmarkError(
                            f"Measured run {i + 1} failed: {e}"
                        ) from e
                    logger.warning(f"Measured run {i + 1} failed: {e}")

            # Phase 5.1: Aggregation
            self._current_phase = ExecutionPhase.AGGREGATION

            if not self._measured_latencies:
                raise BenchmarkError(
                    "No successful measured runs - cannot compute metrics"
                )

            self._current_phase = ExecutionPhase.COMPLETE
            logger.info(
                f"Benchmark complete: {len(self._measured_latencies)} measured runs"
            )

            return ExecutionResult(
                success=True,
                phase=ExecutionPhase.COMPLETE,
                warmup_latencies=self._warmup_latencies.copy(),
                measured_latencies=self._measured_latencies.copy(),
            )

        except BenchmarkError as e:
            self._current_phase = ExecutionPhase.FAILED
            logger.error(f"Benchmark failed: {e}")
            return ExecutionResult(
                success=False,
                phase=ExecutionPhase.FAILED,
                warmup_latencies=self._warmup_latencies.copy(),
                measured_latencies=self._measured_latencies.copy(),
                error_message=str(e),
                error_type=type(e).__name__,
            )

        except Exception as e:
            self._current_phase = ExecutionPhase.FAILED
            logger.error(f"Unexpected error during benchmark: {e}")
            return ExecutionResult(
                success=False,
                phase=ExecutionPhase.FAILED,
                warmup_latencies=self._warmup_latencies.copy(),
                measured_latencies=self._measured_latencies.copy(),
                error_message=f"Unexpected error: {e}",
                error_type=type(e).__name__,
            )


def enforce_no_fallback(backend_type: str, requested_backend: str) -> None:
    """Enforce that no CPU fallback occurred when Hailo was requested.

    Phase 5.2: CPU fallback NOT allowed by default for Hailo.

    Args:
        backend_type: Actual backend being used
        requested_backend: Originally requested backend

    Raises:
        HailoFallbackError: If Hailo was requested but PyTorch is being used
    """
    if requested_backend == "hailo" and backend_type == "pytorch":
        raise HailoFallbackError(
            "Hailo backend was requested but execution fell back to PyTorch. "
            "This is not allowed under Phase 5.2 requirements. "
            "Please ensure Hailo hardware is available and properly configured, "
            "or explicitly request PyTorch backend if intended."
        )


def get_supported_configurations() -> Dict[str, Dict[str, List[str]]]:
    """Get all supported Hailo configurations.

    Returns:
        Dictionary mapping YOLO versions to tasks to model lists
    """
    result = {}
    for version in HAILO_SUPPORTED_VERSIONS:
        result[version] = {}
        for task in HAILO_SUPPORTED_TASKS.get(version, []):
            models = HAILO_OPTIMIZED_MODELS.get(version, {}).get(task, [])
            result[version][task.value] = models
    return result


def format_supported_configurations() -> str:
    """Format supported configurations for display.

    Returns:
        Human-readable string of supported configurations
    """
    configs = get_supported_configurations()
    lines = ["Hailo-supported YOLO configurations:"]
    lines.append("-" * 40)

    for version, tasks in configs.items():
        lines.append(f"\nYOLO {version}:")
        for task_name, models in tasks.items():
            lines.append(f"  {task_name}:")
            for model in models:
                lines.append(f"    - {model}")

    return "\n".join(lines)
