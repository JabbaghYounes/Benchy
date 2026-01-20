# Hailo-specific Metric Collection
#
# This module provides metric collection utilities for Hailo NPU inference,
# including power consumption, memory usage, and performance statistics.
#
# Phase 4 - Task 4.2 of Hailo PRD
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import psutil
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HailoMetrics:
    """Comprehensive metrics for Hailo inference."""

    # Timing metrics
    inference_latency_ms: float = 0.0
    preprocessing_latency_ms: float = 0.0
    postprocessing_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Throughput
    fps: float = 0.0

    # Memory metrics
    host_memory_used_mb: float = 0.0
    host_memory_peak_mb: float = 0.0
    device_memory_used_mb: Optional[float] = None

    # Power metrics
    power_watts: Optional[float] = None
    power_idle_watts: Optional[float] = None
    power_inference_watts: Optional[float] = None
    energy_per_inference_mj: Optional[float] = None

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_percent_during_inference: float = 0.0

    # NPU utilization (if available)
    npu_utilization_percent: Optional[float] = None

    # Statistics over multiple runs
    latency_samples: List[float] = field(default_factory=list)
    latency_mean_ms: Optional[float] = None
    latency_std_ms: Optional[float] = None
    latency_min_ms: Optional[float] = None
    latency_max_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    def compute_statistics(self) -> None:
        """Compute statistical metrics from latency samples."""
        if not self.latency_samples:
            return

        samples = np.array(self.latency_samples)
        self.latency_mean_ms = float(np.mean(samples))
        self.latency_std_ms = float(np.std(samples))
        self.latency_min_ms = float(np.min(samples))
        self.latency_max_ms = float(np.max(samples))
        self.latency_p50_ms = float(np.percentile(samples, 50))
        self.latency_p95_ms = float(np.percentile(samples, 95))
        self.latency_p99_ms = float(np.percentile(samples, 99))

        # Calculate FPS from mean latency
        if self.latency_mean_ms > 0:
            self.fps = 1000.0 / self.latency_mean_ms

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "inference_latency_ms": self.inference_latency_ms,
            "preprocessing_latency_ms": self.preprocessing_latency_ms,
            "postprocessing_latency_ms": self.postprocessing_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "fps": self.fps,
            "host_memory_used_mb": self.host_memory_used_mb,
            "host_memory_peak_mb": self.host_memory_peak_mb,
            "device_memory_used_mb": self.device_memory_used_mb,
            "power_watts": self.power_watts,
            "power_idle_watts": self.power_idle_watts,
            "power_inference_watts": self.power_inference_watts,
            "energy_per_inference_mj": self.energy_per_inference_mj,
            "cpu_percent": self.cpu_percent,
            "cpu_percent_during_inference": self.cpu_percent_during_inference,
            "npu_utilization_percent": self.npu_utilization_percent,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }


class HailoMetricsCollector:
    """Collects metrics during Hailo NPU inference.

    This collector gathers:
    - Inference latency (NPU only, excluding pre/post processing)
    - Total latency (including pre/post processing)
    - Host memory usage
    - Power consumption (on supported platforms)
    - CPU utilization

    Usage:
        collector = HailoMetricsCollector()
        collector.start_monitoring()

        # Run inference
        latency = run_inference()
        collector.record_inference(latency)

        metrics = collector.stop_monitoring()
    """

    # Known power sensor paths on Raspberry Pi
    RPI_POWER_PATHS = [
        "/sys/class/hwmon/hwmon*/power1_input",
        "/sys/class/hwmon/hwmon*/curr1_input",
    ]

    def __init__(self, sample_interval: float = 0.05):
        """Initialize the metrics collector.

        Args:
            sample_interval: Interval between resource samples in seconds
        """
        self.sample_interval = sample_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Sample storage
        self._cpu_samples: List[float] = []
        self._memory_samples: List[float] = []
        self._power_samples: List[float] = []

        # Inference tracking
        self._inference_latencies: List[float] = []
        self._preprocessing_times: List[float] = []
        self._postprocessing_times: List[float] = []

        # Baseline measurements
        self._idle_power: Optional[float] = None
        self._initial_memory: float = 0.0

    def measure_idle_power(self, duration: float = 2.0) -> Optional[float]:
        """Measure idle power consumption.

        Args:
            duration: Duration to measure in seconds

        Returns:
            Average idle power in watts or None
        """
        samples = []
        end_time = time.time() + duration

        while time.time() < end_time:
            power = self._get_power_reading()
            if power is not None:
                samples.append(power)
            time.sleep(0.1)

        if samples:
            self._idle_power = sum(samples) / len(samples)
            logger.debug(f"Idle power: {self._idle_power:.2f}W")
            return self._idle_power

        return None

    def start_monitoring(self) -> None:
        """Start background resource monitoring."""
        if self._monitoring:
            return

        # Record initial state
        self._initial_memory = psutil.Process().memory_info().rss / (1024 ** 2)

        # Clear previous samples
        with self._lock:
            self._cpu_samples = []
            self._memory_samples = []
            self._power_samples = []
            self._inference_latencies = []
            self._preprocessing_times = []
            self._postprocessing_times = []

        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> HailoMetrics:
        """Stop monitoring and return aggregated metrics.

        Returns:
            HailoMetrics with all collected data
        """
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        return self._aggregate_metrics()

    def record_inference(
        self,
        inference_latency_ms: float,
        preprocessing_ms: float = 0.0,
        postprocessing_ms: float = 0.0,
    ) -> None:
        """Record a single inference run.

        Args:
            inference_latency_ms: NPU inference time in milliseconds
            preprocessing_ms: Preprocessing time in milliseconds
            postprocessing_ms: Postprocessing time in milliseconds
        """
        with self._lock:
            self._inference_latencies.append(inference_latency_ms)
            self._preprocessing_times.append(preprocessing_ms)
            self._postprocessing_times.append(postprocessing_ms)

    def _monitoring_loop(self) -> None:
        """Background monitoring thread loop."""
        while self._monitoring:
            sample = self._collect_sample()

            with self._lock:
                self._cpu_samples.append(sample["cpu_percent"])
                self._memory_samples.append(sample["memory_mb"])
                if sample["power_watts"] is not None:
                    self._power_samples.append(sample["power_watts"])

            time.sleep(self.sample_interval)

    def _collect_sample(self) -> Dict:
        """Collect a single resource sample."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_mb": psutil.Process().memory_info().rss / (1024 ** 2),
            "power_watts": self._get_power_reading(),
        }

    def _get_power_reading(self) -> Optional[float]:
        """Get current power consumption reading.

        Returns:
            Power in watts or None if not available
        """
        # Try Raspberry Pi power sensors
        for pattern in self.RPI_POWER_PATHS:
            from glob import glob
            matches = glob(pattern)
            for path in matches:
                try:
                    with open(path, "r") as f:
                        value = float(f.read().strip())
                        # Convert based on sensor type
                        if "power" in path:
                            # Power is typically in microwatts
                            return value / 1_000_000.0
                        elif "curr" in path:
                            # Current in milliamps, assume 5V
                            return (value / 1000.0) * 5.0
                except (IOError, ValueError):
                    continue

        # Try to get power via hailortcli (if available)
        try:
            result = subprocess.run(
                ["hailortcli", "measure-power"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                # Parse power from output
                for line in result.stdout.split("\n"):
                    if "power" in line.lower() and "w" in line.lower():
                        # Try to extract numeric value
                        import re
                        match = re.search(r"(\d+\.?\d*)\s*[wW]", line)
                        if match:
                            return float(match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def _aggregate_metrics(self) -> HailoMetrics:
        """Aggregate all collected samples into final metrics."""
        metrics = HailoMetrics()

        with self._lock:
            # Inference latencies
            if self._inference_latencies:
                metrics.latency_samples = self._inference_latencies.copy()
                metrics.inference_latency_ms = sum(self._inference_latencies) / len(self._inference_latencies)

            if self._preprocessing_times:
                metrics.preprocessing_latency_ms = sum(self._preprocessing_times) / len(self._preprocessing_times)

            if self._postprocessing_times:
                metrics.postprocessing_latency_ms = sum(self._postprocessing_times) / len(self._postprocessing_times)

            metrics.total_latency_ms = (
                metrics.inference_latency_ms +
                metrics.preprocessing_latency_ms +
                metrics.postprocessing_latency_ms
            )

            # CPU samples
            if self._cpu_samples:
                metrics.cpu_percent = sum(self._cpu_samples) / len(self._cpu_samples)
                # Get CPU during inference (higher samples)
                sorted_cpu = sorted(self._cpu_samples, reverse=True)
                top_quarter = sorted_cpu[:max(1, len(sorted_cpu) // 4)]
                metrics.cpu_percent_during_inference = sum(top_quarter) / len(top_quarter)

            # Memory samples
            if self._memory_samples:
                metrics.host_memory_used_mb = sum(self._memory_samples) / len(self._memory_samples)
                metrics.host_memory_peak_mb = max(self._memory_samples)

            # Power samples
            if self._power_samples:
                metrics.power_watts = sum(self._power_samples) / len(self._power_samples)
                if self._idle_power is not None:
                    metrics.power_idle_watts = self._idle_power
                    metrics.power_inference_watts = metrics.power_watts
                    # Energy per inference (mJ = W * s * 1000)
                    if metrics.inference_latency_ms > 0:
                        inference_time_s = metrics.inference_latency_ms / 1000.0
                        power_delta = metrics.power_watts - self._idle_power
                        metrics.energy_per_inference_mj = power_delta * inference_time_s * 1000

        # Compute statistics
        metrics.compute_statistics()

        return metrics


class InferenceTimer:
    """Context manager for timing inference runs.

    Usage:
        with InferenceTimer() as timer:
            # Run inference
            pass
        latency_ms = timer.elapsed_ms
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "InferenceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0


def measure_inference(
    inference_fn: Callable,
    num_runs: int = 10,
    warmup_runs: int = 3,
    collector: Optional[HailoMetricsCollector] = None,
) -> Tuple[List[float], HailoMetrics]:
    """Measure inference performance over multiple runs.

    Args:
        inference_fn: Function that performs a single inference (returns latency_ms)
        num_runs: Number of measured runs
        warmup_runs: Number of warmup runs (excluded from metrics)
        collector: Optional metrics collector

    Returns:
        Tuple of (list of latencies, HailoMetrics)
    """
    if collector is None:
        collector = HailoMetricsCollector()

    # Warmup runs
    logger.info(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        inference_fn()

    # Measure idle power before benchmark
    collector.measure_idle_power(duration=1.0)

    # Start monitoring
    collector.start_monitoring()

    # Measured runs
    logger.info(f"Running {num_runs} measured iterations...")
    latencies = []

    for i in range(num_runs):
        with InferenceTimer() as timer:
            inference_fn()

        latencies.append(timer.elapsed_ms)
        collector.record_inference(timer.elapsed_ms)

    # Stop monitoring and get metrics
    metrics = collector.stop_monitoring()

    return latencies, metrics


def get_hailo_device_memory() -> Optional[float]:
    """Get Hailo device memory usage if available.

    Returns:
        Memory usage in MB or None
    """
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse memory info from output
            for line in result.stdout.split("\n"):
                if "memory" in line.lower():
                    import re
                    match = re.search(r"(\d+\.?\d*)\s*(MB|KB|GB)", line, re.IGNORECASE)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2).upper()
                        if unit == "KB":
                            return value / 1024.0
                        elif unit == "GB":
                            return value * 1024.0
                        return value
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_npu_utilization() -> Optional[float]:
    """Get Hailo NPU utilization if available.

    Note: Hailo doesn't expose utilization directly in most cases.
    This function attempts to estimate or retrieve it if possible.

    Returns:
        Utilization percentage or None
    """
    # Hailo NPUs don't expose utilization metrics directly
    # This is a placeholder for future implementation if Hailo adds this feature
    return None
