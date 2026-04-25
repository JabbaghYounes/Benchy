"""LLM-on-NPU metrics collector for Hailo-10H runs.

The standard `ResourceMonitor` (`benchmark/metrics/collectors.py`) covers
host-side CPU / RAM / Pi power readings. When the LLM is running on the
Hailo-10H (HailoRT GenAI's Ollama-compatible REST endpoint), the host CPU
is mostly idle and the meaningful signal lives on the NPU side: power
draw on the AI HAT+ subsystem, NPU utilization (when a probe exists), and
the HailoRT version label.

This collector mirrors `ResourceMonitor`'s start/stop/get-snapshot shape so
the LLM runner can wire it alongside without conditional branches.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

from benchmark.backends.hailo_utils import (
    get_hailort_version,
    get_npu_utilization_percent,
    get_power_watts,
)

logger = logging.getLogger(__name__)


@dataclass
class HailoLLMSnapshot:
    """Aggregated NPU-side readings over a measured run.

    Means are computed across samples taken between start() and stop();
    peaks track the maximum value seen. None means the probe never produced
    a usable reading (e.g. utilization on a HailoRT 5.x build that does not
    expose a scriptable probe yet).
    """
    npu_utilization_percent: Optional[float] = None
    npu_utilization_peak_percent: Optional[float] = None
    npu_power_watts: Optional[float] = None
    npu_power_peak_watts: Optional[float] = None
    hailort_version: Optional[str] = None
    sample_count: int = 0


class HailoLLMMetricsCollector:
    """Polls Hailo NPU readings on a background thread.

    Usage:
        collector = HailoLLMMetricsCollector()
        collector.start()
        # ... run inference ...
        snapshot = collector.stop()

    Polling cadence defaults to 250 ms. The collector is robust to a
    missing HailoRT install — if every probe returns None, stop() returns
    a snapshot with all-None fields and `sample_count` reflecting how many
    times the thread tried.
    """

    def __init__(self, poll_interval_seconds: float = 0.25):
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._poll_interval = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._utilization_samples: List[float] = []
        self._power_samples: List[float] = []
        self._sample_count = 0
        self._hailort_version: Optional[str] = None
        self._running = False

    def start(self) -> None:
        """Begin background sampling. Idempotent on re-start after stop."""
        if self._running:
            logger.warning("HailoLLMMetricsCollector.start() called while running")
            return
        self._stop_event.clear()
        self._utilization_samples = []
        self._power_samples = []
        self._sample_count = 0
        self._hailort_version = get_hailort_version()
        self._thread = threading.Thread(
            target=self._poll_loop, name="hailo-llm-metrics", daemon=True
        )
        self._running = True
        self._thread.start()

    def stop(self) -> HailoLLMSnapshot:
        """Stop sampling and return the aggregated snapshot. Idempotent."""
        if not self._running:
            return HailoLLMSnapshot()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._running = False
        return self._build_snapshot()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_count += 1
            util = get_npu_utilization_percent()
            if util is not None:
                self._utilization_samples.append(util)
            power = get_power_watts()
            if power is not None:
                self._power_samples.append(power)
            # Wait with the event so stop() interrupts us promptly.
            self._stop_event.wait(self._poll_interval)

    def _build_snapshot(self) -> HailoLLMSnapshot:
        def mean_or_none(values: List[float]) -> Optional[float]:
            return round(sum(values) / len(values), 3) if values else None

        def max_or_none(values: List[float]) -> Optional[float]:
            return round(max(values), 3) if values else None

        return HailoLLMSnapshot(
            npu_utilization_percent=mean_or_none(self._utilization_samples),
            npu_utilization_peak_percent=max_or_none(self._utilization_samples),
            npu_power_watts=mean_or_none(self._power_samples),
            npu_power_peak_watts=max_or_none(self._power_samples),
            hailort_version=self._hailort_version,
            sample_count=self._sample_count,
        )
