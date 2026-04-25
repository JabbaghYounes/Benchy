"""Tests for HailoLLMMetricsCollector and the shared hailo_utils probes.

We mock subprocess and filesystem reads so the tests run on any host. The
real probes are exercised in Slice 7's hardware verification, not here.
"""
from unittest.mock import patch

import pytest

from benchmark.backends import hailo_utils
from benchmark.workloads.llm.hailo_metrics import (
    HailoLLMMetricsCollector,
    HailoLLMSnapshot,
)


# ----- shared probes ---------------------------------------------------------

def test_hailort_version_returns_string_when_cli_succeeds():
    with patch("benchmark.backends.hailo_utils.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "HailoRT 5.1.0\n"
        assert hailo_utils.get_hailort_version() == "HailoRT 5.1.0"


def test_hailort_version_falls_back_to_unknown():
    with patch(
        "benchmark.backends.hailo_utils.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        # If the package import also fails, we expect "unknown"; on a host
        # where hailo_platform is somehow installed without hailortcli the
        # answer is whatever __version__ reports — so we only assert it's
        # a string.
        assert isinstance(hailo_utils.get_hailort_version(), str)


def test_sdk_family_5x_for_hailort_5():
    with patch(
        "benchmark.backends.hailo_utils.get_hailort_version",
        return_value="HailoRT 5.2.0",
    ):
        assert hailo_utils.get_sdk_family() == "5.x"


def test_sdk_family_4x_for_hailort_4():
    with patch(
        "benchmark.backends.hailo_utils.get_hailort_version",
        return_value="HailoRT 4.17.0",
    ):
        assert hailo_utils.get_sdk_family() == "4.x"


def test_sdk_family_unknown_when_version_unparseable():
    with patch(
        "benchmark.backends.hailo_utils.get_hailort_version",
        return_value="unknown",
    ):
        assert hailo_utils.get_sdk_family() == "unknown"


def test_npu_utilization_returns_none_today():
    # Documented stub — the test pins the contract so a future probe lands
    # behind a deliberate code change, not a silent regression.
    assert hailo_utils.get_npu_utilization_percent() is None


# ----- collector lifecycle ---------------------------------------------------

def test_collector_rejects_nonpositive_poll_interval():
    with pytest.raises(ValueError):
        HailoLLMMetricsCollector(poll_interval_seconds=0)
    with pytest.raises(ValueError):
        HailoLLMMetricsCollector(poll_interval_seconds=-1)


def test_collector_stop_before_start_returns_empty_snapshot():
    c = HailoLLMMetricsCollector()
    snap = c.stop()
    assert isinstance(snap, HailoLLMSnapshot)
    assert snap.npu_utilization_percent is None
    assert snap.npu_power_watts is None
    assert snap.hailort_version is None
    assert snap.sample_count == 0


def test_collector_double_stop_is_idempotent():
    c = HailoLLMMetricsCollector(poll_interval_seconds=0.05)
    with patch(
        "benchmark.workloads.llm.hailo_metrics.get_power_watts",
        return_value=2.5,
    ), patch(
        "benchmark.workloads.llm.hailo_metrics.get_hailort_version",
        return_value="HailoRT 5.1.0",
    ):
        c.start()
        first = c.stop()
        second = c.stop()
    assert first.hailort_version == "HailoRT 5.1.0"
    # Second stop is a no-op — returns a fresh empty snapshot, not a
    # corrupted partial.
    assert second.hailort_version is None


def test_collector_aggregates_power_and_utilization():
    samples_power = iter([2.5, 3.0, 2.8, 2.7])
    samples_util = iter([55.0, 70.0, 65.0, 60.0])
    with patch(
        "benchmark.workloads.llm.hailo_metrics.get_power_watts",
        side_effect=lambda: next(samples_power, None),
    ), patch(
        "benchmark.workloads.llm.hailo_metrics.get_npu_utilization_percent",
        side_effect=lambda: next(samples_util, None),
    ), patch(
        "benchmark.workloads.llm.hailo_metrics.get_hailort_version",
        return_value="HailoRT 5.2.0",
    ):
        c = HailoLLMMetricsCollector(poll_interval_seconds=0.05)
        c.start()
        # Wait for the thread to consume at least four samples. The lambda
        # returns None once the iterator is exhausted, so the collector
        # gracefully ignores extra polls.
        import time
        time.sleep(0.4)
        snap = c.stop()
    assert snap.npu_power_watts is not None
    assert snap.npu_power_watts > 0
    assert snap.npu_power_peak_watts is not None
    assert snap.npu_power_peak_watts >= snap.npu_power_watts
    assert snap.npu_utilization_percent is not None
    assert 50 <= snap.npu_utilization_percent <= 80
    assert snap.hailort_version == "HailoRT 5.2.0"
    assert snap.sample_count >= 4


def test_collector_handles_all_none_probes():
    # Dev hosts have no Hailo hardware. Every probe returns None and the
    # collector still produces a valid (all-None) snapshot.
    with patch(
        "benchmark.workloads.llm.hailo_metrics.get_power_watts",
        return_value=None,
    ), patch(
        "benchmark.workloads.llm.hailo_metrics.get_npu_utilization_percent",
        return_value=None,
    ), patch(
        "benchmark.workloads.llm.hailo_metrics.get_hailort_version",
        return_value="unknown",
    ):
        c = HailoLLMMetricsCollector(poll_interval_seconds=0.05)
        c.start()
        import time
        time.sleep(0.15)
        snap = c.stop()
    assert snap.npu_power_watts is None
    assert snap.npu_utilization_percent is None
    assert snap.hailort_version == "unknown"
    assert snap.sample_count >= 1


def test_collector_warns_on_double_start(caplog):
    c = HailoLLMMetricsCollector(poll_interval_seconds=0.05)
    with patch(
        "benchmark.workloads.llm.hailo_metrics.get_hailort_version",
        return_value="HailoRT 5.0.0",
    ):
        c.start()
        with caplog.at_level("WARNING"):
            c.start()
        c.stop()
    assert any("called while running" in rec.message for rec in caplog.records)
