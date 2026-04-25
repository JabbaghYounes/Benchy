"""Hailo runtime probes shared by the YOLO and LLM workloads.

The YOLO Hailo backend has carried these helpers since the original Hailo
integration; they were extracted here when Phase 2 added an LLM-on-NPU
metrics collector that needs the same probes. Keeping the implementations
in one place prevents the two workloads from drifting apart on power
readings and version detection.

All functions are intentionally side-effect-free and degrade gracefully
when `hailortcli` is missing, the `hailo_platform` package is not
installed, or the host has no Hailo-related sysfs nodes.
"""
from __future__ import annotations

import re
import subprocess
from glob import glob
from typing import Optional


def get_hailort_version() -> str:
    """Return the installed HailoRT version string, or "unknown".

    Tries `hailortcli --version` first (catches setups where the SDK is
    installed via the .deb but the Python bindings are not), then falls
    back to the `hailo_platform` package.
    """
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
        from hailo_platform import __version__  # type: ignore[import-not-found]
        return __version__
    except (ImportError, AttributeError):
        pass

    return "unknown"


def get_sdk_family() -> str:
    """Return "4.x" (Hailo-8 family), "5.x" (Hailo-10 family), or "unknown".

    Used by the YOLO Hailo backend's SDK-vs-device compatibility check; the
    LLM collector uses it only to label results.
    """
    version = get_hailort_version()
    if version == "unknown":
        return "unknown"
    match = re.search(r"(\d+)\.", version)
    if not match:
        return "unknown"
    major = int(match.group(1))
    if major == 4:
        return "4.x"
    if major >= 5:
        return "5.x"
    return "unknown"


def get_power_watts() -> Optional[float]:
    """Read the AI HAT+ subsystem power draw in watts.

    Order of preference:
      1. Raspberry Pi power sensors at `/sys/class/hwmon/*/power1_input`
         (microwatts) or `curr1_input` (milliamps, scaled by 5 V).
      2. `hailortcli measure-power` parsed for a watt value.

    Returns None if neither path produces a usable reading. Note: on a
    Pi 5 + AI HAT+ 2 during LLM-on-NPU inference, the HAT subsystem is the
    dominant consumer, so this is approximately the NPU power. It is *not*
    a chip-level NPU power reading.
    """
    for pattern in ("/sys/class/hwmon/hwmon*/power1_input",
                    "/sys/class/hwmon/hwmon*/curr1_input"):
        for path in glob(pattern):
            try:
                with open(path, "r") as f:
                    value = float(f.read().strip())
            except (IOError, ValueError):
                continue
            if "power" in path:
                return value / 1_000_000.0  # microwatts → watts
            if "curr" in path:
                return (value / 1000.0) * 5.0  # milliamps × 5 V

    try:
        result = subprocess.run(
            ["hailortcli", "measure-power"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "power" in line.lower():
                    match = re.search(r"(\d+\.?\d*)\s*[wW]", line)
                    if match:
                        return float(match.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_npu_utilization_percent() -> Optional[float]:
    """Return current NPU utilization as a 0–100 percentage, or None.

    HailoRT 5.x does not expose a stable scriptable utilization probe — the
    interactive `hailortcli monitor` TUI is the only first-party reading
    today, and there is no sysfs entry for utilization on the AI HAT+ 2.

    This function returns None until a working probe lands. It exists so
    the LLM metrics collector can wire the field through end-to-end (CSV
    column, JSON field, dashboard split) without conditional branches in
    the runner. When a probe ships, only this function changes.
    """
    return None
