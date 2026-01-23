# System information and resource utilization collectors
import os
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import psutil

from benchmark.schemas import SystemInfo, ResourceUtilization, Platform


def detect_platform() -> Platform:
    """Detect the hardware platform based on system characteristics."""
    # Check for Jetson by looking for tegra in device tree or /etc/nv_tegra_release
    if Path("/etc/nv_tegra_release").exists():
        return Platform.JETSON_ORIN_NANO

    # Check for Raspberry Pi
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "raspberry pi" in model:
                # Detect AI HAT variant - check for Hailo devices
                # AI HAT+ uses Hailo-8L, AI HAT+ 2 uses Hailo-8
                hailo_devices = list(Path("/dev").glob("hailo*"))
                if hailo_devices:
                    # Try to determine which HAT version
                    try:
                        result = subprocess.run(
                            ["hailortcli", "fw-control", "identify"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if "hailo8" in result.stdout.lower() and "hailo8l" not in result.stdout.lower():
                            return Platform.RPI_AI_HAT_PLUS_2
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                    return Platform.RPI_AI_HAT_PLUS
                else:
                    # Raspberry Pi detected but no Hailo device found
                    # Still return RPi platform (assume AI HAT+ may need setup)
                    return Platform.RPI_AI_HAT_PLUS
    except FileNotFoundError:
        pass

    # Default to Jetson Orin Nano for edge device context
    return Platform.JETSON_ORIN_NANO


def get_cpu_model() -> str:
    """Get CPU model string."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name") or line.startswith("Model"):
                    return line.split(":")[1].strip()
                # ARM processors often use "Hardware" field
                if line.startswith("Hardware"):
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "Unknown"


def get_accelerator_info() -> str:
    """Detect and return accelerator information."""
    # Check for NVIDIA GPU (Jetson)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for Hailo NPU
    hailo_info = detect_hailo_device()
    if hailo_info:
        return hailo_info

    return "None"


def detect_hailo_device() -> Optional[str]:
    """Detect Hailo NPU and return device information.

    Returns:
        Device string (e.g., "Hailo-8 (26 TOPS)") or None
    """
    # Check for device node
    hailo_devices = list(Path("/dev").glob("hailo*"))
    if not hailo_devices:
        return None

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
            if "hailo-8l" in output or "hailo8l" in output:
                return "Hailo-8L NPU (13 TOPS)"
            elif "hailo-8" in output or "hailo8" in output:
                return "Hailo-8 NPU (26 TOPS)"

            # Generic detection
            for line in result.stdout.split("\n"):
                if "device" in line.lower() or "hailo" in line.lower():
                    return line.strip()

            return "Hailo NPU"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: device exists but can't identify type
    return "Hailo NPU (unknown model)"


def get_hailo_device_type() -> Optional[str]:
    """Get the specific Hailo device type.

    Returns:
        "hailo8" or "hailo8l" or None
    """
    hailo_devices = list(Path("/dev").glob("hailo*"))
    if not hailo_devices:
        return None

    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "hailo-8l" in output or "hailo8l" in output:
                return "hailo8l"
            elif "hailo-8" in output or "hailo8" in output:
                return "hailo8"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def is_hailo_available() -> bool:
    """Check if a Hailo device is available.

    Returns:
        True if Hailo device is detected and accessible
    """
    hailo_devices = list(Path("/dev").glob("hailo*"))
    if not hailo_devices:
        return False

    try:
        result = subprocess.run(
            ["hailortcli", "scan"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "hailo" in result.stdout.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_ram_size_gb() -> float:
    """Get total RAM size in GB."""
    mem = psutil.virtual_memory()
    return round(mem.total / (1024 ** 3), 1)


def get_storage_type() -> str:
    """Detect storage type (SD card, SSD, etc.)."""
    root_device = None

    try:
        # Find the device for root filesystem
        result = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE", "/"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            root_device = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if root_device:
        # Check if it's an SD card (mmcblk) or SSD/NVMe
        if "mmcblk" in root_device:
            return "SD Card"
        elif "nvme" in root_device:
            return "NVMe SSD"
        elif "sd" in root_device:
            # Could be SATA SSD or USB storage
            return "SATA/USB Storage"

    return "Unknown"


def get_cooling_config() -> str:
    """Detect cooling configuration."""
    # Check for active fan control
    fan_paths = [
        "/sys/class/hwmon/hwmon*/fan*_input",
        "/sys/devices/pwm-fan/target_pwm",
        "/sys/class/thermal/cooling_device*/cur_state",
    ]

    for pattern in fan_paths:
        from glob import glob
        matches = glob(pattern)
        if matches:
            try:
                with open(matches[0], "r") as f:
                    value = int(f.read().strip())
                    if value > 0:
                        return "Active (Fan)"
            except (ValueError, IOError):
                continue

    return "Passive"


def get_power_mode() -> str:
    """Get current power mode (for Jetson devices)."""
    # Jetson power mode
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "NV Power Mode" in line or "MAXN" in line or "5W" in line or "10W" in line:
                    return line.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check CPU governor as fallback
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        pass

    return "Default"


def get_os_version() -> str:
    """Get OS version string."""
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=")[1].strip().strip('"')
    except FileNotFoundError:
        pass
    return f"{platform.system()} {platform.release()}"


def get_kernel_version() -> str:
    """Get kernel version."""
    return platform.release()


def collect_system_info(platform_override: Optional[Platform] = None) -> SystemInfo:
    """Collect complete system information.

    Args:
        platform_override: Optionally override auto-detected platform

    Returns:
        SystemInfo dataclass with all system details
    """
    detected_platform = platform_override or detect_platform()

    return SystemInfo(
        platform=detected_platform.value,
        cpu_model=get_cpu_model(),
        accelerator=get_accelerator_info(),
        ram_size_gb=get_ram_size_gb(),
        storage_type=get_storage_type(),
        cooling_config=get_cooling_config(),
        power_mode=get_power_mode(),
        os_version=get_os_version(),
        kernel_version=get_kernel_version(),
        hostname=socket.gethostname(),
    )


class ResourceMonitor:
    """Monitor resource utilization during benchmark execution."""

    def __init__(self, sample_interval: float = 0.1):
        """Initialize resource monitor.

        Args:
            sample_interval: Time between samples in seconds
        """
        self.sample_interval = sample_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start background resource monitoring."""
        if self._running:
            return

        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceUtilization:
        """Stop monitoring and return aggregated results."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        return self._aggregate_samples()

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            sample = self._collect_sample()
            with self._lock:
                self._samples.append(sample)
            time.sleep(self.sample_interval)

    def _collect_sample(self) -> dict:
        """Collect a single resource utilization sample."""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        sample = {
            "cpu_percent": cpu_percent,
            "memory_used_mb": memory.used / (1024 ** 2),
            "memory_total_mb": memory.total / (1024 ** 2),
            "accelerator_percent": None,
            "power_watts": None,
        }

        # Try to get GPU utilization (Jetson)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                sample["accelerator_percent"] = float(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        # Try to get power consumption (Jetson)
        power_paths = [
            "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input",
            "/sys/class/hwmon/hwmon*/power1_input",
        ]
        for pattern in power_paths:
            from glob import glob
            matches = glob(pattern)
            if matches:
                try:
                    with open(matches[0], "r") as f:
                        # Power is typically in milliwatts
                        power_mw = float(f.read().strip())
                        sample["power_watts"] = power_mw / 1000.0
                        break
                except (ValueError, IOError):
                    continue

        return sample

    def _aggregate_samples(self) -> ResourceUtilization:
        """Aggregate collected samples into final metrics."""
        with self._lock:
            samples = self._samples.copy()

        if not samples:
            return ResourceUtilization(
                cpu_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
            )

        # Calculate averages
        cpu_avg = sum(s["cpu_percent"] for s in samples) / len(samples)
        mem_used_avg = sum(s["memory_used_mb"] for s in samples) / len(samples)
        mem_total = samples[0]["memory_total_mb"]

        accel_samples = [s["accelerator_percent"] for s in samples if s["accelerator_percent"] is not None]
        accel_avg = sum(accel_samples) / len(accel_samples) if accel_samples else None

        power_samples = [s["power_watts"] for s in samples if s["power_watts"] is not None]
        power_avg = sum(power_samples) / len(power_samples) if power_samples else None

        return ResourceUtilization(
            cpu_percent=round(cpu_avg, 1),
            accelerator_percent=round(accel_avg, 1) if accel_avg else None,
            memory_used_mb=round(mem_used_avg, 1),
            memory_total_mb=round(mem_total, 1),
            power_watts=round(power_avg, 2) if power_avg else None,
        )

    def measure(self, func: Callable, *args, **kwargs):
        """Measure resource utilization during function execution.

        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Tuple of (function result, ResourceUtilization)
        """
        self.start()
        try:
            result = func(*args, **kwargs)
        finally:
            utilization = self.stop()
        return result, utilization
