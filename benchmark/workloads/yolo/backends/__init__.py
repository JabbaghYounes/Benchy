# YOLO Backend implementations
from benchmark.workloads.yolo.backends.base import (
    YOLOBackend,
    BackendType,
    BackendCapabilities,
    ModelInfo,
)
from benchmark.workloads.yolo.backends.pytorch import PyTorchBackend
from benchmark.workloads.yolo.backends.hailo import HailoBackend
from benchmark.workloads.yolo.backends.registry import (
    get_backend,
    get_available_backends,
    select_backend_for_platform,
)

__all__ = [
    "YOLOBackend",
    "BackendType",
    "BackendCapabilities",
    "ModelInfo",
    "PyTorchBackend",
    "HailoBackend",
    "get_backend",
    "get_available_backends",
    "select_backend_for_platform",
]
