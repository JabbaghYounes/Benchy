# Backend registry and auto-selection logic
import logging
from typing import Optional, Type

from benchmark.schemas import Platform
from benchmark.workloads.yolo.backends.base import YOLOBackend, BackendType
from benchmark.workloads.yolo.backends.pytorch import PyTorchBackend
from benchmark.workloads.yolo.backends.hailo import HailoBackend

logger = logging.getLogger(__name__)

# Registry of available backends
_BACKEND_REGISTRY: dict[BackendType, Type[YOLOBackend]] = {
    BackendType.PYTORCH: PyTorchBackend,
    BackendType.HAILO: HailoBackend,
}

# Platform to preferred backend mapping
_PLATFORM_BACKEND_MAP: dict[Platform, BackendType] = {
    Platform.JETSON_NANO: BackendType.PYTORCH,
    Platform.RPI_AI_HAT_PLUS: BackendType.HAILO,
    Platform.RPI_AI_HAT_PLUS_2: BackendType.HAILO,
}


def get_backend(
    backend_type: BackendType,
    device: str = "0",
    **kwargs,
) -> YOLOBackend:
    """Get a backend instance by type.

    Args:
        backend_type: The type of backend to create
        device: Device identifier
        **kwargs: Additional arguments for the backend

    Returns:
        Initialized backend instance

    Raises:
        ValueError: If backend type is not registered
        RuntimeError: If backend is not available
    """
    if backend_type not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Available backends: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_class = _BACKEND_REGISTRY[backend_type]
    backend = backend_class(device=device, **kwargs)

    if not backend.is_available():
        raise RuntimeError(
            f"Backend {backend_type.value} is not available on this system. "
            f"Check that required dependencies are installed."
        )

    return backend


def get_available_backends() -> list[BackendType]:
    """Get list of backends that are available on this system.

    Returns:
        List of available backend types
    """
    available = []

    for backend_type, backend_class in _BACKEND_REGISTRY.items():
        try:
            backend = backend_class()
            if backend.is_available():
                available.append(backend_type)
        except Exception as e:
            logger.debug(f"Backend {backend_type} not available: {e}")

    return available


def select_backend_for_platform(
    platform: Platform,
    fallback_to_pytorch: bool = False,
) -> BackendType:
    """Select the appropriate backend for a given platform.

    Args:
        platform: The detected platform
        fallback_to_pytorch: If True, fall back to PyTorch if preferred backend unavailable

    Returns:
        The selected backend type

    Raises:
        RuntimeError: If no suitable backend is available
    """
    # Get preferred backend for platform
    preferred = _PLATFORM_BACKEND_MAP.get(platform, BackendType.PYTORCH)
    logger.info(f"Platform {platform.value}: preferred backend is {preferred.value}")

    # Check if preferred backend is available
    try:
        backend_class = _BACKEND_REGISTRY[preferred]
        backend = backend_class()
        if backend.is_available():
            logger.info(f"Using {preferred.value} backend")
            return preferred
    except Exception as e:
        logger.warning(f"Preferred backend {preferred.value} check failed: {e}")

    # Preferred backend not available
    if not fallback_to_pytorch:
        raise RuntimeError(
            f"Preferred backend {preferred.value} is not available for {platform.value}. "
            f"CPU fallback is disabled. Install the required dependencies or enable fallback."
        )

    # Try fallback to PyTorch
    logger.warning(
        f"Preferred backend {preferred.value} not available, "
        f"falling back to PyTorch (CPU)"
    )

    try:
        pytorch_backend = PyTorchBackend()
        if pytorch_backend.is_available():
            return BackendType.PYTORCH
    except Exception as e:
        logger.error(f"PyTorch fallback also failed: {e}")

    raise RuntimeError(
        "No suitable backend available. "
        "Install Ultralytics/PyTorch or Hailo SDK."
    )


def auto_select_backend(
    platform: Optional[Platform] = None,
    force_backend: Optional[BackendType] = None,
    allow_fallback: bool = False,
    device: str = "0",
) -> YOLOBackend:
    """Automatically select and instantiate the best available backend.

    Priority:
    1. If force_backend is specified, use that (error if unavailable)
    2. If platform is specified, use platform's preferred backend
    3. Auto-detect platform and use appropriate backend

    Args:
        platform: Override platform detection
        force_backend: Force a specific backend
        allow_fallback: Allow fallback to PyTorch if preferred unavailable
        device: Device identifier

    Returns:
        Initialized backend instance

    Raises:
        RuntimeError: If no suitable backend is available
    """
    # If backend is forced, use it
    if force_backend is not None:
        logger.info(f"Using forced backend: {force_backend.value}")
        return get_backend(force_backend, device=device)

    # Detect platform if not provided
    if platform is None:
        from benchmark.metrics import detect_platform
        platform = detect_platform()
        logger.info(f"Auto-detected platform: {platform.value}")

    # Select backend for platform
    backend_type = select_backend_for_platform(
        platform,
        fallback_to_pytorch=allow_fallback,
    )

    return get_backend(backend_type, device=device)


def register_backend(
    backend_type: BackendType,
    backend_class: Type[YOLOBackend],
) -> None:
    """Register a new backend type.

    This can be used to add custom backends.

    Args:
        backend_type: Backend type identifier
        backend_class: Backend class (must inherit from YOLOBackend)
    """
    if not issubclass(backend_class, YOLOBackend):
        raise TypeError(
            f"Backend class must inherit from YOLOBackend, got {backend_class}"
        )

    _BACKEND_REGISTRY[backend_type] = backend_class
    logger.info(f"Registered backend: {backend_type.value}")


def get_backend_info() -> dict:
    """Get information about all registered backends.

    Returns:
        Dictionary with backend info
    """
    info = {}

    for backend_type, backend_class in _BACKEND_REGISTRY.items():
        try:
            backend = backend_class()
            info[backend_type.value] = {
                "available": backend.is_available(),
                "capabilities": {
                    "tasks": [t.value for t in backend.capabilities.supported_tasks],
                    "versions": backend.capabilities.supported_yolo_versions,
                    "gpu": backend.capabilities.supports_gpu,
                    "npu": backend.capabilities.supports_npu,
                    "requires_compilation": backend.capabilities.requires_compilation,
                },
                "version_info": backend.get_version_info(),
            }
        except Exception as e:
            info[backend_type.value] = {
                "available": False,
                "error": str(e),
            }

    return info
