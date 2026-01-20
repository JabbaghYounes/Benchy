# Model artifact caching for Hailo conversion pipeline
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.schemas import YOLOTask

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path("models/hailo")


@dataclass
class CacheMetadata:
    """Metadata for cached model artifacts.

    This metadata is used to determine if cached artifacts are still valid
    or need to be regenerated due to version changes.
    """

    # Model information
    model_name: str
    yolo_version: str
    task: str
    input_resolution: int

    # Version information for cache invalidation
    ultralytics_version: str
    hailo_sdk_version: Optional[str] = None
    hailo_compiler_version: Optional[str] = None
    hailort_version: Optional[str] = None

    # Target device
    target_device: str = "hailo8l"  # hailo8 or hailo8l

    # File hashes for integrity verification
    pt_model_hash: Optional[str] = None
    onnx_hash: Optional[str] = None
    har_hash: Optional[str] = None
    hef_hash: Optional[str] = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    onnx_created_at: Optional[str] = None
    har_created_at: Optional[str] = None
    hef_created_at: Optional[str] = None

    # Calibration info
    calibration_dataset: Optional[str] = None
    calibration_images: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CacheMetadata":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"Saved cache metadata to {path}")

    @classmethod
    def load(cls, path: Path) -> Optional["CacheMetadata"]:
        """Load metadata from JSON file."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None


def get_cache_path(
    model_name: str,
    yolo_version: str,
    task: YOLOTask,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Get the cache directory path for a model.

    Cache structure:
        models/hailo/{yolo_version}/{task}/{model_name}/
            ├── model.onnx
            ├── model.har
            ├── model.hef
            └── metadata.json

    Args:
        model_name: Model name (e.g., "yolov8n.pt" or "yolov8n")
        yolo_version: YOLO version (e.g., "v8")
        task: Task type
        cache_dir: Optional custom cache directory

    Returns:
        Path to the cache directory for this model
    """
    base_dir = cache_dir or DEFAULT_CACHE_DIR

    # Remove file extension from model name
    base_name = Path(model_name).stem

    return base_dir / yolo_version / task.value / base_name


def compute_file_hash(path: Path, algorithm: str = "sha256") -> Optional[str]:
    """Compute hash of a file for integrity verification.

    Args:
        path: Path to the file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest of the hash, or None if file doesn't exist
    """
    if not path.exists():
        return None

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_ultralytics_version() -> str:
    """Get installed Ultralytics version."""
    try:
        import ultralytics
        return ultralytics.__version__
    except ImportError:
        return "not_installed"


def get_hailo_sdk_version() -> Optional[str]:
    """Get Hailo SDK version."""
    try:
        # Try hailo_sdk module
        from hailo_sdk_client import __version__
        return __version__
    except ImportError:
        pass

    # Try getting version from CLI tool
    try:
        result = subprocess.run(
            ["hailo", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_hailort_version() -> Optional[str]:
    """Get HailoRT version."""
    try:
        result = subprocess.run(
            ["hailortcli", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse version from output
            output = result.stdout.strip()
            # hailortcli outputs like "HailoRT CLI version X.Y.Z"
            return output
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        from hailo_platform import __version__
        return __version__
    except ImportError:
        pass

    return None


def get_dataflow_compiler_version() -> Optional[str]:
    """Get Hailo Dataflow Compiler version."""
    try:
        from hailo_sdk_client import ClientRunner
        # ClientRunner version reflects compiler version
        return getattr(ClientRunner, "__version__", None)
    except ImportError:
        pass

    return None


class ModelCache:
    """Manager for cached model artifacts.

    Handles:
    - Cache directory structure
    - Metadata management
    - Version-based cache invalidation
    - File integrity verification
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the model cache.

        Args:
            cache_dir: Root directory for cached models
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

    def get_model_cache_path(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Path:
        """Get cache directory for a specific model."""
        return get_cache_path(model_name, yolo_version, task, self.cache_dir)

    def get_onnx_path(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Path:
        """Get path to cached ONNX file."""
        return self.get_model_cache_path(model_name, yolo_version, task) / "model.onnx"

    def get_har_path(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Path:
        """Get path to cached HAR file."""
        return self.get_model_cache_path(model_name, yolo_version, task) / "model.har"

    def get_hef_path(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Path:
        """Get path to cached HEF file."""
        return self.get_model_cache_path(model_name, yolo_version, task) / "model.hef"

    def get_metadata_path(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Path:
        """Get path to cache metadata file."""
        return self.get_model_cache_path(model_name, yolo_version, task) / "metadata.json"

    def get_metadata(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> Optional[CacheMetadata]:
        """Load metadata for a cached model."""
        metadata_path = self.get_metadata_path(model_name, yolo_version, task)
        return CacheMetadata.load(metadata_path)

    def save_metadata(
        self,
        metadata: CacheMetadata,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> None:
        """Save metadata for a cached model."""
        metadata_path = self.get_metadata_path(model_name, yolo_version, task)
        metadata.save(metadata_path)

    def has_valid_cache(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        target_device: str = "hailo8l",
        input_resolution: int = 640,
    ) -> bool:
        """Check if valid cached artifacts exist.

        Cache is considered valid if:
        1. HEF file exists
        2. Metadata exists and matches current versions
        3. Target device matches

        Args:
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            target_device: Target Hailo device
            input_resolution: Input resolution

        Returns:
            True if valid cache exists
        """
        hef_path = self.get_hef_path(model_name, yolo_version, task)
        if not hef_path.exists():
            logger.debug(f"Cache miss: HEF not found at {hef_path}")
            return False

        metadata = self.get_metadata(model_name, yolo_version, task)
        if metadata is None:
            logger.debug("Cache miss: No metadata found")
            return False

        # Check target device
        if metadata.target_device != target_device:
            logger.debug(
                f"Cache miss: Target device mismatch "
                f"(cached={metadata.target_device}, requested={target_device})"
            )
            return False

        # Check input resolution
        if metadata.input_resolution != input_resolution:
            logger.debug(
                f"Cache miss: Resolution mismatch "
                f"(cached={metadata.input_resolution}, requested={input_resolution})"
            )
            return False

        # Check Ultralytics version
        current_ultralytics = get_ultralytics_version()
        if metadata.ultralytics_version != current_ultralytics:
            logger.debug(
                f"Cache miss: Ultralytics version mismatch "
                f"(cached={metadata.ultralytics_version}, current={current_ultralytics})"
            )
            return False

        # Check HailoRT version if available
        current_hailort = get_hailort_version()
        if current_hailort and metadata.hailort_version:
            if metadata.hailort_version != current_hailort:
                logger.debug(
                    f"Cache miss: HailoRT version mismatch "
                    f"(cached={metadata.hailort_version}, current={current_hailort})"
                )
                return False

        logger.info(f"Cache hit: Using cached HEF at {hef_path}")
        return True

    def has_onnx(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> bool:
        """Check if ONNX file exists in cache."""
        return self.get_onnx_path(model_name, yolo_version, task).exists()

    def has_har(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> bool:
        """Check if HAR file exists in cache."""
        return self.get_har_path(model_name, yolo_version, task).exists()

    def has_hef(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> bool:
        """Check if HEF file exists in cache."""
        return self.get_hef_path(model_name, yolo_version, task).exists()

    def clear_cache(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
    ) -> None:
        """Clear cached artifacts for a model."""
        cache_path = self.get_model_cache_path(model_name, yolo_version, task)
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            logger.info(f"Cleared cache at {cache_path}")

    def clear_all(self) -> None:
        """Clear all cached artifacts."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared all cache at {self.cache_dir}")

    def create_metadata(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        input_resolution: int,
        target_device: str,
    ) -> CacheMetadata:
        """Create new cache metadata with current version info.

        Args:
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            input_resolution: Input resolution
            target_device: Target Hailo device

        Returns:
            New CacheMetadata instance
        """
        return CacheMetadata(
            model_name=model_name,
            yolo_version=yolo_version,
            task=task.value,
            input_resolution=input_resolution,
            target_device=target_device,
            ultralytics_version=get_ultralytics_version(),
            hailo_sdk_version=get_hailo_sdk_version(),
            hailo_compiler_version=get_dataflow_compiler_version(),
            hailort_version=get_hailort_version(),
        )

    def list_cached_models(self) -> list[dict]:
        """List all cached models.

        Returns:
            List of dictionaries with model info
        """
        models = []

        if not self.cache_dir.exists():
            return models

        for version_dir in self.cache_dir.iterdir():
            if not version_dir.is_dir():
                continue

            for task_dir in version_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                for model_dir in task_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    metadata_path = model_dir / "metadata.json"
                    metadata = CacheMetadata.load(metadata_path)

                    models.append({
                        "model_name": model_dir.name,
                        "yolo_version": version_dir.name,
                        "task": task_dir.name,
                        "has_onnx": (model_dir / "model.onnx").exists(),
                        "has_har": (model_dir / "model.har").exists(),
                        "has_hef": (model_dir / "model.hef").exists(),
                        "metadata": metadata.to_dict() if metadata else None,
                    })

        return models
