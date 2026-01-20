# Calibration Dataset Handling for Hailo Model Compilation
#
# This module provides deterministic calibration dataset handling
# for Hailo model quantization. It uses Ultralytics validation
# datasets with fixed subsets and deterministic ordering.
#
# Phase 3 - Task 3.1 of Hailo PRD
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from benchmark.schemas import YOLOTask

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for calibration dataset loading."""

    # Number of images to use for calibration
    num_samples: int = 100

    # Input resolution (square)
    input_resolution: int = 640

    # Random seed for deterministic shuffling
    seed: int = 42

    # Cache directory for preprocessed calibration data
    cache_dir: Optional[Path] = None

    # Dataset path override (uses Ultralytics default if None)
    dataset_path: Optional[Path] = None


@dataclass
class CalibrationDataset:
    """Container for calibration dataset with metadata."""

    # Preprocessed images ready for model input
    images: List[np.ndarray]

    # Original image paths (for traceability)
    image_paths: List[Path]

    # Configuration used to create this dataset
    config: CalibrationConfig

    # Dataset hash for cache invalidation
    dataset_hash: str = ""

    # Task type
    task: YOLOTask = YOLOTask.DETECTION

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def to_numpy_batch(self) -> np.ndarray:
        """Convert to a single numpy batch array.

        Returns:
            Array of shape (N, H, W, C) where N is num_samples
        """
        return np.stack(self.images, axis=0)


class CalibrationDatasetLoader:
    """Loads calibration datasets from Ultralytics with deterministic ordering.

    This class ensures that:
    1. The same images are used across runs for reproducibility
    2. Images are loaded in a deterministic order (sorted by filename)
    3. A fixed subset is selected using a seeded RNG
    4. Images are preprocessed consistently for Hailo input

    Usage:
        loader = CalibrationDatasetLoader()
        dataset = loader.load(YOLOTask.DETECTION, config)
        # Use dataset.images for calibration
    """

    # Default datasets per task type (from Ultralytics)
    DEFAULT_DATASETS = {
        YOLOTask.DETECTION: "coco128",
        YOLOTask.SEGMENTATION: "coco128-seg",
        YOLOTask.POSE: "coco8-pose",
        YOLOTask.OBB: "dota8",
        YOLOTask.CLASSIFICATION: "imagenet10",
    }

    # Supported image extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the calibration dataset loader.

        Args:
            cache_dir: Directory for caching preprocessed calibration data
        """
        self.cache_dir = cache_dir or Path("models/hailo/calibration_cache")
        self._ultralytics_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ultralytics is available for dataset loading."""
        if self._ultralytics_available is not None:
            return self._ultralytics_available

        try:
            from ultralytics import YOLO
            from ultralytics.data.utils import check_det_dataset

            self._ultralytics_available = True
        except ImportError:
            self._ultralytics_available = False

        return self._ultralytics_available

    def load(
        self,
        task: YOLOTask,
        config: Optional[CalibrationConfig] = None,
    ) -> CalibrationDataset:
        """Load a calibration dataset for the specified task.

        Args:
            task: YOLO task type
            config: Calibration configuration

        Returns:
            CalibrationDataset with preprocessed images

        Raises:
            RuntimeError: If dataset cannot be loaded
        """
        config = config or CalibrationConfig()

        logger.info(f"Loading calibration dataset for task: {task.value}")
        logger.info(f"  Num samples: {config.num_samples}")
        logger.info(f"  Input resolution: {config.input_resolution}")
        logger.info(f"  Seed: {config.seed}")

        # Try to load from cache first
        cached = self._load_from_cache(task, config)
        if cached is not None:
            logger.info("Loaded calibration dataset from cache")
            return cached

        # Get image paths from dataset
        image_paths = self._get_dataset_images(task, config)

        if len(image_paths) == 0:
            raise RuntimeError(
                f"No images found for task {task.value}. "
                f"Ensure Ultralytics datasets are available."
            )

        logger.info(f"Found {len(image_paths)} images in dataset")

        # Select deterministic subset
        selected_paths = self._select_subset(image_paths, config)
        logger.info(f"Selected {len(selected_paths)} images for calibration")

        # Load and preprocess images
        images = self._load_and_preprocess(selected_paths, config)

        # Compute dataset hash for cache invalidation
        dataset_hash = self._compute_hash(selected_paths, config)

        # Create dataset container
        dataset = CalibrationDataset(
            images=images,
            image_paths=selected_paths,
            config=config,
            dataset_hash=dataset_hash,
            task=task,
            metadata={
                "total_images_available": len(image_paths),
                "selected_images": len(selected_paths),
                "loaded_images": len(images),
            },
        )

        # Save to cache
        self._save_to_cache(dataset, task)

        return dataset

    def _get_dataset_images(
        self,
        task: YOLOTask,
        config: CalibrationConfig,
    ) -> List[Path]:
        """Get image paths from the Ultralytics dataset.

        Args:
            task: Task type
            config: Configuration

        Returns:
            List of image paths sorted alphabetically
        """
        # Use custom path if provided
        if config.dataset_path is not None:
            return self._get_images_from_path(config.dataset_path)

        # Use Ultralytics dataset
        dataset_name = self.DEFAULT_DATASETS.get(task)
        if dataset_name is None:
            raise ValueError(f"No default dataset for task: {task.value}")

        return self._get_ultralytics_dataset_images(dataset_name, task)

    def _get_ultralytics_dataset_images(
        self,
        dataset_name: str,
        task: YOLOTask,
    ) -> List[Path]:
        """Get images from an Ultralytics dataset.

        Args:
            dataset_name: Name of the Ultralytics dataset
            task: Task type

        Returns:
            List of sorted image paths
        """
        if not self.is_available():
            logger.warning("Ultralytics not available, using fallback")
            return self._get_fallback_images()

        try:
            # Import Ultralytics utilities
            from ultralytics.data.utils import check_det_dataset
            from ultralytics.utils import DATASETS_DIR

            # For classification, dataset structure is different
            if task == YOLOTask.CLASSIFICATION:
                return self._get_classification_dataset_images(dataset_name)

            # Check/download dataset
            dataset_yaml = f"{dataset_name}.yaml"
            logger.debug(f"Loading dataset: {dataset_yaml}")

            try:
                data_dict = check_det_dataset(dataset_yaml)
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_yaml}: {e}")
                return self._get_fallback_images()

            # Get validation images path
            val_path = data_dict.get("val")
            if val_path is None:
                # Fall back to train path
                val_path = data_dict.get("train")

            if val_path is None:
                logger.warning(f"No image path found in dataset {dataset_name}")
                return self._get_fallback_images()

            val_path = Path(val_path)

            # Handle relative paths
            if not val_path.is_absolute():
                val_path = DATASETS_DIR / val_path

            return self._get_images_from_path(val_path)

        except Exception as e:
            logger.warning(f"Error loading Ultralytics dataset: {e}")
            return self._get_fallback_images()

    def _get_classification_dataset_images(
        self,
        dataset_name: str,
    ) -> List[Path]:
        """Get images from a classification dataset.

        Classification datasets have a different structure with
        images organized by class folders.

        Args:
            dataset_name: Dataset name

        Returns:
            List of sorted image paths
        """
        try:
            from ultralytics.utils import DATASETS_DIR

            dataset_path = DATASETS_DIR / dataset_name

            if not dataset_path.exists():
                # Try to download
                from ultralytics.data.utils import check_cls_dataset

                check_cls_dataset(dataset_name)

            # Look for val or train directory
            for subdir in ["val", "test", "train"]:
                check_path = dataset_path / subdir
                if check_path.exists():
                    # Recursively find all images in class subdirectories
                    all_images = []
                    for ext in self.IMAGE_EXTENSIONS:
                        all_images.extend(check_path.rglob(f"*{ext}"))
                    return sorted(all_images)

            return self._get_images_from_path(dataset_path)

        except Exception as e:
            logger.warning(f"Error loading classification dataset: {e}")
            return self._get_fallback_images()

    def _get_images_from_path(self, path: Path) -> List[Path]:
        """Get all images from a directory path.

        Args:
            path: Directory containing images

        Returns:
            List of sorted image paths
        """
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return []

        if path.is_file():
            return [path] if path.suffix.lower() in self.IMAGE_EXTENSIONS else []

        # Find all images recursively
        all_images = []
        for ext in self.IMAGE_EXTENSIONS:
            all_images.extend(path.rglob(f"*{ext}"))

        # Sort for deterministic ordering
        return sorted(all_images, key=lambda p: str(p))

    def _get_fallback_images(self) -> List[Path]:
        """Return empty list when no dataset is available.

        The actual calibration will use random data if no images are found.
        """
        logger.warning(
            "No dataset images found. Calibration will use random data. "
            "For best results, ensure Ultralytics is installed: pip install ultralytics"
        )
        return []

    def _select_subset(
        self,
        image_paths: List[Path],
        config: CalibrationConfig,
    ) -> List[Path]:
        """Select a deterministic subset of images.

        Args:
            image_paths: All available image paths (already sorted)
            config: Configuration with num_samples and seed

        Returns:
            Selected subset of image paths
        """
        if len(image_paths) <= config.num_samples:
            # Use all available images
            return image_paths

        # Use seeded RNG for deterministic selection
        rng = np.random.default_rng(config.seed)

        # Generate deterministic indices
        indices = rng.choice(
            len(image_paths),
            size=config.num_samples,
            replace=False,
        )

        # Sort indices to maintain some ordering
        indices = sorted(indices)

        return [image_paths[i] for i in indices]

    def _load_and_preprocess(
        self,
        image_paths: List[Path],
        config: CalibrationConfig,
    ) -> List[np.ndarray]:
        """Load and preprocess images for calibration.

        Args:
            image_paths: Paths to images
            config: Configuration

        Returns:
            List of preprocessed numpy arrays
        """
        images = []
        target_size = (config.input_resolution, config.input_resolution)

        # Try to use OpenCV for loading
        try:
            import cv2

            use_cv2 = True
        except ImportError:
            use_cv2 = False
            logger.warning("OpenCV not available, using PIL fallback")

        for path in image_paths:
            try:
                if use_cv2:
                    img = self._load_with_cv2(path, target_size)
                else:
                    img = self._load_with_pil(path, target_size)

                if img is not None:
                    images.append(img)

            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")

        # If not enough images loaded, pad with random data
        if len(images) < config.num_samples:
            logger.warning(
                f"Only loaded {len(images)} images, "
                f"padding to {config.num_samples} with random data"
            )
            rng = np.random.default_rng(config.seed + 1000)
            while len(images) < config.num_samples:
                random_img = rng.random(
                    (config.input_resolution, config.input_resolution, 3),
                    dtype=np.float32,
                )
                images.append(random_img)

        return images

    def _load_with_cv2(
        self,
        path: Path,
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Load and preprocess image using OpenCV.

        Args:
            path: Image path
            target_size: (width, height) to resize to

        Returns:
            Preprocessed image as float32 array normalized to [0, 1]
        """
        import cv2

        img = cv2.imread(str(path))
        if img is None:
            return None

        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def _load_with_pil(
        self,
        path: Path,
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Load and preprocess image using PIL.

        Args:
            path: Image path
            target_size: (width, height) to resize to

        Returns:
            Preprocessed image as float32 array normalized to [0, 1]
        """
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            return img

        except Exception:
            return None

    def _compute_hash(
        self,
        image_paths: List[Path],
        config: CalibrationConfig,
    ) -> str:
        """Compute a hash for the calibration dataset configuration.

        This hash is used for cache invalidation when the dataset
        or configuration changes.

        Args:
            image_paths: Selected image paths
            config: Configuration

        Returns:
            SHA256 hash string
        """
        hasher = hashlib.sha256()

        # Include configuration
        hasher.update(str(config.num_samples).encode())
        hasher.update(str(config.input_resolution).encode())
        hasher.update(str(config.seed).encode())

        # Include sorted image paths
        for path in sorted(image_paths):
            hasher.update(str(path).encode())

        return hasher.hexdigest()[:16]

    def _get_cache_path(self, task: YOLOTask, config: CalibrationConfig) -> Path:
        """Get the cache file path for a dataset.

        Args:
            task: Task type
            config: Configuration

        Returns:
            Path to cache file
        """
        cache_key = f"{task.value}_{config.num_samples}_{config.input_resolution}_{config.seed}"
        return self.cache_dir / f"{cache_key}.npz"

    def _load_from_cache(
        self,
        task: YOLOTask,
        config: CalibrationConfig,
    ) -> Optional[CalibrationDataset]:
        """Try to load calibration data from cache.

        Args:
            task: Task type
            config: Configuration

        Returns:
            CalibrationDataset if cache hit, None otherwise
        """
        cache_path = self._get_cache_path(task, config)

        if not cache_path.exists():
            return None

        try:
            data = np.load(str(cache_path), allow_pickle=True)

            images = [data[f"image_{i}"] for i in range(data["num_images"])]
            image_paths = [Path(p) for p in data["image_paths"]]
            dataset_hash = str(data["dataset_hash"])

            return CalibrationDataset(
                images=images,
                image_paths=image_paths,
                config=config,
                dataset_hash=dataset_hash,
                task=task,
                metadata=data["metadata"].item() if "metadata" in data else {},
            )

        except Exception as e:
            logger.warning(f"Failed to load calibration cache: {e}")
            return None

    def _save_to_cache(self, dataset: CalibrationDataset, task: YOLOTask) -> None:
        """Save calibration data to cache.

        Args:
            dataset: Dataset to cache
            task: Task type
        """
        cache_path = self._get_cache_path(task, dataset.config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_dict = {
                "num_images": len(dataset.images),
                "image_paths": [str(p) for p in dataset.image_paths],
                "dataset_hash": dataset.dataset_hash,
                "metadata": dataset.metadata,
            }

            # Add images
            for i, img in enumerate(dataset.images):
                save_dict[f"image_{i}"] = img

            np.savez(str(cache_path), **save_dict)
            logger.debug(f"Cached calibration data to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to cache calibration data: {e}")

    def clear_cache(self, task: Optional[YOLOTask] = None) -> None:
        """Clear cached calibration data.

        Args:
            task: Specific task to clear, or None to clear all
        """
        if not self.cache_dir.exists():
            return

        if task is not None:
            # Clear specific task caches
            for cache_file in self.cache_dir.glob(f"{task.value}_*.npz"):
                cache_file.unlink()
        else:
            # Clear all caches
            for cache_file in self.cache_dir.glob("*.npz"):
                cache_file.unlink()


def get_calibration_dataset(
    task: YOLOTask,
    num_samples: int = 100,
    input_resolution: int = 640,
    seed: int = 42,
) -> CalibrationDataset:
    """Convenience function to get a calibration dataset.

    This is the primary entry point for getting calibration data.

    Args:
        task: YOLO task type
        num_samples: Number of images to use
        input_resolution: Input resolution (square)
        seed: Random seed for deterministic selection

    Returns:
        CalibrationDataset ready for model quantization

    Example:
        >>> dataset = get_calibration_dataset(YOLOTask.DETECTION, num_samples=100)
        >>> calibration_data = dataset.images  # List of numpy arrays
        >>> hash = dataset.dataset_hash  # For cache invalidation
    """
    config = CalibrationConfig(
        num_samples=num_samples,
        input_resolution=input_resolution,
        seed=seed,
    )

    loader = CalibrationDatasetLoader()
    return loader.load(task, config)
