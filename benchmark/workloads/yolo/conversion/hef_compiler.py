# HEF (Hailo Executable Format) Compilation
#
# Compiles HAR files to HEF format using the Hailo Dataflow Compiler.
# The HEF file is the final format that runs on Hailo NPU.
#
# Uses the calibration module (Phase 3) for deterministic calibration
# dataset handling with Ultralytics validation datasets.
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.cache import (
    ModelCache,
    compute_file_hash,
    get_dataflow_compiler_version,
    get_hailort_version,
)
from benchmark.workloads.yolo.conversion.calibration import (
    CalibrationDatasetLoader,
    CalibrationConfig,
    CalibrationDataset,
)

logger = logging.getLogger(__name__)


@dataclass
class HEFCompilerConfig:
    """Configuration for HEF compilation."""

    # Target device
    target_device: str = "hailo8l"  # hailo8 or hailo8l

    # Optimization settings
    optimization_level: int = 2  # 0-3, higher = more optimization
    compression_level: int = 0  # 0-2, higher = more compression

    # Calibration settings
    calibration_data_path: Optional[Path] = None
    calibration_batch_size: int = 8
    calibration_set_size: int = 100  # Number of images for calibration (Phase 3)

    # Calibration dataset settings (Phase 3)
    use_ultralytics_dataset: bool = True  # Use Ultralytics validation dataset
    calibration_seed: int = 42  # Seed for deterministic calibration

    # Memory settings
    allocator_script: Optional[Path] = None

    # Output settings
    performance_data: bool = True  # Include performance data in HEF

    # Advanced options passed directly to compiler
    extra_args: List[str] = field(default_factory=list)


class HEFCompiler:
    """Compiles HAR files to HEF format.

    The HEF compilation process:
    1. Load the HAR file
    2. Run quantization with calibration data
    3. Optimize the network for the target device
    4. Compile to HEF

    This requires the Hailo Dataflow Compiler which is part of the
    Hailo SDK and requires a license from Hailo Developer Zone.

    Note: Compilation is typically performed on an x86_64 development
    machine. The resulting HEF file is then deployed to the target
    Raspberry Pi with Hailo HAT.
    """

    # Default calibration dataset size (Phase 3)
    DEFAULT_CALIBRATION_SIZE = 100

    def __init__(self, cache: Optional[ModelCache] = None):
        """Initialize the HEF compiler.

        Args:
            cache: Optional model cache
        """
        self.cache = cache or ModelCache()
        self._compiler_available: Optional[bool] = None
        self._calibration_loader = CalibrationDatasetLoader()

    def is_available(self) -> bool:
        """Check if Hailo Dataflow Compiler is available."""
        if self._compiler_available is not None:
            return self._compiler_available

        try:
            from hailo_sdk_client import ClientRunner
            # Check if we can create a runner (indicates full SDK available)
            self._compiler_available = True
        except ImportError:
            self._compiler_available = False

        return self._compiler_available

    def compile(
        self,
        har_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: Optional[HEFCompilerConfig] = None,
        output_path: Optional[Path] = None,
        calibration_data: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """Compile HAR to HEF format.

        Args:
            har_path: Path to the HAR file
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            config: Compiler configuration
            output_path: Custom output path
            calibration_data: Path to calibration dataset
            force: Force recompilation

        Returns:
            Path to the compiled HEF file

        Raises:
            RuntimeError: If compilation fails
            ImportError: If Hailo SDK is not available
        """
        if not self.is_available():
            raise ImportError(
                "HEF compilation requires the Hailo Dataflow Compiler. "
                "Install the Hailo SDK from: https://hailo.ai/developer-zone/"
            )

        config = config or HEFCompilerConfig()

        # Determine output path
        if output_path is None:
            output_path = self.cache.get_hef_path(model_name, yolo_version, task)

        # Check if already exists
        if output_path.exists() and not force:
            logger.info(f"HEF already exists at {output_path}")
            return output_path

        # Verify HAR exists
        if not har_path.exists():
            raise FileNotFoundError(f"HAR file not found: {har_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Compiling HEF from {har_path}...")
        logger.info(f"  Target device: {config.target_device}")
        logger.info(f"  Optimization level: {config.optimization_level}")

        try:
            # Prepare calibration data using Phase 3 calibration module
            if config.use_ultralytics_dataset:
                logger.info("Loading calibration dataset from Ultralytics (Phase 3)")
                calib_dataset = self._load_ultralytics_calibration(task, config)
            else:
                calib_path = calibration_data or config.calibration_data_path
                if calib_path is None:
                    logger.info("No calibration data provided, using random calibration")
                    calib_path = self._create_random_calibration(config)
                calib_dataset = None  # Will be loaded from path

            # Run compilation
            return self._compile_with_sdk(
                har_path, model_name, yolo_version, task, config, output_path,
                calib_path if calib_dataset is None else None,
                calib_dataset,
            )

        except Exception as e:
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"HEF compilation failed: {e}") from e

    def _load_ultralytics_calibration(
        self,
        task: YOLOTask,
        config: HEFCompilerConfig,
    ) -> CalibrationDataset:
        """Load calibration dataset using Phase 3 calibration module.

        Args:
            task: YOLO task type
            config: Compiler configuration

        Returns:
            CalibrationDataset with preprocessed images
        """
        calib_config = CalibrationConfig(
            num_samples=config.calibration_set_size,
            input_resolution=640,  # Standard YOLO input
            seed=config.calibration_seed,
        )

        dataset = self._calibration_loader.load(task, calib_config)

        logger.info(f"Loaded {len(dataset)} calibration images")
        logger.info(f"  Dataset hash: {dataset.dataset_hash}")
        logger.info(f"  Seed: {config.calibration_seed}")

        return dataset

    def _compile_with_sdk(
        self,
        har_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: HEFCompilerConfig,
        output_path: Path,
        calibration_path: Optional[Path] = None,
        calibration_dataset: Optional[CalibrationDataset] = None,
    ) -> Path:
        """Compile using Hailo SDK.

        This is the main compilation method using the ClientRunner.

        Args:
            har_path: Path to HAR file
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            config: Compiler configuration
            output_path: Output path for HEF file
            calibration_path: Path to calibration data (if not using dataset)
            calibration_dataset: Pre-loaded CalibrationDataset (Phase 3)

        Returns:
            Path to compiled HEF file
        """
        from hailo_sdk_client import ClientRunner

        logger.info("Loading HAR file...")
        runner = ClientRunner(har=str(har_path), hw_arch=config.target_device)

        # Get network info
        logger.debug(f"Network loaded for {config.target_device}")

        # Prepare calibration dataset
        if calibration_dataset is not None:
            # Use Phase 3 calibration dataset directly
            calib_data = calibration_dataset.images
            logger.info(f"Using Phase 3 calibration dataset: {len(calib_data)} images")
            logger.info(f"  Hash: {calibration_dataset.dataset_hash}")
        else:
            # Load from path
            calib_data = self._load_calibration_data(
                calibration_path,
                config.calibration_set_size,
                runner,
            )

        # Run optimization (quantization + optimization)
        logger.info("Running model optimization...")
        logger.info(f"  Using {len(calib_data)} calibration samples")

        try:
            # Optimize the model (includes quantization)
            runner.optimize(calib_data)
            logger.info("Optimization complete")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

        # Compile to HEF
        logger.info("Compiling to HEF...")

        try:
            hef = runner.compile()

            # Save the HEF
            with open(output_path, "wb") as f:
                f.write(hef)

            logger.info(f"HEF compiled successfully: {output_path}")

            # Update metadata with calibration info
            self._update_metadata(
                model_name, yolo_version, task, config, output_path,
                calibration_path, calibration_dataset
            )

            return output_path

        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            raise

    def _load_calibration_data(
        self,
        calibration_path: Path,
        num_samples: int,
        runner,
    ) -> list:
        """Load calibration data for quantization.

        Args:
            calibration_path: Path to calibration data
            num_samples: Number of samples to use
            runner: Hailo ClientRunner

        Returns:
            List of calibration samples
        """
        import numpy as np

        # Get input shape from the network
        input_info = runner.get_input_layers_info()
        if not input_info:
            # Default YOLO input shape
            input_shape = (640, 640, 3)
        else:
            # Get shape from first input layer
            first_input = list(input_info.values())[0]
            input_shape = first_input.shape[1:]  # Remove batch dimension

        logger.debug(f"Calibration input shape: {input_shape}")

        # Load or generate calibration data
        if calibration_path.is_dir():
            # Load images from directory
            return self._load_images_from_dir(calibration_path, input_shape, num_samples)
        elif calibration_path.suffix == ".npy":
            # Load from numpy file
            data = np.load(str(calibration_path))
            return [data[i] for i in range(min(len(data), num_samples))]
        else:
            # Generate random data
            logger.warning("Using random calibration data (not recommended for production)")
            return self._generate_random_calibration(input_shape, num_samples)

    def _load_images_from_dir(
        self,
        image_dir: Path,
        input_shape: tuple,
        num_samples: int,
    ) -> list:
        """Load images from a directory for calibration.

        Args:
            image_dir: Directory containing images
            input_shape: Expected input shape (H, W, C)
            num_samples: Number of images to load

        Returns:
            List of preprocessed images
        """
        import numpy as np

        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, using random calibration")
            return self._generate_random_calibration(input_shape, num_samples)

        images = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        # Get list of image files
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        # Sort for deterministic order
        image_files.sort()

        # Limit to num_samples
        image_files = image_files[:num_samples]

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize to expected shape
                img = cv2.resize(img, (input_shape[1], input_shape[0]))

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0

                images.append(img)

            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

        if len(images) < num_samples:
            logger.warning(
                f"Only loaded {len(images)} images, "
                f"padding with random data to reach {num_samples}"
            )
            # Pad with random data
            while len(images) < num_samples:
                images.append(
                    np.random.rand(*input_shape).astype(np.float32)
                )

        logger.info(f"Loaded {len(images)} calibration images")
        return images

    def _generate_random_calibration(
        self,
        input_shape: tuple,
        num_samples: int,
    ) -> list:
        """Generate random calibration data.

        Args:
            input_shape: Input shape (H, W, C)
            num_samples: Number of samples

        Returns:
            List of random samples
        """
        import numpy as np

        # Use fixed seed for reproducibility
        np.random.seed(42)

        samples = [
            np.random.rand(*input_shape).astype(np.float32)
            for _ in range(num_samples)
        ]

        return samples

    def _create_random_calibration(self, config: HEFCompilerConfig) -> Path:
        """Create a temporary directory with random calibration data.

        Args:
            config: Compiler configuration

        Returns:
            Path to temporary calibration directory
        """
        import numpy as np

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="hailo_calib_"))

        # Generate random images
        np.random.seed(42)  # Reproducibility

        for i in range(config.calibration_set_size):
            # Generate random image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Save as numpy file
            np.save(temp_dir / f"sample_{i:04d}.npy", img)

        logger.debug(f"Created random calibration data in {temp_dir}")
        return temp_dir

    def _update_metadata(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: HEFCompilerConfig,
        hef_path: Path,
        calibration_path: Optional[Path] = None,
        calibration_dataset: Optional[CalibrationDataset] = None,
    ) -> None:
        """Update cache metadata after compilation.

        Args:
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            config: Compiler configuration
            hef_path: Path to compiled HEF
            calibration_path: Path to calibration data (legacy)
            calibration_dataset: Phase 3 CalibrationDataset (preferred)
        """
        metadata = self.cache.get_metadata(model_name, yolo_version, task)

        if metadata is None:
            metadata = self.cache.create_metadata(
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                input_resolution=640,  # Default
                target_device=config.target_device,
            )

        # Update HEF-specific fields
        metadata.target_device = config.target_device
        metadata.hef_hash = compute_file_hash(hef_path)
        metadata.hef_created_at = datetime.now().isoformat()
        metadata.hailo_compiler_version = get_dataflow_compiler_version()
        metadata.hailort_version = get_hailort_version()
        metadata.calibration_images = config.calibration_set_size

        # Store calibration dataset info (Phase 3)
        if calibration_dataset is not None:
            metadata.calibration_dataset = f"ultralytics:{task.value}"
            metadata.calibration_hash = calibration_dataset.dataset_hash
            metadata.calibration_seed = config.calibration_seed
        elif calibration_path is not None:
            metadata.calibration_dataset = str(calibration_path)

        self.cache.save_metadata(metadata, model_name, yolo_version, task)

    def verify_hef(self, hef_path: Path) -> dict:
        """Verify a compiled HEF file.

        Args:
            hef_path: Path to HEF file

        Returns:
            Dictionary with HEF information
        """
        result = {
            "valid": False,
            "file_size": 0,
            "info": {},
        }

        if not hef_path.exists():
            result["error"] = "HEF file not found"
            return result

        result["file_size"] = hef_path.stat().st_size

        try:
            from hailo_platform import HEF

            hef = HEF(str(hef_path))

            result["valid"] = True
            result["info"] = {
                "network_groups": len(hef.get_network_group_names()),
                "input_vstreams": len(hef.get_input_vstream_infos()),
                "output_vstreams": len(hef.get_output_vstream_infos()),
            }

            # Get input/output details
            inputs = []
            for vstream_info in hef.get_input_vstream_infos():
                inputs.append({
                    "name": vstream_info.name,
                    "shape": vstream_info.shape,
                    "format": str(vstream_info.format.type),
                })
            result["inputs"] = inputs

            outputs = []
            for vstream_info in hef.get_output_vstream_infos():
                outputs.append({
                    "name": vstream_info.name,
                    "shape": vstream_info.shape,
                    "format": str(vstream_info.format.type),
                })
            result["outputs"] = outputs

        except ImportError:
            result["error"] = "HailoRT not available for verification"
        except Exception as e:
            result["error"] = str(e)

        return result

    def get_compiler_info(self) -> dict:
        """Get information about the compiler.

        Returns:
            Dictionary with compiler info
        """
        return {
            "available": self.is_available(),
            "dataflow_compiler_version": get_dataflow_compiler_version(),
            "hailort_version": get_hailort_version(),
        }
