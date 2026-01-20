# ONNX Export for YOLO models
#
# Converts PyTorch YOLO models (.pt) to ONNX format (.onnx)
# using Ultralytics export functionality.
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.cache import (
    ModelCache,
    compute_file_hash,
    get_ultralytics_version,
)

logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""

    # Input configuration
    input_resolution: int = 640
    batch_size: int = 1

    # ONNX settings
    opset_version: int = 11  # Compatible with Hailo
    simplify: bool = True  # Run onnx-simplifier
    dynamic: bool = False  # Static shapes for Hailo

    # Export options
    half: bool = False  # FP16 export (not recommended for Hailo quantization)
    int8: bool = False  # INT8 export (Hailo does its own quantization)


class ONNXExporter:
    """Exports YOLO models to ONNX format.

    Uses Ultralytics' built-in export functionality to create
    ONNX models compatible with Hailo's model parser.

    Requirements:
    - Ultralytics package
    - onnx package
    - onnxsim package (optional, for simplification)
    """

    def __init__(self, cache: Optional[ModelCache] = None):
        """Initialize the ONNX exporter.

        Args:
            cache: Optional model cache for storing artifacts
        """
        self.cache = cache or ModelCache()
        self._ultralytics_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if ONNX export is available."""
        if self._ultralytics_available is not None:
            return self._ultralytics_available

        try:
            import ultralytics
            import onnx
            self._ultralytics_available = True
        except ImportError as e:
            logger.warning(f"ONNX export not available: {e}")
            self._ultralytics_available = False

        return self._ultralytics_available

    def export(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: Optional[ONNXExportConfig] = None,
        output_path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """Export a YOLO model to ONNX format.

        Args:
            model_name: YOLO model name (e.g., "yolov8n.pt")
            yolo_version: YOLO version (e.g., "v8")
            task: Task type
            config: Export configuration
            output_path: Custom output path (default: use cache)
            force: Force re-export even if cached

        Returns:
            Path to the exported ONNX file

        Raises:
            RuntimeError: If export fails
            ImportError: If Ultralytics is not available
        """
        if not self.is_available():
            raise ImportError(
                "ONNX export requires ultralytics and onnx packages. "
                "Install with: pip install ultralytics onnx onnxsim"
            )

        config = config or ONNXExportConfig()

        # Determine output path
        if output_path is None:
            output_path = self.cache.get_onnx_path(model_name, yolo_version, task)

        # Check if already exported
        if output_path.exists() and not force:
            logger.info(f"ONNX already exists at {output_path}")
            return output_path

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {model_name} to ONNX...")
        logger.info(f"  Resolution: {config.input_resolution}x{config.input_resolution}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Opset: {config.opset_version}")

        try:
            from ultralytics import YOLO

            # Load the model
            model = YOLO(model_name)

            # Export to ONNX
            # Ultralytics export returns the path to the exported file
            export_result = model.export(
                format="onnx",
                imgsz=config.input_resolution,
                batch=config.batch_size,
                opset=config.opset_version,
                simplify=config.simplify,
                dynamic=config.dynamic,
                half=config.half,
                int8=config.int8,
            )

            # Move the exported file to our cache location
            exported_path = Path(export_result)
            if exported_path != output_path:
                import shutil
                shutil.move(str(exported_path), str(output_path))
                logger.debug(f"Moved ONNX from {exported_path} to {output_path}")

            # Verify the export
            if not output_path.exists():
                raise RuntimeError(f"ONNX export failed: file not created at {output_path}")

            # Validate the ONNX model
            self._validate_onnx(output_path)

            # Update cache metadata
            self._update_metadata(
                model_name, yolo_version, task, config, output_path
            )

            logger.info(f"ONNX exported successfully: {output_path}")
            return output_path

        except Exception as e:
            # Clean up partial export
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"ONNX export failed: {e}") from e

    def _validate_onnx(self, onnx_path: Path) -> None:
        """Validate an ONNX model.

        Args:
            onnx_path: Path to the ONNX file

        Raises:
            RuntimeError: If validation fails
        """
        try:
            import onnx

            logger.debug(f"Validating ONNX model: {onnx_path}")

            # Load and check the model
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)

            # Log model info
            graph = model.graph
            logger.debug(f"  Inputs: {[i.name for i in graph.input]}")
            logger.debug(f"  Outputs: {[o.name for o in graph.output]}")
            logger.debug(f"  Nodes: {len(graph.node)}")

            logger.info("ONNX validation passed")

        except Exception as e:
            raise RuntimeError(f"ONNX validation failed: {e}") from e

    def _update_metadata(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: ONNXExportConfig,
        onnx_path: Path,
    ) -> None:
        """Update cache metadata after export."""
        metadata = self.cache.get_metadata(model_name, yolo_version, task)

        if metadata is None:
            # Create new metadata
            metadata = self.cache.create_metadata(
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                input_resolution=config.input_resolution,
                target_device="hailo8l",  # Will be updated later
            )

        # Update ONNX-specific fields
        metadata.onnx_hash = compute_file_hash(onnx_path)
        metadata.onnx_created_at = datetime.now().isoformat()
        metadata.ultralytics_version = get_ultralytics_version()

        self.cache.save_metadata(metadata, model_name, yolo_version, task)

    def get_model_info(self, model_name: str) -> dict:
        """Get information about a YOLO model.

        Args:
            model_name: Model name

        Returns:
            Dictionary with model information
        """
        if not self.is_available():
            return {"error": "Ultralytics not available"}

        try:
            from ultralytics import YOLO

            model = YOLO(model_name)

            return {
                "name": model_name,
                "task": model.task,
                "names": model.names,  # Class names
                "nc": len(model.names) if model.names else None,  # Number of classes
            }
        except Exception as e:
            return {"error": str(e)}

    def estimate_onnx_compatibility(
        self,
        model_name: str,
        task: YOLOTask,
    ) -> dict:
        """Estimate if a model is compatible with Hailo ONNX parser.

        This does a preliminary check before full export.

        Args:
            model_name: Model name
            task: Task type

        Returns:
            Dictionary with compatibility info
        """
        compatibility = {
            "model": model_name,
            "task": task.value,
            "likely_compatible": True,
            "warnings": [],
            "errors": [],
        }

        # Check supported tasks
        supported_tasks = [YOLOTask.DETECTION, YOLOTask.CLASSIFICATION]
        if task not in supported_tasks:
            compatibility["likely_compatible"] = False
            compatibility["errors"].append(
                f"Task {task.value} is not yet supported for Hailo conversion. "
                f"Supported: {[t.value for t in supported_tasks]}"
            )

        # Check model name patterns
        model_lower = model_name.lower()

        # Check for large models (may have memory issues on NPU)
        if any(size in model_lower for size in ["x.pt", "l.pt"]):
            compatibility["warnings"].append(
                "Large models (x, l) may have memory constraints on Hailo NPU"
            )

        # Check for pose/segment/obb models
        if "-pose" in model_lower:
            compatibility["likely_compatible"] = False
            compatibility["errors"].append("Pose models not yet supported")
        if "-seg" in model_lower:
            compatibility["likely_compatible"] = False
            compatibility["errors"].append("Segmentation models not yet supported")
        if "-obb" in model_lower:
            compatibility["likely_compatible"] = False
            compatibility["errors"].append("OBB models not yet supported")

        return compatibility
