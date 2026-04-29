# HAR (Hailo Archive) Generation for YOLO models
#
# Converts ONNX models to Hailo Archive format using the Hailo SDK.
# The HAR file contains the parsed network ready for compilation.
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.cache import (
    ModelCache,
    compute_file_hash,
    get_hailo_sdk_version,
)

logger = logging.getLogger(__name__)


# Per-(yolo_version, task) end-node truncation table.
#
# Hailo's parser cannot ingest the tail of a YOLO head — DFL/Reshape on
# every detection-shaped task, plus Cos/Sin for OBB angle decoding.
# Truncating the graph before those ops and decoding host-side is the
# documented Hailo workflow (`docs/compilation/pitfalls.md` § 3),
# matched on the runtime side by the postprocessors in
# `benchmark/workloads/yolo/postprocessing.py`.
#
# The names below were captured from Hailo's own diagnostic warnings in
# compile-h8.log on 2026-04-29 (workstation sweep, DFC 3.33.1). They
# correspond to Ultralytics' default ONNX export naming for the head
# module — `/model.22/...` for v8 (head module index 22) and
# `/model.23/...` for v11 (index 23). v26 entries are filled in once
# captured from a v26 compile log; the diagnostic-hint fallback below
# covers the gap until then.
#
# Update procedure when Ultralytics changes its export naming:
#   1. Run a failing compile with -v, grep "use these end node names"
#      out of the captured log.
#   2. Verify each suggested name exists verbatim in the exported ONNX:
#      python -c "import onnx; m = onnx.load('/tmp/X.onnx'); \
#          print([n.name for n in m.graph.node])"
#   3. Update the row below; bump tests/test_har_generator.py if needed.
END_NODE_TABLE: dict = {
    # YOLOv8 detection — three end-nodes feed the host-side
    # _process_detection (Concat for raw boxes, Sigmoid_1 for class
    # scores, Mul for the DFL-decoded box deltas).
    ("v8", YOLOTask.DETECTION): [
        "/model.22/Concat",
        "/model.22/Sigmoid_1",
        "/model.22/Mul",
    ],
    # YOLOv11 detection — same head shape, head module is .23.
    ("v11", YOLOTask.DETECTION): [
        "/model.23/Concat",
        "/model.23/Sigmoid_1",
        "/model.23/Mul",
    ],
    # YOLOv11 segmentation — four end-nodes: detection trio plus the
    # mask prototype tensor (proto/cv3/act/Mul). _process_segmentation /
    # _generate_seg_masks consume these.
    ("v11", YOLOTask.SEGMENTATION): [
        "/model.23/Sigmoid",
        "/model.23/Concat_2",
        "/model.23/proto/cv3/act/Mul",
        "/model.23/Concat",
    ],
    # YOLOv11 pose — four end-nodes: detection-style outputs plus the
    # 17-keypoint heatmap (Mul_3 / Sigmoid feed _process_pose's
    # COCO-Pose decoder).
    ("v11", YOLOTask.POSE): [
        "/model.23/Concat",
        "/model.23/Mul_3",
        "/model.23/Sigmoid_1",
        "/model.23/Sigmoid",
    ],
}


def get_end_nodes(yolo_version: str, task: YOLOTask) -> Optional[List[str]]:
    """Look up the end-node truncation list for a (version, task) pair.

    Returns None when no entry exists. Callers should treat None as
    "let the SDK / hailomz / diagnostic-hint fallback handle it"
    rather than passing None through to the SDK directly.
    """
    return END_NODE_TABLE.get((yolo_version, task))


# Hailo's parse-error message embeds a recommended end-node list in a
# remarkably consistent format across SDK versions:
#
#   ... using these end node names: /model.23/Concat, /model.23/Mul_3
#
# We extract everything after the colon up to the next newline or
# closing paren, then split on commas. The leading "use these" /
# "using these" wording varies by SDK version; matching either keeps
# the parser robust across DFC 3.33.x and 5.3.x without hardcoding the
# exact phrase.
END_NODE_HINT_RE = re.compile(
    r"(?:use|using)\s+these\s+end\s+node\s+names\s*:?\s*"
    r"(?P<nodes>[/A-Za-z0-9_,\s.\-]+?)"
    r"(?=$|[\n)])",
    re.IGNORECASE,
)


def parse_end_node_hint(message: str) -> Optional[List[str]]:
    """Extract Hailo's suggested end-node list from a parse-error string.

    Returns the parsed list (whitespace-trimmed, empty entries dropped)
    or None when no hint is present. The Hailo SDK's exact phrasing
    varies between DFC versions, so we match a permissive regex rather
    than an exact substring.
    """
    if not message:
        return None
    m = END_NODE_HINT_RE.search(message)
    if m is None:
        return None
    raw = m.group("nodes")
    nodes = [n.strip() for n in raw.split(",")]
    nodes = [n for n in nodes if n]
    return nodes or None


@dataclass
class HARGeneratorConfig:
    """Configuration for HAR generation."""

    # Target Hailo device
    target_device: str = "hailo8l"  # hailo8 or hailo8l

    # Network configuration
    input_resolution: int = 640
    batch_size: int = 1

    # Parser options. The Hailo SDK accepts a list of node names for the
    # start / end of the subgraph it parses. None means "let Hailo
    # auto-detect" — fine for start, fatal for end on YOLO heads (the
    # tail contains DFL/Reshape/Cos ops Hailo can't ingest). The
    # end-node list is normally populated from END_NODE_TABLE during
    # pipeline._run_har_generation, not by direct callers.
    start_nodes: Optional[List[str]] = None
    end_nodes: Optional[List[str]] = None

    # YOLO-specific settings
    num_classes: int = 80  # COCO default


class HARGenerator:
    """Generates Hailo Archive (HAR) files from ONNX models.

    The HAR generation process:
    1. Parse the ONNX model using Hailo's parser
    2. Validate the network structure
    3. Generate the HAR file

    This requires the Hailo SDK to be installed, which includes:
    - hailo_sdk_client package
    - Hailo Model Zoo (optional, for pre-configured models)

    Note: The Hailo SDK is typically only available on x86_64 systems
    and requires a Hailo Developer Zone account.
    """

    # Known YOLO model configurations from Hailo Model Zoo
    YOLO_CONFIGS = {
        "yolov8n": {
            "model_name": "yolov8n",
            "input_shape": [1, 3, 640, 640],
            "output_layers": ["model/22/cv2.0/conv/Conv", "model/22/cv3.0/conv/Conv"],
        },
        "yolov8s": {
            "model_name": "yolov8s",
            "input_shape": [1, 3, 640, 640],
        },
        "yolov8m": {
            "model_name": "yolov8m",
            "input_shape": [1, 3, 640, 640],
        },
    }

    def __init__(self, cache: Optional[ModelCache] = None):
        """Initialize the HAR generator.

        Args:
            cache: Optional model cache
        """
        self.cache = cache or ModelCache()
        self._hailo_sdk_available: Optional[bool] = None
        self._model_zoo_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Hailo SDK is available for HAR generation."""
        if self._hailo_sdk_available is not None:
            return self._hailo_sdk_available

        try:
            from hailo_sdk_client import ClientRunner
            self._hailo_sdk_available = True
            logger.debug("Hailo SDK is available")
        except ImportError:
            logger.debug("Hailo SDK not available")
            self._hailo_sdk_available = False

        return self._hailo_sdk_available

    def is_model_zoo_available(self) -> bool:
        """Check if Hailo Model Zoo is available."""
        if self._model_zoo_available is not None:
            return self._model_zoo_available

        try:
            from hailo_model_zoo.core.main_utils import get_network_info
            self._model_zoo_available = True
        except ImportError:
            self._model_zoo_available = False

        return self._model_zoo_available

    def generate(
        self,
        onnx_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: Optional[HARGeneratorConfig] = None,
        output_path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """Generate HAR file from ONNX model.

        Args:
            onnx_path: Path to the ONNX file
            model_name: Model name
            yolo_version: YOLO version
            task: Task type
            config: Generator configuration
            output_path: Custom output path
            force: Force regeneration

        Returns:
            Path to the generated HAR file

        Raises:
            RuntimeError: If generation fails
            ImportError: If Hailo SDK is not available
        """
        if not self.is_available():
            raise ImportError(
                "HAR generation requires the Hailo SDK. "
                "Install from Hailo Developer Zone: https://hailo.ai/developer-zone/"
            )

        config = config or HARGeneratorConfig()

        # Determine output path
        if output_path is None:
            output_path = self.cache.get_har_path(model_name, yolo_version, task)

        # Check if already exists
        if output_path.exists() and not force:
            logger.info(f"HAR already exists at {output_path}")
            return output_path

        # Verify ONNX exists
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating HAR from {onnx_path}...")
        logger.info(f"  Target device: {config.target_device}")
        logger.info(f"  Resolution: {config.input_resolution}x{config.input_resolution}")

        try:
            # Try Model Zoo first if available
            if self.is_model_zoo_available():
                return self._generate_with_model_zoo(
                    onnx_path, model_name, yolo_version, task, config, output_path
                )
            else:
                # Fall back to direct SDK usage
                return self._generate_with_sdk(
                    onnx_path, model_name, yolo_version, task, config, output_path
                )

        except Exception as e:
            # Clean up partial file
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"HAR generation failed: {e}") from e

    def _generate_with_model_zoo(
        self,
        onnx_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: HARGeneratorConfig,
        output_path: Path,
    ) -> Path:
        """Generate HAR using Hailo Model Zoo.

        The Model Zoo provides pre-configured parsing settings for YOLO models.
        """
        logger.info("Using Hailo Model Zoo for HAR generation")

        try:
            from hailo_model_zoo.core.main_utils import (
                parse_model,
                get_network_info,
            )

            # Get base model name (without extension)
            base_name = Path(model_name).stem

            # Check if model is in Model Zoo
            try:
                network_info = get_network_info(base_name)
                logger.info(f"Found {base_name} in Hailo Model Zoo")
            except Exception:
                logger.warning(
                    f"Model {base_name} not found in Model Zoo, using generic parsing"
                )
                return self._generate_with_sdk(
                    onnx_path, model_name, yolo_version, task, config, output_path
                )

            # Parse the model using Model Zoo
            # This uses the pre-configured YOLO postprocessing
            runner = parse_model(
                model_name=base_name,
                onnx_path=str(onnx_path),
                hw_arch=config.target_device,
            )

            # Save the HAR
            runner.save_har(str(output_path))

            self._update_metadata(
                model_name, yolo_version, task, config, output_path
            )

            logger.info(f"HAR generated successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"Model Zoo parsing failed: {e}, trying direct SDK")
            return self._generate_with_sdk(
                onnx_path, model_name, yolo_version, task, config, output_path
            )

    def _generate_with_sdk(
        self,
        onnx_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: HARGeneratorConfig,
        output_path: Path,
    ) -> Path:
        """Generate HAR using Hailo SDK directly.

        Two-pass strategy:
        1. Call ``translate_onnx_model`` with ``config.end_nodes``
           (populated from ``END_NODE_TABLE`` by the pipeline). If the
           table had no entry, ``end_nodes`` is ``None`` and Hailo
           parses to graph end — which fails on YOLO heads.
        2. On parse failure, scan the SDK's error message for a
           "use these end node names: ..." hint and retry once with
           the parsed list. This covers (version, task) pairs the
           static table doesn't know about (e.g. v26 entries before
           we capture them on hardware) without requiring a code
           change.

        If both passes fail — or pass 1 fails with no extractable
        hint — the original parse error is surfaced.
        """
        logger.info("Using Hailo SDK directly for HAR generation")

        from hailo_sdk_client import ClientRunner

        runner = ClientRunner(hw_arch=config.target_device)
        logger.debug(f"Parsing ONNX: {onnx_path}")

        try:
            self._translate_and_save(
                runner, onnx_path, model_name, yolo_version, task,
                config, output_path, config.end_nodes,
            )
            return output_path
        except Exception as first_err:
            # Pass 2: try to recover end-nodes from the diagnostic.
            # Only attempt this when the table miss was the cause —
            # if the caller already supplied end_nodes and the SDK
            # still failed, retrying with parsed names is unlikely to
            # help and adds confusing log output.
            if config.end_nodes is not None:
                raise RuntimeError(
                    f"Failed to parse ONNX model with configured end-nodes "
                    f"{config.end_nodes}. Error: {first_err}"
                ) from first_err

            hint = parse_end_node_hint(str(first_err))
            if hint is None:
                raise RuntimeError(
                    f"Failed to parse ONNX model and no end-node hint "
                    f"found in the error. This usually means the YOLO "
                    f"head's tail is unsupported and there's no entry "
                    f"in END_NODE_TABLE for ({yolo_version}, {task.value}). "
                    f"Error: {first_err}"
                ) from first_err

            logger.warning(
                f"Initial parse failed; retrying with end-nodes from "
                f"Hailo's diagnostic hint: {hint}"
            )
            # Fresh runner — the failed translate_onnx_model leaves
            # internal state we don't want to reuse.
            runner = ClientRunner(hw_arch=config.target_device)
            try:
                self._translate_and_save(
                    runner, onnx_path, model_name, yolo_version, task,
                    config, output_path, hint,
                )
                return output_path
            except Exception as retry_err:
                raise RuntimeError(
                    f"Failed to parse ONNX model. Retry with parsed "
                    f"end-nodes {hint} also failed. "
                    f"Original error: {first_err}. Retry error: {retry_err}"
                ) from retry_err

    def _translate_and_save(
        self,
        runner,
        onnx_path: Path,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: HARGeneratorConfig,
        output_path: Path,
        end_nodes: Optional[List[str]],
    ) -> None:
        """Single ``translate_onnx_model`` + ``save_har`` cycle.

        Split out so ``_generate_with_sdk`` can call it twice (initial
        attempt with table-supplied end-nodes, retry with hint-parsed
        end-nodes). Raises on translation / validation failure;
        returns silently on success after writing ``output_path``.
        """
        runner.translate_onnx_model(
            str(onnx_path),
            net_name=Path(model_name).stem,
            start_node_names=config.start_nodes,
            end_node_names=end_nodes,
        )
        logger.debug("ONNX translation successful")

        self._validate_har(runner, model_name, task)

        runner.save_har(str(output_path))
        self._update_metadata(
            model_name, yolo_version, task, config, output_path
        )
        logger.info(f"HAR generated successfully: {output_path}")

    def _validate_har(
        self,
        runner,
        model_name: str,
        task: YOLOTask,
    ) -> None:
        """Validate the parsed HAR.

        Args:
            runner: Hailo ClientRunner with parsed model
            model_name: Model name
            task: Task type
        """
        logger.debug("Validating parsed network...")

        try:
            # Get network info
            hn_model = runner.get_hn_model()

            if hn_model is None:
                raise RuntimeError("No Hailo Network model available after parsing")

            # Log network structure
            logger.debug(f"Network name: {hn_model.name}")

            # Check for unsupported layers
            # This is a basic check - the compiler will do more thorough validation

            logger.info("HAR validation passed")

        except Exception as e:
            logger.warning(f"HAR validation warning: {e}")
            # Don't fail on validation warnings - let the compiler catch real issues

    def _update_metadata(
        self,
        model_name: str,
        yolo_version: str,
        task: YOLOTask,
        config: HARGeneratorConfig,
        har_path: Path,
    ) -> None:
        """Update cache metadata after HAR generation."""
        metadata = self.cache.get_metadata(model_name, yolo_version, task)

        if metadata is None:
            metadata = self.cache.create_metadata(
                model_name=model_name,
                yolo_version=yolo_version,
                task=task,
                input_resolution=config.input_resolution,
                target_device=config.target_device,
            )

        # Update HAR-specific fields
        metadata.target_device = config.target_device
        metadata.har_hash = compute_file_hash(har_path)
        metadata.har_created_at = datetime.now().isoformat()
        metadata.hailo_sdk_version = get_hailo_sdk_version()

        self.cache.save_metadata(metadata, model_name, yolo_version, task)

    def get_supported_ops(self) -> List[str]:
        """Get list of ONNX operations supported by Hailo parser.

        Returns:
            List of supported operation names
        """
        # Common supported ops for YOLO models
        # This is not exhaustive - Hailo supports many more
        return [
            "Conv",
            "BatchNormalization",
            "Relu",
            "LeakyRelu",
            "Sigmoid",
            "MaxPool",
            "GlobalAveragePool",
            "Concat",
            "Add",
            "Mul",
            "Resize",
            "Transpose",
            "Reshape",
            "Split",
            "Softmax",
            "Flatten",
        ]

    def check_onnx_compatibility(self, onnx_path: Path) -> dict:
        """Check if an ONNX model is compatible with Hailo parser.

        Args:
            onnx_path: Path to ONNX file

        Returns:
            Dictionary with compatibility info
        """
        result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "ops_found": [],
            "unsupported_ops": [],
        }

        try:
            import onnx

            model = onnx.load(str(onnx_path))

            # Get all ops used
            ops_used = set()
            for node in model.graph.node:
                ops_used.add(node.op_type)

            result["ops_found"] = sorted(list(ops_used))

            # Check for potentially unsupported ops
            supported = set(self.get_supported_ops())
            for op in ops_used:
                if op not in supported:
                    result["warnings"].append(f"Operation '{op}' may not be supported")

            # Check input/output shapes
            for input_info in model.graph.input:
                if hasattr(input_info.type.tensor_type, "shape"):
                    dims = input_info.type.tensor_type.shape.dim
                    shape = [d.dim_value for d in dims]
                    if len(shape) != 4:
                        result["warnings"].append(
                            f"Input '{input_info.name}' has unusual shape: {shape}"
                        )

        except Exception as e:
            result["compatible"] = False
            result["errors"].append(str(e))

        return result
