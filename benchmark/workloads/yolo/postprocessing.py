# YOLO Output Post-processing for Hailo NPU
#
# This module provides post-processing functions for YOLO model outputs
# including bounding box decoding and Non-Maximum Suppression (NMS).
#
# Phase 4 - Task 4.1 of Hailo PRD
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from benchmark.schemas import YOLOTask

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result."""

    # Bounding box in xyxy format (x1, y1, x2, y2)
    bbox: Tuple[float, float, float, float]

    # Detection confidence score
    confidence: float

    # Class ID
    class_id: int

    # Optional class name
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        """Convert xyxy to xywh format (center x, center y, width, height)."""
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return (cx, cy, w, h)


@dataclass
class OrientedBox:
    """A single rotated detection result (Phase 3a)."""

    cx: float
    cy: float
    w: float
    h: float
    # Canonical angle in radians, normalised to [-pi/2, pi/2].
    angle_rad: float

    confidence: float
    class_id: int
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h,
            "angle_rad": self.angle_rad,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class PoseResult:
    """A single pose-estimation result (Phase 3c).

    `keypoints` is a (K, 3) numpy array with rows `(x, y, visibility)`.
    For COCO-Pose, K = 17 and the row order is the standard
    nose / eyes / ears / shoulders / elbows / wrists / hips / knees /
    ankles convention (see `YOLOPostProcessor.COCO_POSE_KEYPOINTS`).
    The `visibility` channel is the post-sigmoid score in [0, 1]; values
    above ~0.5 are conventionally treated as "visible". The array is
    serialised to JSON via `to_dict()` (unlike segmentation masks, which
    are too large) since 17 × 3 floats per detection is tractable.
    """

    bbox: Tuple[float, float, float, float]  # xyxy
    confidence: float
    class_id: int
    class_name: Optional[str] = None
    # Keypoint array, shape (K, 3) — (x, y, visibility) per keypoint.
    keypoints: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "keypoints": (
                self.keypoints.tolist() if self.keypoints is not None else None
            ),
        }


@dataclass
class SegmentationResult:
    """A single instance-segmentation result (Phase 3b).

    Carries the same bbox/confidence/class info as `Detection` plus an
    optional binary mask cropped to the bbox at the prototype's native
    resolution. `mask` is intentionally *not* serialised through
    `to_dict()` — at typical YOLO output sizes a per-detection 160x160
    boolean mask would inflate the JSON output by orders of magnitude.
    Use the in-process `mask` array for accuracy validation /
    visualisation; the JSON output only carries a `has_mask` flag and
    a coarse `mask_pixel_count` summary.
    """

    bbox: Tuple[float, float, float, float]  # xyxy
    confidence: float
    class_id: int
    class_name: Optional[str] = None
    # Binary mask, cropped to bbox, at the prototype's native resolution.
    # Shape: (mask_h, mask_w), dtype=bool. None when the postprocessor
    # was run with masks disabled or when the bbox crop is degenerate.
    mask: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "has_mask": self.mask is not None,
            "mask_pixel_count": (
                int(self.mask.sum()) if self.mask is not None else None
            ),
        }


@dataclass
class ClassificationResult:
    """A single classification result."""

    class_id: int
    confidence: float
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "class_name": self.class_name,
        }


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""

    # Detection thresholds
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    # Classification settings
    top_k: int = 5  # Top-K classes to return

    # Input image dimensions (for box scaling)
    input_width: int = 640
    input_height: int = 640

    # Original image dimensions (for box rescaling)
    original_width: Optional[int] = None
    original_height: Optional[int] = None

    # YOLO output format hints
    num_classes: int = 80  # Default COCO classes
    num_anchors: Optional[int] = None

    # Class names (optional)
    class_names: Optional[List[str]] = None


class YOLOPostProcessor:
    """Post-processes YOLO model outputs from Hailo NPU.

    This class handles the conversion of raw model outputs to
    usable detection/classification results. It supports:
    - Detection: bounding boxes + class scores
    - Classification: class probabilities

    The post-processing pipeline:
    1. Decode raw outputs based on task type
    2. Apply confidence threshold
    3. Run NMS (for detection)
    4. Scale boxes to original image size
    """

    # DOTA class names (15 classes used by Ultralytics yolo*-obb checkpoints).
    # Ordering matches the Ultralytics DOTA dataset YAML; do not reorder.
    DOTA_CLASSES = [
        "plane", "ship", "storage tank", "baseball diamond", "tennis court",
        "basketball court", "ground track field", "harbor", "bridge",
        "large vehicle", "small vehicle", "helicopter", "roundabout",
        "soccer ball field", "swimming pool",
    ]

    # COCO-Pose keypoint names (17 keypoints used by yolo*-pose checkpoints).
    # Order matches the COCO-Pose dataset convention; do not reorder.
    COCO_POSE_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    # COCO class names (default)
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    def __init__(self, task: YOLOTask, config: Optional[PostProcessConfig] = None):
        """Initialize the post-processor.

        Args:
            task: YOLO task type
            config: Post-processing configuration
        """
        self.task = task
        self.config = config or PostProcessConfig()

        # Set default class names if not provided. OBB defaults to DOTA's 15
        # classes (the yolo*-obb checkpoints are trained on DOTA); detection
        # / classification keep their existing 80-class COCO default.
        if self.config.class_names is None:
            if task == YOLOTask.OBB:
                # Override num_classes too if the caller didn't customise it.
                if self.config.num_classes == 80:
                    self.config.num_classes = 15
                self.config.class_names = self.DOTA_CLASSES
            elif self.config.num_classes == 80:
                self.config.class_names = self.COCO_CLASSES

    def process(
        self,
        outputs: Dict[str, np.ndarray],
        config: Optional[PostProcessConfig] = None,
    ) -> Union[
        List[Detection],
        List[ClassificationResult],
        List[OrientedBox],
        List[SegmentationResult],
        List[PoseResult],
    ]:
        """Process raw model outputs.

        Args:
            outputs: Dictionary of output tensors from the model
            config: Override configuration for this call

        Returns:
            List of Detection or ClassificationResult objects
        """
        cfg = config or self.config

        if self.task == YOLOTask.DETECTION:
            return self._process_detection(outputs, cfg)
        elif self.task == YOLOTask.CLASSIFICATION:
            return self._process_classification(outputs, cfg)
        elif self.task == YOLOTask.OBB:
            return self._process_obb(outputs, cfg)
        elif self.task == YOLOTask.SEGMENTATION:
            return self._process_segmentation(outputs, cfg)
        elif self.task == YOLOTask.POSE:
            return self._process_pose(outputs, cfg)
        else:
            logger.warning(f"Unsupported task: {self.task}, returning empty results")
            return []

    def _process_detection(
        self,
        outputs: Dict[str, np.ndarray],
        config: PostProcessConfig,
    ) -> List[Detection]:
        """Process detection outputs.

        YOLO detection outputs typically have format:
        - Shape: (batch, num_predictions, 4 + num_classes) or
        - Shape: (batch, num_predictions, 5 + num_classes) with objectness

        Args:
            outputs: Raw model outputs
            config: Post-processing configuration

        Returns:
            List of Detection objects
        """
        # Truncated-head HEFs emit per-stride box+cls branches; reassemble
        # them into the combined-head layout that the decoder expects.
        # Returns None for combined-head HEFs, leaving outputs unchanged.
        assembled = self._assemble_truncated_head(
            outputs, YOLOTask.DETECTION, config
        )
        if assembled is not None:
            outputs = assembled

        # Get the main detection output
        # Hailo outputs may have different names depending on model
        detection_output = self._get_detection_output(outputs)

        if detection_output is None:
            logger.warning("No valid detection output found")
            return []

        # Remove batch dimension if present
        if detection_output.ndim == 3:
            detection_output = detection_output[0]

        # Decode based on output shape
        boxes, scores, class_ids = self._decode_detection_output(
            detection_output, config
        )

        if len(boxes) == 0:
            return []

        # Apply confidence threshold
        mask = scores >= config.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Run NMS
        keep_indices = self._nms(boxes, scores, config.iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]

        # Scale boxes if original dimensions provided
        if config.original_width and config.original_height:
            boxes = self._scale_boxes(
                boxes,
                config.input_width, config.input_height,
                config.original_width, config.original_height
            )

        # Create Detection objects
        detections = []
        for i in range(len(boxes)):
            class_name = None
            if config.class_names and class_ids[i] < len(config.class_names):
                class_name = config.class_names[class_ids[i]]

            detections.append(Detection(
                bbox=tuple(boxes[i].tolist()),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
                class_name=class_name,
            ))

        return detections

    # ---------- OBB processing (Phase 3a) ----------

    def _process_obb(
        self,
        outputs: Dict[str, np.ndarray],
        config: PostProcessConfig,
    ) -> List[OrientedBox]:
        """Process oriented-bounding-box outputs from yolo*-obb checkpoints.

        Expected raw shape (post Hailo HEF): one tensor of shape
        (batch, num_anchors, 5 + num_classes) or its transposed variant
        (batch, 5 + num_classes, num_anchors). The +5 is
        (cx, cy, w, h, angle_rad). YOLOv8/v11/v26-obb all use this layout
        per Ultralytics convention; the angle range varies between
        checkpoints ([-pi/4, 3pi/4] vs [-pi/2, pi/2]) so we normalise to
        [-pi/2, pi/2] before NMS.

        Args:
            outputs: Raw model outputs
            config: Post-processing configuration

        Returns:
            List of OrientedBox objects after rotated NMS
        """
        assembled = self._assemble_truncated_head(
            outputs, YOLOTask.OBB, config
        )
        if assembled is not None:
            outputs = assembled

        obb_output = self._get_detection_output(outputs)

        if obb_output is None:
            logger.warning("No valid OBB output found")
            return []

        # Drop the batch dim so we can reason about (anchors, channels).
        if obb_output.ndim == 3:
            obb_output = obb_output[0]

        # Detect transposition: yolo head exports often emit
        # (channels, anchors); transpose to (anchors, channels) when the
        # number of channels matches our expectation.
        expected_channels = 5 + config.num_classes
        if obb_output.shape[0] == expected_channels and obb_output.shape[1] != expected_channels:
            obb_output = obb_output.T
        elif obb_output.shape[-1] != expected_channels:
            logger.warning(
                f"OBB output has unexpected last-dim {obb_output.shape[-1]}, "
                f"expected {expected_channels}; attempting best-effort decode."
            )

        # Split: 5 geometric channels + num_classes class scores.
        cx = obb_output[:, 0]
        cy = obb_output[:, 1]
        w = obb_output[:, 2]
        h = obb_output[:, 3]
        angle = obb_output[:, 4]
        class_scores = obb_output[:, 5:]

        # Normalise angle into the canonical [-pi/2, pi/2] range. Adding
        # multiples of pi rotates the rectangle onto an equivalent OBB
        # because rectangles have 180-degree rotational symmetry.
        angle = ((angle + np.pi / 2) % np.pi) - np.pi / 2

        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Confidence threshold.
        mask = scores >= config.conf_threshold
        if not np.any(mask):
            return []

        boxes = np.stack([cx[mask], cy[mask], w[mask], h[mask], angle[mask]], axis=1)
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Rotated NMS.
        keep = self._rotated_nms(boxes, scores, config.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Scale boxes if original image dims were provided. Note: we scale
        # cx/cy/w/h but leave angle unchanged — under non-uniform scaling
        # an OBB's true orientation would also shift, but YOLO-OBB models
        # are trained at the input resolution so we expect uniform scale
        # in practice.
        if config.original_width and config.original_height:
            scale_x = config.original_width / config.input_width
            scale_y = config.original_height / config.input_height
            boxes = boxes.copy()
            boxes[:, 0] *= scale_x  # cx
            boxes[:, 1] *= scale_y  # cy
            boxes[:, 2] *= scale_x  # w
            boxes[:, 3] *= scale_y  # h

        oriented = []
        for i in range(len(boxes)):
            class_name = None
            if config.class_names and class_ids[i] < len(config.class_names):
                class_name = config.class_names[class_ids[i]]
            oriented.append(OrientedBox(
                cx=float(boxes[i, 0]),
                cy=float(boxes[i, 1]),
                w=float(boxes[i, 2]),
                h=float(boxes[i, 3]),
                angle_rad=float(boxes[i, 4]),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
                class_name=class_name,
            ))
        return oriented

    # ---------- end OBB processing ----------

    # ---------- Segmentation processing (Phase 3b) ----------

    # Number of mask coefficients used by Ultralytics yolo*-seg checkpoints.
    # Both yolov8-seg and yolo11-seg emit 32 mask prototypes; v26-seg
    # follows the same shape (verified during Slice 5 hardware bring-up).
    SEG_NUM_MASK_COEFFS = 32

    # Sigmoid threshold for binarising prototype-blended masks.
    SEG_MASK_THRESHOLD = 0.5

    def _process_segmentation(
        self,
        outputs: Dict[str, np.ndarray],
        config: PostProcessConfig,
    ) -> List[SegmentationResult]:
        """Process instance-segmentation outputs from yolo*-seg checkpoints.

        Expected raw shape (post Hailo HEF):
          - Detection-style tensor: (batch, num_anchors, 4 + num_classes + 32)
            or its transposed variant (batch, 4 + num_classes + 32, num_anchors).
            The +32 mask coefficients live in the last channel block.
          - Prototype tensor: (batch, 32, mask_h, mask_w). For 640 input the
            native prototype resolution is 160x160 (input/4). Different
            input sizes scale this proportionally.

        The mask for each kept detection is `sigmoid(coeffs @ protos)`,
        binarised at SEG_MASK_THRESHOLD, then cropped to the bbox in
        prototype coordinates. We deliberately keep masks at the
        prototype's native resolution rather than upsampling — accurate
        enough for mAP validation, an order of magnitude smaller than
        a full-resolution mask, and the upsampling can be added by a
        consumer if needed.
        """
        assembled = self._assemble_truncated_head(
            outputs, YOLOTask.SEGMENTATION, config
        )
        if assembled is not None:
            outputs = assembled

        det_output, proto_output = self._get_seg_outputs(outputs, config)

        if det_output is None or proto_output is None:
            logger.warning("Segmentation outputs missing or unrecognised")
            return []

        # Drop batch dim on the detection tensor and ensure (anchors, channels).
        if det_output.ndim == 3:
            det_output = det_output[0]
        expected_channels = 4 + config.num_classes + self.SEG_NUM_MASK_COEFFS
        if (
            det_output.shape[0] == expected_channels
            and det_output.shape[1] != expected_channels
        ):
            det_output = det_output.T
        elif det_output.shape[-1] != expected_channels:
            logger.warning(
                f"Seg det-output last-dim {det_output.shape[-1]}, "
                f"expected {expected_channels}; attempting best-effort decode."
            )

        # Decode geometry, scores, class IDs, and the 32-dim mask coefficients.
        boxes_xywh = det_output[:, :4]
        class_scores = det_output[:, 4 : 4 + config.num_classes]
        mask_coeffs = det_output[:, 4 + config.num_classes :]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Confidence threshold.
        keep_conf = scores >= config.conf_threshold
        if not np.any(keep_conf):
            return []
        boxes_xywh = boxes_xywh[keep_conf]
        scores = scores[keep_conf]
        class_ids = class_ids[keep_conf]
        mask_coeffs = mask_coeffs[keep_conf]

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        # Standard axis-aligned NMS on the bboxes.
        keep_idx = self._nms(boxes_xyxy, scores, config.iou_threshold)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        mask_coeffs = mask_coeffs[keep_idx]

        masks = self._generate_seg_masks(mask_coeffs, proto_output, boxes_xyxy, config)

        # Box scaling (input -> original) AFTER mask generation, since masks
        # are computed in input/proto coordinates.
        if config.original_width and config.original_height:
            boxes_xyxy = self._scale_boxes(
                boxes_xyxy,
                config.input_width,
                config.input_height,
                config.original_width,
                config.original_height,
            )

        results: List[SegmentationResult] = []
        for i in range(len(boxes_xyxy)):
            class_name = None
            if config.class_names and class_ids[i] < len(config.class_names):
                class_name = config.class_names[class_ids[i]]
            results.append(
                SegmentationResult(
                    bbox=tuple(boxes_xyxy[i].tolist()),
                    confidence=float(scores[i]),
                    class_id=int(class_ids[i]),
                    class_name=class_name,
                    mask=masks[i] if masks else None,
                )
            )
        return results

    def _get_seg_outputs(
        self,
        outputs: Dict[str, np.ndarray],
        config: PostProcessConfig,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (detection_tensor, prototype_tensor) for a seg head.

        Recognises:
          - Two named outputs ('output0' / 'output1' or 'detections' /
            'protos') with the canonical Ultralytics layout.
          - Otherwise, classifies tensors by ndim: a 4-D tensor with 32 in
            its second axis is treated as the prototype output.
        """
        det = None
        proto = None

        # Common Ultralytics export naming.
        for det_name in ("output0", "detections", "boxes", "predictions"):
            if det_name in outputs:
                det = outputs[det_name]
                break
        for proto_name in ("output1", "protos", "mask_protos", "mask_coefficients"):
            if proto_name in outputs:
                proto = outputs[proto_name]
                break

        # Fallback: scan by shape.
        if det is None or proto is None:
            for tensor in outputs.values():
                if tensor.ndim == 4 and tensor.shape[1] == self.SEG_NUM_MASK_COEFFS:
                    if proto is None:
                        proto = tensor
                elif tensor.ndim >= 2:
                    last_dim = tensor.shape[-1]
                    expected = 4 + config.num_classes + self.SEG_NUM_MASK_COEFFS
                    if det is None and (
                        last_dim == expected or tensor.shape[1] == expected
                    ):
                        det = tensor

        return det, proto

    def _generate_seg_masks(
        self,
        coeffs: np.ndarray,
        proto: np.ndarray,
        boxes_xyxy: np.ndarray,
        config: PostProcessConfig,
    ) -> List[np.ndarray]:
        """Compute one binary mask per detection.

        Args:
            coeffs: (N, 32) mask coefficient vectors (one per kept detection).
            proto: (1, 32, H, W) or (32, H, W) prototype tensor.
            boxes_xyxy: (N, 4) boxes in input-image coordinates.
            config: Post-processing configuration (used for input dims).

        Returns:
            List of (h_box, w_box) bool arrays — one per detection — at the
            prototype's native resolution, cropped to the bbox region.
            Empty list if shapes are unusable.
        """
        if proto.ndim == 4:
            proto = proto[0]
        if proto.ndim != 3 or proto.shape[0] != self.SEG_NUM_MASK_COEFFS:
            logger.warning(
                f"Prototype tensor shape {proto.shape} not "
                f"({self.SEG_NUM_MASK_COEFFS}, H, W); skipping mask generation."
            )
            return []

        c, h, w = proto.shape
        proto_flat = proto.reshape(c, h * w)
        # sigmoid(coeffs @ proto_flat) — clip the input to keep numerics stable.
        logits = coeffs @ proto_flat
        masks = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        masks = masks.reshape(-1, h, w)
        binary = masks >= self.SEG_MASK_THRESHOLD

        # Map input-image bbox coords to prototype grid coords. Input is
        # config.input_width × config.input_height; proto is w × h.
        scale_x = w / max(config.input_width, 1)
        scale_y = h / max(config.input_height, 1)

        cropped: List[np.ndarray] = []
        for i in range(binary.shape[0]):
            x1 = int(np.floor(boxes_xyxy[i, 0] * scale_x))
            y1 = int(np.floor(boxes_xyxy[i, 1] * scale_y))
            x2 = int(np.ceil(boxes_xyxy[i, 2] * scale_x))
            y2 = int(np.ceil(boxes_xyxy[i, 3] * scale_y))
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                cropped.append(np.zeros((0, 0), dtype=bool))
            else:
                cropped.append(binary[i, y1:y2, x1:x2])
        return cropped

    # ---------- end Segmentation processing ----------

    # ---------- Pose processing (Phase 3c) ----------

    # COCO-Pose checkpoints all use 17 keypoints. v8/v11/v26 follow this
    # convention; if a future checkpoint diverges, override via
    # PostProcessConfig.num_classes (which we re-purpose as keypoint count
    # at processor init when task=POSE — see __init__).
    POSE_NUM_KEYPOINTS = 17

    # Channels per keypoint: (x, y, visibility).
    POSE_KEYPOINT_CHANNELS = 3

    # Sigmoid threshold for treating a keypoint as "visible". Used only for
    # downstream filtering by callers; the postprocessor stores the raw
    # post-sigmoid score so consumers can choose their own threshold.
    POSE_KPT_VISIBILITY_THRESHOLD = 0.5

    def _process_pose(
        self,
        outputs: Dict[str, np.ndarray],
        config: PostProcessConfig,
    ) -> List[PoseResult]:
        """Process pose-estimation outputs from yolo*-pose checkpoints.

        Expected raw shape (post Hailo HEF):
          - Single tensor of shape (batch, num_anchors, 4 + num_classes
            + num_keypoints * 3) or its transposed variant
            (batch, 4 + num_classes + num_keypoints * 3, num_anchors).
            For COCO-Pose: 4 + 1 + 17*3 = 56 channels per anchor.

        Decoding:
          - Bbox + class scores -> standard NMS (axis-aligned).
          - Keypoint logits split into (x, y, vis) triples; vis is
            sigmoid-applied so consumers see a [0, 1] visibility score.
          - Keypoint coordinates are scaled when the caller provides
            original_width / original_height (input -> original).
        """
        assembled = self._assemble_truncated_head(
            outputs, YOLOTask.POSE, config
        )
        if assembled is not None:
            outputs = assembled

        pose_output = self._get_detection_output(outputs)

        if pose_output is None:
            logger.warning("No valid pose output found")
            return []

        if pose_output.ndim == 3:
            pose_output = pose_output[0]

        num_kpts = self.POSE_NUM_KEYPOINTS
        kpt_dims = self.POSE_KEYPOINT_CHANNELS
        expected_channels = 4 + config.num_classes + num_kpts * kpt_dims

        # Auto-transpose (channels, anchors) -> (anchors, channels).
        if (
            pose_output.shape[0] == expected_channels
            and pose_output.shape[1] != expected_channels
        ):
            pose_output = pose_output.T
        elif pose_output.shape[-1] != expected_channels:
            # Truncated-head HEFs emit per-stride branches (box / cls / kpts
            # as separate outputs) instead of the combined (4 + nc + 51)
            # tensor this decoder expects. _get_detection_output picks one
            # branch and the kpt reshape later would crash on indivisible
            # sizes. Match the OBB/seg behaviour: return [] gracefully.
            logger.warning(
                f"Pose output last-dim {pose_output.shape[-1]}, expected "
                f"{expected_channels}; truncated-head layout is not yet "
                f"decoded — returning no detections."
            )
            return []

        boxes_xywh = pose_output[:, :4]
        class_scores = pose_output[:, 4 : 4 + config.num_classes]
        kpt_block = pose_output[:, 4 + config.num_classes :]

        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Confidence threshold first.
        keep_conf = scores >= config.conf_threshold
        if not np.any(keep_conf):
            return []
        boxes_xywh = boxes_xywh[keep_conf]
        scores = scores[keep_conf]
        class_ids = class_ids[keep_conf]
        kpt_block = kpt_block[keep_conf]

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        # Standard axis-aligned NMS on bboxes.
        keep_idx = self._nms(boxes_xyxy, scores, config.iou_threshold)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        kpt_block = kpt_block[keep_idx]

        # Reshape keypoint block to (N, num_kpts, 3) and apply sigmoid to
        # the visibility channel only (x, y stay raw, in input pixels).
        keypoints = kpt_block.reshape(-1, num_kpts, kpt_dims).astype(np.float64)
        # Clip the visibility logit before sigmoid for numerical stability.
        vis_logits = np.clip(keypoints[:, :, 2], -50.0, 50.0)
        keypoints[:, :, 2] = 1.0 / (1.0 + np.exp(-vis_logits))

        # Box scaling — also rescale keypoint x/y so they stay aligned.
        if config.original_width and config.original_height:
            scale_x = config.original_width / max(config.input_width, 1)
            scale_y = config.original_height / max(config.input_height, 1)
            boxes_xyxy = self._scale_boxes(
                boxes_xyxy,
                config.input_width,
                config.input_height,
                config.original_width,
                config.original_height,
            )
            keypoints[:, :, 0] *= scale_x
            keypoints[:, :, 1] *= scale_y

        results: List[PoseResult] = []
        for i in range(len(boxes_xyxy)):
            class_name = None
            if config.class_names and class_ids[i] < len(config.class_names):
                class_name = config.class_names[class_ids[i]]
            results.append(
                PoseResult(
                    bbox=tuple(boxes_xyxy[i].tolist()),
                    confidence=float(scores[i]),
                    class_id=int(class_ids[i]),
                    class_name=class_name,
                    keypoints=keypoints[i],
                )
            )
        return results

    # ---------- end Pose processing ----------

    # ---------- Truncated-head assembly --------------------------------------
    #
    # The HEFs in resources/hefs/ are compiled with end-node truncation per
    # END_NODE_TABLE in conversion/har_generator.py — the heads are cut at
    # raw cv*.X.X.2/Conv outputs (one per scale, one per branch). On the
    # chip this means the runtime returns multiple per-stride tensors
    # (3 strides × {box, cls, optional kpts/angle/mask_coeffs}) instead of
    # the single combined-head tensor that combined-head HEFs / .pt models
    # produce. The helpers below reassemble that layout into the combined
    # form so the existing _process_* decoders work unchanged.
    #
    # Per-stride channel layouts (NHWC, batch dim already stripped):
    #
    #   Detection:    box (H, W, 64)        + cls (H, W, nc)
    #   OBB:          box (H, W, 64)        + cls (H, W, nc) + angle (H, W, 1)
    #   Pose:         box (H, W, 64)        + cls (H, W, nc) + kpts  (H, W, 51)
    #   Segmentation: box (H, W, 64)        + cls (H, W, nc) + coeffs(H, W, 32)
    #                 + one shared prototype tensor (H_p, W_p, 32) at input/4
    #
    # Box decoding uses DFL (Distribution Focal Loss): each of the 4 box
    # edges is a 16-bin softmax distribution over distance-from-anchor in
    # feature-map units; expectation over the bins gives a continuous
    # distance, which is then converted to xywh in input pixels.
    #
    # Class scores are raw logits (sigmoid here); pose keypoint xy uses
    # the Ultralytics decode `(2*v + (anchor - 0.5)) * stride`; OBB angle
    # uses `(sigmoid(v) - 0.25) * pi`.

    DFL_BINS = 16  # YOLOv8/v11/v26 all use a 16-bin DFL.
    SEG_PROTO_CHANNELS = 32  # alias of SEG_NUM_MASK_COEFFS for assembler use.
    POSE_KPTS_CHANNELS = 51  # 17 keypoints × (x, y, vis).

    def _assemble_truncated_head(
        self,
        outputs: Dict[str, np.ndarray],
        task: YOLOTask,
        config: PostProcessConfig,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Assemble per-stride truncated-head outputs into the combined layout.

        Returns a new outputs dict whose tensors match what the existing
        combined-head decoders expect:

          - Detection / Pose:   {"combined": (1, anchors, 4 + nc + extras)}
          - OBB:                {"combined": (1, anchors, 5 + nc)}
          - Segmentation:       {"combined": (1, anchors, 4 + nc + 32),
                                 "prototype": (1, 32, H_p, W_p)}

        Returns None when the layout doesn't look like a truncated head
        (combined-head HEFs hit this path), so the caller can fall through
        to its existing decode logic without any change of behaviour.
        """
        if task not in (
            YOLOTask.DETECTION,
            YOLOTask.OBB,
            YOLOTask.POSE,
            YOLOTask.SEGMENTATION,
        ):
            return None

        nc = config.num_classes
        if nc <= 0:
            return None

        # Group every (H, W, C) output by feature-map shape; isolate the
        # seg prototype (4-D NHWC tensor whose feature map is input/4).
        by_hw: Dict[Tuple[int, int], List[np.ndarray]] = {}
        proto: Optional[np.ndarray] = None
        for tensor in outputs.values():
            t = tensor
            # Hailo gives NHWC with batch dim 1.
            if t.ndim == 4 and t.shape[0] == 1:
                t = t[0]
            if t.ndim != 3:
                continue
            h, w, c = t.shape
            if (
                task == YOLOTask.SEGMENTATION
                and c == self.SEG_PROTO_CHANNELS
                and h == config.input_height // 4
                and w == config.input_width // 4
            ):
                proto = t  # (H_p, W_p, 32)
                continue
            by_hw.setdefault((h, w), []).append(t)

        if not by_hw:
            return None

        # Sort strides finest -> coarsest (largest H first).
        strides_sorted = sorted(by_hw.items(), key=lambda kv: -kv[0][0])
        if len(strides_sorted) < 2:
            # A single stride group is more likely a combined-head tensor
            # the caller should decode directly; bail out.
            return None

        per_stride_combined: List[np.ndarray] = []
        for (h, w), branches in strides_sorted:
            stride = max(config.input_height // h, 1)
            box_t = cls_t = kpts_t = angle_t = coeffs_t = None
            for t in branches:
                c = t.shape[-1]
                # Box branch: 64ch = DFL (v8/v11), 4ch = direct distance (v26).
                if c in (4 * self.DFL_BINS, 4) and box_t is None:
                    box_t = t
                elif c == nc and cls_t is None:
                    cls_t = t
                elif (
                    task == YOLOTask.POSE
                    and c == self.POSE_KPTS_CHANNELS
                    and kpts_t is None
                ):
                    kpts_t = t
                elif task == YOLOTask.OBB and c == 1 and angle_t is None:
                    angle_t = t
                elif (
                    task == YOLOTask.SEGMENTATION
                    and c == self.SEG_PROTO_CHANNELS
                    and coeffs_t is None
                ):
                    coeffs_t = t

            if box_t is None or cls_t is None:
                # Missing the required branches at this stride — abort
                # rather than emitting a partial / mismatched decode.
                return None

            n_anchors = h * w
            box_xywh = self._decode_box_branch(box_t, h, w, stride)
            cls_logits = cls_t.reshape(n_anchors, nc).astype(np.float32)
            cls_scores = self._sigmoid(cls_logits)

            if task == YOLOTask.OBB:
                if angle_t is None:
                    return None
                angle_logits = angle_t.reshape(n_anchors, 1).astype(np.float32)
                # Ultralytics OBB angle: (sigmoid(v) - 0.25) * pi  → [-pi/4, 3pi/4]
                angle = (self._sigmoid(angle_logits) - 0.25) * np.pi
                # Combined OBB layout: (cx, cy, w, h, angle, *cls).
                stride_combined = np.concatenate(
                    [box_xywh, angle, cls_scores], axis=1
                )
            elif task == YOLOTask.POSE and kpts_t is not None:
                kpts_decoded = self._decode_kpts_branch(
                    kpts_t.reshape(n_anchors, 17, 3), h, w, stride
                )
                stride_combined = np.concatenate(
                    [box_xywh, cls_scores, kpts_decoded.reshape(n_anchors, 51)],
                    axis=1,
                )
            elif task == YOLOTask.SEGMENTATION and coeffs_t is not None:
                coeffs = coeffs_t.reshape(n_anchors, self.SEG_PROTO_CHANNELS)
                stride_combined = np.concatenate(
                    [box_xywh, cls_scores, coeffs], axis=1
                )
            else:
                stride_combined = np.concatenate([box_xywh, cls_scores], axis=1)

            per_stride_combined.append(stride_combined)

        combined = np.concatenate(per_stride_combined, axis=0)[None, ...]
        result: Dict[str, np.ndarray] = {"combined": combined}
        if task == YOLOTask.SEGMENTATION:
            if proto is None:
                return None
            # _get_seg_outputs scans for a 4-D tensor with shape[1] == 32,
            # i.e. NCHW (1, 32, H_p, W_p). Convert NHWC -> NCHW + batch.
            result["prototype"] = proto.transpose(2, 0, 1)[None, ...]
        return result

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _decode_box_branch(
        self,
        box_t: np.ndarray,
        h: int,
        w: int,
        stride: int,
    ) -> np.ndarray:
        """Decode a per-stride box branch to xywh in input pixels.

        Two layouts in the wild:
          - 64-channel DFL (v8 / v11): 4×16 softmax distributions over
            distance bins; expectation gives the continuous distance.
          - 4-channel direct (v26): the raw conv output already represents
            the (l, t, r, b) distances in feature-map units, no DFL stage.

        Both produce the same (n_anchors, 4) ltrb-distances tensor that
        feeds the shared anchor → xywh conversion.
        """
        n_anchors = h * w
        c = box_t.shape[-1]
        if c == 4 * self.DFL_BINS:
            bins = box_t.reshape(n_anchors, 4, self.DFL_BINS).astype(np.float32)
            # Numerically stable softmax over the 16 bins.
            bins = bins - bins.max(axis=-1, keepdims=True)
            exp = np.exp(bins)
            soft = exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)
            bin_idx = np.arange(self.DFL_BINS, dtype=np.float32)
            dist = (soft * bin_idx).sum(axis=-1)
        elif c == 4:
            dist = box_t.reshape(n_anchors, 4).astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported box branch channel count: {c} "
                f"(expected 4 or {4 * self.DFL_BINS})"
            )
        return self._ltrb_to_xywh(dist, h, w, stride)

    @staticmethod
    def _ltrb_to_xywh(
        dist: np.ndarray,
        h: int,
        w: int,
        stride: int,
    ) -> np.ndarray:
        """Convert (n_anchors, 4) ltrb distances + grid stride to xywh in input pixels."""
        n_anchors = h * w
        gy, gx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        anchor_x = (gx + 0.5).reshape(n_anchors)
        anchor_y = (gy + 0.5).reshape(n_anchors)

        l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
        x1 = (anchor_x - l) * stride
        y1 = (anchor_y - t) * stride
        x2 = (anchor_x + r) * stride
        y2 = (anchor_y + b) * stride
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        ww = x2 - x1
        hh = y2 - y1
        return np.stack([cx, cy, ww, hh], axis=1)

    def _decode_kpts_branch(
        self,
        kpts: np.ndarray,
        h: int,
        w: int,
        stride: int,
    ) -> np.ndarray:
        """Decode (n_anchors, 17, 3) raw kpt logits to input-pixel coords.

        Ultralytics YOLO-pose kpt decode for x/y; visibility is left as a
        raw logit because the downstream `_process_pose` decoder applies
        the sigmoid itself — keeping the layout compatible with the
        combined-head HEFs that this assembler is impersonating.
            x = (raw_x * 2 + (anchor_x - 0.5)) * stride
            y = (raw_y * 2 + (anchor_y - 0.5)) * stride
            v = raw_v   (sigmoid applied later by _process_pose)
        """
        n_anchors = h * w
        gy, gx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        anchor_x = gx.reshape(n_anchors).astype(np.float32)
        anchor_y = gy.reshape(n_anchors).astype(np.float32)

        out = np.empty_like(kpts, dtype=np.float32)
        out[:, :, 0] = (
            kpts[:, :, 0].astype(np.float32) * 2.0 + (anchor_x[:, None] - 0.5)
        ) * stride
        out[:, :, 1] = (
            kpts[:, :, 1].astype(np.float32) * 2.0 + (anchor_y[:, None] - 0.5)
        ) * stride
        out[:, :, 2] = kpts[:, :, 2].astype(np.float32)
        return out

    # ---------- end truncated-head assembly ----------------------------------

    def _get_detection_output(
        self,
        outputs: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Extract the main detection output tensor.

        Args:
            outputs: Dictionary of output tensors

        Returns:
            Detection output tensor or None
        """
        if len(outputs) == 1:
            return list(outputs.values())[0]

        # Try common output names
        for name in ["output", "output0", "detections", "boxes", "predictions"]:
            if name in outputs:
                return outputs[name]

        # Try to find by shape (looking for detection-like shape)
        for name, tensor in outputs.items():
            if tensor.ndim >= 2:
                last_dim = tensor.shape[-1]
                # Detection outputs typically have 4 (bbox) + classes
                if last_dim > 4:
                    return tensor

        # Return first tensor as fallback
        return list(outputs.values())[0] if outputs else None

    def _decode_detection_output(
        self,
        output: np.ndarray,
        config: PostProcessConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw detection output into boxes, scores, and class IDs.

        Args:
            output: Raw output tensor (num_predictions, features)
            config: Configuration

        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        num_features = output.shape[-1]

        # Determine output format
        # Format 1: [x, y, w, h, class_scores...] (YOLOv8 style)
        # Format 2: [x, y, w, h, objectness, class_scores...] (YOLOv5 style)
        # Format 3: [x1, y1, x2, y2, class_scores...] (xyxy format)

        if num_features == 4 + config.num_classes:
            # Format 1: No objectness score
            boxes_xywh = output[:, :4]
            class_scores = output[:, 4:]
            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
            boxes = self._xywh_to_xyxy(boxes_xywh)

        elif num_features == 5 + config.num_classes:
            # Format 2: With objectness score
            boxes_xywh = output[:, :4]
            objectness = output[:, 4]
            class_scores = output[:, 5:]
            scores = objectness * np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
            boxes = self._xywh_to_xyxy(boxes_xywh)

        else:
            # Try to infer format
            # Assume first 4 values are boxes
            boxes_raw = output[:, :4]
            class_scores = output[:, 4:]

            # Check if boxes look like xywh or xyxy
            if np.all(boxes_raw[:, 2:4] <= 1.0):
                # Likely normalized xywh
                boxes = self._xywh_to_xyxy(boxes_raw * config.input_width)
            else:
                # Assume xyxy
                boxes = boxes_raw

            scores = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)

        return boxes, scores, class_ids

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from xywh to xyxy format.

        Args:
            boxes: Boxes in (cx, cy, w, h) format

        Returns:
            Boxes in (x1, y1, x2, y2) format
        """
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        from_width: int,
        from_height: int,
        to_width: int,
        to_height: int,
    ) -> np.ndarray:
        """Scale boxes from one image size to another.

        Args:
            boxes: Boxes in xyxy format
            from_width: Source width
            from_height: Source height
            to_width: Target width
            to_height: Target height

        Returns:
            Scaled boxes
        """
        scale_x = to_width / from_width
        scale_y = to_height / from_height

        scaled = boxes.copy()
        scaled[:, [0, 2]] *= scale_x
        scaled[:, [1, 3]] *= scale_y

        return scaled

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Non-Maximum Suppression.

        Args:
            boxes: Boxes in xyxy format (N, 4)
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        # Sort by score (descending)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # Pick the box with highest score
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            remaining = order[1:]
            ious = self._compute_iou(boxes[i], boxes[remaining])

            # Keep boxes with IoU below threshold
            mask = ious <= iou_threshold
            order = remaining[mask]

        return np.array(keep, dtype=np.int64)

    def _compute_iou(
        self,
        box: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """Compute IoU between one box and multiple boxes.

        Args:
            box: Single box in xyxy format (4,)
            boxes: Multiple boxes in xyxy format (N, 4)

        Returns:
            IoU values (N,)
        """
        # Intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection

        return intersection / (union + 1e-6)

    # ---------- Rotated NMS primitives (Phase 3a, OBB on Hailo) ----------
    # Pure-numpy implementation of rotated-rectangle IoU via Sutherland-Hodgman
    # polygon clipping. Avoids hard-pinning OpenCV (`opencv-python-headless`
    # is installed by the setup scripts but is NOT in setup.py's
    # install_requires, so we keep this self-contained).

    @staticmethod
    def _obb_corners(box: np.ndarray) -> np.ndarray:
        """Compute the four corners of an OBB.

        Args:
            box: Single OBB (5,) = (cx, cy, w, h, angle_rad)

        Returns:
            (4, 2) corner array. Local order (-,-),(+,-),(+,+),(-,+) is
            CCW in math y-up coordinates; image y-down callers should not
            rely on the winding directly — `_rotated_iou` normalises via
            `_ensure_ccw`.
        """
        cx, cy, w, h, angle = (
            float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
        )
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        hw, hh = w / 2.0, h / 2.0
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return local @ rot.T + np.array([cx, cy])

    @staticmethod
    def _polygon_signed_area(poly: np.ndarray) -> float:
        """Shoelace signed area (positive for CCW in math coords)."""
        if poly.shape[0] < 3:
            return 0.0
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    @classmethod
    def _ensure_ccw(cls, poly: np.ndarray) -> np.ndarray:
        """Return polygon in CCW order (reverses if signed area < 0)."""
        if poly.shape[0] < 3:
            return poly
        return poly if cls._polygon_signed_area(poly) >= 0 else poly[::-1]

    @staticmethod
    def _segment_intersect(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
    ) -> np.ndarray:
        """Intersection of the lines through p1-p2 and p3-p4.

        Used inside Sutherland-Hodgman where the inside-test guarantees
        the segments actually cross. Returns p1 unchanged if the lines
        are (nearly) parallel — the caller's outer loop will discard it.
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        x3, y3 = float(p3[0]), float(p3[1])
        x4, y4 = float(p4[0]), float(p4[1])
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1.astype(np.float64)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    @classmethod
    def _polygon_intersect(cls, subj: np.ndarray, clip: np.ndarray) -> np.ndarray:
        """Sutherland-Hodgman polygon intersection.

        Both `subj` and `clip` must be CCW (caller's responsibility — use
        `_ensure_ccw`). Returns the intersection polygon as an (M, 2) array,
        or an empty (0, 2) array if there is no overlap.
        """
        output = subj
        n_clip = clip.shape[0]
        for i in range(n_clip):
            if output.shape[0] == 0:
                return output
            e_start = clip[i]
            e_end = clip[(i + 1) % n_clip]
            edge = e_end - e_start
            new_output = []
            n_out = output.shape[0]
            for j in range(n_out):
                current = output[j]
                previous = output[(j - 1) % n_out]
                # Cross product test: point is "inside" (left of edge for CCW)
                # when cross(edge, point - e_start) >= 0.
                curr_in = (
                    edge[0] * (current[1] - e_start[1])
                    - edge[1] * (current[0] - e_start[0])
                ) >= 0
                prev_in = (
                    edge[0] * (previous[1] - e_start[1])
                    - edge[1] * (previous[0] - e_start[0])
                ) >= 0
                if curr_in:
                    if not prev_in:
                        new_output.append(
                            cls._segment_intersect(previous, current, e_start, e_end)
                        )
                    new_output.append(current)
                elif prev_in:
                    new_output.append(
                        cls._segment_intersect(previous, current, e_start, e_end)
                    )
            output = np.array(new_output) if new_output else np.empty((0, 2))
        return output

    def _rotated_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """IoU between one rotated box and N rotated boxes.

        Mirrors `_compute_iou` but for OBBs.

        Args:
            box: Single OBB (5,) = (cx, cy, w, h, angle_rad)
            boxes: Multiple OBBs (N, 5) in the same format

        Returns:
            IoU values (N,)
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.float64)

        poly_a = self._ensure_ccw(self._obb_corners(box))
        area_a = float(box[2]) * float(box[3])
        n = boxes.shape[0]
        ious = np.zeros(n, dtype=np.float64)
        for i in range(n):
            poly_b = self._ensure_ccw(self._obb_corners(boxes[i]))
            inter_poly = self._polygon_intersect(poly_a, poly_b)
            inter_area = abs(self._polygon_signed_area(inter_poly))
            area_b = float(boxes[i, 2]) * float(boxes[i, 3])
            union = area_a + area_b - inter_area
            ious[i] = inter_area / (union + 1e-6)
        return ious

    def _rotated_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Non-Maximum Suppression on rotated boxes.

        Mirrors `_nms` but uses rotated-rectangle IoU.

        Args:
            boxes: OBBs (N, 5) in (cx, cy, w, h, angle_rad)
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep (descending score order, NMS applied)
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            remaining = order[1:]
            ious = self._rotated_iou(boxes[i], boxes[remaining])
            mask = ious <= iou_threshold
            order = remaining[mask]
        return np.array(keep, dtype=np.int64)

    # ---------- end rotated NMS primitives ----------

    def _process_classification(
        self,
        outputs: Dict[str, np.ndarray],
        config: PostProcessConfig,
    ) -> List[ClassificationResult]:
        """Process classification outputs.

        Args:
            outputs: Raw model outputs
            config: Configuration

        Returns:
            List of ClassificationResult objects (top-K)
        """
        # Get classification output
        output = self._get_classification_output(outputs)

        if output is None:
            logger.warning("No valid classification output found")
            return []

        # Remove batch dimension
        if output.ndim > 1:
            output = output.flatten()

        # Apply softmax if not already probabilities
        if not np.allclose(np.sum(output), 1.0, atol=0.1):
            output = self._softmax(output)

        # Get top-K predictions
        top_k = min(config.top_k, len(output))
        top_indices = np.argsort(output)[::-1][:top_k]

        results = []
        for idx in top_indices:
            class_name = None
            if config.class_names and idx < len(config.class_names):
                class_name = config.class_names[idx]

            results.append(ClassificationResult(
                class_id=int(idx),
                confidence=float(output[idx]),
                class_name=class_name,
            ))

        return results

    def _get_classification_output(
        self,
        outputs: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Extract the classification output tensor.

        Args:
            outputs: Dictionary of output tensors

        Returns:
            Classification output tensor or None
        """
        if len(outputs) == 1:
            return list(outputs.values())[0]

        # Try common output names
        for name in ["output", "output0", "logits", "predictions", "probs"]:
            if name in outputs:
                return outputs[name]

        return list(outputs.values())[0] if outputs else None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values.

        Args:
            x: Input array

        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-6)


def decode_yolo_output(
    outputs: Dict[str, np.ndarray],
    task: YOLOTask,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_size: int = 640,
    num_classes: int = 80,
) -> Union[
    List[Detection],
    List[ClassificationResult],
    List[OrientedBox],
    List[SegmentationResult],
    List[PoseResult],
]:
    """Convenience function to decode YOLO outputs.

    Args:
        outputs: Raw model outputs
        task: YOLO task type
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        input_size: Model input size
        num_classes: Number of classes

    Returns:
        List of Detection or ClassificationResult objects
    """
    config = PostProcessConfig(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        input_width=input_size,
        input_height=input_size,
        num_classes=num_classes,
    )

    processor = YOLOPostProcessor(task, config)
    return processor.process(outputs)


def compute_detection_metrics(
    predictions: List[Detection],
    ground_truth: List[Detection],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute detection accuracy metrics.

    Args:
        predictions: Predicted detections
        ground_truth: Ground truth detections
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with precision, recall, and mAP
    """
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "mAP": 0.0}

    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "mAP": 0.0}

    # Match predictions to ground truth
    pred_boxes = np.array([p.bbox for p in predictions])
    gt_boxes = np.array([g.bbox for g in ground_truth])

    # Compute IoU matrix
    iou_matrix = np.zeros((len(predictions), len(ground_truth)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = _compute_single_iou(pred_box, gt_box)

    # Match predictions (greedy matching)
    matched_gt = set()
    true_positives = 0

    for i in range(len(predictions)):
        best_iou = 0
        best_j = -1
        for j in range(len(ground_truth)):
            if j in matched_gt:
                continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            true_positives += 1
            matched_gt.add(best_j)

    precision = true_positives / len(predictions) if predictions else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0

    # Simplified mAP (single IoU threshold)
    mAP = precision * recall

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP": round(mAP, 4),
    }


def _compute_single_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)
