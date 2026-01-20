# YOLO Output Post-processing for Hailo NPU
#
# This module provides post-processing functions for YOLO model outputs
# including bounding box decoding and Non-Maximum Suppression (NMS).
#
# Phase 4 - Task 4.1 of Hailo PRD
import logging
from dataclasses import dataclass, field
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

        # Set default class names if not provided
        if self.config.class_names is None:
            if self.config.num_classes == 80:
                self.config.class_names = self.COCO_CLASSES

    def process(
        self,
        outputs: Dict[str, np.ndarray],
        config: Optional[PostProcessConfig] = None,
    ) -> Union[List[Detection], List[ClassificationResult]]:
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
) -> Union[List[Detection], List[ClassificationResult]]:
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
