import pytest

from benchmark.cli import _infer_yolo_model_info
from benchmark.schemas import YOLOTask


@pytest.mark.parametrize(
    "model_name,expected_version,expected_task",
    [
        ("yolov8n.pt", "v8", YOLOTask.DETECTION),
        ("yolov8s-seg.pt", "v8", YOLOTask.SEGMENTATION),
        ("yolov8m-pose.pt", "v8", YOLOTask.POSE),
        ("yolov8x-obb.pt", "v8", YOLOTask.OBB),
        ("yolov8l-cls.pt", "v8", YOLOTask.CLASSIFICATION),
        ("yolov8x-pose-p6.pt", "v8", YOLOTask.POSE),
        ("yolo11n.pt", "v11", YOLOTask.DETECTION),
        ("yolo11x-seg.pt", "v11", YOLOTask.SEGMENTATION),
        ("yolo11m-pose.pt", "v11", YOLOTask.POSE),
        ("yolo11s-obb.pt", "v11", YOLOTask.OBB),
        ("yolo11l-cls.pt", "v11", YOLOTask.CLASSIFICATION),
        ("yolo26n.pt", "v26", YOLOTask.DETECTION),
        ("yolo26m-cls.pt", "v26", YOLOTask.CLASSIFICATION),
        ("yolo26x-seg.pt", "v26", YOLOTask.SEGMENTATION),
    ],
)
def test_infer_known_filenames(model_name, expected_version, expected_task):
    version, task = _infer_yolo_model_info(model_name)
    assert version == expected_version
    assert task == expected_task


def test_case_insensitive():
    assert _infer_yolo_model_info("YOLOv8N.pt") == ("v8", YOLOTask.DETECTION)
    assert _infer_yolo_model_info("YOLO11X-SEG.pt") == ("v11", YOLOTask.SEGMENTATION)


def test_unknown_pattern_falls_back_to_v8_detection():
    # CLAUDE.md documents this fallback — keep the contract pinned.
    version, task = _infer_yolo_model_info("not-a-yolo-model.pt")
    assert version == "v8"
    assert task == YOLOTask.DETECTION
