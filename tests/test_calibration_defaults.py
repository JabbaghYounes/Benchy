"""Tests for calibration dataset defaults (Polish 4).

The Hailo conversion pipeline uses
`CalibrationDatasetLoader.DEFAULT_DATASETS` to pick which Ultralytics
dataset to sample 100 calibration images from. Pre-Polish-4 the OBB
and POSE entries were `dota8` / `coco8-pose` — only 8 images each, far
too few for production-quality INT8 quantisation. This test pins the
new contract: full DOTAv1 / coco-pose by default, with the option to
override via `CalibrationConfig.dataset_path` for users with a
pre-curated subset.
"""
from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.calibration import (
    CalibrationConfig,
    CalibrationDatasetLoader,
)


def test_obb_calibration_dataset_is_full_dotav1():
    """OBB calibration must come from a corpus larger than 8 images.
    The previous `dota8` default produced poor quantised mAP because the
    8-image sample didn't cover the angle/scale distribution.
    """
    name = CalibrationDatasetLoader.DEFAULT_DATASETS[YOLOTask.OBB]
    assert name != "dota8", (
        "OBB calibration must not regress to the 8-image dota8 sample"
    )
    assert name.lower().startswith("dota"), (
        f"OBB calibration should still be a DOTA variant; got {name!r}"
    )


def test_pose_calibration_dataset_is_full_coco_pose():
    """Pose calibration must come from a corpus larger than 8 images.
    The previous `coco8-pose` default left visibility-channel
    quantisation under-fit.
    """
    name = CalibrationDatasetLoader.DEFAULT_DATASETS[YOLOTask.POSE]
    assert name != "coco8-pose", (
        "Pose calibration must not regress to the 8-image coco8-pose sample"
    )
    assert "pose" in name.lower(), (
        f"Pose calibration should reference a pose dataset; got {name!r}"
    )


def test_detection_calibration_dataset_unchanged():
    """coco128 (128 images) is already large enough for detection
    calibration; pin it so a future bulk edit doesn't accidentally
    swap to coco8 or similar.
    """
    assert CalibrationDatasetLoader.DEFAULT_DATASETS[YOLOTask.DETECTION] == "coco128"


def test_segmentation_calibration_dataset_unchanged():
    """coco128-seg (128 images) is already large enough for seg
    calibration; same pinning logic as detection.
    """
    assert (
        CalibrationDatasetLoader.DEFAULT_DATASETS[YOLOTask.SEGMENTATION]
        == "coco128-seg"
    )


def test_calibration_num_samples_default_is_100():
    """The 100-sample cap is what makes bumping to full datasets
    tractable — even with DOTAv1 / coco-pose pointing at huge corpora,
    we only sample 100 images for actual calibration. Pin it so a
    future change to the cap is deliberate.
    """
    assert CalibrationConfig().num_samples == 100


def test_calibration_dataset_path_override_is_optional():
    """`CalibrationConfig.dataset_path` is the documented escape hatch
    for users who want to bypass the heavy first-run download. Pin
    that the field exists, defaults to None, and is overridable.
    """
    cfg_default = CalibrationConfig()
    assert cfg_default.dataset_path is None

    from pathlib import Path
    custom = Path("/tmp/my_calibration_subset")
    cfg_override = CalibrationConfig(dataset_path=custom)
    assert cfg_override.dataset_path == custom
