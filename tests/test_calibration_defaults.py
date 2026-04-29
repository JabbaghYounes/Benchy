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


def test_segmentation_calibration_dataset_is_full_coco():
    """Segmentation calibration must come from a corpus large enough
    to enable Hailo's bias-correction passes (≥1024 samples).
    coco128-seg (128 images) was insufficient — the optimizer dropped
    to level 0 and left biases at 16-bit, which then failed chip
    mapping on Hailo-8 with DW-resources / kernel-validation errors
    on the seg head's activation2 (mask coefficient Sigmoid). The
    full coco dataset (~5000 val images) provides enough headroom.
    See the rationale comment on DEFAULT_DATASETS in calibration.py.
    """
    name = CalibrationDatasetLoader.DEFAULT_DATASETS[YOLOTask.SEGMENTATION]
    assert name != "coco128-seg", (
        "Segmentation calibration must not regress to coco128-seg — "
        "it has only 128 images, below Hailo's 1024 threshold for "
        "INT8 bias correction"
    )
    assert "coco" in name.lower(), (
        f"Segmentation calibration should reference a coco-derived "
        f"dataset; got {name!r}"
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


def test_calibration_cache_key_includes_dataset_identity():
    """Changing DEFAULT_DATASETS or CalibrationConfig.dataset_path must
    produce a different cache filename, so changing the dataset on a
    machine with a stale cache invalidates the old entry automatically
    instead of silently returning yesterday's data. This was the bug
    that masked the SEG dataset bump on 2026-04-29 — the cache key
    was (task, num_samples, resolution, seed), so a cached
    coco128-seg run kept being returned even after we switched
    DEFAULT_DATASETS to coco.
    """
    from pathlib import Path
    loader = CalibrationDatasetLoader()
    cfg = CalibrationConfig(num_samples=100, input_resolution=640, seed=42)

    # Default-path key embeds the DEFAULT_DATASETS name.
    default_key = loader._get_cache_path(YOLOTask.SEGMENTATION, cfg).name
    assert "segmentation" in default_key
    assert (
        CalibrationDatasetLoader.DEFAULT_DATASETS[YOLOTask.SEGMENTATION]
        in default_key
    ), (
        f"Cache key {default_key!r} should embed the dataset name "
        f"{CalibrationDatasetLoader.DEFAULT_DATASETS[YOLOTask.SEGMENTATION]!r}"
    )

    # Custom path produces a different key.
    cfg_custom = CalibrationConfig(
        num_samples=100, input_resolution=640, seed=42,
        dataset_path=Path("/tmp/my_subset"),
    )
    custom_key = loader._get_cache_path(YOLOTask.SEGMENTATION, cfg_custom).name
    assert custom_key != default_key

    # Two custom paths produce two distinct keys.
    cfg_other = CalibrationConfig(
        num_samples=100, input_resolution=640, seed=42,
        dataset_path=Path("/tmp/another_subset"),
    )
    other_key = loader._get_cache_path(YOLOTask.SEGMENTATION, cfg_other).name
    assert other_key != custom_key

    # Same custom path produces a deterministic key (cache rehit works).
    cfg_repeat = CalibrationConfig(
        num_samples=100, input_resolution=640, seed=42,
        dataset_path=Path("/tmp/my_subset"),
    )
    assert loader._get_cache_path(YOLOTask.SEGMENTATION, cfg_repeat).name == custom_key


def test_calibration_dataset_path_overrides_default(tmp_path):
    """When CalibrationConfig.dataset_path is set, the loader must
    walk that directory instead of consulting DEFAULT_DATASETS. This
    is the path the compile CLI's --calibration-data-path flag ends
    up using, and the escape hatch from the ~27 GB auto-download.
    """
    # Stage a tiny image dir so the loader has something real to find.
    img_dir = tmp_path / "my_calib"
    img_dir.mkdir()
    # Real-enough JPEG bytes — cv2 / PIL will reject junk, but the
    # path-walking step doesn't care about content. We're testing the
    # selection logic, not the decode logic.
    for i in range(3):
        (img_dir / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")

    loader = CalibrationDatasetLoader()
    cfg = CalibrationConfig(
        num_samples=2, input_resolution=640, seed=42,
        dataset_path=img_dir,
    )

    # _get_dataset_images should walk our directory, not consult
    # DEFAULT_DATASETS or trigger an Ultralytics download.
    paths = loader._get_dataset_images(YOLOTask.SEGMENTATION, cfg)
    assert len(paths) == 3
    assert all(p.parent == img_dir for p in paths)


def test_calibration_cache_key_excludes_legacy_format():
    """The pre-2026-04-29 cache key was
    `<task>_<n>_<resolution>_<seed>.npz` with no dataset identity.
    Pin that we don't regress to that format — otherwise tomorrow's
    bump of DEFAULT_DATASETS to coco-stuff or whatever would silently
    serve coco data."""
    loader = CalibrationDatasetLoader()
    cfg = CalibrationConfig(num_samples=100, input_resolution=640, seed=42)
    key = loader._get_cache_path(YOLOTask.SEGMENTATION, cfg).name
    # legacy format would be exactly "segmentation_100_640_42.npz"
    assert key != "segmentation_100_640_42.npz", (
        f"Cache key regressed to the legacy (task, num, res, seed) "
        f"format: {key}. The dataset name must be part of the key."
    )
