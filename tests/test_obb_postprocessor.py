"""Tests for `YOLOPostProcessor._process_obb` (Phase 3a).

Exercises the OBB dispatch path with synthetic head outputs — no actual
yolo*-obb.pt model load required. Confirms decoding maths, confidence
thresholding, NMS ordering, transposition handling, and the canonical
[-pi/2, pi/2] angle range.
"""
from math import pi

import numpy as np
import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.postprocessing import (
    OrientedBox,
    PostProcessConfig,
    YOLOPostProcessor,
)


@pytest.fixture
def processor() -> YOLOPostProcessor:
    return YOLOPostProcessor(YOLOTask.OBB)


@pytest.fixture
def cfg() -> PostProcessConfig:
    return PostProcessConfig(
        conf_threshold=0.25, iou_threshold=0.5, num_classes=15
    )


def _make_raw(num_classes: int = 15) -> np.ndarray:
    """Build a (1, 3, 5+num_classes) synthetic OBB head output.

    Anchor 0: high-conf 'plane' (class 0) at (100, 100, 50, 30, +0.2 rad).
    Anchor 1: low-conf, below threshold.
    Anchor 2: high-conf 'ship' (class 1) at (200, 200, 40, 40, +pi/4).
    """
    raw = np.zeros((1, 3, 5 + num_classes))
    raw[0, 0, :5] = [100.0, 100.0, 50.0, 30.0, 0.2]
    raw[0, 0, 5] = 0.9
    raw[0, 1, :5] = [50.0, 50.0, 20.0, 20.0, 0.0]
    raw[0, 1, 5] = 0.05
    raw[0, 2, :5] = [200.0, 200.0, 40.0, 40.0, pi / 4]
    raw[0, 2, 6] = 0.85
    return raw


# ----- end-to-end dispatch --------------------------------------------------


def test_process_dispatches_obb_to_oriented_boxes(processor, cfg):
    raw = _make_raw()
    out = processor.process({"output": raw}, cfg)
    assert all(isinstance(o, OrientedBox) for o in out)


def test_process_obb_drops_below_confidence(processor, cfg):
    raw = _make_raw()
    out = processor.process({"output": raw}, cfg)
    # Anchor 1 is below threshold; only anchors 0 and 2 should survive.
    assert len(out) == 2


def test_process_obb_decodes_geometry(processor, cfg):
    raw = _make_raw()
    out = processor.process({"output": raw}, cfg)
    by_class = {o.class_id: o for o in out}
    plane = by_class[0]
    ship = by_class[1]
    assert plane.cx == pytest.approx(100.0)
    assert plane.cy == pytest.approx(100.0)
    assert plane.w == pytest.approx(50.0)
    assert plane.h == pytest.approx(30.0)
    assert plane.angle_rad == pytest.approx(0.2)
    assert ship.angle_rad == pytest.approx(pi / 4)


def test_process_obb_results_sorted_by_score_descending(processor, cfg):
    raw = _make_raw()
    out = processor.process({"output": raw}, cfg)
    # NMS preserves descending-score order. Anchor 0 score=0.9, anchor 2=0.85.
    confidences = [o.confidence for o in out]
    assert confidences == sorted(confidences, reverse=True)


# ----- transposition handling -----------------------------------------------


def test_process_obb_handles_channels_first_transpose(processor, cfg):
    # Some Hailo HEFs emit (channels, anchors) instead of (anchors, channels).
    # The processor should auto-transpose.
    raw = _make_raw()
    raw_t = np.transpose(raw, (0, 2, 1))  # (1, 5+15, 3)
    out = processor.process({"output": raw_t}, cfg)
    assert len(out) == 2


# ----- angle normalisation --------------------------------------------------


@pytest.mark.parametrize(
    "raw_angle",
    [
        -pi / 4,        # already canonical
        3 * pi / 4,     # +3pi/4 -> -pi/4
        pi,             # pi -> 0 (rectangle symmetry)
        -pi,            # -pi -> 0
        1.7 * pi,       # large positive value, must collapse into range
    ],
)
def test_process_obb_normalises_angle_range(processor, cfg, raw_angle):
    raw = np.zeros((1, 1, 5 + cfg.num_classes))
    raw[0, 0, :5] = [50.0, 50.0, 10.0, 10.0, raw_angle]
    raw[0, 0, 5] = 0.9
    out = processor.process({"output": raw}, cfg)
    assert len(out) == 1
    angle = out[0].angle_rad
    # Always in [-pi/2, pi/2] (canonical OBB range).
    assert -pi / 2 - 1e-6 <= angle <= pi / 2 + 1e-6


def test_process_obb_handles_empty_output(processor, cfg):
    raw = np.zeros((1, 1, 5 + cfg.num_classes))  # all confidences 0 -> below threshold
    out = processor.process({"output": raw}, cfg)
    assert out == []


# ----- DOTA defaults --------------------------------------------------------


def test_processor_defaults_to_dota_classes_for_obb():
    p = YOLOPostProcessor(YOLOTask.OBB)
    assert p.config.num_classes == 15
    names = p.config.class_names
    assert names is not None
    assert names == YOLOPostProcessor.DOTA_CLASSES
    assert names[0] == "plane"


def test_processor_uses_coco_for_detection():
    p = YOLOPostProcessor(YOLOTask.DETECTION)
    assert p.config.num_classes == 80
    names = p.config.class_names
    assert names is not None
    assert names == YOLOPostProcessor.COCO_CLASSES
