"""Unit tests for the rotated NMS primitives in `YOLOPostProcessor`.

These pin the mathematical contract of `_rotated_iou` and `_rotated_nms`:
they run on any host (no Hailo hardware needed) and exist so a future
refactor of the polygon-clipping helpers can't silently change the
numerical answer.

Bridge contract (the most important property): at `angle=0`, rotated IoU
must equal axis-aligned IoU within 1e-6. That's what makes the new
implementation a drop-in for the existing `_compute_iou` when the angle
happens to be zero.
"""
from math import pi

import numpy as np
import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.postprocessing import YOLOPostProcessor


@pytest.fixture(scope="module")
def processor() -> YOLOPostProcessor:
    return YOLOPostProcessor(YOLOTask.OBB)


# ----- _rotated_iou bridge contract -----------------------------------------


def test_rotated_iou_matches_axis_aligned_at_zero_angle(processor):
    # Box A at (10,10,10x10,0), Box B at (15,15,10x10,0). Axis-aligned
    # overlap = 25, union = 175, IoU ~= 0.142857.
    box_a = np.array([10.0, 10.0, 10.0, 10.0, 0.0])
    box_b = np.array([[15.0, 15.0, 10.0, 10.0, 0.0]])
    iou = processor._rotated_iou(box_a, box_b)
    assert iou.shape == (1,)
    assert abs(iou[0] - 25.0 / 175.0) < 1e-6


def test_rotated_iou_identical_boxes_is_one(processor):
    box = np.array([10.0, 10.0, 10.0, 10.0, pi / 4])
    boxes = np.array([[10.0, 10.0, 10.0, 10.0, pi / 4]])
    iou = processor._rotated_iou(box, boxes)
    assert iou[0] == pytest.approx(1.0, abs=1e-6)


def test_rotated_iou_disjoint_boxes_is_zero(processor):
    box = np.array([10.0, 10.0, 10.0, 10.0, 0.0])
    boxes = np.array([[100.0, 100.0, 10.0, 10.0, 0.0]])
    iou = processor._rotated_iou(box, boxes)
    assert iou[0] == pytest.approx(0.0, abs=1e-6)


def test_rotated_iou_perpendicular_squares_at_same_centre(processor):
    # Two unit squares centred at the origin, one at 0 deg and one at 90 deg.
    # Both squares are identical (rotation by pi/2 maps a square to itself),
    # so IoU must equal 1.
    box = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    boxes = np.array([[0.0, 0.0, 1.0, 1.0, pi / 2]])
    iou = processor._rotated_iou(box, boxes)
    assert iou[0] == pytest.approx(1.0, abs=1e-6)


def test_rotated_iou_handles_45_degree_overlap(processor):
    # Two unit squares at the same centre, one rotated 45 deg. Box A is
    # axis-aligned [-0.5, 0.5]^2; Box B's corners go to (±sqrt(2)/2, 0)
    # and (0, ±sqrt(2)/2), making B a diamond. The intersection is an
    # octagon: each of A's four corners gets clipped by a triangle of
    # leg (1 - sqrt(2)/2). Intersection = 1 - 2*(1 - sqrt(2)/2)^2;
    # union = 2 - intersection; IoU = intersection / union ~= 0.7071.
    box = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    boxes = np.array([[0.0, 0.0, 1.0, 1.0, pi / 4]])
    iou = processor._rotated_iou(box, boxes)
    triangle_offset = 1.0 - np.sqrt(2.0) / 2.0
    inter = 1.0 - 2.0 * triangle_offset ** 2
    expected = inter / (2.0 - inter)
    assert iou[0] == pytest.approx(expected, abs=1e-3)


def test_rotated_iou_empty_input(processor):
    box = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    boxes = np.empty((0, 5))
    iou = processor._rotated_iou(box, boxes)
    assert iou.shape == (0,)


def test_rotated_iou_zero_size_box_returns_zero(processor):
    # Degenerate box (w=h=0). intersection area is 0, union ~ area_b,
    # IoU == 0 without division-by-zero.
    box = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    boxes = np.array([[0.0, 0.0, 1.0, 1.0, 0.0]])
    iou = processor._rotated_iou(box, boxes)
    assert iou[0] == pytest.approx(0.0, abs=1e-6)


# ----- _rotated_nms behaviour -----------------------------------------------


def test_rotated_nms_keeps_non_overlapping(processor):
    boxes = np.array([
        [10.0, 10.0, 5.0, 5.0, 0.0],
        [50.0, 50.0, 5.0, 5.0, 0.0],
        [100.0, 100.0, 5.0, 5.0, 0.0],
    ])
    scores = np.array([0.9, 0.8, 0.7])
    keep = processor._rotated_nms(boxes, scores, 0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]


def test_rotated_nms_suppresses_overlapping(processor):
    # Three nearly-identical boxes; only the highest-score survives.
    boxes = np.array([
        [10.0, 10.0, 10.0, 10.0, 0.0],
        [10.5, 10.5, 10.0, 10.0, 0.0],
        [11.0, 11.0, 10.0, 10.0, 0.0],
    ])
    scores = np.array([0.7, 0.9, 0.8])
    keep = processor._rotated_nms(boxes, scores, 0.5)
    assert keep.tolist() == [1]


def test_rotated_nms_orders_by_score_descending(processor):
    boxes = np.array([
        [10.0, 10.0, 5.0, 5.0, 0.0],
        [50.0, 50.0, 5.0, 5.0, 0.0],
    ])
    scores = np.array([0.6, 0.95])
    keep = processor._rotated_nms(boxes, scores, 0.5)
    # Both are kept (no overlap), but the higher-score box is first.
    assert keep[0] == 1
    assert keep[1] == 0


def test_rotated_nms_empty_input(processor):
    boxes = np.empty((0, 5))
    scores = np.empty((0,))
    keep = processor._rotated_nms(boxes, scores, 0.5)
    assert keep.shape == (0,)


def test_rotated_nms_threshold_zero_keeps_only_top_per_overlap(processor):
    # iou_threshold=0 means any positive overlap suppresses. Two boxes
    # touching at a corner (zero overlap) survive; pairs with even tiny
    # overlap collapse to one.
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0, 0.0],
        [10.0, 10.0, 10.0, 10.0, 0.0],   # corner-touch, IoU == 0
        [1.0, 1.0, 10.0, 10.0, 0.0],     # heavy overlap with [0]
    ])
    scores = np.array([0.9, 0.8, 0.7])
    keep = processor._rotated_nms(boxes, scores, 0.0)
    assert sorted(keep.tolist()) == [0, 1]


# ----- helper invariants ----------------------------------------------------


def test_obb_corners_returns_four_points(processor):
    box = np.array([0.0, 0.0, 4.0, 2.0, 0.0])
    corners = processor._obb_corners(box)
    assert corners.shape == (4, 2)


def test_polygon_signed_area_zero_for_degenerate(processor):
    poly = np.array([[0.0, 0.0], [1.0, 0.0]])  # only 2 points
    assert processor._polygon_signed_area(poly) == 0.0


def test_ensure_ccw_reverses_clockwise_polygon(processor):
    # Clockwise unit square in math y-up coords.
    cw_square = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    ccw = processor._ensure_ccw(cw_square)
    # After reversal, the signed area should be positive (CCW).
    assert processor._polygon_signed_area(ccw) > 0
