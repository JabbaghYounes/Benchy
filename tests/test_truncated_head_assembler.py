"""Tests for `YOLOPostProcessor._assemble_truncated_head`.

Truncated-head HEFs (per END_NODE_TABLE in conversion/har_generator.py)
emit per-stride raw conv outputs instead of a single combined-head
tensor. The assembler reconstructs the combined layout the existing
decoders expect: DFL-decoded box + sigmoid'd cls + task-specific
extras (kpts, angle, mask coefficients).

Tests build synthetic per-stride NHWC fixtures with one anchor cell
seeded to encode a known box / class / extras at a known position,
then assert the assembled tensor reproduces those numbers — and that
running the full _process_* pipeline with the assembled outputs
yields the seeded detection.
"""
import math

import numpy as np
import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.postprocessing import (
    PostProcessConfig,
    YOLOPostProcessor,
)


# ----- helpers --------------------------------------------------------------


# 640 input → strides 8 / 16 / 32 → feature maps 80×80 / 40×40 / 20×20.
STRIDES = [(80, 80, 8), (40, 40, 16), (20, 20, 32)]
DFL_BINS = YOLOPostProcessor.DFL_BINS  # 16


def _seed_dfl_at_distance(
    box_branch: np.ndarray, h_idx: int, w_idx: int, distances: tuple
) -> None:
    """Seed a 4×16 DFL distribution at one cell so its expectation == distances.

    Uses a delta distribution: one bin set to a large value so softmax
    concentrates almost all mass on the nearest integer bin.
    """
    for edge_idx, dist in enumerate(distances):
        bin_idx = int(round(dist))
        bin_idx = max(0, min(DFL_BINS - 1, bin_idx))
        # Slot offset: edge edge_idx occupies 16 channels in the 64-ch dim.
        for k in range(DFL_BINS):
            box_branch[h_idx, w_idx, edge_idx * DFL_BINS + k] = (
                10.0 if k == bin_idx else -10.0
            )


def _build_empty_strides(num_classes: int, extras: dict) -> dict:
    """Build an outputs dict with all-zero per-stride branches.

    `extras` keys: 'kpts' (51), 'angle' (1), 'coeffs' (32). Add the seg
    prototype separately via `_with_seg_proto` if needed.
    """
    outputs = {}
    for h, w, stride in STRIDES:
        outputs[f"box_s{stride}"] = np.full(
            (1, h, w, 4 * DFL_BINS), -10.0, dtype=np.float32
        )
        outputs[f"cls_s{stride}"] = np.full(
            (1, h, w, num_classes), -10.0, dtype=np.float32
        )
        if "kpts" in extras:
            outputs[f"kpts_s{stride}"] = np.zeros(
                (1, h, w, 51), dtype=np.float32
            )
        if "angle" in extras:
            outputs[f"angle_s{stride}"] = np.zeros(
                (1, h, w, 1), dtype=np.float32
            )
        if "coeffs" in extras:
            outputs[f"coeffs_s{stride}"] = np.zeros(
                (1, h, w, 32), dtype=np.float32
            )
    return outputs


def _with_seg_proto(outputs: dict, input_h: int = 640, input_w: int = 640):
    h_p, w_p = input_h // 4, input_w // 4
    outputs["proto"] = np.zeros((1, h_p, w_p, 32), dtype=np.float32)


# ----- tests: detection ----------------------------------------------------


def test_assemble_detection_recovers_boxes_and_classes():
    proc = YOLOPostProcessor(YOLOTask.DETECTION)
    cfg = PostProcessConfig(num_classes=80)
    outputs = _build_empty_strides(num_classes=80, extras={})

    # Seed: stride-8 cell (10, 10), class 5, ltrb distances (4, 5, 6, 7) in
    # feature-map units. Anchor center is (10.5, 10.5) at stride 8.
    box_s8 = outputs["box_s8"][0]  # (80, 80, 64)
    cls_s8 = outputs["cls_s8"][0]
    _seed_dfl_at_distance(box_s8, 10, 10, (4, 5, 6, 7))
    cls_s8[10, 10, 5] = 8.0  # high logit -> sigmoid ~ 0.999

    assembled = proc._assemble_truncated_head(outputs, YOLOTask.DETECTION, cfg)
    assert assembled is not None
    combined = assembled["combined"][0]  # (anchors, 4 + 80)
    # Total anchors: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
    assert combined.shape == (8400, 84)

    # Find the seeded anchor (stride 8, row 10, col 10) — it sits at index
    # 10 * 80 + 10 = 810 in the stride-8 block (which comes first).
    idx = 10 * 80 + 10
    cx, cy, w, h = combined[idx, :4]
    # Expected:  x1 = (10.5 - 4) * 8 = 52,  x2 = (10.5 + 6) * 8 = 132
    #            y1 = (10.5 - 5) * 8 = 44,  y2 = (10.5 + 7) * 8 = 140
    #            cx = 92, cy = 92, w = 80, h = 96
    assert math.isclose(cx, 92.0, abs_tol=0.5)
    assert math.isclose(cy, 92.0, abs_tol=0.5)
    assert math.isclose(w, 80.0, abs_tol=0.5)
    assert math.isclose(h, 96.0, abs_tol=0.5)
    assert combined[idx, 4 + 5] > 0.99  # class 5 sigmoid'd score


def test_assemble_detection_full_decode_yields_one_detection():
    proc = YOLOPostProcessor(YOLOTask.DETECTION)
    cfg = PostProcessConfig(num_classes=80, conf_threshold=0.5)
    outputs = _build_empty_strides(num_classes=80, extras={})

    box_s8 = outputs["box_s8"][0]
    cls_s8 = outputs["cls_s8"][0]
    _seed_dfl_at_distance(box_s8, 10, 10, (4, 5, 6, 7))
    cls_s8[10, 10, 5] = 8.0

    detections = proc._process_detection(outputs, cfg)
    assert len(detections) == 1
    d = detections[0]
    assert d.class_id == 5
    assert d.confidence > 0.99
    x1, y1, x2, y2 = d.bbox
    assert math.isclose(x1, 52.0, abs_tol=1.0)
    assert math.isclose(y1, 44.0, abs_tol=1.0)
    assert math.isclose(x2, 132.0, abs_tol=1.0)
    assert math.isclose(y2, 140.0, abs_tol=1.0)


# ----- tests: pose ---------------------------------------------------------


def test_assemble_pose_recovers_keypoints():
    proc = YOLOPostProcessor(YOLOTask.POSE)
    cfg = PostProcessConfig(num_classes=1, conf_threshold=0.5)
    outputs = _build_empty_strides(num_classes=1, extras={"kpts"})

    # Seed: stride-8 cell (10, 10), one person, kpt 0 at raw (0.25, 0.25,
    # large_logit). Decoded kpt 0:
    #   x = (0.25 * 2 + (10 - 0.5)) * 8 = (0.5 + 9.5) * 8 = 80
    #   y = (0.25 * 2 + (10 - 0.5)) * 8 = 80
    #   v = sigmoid(5) ~ 0.993
    box_s8 = outputs["box_s8"][0]
    cls_s8 = outputs["cls_s8"][0]
    kpts_s8 = outputs["kpts_s8"][0]
    _seed_dfl_at_distance(box_s8, 10, 10, (4, 5, 6, 7))
    cls_s8[10, 10, 0] = 8.0
    kpts_s8[10, 10, 0] = 0.25  # kpt 0 raw_x
    kpts_s8[10, 10, 1] = 0.25  # kpt 0 raw_y
    kpts_s8[10, 10, 2] = 5.0   # kpt 0 visibility logit

    pose_results = proc._process_pose(outputs, cfg)
    assert len(pose_results) == 1
    pr = pose_results[0]
    assert pr.confidence > 0.99
    assert pr.keypoints is not None
    assert pr.keypoints.shape == (17, 3)
    x0, y0, v0 = pr.keypoints[0]
    assert math.isclose(x0, 80.0, abs_tol=1.0)
    assert math.isclose(y0, 80.0, abs_tol=1.0)
    assert v0 > 0.99


# ----- tests: OBB ----------------------------------------------------------


def test_assemble_obb_recovers_angle():
    proc = YOLOPostProcessor(YOLOTask.OBB)
    cfg = PostProcessConfig(num_classes=15, conf_threshold=0.5)
    outputs = _build_empty_strides(num_classes=15, extras={"angle"})

    # Seed: stride-8 cell (10, 10), class 3, angle raw 0.0.
    # Decoded angle = (sigmoid(0) - 0.25) * pi = (0.5 - 0.25) * pi = pi/4.
    # After OBB normalisation [-pi/2, pi/2]: pi/4 stays at pi/4.
    box_s8 = outputs["box_s8"][0]
    cls_s8 = outputs["cls_s8"][0]
    _seed_dfl_at_distance(box_s8, 10, 10, (4, 5, 6, 7))
    cls_s8[10, 10, 3] = 8.0
    # angle_s8 is already 0.0

    obb_results = proc._process_obb(outputs, cfg)
    assert len(obb_results) == 1
    o = obb_results[0]
    assert o.class_id == 3
    assert math.isclose(o.angle_rad, math.pi / 4, abs_tol=0.05)


# ----- tests: segmentation -------------------------------------------------


def test_assemble_seg_recovers_detection_and_proto_shape():
    proc = YOLOPostProcessor(YOLOTask.SEGMENTATION)
    cfg = PostProcessConfig(num_classes=80, conf_threshold=0.5)
    outputs = _build_empty_strides(num_classes=80, extras={"coeffs"})
    _with_seg_proto(outputs)

    # Seed: stride-8 cell (10, 10), class 5, mask coeffs all zero (will
    # produce a degenerate empty mask — that's fine for the shape test).
    box_s8 = outputs["box_s8"][0]
    cls_s8 = outputs["cls_s8"][0]
    _seed_dfl_at_distance(box_s8, 10, 10, (4, 5, 6, 7))
    cls_s8[10, 10, 5] = 8.0

    assembled = proc._assemble_truncated_head(
        outputs, YOLOTask.SEGMENTATION, cfg
    )
    assert assembled is not None
    assert "combined" in assembled
    assert "prototype" in assembled
    # Prototype must be NCHW (1, 32, 160, 160) so _get_seg_outputs picks it up.
    assert assembled["prototype"].shape == (1, 32, 160, 160)
    # Combined: (1, 8400, 4 + 80 + 32)
    assert assembled["combined"].shape == (1, 8400, 116)

    seg_results = proc._process_segmentation(outputs, cfg)
    assert len(seg_results) == 1
    assert seg_results[0].class_id == 5


# ----- tests: combined-head fallthrough ------------------------------------


def test_assembler_returns_none_for_combined_head_layout():
    """When given a single combined-head tensor (no per-stride split),
    assembler must return None so the caller's existing decode path runs.
    """
    proc = YOLOPostProcessor(YOLOTask.DETECTION)
    cfg = PostProcessConfig(num_classes=80)
    # Single (1, 8400, 84) combined-head tensor.
    combined = np.zeros((1, 8400, 84), dtype=np.float32)
    outputs = {"output0": combined}
    assembled = proc._assemble_truncated_head(outputs, YOLOTask.DETECTION, cfg)
    assert assembled is None


def test_assembler_returns_none_for_single_stride_only():
    """One stride alone is not enough — could be combined-head reshaped.
    Bail rather than guess.
    """
    proc = YOLOPostProcessor(YOLOTask.DETECTION)
    cfg = PostProcessConfig(num_classes=80)
    outputs = {
        "box": np.zeros((1, 80, 80, 64), dtype=np.float32),
        "cls": np.zeros((1, 80, 80, 80), dtype=np.float32),
    }
    assembled = proc._assemble_truncated_head(outputs, YOLOTask.DETECTION, cfg)
    assert assembled is None


def test_assembler_returns_none_when_branches_missing():
    """If a stride has no recognisable cls branch, assembler bails."""
    proc = YOLOPostProcessor(YOLOTask.DETECTION)
    cfg = PostProcessConfig(num_classes=80)
    outputs = {}
    for h, w, stride in STRIDES:
        outputs[f"box_s{stride}"] = np.zeros(
            (1, h, w, 4 * DFL_BINS), dtype=np.float32
        )
        # Deliberately omit cls branches.
    assembled = proc._assemble_truncated_head(outputs, YOLOTask.DETECTION, cfg)
    assert assembled is None
