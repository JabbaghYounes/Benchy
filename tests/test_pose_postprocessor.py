"""Tests for `YOLOPostProcessor._process_pose` (Phase 3c).

Exercises the pose dispatch path with synthetic head outputs — no actual
yolo*-pose.pt model load required. Confirms decoding maths, keypoint
shape, sigmoid visibility transform, COCO-17 default keypoint names,
and that `PoseResult.to_dict()` includes the keypoints array (unlike
segmentation masks, which are deliberately dropped).
"""
import json

import numpy as np
import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.postprocessing import (
    PoseResult,
    PostProcessConfig,
    YOLOPostProcessor,
)


# ----- helpers --------------------------------------------------------------


NUM_KPTS = YOLOPostProcessor.POSE_NUM_KEYPOINTS  # 17 for COCO-Pose
KPT_DIMS = YOLOPostProcessor.POSE_KEYPOINT_CHANNELS  # 3 (x, y, vis)


def _make_pose_outputs(
    num_classes: int = 1,
    num_anchors: int = 3,
    seed: int = 7,
) -> dict:
    """Build a synthetic yolo-pose head output.

    Anchor 0: high-conf person at xywh (320, 320, 100, 200), strong vis logits.
    Anchor 1: below-confidence anchor (will be dropped).
    Anchor 2: high-conf person at xywh (200, 400, 60, 120), weak vis logits.
    """
    rng = np.random.RandomState(seed)
    channels = 4 + num_classes + NUM_KPTS * KPT_DIMS
    raw = np.zeros((1, num_anchors, channels), dtype=np.float32)

    # Anchor 0
    raw[0, 0, :4] = [320, 320, 100, 200]
    raw[0, 0, 4] = 0.9  # person score
    for k in range(NUM_KPTS):
        raw[0, 0, 4 + num_classes + k * KPT_DIMS + 0] = 320 + (k - 8) * 10
        raw[0, 0, 4 + num_classes + k * KPT_DIMS + 1] = 320 + (k - 8) * 5
        raw[0, 0, 4 + num_classes + k * KPT_DIMS + 2] = 3.0  # logit -> ~0.95

    # Anchor 1: below threshold
    raw[0, 1, :4] = [50, 50, 20, 20]
    raw[0, 1, 4] = 0.05

    # Anchor 2
    raw[0, 2, :4] = [200, 400, 60, 120]
    raw[0, 2, 4] = 0.85
    for k in range(NUM_KPTS):
        raw[0, 2, 4 + num_classes + k * KPT_DIMS + 0] = 200 + rng.randn() * 5
        raw[0, 2, 4 + num_classes + k * KPT_DIMS + 1] = 400 + rng.randn() * 5
        raw[0, 2, 4 + num_classes + k * KPT_DIMS + 2] = -1.5  # logit -> ~0.18

    return {"output": raw}


@pytest.fixture
def processor() -> YOLOPostProcessor:
    return YOLOPostProcessor(YOLOTask.POSE)


@pytest.fixture
def cfg() -> PostProcessConfig:
    return PostProcessConfig(
        conf_threshold=0.25,
        iou_threshold=0.5,
        num_classes=1,
        input_width=640,
        input_height=640,
    )


# ----- dispatch -------------------------------------------------------------


def test_process_dispatches_pose_to_pose_results(processor, cfg):
    out = processor.process(_make_pose_outputs(), cfg)
    assert all(isinstance(r, PoseResult) for r in out)


def test_process_pose_drops_below_confidence(processor, cfg):
    out = processor.process(_make_pose_outputs(), cfg)
    # Anchor 1 has score 0.05; should be dropped. Anchors 0 and 2 survive.
    assert len(out) == 2


def test_process_pose_decodes_bbox_correctly(processor, cfg):
    out = processor.process(_make_pose_outputs(), cfg)
    # cx=320, cy=320, w=100, h=200 -> (270, 220, 370, 420)
    by_conf = sorted(out, key=lambda r: r.confidence, reverse=True)
    assert by_conf[0].bbox == pytest.approx((270.0, 220.0, 370.0, 420.0), abs=1e-3)


def test_process_pose_results_sorted_by_score_descending(processor, cfg):
    out = processor.process(_make_pose_outputs(), cfg)
    confidences = [r.confidence for r in out]
    assert confidences == sorted(confidences, reverse=True)


# ----- keypoint shape and content -------------------------------------------


def test_pose_keypoints_have_correct_shape(processor, cfg):
    out = processor.process(_make_pose_outputs(), cfg)
    for r in out:
        assert r.keypoints is not None
        assert r.keypoints.shape == (NUM_KPTS, KPT_DIMS)


def test_pose_visibility_in_unit_interval(processor, cfg):
    """Visibility is sigmoid-applied, so every value must be in [0, 1].
    The raw logit channels are ±3 / ±1.5 in our fixtures; bridge contract
    is that sigmoid maps these into [0, 1] without leaving the bounds.
    """
    out = processor.process(_make_pose_outputs(), cfg)
    for r in out:
        vis = r.keypoints[:, 2]
        assert (vis >= 0.0).all()
        assert (vis <= 1.0).all()


def test_pose_high_logit_maps_to_high_visibility(processor, cfg):
    """Anchor 0 was set with logit=3.0 across all 17 keypoints; sigmoid(3)
    is ~0.953. Anchor 2 was set with logit=-1.5; sigmoid(-1.5) is ~0.182.
    """
    out = processor.process(_make_pose_outputs(), cfg)
    by_conf = sorted(out, key=lambda r: r.confidence, reverse=True)
    high = by_conf[0]  # anchor 0
    low = by_conf[1]   # anchor 2
    assert high.keypoints[:, 2].mean() > 0.9
    assert low.keypoints[:, 2].mean() < 0.3


# ----- coordinate scaling ---------------------------------------------------


def test_pose_keypoints_scale_with_original_size(processor):
    """When original_width/original_height are larger than input, both
    bbox and keypoint coords must scale proportionally.
    """
    cfg = PostProcessConfig(
        conf_threshold=0.25,
        iou_threshold=0.5,
        num_classes=1,
        input_width=640,
        input_height=640,
        original_width=1280,  # 2x scale
        original_height=1280,
    )
    out = processor.process(_make_pose_outputs(), cfg)
    by_conf = sorted(out, key=lambda r: r.confidence, reverse=True)
    high = by_conf[0]
    # Keypoint 0 was at x=240 (320 + (0-8)*10) in input space; scaled 2x -> 480.
    assert high.keypoints[0, 0] == pytest.approx(480.0, abs=1.0)


# ----- to_dict() and JSON serialisability -----------------------------------


def test_pose_to_dict_includes_keypoints(processor, cfg):
    """Unlike segmentation masks (which are deliberately dropped from
    JSON), pose keypoints are small enough to serialise — 17×3 floats.
    """
    out = processor.process(_make_pose_outputs(), cfg)
    payload = out[0].to_dict()
    assert "keypoints" in payload
    assert payload["keypoints"] is not None
    assert len(payload["keypoints"]) == NUM_KPTS
    assert len(payload["keypoints"][0]) == KPT_DIMS


def test_pose_to_dict_is_json_serialisable(processor, cfg):
    out = processor.process(_make_pose_outputs(), cfg)
    json.dumps(out[0].to_dict())  # must not raise


def test_pose_to_dict_with_no_keypoints():
    result = PoseResult(
        bbox=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
        keypoints=None,
    )
    payload = result.to_dict()
    assert payload["keypoints"] is None


# ----- transposition + edge cases -------------------------------------------


def test_pose_handles_channels_first_transpose(processor, cfg):
    raw = _make_pose_outputs()["output"]
    transposed = np.transpose(raw, (0, 2, 1))  # (1, 56, num_anchors)
    out = processor.process({"output": transposed}, cfg)
    assert len(out) == 2


def test_pose_handles_empty_output(processor, cfg):
    raw = np.zeros((1, 1, 4 + 1 + NUM_KPTS * KPT_DIMS))
    out = processor.process({"output": raw}, cfg)
    assert out == []


# ----- COCO-17 keypoint name defaults ---------------------------------------


def test_pose_coco_keypoint_names_have_17_entries():
    assert len(YOLOPostProcessor.COCO_POSE_KEYPOINTS) == NUM_KPTS


def test_pose_coco_keypoint_names_start_with_nose():
    assert YOLOPostProcessor.COCO_POSE_KEYPOINTS[0] == "nose"


def test_pose_coco_keypoint_names_include_left_right_pairs():
    """Sanity check: most non-head keypoints come in left/right pairs.
    Catches accidental edits that drop, say, "left_wrist" but leave
    "right_wrist".
    """
    names = set(YOLOPostProcessor.COCO_POSE_KEYPOINTS)
    paired = ["eye", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
    for kpt in paired:
        assert f"left_{kpt}" in names, f"missing left_{kpt}"
        assert f"right_{kpt}" in names, f"missing right_{kpt}"
