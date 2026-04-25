"""Tests for `YOLOPostProcessor._process_segmentation` (Phase 3b).

Exercises the seg dispatch path with synthetic two-tensor outputs (det
output + mask prototypes). No actual yolo*-seg.pt model load required.
Confirms decoding maths, mask shape correctness, threshold behaviour,
to_dict() not exploding the JSON output, and the dispatch path returning
SegmentationResult instances.
"""
import numpy as np
import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.postprocessing import (
    PostProcessConfig,
    SegmentationResult,
    YOLOPostProcessor,
)


# Synthetic-output helpers --------------------------------------------------


def _make_seg_outputs(
    num_classes: int = 80,
    num_anchors: int = 3,
    proto_h: int = 40,
    proto_w: int = 40,
    seed: int = 42,
) -> dict:
    """Build a (det_output, proto_output) pair that mimics yolov8-seg HEF.

    Anchor 0: high-conf 'person' (class 0) at xywh (320, 320, 100, 200).
    Anchor 1: below-confidence anchor (will be dropped).
    Anchor 2: high-conf 'car' (class 2) at xywh (200, 400, 80, 60).
    """
    rng = np.random.RandomState(seed)
    num_features = 4 + num_classes + YOLOPostProcessor.SEG_NUM_MASK_COEFFS
    det = np.zeros((1, num_anchors, num_features), dtype=np.float32)

    det[0, 0, :4] = [320, 320, 100, 200]
    det[0, 0, 4] = 0.9  # person score
    det[0, 0, 4 + num_classes :] = rng.randn(YOLOPostProcessor.SEG_NUM_MASK_COEFFS) * 0.5

    det[0, 1, :4] = [50, 50, 20, 20]
    det[0, 1, 4] = 0.05  # below threshold

    det[0, 2, :4] = [200, 400, 80, 60]
    det[0, 2, 6] = 0.85  # car score
    det[0, 2, 4 + num_classes :] = rng.randn(YOLOPostProcessor.SEG_NUM_MASK_COEFFS) * 0.5

    proto = (
        rng.randn(1, YOLOPostProcessor.SEG_NUM_MASK_COEFFS, proto_h, proto_w).astype(
            np.float32
        )
        * 0.3
    )
    return {"output0": det, "output1": proto}


@pytest.fixture
def processor() -> YOLOPostProcessor:
    return YOLOPostProcessor(YOLOTask.SEGMENTATION)


@pytest.fixture
def cfg() -> PostProcessConfig:
    return PostProcessConfig(
        conf_threshold=0.25,
        iou_threshold=0.5,
        num_classes=80,
        input_width=640,
        input_height=640,
    )


# ----- end-to-end dispatch --------------------------------------------------


def test_process_dispatches_seg_to_segmentation_results(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    assert all(isinstance(r, SegmentationResult) for r in out)


def test_process_seg_drops_below_confidence(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    # Anchor 1 has score 0.05; should be dropped. Anchors 0 and 2 survive.
    assert len(out) == 2


def test_process_seg_decodes_bboxes_correctly(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    by_class = {r.class_id: r for r in out}
    person = by_class[0]
    car = by_class[2]
    # xywh -> xyxy: cx=320, cy=320, w=100, h=200 -> (270,220)..(370,420)
    assert person.bbox == pytest.approx((270.0, 220.0, 370.0, 420.0), abs=1e-3)
    # cx=200, cy=400, w=80, h=60 -> (160, 370)..(240, 430)
    assert car.bbox == pytest.approx((160.0, 370.0, 240.0, 430.0), abs=1e-3)


def test_process_seg_results_sorted_by_score_descending(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    confidences = [r.confidence for r in out]
    assert confidences == sorted(confidences, reverse=True)


# ----- mask-specific behaviour ---------------------------------------------


def test_seg_results_carry_binary_masks(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    for r in out:
        assert r.mask is not None
        assert r.mask.dtype == np.bool_


def test_seg_mask_dimensions_within_proto_grid(processor, cfg):
    """Each mask should fit inside the prototype grid (40x40 in this test).

    The cropped mask cannot exceed the prototype's spatial extent, and it
    cannot be wider/taller than the bbox it was clipped to (in proto
    coords).
    """
    out = processor.process(_make_seg_outputs(proto_h=40, proto_w=40), cfg)
    for r in out:
        assert r.mask.shape[0] <= 40
        assert r.mask.shape[1] <= 40


def test_seg_mask_threshold_makes_masks_binary(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    for r in out:
        # Boolean array — every entry is True or False, no in-between.
        assert set(np.unique(r.mask).tolist()).issubset({True, False})


# ----- to_dict() behaviour -------------------------------------------------


def test_seg_to_dict_drops_mask_array(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    payload = out[0].to_dict()
    # Mask array is not in the JSON; only metadata flags.
    assert "mask" not in payload
    assert payload["has_mask"] is True
    assert isinstance(payload["mask_pixel_count"], int)


def test_seg_to_dict_pixel_count_matches_mask(processor, cfg):
    out = processor.process(_make_seg_outputs(), cfg)
    for r in out:
        payload = r.to_dict()
        assert payload["mask_pixel_count"] == int(r.mask.sum())


def test_seg_to_dict_with_no_mask():
    result = SegmentationResult(
        bbox=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
        mask=None,
    )
    payload = result.to_dict()
    assert payload["has_mask"] is False
    assert payload["mask_pixel_count"] is None


# ----- output-discovery edge cases ------------------------------------------


def test_seg_handles_proto_without_batch_dim(processor, cfg):
    outputs = _make_seg_outputs()
    # Squeeze batch from prototype to the 3-D variant some HEFs emit.
    outputs["output1"] = outputs["output1"][0]
    out = processor.process(outputs, cfg)
    assert len(out) == 2


def test_seg_handles_empty_when_all_below_threshold(processor):
    cfg_strict = PostProcessConfig(conf_threshold=0.99, num_classes=80)
    out = processor.process(_make_seg_outputs(), cfg_strict)
    assert out == []


def test_seg_returns_empty_when_proto_missing(processor, cfg):
    outputs = _make_seg_outputs()
    del outputs["output1"]  # drop the prototype tensor entirely
    out = processor.process(outputs, cfg)
    assert out == []
