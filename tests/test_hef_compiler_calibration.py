# Regression tests for HEFCompiler calibration data shape.
#
# Hailo's runner.optimize() expects calibration data as a single
# numpy array of shape (N, H, W, C). Passing a Python list of
# (H, W, C) arrays — what CalibrationDataset.images is — fails with
# "Couldn't detect CalibrationDataType" because Hailo's type detector
# sees the list as one-array-per-input-layer rather than
# one-array-per-sample.
#
# These tests fix the optimize() call site by checking the actual
# shape passed in, so a future refactor can't silently regress to the
# list form.
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.calibration import (
    CalibrationConfig,
    CalibrationDataset,
)
from benchmark.workloads.yolo.conversion.hef_compiler import (
    HEFCompiler,
    HEFCompilerConfig,
)


def _make_calibration_dataset(num_samples: int = 4) -> CalibrationDataset:
    """Tiny synthetic CalibrationDataset for tests — the real loader
    pulls coco128 etc. off disk, which is wrong for unit tests."""
    images = [
        (np.random.rand(640, 640, 3) * 255).astype(np.float32) / 255.0
        for _ in range(num_samples)
    ]
    return CalibrationDataset(
        images=images,
        image_paths=[Path(f"/tmp/fake_{i}.jpg") for i in range(num_samples)],
        config=CalibrationConfig(num_samples=num_samples, input_resolution=640),
        dataset_hash="deadbeef",
        task=YOLOTask.SEGMENTATION,
    )


@pytest.fixture
def fake_hailo_sdk(monkeypatch):
    """Inject a fake hailo_sdk_client.ClientRunner that records the
    calib_data shape its optimize() was called with."""
    captured: dict = {}

    class _FakeClientRunner:
        def __init__(self, har=None, hw_arch="hailo8"):
            captured["har"] = har
            captured["hw_arch"] = hw_arch

        def optimize(self, calib_data, *args, **kwargs):
            captured["optimize_called_with"] = calib_data

        def compile(self):
            return b"FAKE-HEF-BYTES"

    fake_module = types.ModuleType("hailo_sdk_client")
    fake_module.ClientRunner = _FakeClientRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", fake_module)
    return captured


def _make_compiler(monkeypatch):
    """HEFCompiler with metadata writes neutered so the test doesn't
    have to set up a real cache directory."""
    compiler = HEFCompiler()
    monkeypatch.setattr(compiler, "_update_metadata", lambda *a, **kw: None)
    # is_available is cached on the instance; force a value rather than
    # trigger another import attempt against the fake module.
    compiler._compiler_available = True
    return compiler


def test_optimize_receives_stacked_numpy_array_not_list(
    fake_hailo_sdk, monkeypatch, tmp_path
):
    """The bug: passing CalibrationDataset.images straight through to
    runner.optimize() fails with 'Couldn't detect CalibrationDataType'
    because Hailo wants (N, H, W, C), not list[(H, W, C)]."""
    compiler = _make_compiler(monkeypatch)

    har_path = tmp_path / "fake.har"
    har_path.write_bytes(b"FAKE-HAR")
    out_path = tmp_path / "out.hef"

    dataset = _make_calibration_dataset(num_samples=5)
    config = HEFCompilerConfig(target_device="hailo8")

    compiler._compile_with_sdk(
        har_path=har_path,
        model_name="yolo11n-seg.pt",
        yolo_version="v11",
        task=YOLOTask.SEGMENTATION,
        config=config,
        output_path=out_path,
        calibration_path=None,
        calibration_dataset=dataset,
    )

    calib_arg = fake_hailo_sdk["optimize_called_with"]
    assert isinstance(calib_arg, np.ndarray), (
        f"optimize() got {type(calib_arg).__name__}, "
        f"expected numpy.ndarray (a Python list trips Hailo's "
        f"calibration-type detector with 'Couldn't detect "
        f"CalibrationDataType')"
    )
    assert calib_arg.shape == (5, 640, 640, 3)
    assert calib_arg.dtype == np.float32


def test_optimize_receives_stacked_array_for_legacy_path(
    fake_hailo_sdk, monkeypatch, tmp_path
):
    """The legacy path (no CalibrationDataset, calibration_path on
    disk) must also stack to (N, H, W, C). Otherwise calibration_data
    formats other than the Phase 3 dataset silently fall back into
    the same trap."""
    compiler = _make_compiler(monkeypatch)

    har_path = tmp_path / "fake.har"
    har_path.write_bytes(b"FAKE-HAR")
    out_path = tmp_path / "out.hef"

    # Bypass real disk loading — stub _load_calibration_data to return
    # a list (mimicking what _load_images_from_dir does).
    fake_list = [
        np.random.rand(640, 640, 3).astype(np.float32) for _ in range(3)
    ]
    monkeypatch.setattr(
        compiler, "_load_calibration_data",
        lambda path, n, runner: fake_list,
    )

    config = HEFCompilerConfig(target_device="hailo8", use_ultralytics_dataset=False)
    compiler._compile_with_sdk(
        har_path=har_path,
        model_name="yolov8n.pt",
        yolo_version="v8",
        task=YOLOTask.DETECTION,
        config=config,
        output_path=out_path,
        calibration_path=tmp_path / "fake_calib_dir",
        calibration_dataset=None,
    )

    calib_arg = fake_hailo_sdk["optimize_called_with"]
    assert isinstance(calib_arg, np.ndarray)
    assert calib_arg.shape == (3, 640, 640, 3)


def test_compile_writes_hef_bytes_to_output(
    fake_hailo_sdk, monkeypatch, tmp_path
):
    """End-to-end smoke: a successful _compile_with_sdk must persist
    the bytes returned by runner.compile() to the output path."""
    compiler = _make_compiler(monkeypatch)

    har_path = tmp_path / "fake.har"
    har_path.write_bytes(b"FAKE-HAR")
    out_path = tmp_path / "out.hef"

    dataset = _make_calibration_dataset(num_samples=2)
    compiler._compile_with_sdk(
        har_path=har_path,
        model_name="yolo11n-seg.pt",
        yolo_version="v11",
        task=YOLOTask.SEGMENTATION,
        config=HEFCompilerConfig(target_device="hailo8"),
        output_path=out_path,
        calibration_path=None,
        calibration_dataset=dataset,
    )

    assert out_path.exists()
    assert out_path.read_bytes() == b"FAKE-HEF-BYTES"


def test_calibration_dataset_to_numpy_batch_is_used():
    """Direct verification that CalibrationDataset.to_numpy_batch
    produces the exact (N, H, W, C) shape the compiler expects.
    A regression here (e.g. switching to channels-first) would
    silently break Hailo even though the compiler's call site looks
    correct."""
    dataset = _make_calibration_dataset(num_samples=7)
    batch = dataset.to_numpy_batch()
    assert batch.shape == (7, 640, 640, 3)
    assert batch.dtype == np.float32
    # And the values match the underlying images, in order.
    np.testing.assert_array_equal(batch[0], dataset.images[0])
    np.testing.assert_array_equal(batch[6], dataset.images[6])
