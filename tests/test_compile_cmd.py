# Tests for the workstation `compile` CLI subcommand.
#
# `cmd_compile` drives the .pt -> .hef pipeline directly so x86_64
# workstations without HailoRT / a Hailo device can produce HEFs and
# stage them under resources/hefs/ for the Pi-side runtime to pick up.
#
# We mock ModelConversionPipeline so the tests don't need the Hailo SDK
# (which is x86_64 + license-gated). The tests focus on the things that
# are easy to break: arg validation, hw_arch threading,
# canonical-filename staging, and continue-on-failure for batches.
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark import cli as cli_mod
from benchmark.schemas import YOLOTask


@pytest.fixture
def mock_pipeline_factory(monkeypatch):
    """Patch ModelConversionPipeline so cmd_compile can run without the SDK.

    Returns a helper that callers parametrise per-test: it lets tests
    decide which model_name should "succeed" / "fail" and what hef_path
    to claim was produced.
    """

    def make(per_model_outcomes: dict, hef_payload: bytes = b"FAKE-HEF"):
        """per_model_outcomes maps model_name -> 'success'|'fail'."""

        def fake_pipeline_init(self, *_, **__):
            self.cache = MagicMock()
            self.calls = []

        def fake_check_requirements(self):
            return {"hef_compilation": True}

        def fake_convert(self, model_name, version, task, config):
            self.calls.append((model_name, version, task, config))

            outcome = per_model_outcomes.get(model_name, "success")
            result = MagicMock()
            result.model_name = model_name
            result.yolo_version = version
            result.task = task.value if hasattr(task, "value") else task

            if outcome == "raise":
                raise RuntimeError("simulated pipeline crash")

            if outcome == "fail":
                result.success = False
                result.hef_path = None
                result.error = "simulated stage failure"
                result.error_stage = "hef_compilation"
                return result

            # success: write a real file so shutil.copy2 has something
            cache_dir = Path("/tmp/benchy_test_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(model_name).stem
            hef_file = cache_dir / f"{stem}_{config.target_device}.hef"
            hef_file.write_bytes(hef_payload)

            result.success = True
            result.hef_path = hef_file
            result.error = None
            result.error_stage = None
            return result

        monkeypatch.setattr(
            "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline.__init__",
            fake_pipeline_init,
        )
        monkeypatch.setattr(
            "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline.check_requirements",
            fake_check_requirements,
        )
        monkeypatch.setattr(
            "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline.convert",
            fake_convert,
        )

    return make


def _make_args(**overrides):
    """Mimic argparse.Namespace for cmd_compile."""

    defaults = dict(
        hw_arch="hailo10h",
        model=None,
        models=None,
        input_resolution=640,
        calibration_set_size=100,
        calibration_data_path=None,
        compression_level=1,
        output_dir=Path("/tmp/benchy_test_out"),
        force_recompile=False,
    )
    defaults.update(overrides)
    return type("Args", (), defaults)()


# ---------------------------------------------------------------- args


def test_compile_rejects_both_model_and_models(tmp_path, mock_pipeline_factory):
    mock_pipeline_factory({})
    args = _make_args(
        model="yolov8n.pt",
        models="yolov8n.pt,yolo11n.pt",
        output_dir=tmp_path,
    )
    assert cli_mod.cmd_compile(args) == 2


def test_compile_requires_one_of_model_or_models(tmp_path, mock_pipeline_factory):
    mock_pipeline_factory({})
    args = _make_args(model=None, models=None, output_dir=tmp_path)
    assert cli_mod.cmd_compile(args) == 2


def test_compile_rejects_empty_models_list(tmp_path, mock_pipeline_factory):
    mock_pipeline_factory({})
    args = _make_args(models="  , ", output_dir=tmp_path)
    assert cli_mod.cmd_compile(args) == 2


# ----------------------------------------------------------- single model


def test_compile_single_model_stages_canonical_filename(
    tmp_path, mock_pipeline_factory
):
    mock_pipeline_factory({"yolov8n-seg.pt": "success"})
    args = _make_args(
        model="yolov8n-seg.pt",
        hw_arch="hailo10h",
        output_dir=tmp_path,
    )
    rc = cli_mod.cmd_compile(args)
    assert rc == 0

    # canonical: <version>_<task>_<size>_<arch>.hef
    expected = tmp_path / "v8_segmentation_n_hailo10h.hef"
    assert expected.exists()
    assert expected.read_bytes() == b"FAKE-HEF"


def test_compile_threads_hw_arch_into_pipeline(tmp_path, monkeypatch):
    captured: dict = {}

    class _SpyPipeline:
        def __init__(self):
            pass

        def check_requirements(self):
            return {"hef_compilation": True}

        def convert(self, model_name, version, task, config):
            captured["target_device"] = config.target_device
            captured["force_recompile"] = config.force_recompile
            captured["calibration_set_size"] = config.calibration_set_size
            hef = tmp_path / f"{Path(model_name).stem}.hef"
            hef.write_bytes(b"x")
            r = MagicMock()
            r.success = True
            r.hef_path = hef
            r.error = None
            r.error_stage = None
            return r

    monkeypatch.setattr(
        "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline",
        _SpyPipeline,
    )

    args = _make_args(
        model="yolov8n.pt",
        hw_arch="hailo8l",
        force_recompile=True,
        calibration_set_size=42,
        output_dir=tmp_path,
    )
    assert cli_mod.cmd_compile(args) == 0
    assert captured["target_device"] == "hailo8l"
    assert captured["force_recompile"] is True
    assert captured["calibration_set_size"] == 42


# ----------------------------------------------------------- batch model


def test_compile_batch_continues_on_failure(tmp_path, mock_pipeline_factory):
    mock_pipeline_factory(
        {
            "yolov8n.pt": "success",
            "yolo11n-pose.pt": "fail",
            "yolo26n-seg.pt": "success",
        }
    )
    args = _make_args(
        models="yolov8n.pt,yolo11n-pose.pt,yolo26n-seg.pt",
        hw_arch="hailo10h",
        output_dir=tmp_path,
    )
    rc = cli_mod.cmd_compile(args)
    # one failure -> non-zero exit
    assert rc == 1

    # successful entries must still have been staged
    assert (tmp_path / "v8_detection_n_hailo10h.hef").exists()
    assert (tmp_path / "v26_segmentation_n_hailo10h.hef").exists()
    # failed entry must NOT have a staged file
    assert not (tmp_path / "v11_pose_n_hailo10h.hef").exists()


def test_compile_pipeline_exception_does_not_kill_batch(
    tmp_path, mock_pipeline_factory
):
    mock_pipeline_factory(
        {
            "yolov8n.pt": "success",
            "yolo11n.pt": "raise",
            "yolo26n.pt": "success",
        }
    )
    args = _make_args(
        models="yolov8n.pt,yolo11n.pt,yolo26n.pt",
        hw_arch="hailo10h",
        output_dir=tmp_path,
    )
    rc = cli_mod.cmd_compile(args)
    assert rc == 1
    assert (tmp_path / "v8_detection_n_hailo10h.hef").exists()
    assert (tmp_path / "v26_detection_n_hailo10h.hef").exists()
    assert not (tmp_path / "v11_detection_n_hailo10h.hef").exists()


# ------------------------------------------------------------ unhappy paths


def test_compile_returns_3_when_sdk_missing(tmp_path, monkeypatch):
    class _NoSDK:
        def __init__(self):
            pass

        def check_requirements(self):
            return {"hef_compilation": False}

    monkeypatch.setattr(
        "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline",
        _NoSDK,
    )

    args = _make_args(model="yolov8n.pt", output_dir=tmp_path)
    assert cli_mod.cmd_compile(args) == 3


def test_compile_skips_models_with_unparseable_size(
    tmp_path, mock_pipeline_factory
):
    mock_pipeline_factory({"yolov8n.pt": "success"})
    # 'random.pt' has no Ultralytics-style size suffix
    args = _make_args(
        models="yolov8n.pt,random.pt",
        output_dir=tmp_path,
    )
    rc = cli_mod.cmd_compile(args)
    assert rc == 1  # one entry failed
    assert (tmp_path / "v8_detection_n_hailo10h.hef").exists()


# ----------------------------------------- skip-if-already-staged (Issue 3)


def test_compile_skips_when_canonical_hef_already_staged(tmp_path, monkeypatch):
    """A HEF already in resources/hefs/ (e.g. from fetch_prebuilt_hefs.py)
    must not trigger another full pipeline run. The pipeline is the
    expensive bit (5-30 min per model); re-staging the same file
    accomplishes nothing."""
    pre_staged = tmp_path / "v8_detection_n_hailo10h.hef"
    pre_staged.write_bytes(b"fetched-from-zoo")

    convert_calls: list = []

    class _SpyPipeline:
        def __init__(self):
            pass

        def check_requirements(self):
            return {"hef_compilation": True}

        def convert(self, *args, **kwargs):
            convert_calls.append((args, kwargs))
            raise AssertionError("pipeline.convert must not run for already-staged HEFs")

    monkeypatch.setattr(
        "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline",
        _SpyPipeline,
    )

    args = _make_args(
        model="yolov8n.pt",
        hw_arch="hailo10h",
        output_dir=tmp_path,
        force_recompile=False,
    )
    rc = cli_mod.cmd_compile(args)
    assert rc == 0
    assert convert_calls == []  # pipeline never invoked
    # Pre-existing file untouched
    assert pre_staged.read_bytes() == b"fetched-from-zoo"


def test_compile_force_recompile_overrides_skip(tmp_path, mock_pipeline_factory):
    """--force-recompile must re-run the pipeline and overwrite the
    pre-staged HEF. This is the escape hatch when the staged file is
    suspect (wrong DFC version, corrupted, etc.)."""
    pre_staged = tmp_path / "v8_detection_n_hailo10h.hef"
    pre_staged.write_bytes(b"old-hef-suspected-bad")

    mock_pipeline_factory({"yolov8n.pt": "success"}, hef_payload=b"freshly-compiled")

    args = _make_args(
        model="yolov8n.pt",
        hw_arch="hailo10h",
        output_dir=tmp_path,
        force_recompile=True,
    )
    rc = cli_mod.cmd_compile(args)
    assert rc == 0
    assert pre_staged.read_bytes() == b"freshly-compiled"


def test_python_m_benchmark_propagates_exit_code(tmp_path):
    """Regression for the __main__.py bug that silently turned every
    failing `python -m benchmark <cmd>` into exit 0. cmd_compile
    returns 3 when the Hailo SDK isn't installed (which is true for
    the dev venv that runs this test) — that has to make it back to
    the shell, otherwise script wrappers like
    scripts/compile_workstation_hefs.sh log spurious PASSes."""
    import subprocess
    import sys as _sys

    r = subprocess.run(
        [
            _sys.executable,
            "-m",
            "benchmark",
            "compile",
            "--hw-arch",
            "hailo8",
            "--model",
            "yolov8n.pt",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # The dev venv has no hailo_sdk_client → cmd_compile returns 3
    # for "SDK not available" rather than 0 or 1. Either way, the
    # exit code must NOT be 0.
    assert r.returncode != 0, (
        f"Expected non-zero exit, got {r.returncode}. "
        f"stderr: {r.stderr[:500]}"
    )


def test_compile_threads_calibration_data_path_to_pipeline(tmp_path, monkeypatch):
    """--calibration-data-path must reach ConversionConfig so the
    HEFCompiler can hand it to CalibrationConfig.dataset_path. This is
    the escape hatch for users who want to avoid Ultralytics' ~27 GB
    coco auto-download and stage just val2017 (~1 GB) themselves."""
    captured: dict = {}

    class _SpyPipeline:
        def __init__(self):
            pass

        def check_requirements(self):
            return {"hef_compilation": True}

        def convert(self, model_name, version, task, config):
            captured["calibration_data_path"] = config.calibration_data_path
            hef = tmp_path / f"{Path(model_name).stem}.hef"
            hef.write_bytes(b"x")
            r = MagicMock()
            r.success = True
            r.hef_path = hef
            r.error = None
            r.error_stage = None
            return r

    monkeypatch.setattr(
        "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline",
        _SpyPipeline,
    )

    custom = Path("/tmp/my_val2017_subset")
    args = _make_args(
        model="yolov8n.pt",
        hw_arch="hailo8",
        calibration_data_path=custom,
        output_dir=tmp_path,
    )
    assert cli_mod.cmd_compile(args) == 0
    assert captured["calibration_data_path"] == custom


def test_compile_calibration_data_path_defaults_to_none(tmp_path, monkeypatch):
    """When --calibration-data-path is omitted, ConversionConfig must
    receive None so the loader falls back to its DEFAULT_DATASETS
    auto-download path. Catches the regression where _make_args
    forgot the field and AttributeError leaked through."""
    captured: dict = {}

    class _SpyPipeline:
        def __init__(self):
            pass

        def check_requirements(self):
            return {"hef_compilation": True}

        def convert(self, model_name, version, task, config):
            captured["calibration_data_path"] = config.calibration_data_path
            hef = tmp_path / "x.hef"
            hef.write_bytes(b"x")
            r = MagicMock()
            r.success = True
            r.hef_path = hef
            r.error = None
            r.error_stage = None
            return r

    monkeypatch.setattr(
        "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline",
        _SpyPipeline,
    )

    args = _make_args(model="yolov8n.pt", output_dir=tmp_path)
    assert cli_mod.cmd_compile(args) == 0
    assert captured["calibration_data_path"] is None


def test_compile_skip_check_uses_per_arch_filename(tmp_path, monkeypatch):
    """A pre-staged hailo8 HEF must NOT cause a hailo10h compile to skip
    — the canonical filename includes the arch, so v8_detection_n_hailo8.hef
    is not the same artefact as v8_detection_n_hailo10h.hef. Mixing them
    on the Pi would crash at HEF load (HEFs are not cross-arch portable)."""
    # Pre-stage a hailo8 file
    (tmp_path / "v8_detection_n_hailo8.hef").write_bytes(b"hailo8-bytes")

    convert_calls: list = []

    class _SpyPipeline:
        def __init__(self):
            pass

        def check_requirements(self):
            return {"hef_compilation": True}

        def convert(self, model_name, version, task, config):
            convert_calls.append(config.target_device)
            r = MagicMock()
            r.success = True
            hef = tmp_path / "fresh_h10h.hef"
            hef.write_bytes(b"hailo10h-bytes")
            r.hef_path = hef
            r.error = None
            r.error_stage = None
            return r

    monkeypatch.setattr(
        "benchmark.workloads.yolo.conversion.pipeline.ModelConversionPipeline",
        _SpyPipeline,
    )

    # Compile for hailo10h — should NOT be skipped despite hailo8 file present
    args = _make_args(
        model="yolov8n.pt",
        hw_arch="hailo10h",
        output_dir=tmp_path,
    )
    rc = cli_mod.cmd_compile(args)
    assert rc == 0
    assert convert_calls == ["hailo10h"]
    # hailo10h artefact now exists alongside the hailo8 one
    assert (tmp_path / "v8_detection_n_hailo10h.hef").exists()
    assert (tmp_path / "v8_detection_n_hailo8.hef").read_bytes() == b"hailo8-bytes"
