# Tests for the prebuilt HEF source layer (Issue 11).
#
# The source layer lets the Hailo backend find HEFs without invoking
# the x86_64-only Dataflow Compiler. We verify naming-convention
# parsing, repo-vs-system search precedence, and the unhappy paths.
from pathlib import Path

import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.hef_source import (
    SYSTEM_PACKAGE_MAP,
    find_prebuilt_hef,
    model_size_from_name,
    repo_filename,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("yolov8n.pt", "n"),
        ("yolov8s-seg.pt", "s"),
        ("yolo11m-pose.pt", "m"),
        ("yolo26l-obb.pt", "l"),
        ("YOLOV8X.PT", "x"),
        ("custom_model.pt", None),
        ("yolo.pt", None),  # no size suffix
    ],
)
def test_model_size_from_name(name: str, expected):
    assert model_size_from_name(name) == expected


def test_repo_filename_format():
    assert (
        repo_filename("v8", YOLOTask.DETECTION, "n", "hailo8")
        == "v8_detection_n_hailo8.hef"
    )
    assert (
        repo_filename("v11", YOLOTask.OBB, "s", "hailo10h")
        == "v11_obb_s_hailo10h.hef"
    )


def test_find_prebuilt_returns_none_when_size_unparseable(tmp_path):
    # Empty resources/hefs/ + empty system dir + unparseable name -> None.
    repo_dir = tmp_path / "hefs"
    repo_dir.mkdir()
    sys_dir = tmp_path / "system"
    sys_dir.mkdir()

    found = find_prebuilt_hef(
        model_name="weird-name.pt",
        yolo_version="v8",
        task=YOLOTask.DETECTION,
        arch="hailo8",
        repo_hef_dir=repo_dir,
        system_hef_dir=sys_dir,
    )
    assert found is None


def test_find_prebuilt_prefers_repo_over_system(tmp_path):
    repo_dir = tmp_path / "hefs"
    repo_dir.mkdir()
    sys_dir = tmp_path / "system"
    sys_dir.mkdir()

    # Stage a HEF in BOTH locations using the system-package mapping.
    repo_file = repo_dir / "v8_detection_s_hailo8.hef"
    repo_file.write_bytes(b"REPO")
    sys_file = sys_dir / "yolov8s_h8.hef"  # mapped name
    sys_file.write_bytes(b"SYS")

    found = find_prebuilt_hef(
        model_name="yolov8s.pt",
        yolo_version="v8",
        task=YOLOTask.DETECTION,
        arch="hailo8",
        repo_hef_dir=repo_dir,
        system_hef_dir=sys_dir,
    )
    assert found == repo_file


def test_find_prebuilt_falls_back_to_system_package(tmp_path):
    repo_dir = tmp_path / "hefs"
    repo_dir.mkdir()  # empty
    sys_dir = tmp_path / "system"
    sys_dir.mkdir()

    sys_file = sys_dir / "yolov8s_h8.hef"
    sys_file.write_bytes(b"SYS")

    found = find_prebuilt_hef(
        model_name="yolov8s.pt",
        yolo_version="v8",
        task=YOLOTask.DETECTION,
        arch="hailo8",
        repo_hef_dir=repo_dir,
        system_hef_dir=sys_dir,
    )
    assert found == sys_file


def test_find_prebuilt_returns_none_when_nothing_matches(tmp_path):
    repo_dir = tmp_path / "hefs"
    repo_dir.mkdir()
    sys_dir = tmp_path / "system"
    sys_dir.mkdir()

    found = find_prebuilt_hef(
        model_name="yolo11n.pt",  # not in system map
        yolo_version="v11",
        task=YOLOTask.DETECTION,
        arch="hailo8",
        repo_hef_dir=repo_dir,
        system_hef_dir=sys_dir,
    )
    assert found is None


def test_system_package_map_only_contains_observed_filenames():
    # Document the contract: every value in the map must reference a
    # file the rpicam-apps-hailo-postprocess Debian package actually
    # ships. Adding entries that don't exist on disk causes a silent
    # cache copy failure later. This test fails if someone invents a
    # filename — they need to verify it's actually in the package.
    OBSERVED_FILENAMES = {
        "yolov8s_h8.hef",
        "yolov8s_h8l.hef",
        "yolov8s_pose_h8.hef",
        "yolov8s_pose_h8l_pi.hef",
    }
    for value in SYSTEM_PACKAGE_MAP.values():
        assert value in OBSERVED_FILENAMES, (
            f"SYSTEM_PACKAGE_MAP references '{value}', which is not in "
            f"the observed-package set. Verify the file exists in "
            f"/usr/share/hailo-models/ before adding it."
        )
