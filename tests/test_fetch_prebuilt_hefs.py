# Tests for scripts/fetch_prebuilt_hefs.py.
#
# Covers URL construction, canonical-filename mapping, dry-run, and
# defensive 404 handling. We mock urllib.request.urlopen so the tests
# never hit the real Hailo S3 bucket.
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# fetch_prebuilt_hefs lives under scripts/ which isn't a package, so
# load it by file path the same way pytest collects ad-hoc scripts.
# Register in sys.modules first so dataclass introspection (which looks
# up cls.__module__ in sys.modules at class-creation time) works on
# Python 3.12+.
_FETCH_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "fetch_prebuilt_hefs.py"
)
_spec = importlib.util.spec_from_file_location("fetch_prebuilt_hefs", _FETCH_PATH)
fetch_mod = importlib.util.module_from_spec(_spec)
sys.modules["fetch_prebuilt_hefs"] = fetch_mod
_spec.loader.exec_module(fetch_mod)


HEFManifestEntry = fetch_mod.HEFManifestEntry


# ---------------------------------------------------------------- URLs


def test_zoo_url_matches_documented_pattern():
    entry = HEFManifestEntry("v8", "detection", "n", "yolov8n.hef")
    url = fetch_mod.zoo_url(entry, "hailo8", "v2.16.0")
    assert url == (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
        "ModelZoo/Compiled/v2.16.0/hailo8/yolov8n.hef"
    )


def test_zoo_url_threads_arch_and_version():
    entry = HEFManifestEntry("v11", "detection", "n", "yolov11n.hef")
    url = fetch_mod.zoo_url(entry, "hailo10h", "v2.18.0")
    assert "/v2.18.0/hailo10h/" in url
    assert url.endswith("yolov11n.hef")


# -------------------------------------------------------- canonical name


@pytest.mark.parametrize(
    "entry, arch, expected",
    [
        (
            HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),
            "hailo8",
            "v8_detection_n_hailo8.hef",
        ),
        (
            HEFManifestEntry("v8", "segmentation", "n", "yolov8n_seg.hef"),
            "hailo10h",
            "v8_segmentation_n_hailo10h.hef",
        ),
        (
            HEFManifestEntry("v8", "pose", "s", "yolov8s_pose.hef"),
            "hailo8l",
            "v8_pose_s_hailo8l.hef",
        ),
    ],
)
def test_canonical_filename_matches_repo_naming_convention(entry, arch, expected):
    assert entry.canonical_filename(arch) == expected


# Cross-check that the manifest's canonical names exactly match the
# Pi-side repo_filename() — drift here would mean the runtime can't
# find a fetched HEF.
def test_manifest_canonical_names_round_trip_with_hef_source():
    from benchmark.schemas import YOLOTask
    from benchmark.workloads.yolo.conversion.hef_source import repo_filename

    task_lookup = {
        "detection": YOLOTask.DETECTION,
        "segmentation": YOLOTask.SEGMENTATION,
        "pose": YOLOTask.POSE,
        "obb": YOLOTask.OBB,
        "classification": YOLOTask.CLASSIFICATION,
    }

    for entry in fetch_mod.ZOO_MANIFEST:
        for arch in ("hailo8", "hailo10h"):
            assert entry.canonical_filename(arch) == repo_filename(
                entry.yolo_version,
                task_lookup[entry.task],
                entry.size,
                arch,
            )


# ------------------------------------------------------------ dry-run


def test_dry_run_does_not_create_files(tmp_path):
    rc = fetch_mod.main(
        ["--arch", "hailo10h", "--dry-run", "--output-dir", str(tmp_path)]
    )
    assert rc == 0
    # Output dir should be created but stay empty
    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []


def test_dry_run_marks_results_as_downloaded(tmp_path):
    results = fetch_mod.fetch_all(
        ("hailo10h",),
        zoo_version="v2.16.0",
        output_dir=tmp_path,
        dry_run=True,
    )
    assert results
    assert all(r.status == "downloaded" for r in results)
    assert list(tmp_path.iterdir()) == []  # never wrote anything


# -------------------------------------------------- defensive 404 handling


class _Fake404(Exception):
    """Stand-in raised by our urlopen monkeypatch."""


def _mk_urlopen_404():
    import urllib.error

    def raise_404(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    return raise_404


def _mk_urlopen_ok(payload: bytes):
    class _CtxResp:
        def __init__(self, body: bytes):
            self._buf = io.BytesIO(body)

        def read(self, n: int) -> bytes:
            return self._buf.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _open(url, timeout=None):  # noqa: ARG001
        return _CtxResp(payload)

    return _open


def test_404_results_in_missing_status_not_exception(tmp_path):
    with patch.object(fetch_mod.urllib.request, "urlopen", _mk_urlopen_404()):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_version="v2.16.0",
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
        )
    assert len(results) == 1
    assert results[0].status == "missing-404"
    assert not results[0].dest.exists()


def test_403_treated_as_missing_not_error(tmp_path):
    """S3 returns 403 instead of 404 for missing objects in buckets that
    forbid ListObjects. We treat 403 the same as 404 — non-fatal, just
    'not in the catalogue at this URL'."""
    import urllib.error

    def raise_403(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", raise_403):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_version="v2.16.0",
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
        )
    assert results[0].status == "missing-403"


def test_404_does_not_set_main_exit_code(tmp_path):
    with patch.object(fetch_mod.urllib.request, "urlopen", _mk_urlopen_404()):
        rc = fetch_mod.main(
            ["--arch", "hailo10h", "--output-dir", str(tmp_path)]
        )
    assert rc == 0  # missing != error


def test_403_does_not_set_main_exit_code(tmp_path):
    """Hailo's public S3 bucket returns 403 for the
    Compiled/v2.16.0/hailo10h/ path because no objects are published
    there. The fetcher must exit 0 in that case so it can be chained
    with --arch both and ignore archs the Zoo doesn't cover."""
    import urllib.error

    def raise_403(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", raise_403):
        rc = fetch_mod.main(
            ["--arch", "hailo10h", "--output-dir", str(tmp_path)]
        )
    assert rc == 0


def test_5xx_remains_a_real_error(tmp_path):
    """Server errors (502/503) shouldn't be silently swallowed."""
    import urllib.error

    def raise_503(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url, 503, "Service Unavailable", hdrs=None, fp=None
        )

    with patch.object(fetch_mod.urllib.request, "urlopen", raise_503):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_version="v2.16.0",
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
        )
    assert results[0].status == "error"
    assert "503" in (results[0].error or "")


def test_successful_download_writes_canonical_file(tmp_path):
    payload = b"FAKE-HEF-PAYLOAD"
    with patch.object(fetch_mod.urllib.request, "urlopen", _mk_urlopen_ok(payload)):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_version="v2.16.0",
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
        )
    assert len(results) == 1
    r = results[0]
    assert r.status == "downloaded"
    assert r.dest == tmp_path / "v8_detection_n_hailo10h.hef"
    assert r.dest.read_bytes() == payload


def test_existing_file_skipped_unless_overwrite(tmp_path):
    from unittest.mock import MagicMock

    existing = tmp_path / "v8_detection_n_hailo10h.hef"
    existing.write_bytes(b"already here")

    spy = MagicMock(side_effect=_mk_urlopen_ok(b"NEW"))
    with patch.object(fetch_mod.urllib.request, "urlopen", spy):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_version="v2.16.0",
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
        )
    assert results[0].status == "skipped-exists"
    assert existing.read_bytes() == b"already here"
    spy.assert_not_called()


def test_overwrite_redownloads(tmp_path):
    existing = tmp_path / "v8_detection_n_hailo10h.hef"
    existing.write_bytes(b"old")

    with patch.object(fetch_mod.urllib.request, "urlopen", _mk_urlopen_ok(b"new")):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_version="v2.16.0",
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
            overwrite=True,
        )
    assert results[0].status == "downloaded"
    assert existing.read_bytes() == b"new"


# --------------------------------------------------- arch arg expansion


def test_arch_both_expands_to_hailo8_and_hailo10h(tmp_path):
    captured_arches = []

    def _spy(url, timeout=None):  # noqa: ARG001
        # url contains "/<arch>/"
        for arch in ("hailo8", "hailo8l", "hailo10h"):
            if f"/{arch}/" in url:
                captured_arches.append(arch)
                break
        import urllib.error

        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", _spy):
        rc = fetch_mod.main(
            ["--arch", "both", "--output-dir", str(tmp_path)]
        )
    assert rc == 0
    assert "hailo8" in captured_arches
    assert "hailo10h" in captured_arches
    assert "hailo8l" not in captured_arches  # 'both' is hailo8 + hailo10h
