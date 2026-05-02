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


# --------------------------------------------------- per-arch defaults


def test_default_zoo_versions_contains_known_arches():
    """Drift guard — DEFAULT_ZOO_VERSIONS must keep hailo8/hailo8l/hailo10h
    entries so unmodified `fetch_prebuilt_hefs.py --arch <X>` always
    resolves a sensible version."""
    assert {"hailo8", "hailo8l", "hailo10h"} <= set(fetch_mod.DEFAULT_ZOO_VERSIONS)


def test_resolve_zoo_versions_uses_per_arch_default_when_no_override():
    """Default resolution: hailo10h -> v2.18.0, hailo8 -> v2.16.0.
    The two arches need different defaults because the public S3 bucket
    has nothing under Compiled/v2.16.0/hailo10h/ — verified live
    2026-04-29 with the real fetcher returning 403 on every URL."""
    versions = fetch_mod.resolve_zoo_versions(("hailo8", "hailo10h"))
    assert versions["hailo8"] == "v2.16.0"
    assert versions["hailo10h"] == "v2.18.0"


def test_resolve_zoo_versions_override_applies_to_all_arches():
    """The --zoo-version CLI flag is intended as a single override for
    all arches in one run, so resolve_zoo_versions should mirror that."""
    versions = fetch_mod.resolve_zoo_versions(
        ("hailo8", "hailo10h"), override="v2.20.0"
    )
    assert versions == {"hailo8": "v2.20.0", "hailo10h": "v2.20.0"}


def test_resolve_zoo_versions_rejects_unknown_arch():
    with pytest.raises(ValueError, match="No DEFAULT_ZOO_VERSIONS entry"):
        fetch_mod.resolve_zoo_versions(("hailo-bogus",))


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


def _mk_head_ok():
    """Mock urlopen for HEAD requests that always succeed."""

    class _CtxResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _open(req_or_url, timeout=None):  # noqa: ARG001
        return _CtxResp()

    return _open


def test_dry_run_does_not_create_files(tmp_path):
    """Dry-run never writes the destination file even when HEAD says 200.

    Pinned to ``--source zoo`` so the test mocks (which respond to HEAD
    only) don't have to handle the release manifest GET that ``--source
    both`` (the default) would issue first.
    """
    with patch.object(fetch_mod.urllib.request, "urlopen", _mk_head_ok()):
        rc = fetch_mod.main(
            [
                "--arch", "hailo10h", "--source", "zoo",
                "--dry-run", "--output-dir", str(tmp_path),
            ]
        )
    assert rc == 0
    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []


def test_dry_run_uses_HEAD_not_GET(tmp_path):
    """Dry-run must HEAD-probe the URL — using GET would defeat the point
    (large transfer for a planning operation). Verifies we pass a
    urllib.request.Request with method='HEAD' rather than a bare URL."""
    captured_methods: list[str] = []

    class _CtxResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _spy(req_or_url, timeout=None):  # noqa: ARG001
        if isinstance(req_or_url, fetch_mod.urllib.request.Request):
            captured_methods.append(req_or_url.get_method())
        else:
            captured_methods.append("GET")
        return _CtxResp()

    with patch.object(fetch_mod.urllib.request, "urlopen", _spy):
        fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_versions={"hailo10h": "v2.18.0", "hailo8": "v2.16.0"},
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
            dry_run=True,
        )
    assert captured_methods == ["HEAD"]


def test_dry_run_reports_actual_status_when_url_missing(tmp_path):
    """The whole reason this fix exists. Before the change, dry-run
    reported 'downloaded' for every URL regardless of whether S3 had
    the object. Now it reflects the real catalogue status — verified
    2026-04-29 against hailo10h@v2.18.0 which 403s on every URL."""
    import urllib.error

    def head_403(req_or_url, timeout=None):  # noqa: ARG001
        url = (
            req_or_url
            if isinstance(req_or_url, str)
            else req_or_url.full_url
        )
        raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", head_403):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_versions={"hailo10h": "v2.18.0", "hailo8": "v2.16.0"},
            output_dir=tmp_path,
            manifest=(HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),),
            dry_run=True,
        )
    assert results[0].status == "missing-403"


def test_dry_run_marks_results_as_downloaded_when_HEAD_ok(tmp_path):
    """When the URL resolves, dry-run reports 'downloaded' (i.e. would
    download). Mock HEAD to always 200 so the test doesn't need network."""
    with patch.object(fetch_mod.urllib.request, "urlopen", _mk_head_ok()):
        results = fetch_mod.fetch_all(
            ("hailo10h",),
            zoo_versions={"hailo10h": "v2.16.0", "hailo8": "v2.16.0"},
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
            zoo_versions={"hailo10h": "v2.16.0", "hailo8": "v2.16.0"},
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
            zoo_versions={"hailo10h": "v2.16.0", "hailo8": "v2.16.0"},
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
            zoo_versions={"hailo10h": "v2.16.0", "hailo8": "v2.16.0"},
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
            zoo_versions={"hailo10h": "v2.16.0", "hailo8": "v2.16.0"},
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
            zoo_versions={"hailo10h": "v2.16.0", "hailo8": "v2.16.0"},
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
            zoo_versions={"hailo10h": "v2.16.0", "hailo8": "v2.16.0"},
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


# =============================================================================
# GitHub Release source — release_url, manifest loading, fetch_from_release,
# fetch_release_for_arches, fetch_with_release_fallback, --source flag.
# =============================================================================

import hashlib
import json
import urllib.error


# ----------------------- URL construction


def test_release_url_uses_canonical_filename_under_release_tag():
    url = fetch_mod.release_url("v8_segmentation_n_hailo10h.hef")
    assert url == (
        "https://github.com/JabbaghYounes/Benchy/releases/download/"
        f"{fetch_mod.HEFS_RELEASE_TAG}/v8_segmentation_n_hailo10h.hef"
    )


def test_release_url_threads_explicit_tag():
    url = fetch_mod.release_url("v26_obb_n_hailo10h.hef", tag="hefs-v2")
    assert "/hefs-v2/v26_obb_n_hailo10h.hef" in url


def test_release_manifest_url_points_at_manifest_json():
    url = fetch_mod.release_manifest_url()
    assert url.endswith(f"/{fetch_mod.HEFS_RELEASE_TAG}/manifest.json")


# ----------------------- manifest loading


def _mk_release_manifest_payload(entries: list[dict]) -> bytes:
    """Build a release manifest.json payload mirroring the publishing
    script's schema (release_tag, hef_count, total_size_bytes, hefs)."""
    return json.dumps(
        {
            "release_tag": fetch_mod.HEFS_RELEASE_TAG,
            "hef_count": len(entries),
            "total_size_bytes": sum(e["size_bytes"] for e in entries),
            "hefs": entries,
        }
    ).encode()


class _CtxBytes:
    """Minimal context-manager wrapping a bytes payload — supports both
    .read(n) chunked reads (used by fetch_from_release) and .read() with
    no args (used by load_release_manifest via json.loads(resp.read()))."""

    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n) if n > 0 else self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_load_release_manifest_parses_canonical_filename_index(tmp_path):
    """load_release_manifest returns a dict keyed by canonical filename
    so callers can do O(1) lookup by the name they want."""
    payload = _mk_release_manifest_payload(
        [
            {
                "filename": "v8_obb_n_hailo10h.hef",
                "sha256": "deadbeef",
                "size_bytes": 100,
                "yolo_version": "v8",
                "task": "obb",
                "size": "n",
                "arch": "hailo10h",
                "source": "workstation-rtx2080ti-2026-04-30",
            }
        ]
    )

    def _open(url, timeout=None):  # noqa: ARG001
        return _CtxBytes(payload)

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        manifest = fetch_mod.load_release_manifest()

    assert "v8_obb_n_hailo10h.hef" in manifest
    assert manifest["v8_obb_n_hailo10h.hef"]["sha256"] == "deadbeef"
    assert manifest["v8_obb_n_hailo10h.hef"]["arch"] == "hailo10h"


def test_load_release_manifest_propagates_404(tmp_path):
    """A missing manifest is the caller's signal to either abort
    (--source release) or degrade to zoo-only (--source both default).
    Function itself just propagates the HTTPError."""

    def _raise_404(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", _raise_404):
        with pytest.raises(urllib.error.HTTPError):
            fetch_mod.load_release_manifest()


# ----------------------- fetch_from_release


def test_fetch_from_release_writes_canonical_file_and_passes_sha(tmp_path):
    payload = b"FAKE-HEF-h10h-segmentation-n"
    sha = hashlib.sha256(payload).hexdigest()

    def _open(url, timeout=None):  # noqa: ARG001
        return _CtxBytes(payload)

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        result = fetch_mod.fetch_from_release(
            "v8_segmentation_n_hailo10h.hef",
            "hailo10h",
            tmp_path,
            sha,
        )

    assert result.status == "downloaded"
    assert result.source == "release"
    assert result.dest == tmp_path / "v8_segmentation_n_hailo10h.hef"
    assert result.dest.read_bytes() == payload


def test_fetch_from_release_sha_mismatch_deletes_file_and_errors(tmp_path):
    """A mismatched SHA-256 means corruption / wrong asset / drift; the
    fetcher deletes the partial file (so a retry doesn't see a stale
    file and skip) and surfaces a clear 'sha-mismatch' status."""
    payload = b"actual bytes from server"

    def _open(url, timeout=None):  # noqa: ARG001
        return _CtxBytes(payload)

    expected_but_wrong = "0" * 64
    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        result = fetch_mod.fetch_from_release(
            "v8_obb_n_hailo10h.hef",
            "hailo10h",
            tmp_path,
            expected_but_wrong,
        )

    assert result.status == "sha-mismatch"
    assert result.source == "release"
    assert result.error is not None and "sha256" in result.error
    assert not result.dest.exists()  # deleted, no stale leftover


def test_fetch_from_release_404_returns_missing_not_error(tmp_path):
    """A 404 on a release asset is "not in this release" (exactly the
    same semantic as Zoo 404), not a hard failure. main() exits 0 on
    missing-* statuses."""

    def _raise_404(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", _raise_404):
        result = fetch_mod.fetch_from_release(
            "v99_nope_n_hailo10h.hef",
            "hailo10h",
            tmp_path,
            "deadbeef",
        )

    assert result.status == "missing-404"
    assert result.source == "release"
    assert not result.dest.exists()


def test_fetch_from_release_existing_file_with_matching_sha_is_skipped(tmp_path):
    payload = b"already-there-and-valid"
    sha = hashlib.sha256(payload).hexdigest()
    existing = tmp_path / "v8_detection_n_hailo10h.hef"
    existing.write_bytes(payload)

    from unittest.mock import MagicMock
    spy = MagicMock()
    with patch.object(fetch_mod.urllib.request, "urlopen", spy):
        result = fetch_mod.fetch_from_release(
            "v8_detection_n_hailo10h.hef",
            "hailo10h",
            tmp_path,
            sha,
        )

    assert result.status == "skipped-exists"
    assert result.source == "release"
    spy.assert_not_called()  # no network call when local sha matches


def test_fetch_from_release_existing_file_with_wrong_sha_reports_mismatch(tmp_path):
    """When the local file exists but doesn't match the manifest sha,
    the fetcher reports sha-mismatch (so the user can decide to
    --overwrite) but does NOT auto-delete the local file — it might be
    an intentional manual stage."""
    existing = tmp_path / "v8_detection_n_hailo10h.hef"
    existing.write_bytes(b"local custom build")

    from unittest.mock import MagicMock
    spy = MagicMock()
    with patch.object(fetch_mod.urllib.request, "urlopen", spy):
        result = fetch_mod.fetch_from_release(
            "v8_detection_n_hailo10h.hef",
            "hailo10h",
            tmp_path,
            "deadbeef" * 8,
        )

    assert result.status == "sha-mismatch"
    assert existing.exists()  # don't auto-delete user-staged files
    spy.assert_not_called()


# ----------------------- fetch_release_for_arches


def test_fetch_release_for_arches_filters_by_arch(tmp_path):
    """Only fetch HEFs whose manifest entry matches a requested arch."""
    payload_h10h = b"h10h-payload"
    payload_h8 = b"h8-payload"
    sha_h10h = hashlib.sha256(payload_h10h).hexdigest()
    sha_h8 = hashlib.sha256(payload_h8).hexdigest()

    manifest = {
        "v8_detection_n_hailo10h.hef": {
            "filename": "v8_detection_n_hailo10h.hef",
            "sha256": sha_h10h,
            "size_bytes": len(payload_h10h),
            "arch": "hailo10h",
        },
        "v8_detection_n_hailo8.hef": {
            "filename": "v8_detection_n_hailo8.hef",
            "sha256": sha_h8,
            "size_bytes": len(payload_h8),
            "arch": "hailo8",
        },
    }

    def _open(url, timeout=None):  # noqa: ARG001
        # Route by canonical filename in the URL
        if "hailo10h" in url:
            return _CtxBytes(payload_h10h)
        return _CtxBytes(payload_h8)

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        results = fetch_mod.fetch_release_for_arches(
            ("hailo10h",),
            output_dir=tmp_path,
            release_manifest=manifest,
        )

    assert len(results) == 1
    assert results[0].dest.name == "v8_detection_n_hailo10h.hef"
    assert results[0].status == "downloaded"


# ----------------------- fetch_with_release_fallback


def test_fallback_uses_zoo_when_zoo_has_the_hef(tmp_path):
    """If the canonical filename has a Zoo equivalent and the Zoo URL
    succeeds, the release URL is never touched — keeps the Zoo as the
    primary source for HEFs both have."""
    zoo_payload = b"FROM-ZOO"
    captured_urls: list[str] = []

    def _open(url_or_req, timeout=None):  # noqa: ARG001
        url = url_or_req if isinstance(url_or_req, str) else url_or_req.full_url
        captured_urls.append(url)
        return _CtxBytes(zoo_payload)

    manifest = {
        "v8_detection_n_hailo8.hef": {
            "filename": "v8_detection_n_hailo8.hef",
            "sha256": hashlib.sha256(zoo_payload).hexdigest(),
            "arch": "hailo8",
        },
    }

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        results = fetch_mod.fetch_with_release_fallback(
            ("hailo8",),
            zoo_versions={"hailo8": "v2.16.0"},
            output_dir=tmp_path,
            release_manifest=manifest,
        )

    assert len(results) == 1
    assert results[0].source == "zoo"
    assert all("releases/download" not in u for u in captured_urls)


def test_fallback_falls_through_to_release_on_zoo_404(tmp_path):
    """Zoo 404 → release attempted. Release returns the bytes; sha
    matches; result tagged source='release'."""
    release_payload = b"FROM-RELEASE"
    sha = hashlib.sha256(release_payload).hexdigest()

    def _open(url_or_req, timeout=None):  # noqa: ARG001
        url = url_or_req if isinstance(url_or_req, str) else url_or_req.full_url
        if "hailo-model-zoo" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        return _CtxBytes(release_payload)

    manifest = {
        "v8_detection_n_hailo8.hef": {
            "filename": "v8_detection_n_hailo8.hef",
            "sha256": sha,
            "arch": "hailo8",
        },
    }

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        results = fetch_mod.fetch_with_release_fallback(
            ("hailo8",),
            zoo_versions={"hailo8": "v2.16.0"},
            output_dir=tmp_path,
            release_manifest=manifest,
        )

    assert len(results) == 1
    assert results[0].source == "release"
    assert results[0].status == "downloaded"


def test_fallback_skips_zoo_for_hefs_zoo_doesnt_ship(tmp_path):
    """OBB HEFs aren't in ZOO_MANIFEST, so the fallback should go
    directly to release without making a Zoo request first."""
    release_payload = b"FROM-RELEASE-OBB"
    sha = hashlib.sha256(release_payload).hexdigest()
    captured_urls: list[str] = []

    def _open(url_or_req, timeout=None):  # noqa: ARG001
        url = url_or_req if isinstance(url_or_req, str) else url_or_req.full_url
        captured_urls.append(url)
        if "hailo-model-zoo" in url:
            raise AssertionError(
                f"Should not have hit Zoo for an OBB HEF; got {url}"
            )
        return _CtxBytes(release_payload)

    manifest = {
        "v11_obb_n_hailo10h.hef": {
            "filename": "v11_obb_n_hailo10h.hef",
            "sha256": sha,
            "arch": "hailo10h",
        },
    }

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        results = fetch_mod.fetch_with_release_fallback(
            ("hailo10h",),
            zoo_versions={"hailo10h": "v2.18.0"},
            output_dir=tmp_path,
            release_manifest=manifest,
        )

    assert len(results) == 1
    assert results[0].source == "release"


def test_fallback_surfaces_zoo_5xx_without_silent_release_fallback(tmp_path):
    """A 503 from the Zoo is a real problem, not 'not in catalogue'.
    Fallback should surface the Zoo error rather than silently retrying
    against the release — paper-over would mask outages."""
    captured_urls: list[str] = []

    def _open(url_or_req, timeout=None):  # noqa: ARG001
        url = url_or_req if isinstance(url_or_req, str) else url_or_req.full_url
        captured_urls.append(url)
        if "hailo-model-zoo" in url:
            raise urllib.error.HTTPError(
                url, 503, "Service Unavailable", hdrs=None, fp=None
            )
        raise AssertionError(f"Should not have hit release after Zoo 5xx; got {url}")

    manifest = {
        "v8_detection_n_hailo8.hef": {
            "filename": "v8_detection_n_hailo8.hef",
            "sha256": "deadbeef" * 8,
            "arch": "hailo8",
        },
    }

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        results = fetch_mod.fetch_with_release_fallback(
            ("hailo8",),
            zoo_versions={"hailo8": "v2.16.0"},
            output_dir=tmp_path,
            release_manifest=manifest,
        )

    assert len(results) == 1
    assert results[0].source == "zoo"
    assert results[0].status == "error"
    assert "503" in (results[0].error or "")


# ----------------------- main() routing for --source


def test_main_source_release_requires_reachable_manifest(tmp_path):
    """--source release with an unreachable manifest is a hard error
    (not a silent fall-through to zoo) — that mode is the user being
    explicit about wanting verified release fetches."""

    def _raise_404(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", _raise_404):
        rc = fetch_mod.main(
            [
                "--arch", "hailo10h", "--source", "release",
                "--output-dir", str(tmp_path),
            ]
        )

    assert rc == 1


def test_main_source_both_degrades_to_zoo_when_manifest_unreachable(tmp_path):
    """The default mode silently falls back to zoo when the release
    manifest isn't reachable — keeps offline / air-gapped workflows
    working without surprises."""

    def _raise_503(url, timeout=None):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url, 503, "Service Unavailable", hdrs=None, fp=None
        )

    with patch.object(fetch_mod.urllib.request, "urlopen", _raise_503):
        rc = fetch_mod.main(
            [
                "--arch", "hailo10h", "--source", "both",
                "--output-dir", str(tmp_path),
            ]
        )

    # Manifest 503 → fall back to zoo → all zoo URLs also 503 → all
    # results are status='error' → main returns 1. The point of this
    # test is that we don't crash with an uncaught exception during
    # the manifest fetch, not that we exit 0 in an outage.
    assert rc == 1  # zoo-side errors still fail the run


def test_main_source_release_sha_mismatch_exits_nonzero(tmp_path):
    """SHA mismatch must fail the run (a corrupt or wrong asset is a
    silent footgun otherwise)."""
    payload = b"corrupt-or-wrong-asset"
    bad_sha = "f" * 64
    manifest_payload = _mk_release_manifest_payload(
        [
            {
                "filename": "v8_obb_n_hailo10h.hef",
                "sha256": bad_sha,
                "size_bytes": len(payload),
                "yolo_version": "v8",
                "task": "obb",
                "size": "n",
                "arch": "hailo10h",
                "source": "workstation-rtx2080ti-2026-04-30",
            }
        ]
    )

    def _open(url_or_req, timeout=None):  # noqa: ARG001
        url = url_or_req if isinstance(url_or_req, str) else url_or_req.full_url
        if url.endswith("manifest.json"):
            return _CtxBytes(manifest_payload)
        return _CtxBytes(payload)

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        rc = fetch_mod.main(
            [
                "--arch", "hailo10h", "--source", "release",
                "--output-dir", str(tmp_path),
            ]
        )

    assert rc == 1
    assert not (tmp_path / "v8_obb_n_hailo10h.hef").exists()


def test_main_release_tag_override_threaded_through(tmp_path):
    """--release-tag should change the manifest URL the fetcher hits
    (so users can pin to a specific batch when reproducing old runs)."""
    captured_urls: list[str] = []

    def _open(url_or_req, timeout=None):  # noqa: ARG001
        url = url_or_req if isinstance(url_or_req, str) else url_or_req.full_url
        captured_urls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with patch.object(fetch_mod.urllib.request, "urlopen", _open):
        fetch_mod.main(
            [
                "--arch", "hailo10h", "--source", "release",
                "--release-tag", "hefs-v99",
                "--output-dir", str(tmp_path),
            ]
        )

    manifest_urls = [u for u in captured_urls if u.endswith("manifest.json")]
    assert manifest_urls, "expected at least one manifest URL"
    assert all("/hefs-v99/" in u for u in manifest_urls)
