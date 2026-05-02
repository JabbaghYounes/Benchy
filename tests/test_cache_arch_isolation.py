# Regression tests for the cache-key arch-isolation fix (Issue 9 in
# resources/session_notes_2026-04-29_nvidia_workstation.md).
#
# Pre-fix: ModelCache.get_*_path() keyed on (model_name, version, task)
# only. Compiling the same model for two different architectures
# (hailo8 then hailo10h, say) reused the same cache directory. The
# second compile would either short-circuit on the wrong-arch cached
# HEF or overwrite the first arch's HEF in place. Either way: the
# staged file in resources/hefs/ ended up md5-identical across
# architectures — bit twice during the 2026-04-29 / 2026-04-30
# sweeps, masquerading as ~10-second "PASS in cache hit" runs.
#
# Post-fix: target_device is a required keyword-only arg on every
# cache-resolving method, and the on-disk path is
# models/hailo/<arch>/<version>/<task>/<model>/. Same model under two
# different archs now lands in two different directories with two
# distinct sets of artifacts.
from __future__ import annotations

from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock

import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion.cache import (
    ModelCache,
    get_cache_path,
)


# ---------------------------------------------------------------- low-level


def test_get_cache_path_includes_arch_in_directory_layout(tmp_path):
    """The on-disk path encodes <arch> as the first directory level
    under the cache root. This is the structural invariant the bug
    violated."""
    h8 = get_cache_path(
        "yolov8n.pt", "v8", YOLOTask.DETECTION,
        target_device="hailo8",
        cache_dir=tmp_path,
    )
    h10h = get_cache_path(
        "yolov8n.pt", "v8", YOLOTask.DETECTION,
        target_device="hailo10h",
        cache_dir=tmp_path,
    )

    assert h8 != h10h, "different archs must yield different cache paths"
    assert h8.parts[-4] == "hailo8", f"arch should be 4 levels up; got {h8}"
    assert h10h.parts[-4] == "hailo10h", f"arch should be 4 levels up; got {h10h}"
    # The remaining structure is shared: <version>/<task>/<model>
    assert h8.parts[-3:] == h10h.parts[-3:]


def test_get_cache_path_requires_target_device_keyword():
    """target_device is keyword-only — positional callers must fail
    loudly so the bug can't sneak back in via a forgotten arg."""
    with pytest.raises(TypeError):
        # Positional 4th arg (would have been valid pre-fix when
        # cache_dir was the 4th positional) — the keyword-only marker
        # makes this a TypeError now.
        get_cache_path("yolov8n.pt", "v8", YOLOTask.DETECTION, "hailo8")


def test_modelcache_methods_require_target_device(tmp_path):
    """The four path-resolving methods on ModelCache also force
    target_device to be passed by keyword."""
    cache = ModelCache(cache_dir=tmp_path)
    args = ("yolov8n.pt", "v8", YOLOTask.DETECTION)

    for method_name in ("get_onnx_path", "get_har_path", "get_hef_path",
                        "get_metadata_path", "get_model_cache_path"):
        method = getattr(cache, method_name)
        with pytest.raises(TypeError):
            method(*args)  # missing target_device — must raise


# ---------------------------------------------------------------- artifact isolation


def _write_arch_distinguishable_hef(
    cache: ModelCache,
    target_device: str,
) -> Tuple[Path, bytes]:
    """Pretend to compile yolov8n-obb for the given arch by writing a
    HEF whose bytes encode the arch tag. Uses the same cache API the
    real pipeline uses, so the test exercises the production path."""
    hef_path = cache.get_hef_path(
        "yolov8n-obb.pt", "v8", YOLOTask.OBB,
        target_device=target_device,
    )
    hef_path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"FAKE-HEF-FOR-{target_device}".encode("ascii")
    hef_path.write_bytes(payload)
    return hef_path, payload


def test_two_archs_same_model_produce_distinct_hefs(tmp_path):
    """End-to-end: compile (mocked) the same model for two archs.
    The two HEFs must land at different paths AND carry different
    bytes. md5-identical-across-archs is the exact symptom Issue 9
    surfaced."""
    cache = ModelCache(cache_dir=tmp_path)

    h8_path, h8_bytes = _write_arch_distinguishable_hef(cache, "hailo8")
    h10h_path, h10h_bytes = _write_arch_distinguishable_hef(cache, "hailo10h")

    assert h8_path != h10h_path
    assert h8_path.exists() and h10h_path.exists()
    # Both still on disk after the second "compile" — neither overwrote
    # the other.
    assert h8_path.read_bytes() == h8_bytes
    assert h10h_path.read_bytes() == h10h_bytes
    # The actual regression: bytes must differ across archs.
    assert h8_bytes != h10h_bytes
    # md5 difference is the smoke test the session notes called out.
    import hashlib
    assert hashlib.md5(h8_bytes).hexdigest() != hashlib.md5(h10h_bytes).hexdigest()


def test_has_hef_does_not_cross_pollute_archs(tmp_path):
    """has_hef(arch=X) must not return True just because there's a
    cached HEF for arch=Y. Pre-fix it would have, since the path
    didn't include arch."""
    cache = ModelCache(cache_dir=tmp_path)

    # Stage a hailo8 HEF only.
    _write_arch_distinguishable_hef(cache, "hailo8")

    args = ("yolov8n-obb.pt", "v8", YOLOTask.OBB)
    assert cache.has_hef(*args, target_device="hailo8") is True
    assert cache.has_hef(*args, target_device="hailo10h") is False
    # Same for the other artifact-presence probes.
    # (ONNX/HAR aren't staged in this test, so both archs are False —
    # the asymmetry on hef is what matters.)


def test_clear_cache_only_touches_one_arch(tmp_path):
    """Clearing one arch's cache leaves the other arch untouched.
    Belt-and-braces — the bug fix's semantics imply this, but worth
    exercising explicitly."""
    cache = ModelCache(cache_dir=tmp_path)
    _write_arch_distinguishable_hef(cache, "hailo8")
    _write_arch_distinguishable_hef(cache, "hailo10h")

    cache.clear_cache(
        "yolov8n-obb.pt", "v8", YOLOTask.OBB,
        target_device="hailo8",
    )

    args = ("yolov8n-obb.pt", "v8", YOLOTask.OBB)
    assert cache.has_hef(*args, target_device="hailo8") is False
    assert cache.has_hef(*args, target_device="hailo10h") is True


# ---------------------------------------------------------------- listing


def test_list_cached_models_walks_arch_level(tmp_path):
    """list_cached_models must walk the new <arch>/<version>/<task>/<model>
    structure correctly (not the pre-fix <version>/<task>/<model>),
    and must report target_device per row so a multi-arch sweep is
    legible."""
    cache = ModelCache(cache_dir=tmp_path)
    _write_arch_distinguishable_hef(cache, "hailo8")
    _write_arch_distinguishable_hef(cache, "hailo10h")

    listing = cache.list_cached_models()

    assert len(listing) == 2
    archs_seen = sorted(row["target_device"] for row in listing)
    assert archs_seen == ["hailo10h", "hailo8"]
    for row in listing:
        assert row["model_name"] == "yolov8n-obb"
        assert row["yolo_version"] == "v8"
        assert row["task"] == "obb"
        assert row["has_hef"] is True
