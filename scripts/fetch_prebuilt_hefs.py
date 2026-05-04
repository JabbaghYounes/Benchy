#!/usr/bin/env python3
"""Fetch prebuilt HEFs from the Hailo Model Zoo and/or the Benchy GitHub Release.

Two sources, picked via ``--source``:

1. **Hailo Model Zoo S3** (``--source zoo``). The original path. Downloads
   from ``hailo-model-zoo.s3.eu-west-2.amazonaws.com`` per the URL pattern::

       https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<zoo_version>/<arch>/<filename>.hef

   The catalogue is incomplete: the Zoo does not publish OBB HEFs at all,
   does not publish pose at the ``n`` size, and as of 2026-04 has no v11
   segmentation/pose/OBB or any v26 prebuilts for Hailo-8. 403/404 are
   expected and treated as "not in the catalogue".

2. **Benchy GitHub Release** (``--source release``). Pulls every HEF
   listed in the release's ``manifest.json`` for the requested arches and
   verifies SHA-256 against the manifest. The release is the canonical
   home for Benchy's workstation-compiled gap-fillers (every Hailo-10H
   HEF, every OBB HEF, every v26 HEF, etc.). URL pattern::

       https://github.com/JabbaghYounes/Benchy/releases/download/<HEFS_RELEASE_TAG>/<canonical_filename>

3. **Both** (``--source both``, default). Iterates the release manifest
   for the requested arches; for each HEF that the Zoo also publishes,
   tries the Zoo first and falls back to the release on 403/404; for
   gap-fillers the Zoo doesn't ship, goes directly to the release. If
   the release manifest is unreachable (offline, GitHub rate-limited),
   degrades silently to ``--source zoo`` with a warning.

Pi-side runtime (``benchmark/workloads/yolo/conversion/hef_source.py``)
picks up staged HEFs from ``resources/hefs/`` automatically. Use
``scripts/compile_workstation_hefs.sh`` to produce HEFs that aren't yet
in either source.

Usage::

    scripts/fetch_prebuilt_hefs.py --arch hailo8
    scripts/fetch_prebuilt_hefs.py --arch hailo10h --dry-run
    scripts/fetch_prebuilt_hefs.py --arch both --source release
    scripts/fetch_prebuilt_hefs.py --arch hailo10h --release-tag hefs-v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import urllib.error
import urllib.request

# Repo-relative output directory; the Pi-side hef_source.py expects
# this exact location.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "resources" / "hefs"

ZOO_BASE_URL = (
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
)

# Per-arch default zoo version. The hailo10h path was not populated
# in v2.16.0 (every URL returned 403 — verified 2026-04-29 against the
# real S3 bucket); v2.18.0 is the earliest version where Hailo started
# publishing hailo10h prebuilts on the public bucket. The hailo8/8l
# defaults stay on v2.16.0 to match the four HEFs originally staged in
# resources/hefs/. Override with --zoo-version on the CLI to force a
# single version across all arches.
DEFAULT_ZOO_VERSIONS: dict[str, str] = {
    "hailo8": "v2.16.0",
    "hailo8l": "v2.16.0",
    "hailo10h": "v2.18.0",
}

ARCHES = tuple(DEFAULT_ZOO_VERSIONS.keys())

# GitHub Release source. Bumped deliberately when a new HEF batch is
# published — the fetcher pins to a specific tag so older Pi setups stay
# reproducible. The release tip is mirrored in
# /tmp/benchy-hefs-v1/release_notes.md (during the publish flow) and on
# the GitHub release page.
HEFS_RELEASE_TAG = "hefs-v2"

GITHUB_RELEASE_BASE_URL = (
    "https://github.com/JabbaghYounes/Benchy/releases/download"
)


@dataclass(frozen=True)
class HEFManifestEntry:
    """One row of the prebuilt HEF manifest.

    Resolves to a Zoo URL via :func:`zoo_url` and to a canonical Benchy
    filename via :meth:`canonical_filename`.
    """

    yolo_version: str  # "v8" / "v11" / "v26"
    task: str          # "detection" / "segmentation" / "pose" / ...
    size: str          # "n" / "s" / "m" / "l" / "x"
    zoo_filename: str  # e.g. "yolov8n.hef"

    def canonical_filename(self, arch: str) -> str:
        return f"{self.yolo_version}_{self.task}_{self.size}_{arch}.hef"


# What the public Hailo Model Zoo S3 bucket publishes (best known map).
# Filenames here are the bare HEF basename inside
# ``Compiled/<zoo_version>/<arch>/``. 404s on download are expected for
# any combination not in the catalogue and are reported, not fatal.
#
# The four entries Benchy already ships in resources/hefs/ for hailo8
# (v2.16.0) are documented in resources/hefs/NAMING.txt:
#   yolov8n.hef        -> v8_detection_n_hailo8.hef
#   yolov8n_seg.hef    -> v8_segmentation_n_hailo8.hef
#   yolov8s_pose.hef   -> v8_pose_s_hailo8.hef
#   yolov11n.hef       -> v11_detection_n_hailo8.hef
ZOO_MANIFEST: tuple[HEFManifestEntry, ...] = (
    # YOLOv8 detection
    HEFManifestEntry("v8", "detection", "n", "yolov8n.hef"),
    HEFManifestEntry("v8", "detection", "s", "yolov8s.hef"),
    HEFManifestEntry("v8", "detection", "m", "yolov8m.hef"),
    HEFManifestEntry("v8", "detection", "l", "yolov8l.hef"),
    HEFManifestEntry("v8", "detection", "x", "yolov8x.hef"),
    # YOLOv8 segmentation
    HEFManifestEntry("v8", "segmentation", "n", "yolov8n_seg.hef"),
    HEFManifestEntry("v8", "segmentation", "s", "yolov8s_seg.hef"),
    HEFManifestEntry("v8", "segmentation", "m", "yolov8m_seg.hef"),
    # YOLOv8 pose (Zoo skips 'n')
    HEFManifestEntry("v8", "pose", "s", "yolov8s_pose.hef"),
    HEFManifestEntry("v8", "pose", "m", "yolov8m_pose.hef"),
    # YOLOv11 detection
    HEFManifestEntry("v11", "detection", "n", "yolov11n.hef"),
    HEFManifestEntry("v11", "detection", "s", "yolov11s.hef"),
    HEFManifestEntry("v11", "detection", "m", "yolov11m.hef"),
)


def zoo_url(entry: HEFManifestEntry, arch: str, zoo_version: str) -> str:
    """Construct the S3 download URL for a manifest entry."""
    return f"{ZOO_BASE_URL}/{zoo_version}/{arch}/{entry.zoo_filename}"


@dataclass
class FetchResult:
    entry: Optional[HEFManifestEntry]  # None for release fetches (no Zoo manifest entry)
    arch: str
    url: str
    dest: Path
    status: str            # "downloaded" / "skipped-exists" / "missing-403/404" / "error" / "sha-mismatch"
    error: str | None = None
    source: str = "zoo"    # "zoo" or "release" — tagged so summary/log output is unambiguous


def _classify_http_error(
    e: urllib.error.HTTPError,
) -> tuple[str, str | None]:
    """Map an HTTPError to (status, error_msg).

    S3 returns 403 instead of 404 for objects that don't exist in a
    bucket that forbids ListObjects, so treat both as "not in the
    public catalogue" — non-fatal, just reported. 5xx are real server
    errors and surfaced as such.
    """
    if e.code in (403, 404):
        return f"missing-{e.code}", None
    return "error", f"HTTP {e.code}"


def fetch_one(
    entry: HEFManifestEntry,
    arch: str,
    zoo_version: str,
    output_dir: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> FetchResult:
    url = zoo_url(entry, arch, zoo_version)
    dest = output_dir / entry.canonical_filename(arch)

    if dest.exists() and not overwrite:
        return FetchResult(entry, arch, url, dest, "skipped-exists")

    if dry_run:
        # HEAD-probe so the user sees the real catalogue status before
        # committing to a (potentially long) download. Without this,
        # dry-run was just URL templating and silently lied when the
        # resolved path didn't exist (verified 2026-04-29 against the
        # real S3 bucket: dry-run claimed all 13 hailo10h@v2.18.0 URLs
        # would download; the GET returned 403 on every one).
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return FetchResult(entry, arch, url, dest, "downloaded")
        except urllib.error.HTTPError as e:
            status, err = _classify_http_error(e)
            return FetchResult(entry, arch, url, dest, status, err)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return FetchResult(entry, arch, url, dest, "error", str(e))

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(
            dest, "wb"
        ) as out:
            while chunk := resp.read(64 * 1024):
                out.write(chunk)
    except urllib.error.HTTPError as e:
        status, err = _classify_http_error(e)
        return FetchResult(entry, arch, url, dest, status, err)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # Partial download cleanup
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return FetchResult(entry, arch, url, dest, "error", str(e))

    return FetchResult(entry, arch, url, dest, "downloaded")


def resolve_zoo_versions(
    arches: Iterable[str],
    override: str | None = None,
) -> dict[str, str]:
    """Pick a zoo_version per arch.

    If ``override`` is given, every arch uses it (matches the
    ``--zoo-version`` CLI flag). Otherwise each arch falls back to its
    entry in :data:`DEFAULT_ZOO_VERSIONS`.
    """
    arches_t = tuple(arches)
    if override:
        return {a: override for a in arches_t}
    out: dict[str, str] = {}
    for a in arches_t:
        if a not in DEFAULT_ZOO_VERSIONS:
            raise ValueError(
                f"No DEFAULT_ZOO_VERSIONS entry for arch '{a}'. "
                f"Known: {sorted(DEFAULT_ZOO_VERSIONS)}."
            )
        out[a] = DEFAULT_ZOO_VERSIONS[a]
    return out


def fetch_all(
    arches: Iterable[str],
    *,
    zoo_versions: dict[str, str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest: Iterable[HEFManifestEntry] = ZOO_MANIFEST,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[FetchResult]:
    arches_t = tuple(arches)
    versions = zoo_versions if zoo_versions is not None else resolve_zoo_versions(arches_t)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    for arch in arches_t:
        for entry in manifest:
            results.append(
                fetch_one(
                    entry,
                    arch,
                    versions[arch],
                    output_dir,
                    dry_run=dry_run,
                    overwrite=overwrite,
                )
            )
    return results


# ============================================================================
# GitHub Release source
# ============================================================================
#
# The release source is the canonical home for Benchy's workstation-compiled
# gap-fillers (every Hailo-10H HEF, every OBB HEF, every v26 HEF). The release
# publishes a manifest.json alongside the .hef assets; we use it to know what
# exists and to verify SHA-256 on every download.


def release_url(canonical_filename: str, tag: str = HEFS_RELEASE_TAG) -> str:
    """Build the asset download URL for a HEF in a Benchy release."""
    return f"{GITHUB_RELEASE_BASE_URL}/{tag}/{canonical_filename}"


def release_manifest_url(tag: str = HEFS_RELEASE_TAG) -> str:
    """Build the URL of the release's manifest.json asset."""
    return f"{GITHUB_RELEASE_BASE_URL}/{tag}/manifest.json"


def _sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream the file through sha256 — handles 50+ MB HEFs without OOM."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_release_manifest(
    tag: str = HEFS_RELEASE_TAG,
    *,
    timeout: float = 30.0,
) -> dict[str, dict]:
    """Download and parse manifest.json from the release.

    Returns a dict keyed by canonical filename, mapping to the per-HEF
    manifest entry (with 'sha256', 'size_bytes', 'yolo_version', 'task',
    'size', 'arch', 'source').

    Raises ``urllib.error.URLError`` / ``HTTPError`` on network failure
    and ``json.JSONDecodeError`` on bad payload — callers decide whether
    to abort or degrade to zoo-only.
    """
    url = release_manifest_url(tag)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return {h["filename"]: h for h in data["hefs"]}


def fetch_from_release(
    canonical_filename: str,
    arch: str,
    output_dir: Path,
    expected_sha256: str,
    *,
    tag: str = HEFS_RELEASE_TAG,
    dry_run: bool = False,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> FetchResult:
    """Download one HEF from the release and verify SHA-256.

    Mirrors :func:`fetch_one` semantics for status values; adds a
    ``"sha-mismatch"`` status that deletes the partial file and surfaces
    a clear error so the caller exits non-zero.
    """
    url = release_url(canonical_filename, tag)
    dest = output_dir / canonical_filename

    if dest.exists() and not overwrite:
        # Existing local file — trust the manifest and re-verify before
        # claiming "skipped-exists". A stale file from a previous botched
        # download would be a silent footgun otherwise.
        actual = _sha256_of(dest)
        if actual == expected_sha256:
            return FetchResult(None, arch, url, dest, "skipped-exists", source="release")
        # Mismatch on existing file — treat the same as a fresh fetch
        # mismatch: report and return so the user can decide whether to
        # --overwrite. Don't auto-delete; the file might be intentional
        # (e.g. a manually-staged variant).
        return FetchResult(
            None, arch, url, dest, "sha-mismatch",
            error=f"existing file sha256 {actual} != manifest {expected_sha256}",
            source="release",
        )

    if dry_run:
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return FetchResult(None, arch, url, dest, "downloaded", source="release")
        except urllib.error.HTTPError as e:
            status, err = _classify_http_error(e)
            return FetchResult(None, arch, url, dest, status, err, source="release")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return FetchResult(None, arch, url, dest, "error", str(e), source="release")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(
            dest, "wb"
        ) as out:
            while chunk := resp.read(64 * 1024):
                out.write(chunk)
    except urllib.error.HTTPError as e:
        status, err = _classify_http_error(e)
        # Partial download cleanup
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return FetchResult(None, arch, url, dest, status, err, source="release")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return FetchResult(None, arch, url, dest, "error", str(e), source="release")

    actual = _sha256_of(dest)
    if actual != expected_sha256:
        # Partial / corrupt / wrong-asset — delete and surface error
        try:
            dest.unlink()
        except OSError:
            pass
        return FetchResult(
            None, arch, url, dest, "sha-mismatch",
            error=f"sha256 {actual} != manifest {expected_sha256}",
            source="release",
        )

    return FetchResult(None, arch, url, dest, "downloaded", source="release")


def fetch_release_for_arches(
    arches: Iterable[str],
    *,
    output_dir: Path,
    release_manifest: dict[str, dict],
    tag: str = HEFS_RELEASE_TAG,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[FetchResult]:
    """Fetch every HEF in the release manifest matching the requested arches."""
    arches_t = set(arches)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    for filename, info in sorted(release_manifest.items()):
        if info["arch"] not in arches_t:
            continue
        results.append(
            fetch_from_release(
                filename,
                info["arch"],
                output_dir,
                info["sha256"],
                tag=tag,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )
    return results


def fetch_with_release_fallback(
    arches: Iterable[str],
    *,
    zoo_versions: dict[str, str],
    output_dir: Path,
    release_manifest: dict[str, dict],
    tag: str = HEFS_RELEASE_TAG,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[FetchResult]:
    """Auto-source: try the Zoo first, fall back to the release.

    Iterates the *release manifest* (not ZOO_MANIFEST) so gap-fillers the
    Zoo doesn't ship are covered. For each canonical filename:

    - If a ZOO_MANIFEST entry produces the same canonical filename for
      this arch, try the Zoo first. On 403/404, fall back to release.
      Other Zoo errors (5xx, network) abort that file with the Zoo
      error — don't paper over real problems with a release fetch.
    - If no Zoo equivalent exists, go straight to release.
    """
    arches_t = set(arches)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build canonical_filename -> (zoo entry, arch) lookup.
    zoo_lookup: dict[str, tuple[HEFManifestEntry, str]] = {}
    for entry in ZOO_MANIFEST:
        for arch in arches_t:
            zoo_lookup[entry.canonical_filename(arch)] = (entry, arch)

    results: list[FetchResult] = []
    for filename, info in sorted(release_manifest.items()):
        if info["arch"] not in arches_t:
            continue

        zoo_match = zoo_lookup.get(filename)
        if zoo_match is not None:
            entry, arch = zoo_match
            zoo_result = fetch_one(
                entry,
                arch,
                zoo_versions[arch],
                output_dir,
                dry_run=dry_run,
                overwrite=overwrite,
            )
            if zoo_result.status in ("downloaded", "skipped-exists"):
                results.append(zoo_result)
                continue
            if zoo_result.status not in ("missing-403", "missing-404"):
                # Real error — surface it instead of silently retrying.
                results.append(zoo_result)
                continue
            # Else: fall through to release (still record the zoo miss
            # for context? Skip — keeps summary uncluttered. The release
            # result below stands as the authoritative outcome.)

        results.append(
            fetch_from_release(
                filename,
                info["arch"],
                output_dir,
                info["sha256"],
                tag=tag,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        )
    return results


def _format_summary(results: list[FetchResult], dry_run: bool) -> str:
    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status == "downloaded":
            source_counts[r.source] = source_counts.get(r.source, 0) + 1

    missing = counts.get("missing-403", 0) + counts.get("missing-404", 0)

    verb = "Would download" if dry_run else "Downloaded"
    lines = [
        "",
        "=== Fetch Summary ===",
        f"  {verb}:        {counts.get('downloaded', 0)}"
        + (f"  (zoo: {source_counts.get('zoo', 0)}, release: {source_counts.get('release', 0)})"
           if source_counts else ""),
        f"  Already on disk: {counts.get('skipped-exists', 0)}",
        f"  Not in catalogue (403/404): {missing}",
        f"  SHA-256 mismatch: {counts.get('sha-mismatch', 0)}",
        f"  Errors:           {counts.get('error', 0)}",
        "",
    ]
    for r in results:
        marker = {
            "downloaded": "  GOT ",
            "skipped-exists": "  SKIP",
            "missing-403": "  MISS",
            "missing-404": "  MISS",
            "sha-mismatch": "  HASH",
            "error": "  ERR ",
        }.get(r.status, "  ?   ")
        suffix = f" ({r.error})" if r.error else ""
        if r.status.startswith("missing-"):
            suffix = f" ({r.status.split('-', 1)[1]})"
        src_tag = f"[{r.source:7s}]"
        lines.append(f"{marker} {src_tag} {r.arch:8s} {r.dest.name:48s} <- {r.url}{suffix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Fetch prebuilt HEFs from the Hailo Model Zoo and/or the Benchy "
            "GitHub Release. Stages them in resources/hefs/ with canonical "
            "naming for the Pi-side runtime."
        )
    )
    p.add_argument(
        "--arch",
        choices=("hailo8", "hailo8l", "hailo10h", "both"),
        default="hailo8",
        help="Target architecture. 'both' = hailo8 + hailo10h.",
    )
    p.add_argument(
        "--source",
        choices=("zoo", "release", "both"),
        default="both",
        help=(
            "Where to fetch from. 'zoo' = Hailo Model Zoo S3 only "
            "(legacy behavior). 'release' = Benchy GitHub Release only "
            "(SHA-256 verified). 'both' (default) = try zoo first, fall "
            "back to release for HEFs the zoo doesn't ship; degrades to "
            "zoo-only if the release manifest is unreachable."
        ),
    )
    p.add_argument(
        "--release-tag",
        default=HEFS_RELEASE_TAG,
        help=(
            f"Benchy release tag to fetch from (default: {HEFS_RELEASE_TAG}). "
            "Bump deliberately to pull a newer batch of compiled HEFs."
        ),
    )
    p.add_argument(
        "--zoo-version",
        default=None,
        help=(
            "Hailo Model Zoo release tag override (forces a single version "
            "across all arches). When omitted, each arch falls back to its "
            "entry in DEFAULT_ZOO_VERSIONS: "
            + ", ".join(f"{a}={v}" for a, v in DEFAULT_ZOO_VERSIONS.items())
            + "."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to stage HEFs (default: resources/hefs/).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended URLs and destinations without downloading.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the canonical file already exists.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose progress logging.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("fetch_prebuilt_hefs")

    if args.arch == "both":
        arches = ("hailo8", "hailo10h")
    else:
        arches = (args.arch,)

    versions = resolve_zoo_versions(arches, override=args.zoo_version)

    # Load release manifest if either source needs it. For default 'both',
    # degrade to zoo-only on failure (keeps existing offline workflows
    # functional). For explicit 'release', a load failure is fatal.
    release_manifest: Optional[dict[str, dict]] = None
    if args.source in ("release", "both"):
        try:
            release_manifest = load_release_manifest(args.release_tag)
            log.info(
                "Loaded release manifest %s with %d entries",
                args.release_tag, len(release_manifest),
            )
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, OSError) as e:
            if args.source == "release":
                log.error(
                    "--source release requires a reachable manifest at %s. "
                    "Failed: %s",
                    release_manifest_url(args.release_tag), e,
                )
                return 1
            log.warning(
                "Release manifest at %s unreachable (%s); falling back to "
                "--source zoo only.",
                release_manifest_url(args.release_tag), e,
            )
            args.source = "zoo"
            release_manifest = None

    log.info(
        "Fetching from %s for arches %s -> %s%s",
        args.source,
        ", ".join(f"{a}@{versions[a]}" for a in arches),
        args.output_dir,
        " (dry-run)" if args.dry_run else "",
    )

    if args.source == "zoo":
        results = fetch_all(
            arches,
            zoo_versions=versions,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.source == "release":
        assert release_manifest is not None  # narrowed by the load block above
        results = fetch_release_for_arches(
            arches,
            output_dir=args.output_dir,
            release_manifest=release_manifest,
            tag=args.release_tag,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    else:  # both
        assert release_manifest is not None
        results = fetch_with_release_fallback(
            arches,
            zoo_versions=versions,
            output_dir=args.output_dir,
            release_manifest=release_manifest,
            tag=args.release_tag,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

    print(_format_summary(results, args.dry_run))

    # SHA mismatches and real network errors fail the run; missing assets
    # and skipped-exists do not (matches Zoo-only behavior for 403/404).
    fatal = sum(1 for r in results if r.status in ("error", "sha-mismatch"))
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
