#!/usr/bin/env python3
"""Fetch prebuilt HEFs from the Hailo Model Zoo S3 catalogue.

Downloads HEFs for the standard YOLO models the Zoo publishes, renames
them to Benchy's canonical ``<version>_<task>_<size>_<arch>.hef``
convention, and stages them under ``resources/hefs/``. The Pi-side
runtime (``benchmark/workloads/yolo/conversion/hef_source.py``) picks
them up automatically.

URL pattern (per ``resources/hefs/NAMING.txt``)::

    https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<zoo_version>/<arch>/<filename>.hef

The catalogue is incomplete: the Zoo does not publish OBB HEFs at all,
does not publish pose at the ``n`` size, and as of 2026-04 has no v11
segmentation/pose/OBB or any v26 prebuilts for Hailo-8. 404s are
expected and treated as "not in the catalogue" — they are reported but
do not abort the sweep.

Use this for the easy half of the bring-up (detection / pose / seg
where the Zoo ships them); use ``scripts/compile_workstation_hefs.sh``
for the gap models the Zoo does not.

Usage::

    scripts/fetch_prebuilt_hefs.py --arch hailo8
    scripts/fetch_prebuilt_hefs.py --arch hailo10h --dry-run
    scripts/fetch_prebuilt_hefs.py --arch both --zoo-version v2.18.0
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import urllib.error
import urllib.request

# Repo-relative output directory; the Pi-side hef_source.py expects
# this exact location.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "resources" / "hefs"

ZOO_BASE_URL = (
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
)
DEFAULT_ZOO_VERSION = "v2.16.0"

ARCHES = ("hailo8", "hailo8l", "hailo10h")


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
    entry: HEFManifestEntry
    arch: str
    url: str
    dest: Path
    status: str            # "downloaded" / "skipped-exists" / "missing-404" / "error"
    error: str | None = None


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
        return FetchResult(entry, arch, url, dest, "downloaded")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(
            dest, "wb"
        ) as out:
            while chunk := resp.read(64 * 1024):
                out.write(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return FetchResult(entry, arch, url, dest, "missing-404")
        return FetchResult(entry, arch, url, dest, "error", f"HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # Partial download cleanup
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return FetchResult(entry, arch, url, dest, "error", str(e))

    return FetchResult(entry, arch, url, dest, "downloaded")


def fetch_all(
    arches: Iterable[str],
    *,
    zoo_version: str = DEFAULT_ZOO_VERSION,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest: Iterable[HEFManifestEntry] = ZOO_MANIFEST,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[FetchResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    for arch in arches:
        for entry in manifest:
            results.append(
                fetch_one(
                    entry,
                    arch,
                    zoo_version,
                    output_dir,
                    dry_run=dry_run,
                    overwrite=overwrite,
                )
            )
    return results


def _format_summary(results: list[FetchResult], dry_run: bool) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    verb = "Would download" if dry_run else "Downloaded"
    lines = [
        "",
        "=== Fetch Summary ===",
        f"  {verb}:        {counts.get('downloaded', 0)}",
        f"  Already on disk: {counts.get('skipped-exists', 0)}",
        f"  Not in catalogue (404): {counts.get('missing-404', 0)}",
        f"  Errors:           {counts.get('error', 0)}",
        "",
    ]
    for r in results:
        marker = {
            "downloaded": "  GOT ",
            "skipped-exists": "  SKIP",
            "missing-404": "  MISS",
            "error": "  ERR ",
        }.get(r.status, "  ?   ")
        suffix = f" ({r.error})" if r.error else ""
        lines.append(f"{marker} {r.arch:8s} {r.dest.name:48s} <- {r.url}{suffix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Fetch prebuilt HEFs from the Hailo Model Zoo S3 catalogue and "
            "stage them in resources/hefs/ with Benchy's canonical naming."
        )
    )
    p.add_argument(
        "--arch",
        choices=("hailo8", "hailo8l", "hailo10h", "both"),
        default="hailo8",
        help="Target architecture. 'both' = hailo8 + hailo10h.",
    )
    p.add_argument(
        "--zoo-version",
        default=DEFAULT_ZOO_VERSION,
        help=f"Hailo Model Zoo release tag (default: {DEFAULT_ZOO_VERSION}).",
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

    log.info(
        "Fetching from Hailo Model Zoo %s for arches %s -> %s%s",
        args.zoo_version,
        ",".join(arches),
        args.output_dir,
        " (dry-run)" if args.dry_run else "",
    )

    results = fetch_all(
        arches,
        zoo_version=args.zoo_version,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    print(_format_summary(results, args.dry_run))

    error_count = sum(1 for r in results if r.status == "error")
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
