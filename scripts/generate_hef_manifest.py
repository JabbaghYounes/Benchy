#!/usr/bin/env python3
"""Generate manifest.json for a Benchy HEF release.

Walks ``resources/hefs/``, computes SHA-256 + size for every ``.hef``
file, parses the canonical filename into
``(yolo_version, task, size, arch)``, looks up provenance from the
embedded PROVENANCE map, and emits a JSON manifest suitable for upload
as a GitHub Release asset alongside the HEFs themselves.

Usage::

    python3 scripts/generate_hef_manifest.py
    python3 scripts/generate_hef_manifest.py --release-tag hefs-v3
    python3 scripts/generate_hef_manifest.py --output -    # stdout
    python3 scripts/generate_hef_manifest.py --hefs-dir /tmp/staged-hefs

Exits non-zero (with a clear message) if:

- Any ``.hef`` file in the directory has no PROVENANCE entry — forces
  explicit provenance tagging of every new HEF before it can ship.
- Any expected file from PROVENANCE is missing from the directory —
  catches incomplete release prep before the upload step.
- Any filename doesn't match the canonical
  ``<version>_<task>_<size>_<arch>.hef`` pattern.

When extending PROVENANCE for a new release, never remove an existing
entry without bumping the release tag — old fetcher pins must keep
working against historical releases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEFS_DIR = REPO_ROOT / "resources" / "hefs"


# Provenance per HEF: which compile session / Hailo Model Zoo version
# produced this exact binary. Audit verified against `git log` on `hef`
# branch up to commit b199532 (2026-05-02) for the original 30 HEFs;
# extended for hefs-v2 with the two v8 pose hailo10h HEFs from the
# 2026-05-03 workstation session (END_NODE_TABLE patch landed pre-compile);
# extended for hefs-v3 with two larger v8 pose hailo10h HEFs (m, l) plus
# the previously-missing hailo8 v8n-pose, all from the 2026-05-04
# workstation session (finetune batch_size=4 ALLS override added to fit
# v8l-pose QAT in 11 GB VRAM — same patch is harmless on smaller variants).
PROVENANCE: dict[str, str] = {
    # Hailo Model Zoo v2.16.0 — 13 HEFs, all hailo8
    "v8_detection_n_hailo8.hef":      "zoo-v2.16.0",
    "v8_detection_s_hailo8.hef":      "zoo-v2.16.0",
    "v8_detection_m_hailo8.hef":      "zoo-v2.16.0",
    "v8_detection_l_hailo8.hef":      "zoo-v2.16.0",
    "v8_detection_x_hailo8.hef":      "zoo-v2.16.0",
    "v8_pose_s_hailo8.hef":           "zoo-v2.16.0",
    "v8_pose_m_hailo8.hef":           "zoo-v2.16.0",
    "v8_segmentation_n_hailo8.hef":   "zoo-v2.16.0",
    "v8_segmentation_s_hailo8.hef":   "zoo-v2.16.0",
    "v8_segmentation_m_hailo8.hef":   "zoo-v2.16.0",
    "v11_detection_n_hailo8.hef":     "zoo-v2.16.0",
    "v11_detection_s_hailo8.hef":     "zoo-v2.16.0",
    "v11_detection_m_hailo8.hef":     "zoo-v2.16.0",

    # Workstation 2026-04-29 sweep (commit 091342f) — 6 HEFs
    "v8_detection_n_hailo10h.hef":     "workstation-rtx2080ti-2026-04-29",
    "v8_detection_s_hailo10h.hef":     "workstation-rtx2080ti-2026-04-29",
    "v11_detection_n_hailo10h.hef":    "workstation-rtx2080ti-2026-04-29",
    "v11_pose_n_hailo8.hef":           "workstation-rtx2080ti-2026-04-29",
    "v11_pose_n_hailo10h.hef":         "workstation-rtx2080ti-2026-04-29",
    "v11_segmentation_n_hailo10h.hef": "workstation-rtx2080ti-2026-04-29",

    # Workstation 2026-04-30 retry sweep (commit 61fc5c2) — 9 HEFs
    "v8_obb_n_hailo8.hef":             "workstation-rtx2080ti-2026-04-30",
    "v8_obb_n_hailo10h.hef":           "workstation-rtx2080ti-2026-04-30",
    "v11_obb_n_hailo8.hef":            "workstation-rtx2080ti-2026-04-30",
    "v11_obb_n_hailo10h.hef":          "workstation-rtx2080ti-2026-04-30",
    "v26_obb_n_hailo8.hef":            "workstation-rtx2080ti-2026-04-30",
    "v26_obb_n_hailo10h.hef":          "workstation-rtx2080ti-2026-04-30",
    "v26_pose_n_hailo8.hef":           "workstation-rtx2080ti-2026-04-30",
    "v26_pose_n_hailo10h.hef":         "workstation-rtx2080ti-2026-04-30",
    "v26_segmentation_n_hailo10h.hef": "workstation-rtx2080ti-2026-04-30",

    # Workstation 2026-04-30 step (b) retry (commit c37bb3f) — 1 HEF
    "v26_detection_n_hailo8.hef":      "workstation-rtx2080ti-2026-04-30",

    # Workstation 2026-05-02 hefs-v1 release prep — 1 HEF
    "v8_segmentation_n_hailo10h.hef":  "workstation-rtx2080ti-2026-05-02",

    # Workstation 2026-05-03 hefs-v2 release prep — 2 HEFs
    # END_NODE_TABLE patch ("v8", YOLOTask.POSE) landed in
    # har_generator.py before compile, unblocking the v8 pose family
    # on hailo10h. v8_pose_s_hailo10h closes the last AI HAT+ 2 verify
    # gap (12/13 -> 13/13); v8_pose_n_hailo10h is a bonus from the
    # same patch — wasn't in the original missing list.
    "v8_pose_s_hailo10h.hef":          "workstation-rtx2080ti-2026-05-03",
    "v8_pose_n_hailo10h.hef":          "workstation-rtx2080ti-2026-05-03",

    # Workstation 2026-05-04 hefs-v3 release prep — 3 HEFs
    # MODEL_SCRIPT_OVERRIDES finetune batch_size=4 patch landed in
    # hef_compiler.py before compile, allowing v8l-pose QAT to fit in
    # 11 GB VRAM on the 2080 Ti. v8_pose_m/l_hailo10h extend the
    # AI HAT+ 2 v8 pose family to n/s/m/l. v8_pose_n_hailo8 closes
    # the matching AI HAT+ gap (Hailo Model Zoo v2.16.0 ships pose s/m
    # for hailo8 but not n).
    "v8_pose_m_hailo10h.hef":          "workstation-rtx2080ti-2026-05-04",
    "v8_pose_l_hailo10h.hef":          "workstation-rtx2080ti-2026-05-04",
    "v8_pose_n_hailo8.hef":            "workstation-rtx2080ti-2026-05-04",
}


_FILENAME_RE = re.compile(
    r"^(?P<version>v8|v11|v26)"
    r"_(?P<task>detection|classification|obb|segmentation|pose)"
    r"_(?P<size>n|s|m|l|x)"
    r"_(?P<arch>hailo8|hailo8l|hailo10h)\.hef$"
)


@dataclass
class HEFManifestEntry:
    filename: str
    sha256: str
    size_bytes: int
    yolo_version: str
    task: str
    size: str
    arch: str
    source: str


def parse_filename(filename: str) -> Optional[dict[str, str]]:
    """Parse a canonical HEF filename into its component fields."""
    m = _FILENAME_RE.match(filename)
    return m.groupdict() if m else None


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream the file through sha256 — handles 50+ MB HEFs without OOM."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(hefs_dir: Path) -> list[HEFManifestEntry]:
    on_disk = sorted(p.name for p in hefs_dir.glob("*.hef"))
    expected = sorted(PROVENANCE.keys())

    extras = set(on_disk) - set(expected)
    missing = set(expected) - set(on_disk)
    errors: list[str] = []

    if extras:
        errors.append(
            f"{len(extras)} HEF(s) in {hefs_dir} have no PROVENANCE entry — "
            f"add them to PROVENANCE before generating the manifest:\n  "
            + "\n  ".join(sorted(extras))
        )
    if missing:
        errors.append(
            f"{len(missing)} HEF(s) listed in PROVENANCE are not in {hefs_dir} — "
            f"either fetch/compile them first or remove from PROVENANCE:\n  "
            + "\n  ".join(sorted(missing))
        )
    if errors:
        raise SystemExit("\n\n".join(errors))

    entries: list[HEFManifestEntry] = []
    for name in on_disk:
        parsed = parse_filename(name)
        if parsed is None:
            raise SystemExit(
                f"Filename does not match canonical pattern "
                f"<version>_<task>_<size>_<arch>.hef: {name}"
            )
        path = hefs_dir / name
        entries.append(
            HEFManifestEntry(
                filename=name,
                sha256=sha256_of(path),
                size_bytes=path.stat().st_size,
                yolo_version=parsed["version"],
                task=parsed["task"],
                size=parsed["size"],
                arch=parsed["arch"],
                source=PROVENANCE[name],
            )
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manifest.json for a Benchy HEF release."
    )
    parser.add_argument(
        "--hefs-dir",
        type=Path,
        default=DEFAULT_HEFS_DIR,
        help="Directory containing the .hef files (default: <repo>/resources/hefs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write manifest.json (default: <hefs-dir>/manifest.json). "
        "Pass '-' to write to stdout.",
    )
    parser.add_argument(
        "--release-tag",
        default="hefs-v3",
        help="Release tag to embed in the manifest (default: hefs-v3)",
    )
    args = parser.parse_args()

    if not args.hefs_dir.is_dir():
        raise SystemExit(f"HEFs directory does not exist: {args.hefs_dir}")

    entries = build_manifest(args.hefs_dir)

    manifest = {
        "release_tag": args.release_tag,
        "hef_count": len(entries),
        "total_size_bytes": sum(e.size_bytes for e in entries),
        "hefs": [asdict(e) for e in entries],
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=False)

    if args.output is None:
        out_path = args.hefs_dir / "manifest.json"
        out_path.write_text(rendered + "\n")
        print(
            f"Wrote {out_path} ({len(entries)} HEFs, "
            f"{manifest['total_size_bytes'] / 1024 / 1024:.1f} MB)",
            file=sys.stderr,
        )
    elif str(args.output) == "-":
        sys.stdout.write(rendered + "\n")
    else:
        args.output.write_text(rendered + "\n")
        print(
            f"Wrote {args.output} ({len(entries)} HEFs, "
            f"{manifest['total_size_bytes'] / 1024 / 1024:.1f} MB)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
