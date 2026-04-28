# Prebuilt HEF source layer.
#
# Background: the Hailo Dataflow Compiler (i.e. `hailo_sdk_client`,
# which the .pt -> .onnx -> .har -> .hef pipeline imports) is x86_64
# Linux only. It will not install on aarch64, so a Pi 5 cannot compile
# its own HEFs. To make the verify suite work without a separate
# workstation, we look for prebuilt HEFs in two project-controlled
# locations before falling through to the (Pi-broken) compile path:
#
#   1. resources/hefs/ in the repo. Users drop HEFs here named per the
#      `repo_filename()` convention. This is also the canonical landing
#      spot for HEFs compiled on a workstation and copied across.
#   2. /usr/share/hailo-models/, populated by the
#      rpicam-apps-hailo-postprocess Debian package. This ships a
#      curated subset of Hailo Model Zoo HEFs vetted by Raspberry Pi.
#
# If neither contains a match, this returns None and the caller logs a
# clear "drop a HEF named X into resources/hefs/ or compile on x86_64"
# message before letting the compile pipeline fail with its existing
# error. See Issue 11 in resources/session_issues_2026-04-27.md.
import logging
import re
from pathlib import Path
from typing import Optional

from benchmark.schemas import YOLOTask

logger = logging.getLogger(__name__)


REPO_HEF_DIR = Path(__file__).resolve().parents[4] / "resources" / "hefs"
SYSTEM_HEF_DIR = Path("/usr/share/hailo-models")


# Curated mapping of (yolo_version, task, model_size, arch) -> filename
# inside /usr/share/hailo-models/. Populated from the
# rpicam-apps-hailo-postprocess package as observed on Pi OS Bookworm
# 2026-04. Extend as new prebuilts ship; do not invent entries that
# aren't actually on disk or the cache copy will fail downstream.
SYSTEM_PACKAGE_MAP: dict[tuple[str, YOLOTask, str, str], str] = {
    ("v8", YOLOTask.DETECTION, "s", "hailo8"): "yolov8s_h8.hef",
    ("v8", YOLOTask.DETECTION, "s", "hailo8l"): "yolov8s_h8l.hef",
    ("v8", YOLOTask.POSE, "s", "hailo8"): "yolov8s_pose_h8.hef",
    ("v8", YOLOTask.POSE, "s", "hailo8l"): "yolov8s_pose_h8l_pi.hef",
}


_SIZE_RE = re.compile(r"yolo(?:v)?\d+([nsmlx])")


def model_size_from_name(model_name: str) -> Optional[str]:
    """Extract size suffix (n/s/m/l/x) from a model name like 'yolov8n.pt'.

    Returns None if the name does not match the standard Ultralytics
    naming pattern (e.g. a custom user-supplied weights file).
    """
    stem = Path(model_name).stem.lower()
    m = _SIZE_RE.search(stem)
    return m.group(1) if m else None


def repo_filename(yolo_version: str, task: YOLOTask, model_size: str, arch: str) -> str:
    """Naming convention for HEFs dropped under resources/hefs/.

    Format: ``<yolo_version>_<task>_<model_size>_<arch>.hef``.
    Example: ``v8_detection_n_hailo8.hef``.
    """
    return f"{yolo_version}_{task.value}_{model_size}_{arch}.hef"


def find_prebuilt_hef(
    model_name: str,
    yolo_version: str,
    task: YOLOTask,
    arch: str,
    repo_hef_dir: Optional[Path] = None,
    system_hef_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Look for a prebuilt HEF.

    Search order:
        1. ``resources/hefs/<repo_filename>`` (or ``repo_hef_dir`` override)
        2. ``/usr/share/hailo-models/<mapped filename>`` per
           SYSTEM_PACKAGE_MAP (or ``system_hef_dir`` override)

    Returns the located ``Path`` or ``None`` if no match exists. The
    caller is responsible for copying the HEF into the runtime cache;
    we never mutate either source location.
    """
    repo_dir = repo_hef_dir or REPO_HEF_DIR
    sys_dir = system_hef_dir or SYSTEM_HEF_DIR

    model_size = model_size_from_name(model_name)
    if model_size is None:
        logger.debug(
            f"Cannot infer size from model name '{model_name}'; "
            "skipping prebuilt HEF lookup."
        )
        return None

    repo_candidate = repo_dir / repo_filename(yolo_version, task, model_size, arch)
    if repo_candidate.exists():
        logger.info(f"Using prebuilt HEF from repo: {repo_candidate}")
        return repo_candidate

    sys_filename = SYSTEM_PACKAGE_MAP.get((yolo_version, task, model_size, arch))
    if sys_filename:
        sys_candidate = sys_dir / sys_filename
        if sys_candidate.exists():
            logger.info(f"Using prebuilt HEF from system package: {sys_candidate}")
            return sys_candidate

    logger.info(
        f"No prebuilt HEF for {yolo_version}/{task.value}/{model_size}/{arch}. "
        f"Drop a file named '{repo_filename(yolo_version, task, model_size, arch)}' "
        f"into {repo_dir} (compile on x86_64), "
        "or rely on the conversion pipeline (which requires the Hailo "
        "Dataflow Compiler — x86_64 only)."
    )
    return None
