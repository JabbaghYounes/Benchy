"""Smoke tests for the `drone` profile (YOLO + LLM).

The drone profile must:
  - parse cleanly out of the shipped YAML configs
  - stay Hailo-compatible (currently detection only — segmentation and
    pose are still blocked on Hailo per docs/hailo.md and
    benchmark/workloads/yolo/execution.py; Phase 3a unblocked OBB)
  - use the bumped 1280 input resolution and a drone-relevant dataset
  - select the curated `drone` prompt set on the LLM side
"""
from pathlib import Path

import pytest
import yaml

from benchmark.cli import _infer_yolo_model_info
from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.execution import (
    HAILO_OPTIMIZED_MODELS,
    HAILO_SUPPORTED_TASKS,
    check_hailo_compatibility,
)
from benchmark.workloads.llm.runner import DRONE_PROMPTS, PromptSet


REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO_CONFIG = REPO_ROOT / "configs" / "yolo_benchmark.yaml"
LLM_CONFIG = REPO_ROOT / "configs" / "llm_benchmark.yaml"


@pytest.fixture(scope="module")
def yolo_cfg() -> dict:
    return yaml.safe_load(YOLO_CONFIG.read_text())


@pytest.fixture(scope="module")
def llm_cfg() -> dict:
    return yaml.safe_load(LLM_CONFIG.read_text())


def test_yolo_drone_profile_exists(yolo_cfg):
    assert "drone" in yolo_cfg, "drone profile missing from yolo_benchmark.yaml"


def test_yolo_drone_profile_is_hailo_compatible(yolo_cfg):
    drone = yolo_cfg["drone"]
    # Every task in the drone profile must be supported on Hailo for every
    # YOLO version the profile lists. This guards against future edits that
    # quietly add seg/pose/OBB and silently break Pi+Hailo runs.
    for version in drone["yolo_versions"]:
        supported = {t.value for t in HAILO_SUPPORTED_TASKS.get(version, [])}
        for task in drone["tasks"]:
            assert task in supported, (
                f"Drone profile task {task!r} not Hailo-compatible for {version}"
            )


def test_yolo_drone_uses_1280_input(yolo_cfg):
    assert yolo_cfg["drone"]["input_resolution"] == 1280


def test_yolo_drone_uses_visdrone_dataset(yolo_cfg):
    assert yolo_cfg["drone"]["datasets"]["detection"] == "VisDrone.yaml"


def test_yolo_drone_drops_large_sizes(yolo_cfg):
    # l/x at 1280×1280 don't fit the realistic Pi 5 budget.
    sizes = yolo_cfg["drone"]["model_sizes"]
    assert "l" not in sizes
    assert "x" not in sizes


def test_model_name_inference_unaffected_by_resolution():
    # Resolution lives in the config; the filename parser must still resolve
    # versions/tasks regardless of input size.
    version, task = _infer_yolo_model_info("yolo11s.pt")
    assert version == "v11"
    assert task == YOLOTask.DETECTION


def test_llm_drone_profile_selects_drone_prompts(llm_cfg):
    drone = llm_cfg["drone"]
    assert drone["prompt_set"] == "drone"


def test_drone_prompt_set_returns_drone_prompts():
    prompts = PromptSet.get_prompts("drone")
    assert prompts == DRONE_PROMPTS


def test_drone_prompt_set_has_five_prompts():
    assert len(DRONE_PROMPTS) == 5


def test_drone_prompts_carry_drone_category():
    for p in DRONE_PROMPTS:
        assert p["category"] == "drone"


def test_drone_prompt_ids_cover_use_cases():
    ids = {p["id"] for p in DRONE_PROMPTS}
    expected = {
        "scene_description",
        "target_identification",
        "mission_preflight",
        "telemetry_interpretation",
        "hazard_reasoning",
    }
    assert ids == expected


# ----- Phase 3a: OBB-on-Hailo invariants ------------------------------------


@pytest.mark.parametrize("yolo_version", ["v8", "v11", "v26"])
def test_obb_optimized_models_pass_hailo_compatibility(yolo_version):
    """Every OBB model in HAILO_OPTIMIZED_MODELS must clear the runtime
    compatibility gate. If a future edit drops OBB from
    HAILO_SUPPORTED_TASKS but leaves models in HAILO_OPTIMIZED_MODELS,
    this test catches the mismatch.
    """
    obb_models = HAILO_OPTIMIZED_MODELS[yolo_version].get(YOLOTask.OBB, [])
    assert obb_models, (
        f"Phase 3a expects HAILO_OPTIMIZED_MODELS[{yolo_version!r}] to "
        f"include OBB entries; found none."
    )
    for model_name in obb_models:
        ok, reason = check_hailo_compatibility(model_name, yolo_version, YOLOTask.OBB)
        assert ok, (
            f"{model_name!r} ({yolo_version}, OBB) failed Hailo "
            f"compatibility: {reason}"
        )


@pytest.mark.parametrize("yolo_version", ["v8", "v11", "v26"])
def test_obb_in_hailo_supported_tasks(yolo_version):
    """The whitelist should accept OBB on every supported version.
    v26 is marked experimental in docs but stays in the whitelist; if it
    needs to be dropped post-Slice-6, this test is the contract that
    must be updated alongside.
    """
    assert YOLOTask.OBB in HAILO_SUPPORTED_TASKS[yolo_version]


# ----- Phase 3b: segmentation-on-Hailo invariants ---------------------------


@pytest.mark.parametrize("yolo_version", ["v8", "v11", "v26"])
def test_seg_optimized_models_pass_hailo_compatibility(yolo_version):
    """Every -seg model in HAILO_OPTIMIZED_MODELS must clear the runtime
    compatibility gate. Same shape as the OBB invariant: catches any
    future edit that drops SEGMENTATION from HAILO_SUPPORTED_TASKS but
    leaves -seg models in HAILO_OPTIMIZED_MODELS.
    """
    seg_models = HAILO_OPTIMIZED_MODELS[yolo_version].get(YOLOTask.SEGMENTATION, [])
    assert seg_models, (
        f"Phase 3b expects HAILO_OPTIMIZED_MODELS[{yolo_version!r}] to "
        f"include SEGMENTATION entries; found none."
    )
    for model_name in seg_models:
        ok, reason = check_hailo_compatibility(
            model_name, yolo_version, YOLOTask.SEGMENTATION
        )
        assert ok, (
            f"{model_name!r} ({yolo_version}, SEGMENTATION) failed Hailo "
            f"compatibility: {reason}"
        )


@pytest.mark.parametrize("yolo_version", ["v8", "v11", "v26"])
def test_segmentation_in_hailo_supported_tasks(yolo_version):
    """Whitelist should accept SEGMENTATION on every supported version
    after Phase 3b. v26-seg follows the same experimental treatment as
    v26-obb until Slice 5 hardware verification.
    """
    assert YOLOTask.SEGMENTATION in HAILO_SUPPORTED_TASKS[yolo_version]


# ----- Phase 3c: pose-on-Hailo invariants -----------------------------------


@pytest.mark.parametrize("yolo_version", ["v8", "v11", "v26"])
def test_pose_optimized_models_pass_hailo_compatibility(yolo_version):
    """Every -pose model in HAILO_OPTIMIZED_MODELS must clear the runtime
    compatibility gate. Same shape as the OBB / seg invariants.
    """
    pose_models = HAILO_OPTIMIZED_MODELS[yolo_version].get(YOLOTask.POSE, [])
    assert pose_models, (
        f"Phase 3c expects HAILO_OPTIMIZED_MODELS[{yolo_version!r}] to "
        f"include POSE entries; found none."
    )
    for model_name in pose_models:
        ok, reason = check_hailo_compatibility(
            model_name, yolo_version, YOLOTask.POSE
        )
        assert ok, (
            f"{model_name!r} ({yolo_version}, POSE) failed Hailo "
            f"compatibility: {reason}"
        )


@pytest.mark.parametrize("yolo_version", ["v8", "v11", "v26"])
def test_pose_in_hailo_supported_tasks(yolo_version):
    """Whitelist should accept POSE on every supported version after
    Phase 3c. With this in place, all five YOLO tasks are unblocked on
    Hailo at the whitelist layer.
    """
    assert YOLOTask.POSE in HAILO_SUPPORTED_TASKS[yolo_version]


@pytest.mark.parametrize("yolo_version", ["v8", "v11", "v26"])
def test_all_yolo_tasks_unblocked_on_hailo(yolo_version):
    """Final invariant — no YOLO task should be missing from the Hailo
    whitelist after Phase 3c. Catches a future regression that drops
    one of the five.
    """
    expected = {
        YOLOTask.DETECTION,
        YOLOTask.CLASSIFICATION,
        YOLOTask.OBB,
        YOLOTask.SEGMENTATION,
        YOLOTask.POSE,
    }
    assert expected.issubset(set(HAILO_SUPPORTED_TASKS[yolo_version]))


# ----- Polish 2: drone_full profile -----------------------------------------


def test_drone_full_profile_exists(yolo_cfg):
    assert "drone_full" in yolo_cfg, (
        "drone_full profile missing from yolo_benchmark.yaml"
    )


def test_drone_full_exercises_phase3_tasks(yolo_cfg):
    drone_full = yolo_cfg["drone_full"]
    # Must run beyond detection — that's the whole point of drone_full.
    tasks = set(drone_full["tasks"])
    assert "detection" in tasks
    assert "obb" in tasks, "drone_full should run OBB (aerial DOTA)"
    assert tasks - {"detection", "obb", "segmentation", "pose", "classification"} == set(), (
        "drone_full has unknown task entries"
    )


def test_drone_full_uses_1280_input(yolo_cfg):
    assert yolo_cfg["drone_full"]["input_resolution"] == 1280


def test_drone_full_drops_large_sizes(yolo_cfg):
    # Sweeping all five tasks at 1280 is already expensive; reject l/x.
    sizes = yolo_cfg["drone_full"]["model_sizes"]
    assert "l" not in sizes
    assert "x" not in sizes


def test_drone_full_uses_aerial_datasets_where_relevant(yolo_cfg):
    drone_full = yolo_cfg["drone_full"]
    datasets = drone_full["datasets"]
    # Detection on a drone-specific dataset, OBB on DOTA (aerial). seg
    # and pose use COCO defaults — there is no widely-used aerial seg
    # or aerial pose dataset, so we don't pretend.
    assert datasets["detection"] == "VisDrone.yaml"
    assert datasets["obb"].lower().startswith("dota"), (
        f"drone_full OBB dataset should be a DOTA variant; got {datasets['obb']!r}"
    )


def test_drone_full_tasks_all_clear_hailo_compatibility(yolo_cfg):
    """Every (version, task) combo in drone_full must clear the Hailo
    compatibility gate. If a future edit drops POSE from
    HAILO_SUPPORTED_TASKS but leaves it in drone_full, this trips.
    """
    drone_full = yolo_cfg["drone_full"]
    for version in drone_full["yolo_versions"]:
        for task_name in drone_full["tasks"]:
            task = YOLOTask(task_name)
            # Pick the first optimised model for this (version, task) so
            # the test exercises something realistic.
            models = HAILO_OPTIMIZED_MODELS[version].get(task, [])
            if not models:
                continue
            ok, reason = check_hailo_compatibility(models[0], version, task)
            assert ok, (
                f"drone_full has ({version}, {task_name}) but "
                f"{models[0]!r} fails Hailo compat: {reason}"
            )
