"""Smoke tests for the `drone` profile (YOLO + LLM).

The drone profile must:
  - parse cleanly out of the shipped YAML configs
  - stay Hailo-compatible (detection only — seg/pose/OBB are blocked on Hailo
    per docs/hailo.md and benchmark/workloads/yolo/execution.py)
  - use the bumped 1280 input resolution and a drone-relevant dataset
  - select the curated `drone` prompt set on the LLM side
"""
from pathlib import Path

import pytest
import yaml

from benchmark.cli import _infer_yolo_model_info
from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.execution import HAILO_SUPPORTED_TASKS
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
