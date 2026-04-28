"""Smoke tests for the LLM-on-NPU profile and the new backend axis.

The `npu` profile must:
  - parse with `backend: hailo-10h`, `npu_metrics: true`, and a non-default
    `api_base` (HailoRT GenAI's Ollama-compatible REST endpoint),
  - list only models that exist as prebuilt HEFs in the Hailo Model Zoo
    GenAI catalogue,
  - flow `backend` and `npu_metrics` cleanly into `LLMBenchmarkConfig` so
    the runner attaches the NPU collector exactly when asked.
"""
from pathlib import Path

import pytest
import yaml

from benchmark.workloads.llm.runner import LLMBenchmarkConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_CONFIG = REPO_ROOT / "configs" / "llm_benchmark.yaml"


# Llama-family models with a published Hailo HEF (per docs/hailo.md).
# Post-Issue-7 the benchmark surface is llama-only, so the qwen2 /
# deepseek prebuilt HEFs are out of scope even though they are present
# in the Hailo Model Zoo GenAI catalogue. Models added to the npu profile
# must be in this set or the test fails — that's the contract that
# prevents the YAML from drifting ahead of what HailoRT GenAI can serve
# *and* what the project has scoped in.
HAILO_GENAI_PREBUILT_HEFS = {
    "llama3.2:3b",
}


@pytest.fixture(scope="module")
def llm_cfg() -> dict:
    return yaml.safe_load(LLM_CONFIG.read_text())


def test_npu_profile_exists(llm_cfg):
    assert "npu" in llm_cfg, "npu profile missing from llm_benchmark.yaml"


def test_npu_profile_has_hailo_backend(llm_cfg):
    assert llm_cfg["npu"]["backend"] == "hailo-10h"


def test_npu_profile_enables_npu_metrics(llm_cfg):
    assert llm_cfg["npu"]["npu_metrics"] is True


def test_npu_profile_overrides_api_base(llm_cfg):
    api_base = llm_cfg["npu"]["api_base"]
    # Must point somewhere other than Ollama's default port. The hailo-ollama
    # README defaults to port 8000 — we pin against that here so a config
    # drift back to :11434 trips a clean failure.
    assert api_base != "http://localhost:11434"
    assert api_base.startswith("http://")


def test_npu_profile_models_are_prebuilt_hefs(llm_cfg):
    models = llm_cfg["npu"]["models"]
    for tag in models:
        assert tag in HAILO_GENAI_PREBUILT_HEFS, (
            f"npu profile lists {tag!r} but no prebuilt HEF exists for it. "
            f"Either remove it or extend HAILO_GENAI_PREBUILT_HEFS with a "
            f"citation to the Hailo Model Zoo GenAI catalogue."
        )


def test_npu_profile_stays_at_or_below_3b(llm_cfg):
    # The npu profile must stay within the llama-family models that have
    # a published Hailo HEF. Currently that is llama3.2:3b only; if a
    # llama 1B HEF lands later it can be added to SMALL.
    models = llm_cfg["npu"]["models"]
    SMALL = {"llama3.2:3b"}
    for tag in models:
        assert tag in SMALL, f"npu profile too aggressive: {tag} > 3B"


def test_npu_profile_uses_drone_prompts(llm_cfg):
    # Drone prompts double as the realistic LLM workload for an aerial
    # platform — that's why we lean on the same set when first measuring
    # NPU performance, rather than the legacy "haiku" prompts.
    assert llm_cfg["npu"]["prompt_set"] == "drone"


def test_llm_config_accepts_backend_field():
    cfg = LLMBenchmarkConfig(backend="hailo-10h", npu_metrics=True)
    assert cfg.backend == "hailo-10h"
    assert cfg.npu_metrics is True


def test_llm_config_defaults_to_ollama_cpu():
    cfg = LLMBenchmarkConfig()
    assert cfg.backend == "ollama-cpu"
    assert cfg.npu_metrics is False


def test_npu_profile_does_not_activate_quant_sweep(llm_cfg):
    # Backends and quant sweeps are orthogonal axes. The npu profile uses
    # prebuilt HEFs whose tag namespace doesn't follow the "<base>-<quant>"
    # template, so accidentally inheriting the default profile's quants
    # would explode into nonsense tags.
    npu = llm_cfg["npu"]
    assert "quants" not in npu
    assert "quant_tag_template" not in npu
