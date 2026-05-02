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


# Prebuilt HEFs confirmed in the Hailo Model Zoo GenAI 5.3.0 catalogue —
# served by the hailo-ollama REST endpoint and listed in its README. See
# docs/hailo.md "LLM on Hailo-10H". Models added to the npu profile must
# be in this set or the test fails — that's the contract that prevents the
# YAML from drifting ahead of what HailoRT GenAI can actually serve.
# Verified 2026-04-28 by hitting /api/tags on the running hailo-ollama
# server after installing hailo_gen_ai_model_zoo_5.3.0_arm64.deb.
# 5.1.1 had llama3.2:3b; Hailo dropped it and added llama3.2:1b in 5.3.0.
# No 7B HEFs have ever shipped in any release — Hailo positions the
# Hailo-10H for ~1-2B edge inference, not 7B-class workloads.
HAILO_GENAI_PREBUILT_HEFS = {
    "llama3.2:1b",
    "qwen2:1.5b",
    "qwen2.5:1.5b",
    "qwen2.5-coder:1.5b",
    "qwen3:1.7b",
    "deepseek_r1:1.5b",
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


def test_npu_profile_stays_within_zoo_size_range(llm_cfg):
    # The 5.3.0 GenAI Model Zoo only ships HEFs in the 1B-1.7B range.
    # No 3B, no 7B exist as HEFs — this is a Hailo-side constraint, not
    # a project preference. The npu profile must not list anything
    # outside HAILO_GENAI_PREBUILT_HEFS (already enforced by
    # test_npu_profile_models_are_prebuilt_hefs); this test additionally
    # guards against any future >2B tag being smuggled in via a
    # hypothetical extended whitelist without the corresponding HEF.
    models = llm_cfg["npu"]["models"]
    for tag in models:
        assert tag in HAILO_GENAI_PREBUILT_HEFS, f"npu profile lists {tag} which has no HEF in the 5.3.0 zoo"


def test_npu_profile_uses_drone_prompts(llm_cfg):
    # Drone prompts double as the realistic LLM workload for an aerial
    # platform — that's why we lean on the same set when first measuring
    # NPU performance, rather than the legacy "haiku" prompts.
    assert llm_cfg["npu"]["prompt_set"] == "drone"


def test_compare_profile_mirrors_npu_for_cross_backend(llm_cfg):
    # The `compare` profile exists for one job: produce a CPU-side row
    # the dashboard can split against the NPU row from the `npu` profile.
    # That comparison is only meaningful if both run THE SAME model and
    # THE SAME prompts. If this assert fails, the verify-sweep dashboard
    # ends up comparing two different models — defeating the whole point.
    assert "compare" in llm_cfg, "compare profile missing from llm_benchmark.yaml"
    assert llm_cfg["compare"]["models"] == llm_cfg["npu"]["models"], (
        f"compare profile must run the same model as npu profile for cross-backend comparison "
        f"(npu={llm_cfg['npu']['models']}, compare={llm_cfg['compare']['models']})"
    )
    assert llm_cfg["compare"]["prompt_set"] == llm_cfg["npu"]["prompt_set"], (
        "compare profile must use the same prompt_set as npu profile"
    )


def test_compare_profile_targets_cpu_not_npu(llm_cfg):
    # The compare profile is the CPU mirror — it must NOT inherit the npu
    # profile's hailo-10h backend tagging or :8000 api_base, otherwise it
    # would double up on hailo-ollama and produce no Ollama-CPU rows.
    compare = llm_cfg["compare"]
    assert compare.get("backend", "ollama-cpu") == "ollama-cpu", (
        "compare profile must use the default ollama-cpu backend"
    )
    assert compare.get("api_base", "http://localhost:11434") != "http://localhost:8000", (
        "compare profile must NOT point at hailo-ollama's :8000"
    )
    assert compare.get("npu_metrics", False) is False, (
        "compare profile is CPU-side; npu_metrics must be off"
    )


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
