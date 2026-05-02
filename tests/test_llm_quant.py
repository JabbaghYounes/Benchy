"""Tests for the LLM quantization sweep axis.

Profile-level `quants:` × `models:` is expanded into Ollama tags via the
`quant_tag_template` string. The runner records the actual quantization
level reported by Ollama's /api/show, but the schema and writers must still
carry the field end-to-end.
"""
from pathlib import Path

import pytest
import yaml

from benchmark.cli import _expand_quant_sweep
from benchmark.schemas import LLMResult


REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_CONFIG = REPO_ROOT / "configs" / "llm_benchmark.yaml"


@pytest.fixture(scope="module")
def llm_cfg() -> dict:
    return yaml.safe_load(LLM_CONFIG.read_text())


def test_default_profile_does_not_declare_quant_sweep(llm_cfg):
    # Llama-only policy (2026-04-27): chat-quant variants are too heavy
    # on disk for an SD-card-backed Pi (~16 GB combined for the three
    # llama2:7b chat-quants). The default profile now ships a single
    # bare tag. Re-add `quants:` and `quant_tag_template:` to opt back
    # in — see configs/llm_benchmark.yaml comment + git history.
    default = llm_cfg["default"]
    assert "quants" not in default
    assert "quant_tag_template" not in default


def test_default_profile_expands_to_single_tag(llm_cfg):
    # With no quant sweep declared, _expand_quant_sweep returns the
    # model list unchanged — exactly one Ollama tag per model.
    default = llm_cfg["default"]
    expanded = _expand_quant_sweep(
        default["models"],
        default.get("quants", []),
        default.get("quant_tag_template", "{base}-{quant}"),
    )
    assert expanded == ["llama2:7b"]


def test_expand_with_default_template():
    out = _expand_quant_sweep(["llama3.2:3b"], ["q4_K_M", "q8_0"])
    assert out == ["llama3.2:3b-q4_K_M", "llama3.2:3b-q8_0"]


def test_expand_with_instruct_template():
    out = _expand_quant_sweep(
        ["llama3.2:3b"], ["q4_K_M", "q5_K_M"], "{base}-instruct-{quant}"
    )
    assert out == ["llama3.2:3b-instruct-q4_K_M", "llama3.2:3b-instruct-q5_K_M"]


def test_expand_cross_product():
    # 2 base models × 2 quants → 4 tags, base-major order
    out = _expand_quant_sweep(["a:7b", "b:7b"], ["q4", "q8"])
    assert out == ["a:7b-q4", "a:7b-q8", "b:7b-q4", "b:7b-q8"]


def test_no_quants_returns_models_unchanged():
    out = _expand_quant_sweep(["llama3.2:1b", "llama2:7b"], [])
    assert out == ["llama3.2:1b", "llama2:7b"]


def test_no_quants_returns_a_copy():
    # Callers may mutate the returned list — make sure we don't alias.
    base = ["llama2:7b"]
    out = _expand_quant_sweep(base, [])
    out.append("mutated")
    assert base == ["llama2:7b"]


def test_empty_models_with_quants_returns_empty():
    assert _expand_quant_sweep([], ["q4_K_M"]) == []


def test_llm_result_carries_quantization_through_to_dict():
    result = LLMResult(
        model_name="llama2:7b-chat-q4_K_M",
        model_size="7B",
        quantization="Q4_K_M",
    )
    payload = result.to_dict()
    assert payload["quantization"] == "Q4_K_M"


def test_llm_result_quantization_optional():
    # Existing runs without quantization info must still serialize.
    result = LLMResult(model_name="custom:tag", model_size="7B")
    payload = result.to_dict()
    assert payload["quantization"] is None


def test_drone_profile_does_not_mix_with_quant_sweep(llm_cfg):
    # The drone profile is about prompts, not quants. Keep the axes orthogonal
    # so dashboards don't get confused by overlapping sweeps.
    drone = llm_cfg["drone"]
    assert "quants" not in drone
