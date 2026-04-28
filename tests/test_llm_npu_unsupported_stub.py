"""Tests for the `npu` profile's unsupported-on-this-hardware stub path.

When `--profile npu` runs on anything other than Platform.RPI_AI_HAT_PLUS_2
the runner cannot legally execute (Hailo-8/8L can't host LLMs on the NPU
and a CPU fallback under backend=hailo-10h would be a fairness violation).
Instead of returning an empty list — which would give the cross-platform
dashboard a missing row — the runner emits zero-valued LLMResult stubs
tagged backend=hailo-10h so the chart shows an explicit "tried, 0 TPS"
bar on the Hailo-10H axis. The verify_ai_hat_plus.sh runner depends on
this for chart-comparison parity with verify_ai_hat_plus_2.sh.
"""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from benchmark.cli import _build_unsupported_npu_stubs, run_llm_benchmark
from benchmark.schemas import SystemInfo


REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_CONFIG = REPO_ROOT / "configs" / "llm_benchmark.yaml"


@pytest.fixture(scope="module")
def llm_cfg() -> dict:
    return yaml.safe_load(LLM_CONFIG.read_text())


def _fake_system_info(platform: str) -> SystemInfo:
    return SystemInfo(
        platform=platform,
        cpu_model="dummy",
        accelerator="none",
        ram_size_gb=8.0,
        storage_type="ssd",
        cooling_config="passive",
        power_mode="default",
        os_version="dummy-os",
        kernel_version="0.0.0",
    )


def test_helper_emits_one_stub_per_model():
    profile_config = {
        "models": ["llama3.2:3b", "llama2:7b"],
        "model_groups": ["3B", "7B"],
    }
    stubs = _build_unsupported_npu_stubs(profile_config, "hailo-10h")
    assert len(stubs) == 2
    assert {s.model_name for s in stubs} == {"llama3.2:3b", "llama2:7b"}


def test_helper_zeroes_all_perf_fields():
    profile_config = {"models": ["llama3.2:3b"], "model_groups": ["3B"]}
    stub = _build_unsupported_npu_stubs(profile_config, "hailo-10h")[0]
    assert stub.tokens_per_second == 0.0
    assert stub.tps_mean == 0.0
    assert stub.ttft_mean_ms == 0.0
    assert stub.latency_mean_ms == 0.0
    assert stub.total_latency_ms == 0.0
    assert stub.warmup_runs == 0
    assert stub.measured_runs == 0


def test_helper_tags_backend_for_dashboard_axis():
    profile_config = {"models": ["llama3.2:3b"], "model_groups": ["3B"]}
    stub = _build_unsupported_npu_stubs(profile_config, "hailo-10h")[0]
    # backend must be hailo-10h so the stub lands on the same dashboard
    # axis as the real Hailo-10H runs from verify_ai_hat_plus_2.sh.
    assert stub.backend == "hailo-10h"


def test_helper_marks_prompt_id_for_provenance():
    profile_config = {"models": ["llama3.2:3b"], "model_groups": ["3B"]}
    stub = _build_unsupported_npu_stubs(profile_config, "hailo-10h")[0]
    # Prompt-id is the discriminator a reader uses to tell a stub from a
    # real-but-very-slow run; LLMResult has no dedicated status field.
    assert stub.prompt_id == "unsupported-on-this-hardware"


def test_helper_handles_empty_models_list():
    # Defensive: a malformed npu profile shouldn't crash the runner.
    assert _build_unsupported_npu_stubs({"models": []}, "hailo-10h") == []
    assert _build_unsupported_npu_stubs({}, "hailo-10h") == []


def test_runner_emits_stubs_when_npu_profile_on_wrong_platform(llm_cfg, tmp_path):
    # Patch check_ollama_status so the test doesn't need a live Ollama
    # server — the platform gate runs *before* that check anyway, but the
    # import path resolves through the runner module.
    fake_status = {"server_running": True}
    with patch("benchmark.workloads.llm.check_ollama_status", return_value=fake_status):
        results = run_llm_benchmark(
            llm_cfg,
            profile="npu",
            output_dir=tmp_path,
            system_info=_fake_system_info("rpi_ai_hat_plus"),  # the wrong Pi
        )
    # We get stub rows back, not an empty list. One per model in the npu
    # profile YAML. Their backend axis matches the real Hailo-10H runs.
    assert len(results) == len(llm_cfg["npu"]["models"])
    assert all(r.backend == "hailo-10h" for r in results)
    assert all(r.tokens_per_second == 0.0 for r in results)
    assert all(r.prompt_id == "unsupported-on-this-hardware" for r in results)


def test_runner_does_not_emit_stubs_for_non_npu_profile(llm_cfg, tmp_path):
    # Sanity: the stub path is gated by `npu_metrics: true`. The default
    # profile must not accidentally trip it on any platform.
    fake_status = {"server_running": True}
    with patch("benchmark.workloads.llm.check_ollama_status", return_value=fake_status):
        with patch("benchmark.workloads.llm.LLMBenchmarkRunner") as MockRunner:
            MockRunner.return_value.run.return_value = []
            results = run_llm_benchmark(
                llm_cfg,
                profile="default",
                output_dir=tmp_path,
                system_info=_fake_system_info("rpi_ai_hat_plus"),
            )
    # Default profile on a non-Hailo-10H Pi should *not* return stubs —
    # it should attempt a real run (which our mock makes empty).
    assert results == []
