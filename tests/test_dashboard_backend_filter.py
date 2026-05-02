"""Tests for the Phase 7 backend axis on the dashboard.

Builds a synthetic aggregator with two LLMResults that share
model_name + prompt_id but differ in `backend` (Ollama-CPU vs
Hailo-10H). Confirms:

  1. The aggregator's default group_by includes `backend`, so the two
     runs do NOT collapse into a single LLMAggregatedMetrics row.
  2. `LLMAggregatedMetrics.backend` flows through `to_dict()` so the
     dashboard's `llmData` JSON has it.
  3. The rendered HTML carries the new filter chip, the LLM table's
     Backend column, and the JS hooks (`filter-backend` ID,
     `applyFilters` reading it).
"""
import re
from pathlib import Path

import pytest

from benchmark.aggregation.aggregator import ResultsAggregator
from benchmark.reporting.dashboard import DashboardGenerator
from benchmark.schemas import LLMResult, SystemInfo


def _make_system_info(platform: str = "rpi_ai_hat_plus_2") -> SystemInfo:
    return SystemInfo(
        platform=platform,
        cpu_model="Cortex-A76",
        accelerator="Hailo-10H",
        ram_size_gb=8.0,
        storage_type="SD Card",
        cooling_config="passive",
        power_mode="default",
        os_version="Raspberry Pi OS",
        kernel_version="6.6.0",
    )


def _make_llm_result(
    backend: str | None = "ollama-cpu",
    tps: float = 12.5,
) -> LLMResult:
    return LLMResult(
        model_name="llama3.2:3b",
        model_size="3B",
        prompt_id="scene_description",
        prompt_tokens=42,
        output_tokens=120,
        time_to_first_token_ms=180.0,
        tokens_per_second=tps,
        total_latency_ms=950.0,
        ttft_mean_ms=180.0,
        tps_mean=tps,
        tps_std=0.4,
        tps_median=tps,
        latency_mean_ms=950.0,
        backend=backend,
        # NPU-side fields populate only on hailo-10h; both rows go through
        # the same code path either way.
        npu_utilization_percent=72.0 if backend == "hailo-10h" else None,
        npu_power_watts=2.1 if backend == "hailo-10h" else None,
        hailort_version="HailoRT 5.2.0" if backend == "hailo-10h" else None,
    )


@pytest.fixture
def aggregator_with_two_backends() -> ResultsAggregator:
    """Aggregator carrying one Ollama-CPU and one Hailo-10H run for the
    same (model_name, model_size, prompt_id) tuple. Phase 7 group_by
    must keep them separate.
    """
    agg = ResultsAggregator()
    sys_info = _make_system_info()
    agg.raw_llm_results.append((sys_info, _make_llm_result(backend="ollama-cpu", tps=8.0)))
    agg.raw_llm_results.append((sys_info, _make_llm_result(backend="hailo-10h", tps=18.5)))
    return agg


# ----- aggregator-level invariants -----------------------------------------


def test_aggregator_groups_by_backend(aggregator_with_two_backends):
    metrics = aggregator_with_two_backends.aggregate_llm_results()
    # Two backends => two rows, even though model_name/prompt_id match.
    assert len(metrics) == 2
    backends = {m.backend for m in metrics}
    assert backends == {"ollama-cpu", "hailo-10h"}


def test_aggregated_metrics_carry_npu_fields(aggregator_with_two_backends):
    metrics = aggregator_with_two_backends.aggregate_llm_results()
    by_backend = {m.backend: m for m in metrics}
    # NPU-side fields populate only for the hailo-10h row.
    npu = by_backend["hailo-10h"]
    cpu = by_backend["ollama-cpu"]
    assert npu.npu_utilization_percent_mean == pytest.approx(72.0)
    assert npu.npu_power_watts_mean == pytest.approx(2.1)
    assert npu.hailort_version == "HailoRT 5.2.0"
    assert cpu.npu_utilization_percent_mean is None
    assert cpu.npu_power_watts_mean is None
    assert cpu.hailort_version is None


def test_aggregated_metrics_to_dict_emits_backend(aggregator_with_two_backends):
    metrics = aggregator_with_two_backends.aggregate_llm_results()
    payloads = [m.to_dict() for m in metrics]
    backends_in_json = {p["backend"] for p in payloads}
    assert backends_in_json == {"ollama-cpu", "hailo-10h"}
    # JSON must also carry the NPU keys (None on the CPU row, populated
    # on the NPU row) so the dashboard JS can split confidently.
    for p in payloads:
        assert "npu_utilization_percent_mean" in p
        assert "npu_power_watts_mean" in p
        assert "hailort_version" in p


# ----- dashboard render invariants -----------------------------------------


def test_dashboard_renders_backend_filter(
    tmp_path: Path, aggregator_with_two_backends
):
    out = tmp_path / "dashboard.html"
    DashboardGenerator(aggregator_with_two_backends).generate(out)
    html = out.read_text()
    # Filter chip must be present with both option values.
    assert 'id="filter-backend"' in html
    assert 'value="ollama-cpu"' in html
    assert 'value="hailo-10h"' in html


def test_dashboard_llm_table_has_backend_column(
    tmp_path: Path, aggregator_with_two_backends
):
    out = tmp_path / "dashboard.html"
    DashboardGenerator(aggregator_with_two_backends).generate(out)
    html = out.read_text()
    # The LLM raw-data section must include a "Backend" column header,
    # and the actual backend strings must show up in the body.
    assert "<th>Backend</th>" in html
    assert "ollama-cpu" in html
    assert "hailo-10h" in html


def test_dashboard_apply_filters_reads_backend(
    tmp_path: Path, aggregator_with_two_backends
):
    out = tmp_path / "dashboard.html"
    DashboardGenerator(aggregator_with_two_backends).generate(out)
    html = out.read_text()
    # JS should both read filter-backend and branch on its value (legacy
    # vs explicit backend label) — pin both contracts.
    assert "filter-backend" in html
    assert re.search(r"d\.backend\s*===\s*backend", html), (
        "applyFilters should compare d.backend === backend for explicit values"
    )
    assert "!d.backend" in html, (
        "applyFilters should treat 'legacy' option as backend-less rows"
    )


def test_dashboard_legacy_option_only_when_legacy_data_present(tmp_path: Path):
    """When every LLM row has an explicit backend, the 'legacy' option
    should NOT appear — the dropdown only surfaces it when there are
    actually pre-Phase-7 rows.
    """
    agg = ResultsAggregator()
    sys_info = _make_system_info()
    agg.raw_llm_results.append((sys_info, _make_llm_result(backend="ollama-cpu")))
    agg.raw_llm_results.append((sys_info, _make_llm_result(backend="hailo-10h")))
    out = tmp_path / "dashboard.html"
    DashboardGenerator(agg).generate(out)
    html = out.read_text()
    assert 'value="legacy"' not in html


def test_dashboard_legacy_option_appears_for_pre_phase7_data(tmp_path: Path):
    """Conversely, when at least one LLM row has backend=None (data
    written before Phase 7's tagging landed), the dropdown surfaces a
    'Legacy' bucket so the operator can isolate it.
    """
    agg = ResultsAggregator()
    sys_info = _make_system_info()
    agg.raw_llm_results.append((sys_info, _make_llm_result(backend=None)))
    agg.raw_llm_results.append((sys_info, _make_llm_result(backend="hailo-10h")))
    out = tmp_path / "dashboard.html"
    DashboardGenerator(agg).generate(out)
    html = out.read_text()
    assert 'value="legacy"' in html


def test_dashboard_friendly_backend_labels(
    tmp_path: Path, aggregator_with_two_backends
):
    """The dropdown should show human-friendly labels for known backends
    rather than the raw schema strings.
    """
    out = tmp_path / "dashboard.html"
    DashboardGenerator(aggregator_with_two_backends).generate(out)
    html = out.read_text()
    assert "Ollama (CPU)" in html
    assert "Hailo-10H (NPU)" in html
