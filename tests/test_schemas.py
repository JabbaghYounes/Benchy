import json

from benchmark.schemas import (
    BenchmarkRun,
    LatencyMetrics,
    LLMResult,
    ResourceUtilization,
    SystemInfo,
    YOLOResult,
)


def _is_json_serializable(payload: dict) -> bool:
    json.dumps(payload)
    return True


def test_system_info_to_dict_is_json_safe():
    info = SystemInfo(
        platform="jetson_orin_nano",
        cpu_model="ARMv8 Cortex-A78AE",
        accelerator="Ampere GPU",
        ram_size_gb=8.0,
        storage_type="NVMe",
        cooling_config="active",
        power_mode="MAXN",
        os_version="Ubuntu 22.04",
        kernel_version="5.15.0",
    )
    payload = info.to_dict()
    assert payload["platform"] == "jetson_orin_nano"
    assert isinstance(payload["timestamp"], str)
    assert _is_json_serializable(payload)


def test_latency_metrics_round_trip():
    metrics = LatencyMetrics(
        first_inference_ms=12.3,
        mean_ms=8.1,
        std_ms=0.4,
        min_ms=7.5,
        max_ms=9.0,
        p50_ms=8.0,
        p95_ms=8.9,
    )
    payload = metrics.to_dict()
    assert payload["mean_ms"] == 8.1
    assert payload["p99_ms"] is None
    assert _is_json_serializable(payload)


def test_yolo_result_with_nested_objects():
    util = ResourceUtilization(
        cpu_percent=23.0,
        accelerator_percent=88.5,
        memory_used_mb=1024.0,
        memory_total_mb=8192.0,
        power_watts=12.4,
    )
    latency = LatencyMetrics(
        first_inference_ms=15.0,
        mean_ms=10.0,
        std_ms=0.5,
        min_ms=9.5,
        max_ms=11.0,
        p50_ms=10.0,
        p95_ms=10.8,
    )
    result = YOLOResult(
        model_name="yolov8n.pt",
        yolo_version="v8",
        task="detection",
        input_resolution="640x640",
        latency=latency,
        throughput_fps=100.0,
        backend="hailo",
        resource_utilization=util,
    )
    payload = result.to_dict()
    assert payload["backend"] == "hailo"
    assert payload["latency"]["mean_ms"] == 10.0
    assert payload["resource_utilization"]["accelerator_percent"] == 88.5
    assert _is_json_serializable(payload)


def test_yolo_result_handles_missing_resource_utilization():
    latency = LatencyMetrics(
        first_inference_ms=15.0,
        mean_ms=10.0,
        std_ms=0.5,
        min_ms=9.5,
        max_ms=11.0,
        p50_ms=10.0,
        p95_ms=10.8,
    )
    result = YOLOResult(
        model_name="yolov8n.pt",
        yolo_version="v8",
        task="detection",
        input_resolution="640x640",
        latency=latency,
        throughput_fps=100.0,
    )
    payload = result.to_dict()
    assert payload["resource_utilization"] is None
    assert _is_json_serializable(payload)


def test_llm_result_includes_phase_metadata():
    result = LLMResult(
        model_name="llama2:7b",
        model_size="7B",
        parameter_group="7B",
        architecture="dense",
        specialization="general",
        peak_memory_mb=4500.0,
        truncated=False,
        prompt_category="general",
    )
    payload = result.to_dict()
    assert payload["parameter_group"] == "7B"
    assert payload["architecture"] == "dense"
    assert payload["truncated"] is False
    assert _is_json_serializable(payload)


def test_benchmark_run_with_empty_result_lists():
    info = SystemInfo(
        platform="rpi_ai_hat_plus",
        cpu_model="Cortex-A76",
        accelerator="Hailo-8 26 TOPS",
        ram_size_gb=8.0,
        storage_type="NVMe",
        cooling_config="active",
        power_mode="default",
        os_version="Raspberry Pi OS Bookworm",
        kernel_version="6.6.0",
    )
    run = BenchmarkRun(
        run_id="abc123",
        system_info=info,
        workload_type="yolo",
        started_at="2026-04-25T12:00:00",
    )
    payload = run.to_dict()
    assert payload["run_id"] == "abc123"
    assert payload["yolo_results"] == []
    assert payload["llm_results"] == []
    assert _is_json_serializable(payload)
