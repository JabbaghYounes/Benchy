"""Tests for `scripts/hw_verify_validators.py`.

The bash hardware-verification runner (`hw_verify_common.sh:hw_run_step`)
calls this validator after each `python -m benchmark run …` so empty or
malformed JSON gets demoted from green-exit to FAIL. Bash scripts are
hard to unit-test cleanly on a dev laptop without a Pi, so the validator
is the surface we pin here.

The validator imports cleanly as a module — no Hailo SDK or other
hardware deps — which is why we can test the contract logic end-to-end
on any host.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_PATH = REPO_ROOT / "scripts" / "hw_verify_validators.py"


# Load the script as a module — `scripts/` isn't on the regular Python
# import path, but importlib lets us reach it directly.
def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "hw_verify_validators", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def validator():
    return _load_validator()


# ----- Helpers -------------------------------------------------------------


def _yolo_run(
    *,
    backend: str = "hailo",
    task: str = "detection",
    throughput: float = 12.5,
    latency: float = 80.0,
    model_name: str = "yolov8n.pt",
) -> dict:
    """Build a minimal benchmark-run dict with one YOLOResult."""
    return {
        "yolo_results": [
            {
                "model_name": model_name,
                "yolo_version": "v8",
                "task": task,
                "throughput_fps": throughput,
                "backend": backend,
                "latency": {"mean_ms": latency},
            }
        ]
    }


def _llm_run(
    *,
    backend: str = "hailo-10h",
    tps: float = 18.5,
    hailort_version: str | None = "HailoRT 5.2.0",
    model_name: str = "llama3.2:3b",
) -> dict:
    """Build a minimal benchmark-run dict with one LLMResult."""
    return {
        "llm_results": [
            {
                "model_name": model_name,
                "tokens_per_second": tps,
                "backend": backend,
                "hailort_version": hailort_version,
            }
        ]
    }


def _write_bench(tmp_path: Path, payload: dict, name: str = "bench_test.json") -> Path:
    out = tmp_path / name
    out.write_text(json.dumps(payload))
    return out


# ----- validate_yolo --------------------------------------------------------


def test_yolo_passes_clean_run(validator, tmp_path):
    out = _write_bench(tmp_path, _yolo_run())
    errors = validator.validate_yolo(json.loads(out.read_text()), task="detection", backend="hailo")
    assert errors == []


def test_yolo_empty_results_is_failure(validator):
    errors = validator.validate_yolo({"yolo_results": []}, task=None, backend=None)
    assert any("empty" in e for e in errors)


def test_yolo_backend_mismatch(validator):
    payload = _yolo_run(backend="pytorch")
    errors = validator.validate_yolo(payload, task=None, backend="hailo")
    assert any("backend=" in e and "pytorch" in e for e in errors)


def test_yolo_task_mismatch(validator):
    payload = _yolo_run(task="detection")
    errors = validator.validate_yolo(payload, task="obb", backend=None)
    assert any("task=" in e for e in errors)


def test_yolo_zero_throughput_is_failure(validator):
    payload = _yolo_run(throughput=0.0)
    errors = validator.validate_yolo(payload, task=None, backend=None)
    assert any("throughput_fps" in e for e in errors)


def test_yolo_missing_latency_is_failure(validator):
    payload = _yolo_run()
    payload["yolo_results"][0]["latency"] = {}  # no mean_ms
    errors = validator.validate_yolo(payload, task=None, backend=None)
    assert any("latency.mean_ms" in e for e in errors)


# ----- validate_llm ---------------------------------------------------------


def test_llm_passes_clean_npu_run(validator):
    payload = _llm_run()
    errors = validator.validate_llm(payload, backend="hailo-10h", require_npu_metrics=True)
    assert errors == []


def test_llm_passes_clean_cpu_run(validator):
    """A CPU-side run won't have hailort_version; the validator must
    accept that when --require-npu-metrics isn't set.
    """
    payload = _llm_run(backend="ollama-cpu", hailort_version=None)
    errors = validator.validate_llm(payload, backend="ollama-cpu", require_npu_metrics=False)
    assert errors == []


def test_llm_empty_results_is_failure(validator):
    errors = validator.validate_llm(
        {"llm_results": []}, backend=None, require_npu_metrics=False
    )
    assert any("empty" in e for e in errors)


def test_llm_backend_mismatch(validator):
    payload = _llm_run(backend="ollama-cpu")
    errors = validator.validate_llm(payload, backend="hailo-10h", require_npu_metrics=False)
    assert any("backend=" in e for e in errors)


def test_llm_zero_tps_is_failure(validator):
    payload = _llm_run(tps=0.0)
    errors = validator.validate_llm(payload, backend=None, require_npu_metrics=False)
    assert any("tokens_per_second" in e for e in errors)


def test_llm_npu_run_missing_hailort_version_is_failure(validator):
    payload = _llm_run(hailort_version=None)
    errors = validator.validate_llm(payload, backend="hailo-10h", require_npu_metrics=True)
    assert any("hailort_version" in e for e in errors)


def test_llm_cpu_run_missing_hailort_version_is_ok(validator):
    """Symmetric to test_llm_passes_clean_cpu_run — pin both directions
    so a future change can't accidentally tighten the contract.
    """
    payload = _llm_run(hailort_version=None, backend="ollama-cpu")
    errors = validator.validate_llm(payload, backend="ollama-cpu", require_npu_metrics=False)
    assert errors == []


# ----- find_latest_bench_json + main ---------------------------------------


def test_find_latest_bench_json_picks_most_recent(validator, tmp_path):
    import time
    _write_bench(tmp_path, _yolo_run(), name="bench_old.json")
    time.sleep(0.05)  # ensure mtime distinct on coarse-resolution filesystems
    newer = _write_bench(tmp_path, _yolo_run(), name="bench_new.json")
    found = validator.find_latest_bench_json(tmp_path)
    assert found == newer


def test_find_latest_bench_json_returns_none_when_empty(validator, tmp_path):
    assert validator.find_latest_bench_json(tmp_path) is None


def test_main_exit_zero_on_clean_yolo(validator, tmp_path):
    out = _write_bench(tmp_path, _yolo_run())
    rc = validator.main([str(out), "--workload", "yolo", "--backend", "hailo"])
    assert rc == 0


def test_main_exit_one_on_validation_failure(validator, tmp_path, capsys):
    out = _write_bench(tmp_path, _yolo_run(backend="pytorch"))
    rc = validator.main([str(out), "--workload", "yolo", "--backend", "hailo"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Validation FAILED" in captured.err


def test_main_exit_two_on_missing_file(validator, tmp_path):
    rc = validator.main([str(tmp_path / "does_not_exist.json"), "--workload", "yolo"])
    assert rc == 2


def test_main_resolves_directory_to_latest_bench_json(validator, tmp_path):
    """Pointing the validator at a directory (not a file) is the
    runner's primary call shape — `hw_run_step` passes
    "$HW_RESULTS_DIR" without knowing which timestamped file was just
    written.
    """
    _write_bench(tmp_path, _yolo_run())
    rc = validator.main([str(tmp_path), "--workload", "yolo", "--backend", "hailo"])
    assert rc == 0


def test_main_exit_two_on_empty_directory(validator, tmp_path, capsys):
    rc = validator.main([str(tmp_path), "--workload", "yolo"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "No bench_*.json found" in captured.err
