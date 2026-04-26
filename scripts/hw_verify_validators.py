#!/usr/bin/env python3
"""Validate the JSON output of a benchmark run against per-task contracts.

The hardware-verification bash runner (`hw_verify_common.sh`) calls this
helper after each `python -m benchmark run …` command to demote
green-exit-but-empty-result outcomes to FAIL. Bash + jq alone could check
"file exists, parses as JSON" but couldn't easily assert structural
contracts (e.g. "every YOLOResult has backend == hailo and
throughput_fps > 0" or "every LLMResult on the npu profile has
hailort_version populated").

CLI shape:
    python scripts/hw_verify_validators.py PATH \\
        --workload {yolo,llm} \\
        [--task TASK]                 # YOLO only
        [--backend BACKEND]           # expected `backend` field
        [--require-npu-metrics]       # LLM-on-NPU runs must have hailort_version

PATH may be a `bench_*.json` file directly or a directory containing one
(in which case the most recent matching file is picked — that's how the
runner finds the freshly-written result).

Exit codes:
    0  contracts met
    1  validation failures (printed to stderr, one per line)
    2  argument or I/O errors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def find_latest_bench_json(directory: Path) -> Optional[Path]:
    """Return the most recently-modified bench_*.json in a directory."""
    candidates = list(directory.glob("bench_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def validate_yolo(
    data: dict, task: Optional[str], backend: Optional[str]
) -> List[str]:
    """Return a list of contract failures for a YOLO benchmark JSON."""
    errors: List[str] = []
    results = data.get("yolo_results", [])
    if not results:
        errors.append("yolo_results is empty — the runner produced no rows")
        return errors

    for i, r in enumerate(results):
        prefix = f"yolo_results[{i}] ({r.get('model_name', '?')}):"
        if backend and r.get("backend") != backend:
            errors.append(
                f"{prefix} backend={r.get('backend')!r}, expected {backend!r}"
            )
        if task and r.get("task") != task:
            errors.append(
                f"{prefix} task={r.get('task')!r}, expected {task!r}"
            )
        fps = r.get("throughput_fps")
        if fps is None or fps <= 0:
            errors.append(
                f"{prefix} throughput_fps={fps!r}, expected > 0"
            )
        latency = r.get("latency") or {}
        mean_ms = latency.get("mean_ms")
        if mean_ms is None or mean_ms <= 0:
            errors.append(
                f"{prefix} latency.mean_ms={mean_ms!r}, expected > 0"
            )
    return errors


def validate_llm(
    data: dict, backend: Optional[str], require_npu_metrics: bool
) -> List[str]:
    """Return a list of contract failures for an LLM benchmark JSON."""
    errors: List[str] = []
    results = data.get("llm_results", [])
    if not results:
        errors.append("llm_results is empty — the runner produced no rows")
        return errors

    for i, r in enumerate(results):
        prefix = f"llm_results[{i}] ({r.get('model_name', '?')}):"
        if backend and r.get("backend") != backend:
            errors.append(
                f"{prefix} backend={r.get('backend')!r}, expected {backend!r}"
            )
        tps = r.get("tokens_per_second")
        if tps is None or tps <= 0:
            errors.append(
                f"{prefix} tokens_per_second={tps!r}, expected > 0"
            )
        if require_npu_metrics and not r.get("hailort_version"):
            # NPU-side power may legitimately be None on Pis whose
            # /sys/class/hwmon doesn't expose AI HAT+ power, so we don't
            # assert npu_power_watts. hailort_version, however, is
            # captured from `hailortcli --version` and should always
            # populate on any host where HailoRT is installed.
            errors.append(
                f"{prefix} hailort_version not populated; "
                f"expected NPU run via hailo-ollama"
            )
    return errors


def main(argv: Optional[List[str]] = None) -> int:
    description = (__doc__ or "").split("\n\n", 1)[0]
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="bench_*.json file or directory containing the latest one",
    )
    parser.add_argument(
        "--workload",
        choices=["yolo", "llm"],
        required=True,
        help="Which result list to validate",
    )
    parser.add_argument(
        "--task",
        help="Expected YOLO task value (yolo workload only)",
    )
    parser.add_argument(
        "--backend",
        help="Expected `backend` field value on every result",
    )
    parser.add_argument(
        "--require-npu-metrics",
        action="store_true",
        help="Demand hailort_version populated (LLM-on-NPU runs)",
    )
    args = parser.parse_args(argv)

    path = args.path
    if path.is_dir():
        latest = find_latest_bench_json(path)
        if latest is None:
            print(
                f"No bench_*.json found in {path}", file=sys.stderr
            )
            return 2
        path = latest

    if not path.exists():
        print(f"File does not exist: {path}", file=sys.stderr)
        return 2

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to parse {path}: {e}", file=sys.stderr)
        return 2

    if args.workload == "yolo":
        errors = validate_yolo(data, args.task, args.backend)
    else:
        errors = validate_llm(data, args.backend, args.require_npu_metrics)

    if errors:
        print(f"Validation FAILED for {path}:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    n_results = len(data.get(f"{args.workload}_results", []))
    print(f"OK: {path} ({n_results} {args.workload} result row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
