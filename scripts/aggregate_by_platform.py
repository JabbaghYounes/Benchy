#!/usr/bin/env python3
"""Platform-aware aggregation across all bench_*.json files.

The default ResultsAggregator groups YOLO by (model, task) and LLM by
(model, prompt) — *not* by platform — so the same model run on both Pis
gets merged into one row with a meaningless std. This script does the
right thing: groups by platform too, so AI HAT+ and AI HAT+ 2 get
separate rows for the same model.

Outputs land under ``docs/showcase/`` (CSVs consumed by the chart
generator and embedded in ``docs/showcase.md``):

  docs/showcase/yolo_by_platform.csv
  docs/showcase/llm_by_platform.csv
  docs/showcase/system_info_by_platform.csv

Run from anywhere — paths are derived from the script's location, not
the current working directory.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "docs" / "showcase"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect():
    """Walk every bench_*.json. Return:
       yolo: list of (platform, system_info, yolo_result_dict)
       llm:  list of (platform, system_info, llm_result_dict)
       system_info_by_platform: most-recent SystemInfo per platform
    """
    yolo, llm = [], []
    sysinfo_by_platform: dict[str, dict] = {}
    bench_files = sorted(RESULTS_ROOT.rglob("bench_*.json"))

    for bf in bench_files:
        try:
            d = json.load(open(bf))
        except Exception:
            continue
        si = d.get("system_info", {})
        plat = si.get("platform", "unknown")
        # Prefer the bundle with non-trivial system_info (some early
        # bundles have placeholders); pick by mtime as tiebreaker.
        existing = sysinfo_by_platform.get(plat)
        if (existing is None
            or (existing.get("ram_size_gb") in (None, 0, 0.0, 4.0)
                and si.get("ram_size_gb", 0) > 4.0)
            or si.get("accelerator") not in (None, "None")):
            # Hold the most informative system_info we've seen
            if existing is None or si.get("accelerator") not in (None, "None"):
                sysinfo_by_platform[plat] = si

        for r in d.get("yolo_results", []):
            yolo.append((plat, si, r))
        for r in d.get("llm_results", []):
            llm.append((plat, si, r))

    return yolo, llm, sysinfo_by_platform


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None, None
    return (statistics.mean(xs),
            statistics.stdev(xs) if len(xs) > 1 else 0.0,
            min(xs), max(xs))


def aggregate_yolo(yolo_records):
    groups = defaultdict(list)
    for plat, si, r in yolo_records:
        # Filter out the unsupported-on-this-hardware stub rows
        if r.get("model_name") == "unsupported-on-this-hardware":
            continue
        key = (plat, r["model_name"], r.get("task", ""))
        groups[key].append(r)

    rows = []
    for (plat, model, task), rs in sorted(groups.items()):
        # latency is nested under .latency.mean_ms in the per-run JSON
        lats = [r.get("latency", {}).get("mean_ms") for r in rs]
        fpses = [r.get("throughput_fps") for r in rs]
        cpu = [r.get("resource_utilization", {}).get("cpu_percent") for r in rs]
        mem = [r.get("resource_utilization", {}).get("memory_used_mb") for r in rs]

        l_mean, l_std, l_min, l_max = mean_std(lats)
        f_mean, f_std, f_min, f_max = mean_std(fpses)
        c_mean, _, _, _ = mean_std(cpu)
        m_mean, _, _, _ = mean_std(mem)

        rows.append({
            "platform": plat,
            "model_name": model,
            "task": task,
            "num_runs": len(rs),
            "throughput_mean_fps": f_mean,
            "throughput_std_fps": f_std,
            "throughput_min_fps": f_min,
            "throughput_max_fps": f_max,
            "latency_mean_ms": l_mean,
            "latency_std_ms": l_std,
            "latency_min_ms": l_min,
            "latency_max_ms": l_max,
            "cpu_percent_mean": c_mean,
            "memory_used_mb_mean": m_mean,
        })
    return rows


def aggregate_llm(llm_records):
    groups = defaultdict(list)
    for plat, si, r in llm_records:
        # Filter out unsupported-on-this-hardware stubs (zero-valued)
        if r.get("prompt_id") == "unsupported-on-this-hardware":
            continue
        key = (plat,
               r.get("model_name", "?"),
               r.get("backend", "?"),
               r.get("prompt_id", "?"))
        groups[key].append(r)

    rows = []
    for (plat, model, backend, prompt), rs in sorted(groups.items()):
        # tps_mean / ttft_mean_ms are the per-run averages across the
        # measured iterations inside a single bench file. Aggregating
        # those across multiple bench files gives mean-of-means, which
        # is what we want for the cross-run showcase.
        tps = [r.get("tps_mean") for r in rs]
        ttft = [r.get("ttft_mean_ms") for r in rs]
        out_tokens = [r.get("output_tokens") for r in rs]

        t_mean, t_std, _, _ = mean_std(tps)
        f_mean, f_std, _, _ = mean_std(ttft)
        ot_mean, _, _, _ = mean_std(out_tokens)

        rows.append({
            "platform": plat,
            "model_name": model,
            "backend": backend,
            "prompt_id": prompt,
            "num_runs": len(rs),
            "tps_mean": t_mean,
            "tps_std": t_std,
            "ttft_mean_ms": f_mean,
            "ttft_std_ms": f_std,
            "output_tokens_mean": ot_mean,
        })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    yolo_recs, llm_recs, sysinfo = collect()
    print(f"Collected: {len(yolo_recs)} YOLO records, {len(llm_recs)} LLM records, "
          f"{len(sysinfo)} platforms")

    yolo_agg = aggregate_yolo(yolo_recs)
    write_csv(OUT_DIR / "yolo_by_platform.csv", yolo_agg, [
        "platform", "model_name", "task", "num_runs",
        "throughput_mean_fps", "throughput_std_fps",
        "throughput_min_fps", "throughput_max_fps",
        "latency_mean_ms", "latency_std_ms",
        "latency_min_ms", "latency_max_ms",
        "cpu_percent_mean", "memory_used_mb_mean",
    ])

    llm_agg = aggregate_llm(llm_recs)
    write_csv(OUT_DIR / "llm_by_platform.csv", llm_agg, [
        "platform", "model_name", "backend", "prompt_id", "num_runs",
        "tps_mean", "tps_std",
        "ttft_mean_ms", "ttft_std_ms",
        "output_tokens_mean",
    ])

    # System info per platform — pick the one with most informative fields
    si_path = OUT_DIR / "system_info_by_platform.csv"
    with open(si_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "platform", "cpu_model", "accelerator", "ram_size_gb",
            "cooling_config", "power_mode", "os_version", "kernel_version",
        ])
        w.writeheader()
        for plat, si in sorted(sysinfo.items()):
            w.writerow({k: si.get(k) for k in w.fieldnames})
    print(f"  wrote {si_path}  ({len(sysinfo)} platforms)")


if __name__ == "__main__":
    main()
