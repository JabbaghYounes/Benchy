#!/usr/bin/env python3
"""Regenerate the cross-platform dashboard + aggregated JSON for the showcase.

The default ``python -m benchmark report --input results/`` uses
non-recursive glob, so it misses everything under
``results/<platform>/hw_verify_<ts>/``. This wrapper loads every
``bench_*.json`` recursively, then drives the same
``DashboardGenerator`` so the showcase dashboard reflects the full
multi-bundle dataset.

Output (overwrites in place):
  docs/showcase/dashboard.html
  docs/showcase/aggregated.json

Pair with:
  scripts/aggregate_by_platform.py     -> per-platform CSVs (showcase tables)
  scripts/generate_showcase_charts.py  -> static PNGs (embedded in showcase.md)

Run from anywhere — paths are derived from the script's location.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.aggregation import ResultsAggregator
from benchmark.reporting import DashboardGenerator


def main() -> int:
    results_root = REPO_ROOT / "results"
    output_dir = REPO_ROOT / "docs" / "showcase"
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_files = sorted(results_root.rglob("bench_*.json"))
    print(f"Found {len(bench_files)} bench_*.json files under {results_root}")

    aggregator = ResultsAggregator()
    loaded = failed = 0
    for path in bench_files:
        try:
            aggregator.load_json_file(path)
            loaded += 1
        except Exception as e:
            print(f"  FAILED {path.relative_to(results_root)}: {e}")
            failed += 1

    print(f"Loaded {loaded}, failed {failed}")
    print(f"  YOLO results: {len(aggregator.raw_yolo_results)}")
    print(f"  LLM results:  {len(aggregator.raw_llm_results)}")
    print(f"  Platforms:    {len(aggregator.get_platform_summaries())}")

    if not aggregator.raw_yolo_results and not aggregator.raw_llm_results:
        print("No results to aggregate.")
        return 1

    # Aggregated JSON
    results = aggregator.create_aggregated_results("showcase")
    json_path = output_dir / "aggregated.json"
    with open(json_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")

    # Dashboard
    dashboard_path = output_dir / "dashboard.html"
    DashboardGenerator(aggregator).generate(
        dashboard_path,
        title="Benchy Edge AI Benchmark Suite — Cross-Platform Showcase",
    )
    print(f"Wrote {dashboard_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
