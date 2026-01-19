# Results aggregation for benchmark runs
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.schemas import (
    BenchmarkRun,
    YOLOResult,
    LLMResult,
    AggregatedResults,
    SystemInfo,
    LatencyMetrics,
    ResourceUtilization,
)

logger = logging.getLogger(__name__)


@dataclass
class YOLOAggregatedMetrics:
    """Aggregated metrics for a YOLO model across multiple runs."""

    model_name: str
    yolo_version: str
    task: str
    input_resolution: str
    num_runs: int = 0
    # Latency aggregates
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    # Throughput aggregates
    throughput_mean_fps: float = 0.0
    throughput_std_fps: float = 0.0
    throughput_min_fps: float = 0.0
    throughput_max_fps: float = 0.0
    # Accuracy aggregates
    map_mean: Optional[float] = None
    map_std: Optional[float] = None
    precision_mean: Optional[float] = None
    recall_mean: Optional[float] = None
    # Resource utilization aggregates
    cpu_percent_mean: Optional[float] = None
    accelerator_percent_mean: Optional[float] = None
    memory_used_mb_mean: Optional[float] = None
    power_watts_mean: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "yolo_version": self.yolo_version,
            "task": self.task,
            "input_resolution": self.input_resolution,
            "num_runs": self.num_runs,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "throughput_mean_fps": self.throughput_mean_fps,
            "throughput_std_fps": self.throughput_std_fps,
            "throughput_min_fps": self.throughput_min_fps,
            "throughput_max_fps": self.throughput_max_fps,
            "map_mean": self.map_mean,
            "map_std": self.map_std,
            "precision_mean": self.precision_mean,
            "recall_mean": self.recall_mean,
            "cpu_percent_mean": self.cpu_percent_mean,
            "accelerator_percent_mean": self.accelerator_percent_mean,
            "memory_used_mb_mean": self.memory_used_mb_mean,
            "power_watts_mean": self.power_watts_mean,
        }


@dataclass
class LLMAggregatedMetrics:
    """Aggregated metrics for an LLM model across multiple runs."""

    model_name: str
    model_size: str
    prompt_id: str
    num_runs: int = 0
    # TTFT aggregates
    ttft_mean_ms: float = 0.0
    ttft_std_ms: float = 0.0
    ttft_min_ms: float = 0.0
    ttft_max_ms: float = 0.0
    # Tokens per second aggregates
    tps_mean: float = 0.0
    tps_std: float = 0.0
    tps_min: float = 0.0
    tps_max: float = 0.0
    # Latency aggregates
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    # Token counts
    prompt_tokens_mean: float = 0.0
    output_tokens_mean: float = 0.0
    # Resource utilization aggregates
    cpu_percent_mean: Optional[float] = None
    accelerator_percent_mean: Optional[float] = None
    memory_used_mb_mean: Optional[float] = None
    power_watts_mean: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "prompt_id": self.prompt_id,
            "num_runs": self.num_runs,
            "ttft_mean_ms": self.ttft_mean_ms,
            "ttft_std_ms": self.ttft_std_ms,
            "ttft_min_ms": self.ttft_min_ms,
            "ttft_max_ms": self.ttft_max_ms,
            "tps_mean": self.tps_mean,
            "tps_std": self.tps_std,
            "tps_min": self.tps_min,
            "tps_max": self.tps_max,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "prompt_tokens_mean": self.prompt_tokens_mean,
            "output_tokens_mean": self.output_tokens_mean,
            "cpu_percent_mean": self.cpu_percent_mean,
            "accelerator_percent_mean": self.accelerator_percent_mean,
            "memory_used_mb_mean": self.memory_used_mb_mean,
            "power_watts_mean": self.power_watts_mean,
        }


@dataclass
class PlatformSummary:
    """Summary statistics for a platform."""

    platform: str
    num_benchmark_runs: int = 0
    num_yolo_results: int = 0
    num_llm_results: int = 0
    # System info
    cpu_model: str = ""
    accelerator: str = ""
    ram_size_gb: float = 0.0
    os_version: str = ""
    # Overall performance summaries
    yolo_avg_fps: Optional[float] = None
    yolo_avg_latency_ms: Optional[float] = None
    llm_avg_tps: Optional[float] = None
    llm_avg_ttft_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "num_benchmark_runs": self.num_benchmark_runs,
            "num_yolo_results": self.num_yolo_results,
            "num_llm_results": self.num_llm_results,
            "cpu_model": self.cpu_model,
            "accelerator": self.accelerator,
            "ram_size_gb": self.ram_size_gb,
            "os_version": self.os_version,
            "yolo_avg_fps": self.yolo_avg_fps,
            "yolo_avg_latency_ms": self.yolo_avg_latency_ms,
            "llm_avg_tps": self.llm_avg_tps,
            "llm_avg_ttft_ms": self.llm_avg_ttft_ms,
        }


class ResultsAggregator:
    """Aggregates benchmark results from multiple runs."""

    def __init__(self):
        self.benchmark_runs: list[BenchmarkRun] = []
        self.raw_yolo_results: list[tuple[SystemInfo, YOLOResult]] = []
        self.raw_llm_results: list[tuple[SystemInfo, LLMResult]] = []

    def load_json_file(self, path: Path) -> None:
        """Load benchmark results from a JSON file.

        Args:
            path: Path to JSON results file
        """
        logger.info(f"Loading results from: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Check if this is a BenchmarkRun or raw results file
        if "run_id" in data:
            # Full benchmark run
            benchmark_run = self._parse_benchmark_run(data)
            self.benchmark_runs.append(benchmark_run)

            # Extract individual results
            for yolo_result in benchmark_run.yolo_results:
                self.raw_yolo_results.append((benchmark_run.system_info, yolo_result))
            for llm_result in benchmark_run.llm_results:
                self.raw_llm_results.append((benchmark_run.system_info, llm_result))

        elif "system_info" in data and "results" in data:
            # Raw results file (YOLO or LLM)
            system_info = self._parse_system_info(data["system_info"])

            for result_data in data["results"]:
                if "yolo_version" in result_data:
                    yolo_result = self._parse_yolo_result(result_data)
                    self.raw_yolo_results.append((system_info, yolo_result))
                elif "model_size" in result_data:
                    llm_result = self._parse_llm_result(result_data)
                    self.raw_llm_results.append((system_info, llm_result))

    def load_directory(self, directory: Path, pattern: str = "*.json") -> None:
        """Load all JSON result files from a directory.

        Args:
            directory: Directory containing result files
            pattern: Glob pattern for files to load
        """
        for path in directory.glob(pattern):
            if path.name.startswith("example_"):
                continue  # Skip example files
            try:
                self.load_json_file(path)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    def _parse_system_info(self, data: dict) -> SystemInfo:
        """Parse SystemInfo from dict."""
        return SystemInfo(
            platform=data.get("platform", "unknown"),
            cpu_model=data.get("cpu_model", "unknown"),
            accelerator=data.get("accelerator", "unknown"),
            ram_size_gb=data.get("ram_size_gb", 0.0),
            storage_type=data.get("storage_type", "unknown"),
            cooling_config=data.get("cooling_config", "unknown"),
            power_mode=data.get("power_mode", "unknown"),
            os_version=data.get("os_version", "unknown"),
            kernel_version=data.get("kernel_version", "unknown"),
            hostname=data.get("hostname"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

    def _parse_latency_metrics(self, data: dict) -> LatencyMetrics:
        """Parse LatencyMetrics from dict."""
        return LatencyMetrics(
            first_inference_ms=data.get("first_inference_ms", 0.0),
            mean_ms=data.get("mean_ms", 0.0),
            std_ms=data.get("std_ms", 0.0),
            min_ms=data.get("min_ms", 0.0),
            max_ms=data.get("max_ms", 0.0),
            p50_ms=data.get("p50_ms", 0.0),
            p95_ms=data.get("p95_ms", 0.0),
            p99_ms=data.get("p99_ms"),
        )

    def _parse_resource_utilization(self, data: Optional[dict]) -> Optional[ResourceUtilization]:
        """Parse ResourceUtilization from dict."""
        if not data:
            return None
        return ResourceUtilization(
            cpu_percent=data.get("cpu_percent", 0.0),
            accelerator_percent=data.get("accelerator_percent"),
            memory_used_mb=data.get("memory_used_mb", 0.0),
            memory_total_mb=data.get("memory_total_mb", 0.0),
            power_watts=data.get("power_watts"),
        )

    def _parse_yolo_result(self, data: dict) -> YOLOResult:
        """Parse YOLOResult from dict."""
        return YOLOResult(
            model_name=data.get("model_name", ""),
            yolo_version=data.get("yolo_version", ""),
            task=data.get("task", ""),
            input_resolution=data.get("input_resolution", ""),
            latency=self._parse_latency_metrics(data.get("latency", {})),
            throughput_fps=data.get("throughput_fps", 0.0),
            map_score=data.get("map_score"),
            precision=data.get("precision"),
            recall=data.get("recall"),
            resource_utilization=self._parse_resource_utilization(data.get("resource_utilization")),
            power_idle_watts=data.get("power_idle_watts"),
            power_inference_watts=data.get("power_inference_watts"),
            warmup_runs=data.get("warmup_runs", 3),
            measured_runs=data.get("measured_runs", 10),
        )

    def _parse_llm_result(self, data: dict) -> LLMResult:
        """Parse LLMResult from dict."""
        return LLMResult(
            model_name=data.get("model_name", ""),
            model_size=data.get("model_size", ""),
            model_hash=data.get("model_hash"),
            quantization=data.get("quantization"),
            prompt_id=data.get("prompt_id"),
            prompt_tokens=data.get("prompt_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            time_to_first_token_ms=data.get("time_to_first_token_ms", 0.0),
            tokens_per_second=data.get("tokens_per_second", 0.0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            resource_utilization=self._parse_resource_utilization(data.get("resource_utilization")),
            power_watts=data.get("power_watts"),
            warmup_runs=data.get("warmup_runs", 3),
            measured_runs=data.get("measured_runs", 10),
            ttft_mean_ms=data.get("ttft_mean_ms"),
            ttft_std_ms=data.get("ttft_std_ms"),
            tps_mean=data.get("tps_mean"),
            tps_std=data.get("tps_std"),
            latency_mean_ms=data.get("latency_mean_ms"),
            latency_std_ms=data.get("latency_std_ms"),
        )

    def _parse_benchmark_run(self, data: dict) -> BenchmarkRun:
        """Parse BenchmarkRun from dict."""
        system_info = self._parse_system_info(data.get("system_info", {}))
        yolo_results = [self._parse_yolo_result(r) for r in data.get("yolo_results", [])]
        llm_results = [self._parse_llm_result(r) for r in data.get("llm_results", [])]

        return BenchmarkRun(
            run_id=data.get("run_id", ""),
            system_info=system_info,
            workload_type=data.get("workload_type", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            yolo_results=yolo_results,
            llm_results=llm_results,
            metadata=data.get("metadata", {}),
        )

    def aggregate_yolo_results(
        self, group_by: list[str] = None
    ) -> list[YOLOAggregatedMetrics]:
        """Aggregate YOLO results.

        Args:
            group_by: Fields to group by (default: model_name, yolo_version, task)

        Returns:
            List of aggregated metrics
        """
        if group_by is None:
            group_by = ["model_name", "yolo_version", "task"]

        # Group results
        groups: dict[tuple, list[tuple[SystemInfo, YOLOResult]]] = defaultdict(list)
        for system_info, result in self.raw_yolo_results:
            key = tuple(getattr(result, field) for field in group_by)
            groups[key].append((system_info, result))

        # Aggregate each group
        aggregated = []
        for key, results in groups.items():
            metrics = self._aggregate_yolo_group(results, group_by, key)
            aggregated.append(metrics)

        return aggregated

    def _aggregate_yolo_group(
        self,
        results: list[tuple[SystemInfo, YOLOResult]],
        group_by: list[str],
        key: tuple,
    ) -> YOLOAggregatedMetrics:
        """Aggregate a group of YOLO results."""
        first_result = results[0][1]

        # Extract values for aggregation
        latency_means = [r.latency.mean_ms for _, r in results]
        latency_p50s = [r.latency.p50_ms for _, r in results]
        latency_p95s = [r.latency.p95_ms for _, r in results]
        throughputs = [r.throughput_fps for _, r in results]
        maps = [r.map_score for _, r in results if r.map_score is not None]
        precisions = [r.precision for _, r in results if r.precision is not None]
        recalls = [r.recall for _, r in results if r.recall is not None]

        # Resource utilization
        cpu_percents = [r.resource_utilization.cpu_percent for _, r in results
                       if r.resource_utilization]
        accel_percents = [r.resource_utilization.accelerator_percent for _, r in results
                         if r.resource_utilization and r.resource_utilization.accelerator_percent]
        memory_usages = [r.resource_utilization.memory_used_mb for _, r in results
                        if r.resource_utilization]
        power_usages = [r.resource_utilization.power_watts for _, r in results
                       if r.resource_utilization and r.resource_utilization.power_watts]

        return YOLOAggregatedMetrics(
            model_name=first_result.model_name,
            yolo_version=first_result.yolo_version,
            task=first_result.task,
            input_resolution=first_result.input_resolution,
            num_runs=len(results),
            latency_mean_ms=round(statistics.mean(latency_means), 3),
            latency_std_ms=round(statistics.stdev(latency_means), 3) if len(latency_means) > 1 else 0.0,
            latency_min_ms=round(min(latency_means), 3),
            latency_max_ms=round(max(latency_means), 3),
            latency_p50_ms=round(statistics.mean(latency_p50s), 3),
            latency_p95_ms=round(statistics.mean(latency_p95s), 3),
            throughput_mean_fps=round(statistics.mean(throughputs), 2),
            throughput_std_fps=round(statistics.stdev(throughputs), 2) if len(throughputs) > 1 else 0.0,
            throughput_min_fps=round(min(throughputs), 2),
            throughput_max_fps=round(max(throughputs), 2),
            map_mean=round(statistics.mean(maps), 4) if maps else None,
            map_std=round(statistics.stdev(maps), 4) if len(maps) > 1 else None,
            precision_mean=round(statistics.mean(precisions), 4) if precisions else None,
            recall_mean=round(statistics.mean(recalls), 4) if recalls else None,
            cpu_percent_mean=round(statistics.mean(cpu_percents), 1) if cpu_percents else None,
            accelerator_percent_mean=round(statistics.mean(accel_percents), 1) if accel_percents else None,
            memory_used_mb_mean=round(statistics.mean(memory_usages), 1) if memory_usages else None,
            power_watts_mean=round(statistics.mean(power_usages), 2) if power_usages else None,
        )

    def aggregate_llm_results(
        self, group_by: list[str] = None
    ) -> list[LLMAggregatedMetrics]:
        """Aggregate LLM results.

        Args:
            group_by: Fields to group by (default: model_name, model_size, prompt_id)

        Returns:
            List of aggregated metrics
        """
        if group_by is None:
            group_by = ["model_name", "model_size", "prompt_id"]

        # Group results
        groups: dict[tuple, list[tuple[SystemInfo, LLMResult]]] = defaultdict(list)
        for system_info, result in self.raw_llm_results:
            key = tuple(getattr(result, field) for field in group_by)
            groups[key].append((system_info, result))

        # Aggregate each group
        aggregated = []
        for key, results in groups.items():
            metrics = self._aggregate_llm_group(results, group_by, key)
            aggregated.append(metrics)

        return aggregated

    def _aggregate_llm_group(
        self,
        results: list[tuple[SystemInfo, LLMResult]],
        group_by: list[str],
        key: tuple,
    ) -> LLMAggregatedMetrics:
        """Aggregate a group of LLM results."""
        first_result = results[0][1]

        # Extract values - use aggregated stats if available, otherwise raw values
        ttfts = []
        tps_values = []
        latencies = []

        for _, r in results:
            if r.ttft_mean_ms is not None:
                ttfts.append(r.ttft_mean_ms)
            else:
                ttfts.append(r.time_to_first_token_ms)

            if r.tps_mean is not None:
                tps_values.append(r.tps_mean)
            else:
                tps_values.append(r.tokens_per_second)

            if r.latency_mean_ms is not None:
                latencies.append(r.latency_mean_ms)
            else:
                latencies.append(r.total_latency_ms)

        prompt_tokens = [r.prompt_tokens for _, r in results]
        output_tokens = [r.output_tokens for _, r in results]

        # Resource utilization
        cpu_percents = [r.resource_utilization.cpu_percent for _, r in results
                       if r.resource_utilization]
        accel_percents = [r.resource_utilization.accelerator_percent for _, r in results
                         if r.resource_utilization and r.resource_utilization.accelerator_percent]
        memory_usages = [r.resource_utilization.memory_used_mb for _, r in results
                        if r.resource_utilization]
        power_usages = [r.power_watts for _, r in results if r.power_watts]

        return LLMAggregatedMetrics(
            model_name=first_result.model_name,
            model_size=first_result.model_size,
            prompt_id=first_result.prompt_id or "unknown",
            num_runs=len(results),
            ttft_mean_ms=round(statistics.mean(ttfts), 2),
            ttft_std_ms=round(statistics.stdev(ttfts), 2) if len(ttfts) > 1 else 0.0,
            ttft_min_ms=round(min(ttfts), 2),
            ttft_max_ms=round(max(ttfts), 2),
            tps_mean=round(statistics.mean(tps_values), 2),
            tps_std=round(statistics.stdev(tps_values), 2) if len(tps_values) > 1 else 0.0,
            tps_min=round(min(tps_values), 2),
            tps_max=round(max(tps_values), 2),
            latency_mean_ms=round(statistics.mean(latencies), 2),
            latency_std_ms=round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
            latency_min_ms=round(min(latencies), 2),
            latency_max_ms=round(max(latencies), 2),
            prompt_tokens_mean=round(statistics.mean(prompt_tokens), 1),
            output_tokens_mean=round(statistics.mean(output_tokens), 1),
            cpu_percent_mean=round(statistics.mean(cpu_percents), 1) if cpu_percents else None,
            accelerator_percent_mean=round(statistics.mean(accel_percents), 1) if accel_percents else None,
            memory_used_mb_mean=round(statistics.mean(memory_usages), 1) if memory_usages else None,
            power_watts_mean=round(statistics.mean(power_usages), 2) if power_usages else None,
        )

    def get_platform_summaries(self) -> list[PlatformSummary]:
        """Get summary statistics per platform.

        Returns:
            List of PlatformSummary objects
        """
        # Group by platform
        platform_data: dict[str, dict] = defaultdict(lambda: {
            "runs": [],
            "yolo_results": [],
            "llm_results": [],
            "system_info": None,
        })

        for run in self.benchmark_runs:
            platform = run.system_info.platform
            platform_data[platform]["runs"].append(run)
            if platform_data[platform]["system_info"] is None:
                platform_data[platform]["system_info"] = run.system_info

        for system_info, result in self.raw_yolo_results:
            platform_data[system_info.platform]["yolo_results"].append(result)
            if platform_data[system_info.platform]["system_info"] is None:
                platform_data[system_info.platform]["system_info"] = system_info

        for system_info, result in self.raw_llm_results:
            platform_data[system_info.platform]["llm_results"].append(result)
            if platform_data[system_info.platform]["system_info"] is None:
                platform_data[system_info.platform]["system_info"] = system_info

        # Create summaries
        summaries = []
        for platform, data in platform_data.items():
            system_info = data["system_info"]
            yolo_results = data["yolo_results"]
            llm_results = data["llm_results"]

            summary = PlatformSummary(
                platform=platform,
                num_benchmark_runs=len(data["runs"]),
                num_yolo_results=len(yolo_results),
                num_llm_results=len(llm_results),
                cpu_model=system_info.cpu_model if system_info else "",
                accelerator=system_info.accelerator if system_info else "",
                ram_size_gb=system_info.ram_size_gb if system_info else 0.0,
                os_version=system_info.os_version if system_info else "",
            )

            # Calculate averages
            if yolo_results:
                fps_values = [r.throughput_fps for r in yolo_results]
                latency_values = [r.latency.mean_ms for r in yolo_results]
                summary.yolo_avg_fps = round(statistics.mean(fps_values), 2)
                summary.yolo_avg_latency_ms = round(statistics.mean(latency_values), 2)

            if llm_results:
                tps_values = [r.tps_mean or r.tokens_per_second for r in llm_results]
                ttft_values = [r.ttft_mean_ms or r.time_to_first_token_ms for r in llm_results]
                summary.llm_avg_tps = round(statistics.mean(tps_values), 2)
                summary.llm_avg_ttft_ms = round(statistics.mean(ttft_values), 2)

            summaries.append(summary)

        return summaries

    def create_aggregated_results(self, aggregation_id: str = None) -> AggregatedResults:
        """Create complete aggregated results object.

        Args:
            aggregation_id: Optional ID for the aggregation

        Returns:
            AggregatedResults object
        """
        if aggregation_id is None:
            aggregation_id = f"agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        platform_summaries = self.get_platform_summaries()
        yolo_aggregated = self.aggregate_yolo_results()
        llm_aggregated = self.aggregate_llm_results()

        return AggregatedResults(
            aggregation_id=aggregation_id,
            created_at=datetime.now().isoformat(),
            runs=[run.run_id for run in self.benchmark_runs],
            platform_summary={s.platform: s.to_dict() for s in platform_summaries},
            yolo_summary={
                "models": [m.to_dict() for m in yolo_aggregated],
                "total_results": len(self.raw_yolo_results),
            },
            llm_summary={
                "models": [m.to_dict() for m in llm_aggregated],
                "total_results": len(self.raw_llm_results),
            },
        )


def aggregate_results(
    input_dir: Path,
    output_dir: Path,
    aggregation_id: str = None,
) -> AggregatedResults:
    """Convenience function to aggregate all results in a directory.

    Args:
        input_dir: Directory containing raw result JSON files
        output_dir: Directory to write aggregated outputs
        aggregation_id: Optional aggregation ID

    Returns:
        AggregatedResults object
    """
    aggregator = ResultsAggregator()
    aggregator.load_directory(input_dir)

    results = aggregator.create_aggregated_results(aggregation_id)

    # Write aggregated results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = output_dir / f"{results.aggregation_id}.json"
    with open(json_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info(f"Aggregated JSON written to: {json_path}")

    return results
