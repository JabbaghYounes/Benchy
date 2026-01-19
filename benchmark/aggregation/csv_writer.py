# CSV writers for aggregated benchmark results
import csv
import logging
from pathlib import Path

from benchmark.aggregation.aggregator import (
    YOLOAggregatedMetrics,
    LLMAggregatedMetrics,
    PlatformSummary,
    ResultsAggregator,
)

logger = logging.getLogger(__name__)


# Column definitions for aggregated YOLO results
YOLO_AGGREGATED_COLUMNS = [
    "model_name",
    "yolo_version",
    "task",
    "input_resolution",
    "num_runs",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_min_ms",
    "latency_max_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "throughput_mean_fps",
    "throughput_std_fps",
    "throughput_min_fps",
    "throughput_max_fps",
    "map_mean",
    "map_std",
    "precision_mean",
    "recall_mean",
    "cpu_percent_mean",
    "accelerator_percent_mean",
    "memory_used_mb_mean",
    "power_watts_mean",
]

# Column definitions for aggregated LLM results
LLM_AGGREGATED_COLUMNS = [
    "model_name",
    "model_size",
    "prompt_id",
    "num_runs",
    "ttft_mean_ms",
    "ttft_std_ms",
    "ttft_min_ms",
    "ttft_max_ms",
    "tps_mean",
    "tps_std",
    "tps_min",
    "tps_max",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_min_ms",
    "latency_max_ms",
    "prompt_tokens_mean",
    "output_tokens_mean",
    "cpu_percent_mean",
    "accelerator_percent_mean",
    "memory_used_mb_mean",
    "power_watts_mean",
]

# Column definitions for platform summary
PLATFORM_SUMMARY_COLUMNS = [
    "platform",
    "cpu_model",
    "accelerator",
    "ram_size_gb",
    "os_version",
    "num_benchmark_runs",
    "num_yolo_results",
    "num_llm_results",
    "yolo_avg_fps",
    "yolo_avg_latency_ms",
    "llm_avg_tps",
    "llm_avg_ttft_ms",
]


class AggregatedCSVWriter:
    """Writes aggregated benchmark results to CSV files."""

    def write_yolo_aggregated(
        self,
        metrics: list[YOLOAggregatedMetrics],
        path: Path,
    ) -> None:
        """Write aggregated YOLO metrics to CSV.

        Args:
            metrics: List of aggregated YOLO metrics
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=YOLO_AGGREGATED_COLUMNS)
            writer.writeheader()

            for metric in metrics:
                writer.writerow(metric.to_dict())

        logger.info(f"YOLO aggregated CSV written to: {path}")

    def write_llm_aggregated(
        self,
        metrics: list[LLMAggregatedMetrics],
        path: Path,
    ) -> None:
        """Write aggregated LLM metrics to CSV.

        Args:
            metrics: List of aggregated LLM metrics
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LLM_AGGREGATED_COLUMNS)
            writer.writeheader()

            for metric in metrics:
                writer.writerow(metric.to_dict())

        logger.info(f"LLM aggregated CSV written to: {path}")

    def write_platform_summary(
        self,
        summaries: list[PlatformSummary],
        path: Path,
    ) -> None:
        """Write platform summaries to CSV.

        Args:
            summaries: List of platform summaries
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PLATFORM_SUMMARY_COLUMNS)
            writer.writeheader()

            for summary in summaries:
                writer.writerow(summary.to_dict())

        logger.info(f"Platform summary CSV written to: {path}")

    def write_all(
        self,
        aggregator: ResultsAggregator,
        output_dir: Path,
        prefix: str = "aggregated",
    ) -> dict[str, Path]:
        """Write all aggregated results to CSV files.

        Args:
            aggregator: ResultsAggregator with loaded data
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Dictionary mapping result type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Write YOLO aggregated results
        yolo_metrics = aggregator.aggregate_yolo_results()
        if yolo_metrics:
            yolo_path = output_dir / f"{prefix}_yolo.csv"
            self.write_yolo_aggregated(yolo_metrics, yolo_path)
            paths["yolo"] = yolo_path

        # Write LLM aggregated results
        llm_metrics = aggregator.aggregate_llm_results()
        if llm_metrics:
            llm_path = output_dir / f"{prefix}_llm.csv"
            self.write_llm_aggregated(llm_metrics, llm_path)
            paths["llm"] = llm_path

        # Write platform summaries
        platform_summaries = aggregator.get_platform_summaries()
        if platform_summaries:
            platform_path = output_dir / f"{prefix}_platforms.csv"
            self.write_platform_summary(platform_summaries, platform_path)
            paths["platforms"] = platform_path

        return paths


def write_aggregated_csvs(
    input_dir: Path,
    output_dir: Path,
    prefix: str = "aggregated",
) -> dict[str, Path]:
    """Convenience function to aggregate and write all CSVs.

    Args:
        input_dir: Directory containing raw result JSON files
        output_dir: Directory to write CSV outputs
        prefix: Filename prefix

    Returns:
        Dictionary mapping result type to file path
    """
    aggregator = ResultsAggregator()
    aggregator.load_directory(input_dir)

    writer = AggregatedCSVWriter()
    return writer.write_all(aggregator, output_dir, prefix)
