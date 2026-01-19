# Result writers for JSON and CSV output formats
import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from benchmark.schemas import (
    BenchmarkRun,
    YOLOResult,
    LLMResult,
    AggregatedResults,
    SystemInfo,
    LatencyMetrics,
    ResourceUtilization,
)


class ResultWriter(ABC):
    """Abstract base class for result writers."""

    @abstractmethod
    def write(self, data: Union[BenchmarkRun, AggregatedResults], path: Path) -> None:
        """Write benchmark data to file."""
        pass

    @abstractmethod
    def write_yolo_results(self, results: list[YOLOResult], system_info: SystemInfo, path: Path) -> None:
        """Write YOLO results to file."""
        pass

    @abstractmethod
    def write_llm_results(self, results: list[LLMResult], system_info: SystemInfo, path: Path) -> None:
        """Write LLM results to file."""
        pass


class JSONResultWriter(ResultWriter):
    """Writes benchmark results to JSON format."""

    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def write(self, data: Union[BenchmarkRun, AggregatedResults], path: Path) -> None:
        """Write complete benchmark run or aggregated results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data.to_dict(), f, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def write_yolo_results(self, results: list[YOLOResult], system_info: SystemInfo, path: Path) -> None:
        """Write YOLO benchmark results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "system_info": system_info.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def write_llm_results(self, results: list[LLMResult], system_info: SystemInfo, path: Path) -> None:
        """Write LLM benchmark results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "system_info": system_info.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def append_result(self, result: Union[YOLOResult, LLMResult], path: Path) -> None:
        """Append a single result to an existing JSON file."""
        path = Path(path)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"results": []}

        data["results"].append(result.to_dict())

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=self.ensure_ascii)


class CSVResultWriter(ResultWriter):
    """Writes benchmark results to CSV format."""

    # Column definitions for YOLO results
    YOLO_COLUMNS = [
        "platform", "cpu_model", "accelerator", "ram_size_gb",
        "model_name", "yolo_version", "task", "input_resolution",
        "latency_first_ms", "latency_mean_ms", "latency_std_ms",
        "latency_min_ms", "latency_max_ms", "latency_p50_ms", "latency_p95_ms",
        "throughput_fps", "map_score", "precision", "recall",
        "cpu_percent", "accelerator_percent", "memory_used_mb",
        "power_idle_watts", "power_inference_watts",
        "warmup_runs", "measured_runs",
    ]

    # Column definitions for LLM results
    LLM_COLUMNS = [
        "platform", "cpu_model", "accelerator", "ram_size_gb",
        "model_name", "model_size", "quantization",
        "prompt_tokens", "output_tokens",
        "ttft_ms", "tokens_per_second", "total_latency_ms",
        "ttft_mean_ms", "ttft_std_ms", "tps_mean", "tps_std",
        "latency_mean_ms", "latency_std_ms",
        "cpu_percent", "accelerator_percent", "memory_used_mb",
        "power_watts", "warmup_runs", "measured_runs",
    ]

    def write(self, data: Union[BenchmarkRun, AggregatedResults], path: Path) -> None:
        """Write complete benchmark run to CSV files (one for YOLO, one for LLM)."""
        path = Path(path)
        base_path = path.parent / path.stem

        if isinstance(data, BenchmarkRun):
            if data.yolo_results:
                self.write_yolo_results(
                    data.yolo_results,
                    data.system_info,
                    Path(f"{base_path}_yolo.csv"),
                )
            if data.llm_results:
                self.write_llm_results(
                    data.llm_results,
                    data.system_info,
                    Path(f"{base_path}_llm.csv"),
                )

    def write_yolo_results(self, results: list[YOLOResult], system_info: SystemInfo, path: Path) -> None:
        """Write YOLO benchmark results to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.YOLO_COLUMNS)
            writer.writeheader()

            for result in results:
                row = self._yolo_result_to_row(result, system_info)
                writer.writerow(row)

    def write_llm_results(self, results: list[LLMResult], system_info: SystemInfo, path: Path) -> None:
        """Write LLM benchmark results to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.LLM_COLUMNS)
            writer.writeheader()

            for result in results:
                row = self._llm_result_to_row(result, system_info)
                writer.writerow(row)

    def append_yolo_result(self, result: YOLOResult, system_info: SystemInfo, path: Path) -> None:
        """Append a single YOLO result to CSV file."""
        path = Path(path)
        file_exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.YOLO_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(self._yolo_result_to_row(result, system_info))

    def append_llm_result(self, result: LLMResult, system_info: SystemInfo, path: Path) -> None:
        """Append a single LLM result to CSV file."""
        path = Path(path)
        file_exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.LLM_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(self._llm_result_to_row(result, system_info))

    def _yolo_result_to_row(self, result: YOLOResult, system_info: SystemInfo) -> dict:
        """Convert YOLO result to CSV row dictionary."""
        ru = result.resource_utilization
        return {
            "platform": system_info.platform,
            "cpu_model": system_info.cpu_model,
            "accelerator": system_info.accelerator,
            "ram_size_gb": system_info.ram_size_gb,
            "model_name": result.model_name,
            "yolo_version": result.yolo_version,
            "task": result.task,
            "input_resolution": result.input_resolution,
            "latency_first_ms": result.latency.first_inference_ms,
            "latency_mean_ms": result.latency.mean_ms,
            "latency_std_ms": result.latency.std_ms,
            "latency_min_ms": result.latency.min_ms,
            "latency_max_ms": result.latency.max_ms,
            "latency_p50_ms": result.latency.p50_ms,
            "latency_p95_ms": result.latency.p95_ms,
            "throughput_fps": result.throughput_fps,
            "map_score": result.map_score,
            "precision": result.precision,
            "recall": result.recall,
            "cpu_percent": ru.cpu_percent if ru else None,
            "accelerator_percent": ru.accelerator_percent if ru else None,
            "memory_used_mb": ru.memory_used_mb if ru else None,
            "power_idle_watts": result.power_idle_watts,
            "power_inference_watts": result.power_inference_watts,
            "warmup_runs": result.warmup_runs,
            "measured_runs": result.measured_runs,
        }

    def _llm_result_to_row(self, result: LLMResult, system_info: SystemInfo) -> dict:
        """Convert LLM result to CSV row dictionary."""
        ru = result.resource_utilization
        return {
            "platform": system_info.platform,
            "cpu_model": system_info.cpu_model,
            "accelerator": system_info.accelerator,
            "ram_size_gb": system_info.ram_size_gb,
            "model_name": result.model_name,
            "model_size": result.model_size,
            "quantization": result.quantization,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "ttft_ms": result.time_to_first_token_ms,
            "tokens_per_second": result.tokens_per_second,
            "total_latency_ms": result.total_latency_ms,
            "ttft_mean_ms": result.ttft_mean_ms,
            "ttft_std_ms": result.ttft_std_ms,
            "tps_mean": result.tps_mean,
            "tps_std": result.tps_std,
            "latency_mean_ms": result.latency_mean_ms,
            "latency_std_ms": result.latency_std_ms,
            "cpu_percent": ru.cpu_percent if ru else None,
            "accelerator_percent": ru.accelerator_percent if ru else None,
            "memory_used_mb": ru.memory_used_mb if ru else None,
            "power_watts": result.power_watts,
            "warmup_runs": result.warmup_runs,
            "measured_runs": result.measured_runs,
        }


class ResultWriterFactory:
    """Factory for creating result writers."""

    _writers = {
        "json": JSONResultWriter,
        "csv": CSVResultWriter,
    }

    @classmethod
    def create(cls, format: str, **kwargs) -> ResultWriter:
        """Create a result writer for the specified format.

        Args:
            format: Output format ('json' or 'csv')
            **kwargs: Additional arguments passed to the writer constructor

        Returns:
            ResultWriter instance

        Raises:
            ValueError: If format is not supported
        """
        format = format.lower()
        if format not in cls._writers:
            raise ValueError(f"Unsupported format: {format}. Supported: {list(cls._writers.keys())}")
        return cls._writers[format](**kwargs)

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Return list of supported output formats."""
        return list(cls._writers.keys())

    @classmethod
    def register(cls, format: str, writer_class: type) -> None:
        """Register a custom writer class for a format."""
        cls._writers[format.lower()] = writer_class
