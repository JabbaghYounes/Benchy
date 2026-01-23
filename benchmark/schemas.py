# Result schemas for Edge AI Benchmark Suite
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from enum import Enum


class Platform(Enum):
    """Supported hardware platforms."""
    JETSON_ORIN_NANO = "jetson_orin_nano"
    RPI_AI_HAT_PLUS = "rpi_ai_hat_plus"
    RPI_AI_HAT_PLUS_2 = "rpi_ai_hat_plus_2"


class WorkloadType(Enum):
    """Benchmark workload types."""
    YOLO = "yolo"
    LLM = "llm"


class YOLOTask(Enum):
    """YOLO task types."""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    OBB = "obb"
    CLASSIFICATION = "classification"


class Backend(Enum):
    """Inference backend types."""
    PYTORCH = "pytorch"
    HAILO = "hailo"


class HailoDevice(Enum):
    """Hailo NPU device types."""
    HAILO8 = "hailo8"
    HAILO8L = "hailo8l"


@dataclass
class SystemInfo:
    """System information recorded for each benchmark run."""
    platform: str
    cpu_model: str
    accelerator: str
    ram_size_gb: float
    storage_type: str
    cooling_config: str
    power_mode: str
    os_version: str
    kernel_version: str
    hostname: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResourceUtilization:
    """Resource utilization metrics."""
    cpu_percent: float
    accelerator_percent: Optional[float] = None
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    power_watts: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LatencyMetrics:
    """Latency statistics for benchmark runs."""
    first_inference_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class YOLOResult:
    """Results from a YOLO benchmark run."""
    model_name: str
    yolo_version: str
    task: str
    input_resolution: str
    latency: LatencyMetrics
    throughput_fps: float
    backend: str = "pytorch"  # Backend used for inference (pytorch, hailo)
    map_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    resource_utilization: Optional[ResourceUtilization] = None
    power_idle_watts: Optional[float] = None
    power_inference_watts: Optional[float] = None
    warmup_runs: int = 3
    measured_runs: int = 10

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "yolo_version": self.yolo_version,
            "task": self.task,
            "input_resolution": self.input_resolution,
            "latency": self.latency.to_dict(),
            "throughput_fps": self.throughput_fps,
            "backend": self.backend,
            "map_score": self.map_score,
            "precision": self.precision,
            "recall": self.recall,
            "resource_utilization": self.resource_utilization.to_dict() if self.resource_utilization else None,
            "power_idle_watts": self.power_idle_watts,
            "power_inference_watts": self.power_inference_watts,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
        }
        return result


@dataclass
class LLMResult:
    """Results from an LLM (Ollama) benchmark run.

    Model Expansion PRD additions:
    - parameter_group: Model size group (1B, 3B, 7B, 8B, 9B)
    - architecture: Model architecture (dense, moe)
    - specialization: Model specialization (general, code)
    - peak_memory_mb: Peak memory usage during inference
    - truncated: Whether output was truncated
    - prompt_category: Category of the prompt (general, code)
    """
    model_name: str
    model_size: str
    model_hash: Optional[str] = None
    quantization: Optional[str] = None
    prompt_id: Optional[str] = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_latency_ms: float = 0.0
    resource_utilization: Optional[ResourceUtilization] = None
    power_watts: Optional[float] = None
    warmup_runs: int = 3
    measured_runs: int = 10
    # Aggregated stats across runs
    ttft_mean_ms: Optional[float] = None
    ttft_std_ms: Optional[float] = None
    tps_mean: Optional[float] = None
    tps_std: Optional[float] = None
    latency_mean_ms: Optional[float] = None
    latency_std_ms: Optional[float] = None
    # Model Expansion PRD - Phase 1 metadata
    parameter_group: Optional[str] = None  # "1B", "3B", "7B", etc.
    architecture: Optional[str] = None  # "dense" or "moe"
    specialization: Optional[str] = None  # "general" or "code"
    # Model Expansion PRD - Phase 5 metrics
    peak_memory_mb: Optional[float] = None
    truncated: bool = False
    prompt_category: Optional[str] = None  # "general" or "code"
    # Phase 6 aggregation additions
    ttft_median_ms: Optional[float] = None
    ttft_min_ms: Optional[float] = None
    ttft_max_ms: Optional[float] = None
    tps_median: Optional[float] = None
    tps_min: Optional[float] = None
    tps_max: Optional[float] = None

    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "model_hash": self.model_hash,
            "quantization": self.quantization,
            "prompt_id": self.prompt_id,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "tokens_per_second": self.tokens_per_second,
            "total_latency_ms": self.total_latency_ms,
            "resource_utilization": self.resource_utilization.to_dict() if self.resource_utilization else None,
            "power_watts": self.power_watts,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "ttft_mean_ms": self.ttft_mean_ms,
            "ttft_std_ms": self.ttft_std_ms,
            "tps_mean": self.tps_mean,
            "tps_std": self.tps_std,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            # Model Expansion PRD fields
            "parameter_group": self.parameter_group,
            "architecture": self.architecture,
            "specialization": self.specialization,
            "peak_memory_mb": self.peak_memory_mb,
            "truncated": self.truncated,
            "prompt_category": self.prompt_category,
            "ttft_median_ms": self.ttft_median_ms,
            "ttft_min_ms": self.ttft_min_ms,
            "ttft_max_ms": self.ttft_max_ms,
            "tps_median": self.tps_median,
            "tps_min": self.tps_min,
            "tps_max": self.tps_max,
        }
        return result


@dataclass
class BenchmarkRun:
    """Complete benchmark run containing system info and results."""
    run_id: str
    system_info: SystemInfo
    workload_type: str
    started_at: str
    completed_at: Optional[str] = None
    yolo_results: list[YOLOResult] = field(default_factory=list)
    llm_results: list[LLMResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "system_info": self.system_info.to_dict(),
            "workload_type": self.workload_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "yolo_results": [r.to_dict() for r in self.yolo_results],
            "llm_results": [r.to_dict() for r in self.llm_results],
            "metadata": self.metadata,
        }


@dataclass
class AggregatedResults:
    """Aggregated results across multiple benchmark runs."""
    aggregation_id: str
    created_at: str
    runs: list[str]
    platform_summary: dict = field(default_factory=dict)
    yolo_summary: dict = field(default_factory=dict)
    llm_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
