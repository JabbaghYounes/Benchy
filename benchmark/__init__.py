# Edge AI Benchmark Suite
from benchmark.schemas import (
    Platform,
    WorkloadType,
    YOLOTask,
    SystemInfo,
    ResourceUtilization,
    LatencyMetrics,
    YOLOResult,
    LLMResult,
    BenchmarkRun,
    AggregatedResults,
)
from benchmark.results import JSONResultWriter, CSVResultWriter, ResultWriterFactory
from benchmark.metrics import collect_system_info, detect_platform, ResourceMonitor
from benchmark.workloads import (
    YOLOBenchmarkRunner,
    YOLOBenchmarkConfig,
    run_yolo_benchmark,
    LLMBenchmarkRunner,
    LLMBenchmarkConfig,
    run_llm_benchmark,
    check_ollama_status,
)
from benchmark.aggregation import (
    ResultsAggregator,
    YOLOAggregatedMetrics,
    LLMAggregatedMetrics,
    PlatformSummary,
    AggregatedCSVWriter,
    aggregate_results,
    write_aggregated_csvs,
)
from benchmark.reporting import (
    DashboardGenerator,
    generate_dashboard,
)

__version__ = "0.1.0"

__all__ = [
    # Schemas
    "Platform",
    "WorkloadType",
    "YOLOTask",
    "SystemInfo",
    "ResourceUtilization",
    "LatencyMetrics",
    "YOLOResult",
    "LLMResult",
    "BenchmarkRun",
    "AggregatedResults",
    # Writers
    "JSONResultWriter",
    "CSVResultWriter",
    "ResultWriterFactory",
    # Metrics
    "collect_system_info",
    "detect_platform",
    "ResourceMonitor",
    # Workloads
    "YOLOBenchmarkRunner",
    "YOLOBenchmarkConfig",
    "run_yolo_benchmark",
    "LLMBenchmarkRunner",
    "LLMBenchmarkConfig",
    "run_llm_benchmark",
    "check_ollama_status",
    # Aggregation
    "ResultsAggregator",
    "YOLOAggregatedMetrics",
    "LLMAggregatedMetrics",
    "PlatformSummary",
    "AggregatedCSVWriter",
    "aggregate_results",
    "write_aggregated_csvs",
    # Reporting
    "DashboardGenerator",
    "generate_dashboard",
]
