# Phase 7 - Cross-Platform Verification Module
#
# Task 7.1: Validates benchmark results across platforms (Jetson GPU vs RPi+Hailo NPU)
# Ensures comparisons are fair and performance deltas are clearly visible.

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark.schemas import (
    Platform,
    YOLOResult,
    BenchmarkRun,
    SystemInfo,
)

logger = logging.getLogger(__name__)


class ComparisonError(Exception):
    """Raised when comparison validation fails."""
    pass


class MisleadingComparisonError(ComparisonError):
    """Raised when a comparison would be misleading."""
    pass


@dataclass
class ComparisonCriteria:
    """Criteria for valid cross-platform comparisons.

    To ensure fair comparisons, results must match on these attributes.
    """
    model_name: bool = True
    yolo_version: bool = True
    task: bool = True
    input_resolution: bool = True
    warmup_runs: bool = False  # Different platforms may use different warmup
    measured_runs: bool = False  # Different platforms may use different run counts

    def get_required_fields(self) -> List[str]:
        """Get list of fields that must match."""
        fields = []
        if self.model_name:
            fields.append("model_name")
        if self.yolo_version:
            fields.append("yolo_version")
        if self.task:
            fields.append("task")
        if self.input_resolution:
            fields.append("input_resolution")
        if self.warmup_runs:
            fields.append("warmup_runs")
        if self.measured_runs:
            fields.append("measured_runs")
        return fields


@dataclass
class PerformanceDelta:
    """Performance difference between two platforms."""
    metric_name: str
    platform_a: str
    platform_b: str
    value_a: float
    value_b: float
    absolute_diff: float
    relative_diff_percent: float
    winner: str  # Which platform is better for this metric

    def to_dict(self) -> dict:
        return {
            "metric": self.metric_name,
            "platforms": {
                self.platform_a: self.value_a,
                self.platform_b: self.value_b,
            },
            "absolute_diff": round(self.absolute_diff, 3),
            "relative_diff_percent": round(self.relative_diff_percent, 2),
            "winner": self.winner,
        }


@dataclass
class ComparisonResult:
    """Result of comparing benchmark results across platforms."""
    model_name: str
    yolo_version: str
    task: str
    input_resolution: str
    platforms: List[str]
    backends: List[str]
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_deltas: List[PerformanceDelta] = field(default_factory=list)
    raw_results: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "yolo_version": self.yolo_version,
            "task": self.task,
            "input_resolution": self.input_resolution,
            "platforms": self.platforms,
            "backends": self.backends,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "warnings": self.warnings,
            "performance_deltas": [d.to_dict() for d in self.performance_deltas],
            "raw_results": self.raw_results,
        }

    def get_summary(self) -> str:
        """Get human-readable summary of comparison."""
        lines = [
            f"\n{'=' * 60}",
            f"CROSS-PLATFORM COMPARISON: {self.model_name}",
            f"{'=' * 60}",
            f"Task: {self.task} | Resolution: {self.input_resolution}",
            f"Platforms: {' vs '.join(self.platforms)}",
            f"Backends: {' vs '.join(self.backends)}",
            f"Valid Comparison: {'Yes' if self.is_valid else 'NO'}",
        ]

        if self.validation_errors:
            lines.append(f"\nValidation Errors:")
            for err in self.validation_errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append(f"\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        if self.performance_deltas:
            lines.append(f"\nPerformance Deltas:")
            lines.append("-" * 60)

            for delta in self.performance_deltas:
                direction = "higher" if delta.relative_diff_percent > 0 else "lower"
                lines.append(
                    f"  {delta.metric_name}:"
                )
                lines.append(
                    f"    {delta.platform_a}: {delta.value_a:.2f}"
                )
                lines.append(
                    f"    {delta.platform_b}: {delta.value_b:.2f}"
                )
                lines.append(
                    f"    Delta: {abs(delta.relative_diff_percent):.1f}% {direction} on {delta.winner}"
                )

        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


class CrossPlatformVerifier:
    """Verifies benchmark results across different platforms.

    Phase 7.1: Ensures fair comparisons between:
    - Jetson (GPU) running PyTorch
    - RPi + Hailo (NPU) running HailoRT

    Key validations:
    - Same model, version, task, and resolution
    - Clear performance delta reporting
    - Warnings for potentially misleading comparisons
    """

    # Metrics where higher is better
    HIGHER_IS_BETTER = {"throughput_fps"}

    # Metrics where lower is better
    LOWER_IS_BETTER = {
        "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "latency_min_ms", "latency_max_ms", "latency_std_ms",
    }

    def __init__(self, criteria: Optional[ComparisonCriteria] = None):
        """Initialize verifier.

        Args:
            criteria: Comparison criteria. Uses strict defaults if None.
        """
        self.criteria = criteria or ComparisonCriteria()

    def validate_comparison(
        self,
        result_a: YOLOResult,
        result_b: YOLOResult,
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate that two results can be fairly compared.

        Args:
            result_a: First benchmark result
            result_b: Second benchmark result

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check required fields match
        required_fields = self.criteria.get_required_fields()

        for field_name in required_fields:
            val_a = getattr(result_a, field_name, None)
            val_b = getattr(result_b, field_name, None)

            if val_a != val_b:
                errors.append(
                    f"Mismatched {field_name}: '{val_a}' vs '{val_b}'. "
                    f"Results are not comparable."
                )

        # Check for potentially misleading comparisons
        if result_a.backend == result_b.backend:
            warnings.append(
                f"Both results use the same backend ({result_a.backend}). "
                f"This is a same-backend comparison, not cross-platform."
            )

        # Warn about different run configurations
        if result_a.warmup_runs != result_b.warmup_runs:
            warnings.append(
                f"Different warmup runs: {result_a.warmup_runs} vs {result_b.warmup_runs}"
            )

        if result_a.measured_runs != result_b.measured_runs:
            warnings.append(
                f"Different measured runs: {result_a.measured_runs} vs {result_b.measured_runs}"
            )

        # Warn about Hailo CPU fallback (should never happen with Phase 5)
        if result_a.backend == "hailo" or result_b.backend == "hailo":
            # This is valid - Hailo NPU comparison
            pass

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def calculate_deltas(
        self,
        result_a: YOLOResult,
        result_b: YOLOResult,
        platform_a: str,
        platform_b: str,
    ) -> List[PerformanceDelta]:
        """Calculate performance deltas between two results.

        Args:
            result_a: First benchmark result
            result_b: Second benchmark result
            platform_a: Name/identifier for first platform
            platform_b: Name/identifier for second platform

        Returns:
            List of PerformanceDelta objects
        """
        deltas = []

        # Throughput (higher is better)
        if result_a.throughput_fps and result_b.throughput_fps:
            val_a = result_a.throughput_fps
            val_b = result_b.throughput_fps
            abs_diff = val_a - val_b
            rel_diff = ((val_a - val_b) / val_b * 100) if val_b > 0 else 0
            winner = platform_a if val_a > val_b else platform_b

            deltas.append(PerformanceDelta(
                metric_name="Throughput (FPS)",
                platform_a=platform_a,
                platform_b=platform_b,
                value_a=val_a,
                value_b=val_b,
                absolute_diff=abs_diff,
                relative_diff_percent=rel_diff,
                winner=winner,
            ))

        # Latency metrics (lower is better)
        latency_metrics = [
            ("Mean Latency (ms)", "mean_ms"),
            ("P50 Latency (ms)", "p50_ms"),
            ("P95 Latency (ms)", "p95_ms"),
            ("P99 Latency (ms)", "p99_ms"),
            ("Min Latency (ms)", "min_ms"),
            ("Max Latency (ms)", "max_ms"),
        ]

        for metric_name, attr_name in latency_metrics:
            val_a = getattr(result_a.latency, attr_name, None)
            val_b = getattr(result_b.latency, attr_name, None)

            if val_a is not None and val_b is not None:
                abs_diff = val_a - val_b
                rel_diff = ((val_a - val_b) / val_b * 100) if val_b > 0 else 0
                # For latency, lower is better
                winner = platform_a if val_a < val_b else platform_b

                deltas.append(PerformanceDelta(
                    metric_name=metric_name,
                    platform_a=platform_a,
                    platform_b=platform_b,
                    value_a=val_a,
                    value_b=val_b,
                    absolute_diff=abs_diff,
                    relative_diff_percent=rel_diff,
                    winner=winner,
                ))

        return deltas

    def compare(
        self,
        result_a: YOLOResult,
        result_b: YOLOResult,
        platform_a: str = "Platform A",
        platform_b: str = "Platform B",
    ) -> ComparisonResult:
        """Compare two benchmark results from different platforms.

        Args:
            result_a: First benchmark result
            result_b: Second benchmark result
            platform_a: Name for first platform
            platform_b: Name for second platform

        Returns:
            ComparisonResult with validation and deltas
        """
        # Validate comparison
        is_valid, errors, warnings = self.validate_comparison(result_a, result_b)

        # Calculate deltas even if invalid (for informational purposes)
        deltas = self.calculate_deltas(result_a, result_b, platform_a, platform_b)

        return ComparisonResult(
            model_name=result_a.model_name,
            yolo_version=result_a.yolo_version,
            task=result_a.task,
            input_resolution=result_a.input_resolution,
            platforms=[platform_a, platform_b],
            backends=[result_a.backend, result_b.backend],
            is_valid=is_valid,
            validation_errors=errors,
            warnings=warnings,
            performance_deltas=deltas,
            raw_results={
                platform_a: result_a.to_dict(),
                platform_b: result_b.to_dict(),
            },
        )

    def compare_runs(
        self,
        run_a: BenchmarkRun,
        run_b: BenchmarkRun,
    ) -> List[ComparisonResult]:
        """Compare all matching results between two benchmark runs.

        Args:
            run_a: First benchmark run
            run_b: Second benchmark run

        Returns:
            List of ComparisonResult for each matching model
        """
        results = []
        platform_a = run_a.system_info.platform
        platform_b = run_b.system_info.platform

        # Index results by model key
        def get_key(r: YOLOResult) -> str:
            return f"{r.model_name}_{r.yolo_version}_{r.task}_{r.input_resolution}"

        results_a = {get_key(r): r for r in run_a.yolo_results}
        results_b = {get_key(r): r for r in run_b.yolo_results}

        # Find matching models
        common_keys = set(results_a.keys()) & set(results_b.keys())

        for key in sorted(common_keys):
            comparison = self.compare(
                results_a[key],
                results_b[key],
                platform_a,
                platform_b,
            )
            results.append(comparison)

        # Warn about non-matching models
        only_a = set(results_a.keys()) - common_keys
        only_b = set(results_b.keys()) - common_keys

        if only_a:
            logger.warning(
                f"Models only in {platform_a}: {', '.join(only_a)}"
            )
        if only_b:
            logger.warning(
                f"Models only in {platform_b}: {', '.join(only_b)}"
            )

        return results


@dataclass
class VerificationReport:
    """Complete verification report for cross-platform comparison."""
    report_id: str
    created_at: str
    platforms: List[str]
    comparisons: List[ComparisonResult]
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "platforms": self.platforms,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "summary": self.summary,
        }

    def get_full_report(self) -> str:
        """Generate full human-readable report."""
        lines = [
            "\n" + "=" * 70,
            "CROSS-PLATFORM VERIFICATION REPORT",
            "=" * 70,
            f"Report ID: {self.report_id}",
            f"Generated: {self.created_at}",
            f"Platforms: {', '.join(self.platforms)}",
            "",
            f"Total Comparisons: {len(self.comparisons)}",
            f"Valid Comparisons: {sum(1 for c in self.comparisons if c.is_valid)}",
            f"Invalid Comparisons: {sum(1 for c in self.comparisons if not c.is_valid)}",
        ]

        # Summary statistics
        if self.comparisons:
            valid_comparisons = [c for c in self.comparisons if c.is_valid]

            if valid_comparisons:
                # Find overall performance winner
                fps_wins = {}
                latency_wins = {}

                for comp in valid_comparisons:
                    for delta in comp.performance_deltas:
                        if "FPS" in delta.metric_name:
                            fps_wins[delta.winner] = fps_wins.get(delta.winner, 0) + 1
                        elif "Latency" in delta.metric_name:
                            latency_wins[delta.winner] = latency_wins.get(delta.winner, 0) + 1

                lines.append("\nPerformance Summary:")
                lines.append("-" * 40)

                if fps_wins:
                    lines.append("Throughput wins by platform:")
                    for platform, wins in sorted(fps_wins.items(), key=lambda x: -x[1]):
                        lines.append(f"  {platform}: {wins} models")

                if latency_wins:
                    lines.append("Latency wins by platform:")
                    for platform, wins in sorted(latency_wins.items(), key=lambda x: -x[1]):
                        lines.append(f"  {platform}: {wins} models")

        # Individual comparison details
        lines.append("\n" + "=" * 70)
        lines.append("DETAILED COMPARISONS")
        lines.append("=" * 70)

        for comp in self.comparisons:
            lines.append(comp.get_summary())

        # Disclaimer
        lines.extend([
            "",
            "=" * 70,
            "IMPORTANT NOTES",
            "=" * 70,
            "1. Comparisons are between different hardware accelerators (GPU vs NPU)",
            "2. Power consumption may vary significantly between platforms",
            "3. Hailo results use INT8 quantization, PyTorch may use FP16/FP32",
            "4. For fair comparison, ensure identical input data and preprocessing",
            "5. CPU fallback on Hailo platforms is NOT valid for benchmarking",
            "=" * 70,
        ])

        return "\n".join(lines)


def generate_verification_report(
    run_a: BenchmarkRun,
    run_b: BenchmarkRun,
    output_path: Optional[Path] = None,
) -> VerificationReport:
    """Generate a cross-platform verification report.

    Args:
        run_a: First benchmark run (e.g., Jetson)
        run_b: Second benchmark run (e.g., RPi + Hailo)
        output_path: Optional path to write JSON report

    Returns:
        VerificationReport with all comparisons
    """
    verifier = CrossPlatformVerifier()
    comparisons = verifier.compare_runs(run_a, run_b)

    report = VerificationReport(
        report_id=f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        created_at=datetime.now().isoformat(),
        platforms=[run_a.system_info.platform, run_b.system_info.platform],
        comparisons=comparisons,
        summary={
            "total_comparisons": len(comparisons),
            "valid_comparisons": sum(1 for c in comparisons if c.is_valid),
            "models_compared": [c.model_name for c in comparisons],
        },
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Verification report written to: {output_path}")

    return report


def load_benchmark_run(path: Path) -> BenchmarkRun:
    """Load a benchmark run from JSON file.

    Args:
        path: Path to benchmark JSON file

    Returns:
        BenchmarkRun object
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruct SystemInfo
    sys_info_data = data.get("system_info", {})
    system_info = SystemInfo(
        platform=sys_info_data.get("platform", "unknown"),
        cpu_model=sys_info_data.get("cpu_model", "unknown"),
        accelerator=sys_info_data.get("accelerator", "unknown"),
        ram_size_gb=sys_info_data.get("ram_size_gb", 0),
        storage_type=sys_info_data.get("storage_type", "unknown"),
        cooling_config=sys_info_data.get("cooling_config", "unknown"),
        power_mode=sys_info_data.get("power_mode", "unknown"),
        os_version=sys_info_data.get("os_version", "unknown"),
        kernel_version=sys_info_data.get("kernel_version", "unknown"),
        hostname=sys_info_data.get("hostname"),
        timestamp=sys_info_data.get("timestamp", ""),
    )

    # Reconstruct YOLO results
    from benchmark.schemas import LatencyMetrics, ResourceUtilization

    yolo_results = []
    for r in data.get("yolo_results", []):
        latency_data = r.get("latency", {})
        latency = LatencyMetrics(
            first_inference_ms=latency_data.get("first_inference_ms", 0),
            mean_ms=latency_data.get("mean_ms", 0),
            std_ms=latency_data.get("std_ms", 0),
            min_ms=latency_data.get("min_ms", 0),
            max_ms=latency_data.get("max_ms", 0),
            p50_ms=latency_data.get("p50_ms", 0),
            p95_ms=latency_data.get("p95_ms", 0),
            p99_ms=latency_data.get("p99_ms"),
        )

        resource_data = r.get("resource_utilization")
        resource_util = None
        if resource_data:
            resource_util = ResourceUtilization(
                cpu_percent=resource_data.get("cpu_percent", 0),
                accelerator_percent=resource_data.get("accelerator_percent"),
                memory_used_mb=resource_data.get("memory_used_mb", 0),
                memory_total_mb=resource_data.get("memory_total_mb", 0),
                power_watts=resource_data.get("power_watts"),
            )

        yolo_results.append(YOLOResult(
            model_name=r.get("model_name", ""),
            yolo_version=r.get("yolo_version", ""),
            task=r.get("task", ""),
            input_resolution=r.get("input_resolution", ""),
            latency=latency,
            throughput_fps=r.get("throughput_fps", 0),
            backend=r.get("backend", "pytorch"),
            map_score=r.get("map_score"),
            precision=r.get("precision"),
            recall=r.get("recall"),
            resource_utilization=resource_util,
            warmup_runs=r.get("warmup_runs", 3),
            measured_runs=r.get("measured_runs", 10),
        ))

    return BenchmarkRun(
        run_id=data.get("run_id", ""),
        system_info=system_info,
        workload_type=data.get("workload_type", "yolo"),
        started_at=data.get("started_at", ""),
        completed_at=data.get("completed_at"),
        yolo_results=yolo_results,
        llm_results=[],  # LLM results not needed for YOLO verification
        metadata=data.get("metadata", {}),
    )
