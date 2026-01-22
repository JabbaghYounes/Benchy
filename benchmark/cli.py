# Benchmark CLI entry point
import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from benchmark.schemas import (
    Platform,
    WorkloadType,
    YOLOTask,
    Backend,
    BenchmarkRun,
    SystemInfo,
)
from benchmark.metrics import collect_system_info
from benchmark.results import JSONResultWriter, CSVResultWriter


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _infer_yolo_model_info(model_name: str) -> tuple:
    """Infer YOLO version and task from model name.

    Args:
        model_name: Model filename (e.g., yolo26n.pt, yolov8s-seg.pt)

    Returns:
        Tuple of (version, task) where version is 'v8', 'v11', or 'v26'
        and task is a YOLOTask enum value.
    """
    from benchmark.schemas import YOLOTask

    model_lower = model_name.lower()

    # Determine version
    if "yolov8" in model_lower or "yolo8" in model_lower:
        version = "v8"
    elif "yolo11" in model_lower:
        version = "v11"
    elif "yolo26" in model_lower:
        version = "v26"
    else:
        # Default to v8 for unknown patterns
        version = "v8"

    # Determine task from suffix
    if "-seg" in model_lower:
        task = YOLOTask.SEGMENTATION
    elif "-pose" in model_lower:
        task = YOLOTask.POSE
    elif "-obb" in model_lower:
        task = YOLOTask.OBB
    elif "-cls" in model_lower:
        task = YOLOTask.CLASSIFICATION
    else:
        task = YOLOTask.DETECTION

    return version, task


def _run_single_yolo_model(
    model_name: str,
    benchmark_settings: dict,
    inference_settings: dict,
    backend: Optional[str],
    force_recompile: bool,
) -> list:
    """Run benchmark for a single specific YOLO model.

    Args:
        model_name: Model filename (e.g., yolo26n.pt)
        benchmark_settings: Benchmark configuration settings
        inference_settings: Inference configuration settings
        backend: Backend to use
        force_recompile: Force recompilation of Hailo models

    Returns:
        List containing single YOLOResult or empty list on failure
    """
    from benchmark.workloads.yolo import YOLOBenchmarkRunner, YOLOBenchmarkConfig

    logger = logging.getLogger(__name__)

    version, task = _infer_yolo_model_info(model_name)
    logger.info(f"Inferred version: {version}, task: {task.value}")

    bench_config = YOLOBenchmarkConfig(
        model_name=model_name,
        yolo_version=version,
        task=task,
        input_resolution=benchmark_settings.get("input_resolution", 640),
        warmup_runs=benchmark_settings.get("warmup_runs", 3),
        measured_runs=benchmark_settings.get("measured_runs", 10),
        device=inference_settings.get("device", "0"),
        conf_threshold=inference_settings.get("conf_threshold", 0.25),
        iou_threshold=inference_settings.get("iou_threshold", 0.45),
        backend=backend,
        force_recompile=force_recompile,
    )

    try:
        runner = YOLOBenchmarkRunner(bench_config)
        result = runner.run()
        logger.info(f"  [{result.backend}] Throughput: {result.throughput_fps:.2f} FPS, Latency: {result.latency.mean_ms:.2f}ms")
        return [result]
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return []


def run_yolo_benchmark(
    config: dict,
    profile: str,
    output_dir: Path,
    system_info: SystemInfo,
    skip_validation: bool = False,
    backend: Optional[str] = None,
    force_recompile: bool = False,
    yolo_version: Optional[str] = None,
    yolo_model: Optional[str] = None,
) -> list:
    """Run YOLO benchmarks based on configuration.

    Args:
        config: YOLO benchmark configuration
        profile: Benchmark profile (default or full)
        output_dir: Output directory for results
        system_info: System information
        skip_validation: Skip accuracy validation
        backend: Backend to use (pytorch, hailo, or None for auto)
        force_recompile: Force recompilation of Hailo models
        yolo_version: Specific YOLO version to run (overrides profile)
        yolo_model: Specific YOLO model to run (overrides profile)

    Returns:
        List of YOLOResult objects
    """
    from benchmark.workloads.yolo import YOLOBenchmarkRunner, YOLOBenchmarkConfig
    from benchmark.workloads.yolo.runner import YOLO_MODELS

    logger = logging.getLogger(__name__)

    # Get profile settings
    profile_config = config.get(profile, config.get("default", {}))
    benchmark_settings = config.get("benchmark", {})
    inference_settings = config.get("inference", {})

    yolo_versions = profile_config.get("yolo_versions", ["v8"])
    tasks = profile_config.get("tasks", ["detection"])
    model_sizes = profile_config.get("model_sizes", ["n"])

    # Override with CLI arguments if provided
    if yolo_version:
        yolo_versions = [yolo_version]
        logger.info(f"Using CLI-specified YOLO version: {yolo_version}")

    if yolo_model:
        # When specific model is provided, run only that model
        logger.info(f"Using CLI-specified YOLO model: {yolo_model}")
        return _run_single_yolo_model(
            yolo_model,
            benchmark_settings,
            inference_settings,
            backend,
            force_recompile,
        )

    all_results = []

    for version in yolo_versions:
        for task_name in tasks:
            task = YOLOTask(task_name)
            models = config.get("models", {}).get(version, {}).get(task_name, [])

            # Filter by model size
            filtered_models = []
            for m in models:
                for size in model_sizes:
                    if f"{size}." in m or f"{size}-" in m:
                        filtered_models.append(m)
                        break

            for model_name in filtered_models:
                logger.info(f"Benchmarking {model_name} ({task_name})")

                bench_config = YOLOBenchmarkConfig(
                    model_name=model_name,
                    yolo_version=version,
                    task=task,
                    input_resolution=benchmark_settings.get("input_resolution", 640),
                    warmup_runs=benchmark_settings.get("warmup_runs", 3),
                    measured_runs=benchmark_settings.get("measured_runs", 10),
                    device=inference_settings.get("device", "0"),
                    conf_threshold=inference_settings.get("conf_threshold", 0.25),
                    iou_threshold=inference_settings.get("iou_threshold", 0.45),
                    backend=backend,
                    force_recompile=force_recompile,
                )

                try:
                    runner = YOLOBenchmarkRunner(bench_config)
                    result = runner.run()
                    all_results.append(result)
                    logger.info(f"  [{result.backend}] Throughput: {result.throughput_fps:.2f} FPS, Latency: {result.latency.mean_ms:.2f}ms")
                except Exception as e:
                    logger.error(f"  Failed: {e}")

    return all_results


def run_llm_benchmark(
    config: dict,
    profile: str,
    output_dir: Path,
    system_info: SystemInfo,
) -> list:
    """Run LLM benchmarks based on configuration.

    Args:
        config: LLM benchmark configuration
        profile: Benchmark profile (default or full)
        output_dir: Output directory for results
        system_info: System information

    Returns:
        List of LLMResult objects
    """
    from benchmark.workloads.llm import LLMBenchmarkRunner, LLMBenchmarkConfig, check_ollama_status

    logger = logging.getLogger(__name__)

    # Check Ollama status
    status = check_ollama_status()
    if not status["server_running"]:
        logger.error("Ollama server is not running. Start with 'ollama serve'")
        return []

    # Get profile settings
    profile_config = config.get(profile, config.get("default", {}))
    benchmark_settings = config.get("benchmark", {})
    generation_settings = config.get("generation", {})
    ollama_settings = config.get("ollama", {})

    model_groups = profile_config.get("model_groups", ["7B"])
    specific_models = profile_config.get("models", None)

    prompts = config.get("prompts", [])

    all_results = []

    for group in model_groups:
        models = specific_models or config.get("models", {}).get(group, [])

        for model_name in models:
            logger.info(f"Benchmarking {model_name}")

            bench_config = LLMBenchmarkConfig(
                model_name=model_name,
                model_size=group,
                warmup_runs=benchmark_settings.get("warmup_runs", 3),
                measured_runs=benchmark_settings.get("measured_runs", 10),
                api_base=ollama_settings.get("api_base", "http://localhost:11434"),
                temperature=generation_settings.get("temperature", 0.0),
                top_p=generation_settings.get("top_p", 1.0),
                top_k=generation_settings.get("top_k", 1),
                seed=generation_settings.get("seed", 42),
                max_tokens=generation_settings.get("max_tokens", 256),
                prompts=prompts,
            )

            try:
                runner = LLMBenchmarkRunner(bench_config)
                results = runner.run()
                all_results.extend(results)

                for result in results:
                    logger.info(f"  [{result.prompt_id}] TPS: {result.tps_mean:.2f}, TTFT: {result.ttft_mean_ms:.1f}ms")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    return all_results


def cmd_benchmark(args) -> int:
    """Run benchmarks command handler."""
    logger = logging.getLogger(__name__)

    # Collect system info
    platform_override = None
    if args.platform:
        platform_override = Platform(args.platform)

    system_info = collect_system_info(platform_override)

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate run ID
    run_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now().isoformat()

    logger.info(f"Starting benchmark run: {run_id}")
    logger.info(f"Platform: {system_info.platform}")
    logger.info(f"Profile: {args.profile}")

    yolo_results = []
    llm_results = []

    # Run YOLO benchmarks
    if args.workload in ["yolo", "all"]:
        config_path = args.config_dir / "yolo_benchmark.yaml"
        if config_path.exists():
            config = load_config(config_path)
            logger.info("Running YOLO benchmarks...")
            yolo_results = run_yolo_benchmark(
                config,
                args.profile,
                args.output,
                system_info,
                args.skip_validation,
                backend=getattr(args, "backend", None),
                force_recompile=getattr(args, "force_recompile", False),
                yolo_version=getattr(args, "yolo_version", None),
                yolo_model=getattr(args, "yolo_model", None),
            )
        else:
            logger.warning(f"YOLO config not found: {config_path}")

    # Run LLM benchmarks
    if args.workload in ["llm", "all"]:
        config_path = args.config_dir / "llm_benchmark.yaml"
        if config_path.exists():
            config = load_config(config_path)
            logger.info("Running LLM benchmarks...")
            llm_results = run_llm_benchmark(
                config,
                args.profile,
                args.output,
                system_info,
            )
        else:
            logger.warning(f"LLM config not found: {config_path}")

    # Create benchmark run record
    completed_at = datetime.now().isoformat()
    benchmark_run = BenchmarkRun(
        run_id=run_id,
        system_info=system_info,
        workload_type=args.workload,
        started_at=started_at,
        completed_at=completed_at,
        yolo_results=yolo_results,
        llm_results=llm_results,
        metadata={
            "profile": args.profile,
            "config_dir": str(args.config_dir),
        },
    )

    # Write results
    if args.format in ["json", "both"]:
        json_writer = JSONResultWriter()
        output_path = args.output / f"{run_id}.json"
        json_writer.write(benchmark_run, output_path)
        logger.info(f"JSON results written to: {output_path}")

    if args.format in ["csv", "both"]:
        csv_writer = CSVResultWriter()
        if yolo_results:
            yolo_csv = args.output / f"{run_id}_yolo.csv"
            csv_writer.write_yolo_results(yolo_results, system_info, yolo_csv)
            logger.info(f"YOLO CSV results written to: {yolo_csv}")
        if llm_results:
            llm_csv = args.output / f"{run_id}_llm.csv"
            csv_writer.write_llm_results(llm_results, system_info, llm_csv)
            logger.info(f"LLM CSV results written to: {llm_csv}")

    # Print summary
    print(f"\nBenchmark Complete: {run_id}")
    print("-" * 40)
    if yolo_results:
        print(f"YOLO models benchmarked: {len(yolo_results)}")
        avg_fps = sum(r.throughput_fps for r in yolo_results) / len(yolo_results)
        print(f"  Average throughput: {avg_fps:.2f} FPS")
    if llm_results:
        print(f"LLM prompts benchmarked: {len(llm_results)}")
        avg_tps = sum(r.tps_mean for r in llm_results if r.tps_mean) / len(llm_results)
        print(f"  Average tokens/sec: {avg_tps:.2f}")
    print(f"Results saved to: {args.output}")

    return 0


def cmd_info(args) -> int:
    """Show system information command handler."""
    platform_override = None
    if args.platform:
        platform_override = Platform(args.platform)

    system_info = collect_system_info(platform_override)

    print("\nSystem Information:")
    print("-" * 40)
    for key, value in system_info.to_dict().items():
        print(f"  {key}: {value}")
    return 0


def cmd_aggregate(args) -> int:
    """Aggregate results command handler."""
    from benchmark.aggregation import (
        ResultsAggregator,
        AggregatedCSVWriter,
        aggregate_results,
    )

    logger = logging.getLogger(__name__)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    logger.info(f"Aggregating results from: {input_dir}")

    # Load and aggregate results
    aggregator = ResultsAggregator()
    aggregator.load_directory(input_dir)

    if not aggregator.raw_yolo_results and not aggregator.raw_llm_results:
        logger.warning("No results found to aggregate")
        return 0

    # Generate aggregation ID
    agg_id = f"agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Write aggregated JSON
    results = aggregator.create_aggregated_results(agg_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{agg_id}.json"
    with open(json_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info(f"Aggregated JSON written to: {json_path}")

    # Write aggregated CSVs
    csv_writer = AggregatedCSVWriter()
    csv_paths = csv_writer.write_all(aggregator, output_dir, agg_id)

    # Print summary
    print(f"\nAggregation Complete: {agg_id}")
    print("-" * 40)
    print(f"YOLO results aggregated: {len(aggregator.raw_yolo_results)}")
    print(f"LLM results aggregated: {len(aggregator.raw_llm_results)}")
    print(f"Platforms: {len(aggregator.get_platform_summaries())}")
    print(f"\nOutput files:")
    print(f"  JSON: {json_path}")
    for name, path in csv_paths.items():
        print(f"  {name.upper()} CSV: {path}")

    return 0


def cmd_dashboard(args) -> int:
    """Generate dashboard command handler."""
    from benchmark.aggregation import ResultsAggregator
    from benchmark.reporting import DashboardGenerator, generate_dashboard

    logger = logging.getLogger(__name__)

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    logger.info(f"Generating dashboard from: {input_dir}")

    # Load results
    aggregator = ResultsAggregator()
    aggregator.load_directory(input_dir)

    if not aggregator.raw_yolo_results and not aggregator.raw_llm_results:
        logger.warning("No results found for dashboard")
        return 0

    # Generate dashboard
    generator = DashboardGenerator(aggregator)
    generator.generate(output_path, title=args.title)

    print(f"\nDashboard Generated: {output_path}")
    print("-" * 40)
    print(f"YOLO results: {len(aggregator.raw_yolo_results)}")
    print(f"LLM results: {len(aggregator.raw_llm_results)}")
    print(f"Platforms: {len(aggregator.get_platform_summaries())}")
    print(f"\nOpen in browser: file://{output_path.absolute()}")

    return 0


def cmd_report(args) -> int:
    """Generate full report (aggregate + dashboard) command handler."""
    from benchmark.aggregation import ResultsAggregator, AggregatedCSVWriter
    from benchmark.reporting import DashboardGenerator

    logger = logging.getLogger(__name__)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    logger.info(f"Generating report from: {input_dir}")

    # Load results
    aggregator = ResultsAggregator()
    aggregator.load_directory(input_dir)

    if not aggregator.raw_yolo_results and not aggregator.raw_llm_results:
        logger.warning("No results found for report")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate aggregation ID
    report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Write aggregated JSON
    results = aggregator.create_aggregated_results(report_id)
    json_path = output_dir / f"{report_id}.json"
    with open(json_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info(f"Aggregated JSON written to: {json_path}")

    # Write aggregated CSVs
    csv_writer = AggregatedCSVWriter()
    csv_paths = csv_writer.write_all(aggregator, output_dir, report_id)

    # Generate dashboard
    dashboard_path = output_dir / f"{report_id}_dashboard.html"
    generator = DashboardGenerator(aggregator)
    generator.generate(dashboard_path, title=args.title)

    # Print summary
    print(f"\nReport Generated: {report_id}")
    print("-" * 40)
    print(f"YOLO results: {len(aggregator.raw_yolo_results)}")
    print(f"LLM results: {len(aggregator.raw_llm_results)}")
    print(f"Platforms: {len(aggregator.get_platform_summaries())}")
    print(f"\nOutput files:")
    print(f"  JSON: {json_path}")
    for name, path in csv_paths.items():
        print(f"  {name.upper()} CSV: {path}")
    print(f"  Dashboard: {dashboard_path}")
    print(f"\nOpen dashboard in browser: file://{dashboard_path.absolute()}")

    return 0


def cmd_list_models(args) -> int:
    """List supported models command handler (Phase 6 - Task 6.1)."""
    from benchmark.workloads.yolo.execution import (
        format_supported_configurations,
        get_supported_configurations,
        HAILO_SUPPORTED_VERSIONS,
        HAILO_SUPPORTED_TASKS,
    )
    from benchmark.workloads.yolo.runner import YOLO_MODELS

    # Determine output format
    if args.format == "json":
        if args.backend == "hailo":
            configs = get_supported_configurations()
            print(json.dumps(configs, indent=2))
        else:
            # All models for all backends
            output = {
                "pytorch": {},
                "hailo": get_supported_configurations(),
            }
            for version, tasks in YOLO_MODELS.items():
                output["pytorch"][version] = {}
                for task, models in tasks.items():
                    output["pytorch"][version][task.value] = models
            print(json.dumps(output, indent=2))
    else:
        # Human-readable format
        if args.backend == "hailo":
            print(format_supported_configurations())
        elif args.backend == "pytorch":
            print("\nPyTorch-supported YOLO configurations:")
            print("-" * 40)
            for version, tasks in YOLO_MODELS.items():
                print(f"\nYOLO {version}:")
                for task, models in tasks.items():
                    print(f"  {task.value}:")
                    for model in models:
                        print(f"    - {model}")
        else:
            # Show all backends
            print("\n" + "=" * 50)
            print("SUPPORTED YOLO MODELS BY BACKEND")
            print("=" * 50)

            print("\n[PyTorch Backend]")
            print("-" * 40)
            print("Supports all YOLO models on CPU/GPU")
            for version, tasks in YOLO_MODELS.items():
                print(f"\nYOLO {version}:")
                for task, models in tasks.items():
                    print(f"  {task.value}: {len(models)} models")

            print("\n" + "-" * 50)
            print(format_supported_configurations())

            print("\n" + "=" * 50)
            print("NOTE: Hailo backend requires Hailo-8/8L NPU hardware")
            print("      and HailoRT runtime installed.")
            print("=" * 50)

    return 0


def cmd_backends(args) -> int:
    """List available backends command handler (Phase 6 - Task 6.1)."""
    from benchmark.workloads.yolo.backends import get_backend_info

    print("\nAvailable Inference Backends:")
    print("-" * 40)

    backend_info = get_backend_info()

    for backend_name, info in backend_info.items():
        status = "Available" if info.get("available") else "Not Available"
        print(f"\n{backend_name.upper()}: {status}")

        if info.get("available"):
            caps = info.get("capabilities", {})
            tasks = caps.get("tasks", [])
            versions = caps.get("versions", [])
            print(f"  Supported tasks: {', '.join(tasks)}")
            print(f"  YOLO versions: {', '.join(versions)}")
            if caps.get("gpu"):
                print("  GPU acceleration: Yes")
            if caps.get("npu"):
                print("  NPU acceleration: Yes")
            if caps.get("requires_compilation"):
                print("  Requires model compilation: Yes")

            version_info = info.get("version_info", {})
            if version_info:
                for key, value in version_info.items():
                    if value:
                        print(f"  {key}: {value}")
        else:
            error = info.get("error", "Unknown reason")
            print(f"  Reason: {error}")

    return 0


def cmd_verify(args) -> int:
    """Cross-platform verification command handler (Phase 7 - Task 7.1)."""
    from benchmark.verification import (
        load_benchmark_run,
        generate_verification_report,
    )

    logger = logging.getLogger(__name__)

    run_a_path = Path(args.run_a)
    run_b_path = Path(args.run_b)

    if not run_a_path.exists():
        logger.error(f"Benchmark run file not found: {run_a_path}")
        return 1

    if not run_b_path.exists():
        logger.error(f"Benchmark run file not found: {run_b_path}")
        return 1

    logger.info(f"Loading benchmark runs for comparison...")

    try:
        run_a = load_benchmark_run(run_a_path)
        run_b = load_benchmark_run(run_b_path)
    except Exception as e:
        logger.error(f"Failed to load benchmark runs: {e}")
        return 1

    logger.info(f"Comparing: {run_a.system_info.platform} vs {run_b.system_info.platform}")

    # Generate report
    output_path = None
    if args.output:
        output_path = Path(args.output)

    report = generate_verification_report(run_a, run_b, output_path)

    # Print report
    if args.format == "text":
        print(report.get_full_report())
    else:
        print(json.dumps(report.to_dict(), indent=2))

    # Summary
    valid_count = sum(1 for c in report.comparisons if c.is_valid)
    total_count = len(report.comparisons)

    print(f"\nVerification Complete: {valid_count}/{total_count} valid comparisons")

    if output_path:
        print(f"Report saved to: {output_path}")

    return 0 if valid_count == total_count else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Edge AI Benchmark Suite - Benchmark AI inference on edge devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "run",
        help="Run benchmarks",
        description="Run YOLO and/or LLM benchmarks",
    )
    bench_parser.add_argument(
        "workload",
        choices=["yolo", "llm", "all"],
        help="Workload to benchmark",
    )
    bench_parser.add_argument(
        "--profile",
        choices=["default", "full"],
        default="default",
        help="Benchmark profile (default: default)",
    )
    bench_parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent.parent / "configs",
        help="Configuration directory",
    )
    bench_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Output directory for results",
    )
    bench_parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format (default: both)",
    )
    bench_parser.add_argument(
        "--platform",
        choices=["jetson_nano", "rpi_ai_hat_plus", "rpi_ai_hat_plus_2"],
        help="Override platform detection",
    )
    bench_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip accuracy validation for YOLO benchmarks",
    )
    bench_parser.add_argument(
        "--backend",
        choices=["pytorch", "hailo", "auto"],
        default=None,
        help="Inference backend (default: auto-select based on platform)",
    )
    bench_parser.add_argument(
        "--force-recompile",
        action="store_true",
        help="Force recompilation of Hailo models (ignore cache)",
    )
    bench_parser.add_argument(
        "--yolo-version",
        choices=["v8", "v11", "v26"],
        help="Run specific YOLO version only (overrides profile)",
    )
    bench_parser.add_argument(
        "--yolo-model",
        type=str,
        help="Run specific YOLO model only (e.g., yolo26n.pt, yolov8s-seg.pt)",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display detected system information",
    )
    info_parser.add_argument(
        "--platform",
        choices=["jetson_nano", "rpi_ai_hat_plus", "rpi_ai_hat_plus_2"],
        help="Override platform detection",
    )

    # Aggregate command
    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate benchmark results",
        description="Aggregate raw benchmark results into summary CSVs",
    )
    agg_parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Input directory containing raw results",
    )
    agg_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "aggregated",
        help="Output directory for aggregated results",
    )

    # Dashboard command
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Generate HTML dashboard",
        description="Generate interactive HTML dashboard from results",
    )
    dash_parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Input directory containing results",
    )
    dash_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "dashboard.html",
        help="Output path for HTML dashboard",
    )
    dash_parser.add_argument(
        "--title",
        default="Edge AI Benchmark Dashboard",
        help="Dashboard title",
    )

    # Report command (aggregate + dashboard)
    report_parser = subparsers.add_parser(
        "report",
        help="Generate full report",
        description="Generate aggregated results and HTML dashboard",
    )
    report_parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Input directory containing results",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "report",
        help="Output directory for report",
    )
    report_parser.add_argument(
        "--title",
        default="Edge AI Benchmark Report",
        help="Report title",
    )

    # Phase 6: List supported models command
    models_parser = subparsers.add_parser(
        "list-models",
        help="List supported YOLO models",
        description="List YOLO models supported by each backend (Phase 6 - Task 6.1)",
    )
    models_parser.add_argument(
        "--backend",
        choices=["pytorch", "hailo", "all"],
        default="all",
        help="Filter by backend (default: all)",
    )
    models_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Phase 6: List backends command
    backends_parser = subparsers.add_parser(
        "backends",
        help="List available inference backends",
        description="Show available backends and their capabilities",
    )

    # Phase 7: Cross-platform verification command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Compare benchmark results across platforms",
        description="Cross-platform verification (Phase 7 - Task 7.1). "
                    "Compares results from different platforms (e.g., Jetson GPU vs RPi+Hailo NPU)",
    )
    verify_parser.add_argument(
        "run_a",
        type=str,
        help="Path to first benchmark run JSON file (e.g., Jetson results)",
    )
    verify_parser.add_argument(
        "run_b",
        type=str,
        help="Path to second benchmark run JSON file (e.g., RPi+Hailo results)",
    )
    verify_parser.add_argument(
        "--output",
        type=Path,
        help="Output path for verification report JSON",
    )
    verify_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Handle commands
    if args.command == "run":
        return cmd_benchmark(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "aggregate":
        return cmd_aggregate(args)
    elif args.command == "dashboard":
        return cmd_dashboard(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "list-models":
        return cmd_list_models(args)
    elif args.command == "backends":
        return cmd_backends(args)
    elif args.command == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
