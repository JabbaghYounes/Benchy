# LLM Benchmark Runner using Ollama
import json
import logging
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator

import requests

from benchmark.schemas import LLMResult
from benchmark.metrics import ResourceMonitor

logger = logging.getLogger(__name__)

# Default Ollama API endpoint
OLLAMA_API_BASE = "http://localhost:11434"


@dataclass
class LLMBenchmarkConfig:
    """Configuration for LLM benchmark runs."""

    model_name: str = "llama2:7b"
    model_size: str = "7B"
    warmup_runs: int = 3
    measured_runs: int = 10
    api_base: str = OLLAMA_API_BASE
    # Generation parameters for deterministic output
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 42
    max_tokens: int = 256
    # Prompts to use
    prompts: list[str] = field(default_factory=list)


# Model groups as defined in PRD
LLM_MODELS = {
    "7B": ["llama2:7b", "mistral:7b", "olmo2:7b"],
    "8B": ["llama3.1:8b", "dolphin3:8b", "dolphin-llama3:8b"],
    "9B": ["gemma2:9b"],
}

# Default benchmark prompts - fixed and version-controlled
DEFAULT_PROMPTS = [
    {
        "id": "simple_qa",
        "prompt": "What is the capital of France? Answer in one word.",
        "expected_tokens": 10,
    },
    {
        "id": "reasoning",
        "prompt": "If a train travels at 60 mph for 2 hours, how far does it travel? Show your calculation step by step.",
        "expected_tokens": 100,
    },
    {
        "id": "code_generation",
        "prompt": "Write a Python function that checks if a number is prime. Include a docstring.",
        "expected_tokens": 150,
    },
    {
        "id": "summarization",
        "prompt": "Summarize the concept of machine learning in exactly three sentences.",
        "expected_tokens": 80,
    },
    {
        "id": "creative",
        "prompt": "Write a haiku about artificial intelligence.",
        "expected_tokens": 30,
    },
]


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = OLLAMA_API_BASE):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_info(self, model_name: str) -> Optional[dict]:
        """Get information about a specific model."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return None

    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not already available."""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=3600,  # Long timeout for large model downloads
            )
            response.raise_for_status()

            # Stream the response to show progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status or "downloading" in status:
                        logger.debug(status)
                    elif "success" in status.lower():
                        logger.info(f"Model {model_name} pulled successfully")
                        return True
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    def generate_stream(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        seed: int = 42,
        max_tokens: int = 256,
    ) -> Generator[dict, None, None]:
        """Generate completion with streaming response.

        Yields dictionaries with token data including timing information.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        except requests.RequestException as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        seed: int = 42,
        max_tokens: int = 256,
    ) -> dict:
        """Generate completion without streaming.

        Returns dict with response and timing metrics.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Generation failed: {e}")
            raise


@dataclass
class InferenceMetrics:
    """Metrics from a single inference run."""

    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    prompt_tokens: int
    response_text: str


class LLMBenchmarkRunner:
    """Runner for LLM inference benchmarks using Ollama."""

    def __init__(self, config: LLMBenchmarkConfig):
        """Initialize the LLM benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.client = OllamaClient(config.api_base)
        self._resource_monitor = ResourceMonitor(sample_interval=0.1)
        self._model_info: Optional[dict] = None

    def _ensure_model_available(self) -> bool:
        """Ensure the model is available, pulling if necessary."""
        if not self.client.is_available():
            raise RuntimeError("Ollama server is not running. Start with 'ollama serve'")

        models = self.client.list_models()
        model_names = [m.get("name", "").split(":")[0] + ":" + m.get("name", "").split(":")[-1]
                       for m in models]

        # Check if model exists (handle tag variations)
        model_base = self.config.model_name.split(":")[0]
        if not any(model_base in name for name in model_names):
            logger.info(f"Model {self.config.model_name} not found, pulling...")
            if not self.client.pull_model(self.config.model_name):
                return False

        # Get model info
        self._model_info = self.client.get_model_info(self.config.model_name)
        return True

    def _run_single_inference(self, prompt: str) -> InferenceMetrics:
        """Run a single inference and capture detailed metrics.

        Args:
            prompt: The prompt to send to the model

        Returns:
            InferenceMetrics with timing and token data
        """
        start_time = time.perf_counter()
        first_token_time = None
        tokens_generated = 0
        response_text = ""

        # Use streaming to capture TTFT accurately
        for chunk in self.client.generate_stream(
            model=self.config.model_name,
            prompt=prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            seed=self.config.seed,
            max_tokens=self.config.max_tokens,
        ):
            if first_token_time is None and chunk.get("response"):
                first_token_time = time.perf_counter()

            response_text += chunk.get("response", "")

            if chunk.get("done", False):
                # Extract token counts from final response
                tokens_generated = chunk.get("eval_count", len(response_text.split()))
                prompt_tokens = chunk.get("prompt_eval_count", len(prompt.split()))
                break

        end_time = time.perf_counter()

        # Calculate metrics
        total_latency_ms = (end_time - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_latency_ms

        # Calculate tokens per second (excluding TTFT for generation speed)
        generation_time_s = (end_time - (first_token_time or start_time))
        tps = tokens_generated / generation_time_s if generation_time_s > 0 else 0.0

        return InferenceMetrics(
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens if 'prompt_tokens' in dir() else len(prompt.split()),
            response_text=response_text,
        )

    def run(self, prompt_id: Optional[str] = None) -> list[LLMResult]:
        """Run the complete benchmark.

        Args:
            prompt_id: Optional specific prompt ID to benchmark. If None, runs all prompts.

        Returns:
            List of LLMResult for each prompt
        """
        # Ensure model is available
        if not self._ensure_model_available():
            raise RuntimeError(f"Failed to load model: {self.config.model_name}")

        # Get prompts to run
        prompts = self.config.prompts if self.config.prompts else DEFAULT_PROMPTS
        if prompt_id:
            prompts = [p for p in prompts if p.get("id") == prompt_id]
            if not prompts:
                raise ValueError(f"Prompt ID not found: {prompt_id}")

        results = []

        for prompt_data in prompts:
            prompt_text = prompt_data["prompt"] if isinstance(prompt_data, dict) else prompt_data
            current_prompt_id = prompt_data.get("id", "custom") if isinstance(prompt_data, dict) else "custom"

            logger.info(f"Benchmarking prompt: {current_prompt_id}")
            logger.info(f"  Warmup runs: {self.config.warmup_runs}, Measured runs: {self.config.measured_runs}")

            # Warmup runs
            logger.debug("Running warmup iterations...")
            for i in range(self.config.warmup_runs):
                self._run_single_inference(prompt_text)
                logger.debug(f"Warmup run {i + 1}/{self.config.warmup_runs} complete")

            # Measured runs with resource monitoring
            logger.debug("Running measured iterations...")
            self._resource_monitor.start()

            run_metrics: list[InferenceMetrics] = []
            for i in range(self.config.measured_runs):
                metrics = self._run_single_inference(prompt_text)
                run_metrics.append(metrics)
                logger.debug(
                    f"Run {i + 1}/{self.config.measured_runs}: "
                    f"TTFT={metrics.time_to_first_token_ms:.1f}ms, "
                    f"TPS={metrics.tokens_per_second:.1f}"
                )

            resource_utilization = self._resource_monitor.stop()

            # Aggregate metrics across runs
            ttft_values = [m.time_to_first_token_ms for m in run_metrics]
            tps_values = [m.tokens_per_second for m in run_metrics]
            latency_values = [m.total_latency_ms for m in run_metrics]

            # Get model hash and quantization info
            model_hash = None
            quantization = None
            if self._model_info:
                model_hash = self._model_info.get("digest", "")[:12]
                details = self._model_info.get("details", {})
                quantization = details.get("quantization_level", "")

            result = LLMResult(
                model_name=self.config.model_name,
                model_size=self.config.model_size,
                model_hash=model_hash,
                quantization=quantization,
                prompt_id=current_prompt_id,
                prompt_tokens=run_metrics[0].prompt_tokens if run_metrics else 0,
                output_tokens=run_metrics[0].tokens_generated if run_metrics else 0,
                time_to_first_token_ms=round(run_metrics[0].time_to_first_token_ms, 2) if run_metrics else 0,
                tokens_per_second=round(run_metrics[0].tokens_per_second, 2) if run_metrics else 0,
                total_latency_ms=round(run_metrics[0].total_latency_ms, 2) if run_metrics else 0,
                resource_utilization=resource_utilization,
                warmup_runs=self.config.warmup_runs,
                measured_runs=self.config.measured_runs,
                # Aggregated statistics
                ttft_mean_ms=round(statistics.mean(ttft_values), 2),
                ttft_std_ms=round(statistics.stdev(ttft_values), 2) if len(ttft_values) > 1 else 0.0,
                tps_mean=round(statistics.mean(tps_values), 2),
                tps_std=round(statistics.stdev(tps_values), 2) if len(tps_values) > 1 else 0.0,
                latency_mean_ms=round(statistics.mean(latency_values), 2),
                latency_std_ms=round(statistics.stdev(latency_values), 2) if len(latency_values) > 1 else 0.0,
            )

            results.append(result)

        return results


def run_llm_benchmark(
    model_name: str = "llama2:7b",
    model_size: str = "7B",
    warmup_runs: int = 3,
    measured_runs: int = 10,
    prompts: Optional[list] = None,
    api_base: str = OLLAMA_API_BASE,
) -> list[LLMResult]:
    """Convenience function to run an LLM benchmark.

    Args:
        model_name: Ollama model name
        model_size: Model size category (7B, 8B, 9B)
        warmup_runs: Number of warmup iterations
        measured_runs: Number of measured iterations
        prompts: Custom prompts (uses defaults if None)
        api_base: Ollama API base URL

    Returns:
        List of LLMResult for each prompt
    """
    config = LLMBenchmarkConfig(
        model_name=model_name,
        model_size=model_size,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        api_base=api_base,
        prompts=prompts or [],
    )

    runner = LLMBenchmarkRunner(config)
    return runner.run()


def get_available_models(size_group: Optional[str] = None) -> list[str]:
    """Get list of supported LLM models.

    Args:
        size_group: Optional filter by size (7B, 8B, 9B)

    Returns:
        List of model names
    """
    if size_group:
        return LLM_MODELS.get(size_group, [])
    return [model for models in LLM_MODELS.values() for model in models]


def check_ollama_status() -> dict:
    """Check Ollama server status and available models.

    Returns:
        Dict with status information
    """
    client = OllamaClient()

    status = {
        "server_running": client.is_available(),
        "available_models": [],
        "api_base": OLLAMA_API_BASE,
    }

    if status["server_running"]:
        models = client.list_models()
        status["available_models"] = [m.get("name") for m in models]

    return status
