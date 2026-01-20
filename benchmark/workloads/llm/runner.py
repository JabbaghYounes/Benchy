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
    """Configuration for LLM benchmark runs.

    Phase 3 (Model Expansion PRD) defaults for 1B/3B models:
    - warmup_runs: 2 (per PRD section 10)
    - measured_runs: 10
    - temperature: 0.2 (per PRD section 8)
    - top_p: 0.95 (per PRD section 8)
    - max_tokens: 256
    - streaming: disabled (per PRD section 8)
    - prompt_batch_size: 3 (per PRD section 10)
    """

    model_name: str = "llama2:7b"
    model_size: str = "7B"
    warmup_runs: int = 3  # Default for 7B+; use 2 for 1B/3B
    measured_runs: int = 10
    api_base: str = OLLAMA_API_BASE
    # Generation parameters for deterministic output
    # Phase 3 - Task 3.2: Decoding parameter enforcement
    temperature: float = 0.0  # 0.0 for legacy, 0.2 for 1B/3B
    top_p: float = 1.0  # 1.0 for legacy, 0.95 for 1B/3B
    top_k: int = 1
    seed: int = 42
    max_tokens: int = 256
    # Streaming control (Phase 3: disabled for 1B/3B)
    use_streaming: bool = True  # False for 1B/3B models
    # Prompts to use
    prompts: list[str] = field(default_factory=list)
    # Phase 4 - Batching for small models
    prompt_batch_size: int = 1  # 3 for 1B/3B models per PRD
    # Prompt set selection
    prompt_set: str = "legacy"  # "legacy", "general", "code", "all"
    # Phase 3 - Memory preflight
    enable_memory_check: bool = False  # True for 1B/3B models

    @classmethod
    def for_lightweight_model(
        cls,
        model_name: str,
        model_size: str,
        api_base: str = OLLAMA_API_BASE,
    ) -> "LLMBenchmarkConfig":
        """Create config optimized for 1B/3B lightweight models.

        Per Model Expansion PRD sections 8 and 10.
        """
        return cls(
            model_name=model_name,
            model_size=model_size,
            warmup_runs=2,  # PRD: 2 warmup runs
            measured_runs=10,
            api_base=api_base,
            temperature=0.2,  # PRD: temperature 0.2
            top_p=0.95,  # PRD: top_p 0.95
            top_k=1,
            seed=42,
            max_tokens=256,
            use_streaming=False,  # PRD: streaming disabled
            prompt_batch_size=3,  # PRD: batch 3 prompts
            prompt_set="all",  # Use new prompt sets
            enable_memory_check=True,  # PRD: memory preflight
        )


# Model groups as defined in PRD
# Phase 1: Extended with 1B and 3B models per Model Expansion PRD
LLM_MODELS = {
    "1B": ["llama3.2:1b", "granite3.1-moe:1b", "sailor2:1b"],
    "3B": ["llama3.2:3b", "granite3.1-moe:3b", "starcoder2:3b"],
    "7B": ["llama2:7b", "mistral:7b", "olmo2:7b"],
    "8B": ["llama3.1:8b", "dolphin3:8b", "dolphin-llama3:8b"],
    "9B": ["gemma2:9b"],
}


@dataclass
class ModelMetadata:
    """Metadata for LLM models per Model Expansion PRD Task 1.2."""
    parameter_group: str  # "1B", "3B", "7B", "8B", "9B"
    architecture: str  # "dense" or "moe"
    specialization: str  # "general" or "code"


# Model metadata registry - Task 1.2
MODEL_METADATA: dict[str, ModelMetadata] = {
    # 1B models
    "llama3.2:1b": ModelMetadata("1B", "dense", "general"),
    "granite3.1-moe:1b": ModelMetadata("1B", "moe", "general"),
    "sailor2:1b": ModelMetadata("1B", "dense", "general"),
    # 3B models
    "llama3.2:3b": ModelMetadata("3B", "dense", "general"),
    "granite3.1-moe:3b": ModelMetadata("3B", "moe", "general"),
    "starcoder2:3b": ModelMetadata("3B", "dense", "code"),
    # 7B models
    "llama2:7b": ModelMetadata("7B", "dense", "general"),
    "mistral:7b": ModelMetadata("7B", "dense", "general"),
    "olmo2:7b": ModelMetadata("7B", "dense", "general"),
    # 8B models
    "llama3.1:8b": ModelMetadata("8B", "dense", "general"),
    "dolphin3:8b": ModelMetadata("8B", "dense", "general"),
    "dolphin-llama3:8b": ModelMetadata("8B", "dense", "general"),
    # 9B models
    "gemma2:9b": ModelMetadata("9B", "dense", "general"),
}


def get_model_metadata(model_name: str) -> Optional[ModelMetadata]:
    """Get metadata for a model.

    Args:
        model_name: Ollama model name

    Returns:
        ModelMetadata or None if not found
    """
    return MODEL_METADATA.get(model_name)

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


# Phase 2 - Prompt Sets for 1B/3B Models (Model Expansion PRD)
# Task 2.1: General Reasoning Prompts
GENERAL_REASONING_PROMPTS = [
    {
        "id": "summarization",
        "prompt": "Summarize the following in one sentence: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "expected_tokens": 50,
        "category": "general",
    },
    {
        "id": "instruction_following",
        "prompt": "List exactly three benefits of regular exercise. Use bullet points.",
        "expected_tokens": 60,
        "category": "general",
    },
    {
        "id": "short_reasoning",
        "prompt": "If all cats are mammals, and all mammals are animals, what can we conclude about cats? Answer in one sentence.",
        "expected_tokens": 30,
        "category": "general",
    },
]

# Task 2.2: Code Generation Prompts
CODE_GENERATION_PROMPTS = [
    {
        "id": "function_generation",
        "prompt": "Write a function called 'add_numbers' that takes two parameters and returns their sum.",
        "expected_tokens": 80,
        "category": "code",
    },
    {
        "id": "code_completion",
        "prompt": "Complete this function:\ndef is_even(n):\n    # Return True if n is even, False otherwise\n",
        "expected_tokens": 40,
        "category": "code",
    },
    {
        "id": "syntax_validation",
        "prompt": "Write a for loop that prints numbers 1 to 5.",
        "expected_tokens": 50,
        "category": "code",
    },
]

# Combined prompt sets for lightweight models
LIGHTWEIGHT_MODEL_PROMPTS = GENERAL_REASONING_PROMPTS + CODE_GENERATION_PROMPTS


class PromptSet:
    """Prompt set selector for different model types."""

    GENERAL = "general"
    CODE = "code"
    ALL = "all"
    LEGACY = "legacy"  # Original prompts for backward compatibility

    @staticmethod
    def get_prompts(
        prompt_set: str = "all",
        model_specialization: str = "general",
    ) -> list[dict]:
        """Get prompts based on set type and model specialization.

        Args:
            prompt_set: "general", "code", "all", or "legacy"
            model_specialization: Model's specialization from metadata

        Returns:
            List of prompt dictionaries
        """
        if prompt_set == PromptSet.LEGACY:
            return DEFAULT_PROMPTS
        elif prompt_set == PromptSet.GENERAL:
            return GENERAL_REASONING_PROMPTS
        elif prompt_set == PromptSet.CODE:
            return CODE_GENERATION_PROMPTS
        elif prompt_set == PromptSet.ALL:
            # For code-specialized models, prioritize code prompts
            if model_specialization == "code":
                return CODE_GENERATION_PROMPTS + GENERAL_REASONING_PROMPTS
            return LIGHTWEIGHT_MODEL_PROMPTS
        else:
            return DEFAULT_PROMPTS


# Phase 3 - Execution Constraints
class MemoryPreflightError(Exception):
    """Raised when memory preflight check fails."""
    pass


class PromptComplianceError(Exception):
    """Raised when prompt compliance validation fails."""
    pass


def check_memory_preflight(min_available_gb: float = 1.0) -> dict:
    """Phase 3 - Task 3.1: Memory preflight check.

    Checks available RAM before benchmark execution.
    Aborts if insufficient memory or swap usage required.

    Args:
        min_available_gb: Minimum required available RAM in GB

    Returns:
        Dict with memory status

    Raises:
        MemoryPreflightError: If memory check fails
    """
    import psutil

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    available_gb = memory.available / (1024 ** 3)
    total_gb = memory.total / (1024 ** 3)
    swap_used_gb = swap.used / (1024 ** 3)

    status = {
        "total_gb": round(total_gb, 2),
        "available_gb": round(available_gb, 2),
        "used_percent": memory.percent,
        "swap_used_gb": round(swap_used_gb, 2),
        "passed": True,
        "reason": None,
    }

    # Check if swap is being used significantly
    if swap_used_gb > 0.5:
        status["passed"] = False
        status["reason"] = f"Swap usage detected ({swap_used_gb:.2f}GB). Model may cause OOM."
        logger.warning(f"Memory preflight warning: {status['reason']}")

    # Check available memory
    if available_gb < min_available_gb:
        status["passed"] = False
        status["reason"] = (
            f"Insufficient memory: {available_gb:.2f}GB available, "
            f"{min_available_gb:.2f}GB required"
        )
        logger.error(f"Memory preflight failed: {status['reason']}")

    return status


def validate_prompt_compliance(
    client: "OllamaClient",
    model_name: str,
    test_prompt: str = "Say 'hello' and nothing else.",
    config: Optional[LLMBenchmarkConfig] = None,
) -> dict:
    """Phase 2 - Task 2.3: Prompt compliance validation.

    Validates that a model:
    - Respects system prompts
    - Produces non-empty output
    - Has no streaming artifacts or tool calls

    Args:
        client: OllamaClient instance
        model_name: Model to validate
        test_prompt: Simple test prompt
        config: Optional config for generation parameters

    Returns:
        Dict with compliance status
    """
    config = config or LLMBenchmarkConfig()

    result = {
        "model": model_name,
        "compliant": True,
        "issues": [],
        "response": None,
    }

    try:
        response = client.generate(
            model=model_name,
            prompt=test_prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            seed=config.seed,
            max_tokens=50,
        )

        response_text = response.get("response", "")
        result["response"] = response_text

        # Check for empty output
        if not response_text or not response_text.strip():
            result["compliant"] = False
            result["issues"].append("Empty response")

        # Check for tool call artifacts
        tool_indicators = ["<tool_call>", "<function_call>", "```tool", "[TOOL]"]
        for indicator in tool_indicators:
            if indicator in response_text:
                result["compliant"] = False
                result["issues"].append(f"Tool call artifact detected: {indicator}")

        # Check for streaming artifacts
        stream_indicators = ["<|im_", "<|assistant|>", "[STREAMING]"]
        for indicator in stream_indicators:
            if indicator in response_text:
                result["compliant"] = False
                result["issues"].append(f"Streaming artifact detected: {indicator}")

    except Exception as e:
        result["compliant"] = False
        result["issues"].append(f"Generation failed: {str(e)}")

    if not result["compliant"]:
        logger.warning(
            f"Prompt compliance check failed for {model_name}: {result['issues']}"
        )

    return result


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
    """Metrics from a single inference run.

    Phase 5: Extended with peak_memory_mb and truncated flag.
    """

    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    prompt_tokens: int
    response_text: str
    # Phase 5 additions
    peak_memory_mb: Optional[float] = None
    truncated: bool = False  # True if output was truncated at max_tokens


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

        Supports both streaming and non-streaming modes per Phase 3 requirements.

        Args:
            prompt: The prompt to send to the model

        Returns:
            InferenceMetrics with timing and token data
        """
        import psutil

        # Capture memory before inference
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.perf_counter()
        first_token_time = None
        tokens_generated = 0
        response_text = ""
        prompt_tokens = len(prompt.split())
        truncated = False

        if self.config.use_streaming:
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
                    # Check if truncated
                    if tokens_generated >= self.config.max_tokens:
                        truncated = True
                    break
        else:
            # Non-streaming mode for 1B/3B models (Phase 3)
            response = self.client.generate(
                model=self.config.model_name,
                prompt=prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                seed=self.config.seed,
                max_tokens=self.config.max_tokens,
            )

            # For non-streaming, TTFT is approximated as the full latency
            # divided by number of tokens (rough estimate)
            first_token_time = start_time  # Can't measure TTFT without streaming

            response_text = response.get("response", "")
            tokens_generated = response.get("eval_count", len(response_text.split()))
            prompt_tokens = response.get("prompt_eval_count", len(prompt.split()))

            if tokens_generated >= self.config.max_tokens:
                truncated = True

        end_time = time.perf_counter()

        # Capture peak memory
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_memory_mb = max(mem_before, mem_after)

        # Calculate metrics
        total_latency_ms = (end_time - start_time) * 1000

        if self.config.use_streaming and first_token_time:
            ttft_ms = (first_token_time - start_time) * 1000
        else:
            # For non-streaming, estimate TTFT as latency / tokens * 1
            ttft_ms = total_latency_ms / max(tokens_generated, 1)

        # Calculate tokens per second (excluding TTFT for generation speed)
        generation_time_s = (end_time - (first_token_time or start_time))
        tps = tokens_generated / generation_time_s if generation_time_s > 0 else 0.0

        return InferenceMetrics(
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_generated=tokens_generated,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
            response_text=response_text,
            peak_memory_mb=peak_memory_mb,
            truncated=truncated,
        )

    def run(self, prompt_id: Optional[str] = None) -> list[LLMResult]:
        """Run the complete benchmark.

        Phase 3-5 enhancements:
        - Memory preflight check for 1B/3B models
        - New prompt sets for lightweight models
        - Model metadata in results
        - Peak memory tracking

        Args:
            prompt_id: Optional specific prompt ID to benchmark. If None, runs all prompts.

        Returns:
            List of LLMResult for each prompt
        """
        # Phase 3 - Task 3.1: Memory preflight check
        if self.config.enable_memory_check:
            # Minimum memory varies by model size
            min_memory_gb = {
                "1B": 2.0,
                "3B": 4.0,
                "7B": 8.0,
                "8B": 10.0,
                "9B": 12.0,
            }.get(self.config.model_size, 4.0)

            mem_status = check_memory_preflight(min_memory_gb)
            if not mem_status["passed"]:
                raise MemoryPreflightError(
                    f"Memory preflight failed for {self.config.model_name}: "
                    f"{mem_status['reason']}"
                )
            logger.info(
                f"Memory preflight passed: {mem_status['available_gb']:.1f}GB available"
            )

        # Ensure model is available
        if not self._ensure_model_available():
            raise RuntimeError(f"Failed to load model: {self.config.model_name}")

        # Get model metadata for prompt selection
        metadata = get_model_metadata(self.config.model_name)
        model_specialization = metadata.specialization if metadata else "general"

        # Get prompts to run (Phase 2: Support new prompt sets)
        if self.config.prompts:
            prompts = self.config.prompts
        elif self.config.prompt_set != "legacy":
            prompts = PromptSet.get_prompts(
                self.config.prompt_set,
                model_specialization,
            )
        else:
            prompts = DEFAULT_PROMPTS

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

            # Phase 5 - Peak memory and truncation detection
            peak_memory_values = [m.peak_memory_mb for m in run_metrics if m.peak_memory_mb]
            peak_memory_mb = max(peak_memory_values) if peak_memory_values else None
            any_truncated = any(m.truncated for m in run_metrics)

            # Get prompt category from prompt data
            prompt_category = prompt_data.get("category", "general") if isinstance(prompt_data, dict) else "general"

            # Phase 6 - Median, min, max aggregation
            ttft_median_ms = round(statistics.median(ttft_values), 2) if ttft_values else None
            ttft_min_ms = round(min(ttft_values), 2) if ttft_values else None
            ttft_max_ms = round(max(ttft_values), 2) if ttft_values else None
            tps_median = round(statistics.median(tps_values), 2) if tps_values else None
            tps_min = round(min(tps_values), 2) if tps_values else None
            tps_max = round(max(tps_values), 2) if tps_values else None

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
                # Aggregated statistics (mean/std)
                ttft_mean_ms=round(statistics.mean(ttft_values), 2),
                ttft_std_ms=round(statistics.stdev(ttft_values), 2) if len(ttft_values) > 1 else 0.0,
                tps_mean=round(statistics.mean(tps_values), 2),
                tps_std=round(statistics.stdev(tps_values), 2) if len(tps_values) > 1 else 0.0,
                latency_mean_ms=round(statistics.mean(latency_values), 2),
                latency_std_ms=round(statistics.stdev(latency_values), 2) if len(latency_values) > 1 else 0.0,
                # Model Expansion PRD - Phase 1 metadata
                parameter_group=metadata.parameter_group if metadata else None,
                architecture=metadata.architecture if metadata else None,
                specialization=metadata.specialization if metadata else None,
                # Model Expansion PRD - Phase 5 metrics
                peak_memory_mb=peak_memory_mb,
                truncated=any_truncated,
                prompt_category=prompt_category,
                # Phase 6 aggregation (median, min, max)
                ttft_median_ms=ttft_median_ms,
                ttft_min_ms=ttft_min_ms,
                ttft_max_ms=ttft_max_ms,
                tps_median=tps_median,
                tps_min=tps_min,
                tps_max=tps_max,
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


def verify_reproducibility(
    model_name: str,
    model_size: str,
    num_runs: int = 3,
    variance_threshold: float = 0.15,
    api_base: str = OLLAMA_API_BASE,
) -> dict:
    """Phase 7 - Task 7.1: Verify benchmark reproducibility.

    Re-runs a model multiple times and checks that metrics variance
    is within acceptable bounds.

    Args:
        model_name: Ollama model name to test
        model_size: Model size category
        num_runs: Number of benchmark runs to compare
        variance_threshold: Maximum acceptable coefficient of variation (default 15%)
        api_base: Ollama API base URL

    Returns:
        Dict with reproducibility status and metrics
    """
    logger.info(f"Verifying reproducibility for {model_name} ({num_runs} runs)")

    # Get appropriate config based on model size
    if model_size in ("1B", "3B"):
        config = LLMBenchmarkConfig.for_lightweight_model(
            model_name=model_name,
            model_size=model_size,
            api_base=api_base,
        )
    else:
        config = LLMBenchmarkConfig(
            model_name=model_name,
            model_size=model_size,
            api_base=api_base,
        )

    # Run benchmarks multiple times
    all_results: list[list[LLMResult]] = []
    for i in range(num_runs):
        logger.info(f"Reproducibility run {i + 1}/{num_runs}")
        runner = LLMBenchmarkRunner(config)
        results = runner.run()
        all_results.append(results)

    # Aggregate metrics across runs
    ttft_means = []
    tps_means = []

    for run_results in all_results:
        for result in run_results:
            if result.ttft_mean_ms is not None:
                ttft_means.append(result.ttft_mean_ms)
            if result.tps_mean is not None:
                tps_means.append(result.tps_mean)

    # Calculate coefficient of variation (CV = std/mean)
    def calc_cv(values):
        if len(values) < 2:
            return 0.0
        mean = statistics.mean(values)
        if mean == 0:
            return 0.0
        return statistics.stdev(values) / mean

    ttft_cv = calc_cv(ttft_means)
    tps_cv = calc_cv(tps_means)

    # Check if variance is acceptable
    ttft_acceptable = ttft_cv <= variance_threshold
    tps_acceptable = tps_cv <= variance_threshold
    overall_pass = ttft_acceptable and tps_acceptable

    result = {
        "model_name": model_name,
        "model_size": model_size,
        "num_runs": num_runs,
        "variance_threshold": variance_threshold,
        "passed": overall_pass,
        "metrics": {
            "ttft": {
                "mean_ms": round(statistics.mean(ttft_means), 2) if ttft_means else None,
                "std_ms": round(statistics.stdev(ttft_means), 2) if len(ttft_means) > 1 else 0.0,
                "cv": round(ttft_cv, 4),
                "acceptable": ttft_acceptable,
            },
            "tps": {
                "mean": round(statistics.mean(tps_means), 2) if tps_means else None,
                "std": round(statistics.stdev(tps_means), 2) if len(tps_means) > 1 else 0.0,
                "cv": round(tps_cv, 4),
                "acceptable": tps_acceptable,
            },
        },
    }

    if overall_pass:
        logger.info(f"Reproducibility check PASSED for {model_name}")
    else:
        logger.warning(
            f"Reproducibility check FAILED for {model_name}: "
            f"TTFT CV={ttft_cv:.2%}, TPS CV={tps_cv:.2%}"
        )

    return result


def verify_parameter_group_reproducibility(
    parameter_group: str = "1B",
    variance_threshold: float = 0.15,
    api_base: str = OLLAMA_API_BASE,
) -> dict:
    """Phase 7 - Task 7.1: Verify reproducibility for one model per group.

    Args:
        parameter_group: Parameter group to test ("1B", "3B", "7B", etc.)
        variance_threshold: Maximum acceptable coefficient of variation
        api_base: Ollama API base URL

    Returns:
        Dict with reproducibility results for the selected model
    """
    models = LLM_MODELS.get(parameter_group, [])
    if not models:
        return {
            "error": f"No models found for parameter group: {parameter_group}",
            "passed": False,
        }

    # Select first model from the group
    test_model = models[0]
    logger.info(f"Testing reproducibility for {parameter_group} group with {test_model}")

    return verify_reproducibility(
        model_name=test_model,
        model_size=parameter_group,
        variance_threshold=variance_threshold,
        api_base=api_base,
    )
