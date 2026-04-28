# Tests for LLM prewarm + configurable timeout (Issue 12).
#
# We don't have a live Ollama server in CI, so this test stubs out the
# OllamaClient methods on the runner and asserts the prewarm call is
# issued exactly once with the configured prewarm timeout and
# keep_alive=-1 — that's the contract that prevents cold-load latency
# from contaminating the measured loop.
from __future__ import annotations

from unittest.mock import MagicMock

from benchmark.workloads.llm.runner import (
    LLMBenchmarkConfig,
    LLMBenchmarkRunner,
    OllamaClient,
)


def test_config_has_timeout_fields_with_documented_defaults():
    cfg = LLMBenchmarkConfig(model_name="llama2:7b", model_size="7B")
    assert cfg.http_timeout_seconds == 600
    assert cfg.prewarm_timeout_seconds == 1800


def test_ollama_client_honors_request_timeout():
    client = OllamaClient(request_timeout=900)
    assert client.request_timeout == 900


def test_ollama_client_default_timeout_is_600():
    client = OllamaClient()
    assert client.request_timeout == 600


def test_prewarm_issues_one_keep_alive_call_with_long_timeout(monkeypatch):
    cfg = LLMBenchmarkConfig(
        model_name="llama2:7b",
        model_size="7B",
        prewarm_timeout_seconds=1800,
    )
    runner = LLMBenchmarkRunner(cfg)

    fake_generate = MagicMock(return_value={"response": "ok"})
    monkeypatch.setattr(runner.client, "generate", fake_generate)

    runner._prewarm_model()

    fake_generate.assert_called_once()
    kwargs = fake_generate.call_args.kwargs
    assert kwargs["model"] == "llama2:7b"
    assert kwargs["max_tokens"] == 1
    assert kwargs["timeout"] == 1800
    assert kwargs["keep_alive"] == "-1"


def test_prewarm_does_not_raise_on_failure(monkeypatch):
    # The prewarm path must not abort the benchmark on transient HTTP
    # errors. The first measured request will surface a real failure
    # if the server is genuinely down, with a more useful message.
    import requests

    cfg = LLMBenchmarkConfig(model_name="llama2:7b", model_size="7B")
    runner = LLMBenchmarkRunner(cfg)

    def boom(**_kwargs):
        raise requests.ConnectionError("simulated")

    monkeypatch.setattr(runner.client, "generate", boom)

    # Should not raise.
    runner._prewarm_model()


def test_runner_passes_config_timeout_to_client():
    cfg = LLMBenchmarkConfig(
        model_name="llama2:7b",
        model_size="7B",
        http_timeout_seconds=750,
    )
    runner = LLMBenchmarkRunner(cfg)
    assert runner.client.request_timeout == 750
