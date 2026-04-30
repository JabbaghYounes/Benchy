# Tests for HAR generation: end-node truncation table, pipeline
# threading, diagnostic-hint parser, and the fallback-retry path in
# `_generate_with_sdk`.
#
# Hailo's parser cannot ingest the tail of a YOLO head (DFL/Reshape on
# detection-shaped tasks; Cos/Sin for OBB angles). The fix is to
# truncate the graph at the right end-nodes and decode host-side. Two
# mechanisms cooperate:
#
#   1. END_NODE_TABLE — static (yolo_version, task) -> list[str] map.
#      Populated from Hailo's diagnostic warnings captured during real
#      compile runs. Pipeline-driven; deterministic; cheap to test.
#   2. parse_end_node_hint + retry path — when the table misses,
#      _generate_with_sdk re-runs once using whatever Hailo's error
#      message suggests. Robust to Ultralytics export-naming changes
#      and to YOLO versions / tasks the table doesn't yet cover.
#
# These tests mock hailo_sdk_client so they pass on any dev host.
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmark.schemas import YOLOTask
from benchmark.workloads.yolo.conversion import har_generator
from benchmark.workloads.yolo.conversion.har_generator import (
    END_NODE_HINT_RE,
    END_NODE_TABLE,
    HARGenerator,
    HARGeneratorConfig,
    get_end_nodes,
    parse_end_node_hint,
)


# ----------------------------------------------------- coverage of END_NODE_TABLE


# (yolo_version, task) pairs that aren't in END_NODE_TABLE because we
# haven't captured Hailo's diagnostic hint for them yet. The
# diagnostic-hint fallback parser (`parse_end_node_hint` +
# `_generate_with_sdk` retry) handles them at runtime, so they are
# soft gaps — not blockers, just future work.
#
# Update procedure when we capture a new hint: add the end-nodes to
# END_NODE_TABLE and remove the (version, task) tuple from this set.
KNOWN_GAPS = {
    ("v8", YOLOTask.POSE),
    ("v8", YOLOTask.CLASSIFICATION),
    ("v11", YOLOTask.CLASSIFICATION),
    ("v26", YOLOTask.CLASSIFICATION),
}


def test_end_node_table_covers_all_known_combinations():
    """Every (version, task) Benchy claims Hailo support for is either
    in END_NODE_TABLE or explicitly marked as a known gap. Adding a
    new task / version without making a conscious choice between
    those two should fail this test."""
    from benchmark.workloads.yolo.execution import HAILO_SUPPORTED_TASKS

    missing = []
    for version, tasks in HAILO_SUPPORTED_TASKS.items():
        for task in tasks:
            key = (version, task)
            in_table = key in END_NODE_TABLE
            in_gaps = key in KNOWN_GAPS
            if in_table and in_gaps:
                pytest.fail(
                    f"({version}, {task.value}) is in BOTH END_NODE_TABLE "
                    f"and KNOWN_GAPS — drop it from KNOWN_GAPS now."
                )
            if not in_table and not in_gaps:
                missing.append(key)

    assert missing == [], (
        f"These (version, task) combinations are HAILO_SUPPORTED but not "
        f"in END_NODE_TABLE and not marked as known gaps. Either add "
        f"them to END_NODE_TABLE (preferred) or to KNOWN_GAPS in this "
        f"test file: {missing}"
    )


def test_end_node_table_entries_have_nonempty_lists():
    """Sanity check the table contents — every value should be a
    non-empty list of strings (an empty list would mean "tell Hailo to
    parse to graph end" which is exactly the failure mode this table
    exists to avoid)."""
    for key, nodes in END_NODE_TABLE.items():
        version, task = key
        assert isinstance(nodes, list), f"{key} has non-list value"
        assert nodes, f"{key} has empty end-node list"
        for n in nodes:
            assert isinstance(n, str) and n.startswith("/"), (
                f"{key} contains non-onnx-name entry: {n!r}"
            )


def test_get_end_nodes_returns_list_or_none():
    """get_end_nodes returns the list verbatim when the lookup hits
    and None when it misses. Crucial: it does not raise."""
    nodes = get_end_nodes("v11", YOLOTask.SEGMENTATION)
    assert nodes == END_NODE_TABLE[("v11", YOLOTask.SEGMENTATION)]

    # A combination that's intentionally not in the table.
    assert get_end_nodes("v8", YOLOTask.POSE) is None


# --------------------------------------------- pipeline -> HARGeneratorConfig wiring


def test_end_nodes_threaded_into_har_config(monkeypatch, tmp_path):
    """The pipeline must populate HARGeneratorConfig.end_nodes from
    END_NODE_TABLE before invoking the generator. We swap in a spy
    HARGenerator and verify the config it receives carries the
    correct end-node list for (v11, SEGMENTATION)."""
    from benchmark.workloads.yolo.conversion.pipeline import (
        ConversionConfig,
        ModelConversionPipeline,
    )

    captured: dict = {}

    class _SpyGenerator:
        def __init__(self, _cache=None):
            pass

        def is_available(self):
            return True

        def is_model_zoo_available(self):
            return False

        def check_onnx_compatibility(self, _onnx_path):
            return {"compatible": True, "warnings": [], "errors": []}

        def generate(self, *, onnx_path, model_name, yolo_version, task,
                     config, force=False):
            captured["config"] = config
            captured["yolo_version"] = yolo_version
            captured["task"] = task
            har = tmp_path / f"{Path(model_name).stem}.har"
            har.write_bytes(b"FAKE-HAR")
            return har

    pipeline = ModelConversionPipeline()
    pipeline.har_generator = _SpyGenerator()

    # Fake ONNX file; the compatibility check is mocked to pass.
    onnx_path = tmp_path / "yolo11n-seg.onnx"
    onnx_path.write_bytes(b"FAKE-ONNX")

    config = ConversionConfig(target_device="hailo8")
    pipeline._run_har_generation(
        onnx_path=onnx_path,
        model_name="yolo11n-seg.pt",
        yolo_version="v11",
        task=YOLOTask.SEGMENTATION,
        config=config,
    )

    assert captured["yolo_version"] == "v11"
    assert captured["task"] == YOLOTask.SEGMENTATION
    har_cfg = captured["config"]
    assert isinstance(har_cfg, HARGeneratorConfig)
    assert har_cfg.end_nodes == END_NODE_TABLE[("v11", YOLOTask.SEGMENTATION)]
    assert har_cfg.target_device == "hailo8"


def test_end_nodes_none_for_table_miss(monkeypatch, tmp_path):
    """When the (version, task) pair isn't in the table, the pipeline
    must still build a config — just with end_nodes=None — so the
    diagnostic-hint fallback in _generate_with_sdk has a chance to
    recover. A KeyError or empty-list here would mask that path."""
    from benchmark.workloads.yolo.conversion.pipeline import (
        ConversionConfig,
        ModelConversionPipeline,
    )

    captured: dict = {}

    class _SpyGenerator:
        def __init__(self, _cache=None):
            pass

        def is_available(self):
            return True

        def is_model_zoo_available(self):
            return False

        def check_onnx_compatibility(self, _onnx_path):
            return {"compatible": True, "warnings": [], "errors": []}

        def generate(self, *, onnx_path, model_name, yolo_version, task,
                     config, force=False):
            captured["config"] = config
            har = tmp_path / "x.har"
            har.write_bytes(b"x")
            return har

    pipeline = ModelConversionPipeline()
    pipeline.har_generator = _SpyGenerator()

    onnx_path = tmp_path / "yolov8n-pose.onnx"
    onnx_path.write_bytes(b"x")

    pipeline._run_har_generation(
        onnx_path=onnx_path,
        model_name="yolov8n-pose.pt",
        yolo_version="v8",
        task=YOLOTask.POSE,
        config=ConversionConfig(target_device="hailo10h"),
    )

    assert captured["config"].end_nodes is None


# -------------------------------------------- diagnostic-hint regex parser


def test_diagnostic_parser_extracts_end_nodes_basic():
    """The exact phrasing observed in compile-h8.log."""
    msg = (
        "UnsupportedShuffleLayerError: Failed to parse model. "
        "Try using these end node names: /model.23/Concat, "
        "/model.23/Sigmoid_1, /model.23/Mul"
    )
    nodes = parse_end_node_hint(msg)
    assert nodes == [
        "/model.23/Concat",
        "/model.23/Sigmoid_1",
        "/model.23/Mul",
    ]


def test_diagnostic_parser_handles_trailing_paren():
    """Hailo often wraps the suggestion in parentheses with extra
    diagnostic text after the closing paren."""
    msg = (
        "Parse failure (use these end node names: /model.22/Concat, "
        "/model.22/Sigmoid_1, /model.22/Mul) — falling back to direct SDK"
    )
    nodes = parse_end_node_hint(msg)
    assert nodes == [
        "/model.22/Concat",
        "/model.22/Sigmoid_1",
        "/model.22/Mul",
    ]


def test_diagnostic_parser_handles_extra_whitespace():
    msg = (
        "use these end node names :   /model.23/Concat ,  /model.23/Mul_3 , "
        "/model.23/Sigmoid_1, /model.23/Sigmoid"
    )
    nodes = parse_end_node_hint(msg)
    assert nodes == [
        "/model.23/Concat",
        "/model.23/Mul_3",
        "/model.23/Sigmoid_1",
        "/model.23/Sigmoid",
    ]


def test_diagnostic_parser_returns_none_on_no_hint():
    """An error string with no hint must return None, not a partial
    list. The retry path uses None as the signal to give up."""
    assert parse_end_node_hint("totally unrelated error") is None
    assert parse_end_node_hint("") is None
    assert parse_end_node_hint(None) is None  # type: ignore[arg-type]


def test_diagnostic_parser_case_insensitive():
    """SDK versions vary 'use'/'using' and capitalization; the regex
    must match either form."""
    a = parse_end_node_hint("USE THESE END NODE NAMES: /a, /b")
    b = parse_end_node_hint("Using these end node names: /a, /b")
    assert a == ["/a", "/b"]
    assert b == ["/a", "/b"]


# ---------------------------------------------------- fallback-retry path


@pytest.fixture
def fake_hailo_sdk(monkeypatch):
    """Inject a fake hailo_sdk_client module so _generate_with_sdk's
    `from hailo_sdk_client import ClientRunner` succeeds without the
    real SDK installed.

    Yields a list — `runners` — capturing every ClientRunner instance
    the test creates. Each instance carries a list `calls` of
    translate_onnx_model invocations and a configurable
    `translate_should_raise` to simulate parse failures.
    """
    runners: list = []

    class _FakeClientRunner:
        def __init__(self, hw_arch="hailo8"):
            self.hw_arch = hw_arch
            self.calls: list = []
            self.translate_should_raise: Exception | None = None
            self.saved_har_to: str | None = None
            runners.append(self)

        def translate_onnx_model(self, onnx_path, *, net_name=None,
                                 start_node_names=None, end_node_names=None):
            self.calls.append({
                "onnx_path": onnx_path,
                "net_name": net_name,
                "start_node_names": start_node_names,
                "end_node_names": end_node_names,
            })
            if self.translate_should_raise is not None:
                err = self.translate_should_raise
                self.translate_should_raise = None  # only fire once
                raise err
            return ("fake-hn", "fake-npz")

        def get_hn_model(self):
            m = MagicMock()
            m.name = "fake-net"
            return m

        def save_har(self, path):
            self.saved_har_to = path
            Path(path).write_bytes(b"FAKE-HAR")

    fake_module = types.ModuleType("hailo_sdk_client")
    fake_module.ClientRunner = _FakeClientRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", fake_module)

    # The HARGenerator caches is_available; reset so the fake module
    # is picked up.
    yield runners


def _make_generator_with_mocked_metadata(monkeypatch, tmp_path):
    """HARGenerator with cache writes neutered. _update_metadata
    requires a real cache directory; tests don't care about that
    side-effect, only about the SDK call sequence."""
    gen = HARGenerator()
    monkeypatch.setattr(gen, "_update_metadata", lambda *a, **kw: None)
    monkeypatch.setattr(gen, "_validate_har", lambda *a, **kw: None)
    return gen


def test_sdk_first_call_uses_table_end_nodes(fake_hailo_sdk, monkeypatch, tmp_path):
    """When config.end_nodes is set, _generate_with_sdk passes them
    straight to translate_onnx_model on the first call. No retry,
    no diagnostic parsing."""
    gen = _make_generator_with_mocked_metadata(monkeypatch, tmp_path)
    onnx_path = tmp_path / "yolo11n-seg.onnx"
    onnx_path.write_bytes(b"x")
    out = tmp_path / "out.har"

    table_nodes = END_NODE_TABLE[("v11", YOLOTask.SEGMENTATION)]
    cfg = HARGeneratorConfig(target_device="hailo8", end_nodes=table_nodes)

    gen._generate_with_sdk(
        onnx_path=onnx_path,
        model_name="yolo11n-seg.pt",
        yolo_version="v11",
        task=YOLOTask.SEGMENTATION,
        config=cfg,
        output_path=out,
    )

    # Exactly one runner created, exactly one translate call, with the
    # table-supplied end-nodes.
    assert len(fake_hailo_sdk) == 1
    assert len(fake_hailo_sdk[0].calls) == 1
    assert fake_hailo_sdk[0].calls[0]["end_node_names"] == table_nodes
    assert out.exists()


def test_sdk_retry_path_parses_diagnostic_and_succeeds(
    fake_hailo_sdk, monkeypatch, tmp_path
):
    """When end_nodes is None and the first translate fails with a
    Hailo-style hint, the second runner instance must retry with
    end-nodes parsed from the error message."""
    gen = _make_generator_with_mocked_metadata(monkeypatch, tmp_path)
    onnx_path = tmp_path / "yolo26n-obb.onnx"
    onnx_path.write_bytes(b"x")
    out = tmp_path / "out.har"

    # Make the first ClientRunner raise; subsequent ones succeed.
    real_init = fake_hailo_sdk.__class__  # the captured-list-as-fixture quirk
    # We can't easily mutate the list before instances exist, so we
    # subclass the fake module's runner factory to flip the flag.
    fake_module = sys.modules["hailo_sdk_client"]
    original_runner_cls = fake_module.ClientRunner

    def _runner_factory(*a, **kw):
        inst = original_runner_cls(*a, **kw)
        if len(fake_hailo_sdk) == 1:
            # First runner — make it fail with a parse hint.
            inst.translate_should_raise = RuntimeError(
                "Failed to parse: try using these end node names: "
                "/model.23/Concat, /model.23/Mul"
            )
        return inst

    fake_module.ClientRunner = _runner_factory  # type: ignore[attr-defined]

    cfg = HARGeneratorConfig(target_device="hailo10h", end_nodes=None)
    gen._generate_with_sdk(
        onnx_path=onnx_path,
        model_name="yolo26n-obb.pt",
        yolo_version="v26",
        task=YOLOTask.OBB,
        config=cfg,
        output_path=out,
    )

    # Two runners created (initial + retry).
    assert len(fake_hailo_sdk) == 2
    # First call had end_node_names=None (table miss).
    assert fake_hailo_sdk[0].calls[0]["end_node_names"] is None
    # Second call used the parsed hint.
    assert fake_hailo_sdk[1].calls[0]["end_node_names"] == [
        "/model.23/Concat",
        "/model.23/Mul",
    ]
    assert out.exists()


def test_sdk_no_retry_when_end_nodes_already_supplied(
    fake_hailo_sdk, monkeypatch, tmp_path
):
    """If config.end_nodes was set and the first parse still failed,
    we must NOT retry. The most likely cause is a wrong table entry
    (Ultralytics renamed nodes); auto-retrying with parsed names would
    silently mask that and produce a HEF whose outputs don't match the
    postprocessor's expectations."""
    gen = _make_generator_with_mocked_metadata(monkeypatch, tmp_path)
    onnx_path = tmp_path / "yolov8n.onnx"
    onnx_path.write_bytes(b"x")
    out = tmp_path / "out.har"

    fake_module = sys.modules["hailo_sdk_client"]
    original_runner_cls = fake_module.ClientRunner

    def _runner_factory(*a, **kw):
        inst = original_runner_cls(*a, **kw)
        # Always fail with a hint; if the retry path were buggy, we'd
        # see two runner instances instead of one.
        inst.translate_should_raise = RuntimeError(
            "use these end node names: /model.22/Concat, /model.22/Mul"
        )
        return inst

    fake_module.ClientRunner = _runner_factory  # type: ignore[attr-defined]

    cfg = HARGeneratorConfig(
        target_device="hailo8",
        end_nodes=["/model.22/some/wrong/path"],
    )
    with pytest.raises(RuntimeError, match="configured end-nodes"):
        gen._generate_with_sdk(
            onnx_path=onnx_path,
            model_name="yolov8n.pt",
            yolo_version="v8",
            task=YOLOTask.DETECTION,
            config=cfg,
            output_path=out,
        )

    # Exactly one runner — no retry.
    assert len(fake_hailo_sdk) == 1


def test_sdk_no_retry_when_no_hint_in_error(
    fake_hailo_sdk, monkeypatch, tmp_path
):
    """A failure whose message contains no hint can't be auto-recovered.
    The error must surface as a RuntimeError mentioning END_NODE_TABLE."""
    gen = _make_generator_with_mocked_metadata(monkeypatch, tmp_path)
    onnx_path = tmp_path / "yolo26n-pose.onnx"
    onnx_path.write_bytes(b"x")
    out = tmp_path / "out.har"

    fake_module = sys.modules["hailo_sdk_client"]
    original_runner_cls = fake_module.ClientRunner

    def _runner_factory(*a, **kw):
        inst = original_runner_cls(*a, **kw)
        inst.translate_should_raise = RuntimeError(
            "Generic error with no recoverable end-node hint"
        )
        return inst

    fake_module.ClientRunner = _runner_factory  # type: ignore[attr-defined]

    cfg = HARGeneratorConfig(target_device="hailo10h", end_nodes=None)
    with pytest.raises(RuntimeError, match="END_NODE_TABLE"):
        gen._generate_with_sdk(
            onnx_path=onnx_path,
            model_name="yolo26n-pose.pt",
            yolo_version="v26",
            task=YOLOTask.POSE,
            config=cfg,
            output_path=out,
        )

    # Single runner; retry never attempted.
    assert len(fake_hailo_sdk) == 1


def test_sdk_retry_failure_surfaces_both_errors(
    fake_hailo_sdk, monkeypatch, tmp_path
):
    """When the retry also fails, both errors must be visible in the
    final RuntimeError. Otherwise debugging a wrong-hint case (Hailo's
    suggestion was itself unparseable) requires scrolling through
    layered tracebacks."""
    gen = _make_generator_with_mocked_metadata(monkeypatch, tmp_path)
    onnx_path = tmp_path / "yolo26n-seg.onnx"
    onnx_path.write_bytes(b"x")
    out = tmp_path / "out.har"

    fake_module = sys.modules["hailo_sdk_client"]
    original_runner_cls = fake_module.ClientRunner

    def _runner_factory(*a, **kw):
        inst = original_runner_cls(*a, **kw)
        # Both runners fail, but the first one provides a hint so the
        # retry is attempted.
        if len(fake_hailo_sdk) == 1:
            inst.translate_should_raise = RuntimeError(
                "use these end node names: /model.23/foo, /model.23/bar"
            )
        else:
            inst.translate_should_raise = RuntimeError(
                "retry failed too — wrong nodes"
            )
        return inst

    fake_module.ClientRunner = _runner_factory  # type: ignore[attr-defined]

    cfg = HARGeneratorConfig(target_device="hailo10h", end_nodes=None)
    with pytest.raises(RuntimeError) as excinfo:
        gen._generate_with_sdk(
            onnx_path=onnx_path,
            model_name="yolo26n-seg.pt",
            yolo_version="v26",
            task=YOLOTask.SEGMENTATION,
            config=cfg,
            output_path=out,
        )

    msg = str(excinfo.value)
    assert "/model.23/foo" in msg or "/model.23/bar" in msg
    assert "retry" in msg.lower()
    assert len(fake_hailo_sdk) == 2
