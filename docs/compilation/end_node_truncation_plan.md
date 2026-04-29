# End-node truncation: HAR generator + per-task end-node table

Plan for the next session. Scope is the HAR-generation stage of the
workstation compile pipeline — the immediate blocker keeping
`scripts/compile_workstation_hefs.sh` from producing usable HEFs for
gap models on either AI HAT+ chip.

## Problem

`python -m benchmark compile --hw-arch hailo8 --model <gap-model>.pt`
fails at HAR generation with errors like:

```
UnsupportedShuffleLayerError /model.23/dfl/Reshape
UnsupportedOperationError /model.23/Cos: Cos operation is unsupported
UnsupportedModelError /model.23/Add_3: ... must be broadcastable to ...
```

Hailo's diagnostic explicitly tells us how to fix it (truncate the
model before the unsupported tail) and gives concrete end-node names
per task. A 7-model hailo8 sweep on the workstation produced 0
usable HEFs because the pipeline never passes those end-node names
to the parser.

This is not a bug in Hailo — it's the documented Hailo workflow
(`docs/compilation/pitfalls.md` § 3 already mentions
`--end-node-names` for OBB / pose). Benchy's host-side
postprocessing (`benchmark/workloads/yolo/postprocessing.py`)
already handles the decoded outputs the truncated model would emit:
`_process_detection`, `_process_obb` (with `_rotated_nms`),
`_process_segmentation` (with `_generate_seg_masks`), `_process_pose`
(17-keypoint COCO decoder). Truncation is the correct design, not a
workaround.

## What we know (from this session's failed run)

`compile-h8.log` captured Hailo's per-model end-node hints. Distinct
suggestions observed:

| YOLO version | Task | Hailo's suggested end nodes |
|---|---|---|
| v8 | det | `/model.22/Concat, /model.22/Sigmoid_1, /model.22/Mul` |
| v11 | det | `/model.23/Concat, /model.23/Sigmoid_1, /model.23/Mul` |
| v11 | seg | `/model.23/Sigmoid, /model.23/Concat_2, /model.23/proto/cv3/act/Mul, /model.23/Concat` |
| v11 | pose | `/model.23/Concat, /model.23/Mul_3, /model.23/Sigmoid_1, /model.23/Sigmoid` |
| v26 | det | (need a separate run — wasn't in the gap-models batch) |

Plus the `Cos` unsupported on every OBB compile (v8/v11/v26) — those
need an end-node before the angle-decoding head.

The exact node IDs depend on (a) the YOLO version's head module index
(`/model.22` for v8, `/model.23` for v11) and (b) Ultralytics' export
naming. We must verify these against fresh `.onnx` exports before
hardcoding them — Ultralytics has changed export naming between minor
versions.

## Current code, files of interest

- **`benchmark/workloads/yolo/conversion/har_generator.py`**
  - `HARGeneratorConfig` (line 23) has `start_node: Optional[str]` and `end_node: Optional[str]` fields, both default `None`. Singular type — but the Hailo SDK call expects lists.
  - `_generate_with_sdk` (line 246) passes `start_node_names=config.start_node, end_node_names=config.end_node`. SDK accepts both string and list, but ours is `None` → Hailo parses the whole graph → fail.
  - `_generate_with_model_zoo` (line ~190) tries `parse_model` from `hailo_model_zoo.core.main_utils` first. **This also fails** — the user's log showed "Model Zoo parsing failed" before each fallback to `_generate_with_sdk`. Investigate why before assuming we need a custom path.
- **`benchmark/workloads/yolo/conversion/pipeline.py`**
  - `_run_har_generation` (line ~416) builds `HARGeneratorConfig` with only `target_device`, `input_resolution`, `batch_size`. Never sets end-nodes.
  - `ConversionConfig` (line ~54) has no end-node fields either; would need new fields if we expose end-node configuration to callers (probably not — better to derive from version/task internally).
- **Postprocessor reference** (already correct, no changes needed):
  - `benchmark/workloads/yolo/postprocessing.py` — `_process_detection` / `_process_obb` / `_process_segmentation` / `_process_pose` consume the truncated-model outputs. Confirm tensor shapes/order match what each task's end-node set produces.
- **hailomz YAML reference** (workstation venvs, not committed):
  - `venv-compile-h8/lib/python3.10/site-packages/hailo_model_zoo/cfg/networks/`
  - Has `yolov8n.yaml`, `yolov8n_seg.yaml`, `yolov8s_pose.yaml`, `yolov11n.yaml` …
  - **Does NOT have**: `yolov8*_obb.yaml`, `yolov11*_seg.yaml`, `yolov11*_pose.yaml`, any `yolov26*` (these are the gap models we need to compile manually).
  - YAML structure example: `nodes: - null - [/model.22/cv2.2/cv2.2.2/Conv]` — task-specific end-nodes baked in.

## Investigation step (do this first)

Before committing to any approach, run **one** failing model with
extra logging and figure out **why the hailomz fallback failed**:

```bash
BENCHY_VENV=venv-compile-h8 ./venv-compile-h8/bin/python -m benchmark \
    compile --hw-arch hailo8 --model yolo11n-seg.pt \
    --output-dir /tmp/test-out -v 2>&1 | tee /tmp/yolo11n-seg-debug.log
```

Specifically figure out:

1. Does `get_network_info("yolo11n-seg")` find a Model Zoo entry?
   (Likely no — the YAML directory only has `yolov11*.yaml` for
   detection.)
2. If yes, does `parse_model` fail because Ultralytics-exported
   ONNX node names diverge from what the YAML expects?
3. If no Model Zoo entry, the fallback path needs end-nodes. Does
   `runner.translate_onnx_model(...)` accept a list of strings for
   `end_node_names`? (Should — Hailo SDK docs say yes.)
4. Cross-check Hailo's suggested end-nodes against the actual ONNX
   node names with `onnx.load(path); print([n.name for n in
   model.graph.node])` — make sure the names we're going to put in
   the table actually exist in the exported ONNX.

This investigation gates the implementation choice.

## Implementation approach

Two viable paths. Pick after the investigation step.

### Path A — Static per-(version, task) end-node table

Add a module-level dict in `har_generator.py` (or a sibling module):

```python
END_NODE_TABLE: dict[tuple[str, YOLOTask], list[str]] = {
    ("v8", YOLOTask.DETECTION):    ["/model.22/Concat", "/model.22/Sigmoid_1", "/model.22/Mul"],
    ("v11", YOLOTask.DETECTION):   ["/model.23/Concat", "/model.23/Sigmoid_1", "/model.23/Mul"],
    ("v11", YOLOTask.SEGMENTATION):["/model.23/Sigmoid", "/model.23/Concat_2", "/model.23/proto/cv3/act/Mul", "/model.23/Concat"],
    ("v11", YOLOTask.POSE):        ["/model.23/Concat", "/model.23/Mul_3", "/model.23/Sigmoid_1", "/model.23/Sigmoid"],
    # … fill in OBB, classification, v8 seg/pose, v26 entries as we
    # verify them against actual ONNX exports.
}
```

Wire it into the pipeline:

1. **`HARGeneratorConfig`** — change `end_node: Optional[str]` to
   `end_nodes: Optional[list[str]]`. Pass through to SDK call as
   `end_node_names=config.end_nodes`. Same for `start_nodes`.
2. **`pipeline.py:_run_har_generation`** — when building
   `HARGeneratorConfig`, look up `(yolo_version, task)` in
   `END_NODE_TABLE` and set `end_nodes`. If the lookup misses, log a
   warning and try Hailo Model Zoo path (which may know about more
   models than our table).
3. **`har_generator.py:_generate_with_sdk`** — always pass the list
   of end-nodes (drop the singular default).
4. **`har_generator.py:_generate_with_model_zoo`** — keep the
   hailomz path as a primary attempt for `(version, task)` pairs the
   table doesn't know about (e.g. yolov8n.pt detection, where
   hailomz has `yolov8n.yaml`).

**Pros:** explicit, easy to reason about, easy to test in isolation
(just verify the table is populated correctly).
**Cons:** brittle to Ultralytics export-naming changes; we'll have to
update the table when that happens. Mitigated by versioning the
ultralytics dep and testing against a known-good version.

### Path B — Parse Hailo's diagnostic and retry

When `_generate_with_sdk` fails with a `Failed to parse ONNX model`
error whose message includes "use these end node names: …", parse
that out and retry with the suggested list. Hailo's parser is
self-documenting in this case.

```python
import re
END_NODE_HINT_RE = re.compile(
    r"using these end node names:\s*(?P<nodes>[^\n)]+)"
)
```

**Pros:** zero hardcoding — adapts automatically when Ultralytics
renames nodes, when new YOLO versions ship, when Hailo changes its
parser.
**Cons:** depends on the exact format of Hailo's error message
(could change between SDK versions); double-compile cost (first
attempt fails, retry succeeds); harder to unit-test cleanly.

### Recommended: **A as the primary path, with B as a fallback**

Per-version-and-task lookup → if missing → fall back to error-message
parsing → if that also fails, surface the original error. The static
table guarantees fast/correct compiles for the canonical YOLO models
we care about; the fallback handles surprises (e.g. somebody passes
a custom-trained model the table doesn't know about) without crashing
the sweep.

## Implementation steps

In rough order. Each numbered item ends with a checkpoint where you
run the matching subset of `tests/test_compile_cmd.py` plus one
real failing model from the list to confirm progress.

1. **Investigation step above** (cheap; 30–60 min).
2. **Add `END_NODE_TABLE`** to `benchmark/workloads/yolo/conversion/har_generator.py`. Initial entries: the four rows from the `compile-h8.log` capture above. Verify each list against `onnx.load("/tmp/.../yolo11n-seg.onnx").graph.node` — node names must exist verbatim.
3. **Refactor `HARGeneratorConfig`** — `end_node: Optional[str]` → `end_nodes: Optional[list[str]]`. Same for start. Update the SDK call site to pass the list.
4. **Update `pipeline.py:_run_har_generation`** to populate `end_nodes` from `END_NODE_TABLE.get((yolo_version, task))`. Pass into `HARGeneratorConfig`.
5. **Add fallback path B** to `_generate_with_sdk`'s exception handler — extract end-nodes from the failure message and retry once. Cap the retry at one attempt to avoid infinite loops.
6. **Run `yolo11n-seg.pt` end-to-end** (a known-failing model, ~10 min). Confirm: HEF lands in `~/.cache/benchy/hailo/hailo8/...`, gets staged into `resources/hefs/v11_segmentation_n_hailo8.hef`, file size > 1 MB.
7. **Verify output tensor count and shapes** match `_process_segmentation`'s expectations. If not, the end-nodes are wrong (a retraining of step 2). This is the slowest iteration loop: every wrong end-node set = 5–30 min wasted compile + Pi-side load test.
8. **Extend the table** to cover the remaining six gap models (`yolo11n-pose`, `yolov8n-obb`, `yolo11n-obb`, `yolo26n-obb`, `yolo26n-seg`, `yolo26n-pose`). v26 may need its own entries — Hailo's hint for v26 wasn't in the first sweep's log because v26 detection passed via hailomz. Capture v26 hints by running each v26 gap model with `-v` and checking the log.
9. **Hailo-10H pass.** `BENCHY_VENV=venv-compile-h10h scripts/compile_workstation_hefs.sh --arch hailo10h --include-detection`. The end-nodes should be identical (head architecture is the same; only the chip target differs). If not, add hailo10h-specific entries to the table (column or separate dict).

## Tests to add

In `tests/test_har_generator.py` (new file):

- `test_end_node_table_covers_all_known_combinations` — every
  `(yolo_version, task)` pair Benchy lists in `YOLO_MODELS` /
  `HAILO_SUPPORTED_TASKS` has a non-empty entry. Failing this means
  someone added a new task / version without updating the table.
- `test_end_nodes_threaded_into_har_config` — given a
  `ConversionConfig(target_device="hailo8")` for `(v11, SEG)`,
  `pipeline._run_har_generation` builds a `HARGeneratorConfig` whose
  `end_nodes` matches `END_NODE_TABLE[("v11", YOLOTask.SEGMENTATION)]`.
  Mock `HARGenerator.generate` to capture the config it receives.
- `test_diagnostic_parser_extracts_end_nodes` — given the literal
  Hailo error string from `compile-h8.log`, the regex extracts the
  expected list. Edge cases: trailing comma, whitespace, the
  ", trying direct SDK" suffix observed in the actual log.
- `test_fallback_retry_path` — mock `_generate_with_sdk` to fail
  once with a Hailo-style "use these end node names" error and
  succeed on the retry; verify the retry is invoked with the parsed
  end-nodes.

Plus a smoke test that `python -m benchmark compile` exit codes still
propagate correctly (already added to `tests/test_compile_cmd.py` —
keep passing).

All tests should mock `hailo_sdk_client` so they pass on any dev
machine. The "does the truncated HEF actually load on the Pi" check
is integration-level and gets done via
`scripts/verify_ai_hat_plus.sh` once the HEFs land.

## Iteration cycle / rollout

- **Per-attempt cost**: a compile that gets past ONNX export but
  fails at HAR generation is fast (~10s). One that succeeds is slow
  (5–30 min on CPU; ~10× slower than CUDA, and this workstation has
  AMD GPU so CPU is the only path).
- **Plan for**: 30–60 min investigation + 1–2 hr writing/refactoring
  + 2–4 hr iterating on end-nodes for one model + 4–8 hr running
  the full hailo8 sweep + 4–8 hr for the hailo10h sweep.
- **Don't** rebuild the venvs unless something breaks — the existing
  `venv-compile-h8` and `venv-compile-h10h` are slow to set up
  (pygraphviz CFLAGS workaround, opencv pin, ~1 GB of wheels, ~30 min
  of pip).

## Out of scope for this fix

- **CUDA acceleration**: workstation is AMD-only; Hailo doesn't
  support ROCm. Compiles run on CPU. Don't try to get GPU
  acceleration working — separate problem with no clean fix.
- **Migrating `resources/hefs/` to git-LFS**: GitHub already warned
  that one HEF passed 50 MB, and the compile sweep will produce
  ~14 more files. Worth doing, but separate from making compiles
  work in the first place.
- **YOLOv26 hailomz support**: Hailo's Model Zoo doesn't ship v26
  configs at all. Our table covers it instead. If/when Hailo adds
  v26 to hailomz, we can drop our v26 entries.
- **Hailo's runtime version pairing**: `resources/hailo-8/` ships
  HailoRT 4.23 which pairs with DFC 3.33.x; `resources/hailo-10H/`
  ships HailoRT 5.2 which pairs with DFC 5.3.x. Separate workstream
  documented in `resources/hailo-sdk/README.md`.

## What's already done (don't redo)

- `python -m benchmark compile` subcommand works end-to-end
  (commit `1175ce5`).
- `scripts/compile_workstation_hefs.sh` driver works, supports
  `--venv` / `BENCHY_VENV`, and propagates exit codes correctly
  (commits `9e585f6`, `aa7bf26`).
- `scripts/fetch_prebuilt_hefs.py` Hailo Model Zoo S3 fetcher works
  with per-arch defaults and HEAD-based dry-run (commit `1175ce5`).
- 13 hailo8 detection / seg / pose HEFs are already staged from
  Plan B (commit `9e585f6`); hailo10h has zero — Hailo doesn't
  publish hailo10h on the public S3 bucket.
- `venv-compile-h8` (Python 3.10 + DFC 3.33.1 + Model Zoo 2.18.0)
  and `venv-compile-h10h` (Python 3.10 + DFC 5.3.0 + Model Zoo
  5.3.0) both built and verified (commit `1286a75`).
- `benchmark/__main__.py` exit-code propagation fix + driver
  `set -u` empty-array fix (commit `aa7bf26`).
- 253/253 tests pass; the new ones cover the compile subcommand,
  fetch-script behavior, skip-if-staged, exit-code propagation, and
  per-arch zoo defaults.

## Quick references

- `compile-h8.log` (workstation, gitignored): full output of the
  failed sweep with Hailo's end-node hints in the warnings. Don't
  delete until the table is filled in.
- `docs/compilation/pitfalls.md` § 3 / § 4: existing end-node
  guidance.
- `docs/compilation/hailo8/workflow.md` § 2 / § 4: shows the manual
  `--start-node-names` / `--end-node-names` invocation pattern.
- `resources/hailo-sdk/README.md`: two-venv setup explanation.
- `resources/session_issues_2026-04-27.md`: catalogue of prior
  bring-up issues — read before assuming a problem is new.
