# End-node truncation — see `har_generator.py` and pitfalls § 10

This document was the planning note for the HAR-generation
end-node table when truncation was still future work. The table is
now implemented and populated for v8/v11 detection + segmentation
and v11 pose.

**The plan's specific recommendations were also wrong** in light of
the 2026-04-29 NVIDIA bring-up. It suggested cutting at Hailo's
diagnostic-hint end-nodes (deep post-processing layers like
`/model.23/Sigmoid`, `/model.23/Concat`, `/model.23/Mul`). Those
cuts pass HAR generation but fail chip mapping on Hailo-8 with
`16x4 not supported in activation1/2` because the high-precision-
bias activations end up on the chip subgraph. The correct cut is at
the raw `cv*.X.X.2/Conv` outputs, matching what the Hailo Model Zoo
YAMLs use under `parser.nodes`.

For the current state, see:

- **`benchmark/workloads/yolo/conversion/har_generator.py:END_NODE_TABLE`**
  — populated entries: `(v8, det)`, `(v8, seg)`, `(v11, det)`,
  `(v11, seg)`, `(v11, pose)`. Gap entries: `(v8, pose)`, `(v8, obb)`,
  `(v11, obb)`, all v26 tasks.
- **`docs/compilation/pitfalls.md` § 10** — the rule for adding new
  entries (cut at raw Conv outputs; verify against Hailo Model Zoo
  YAMLs; cross-check against `postprocessing.py`).
- **`resources/session_notes_2026-04-29_nvidia_workstation.md` Issue 6
  + Issue 10** — full forensics on why the deep-layer cut was wrong
  and the procedure for deriving the missing entries.
