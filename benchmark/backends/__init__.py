"""Cross-workload backend utilities.

Helpers that both the YOLO and LLM workloads need to talk to the same
underlying accelerator (currently: Hailo). Keeping them here prevents
copy-paste drift between `benchmark/workloads/yolo/backends/hailo.py` and
`benchmark/workloads/llm/hailo_metrics.py`.
"""
