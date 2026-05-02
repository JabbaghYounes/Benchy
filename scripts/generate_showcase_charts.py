#!/usr/bin/env python3
"""Generate static PNG charts for the SHOWCASE.md page from the
platform-aware aggregated CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_DIR = REPO_ROOT / "docs" / "showcase"
CHARTS_DIR = SHOWCASE_DIR / "charts"
DATA_DIR = SHOWCASE_DIR  # CSVs land in docs/showcase/ via aggregate_by_platform.py

PLATFORM_LABEL = {
    "rpi_ai_hat_plus": "AI HAT+ (Hailo-8 26 TOPS)",
    "rpi_ai_hat_plus_2": "AI HAT+ 2 (Hailo-10H)",
}
PLATFORM_COLOR = {
    "rpi_ai_hat_plus": "#1f77b4",   # blue
    "rpi_ai_hat_plus_2": "#ff7f0e", # orange
}
BACKEND_LABEL = {
    "hailo-10h": "Hailo-10H NPU",
    "ollama-cpu": "Pi 5 CPU (ollama)",
}
BACKEND_COLOR = {
    "hailo-10h": "#2ca02c",    # green
    "ollama-cpu": "#d62728",   # red
}


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def yolo_throughput_chart():
    """Side-by-side bar chart: throughput (FPS) per model, grouped by platform.
    Only shows models that BOTH platforms have data for."""
    rows = load_csv(DATA_DIR / "yolo_by_platform.csv")
    by_plat_model = {(r["platform"], r["model_name"]): r for r in rows}
    plat_models = {r["model_name"] for r in rows if r["platform"] == "rpi_ai_hat_plus"}
    plat2_models = {r["model_name"] for r in rows if r["platform"] == "rpi_ai_hat_plus_2"}
    common = sorted(plat_models & plat2_models)

    if not common:
        print("  no common models — skipping yolo throughput chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(common))
    width = 0.4

    h8_fps = [float(by_plat_model[("rpi_ai_hat_plus", m)]["throughput_mean_fps"])
              for m in common]
    h8_std = [float(by_plat_model[("rpi_ai_hat_plus", m)]["throughput_std_fps"])
              for m in common]
    h10_fps = [float(by_plat_model[("rpi_ai_hat_plus_2", m)]["throughput_mean_fps"])
               for m in common]
    h10_std = [float(by_plat_model[("rpi_ai_hat_plus_2", m)]["throughput_std_fps"])
               for m in common]

    bars1 = ax.bar(x - width/2, h8_fps, width, yerr=h8_std,
                    label=PLATFORM_LABEL["rpi_ai_hat_plus"],
                    color=PLATFORM_COLOR["rpi_ai_hat_plus"], capsize=3)
    bars2 = ax.bar(x + width/2, h10_fps, width, yerr=h10_std,
                    label=PLATFORM_LABEL["rpi_ai_hat_plus_2"],
                    color=PLATFORM_COLOR["rpi_ai_hat_plus_2"], capsize=3)

    # Annotate ratio above each pair
    for i, (a, b) in enumerate(zip(h8_fps, h10_fps)):
        ratio = a / b if b > 0 else 0
        ymax = max(a, b)
        ax.text(i, ymax * 1.05, f"{ratio:.1f}×",
                ha="center", va="bottom", fontsize=9, color="dimgray", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(".pt", "") for m in common], rotation=30, ha="right")
    ax.set_ylabel("Throughput (FPS, mean ± std)")
    ax.set_title("YOLO inference throughput — head-to-head, 640×640")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(h8_fps), max(h10_fps)) * 1.25)

    fig.tight_layout()
    out = CHARTS_DIR / "yolo_throughput_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def yolo_latency_chart():
    """Latency p50 with p95 markers, per model, grouped by platform."""
    rows = load_csv(DATA_DIR / "yolo_by_platform.csv")
    by_plat_model = {(r["platform"], r["model_name"]): r for r in rows}
    plat_models = {r["model_name"] for r in rows if r["platform"] == "rpi_ai_hat_plus"}
    plat2_models = {r["model_name"] for r in rows if r["platform"] == "rpi_ai_hat_plus_2"}
    common = sorted(plat_models & plat2_models)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(common))
    width = 0.4

    h8_lat = [float(by_plat_model[("rpi_ai_hat_plus", m)]["latency_mean_ms"])
              for m in common]
    h10_lat = [float(by_plat_model[("rpi_ai_hat_plus_2", m)]["latency_mean_ms"])
               for m in common]

    ax.bar(x - width/2, h8_lat, width,
           label=PLATFORM_LABEL["rpi_ai_hat_plus"],
           color=PLATFORM_COLOR["rpi_ai_hat_plus"])
    ax.bar(x + width/2, h10_lat, width,
           label=PLATFORM_LABEL["rpi_ai_hat_plus_2"],
           color=PLATFORM_COLOR["rpi_ai_hat_plus_2"])

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(".pt", "") for m in common], rotation=30, ha="right")
    ax.set_ylabel("Latency mean (ms)")
    ax.set_title("YOLO inference latency — lower is better, 640×640")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = CHARTS_DIR / "yolo_latency_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def llm_npu_vs_cpu_chart():
    """Bar chart: NPU vs CPU TPS on AI HAT+ 2, per drone prompt.
    Authoritative LLM comparison — both backends on the same Pi."""
    rows = load_csv(DATA_DIR / "llm_by_platform.csv")
    h10h = [r for r in rows
            if r["platform"] == "rpi_ai_hat_plus_2" and r["backend"] == "hailo-10h"]
    cpu = [r for r in rows
           if r["platform"] == "rpi_ai_hat_plus_2" and r["backend"] == "ollama-cpu"]
    by_prompt_npu = {r["prompt_id"]: r for r in h10h}
    by_prompt_cpu = {r["prompt_id"]: r for r in cpu}
    common = sorted(set(by_prompt_npu) & set(by_prompt_cpu))

    if not common:
        print("  no common LLM prompts — skipping NPU vs CPU chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(common))
    width = 0.4

    npu_tps = [float(by_prompt_npu[p]["tps_mean"]) for p in common]
    npu_std = [float(by_prompt_npu[p]["tps_std"]) for p in common]
    cpu_tps = [float(by_prompt_cpu[p]["tps_mean"]) for p in common]
    cpu_std = [float(by_prompt_cpu[p]["tps_std"]) for p in common]

    ax.bar(x - width/2, npu_tps, width, yerr=npu_std,
           label=BACKEND_LABEL["hailo-10h"],
           color=BACKEND_COLOR["hailo-10h"], capsize=3)
    ax.bar(x + width/2, cpu_tps, width, yerr=cpu_std,
           label=BACKEND_LABEL["ollama-cpu"],
           color=BACKEND_COLOR["ollama-cpu"], capsize=3)

    for i, (n, c) in enumerate(zip(npu_tps, cpu_tps)):
        ratio = n / c if c > 0 else 0
        ymax = max(n, c)
        ax.text(i, ymax * 1.05, f"{ratio:.2f}×",
                ha="center", va="bottom", fontsize=9, color="dimgray", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in common], fontsize=9)
    ax.set_ylabel("Tokens / second (mean ± std)")
    ax.set_title("LLM `llama3.2:1b` decode throughput — NPU vs CPU on AI HAT+ 2 Pi")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(npu_tps + cpu_tps) * 1.25)

    fig.tight_layout()
    out = CHARTS_DIR / "llm_npu_vs_cpu.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def llm_ttft_chart():
    """TTFT (prefill) is essentially identical NPU vs CPU since prefill
    is the same work; this chart makes that visually clear."""
    rows = load_csv(DATA_DIR / "llm_by_platform.csv")
    h10h = [r for r in rows
            if r["platform"] == "rpi_ai_hat_plus_2" and r["backend"] == "hailo-10h"]
    cpu = [r for r in rows
           if r["platform"] == "rpi_ai_hat_plus_2" and r["backend"] == "ollama-cpu"]
    by_prompt_npu = {r["prompt_id"]: r for r in h10h}
    by_prompt_cpu = {r["prompt_id"]: r for r in cpu}
    common = sorted(set(by_prompt_npu) & set(by_prompt_cpu))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(common))
    width = 0.4

    npu_ttft = [float(by_prompt_npu[p]["ttft_mean_ms"]) for p in common]
    cpu_ttft = [float(by_prompt_cpu[p]["ttft_mean_ms"]) for p in common]

    ax.bar(x - width/2, npu_ttft, width,
           label=BACKEND_LABEL["hailo-10h"],
           color=BACKEND_COLOR["hailo-10h"])
    ax.bar(x + width/2, cpu_ttft, width,
           label=BACKEND_LABEL["ollama-cpu"],
           color=BACKEND_COLOR["ollama-cpu"])

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in common], fontsize=9)
    ax.set_ylabel("TTFT — Time to first token (ms)")
    ax.set_title("LLM prefill latency (TTFT) — NPU vs CPU. Essentially identical.")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = CHARTS_DIR / "llm_ttft.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    yolo_throughput_chart()
    yolo_latency_chart()
    llm_npu_vs_cpu_chart()
    llm_ttft_chart()


if __name__ == "__main__":
    main()
