#!/usr/bin/env python3
"""Plot conditional per-class typology F1 from the released aggregate CSV.

This redraws the HI-Small figure without inference or metric recomputation.
The 10 model series, eight classes, macro-F1 labels and colour assignments
match the existing manuscript figure. Each series uses that model's true
positives, not the all-illicit TF1 denominator used in the main task table.
Use --source or --output to override the portable checkout defaults.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amlc import paths
from amlc.figures._paper_style import apply_style, TEXT_W

SOURCE = paths.results() / "error_analysis/typology_f1.csv"
OUTPUT = paths.figures() / "typology_f1_hi_small.pdf"
CLASSES = ["fan-out", "fan-in", "cycle", "scatter-gather", "gather-scatter",
           "stack", "bipartite", "random"]
# Explicit series/colour order reproduces the original ten-series figure.
MODELS = [
    ("LightGBM-GFP", "LightGBM-GFP", "#1f4e79"),
    ("XGBoost-GFP", "XGBoost-GFP", "#2e75b6"),
    ("Ensemble-Soft", "Ensemble-Soft", "#5b9bd5"),
    ("GPT-OSS-20B", "GPT-OSS-20B", "#c55a11"),
    ("Nemotron-3-Nano-30B", "Nem-Nano-30B", "#ed7d31"),
    ("GPT-OSS-120B", "GPT-OSS-120B", "#f4b183"),
    ("Nemotron-3-Super-120B", "Nem-Super-120B", "#996633"),
    ("Qwen3-5-397B-A17B", "Qwen3.5-397B", "#7030a0"),
    ("Qwen3-5-35B-A3B", "Qwen3.5-35B", "#c55a11"),
    ("Qwen3-5-27B", "Qwen3.5-27B", "#ed7d31"),
]


def load_values(source: Path) -> tuple[np.ndarray, np.ndarray]:
    with source.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["dataset"] == "HI-Small"]
    selected = {row["model"]: row for row in rows}
    if len(rows) != 10 or set(selected) != {model for model, _, _ in MODELS}:
        raise ValueError("Expected exactly the ten released HI-Small model rows")
    values = np.asarray([[float(selected[model][f"f1_{label}"])
                          for label in CLASSES] for model, _, _ in MODELS])
    macro = np.asarray([float(selected[model]["typology_macro_f1"])
                        for model, _, _ in MODELS])
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError("Class F1 values must be finite proportions in [0, 1]")
    return values, macro


def plot(values: np.ndarray, macro: np.ndarray, output: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_W, 3.8))
    # Reserve four legend rows and slanted class labels without shrinking text.
    fig.subplots_adjust(left=0.11, right=0.995, top=0.90, bottom=0.40)
    x = np.arange(len(CLASSES))
    width = 0.8 / len(MODELS)
    for i, (_, label, color) in enumerate(MODELS):
        ax.bar(x + i * width, values[i] * 100, width,
               label=f"{label} ({macro[i]:.1%})", color=color,
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x + width * (len(MODELS) - 1) / 2,
                  CLASSES, rotation=25, ha="right", fontsize=9.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Typology F1 (%)", fontsize=11)
    ax.set_title("Per-class typology F1 on HI-Small", fontsize=11, pad=7)
    ax.set_ylim(0, 110)
    # A figure-level legend has a stable position independent of tick extents.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center",
               bbox_to_anchor=(0.515, 0.012), fontsize=9.5,
               columnspacing=1.0, handletextpad=0.45, handlelength=1.25,
               borderaxespad=0, labelspacing=0.4, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None,
                metadata={"Title": "Conditional per-class typology F1 on HI-Small",
                          "Subject": "Few-shot, run 42; each model's correctly detected illicit edges",
                          "Creator": Path(__file__).name})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    values, macro = load_values(args.source)
    plot(values, macro, args.output)
    print(json.dumps({"source": str(args.source), "source_sha256":
                      hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      "output": str(args.output), "models": len(MODELS),
                      "classes": CLASSES, "figure_inches": [TEXT_W, 3.8],
                      "values": values.tolist(), "macro_f1": macro.tolist()}, indent=2))


if __name__ == "__main__":
    main()
