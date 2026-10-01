#!/usr/bin/env python3
"""Render the released regex audit heatmap at the ACL 77 mm column width.

Uses the original 1,000-record annotation CSV, including empty traces.
No API calls or re-annotation are performed. The supporting pooled funnel
figure is deliberately independent of this manuscript figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amlc import paths
from amlc.figures._paper_style import COL_W, apply_style, short_model


STEPS = ["parse", "recall", "match", "conclude"]
MODEL_ORDER = [
    "GPT-OSS-20B", "Nemotron-3-Nano-30B", "Qwen3.5-35B-A3B",
    "GPT-OSS-120B", "Nemotron-3-Super-120B", "Qwen3.5-397B-A17B",
    "Qwen3.5-27B",
]


def pass_rates(source: Path) -> np.ndarray:
    df = pd.read_csv(source)
    return np.array([
        [df.loc[df["model"].eq(model), step].mean() * 100 for step in STEPS]
        for model in MODEL_ORDER
    ])


def plot(output: Path, source: Path) -> None:
    apply_style()
    matrix = pass_rates(source)
    fig = plt.figure(figsize=(COL_W, 72 / 25.4))
    ax = fig.add_axes([0.355, 0.31, 0.63, 0.625])
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(STEPS)), [step.capitalize() for step in STEPS])
    ax.set_yticks(range(len(MODEL_ORDER)), [short_model(model) for model in MODEL_ORDER])
    ax.tick_params(axis="both", labelsize=9.5, length=0, pad=3)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            color = "white" if value < 35 or value > 75 else "black"
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=10, color=color)

    cax = fig.add_axes([0.405, 0.18, 0.54, 0.045])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cbar.set_label("Pass rate (%)", fontsize=10, labelpad=2)
    cbar.ax.tick_params(labelsize=9.5, length=2, pad=2)
    cbar.outline.set_linewidth(0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None)
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=paths.figures() / "step_heatmap_per_model.pdf")
    parser.add_argument("--input", type=Path, default=paths.results() / "audit" / "trace_4step_annotations_n1000.csv")
    args = parser.parse_args()
    plot(args.output, args.input)


if __name__ == "__main__":
    main()
