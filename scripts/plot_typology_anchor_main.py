#!/usr/bin/env python3
"""Plot the main-paper typology heatmap as two panels with shared rows.

Run from any directory with ``python scripts/plot_typology_anchor_main.py``.
Only the output figure is written; the aggregate CSV is left unchanged.
Use ``--output`` to write directly to a manuscript figure directory.

Source: ``results/analysis/per_run.csv``. Select the
``gt=benign`` rows for ICL-FS/ICL-ZS. The denominator includes every benign
prediction from all five seeds, including ``legitimate`` and ``<none>``.
The CSV is an exact copy of the archived typology-frequency aggregates
after renaming the prompting methods. HI-Small has 2,502 benign cases per
seed and LI-Small has 1,512. No class is renormalised or pooled with another.

All ten columns in the original figure are preserved. The extra CSV column
``<other>`` is asserted to be zero before omission. Cell colours use the
unrounded percentages; annotations round to whole percentages and values
below 0.5% are left unannotated, as in the original figure.
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

SOURCE = paths.results() / "analysis/per_run.csv"
OUTPUT = paths.figures() / "typology_anchor_heatmap_main.pdf"
MODELS = [
    ("GPT-OSS-20B", "GPT-OSS-20B"),
    ("Qwen3.5-27B", "Qwen3.5-27B"),
    ("Nemotron-3-Nano-30B", "Nem-Nano-30B"),
    ("Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B"),
    ("GPT-OSS-120B", "GPT-OSS-120B"),
    ("Nemotron-3-Super-120B", "Nem-Super-120B"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B"),
]
PROMPTING = [("ICL-FS", "FS"), ("ICL-ZS", "ZS")]
CLASSES = [
    "fan-out", "fan-in", "cycle", "scatter-gather", "gather-scatter",
    "random", "bipartite", "stack", "legitimate", "<none>",
]
BENIGN_PER_SEED = {"HI-Small": 2502, "LI-Small": 1512}


def load_percentages(source: Path) -> tuple[dict[str, np.ndarray], dict]:
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for row in rows:
        if row["split"] != "gt=benign" or row["method"] not in dict(PROMPTING):
            continue
        if row["model"] not in dict(MODELS) or row["dataset"] not in BENIGN_PER_SEED:
            continue
        key = row["dataset"], row["model"], row["method"]
        if key in selected:
            raise ValueError(f"Duplicate aggregate row: {key}")
        selected[key] = row
    if len(selected) != 28:
        raise ValueError(f"Expected 28 aggregate rows, found {len(selected)}")

    matrices = {}
    audit = {"source": str(source),
             "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "rows": len(selected), "columns": CLASSES,
             "denominator": "all GT-benign predictions across five seeds",
             "datasets": {}}
    for dataset, benign_count in BENIGN_PER_SEED.items():
        counts = []
        expected = benign_count * 5
        for model, _ in MODELS:
            for method, _ in PROMPTING:
                row = selected[dataset, model, method]
                values = [int(row[label]) for label in CLASSES]
                if int(row["n_seeds"]) != 5 or int(row["n_total"]) != expected:
                    raise ValueError(f"Unexpected seed count or denominator: {dataset, model, method}")
                if int(row["<other>"]) != 0 or sum(values) != expected:
                    raise ValueError(f"Label counts do not exhaust the denominator: {dataset, model, method}")
                counts.append(values)
        matrices[dataset] = np.asarray(counts, dtype=float) * 100 / expected
        audit["datasets"][dataset] = {
            "rows": len(counts), "denominator_per_row": expected,
            "plotted_count_total": int(np.asarray(counts).sum()),
            "row_sum_percent_min": float(matrices[dataset].sum(axis=1).min()),
            "row_sum_percent_max": float(matrices[dataset].sum(axis=1).max()),
            "omitted_other_count": 0,
        }
    return matrices, audit


def plot(matrices: dict[str, np.ndarray], output: Path) -> None:
    apply_style()
    fig = plt.figure(figsize=(TEXT_W, 3.9))
    row_labels = [f"{short}  {label}" for _, short in MODELS for _, label in PROMPTING]
    axes = [fig.add_axes([0.285, 0.365, 0.327, 0.562]),
            fig.add_axes([0.635, 0.365, 0.327, 0.562])]
    for panel, (dataset, values) in enumerate(matrices.items()):
        ax = axes[panel]
        # pcolormesh preserves vector cells in the PDF (imshow would rasterise).
        mesh = ax.pcolormesh(np.arange(11) - 0.5, np.arange(15) - 0.5,
                            values, cmap="YlOrRd", vmin=0, vmax=100,
                            shading="flat", rasterized=False,
                            edgecolors="none")
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(13.5, -0.5)
        ax.set_title(dataset, fontsize=11, pad=5)
        display_classes = [r"$\langle\mathrm{none}\rangle$" if label == "<none>" else label
                           for label in CLASSES]
        ax.set_xticks(range(10), display_classes, rotation=62, ha="right",
                      rotation_mode="anchor", fontsize=9.5)
        ax.set_yticks(range(14), row_labels if panel == 0 else [""] * 14,
                      fontsize=9.5)
        ax.tick_params(axis="both", which="both", length=0, pad=3)
        for boundary in np.arange(1.5, 13.5, 2):
            ax.axhline(boundary, color="white", linewidth=0.7)
        for boundary in (7.5,):
            ax.axvline(boundary, color="#aaaaaa", linewidth=0.5)
        for r in range(14):
            for c in range(10):
                value = values[r, c]
                if value >= 0.5:
                    ax.text(c, r, f"{value:.0f}", ha="center", va="center",
                            fontsize=9.5, color="white" if value >= 60 else "#111111")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#777777")
            spine.set_linewidth(0.45)
    color_ax = fig.add_axes([0.34, 0.11, 0.54, 0.026])
    bar = fig.colorbar(mesh, cax=color_ax, orientation="horizontal",
                       ticks=np.arange(0, 101, 20))
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor("face")
    bar.ax.tick_params(labelsize=9.5, length=2, pad=2)
    bar.set_label("Predictions on GT-benign cases (%)", fontsize=10, labelpad=4)
    bar.outline.set_linewidth(0.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None,
                metadata={"Title": "Predicted labels on GT-benign cases",
                          "Subject": "Five-seed aggregate; HI-Small and LI-Small; all benign predictions",
                          "Creator": Path(__file__).name})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    matrices, audit = load_percentages(args.source)
    plot(matrices, args.output)
    audit["output"] = str(args.output)
    audit["figure_inches"] = [TEXT_W, 3.9]
    audit["colour_scale_percent"] = [0, 100]
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
