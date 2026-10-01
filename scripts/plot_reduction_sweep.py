#!/usr/bin/env python3
"""Render the released coreset sweep and baseline results at ACL text width.

Uses only results/coreset CSVs. No training or sampling is performed.
The PDF has a fixed 160 mm width so labels retain their point sizes when
included with ``width=\\textwidth`` in the ACL manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amlc import paths
from amlc.figures._paper_style import TEXT_W, apply_style


FULL = {"HI-Small": 1_015_669, "LI-Small": 1_384_810}
BLUE, RED, GRAY = "#1f77b4", "#d62728", "#7f7f7f"
BASELINE_ORDER = [
    ("B1_random_uniform", "B1 Random"),
    ("B2_stratified_proportional", "B2 Stratified"),
    ("B3_keep_ill_no_iw", "B3 No HT"),
    ("B4_v2_no_hardneg", "B4 No Hard-Neg"),
    ("B5_fogliato_neyman", "B5 Fogliato"),
    ("B6_leskovec_rw", "B6 Leskovec"),
    ("B7_gao_stratified", "B7 Gao"),
]


def load_inputs(directory: Path):
    """Retain the original 12 feasible sweep points and 7 baseline rows."""
    sweeps = {}
    for dataset in FULL:
        df = pd.read_csv(directory / f"downsample_v2_{dataset}_sizes.csv")
        df = df[df["feasible"].astype(str).str.lower().eq("true")].copy()
        df["delta_f1_pp"] = df["delta_f1_w_mean"].abs() * 100
        df["reduction"] = FULL[dataset] / df["n_total"]
        sweeps[dataset] = df.sort_values("reduction")
    baseline = pd.read_csv(directory / "baselines_agg.csv")
    baseline["delta_f1_pp"] = baseline["delta_f1_mean"].abs() * 100
    return sweeps, baseline


def plot(output: Path, directory: Path) -> None:
    apply_style()
    sweeps, baseline = load_inputs(directory)
    fig = plt.figure(figsize=(TEXT_W, 105 / 25.4))

    for row, dataset in enumerate(FULL):
        # Fixed geometry keeps long baseline names clear of the second y-axis.
        bottom = 0.60 if row == 0 else 0.12
        ax = fig.add_axes([0.082, bottom, 0.358, 0.325])
        ax_b = fig.add_axes([0.738, bottom, 0.227, 0.325])
        df = sweeps[dataset]
        endpoint = df.iloc[-1]
        r_max = float(endpoint["reduction"])

        ax.axhspan(5, 30, color=RED, alpha=0.07, zorder=0)
        ax.axhline(5, color=RED, linewidth=0.8, linestyle="--", alpha=0.65)
        ax.text(1.12, 6.7, "5 pp tolerance", fontsize=9.5, color=RED)
        ax.plot(
            df["reduction"], df["delta_f1_pp"], color=BLUE,
            linewidth=1.4, marker="o", markersize=3.8,
            markerfacecolor="white", markeredgewidth=0.9, zorder=3,
        )
        ax.scatter(
            [r_max], [endpoint["delta_f1_pp"]], s=100, marker="*",
            color="gold", edgecolors=BLUE, linewidths=0.9, zorder=5,
        )
        ax.text(
            0.025, 0.81, "HT-Coreset: " + r"$|\Delta F1|<10^{-13}$ pp",
            transform=ax.transAxes, fontsize=9.5, color=BLUE,
        )
        ax.text(
            0.025, 0.66, f"{len(df)} sizes; up to {r_max:.0f}" + r"$\times$",
            transform=ax.transAxes, fontsize=9.5, color=BLUE,
        )
        ax.set_title(dataset + f": {FULL[dataset]:,} edges", loc="left", pad=10, fontsize=11)
        ax.set_xscale("log")
        ax.set_xlim(0.85, r_max * 1.35)
        ax.set_ylim(-3, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=9.5, length=3, pad=2)
        ax.tick_params(axis="y", labelcolor=BLUE)
        ax.set_xlabel(r"Target-count reduction ($\times$, log)", fontsize=10, labelpad=3)
        ax.set_ylabel(r"$|\Delta F1|$ (pp)", color=BLUE, fontsize=10, labelpad=2)
        ax.grid(True, alpha=0.22, which="major")

        rate = ax.twinx()
        rate.plot(
            df["reduction"], df["illicit_rate_pct"], color=GRAY,
            linewidth=1.0, linestyle="--", marker="^", markersize=3.8,
            markerfacecolor="white", markeredgewidth=0.8, alpha=0.9,
        )
        rate.set_ylim(-3, 40)
        rate.set_yticks([0, 20, 40])
        rate.tick_params(axis="y", labelcolor=GRAY, labelsize=9.5, length=3, pad=2)
        rate.set_ylabel("Subset illicit rate (%)", color=GRAY, fontsize=10, labelpad=2)
        rate.annotate(
            "1:2", xy=(r_max, endpoint["illicit_rate_pct"]),
            xytext=(-7, 5), textcoords="offset points", fontsize=9.5,
            color=GRAY, ha="right", va="bottom",
        )

        bds = baseline[baseline["dataset"].eq(dataset)].set_index("method")
        names = [label for _, label in BASELINE_ORDER] + ["HT-Coreset"]
        values = [float(bds.loc[key, "delta_f1_pp"]) for key, _ in BASELINE_ORDER]
        # The archived HT redraw has numerical-zero error (at most 1.2e-14 pp).
        values.append(float(bds.loc["ablation-redraw", "delta_f1_pp"]))
        bars = ax_b.barh(
            np.arange(len(names)), values, height=0.66,
            color=[RED] * 7 + [BLUE], alpha=0.85,
            edgecolor="black", linewidth=0.4,
        )
        offset = max(values) * 0.035
        for bar, value in zip(bars, values):
            ax_b.text(
                value + offset, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", fontsize=9.5, va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
                zorder=4,
            )
        ax_b.axvline(5, color=RED, linewidth=0.8, linestyle="--", alpha=0.65)
        ax_b.set_yticks(np.arange(len(names)), names, fontsize=9.5)
        ax_b.tick_params(axis="y", length=0, pad=3)
        ax_b.tick_params(axis="x", labelsize=9.5, length=3, pad=2)
        ax_b.invert_yaxis()
        ax_b.set_xlim(0, max(values) * 1.33)
        ax_b.set_xticks([0, 10, 20, 30] if row == 0 else [0, 20, 40])
        ax_b.set_xlabel(r"$|\Delta F1|$ (pp)", fontsize=10, labelpad=3)
        ax_b.set_title(f"At {r_max:.0f}" + r"$\times$ reduction", loc="left", fontsize=11, pad=10)
        ax_b.grid(True, axis="x", alpha=0.22)
        ax_b.set_axisbelow(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches=None)
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=paths.figures() / "reduction_sweep.pdf")
    parser.add_argument("--input-dir", type=Path, default=paths.results() / "coreset")
    args = parser.parse_args()
    plot(args.output, args.input_dir)


if __name__ == "__main__":
    main()
