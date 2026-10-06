#!/usr/bin/env python3
"""Draw the website overview for the repository and dataset cards.

This horizontal schematic is independent of the manuscript figure. It uses
the released target counts; its symbols are illustrative evaluation targets.
Only PNG and SVG assets are written. No models or dataset tables are loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

WIDTH, HEIGHT = 1400, 750
INK = "#23374D"
MUTED = "#5D7084"
BLUE = "#355F9A"
ILLICIT = "#B75B35"
BENIGN = "#527FA6"
PALE_BENIGN = "#AEBFD0"
BACKGROUND = "#F6F8FB"
BORDER = "#DCE4ED"
COUNTS = [("HI-Small", 1_015_669, 3_753), ("LI-Small", 1_384_810, 2_268)]


def draw(output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.usetex": False,
        "svg.fonttype": "path",
        "svg.hashsalt": "amlworld-compact-readme",
        "savefig.bbox": None,
        "savefig.pad_inches": 0,
        "savefig.transparent": False,
        "savefig.facecolor": "auto",
    })
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100,
                     facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    labels = []

    def box(x, y, width, height, fill="white", border=BORDER, radius=16, lw=1):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=border, linewidth=lw,
        ))
        return (x, y, x + width, y + height)

    def text(x, y, value, size=17, weight="normal", color=INK,
             ha="left", bounds=None):
        artist = ax.text(x, y, value, fontsize=size, fontweight=weight,
                         color=color, ha=ha, va="center", linespacing=1.3)
        labels.append((artist, bounds))
        return artist

    def target(x, y, illicit=False, color=BENIGN):
        if illicit:
            ax.add_patch(Polygon([(x, y - 7), (x + 7, y), (x, y + 7),
                                  (x - 7, y)], closed=True,
                                 facecolor=ILLICIT, edgecolor="none"))
        else:
            ax.add_patch(Circle((x, y), 5, facecolor=color, edgecolor="none"))

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=18,
            linewidth=1.8, color=BLUE, shrinkA=0, shrinkB=0,
        ))

    text(48, 53, "AMLworld-Compact", size=30, weight="bold")
    text(48, 104, "Importance-weighted transaction graphs for model evaluation",
         size=18, color=MUTED)
    badge = box(1120, 30, 232, 55, fill="#E7EEF8", border="#E7EEF8", radius=12)
    text(1236, 57, f"{sum(row[2] for row in COUNTS):,} cases", size=18,
         weight="bold", color=BLUE, ha="center", bounds=badge)

    source = box(48, 165, 344, 352)
    subset = box(552, 165, 344, 352, border="#93AED0", lw=1.4)
    scoring = box(976, 165, 376, 352)

    text(72, 203, "Full test partitions", size=21, weight="bold", bounds=source)
    text(72, 242, "HI-Small and LI-Small", size=17, color=MUTED, bounds=source)
    text(72, 293, "Illicit", size=17, color=ILLICIT, bounds=source)
    text(72, 377, "Benign", size=17, color=MUTED, bounds=source)
    for x in [186, 230, 274, 318]:
        target(x, 293, illicit=True)
    for row in range(4):
        for col in range(8):
            target(179 + 24 * col, 345 + 25 * row, color=PALE_BENIGN)
    text(72, 475, "Full test populations", size=17, color=MUTED, bounds=source)

    text(472, 296, "HT-Coreset", size=16, weight="bold", color=BLUE,
         ha="center", bounds=(396, 270, 548, 325))
    arrow((416, 335), (528, 335))

    text(576, 203, "Retained targets", size=21, weight="bold", bounds=subset)
    text(576, 242, "AMLworld-Compact", size=17, color=BLUE, bounds=subset)
    text(576, 293, "Illicit", size=17, color=ILLICIT, bounds=subset)
    text(576, 368, "Benign", size=17, color=BENIGN, bounds=subset)
    for x in [690, 734, 778, 822]:
        target(x, 293, illicit=True)
        target(x, 345)
        target(x, 382)
    text(576, 448, "Keep every illicit target", size=17, bounds=subset)
    text(576, 483, "Sample benign in strata", size=17, bounds=subset)

    arrow((917, 335), (955, 335))
    text(1000, 203, "Model predictions", size=21, weight="bold", bounds=scoring)
    text(1000, 242, "Same predictions, two reports", size=16, color=MUTED,
         bounds=scoring)
    weighted = box(1000, 280, 328, 90, fill="#EDF2FA", border="#EDF2FA", radius=12)
    text(1018, 311, "HT-weighted", size=19, weight="bold", color=BLUE, bounds=weighted)
    text(1018, 347, "Full-test estimates", size=17, color=MUTED, bounds=weighted)
    unweighted = box(1000, 396, 328, 90, fill="#F1F4F7", border="#F1F4F7", radius=12)
    text(1018, 427, "Unweighted", size=19, weight="bold", bounds=unweighted)
    text(1018, 463, "Subset diagnostics", size=17, color=MUTED, bounds=unweighted)

    for x, (split, full, kept) in zip([48, 724], COUNTS):
        panel = box(x, 550, 628, 124)
        text(x + 24, 581, split, size=17, weight="bold", bounds=panel)
        text(x + 24, 630, f"{full:,} → {kept:,} targets", size=17,
             color=MUTED, bounds=panel)
        text(x + 518, 594, f"≈{full / kept:.0f}×", size=30, weight="bold",
             color=BLUE, ha="center", bounds=panel)
        text(x + 518, 639, "fewer targets", size=14, color=MUTED,
             ha="center", bounds=panel)

    text(48, 717, "Selection symbols are schematic evaluation targets.",
         size=14, color=MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, bounds in labels:
        display = artist.get_window_extent(renderer)
        points = ax.transData.inverted().transform(display.get_points())
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        left, top, right, bottom = bounds or (0, 0, WIDTH, HEIGHT)
        if x0 < left or x1 > right or y0 < top or y1 > bottom:
            raise ValueError(f"Text outside its panel: {artist.get_text()!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg"]:
        target_path = output_dir / f"readme_overview.{suffix}"
        metadata = {"Title": "AMLworld-Compact evaluation overview"}
        if suffix == "svg":
            metadata["Date"] = None
        fig.savefig(target_path, dpi=160, metadata=metadata)
        print(target_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/figures")
    args = parser.parse_args()
    draw(args.output_dir)


if __name__ == "__main__":
    main()
