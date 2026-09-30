#!/usr/bin/env python3
"""Render Figure 1 as a publication-size vector PDF.

Run from any directory. Counts and reduction factors are the released
HI-Small / LI-Small values in Table 1. Tokens illustrate selected target
edges, NOT a sampled graph, and are intentionally not to scale. The four
illicit tokens are retained in both panels; neither panel represents
individual experimental observations.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WIDTH, HEIGHT = 3.25, 3.00
INK = '#203442'
MUTED = '#536571'
TEAL = '#096F78'
ORANGE = '#B65225'
BENIGN = '#B7C5CE'
RULE = '#D9E1E5'


def main():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.6,
        'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()

    def text(x, y, value, size=8.6, weight='normal', color=INK, ha='center'):
        return ax.text(x, y, value, fontsize=size, fontweight=weight,
                       color=color, ha=ha, va='center')

    def arrow(start, end, color=MUTED, lw=0.9):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                     mutation_scale=8, linewidth=lw, color=color,
                     shrinkA=0, shrinkB=0))

    def tokens(xs, ys, illicit_positions):
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                illicit = (row, col) in illicit_positions
                ax.scatter(x, y, s=18 if illicit else 12,
                           marker='D' if illicit else 'o',
                           facecolor=ORANGE if illicit else BENIGN,
                           edgecolors='none', zorder=3)

    # Split labels avoid using a slash as an ambiguous pair separator.
    # Reduction factors are rounded ratios of full-test / retained counts.
    for x, split, reduction in [(0.80, 'HI-Small', '271'),
                                (2.45, 'LI-Small', '611')]:
        text(x, 2.83, split, size=9.0, weight='bold')
        text(x, 2.58, r'$\approx$' + reduction + r'$\times$', size=18,
             weight='bold', color=TEAL)
    text(1.625, 2.34, 'target-count reduction', size=9.0)
    ax.plot([0.05, 3.20], [2.19, 2.19], color=RULE, lw=0.65)

    text(0.74, 2.03, 'Full test', size=9.5, weight='bold')
    text(2.51, 2.03, 'HT-Coreset', size=9.5, weight='bold', color=TEAL)
    text(0.74, 1.85, 'HI: 1,015,669 edges', size=8.1, color=MUTED)
    text(0.74, 1.69, 'LI: 1,384,810 edges', size=8.1, color=MUTED)
    text(2.51, 1.85, 'HI: 3,753 edges', size=8.1, color=MUTED)
    text(2.51, 1.69, 'LI: 2,268 edges', size=8.1, color=MUTED)

    ys = [1.52, 1.40, 1.28, 1.16]
    # A schematic population and subset; every illicit token is retained.
    tokens([0.18 + 0.124 * i for i in range(10)], ys,
           {(0, 1), (1, 7), (2, 3), (3, 8)})
    tokens([2.21, 2.51, 2.81], ys, {(i, 0) for i in range(4)})
    arrow((1.49, 1.34), (1.94, 1.34), color=TEAL, lw=1.2)

    text(1.625, 0.98, 'All illicit kept; benign sampled by stratum',
         size=8.1, weight='bold')
    ax.scatter(0.46, 0.79, s=17, marker='D', c=ORANGE, edgecolors='none')
    text(0.56, 0.79, 'Illicit', size=8.0, color=MUTED, ha='left')
    ax.scatter(1.14, 0.79, s=12, marker='o', c=BENIGN, edgecolors='none')
    text(1.24, 0.79, 'Benign', size=8.0, color=MUTED, ha='left')
    text(2.60, 0.79, 'Not to scale', size=7.7, color=MUTED)

    # One prediction set feeds both reports; no full-test inference is shown.
    text(1.625, 0.61, 'Predictions on AMLworld-Compact', size=8.6, weight='bold')
    ax.plot([1.625, 1.625], [0.52, 0.48], color=MUTED, lw=0.8)
    ax.plot([0.80, 2.45], [0.48, 0.48], color=MUTED, lw=0.8)
    arrow((0.80, 0.48), (0.80, 0.425))
    arrow((2.45, 0.48), (2.45, 0.425))

    # Restrained emphasis: the population estimate is the primary report.
    for x, width, fill, edge in [
        (0.035, 1.53, '#EAF3F3', '#AACACB'),
        (1.685, 1.53, '#F5F7F8', RULE),
    ]:
        ax.add_patch(FancyBboxPatch((x, 0.035), width, 0.385,
                     boxstyle='round,pad=0,rounding_size=0.035',
                     facecolor=fill, edgecolor=edge, linewidth=0.65))
    text(0.80, 0.31, 'HT-weighted', size=9.0, weight='bold', color=TEAL)
    text(2.45, 0.31, 'Unweighted', size=9.0, weight='bold')
    text(0.80, 0.14, 'Full-test estimates', size=8.1, color=TEAL)
    text(2.45, 0.14, 'Subset diagnostics', size=8.1)

    out = Path(__file__).resolve().parents[1] / 'results/figures/evaluation_overview.pdf'
    fig.savefig(out, metadata={'Title': 'AMLworld-Compact: target selection and evaluation'})
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
