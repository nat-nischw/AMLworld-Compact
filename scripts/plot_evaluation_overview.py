#!/usr/bin/env python3
"""Draw the manuscript's target-selection overview at its final column size.

The symbols represent evaluation targets, not a graph being downsampled.
Their class proportions are schematic. The actual released target counts
are printed above them; the same illicit symbols survive selection.
Run from any directory. No experimental outputs are recomputed.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

WIDTH_MM, HEIGHT_MM = 77, 71
INK = '#202F39'
MUTED = '#54636D'
TEAL = '#086A72'
ILLICIT = '#B65329'
BENIGN = '#8098A7'
RULE = '#D2DADF'
ARROW_HEAD_PT = 7.2
SIDE_PAD_PT = 2


def main():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'mathtext.fontset': 'dejavusans',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    width, height = WIDTH_MM / 25.4 * 72, HEIGHT_MM / 25.4 * 72
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
                     facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()

    def text(x, y, value, size=8, weight='normal', color=INK, ha='center'):
        return ax.text(x, y, value, fontsize=size, fontweight=weight,
                       color=color, ha=ha, va='center')

    def rule(start, end, color=RULE, lw=0.65):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=lw, solid_capstyle='butt')

    def arrow(start, end, color=MUTED, lw=0.85):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                     mutation_scale=ARROW_HEAD_PT, linewidth=lw, color=color,
                     shrinkA=0, shrinkB=0, capstyle='butt', joinstyle='miter'))

    def token(x, y, illicit=False):
        ax.scatter(x, y, s=15 if illicit else 10,
                   marker='D' if illicit else 'o',
                   color=ILLICIT if illicit else BENIGN,
                   linewidths=0, zorder=3)

    # Exact target counts carry the scale; schematic marks explain selection.
    text(SIDE_PAD_PT, 194, 'Fewer targets, full-test estimates', size=10.2,
         weight='bold', ha='left')
    for y, split, full, kept, reduction in [
        (175, 'HI-Small', '1,015,669', '3,753', '271'),
        (158, 'LI-Small', '1,384,810', '2,268', '611'),
    ]:
        text(SIDE_PAD_PT, y, split, size=8.1, weight='bold', ha='left')
        text(94, y, full, size=8.1, ha='right', color=MUTED)
        arrow((99, y), (114, y), color=TEAL)
        text(119, y, kept, size=8.5, weight='bold', ha='left')
        text(width - SIDE_PAD_PT, y, '\N{ALMOST EQUAL TO}' + reduction + '\N{MULTIPLICATION SIGN}',
             size=12.2, weight='bold', color=TEAL, ha='right')
    rule((SIDE_PAD_PT, 146), (width - SIDE_PAD_PT, 146))

    text(SIDE_PAD_PT, 135, 'HT-Coreset', size=9.1, weight='bold', color=TEAL,
         ha='left')
    text(width - SIDE_PAD_PT, 135, 'Target selection', size=8, color=MUTED, ha='right')
    full_x, subset_x = 36, 172
    text(full_x, 123, 'Full test', size=8.2, weight='bold')
    text(subset_x, 123, 'AMLworld-Compact', size=8.0, weight='bold')

    # Retention and sampling are separate visual lanes. All three illicit
    # symbols are retained; the illustrative subset has six benign symbols.
    for offset in [-13, 0, 13]:
        token(full_x + offset, 105, illicit=True)
        token(subset_x + offset, 105, illicit=True)
    text(107, 113, 'Keep all illicit', size=8, color=ILLICIT)
    arrow((71, 105), (145, 105), color=ILLICIT)

    for y in [88, 80, 72]:
        for offset in range(-27, 28, 9):
            token(full_x + offset, y)
        for offset in [-7, 7]:
            token(subset_x + offset, y)
    text(107, 91, 'Sample benign', size=8, color=MUTED)
    arrow((71, 80), (145, 80), color=MUTED)
    text(107, 74, 'by stratum', size=8, color=MUTED)

    # All predictions are generated on the selected targets. A single shared
    # prediction node then feeds both reporting choices, without new inference.
    rule((207, 109), (213, 109), color=MUTED, lw=0.65)
    rule((213, 109), (213, 66.5), color=MUTED, lw=0.8)
    center = width / 2
    card_width = (width - 2 * SIDE_PAD_PT - 8) / 2
    left = SIDE_PAD_PT + card_width / 2
    right = width - left
    rule((213, 66.5), (center, 66.5), color=MUTED, lw=0.8)
    arrow((center, 66.5), (center, 61), color=MUTED)
    text(center, 54.5, 'One set of predictions', size=8.7, weight='bold')
    rule((center, 47.8), (center, 43.5), color=MUTED, lw=0.8)
    rule((left, 43.5), (right, 43.5), color=MUTED, lw=0.8)
    # Matching vertical shafts, with 2.7 pt between each tip and its panel rule.
    arrow((left, 43.5), (left, 36.8))
    arrow((right, 43.5), (right, 36.8))

    # Straight rules and restrained fill distinguish the primary estimate.
    ax.add_patch(Rectangle((SIDE_PAD_PT, 1.5), card_width, 32.6, facecolor='#EFF6F5',
                           edgecolor='none'))
    rule((SIDE_PAD_PT, 34.1), (SIDE_PAD_PT + card_width, 34.1), color=TEAL, lw=1.15)
    rule((width - SIDE_PAD_PT - card_width, 34.1),
         (width - SIDE_PAD_PT, 34.1), color=MUTED, lw=0.8)
    text(left, 26.5, 'HT-weighted', size=9, weight='bold', color=TEAL)
    text(right, 26.5, 'Unweighted', size=9, weight='bold')
    text(left, 16, r'$w_e=1/\pi_e$', size=8.1, color=TEAL)
    text(right, 16, r'$w_e=1$', size=8.1, color=MUTED)
    text(left, 6.3, 'Full-test estimates', size=8.1, color=TEAL)
    text(right, 6.3, 'Subset diagnostics', size=8.1)

    out = Path(__file__).resolve().parents[1] / 'results/figures/evaluation_overview.pdf'
    fig.savefig(out, metadata={
        'Title': 'AMLworld-Compact: fewer targets, full-test estimates',
        'Creator': 'Matplotlib; editable figure source accompanies the paper',
    })
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
