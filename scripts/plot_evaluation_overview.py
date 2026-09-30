#!/usr/bin/env python3
"""Draw target retention, stratified sampling, and the two evaluation reports.

All marks denote evaluation targets, not graph nodes or extracted contexts.
The schematic keeps four illicit targets and samples eight benign targets.
Hard-negative census applies to the released budgets; other strata are sampled.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

WIDTH_MM, HEIGHT_MM = 77, 83
INK = '#243642'
MUTED = '#5B6B75'
ACCENT = '#355C9A'
ILLICIT = '#B65329'
HARD = '#496579'
BENIGN = '#748F9F'
RULE = '#D6E0E4'
STRATUM_FILL = '#F1F5F7'
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
    labels = []

    def text(x, y, value, size=8, weight='normal', color=INK, ha='center'):
        artist = ax.text(x, y, value, fontsize=size, fontweight=weight,
                         color=color, ha=ha, va='center', linespacing=1.1)
        labels.append(artist)
        return artist

    def rule(start, end, color=RULE, lw=0.65):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=lw, solid_capstyle='butt')

    def arrow(start, end, color=MUTED, lw=0.8):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle='-|>', mutation_scale=6.8,
            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
            capstyle='butt', joinstyle='miter'))

    def token(x, y, illicit=False, hard=False):
        ax.scatter(x, y, s=14 if illicit else 9.5,
                   marker='D' if illicit else 'o',
                   color=ILLICIT if illicit else HARD if hard else BENIGN,
                   linewidths=0, zorder=3)

    # Counts describe actual target reduction; the selection diagram is schematic.
    text(SIDE_PAD_PT, 227, 'Fewer targets, full-test estimates', size=10.2,
         weight='bold', ha='left')
    for y, split, full, kept, reduction in [
        (209, 'HI-Small', '1,015,669', '3,753', '271'),
        (192, 'LI-Small', '1,384,810', '2,268', '611'),
    ]:
        text(SIDE_PAD_PT, y, split, size=8.1, weight='bold', ha='left')
        text(94, y, full, size=8.1, ha='right', color=MUTED)
        arrow((99, y), (114, y), color=ACCENT)
        text(119, y, kept, size=8.5, weight='bold', ha='left')
        text(width - SIDE_PAD_PT, y,
             '\N{ALMOST EQUAL TO}' + reduction + '\N{MULTIPLICATION SIGN}',
             size=12.2, weight='bold', color=ACCENT, ha='right')
    rule((SIDE_PAD_PT, 180), (width - SIDE_PAD_PT, 180))

    # The method heading describes the whole selection process. The two
    # data-set headings sit at the same level above their target columns.
    text(117, 169, 'HT-Coreset', size=9.1, weight='bold', color=ACCENT)
    text(57, 155, 'Full test', size=8.2, weight='bold')
    text(174, 155, 'AMLworld-Compact', size=7.8, weight='bold')

    # Equal illicit counts on both sides; all hard benign targets fit here.
    text(3, 134, 'Illicit', size=7.3, color=ILLICIT, ha='left')
    for x in [39, 51, 63, 75]:
        token(x, 134, illicit=True)
    for x in [153, 167, 181, 195]:
        token(x, 134, illicit=True)
    text(117, 143, 'Keep all', size=7.6, color=ILLICIT)
    arrow((90, 134), (144, 134), color=ILLICIT)

    text(3, 114, 'Hard\nbenign', size=7.1, color=HARD, ha='left')
    for x in [51, 63]:
        token(x, 114, hard=True)
    for x in [167, 181]:
        token(x, 114, hard=True)
    text(117, 123, 'Keep all*', size=7.6, color=HARD)
    arrow((90, 114), (144, 114), color=HARD)

    # Three visible bands are the remaining benign difficulty strata.
    # Each source band contains ten targets; two are selected and arranged
    # in the corresponding compact-set row. Equal marker area prevents
    # confusion between a sampled target and its importance weight.
    text(3, 80, 'Other\nbenign', size=7.1, color=MUTED, ha='left')
    text(117, 103, 'Sample within strata', size=7.2, color=MUTED)
    for y in [92, 80, 68]:
        ax.add_patch(Rectangle((31, y - 4.8), 52, 9.6,
                              facecolor=STRATUM_FILL, edgecolor='none'))
        ax.add_patch(Rectangle((150, y - 4.8), 48, 9.6,
                              facecolor=STRATUM_FILL, edgecolor='none'))
        for x in [35, 46, 57, 68, 79]:
            for dy in [-2.35, 2.35]:
                token(x, y + dy)
        for x in [167, 181]:
            token(x, y)
        arrow((90, y), (144, y), color=BENIGN, lw=0.7)

    # One connector includes every selected target and reaches a shared
    # prediction node. The two reports reuse those same predictions.
    rule((204, 139), (213, 139), color=MUTED, lw=0.7)
    rule((213, 139), (213, 60), color=MUTED, lw=0.7)
    center = width / 2
    rule((213, 60), (center, 60), color=MUTED, lw=0.7)
    arrow((center, 60), (center, 52.2), lw=0.7)
    text(center, 45.5, 'One set of predictions', size=8.7, weight='bold')

    card_width = (width - 2 * SIDE_PAD_PT - 8) / 2
    left = SIDE_PAD_PT + card_width / 2
    right = width - left
    rule((center, 39.3), (center, 37), color=MUTED, lw=0.7)
    rule((left, 37), (right, 37), color=MUTED, lw=0.7)
    arrow((left, 37), (left, 31.7), lw=0.7)
    arrow((right, 37), (right, 31.7), lw=0.7)

    ax.add_patch(Rectangle((SIDE_PAD_PT, 1.5), card_width, 27.5,
                          facecolor='#EFF3FA', edgecolor='none'))
    rule((SIDE_PAD_PT, 29), (SIDE_PAD_PT + card_width, 29), color=ACCENT, lw=1.05)
    rule((width - SIDE_PAD_PT - card_width, 29),
         (width - SIDE_PAD_PT, 29), color=MUTED, lw=0.7)
    text(left, 22.4, 'HT-weighted', size=8.8, weight='bold', color=ACCENT)
    text(right, 22.4, 'Unweighted', size=8.8, weight='bold')
    text(left, 14, r'$w_e=1/\pi_e$', size=8, color=ACCENT)
    text(right, 14, r'$w_e=1$', size=8, color=MUTED)
    text(left, 5.6, 'Full-test estimates', size=7.9, color=ACCENT)
    text(right, 5.6, 'Subset diagnostics', size=7.9)

    # Check the physical figure boundary at export size, including all labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        bounds = label.get_window_extent(renderer)
        if not (fig.bbox.x0 <= bounds.x0 and bounds.x1 <= fig.bbox.x1
                and fig.bbox.y0 <= bounds.y0 and bounds.y1 <= fig.bbox.y1):
            raise ValueError(f'Label outside figure: {label.get_text()}')
    out = Path(__file__).resolve().parents[1] / 'results/figures/evaluation_overview.pdf'
    fig.savefig(out, metadata={
        'Title': 'AMLworld-Compact: retain illicit targets, sample benign strata',
        'Creator': 'Matplotlib; editable figure source accompanies the paper',
    })
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
