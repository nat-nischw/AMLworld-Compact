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
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

WIDTH_MM, HEIGHT_MM = 77, 87
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
    # ACL permits Computer Modern Roman when Times is unavailable. Register
    # the bundled regular/bold faces under one family for portable rendering.
    family = 'ACL Computer Modern'
    font_dir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    if not any(entry.name == family for entry in font_manager.fontManager.ttflist):
        for file, weight in [('cmr10.ttf', 'normal'), ('cmb10.ttf', 'bold')]:
            font_manager.fontManager.ttflist.append(font_manager.FontEntry(
                fname=str(font_dir / file), name=family, weight=weight,
                style='normal', stretch='normal'))
    plt.rcParams.update({
        'font.family': family,
        'font.size': 10,
        'axes.formatter.use_mathtext': True,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'cm',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.bbox': None,
        'savefig.pad_inches': 0,
    })
    width, height = WIDTH_MM / 25.4 * 72, HEIGHT_MM / 25.4 * 72
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
                     facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    labels = []

    def text(x, y, value, size=10, weight='normal', color=INK, ha='center'):
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
    text(SIDE_PAD_PT, 238.5, 'Fewer targets, full-test estimates', size=11,
         weight='bold', ha='left')
    for y, split, full, kept, reduction in [
        (220.5, 'HI-Small', '1,015,669', '3,753', '271'),
        (203.5, 'LI-Small', '1,384,810', '2,268', '611'),
    ]:
        text(SIDE_PAD_PT, y, split, size=9.5, weight='bold', ha='left')
        text(94, y, full, size=9.5, ha='right', color=MUTED)
        arrow((99, y), (114, y), color=ACCENT)
        text(119, y, kept, size=10, weight='bold', ha='left')
        text(width - SIDE_PAD_PT, y,
             r'$\approx' + reduction + r'\times$',
             size=11, weight='bold', color=ACCENT, ha='right')
    rule((SIDE_PAD_PT, 191.5), (width - SIDE_PAD_PT, 191.5))

    # The method heading describes the whole selection process. The two
    # data-set headings sit at the same level above their target columns.
    text(SIDE_PAD_PT, 182.5, '(a) Target selection: HT-Coreset',
         size=10, weight='bold', color=ACCENT, ha='left')
    text(57, 166.5, 'Full test', size=9.5, weight='bold')
    text(174, 166.5, 'AMLworld-Compact', size=9.5, weight='bold')

    # Equal illicit counts on both sides; all hard benign targets fit here.
    text(3, 145.5, 'Illicit', size=9.5, color=ILLICIT, ha='left')
    for x in [39, 51, 63, 75]:
        token(x, 145.5, illicit=True)
    for x in [153, 167, 181, 195]:
        token(x, 145.5, illicit=True)
    text(117, 154.5, 'Keep all', size=9.5, color=ILLICIT)
    arrow((90, 145.5), (144, 145.5), color=ILLICIT)

    text(3, 125.5, 'High-score\nbenign', size=9.5, color=HARD, ha='left')
    for x in [63, 75]:
        token(x, 125.5, hard=True)
    for x in [167, 181]:
        token(x, 125.5, hard=True)
    text(117, 134.5, 'Keep all at release', size=9.5, color=HARD)
    arrow((90, 125.5), (144, 125.5), color=HARD)

    # Three visible bands are the remaining benign difficulty strata.
    # Each source band contains ten targets; two are selected and arranged
    # in the corresponding compact-set row. Equal marker area prevents
    # confusion between a sampled target and its importance weight.
    text(3, 91.5, 'Other\nbenign', size=9.5, color=MUTED, ha='left')
    text(117, 114.5, '3 difficulty strata', size=9.5, color=MUTED)
    for y in [103.5, 91.5, 79.5]:
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

    # Group only the three other-benign bands; the high-score stratum above
    # remains separate. Leave clearance from each sampling arrow's shaft.
    rule((83.5, 108.3), (86.5, 108.3), color=BENIGN)
    rule((86.5, 108.3), (86.5, 74.7), color=BENIGN)
    rule((83.5, 74.7), (86.5, 74.7), color=BENIGN)

    # A bracket collects every selected row. Its connection to the model
    # prediction node stays outside both panel headings and the row labels.
    rule((204, 150.5), (211, 150.5), color=MUTED, lw=0.7)
    rule((211, 150.5), (211, 74.5), color=MUTED, lw=0.7)
    rule((204, 74.5), (211, 74.5), color=MUTED, lw=0.7)
    rule((211, 112.5), (215, 112.5), color=MUTED, lw=0.7)
    rule((215, 112.5), (215, 52.5), color=MUTED, lw=0.7)

    text(SIDE_PAD_PT, 67.5, '(b) Evaluation and reporting',
         size=10, weight='bold', ha='left')
    center = width / 2
    arrow((215, 52.5), (164, 52.5), lw=0.7)
    text(center, 52.5, 'Model predictions', size=10.5, weight='bold')

    card_width = (width - 2 * SIDE_PAD_PT - 8) / 2
    left = SIDE_PAD_PT + card_width / 2
    right = width - left
    # Equal vertical branches leave a visible shaft above each arrowhead,
    # with matching clearance from both report headers.
    branch_y = 43
    rule((center, 46.5), (center, branch_y), color=MUTED, lw=0.7)
    rule((left, branch_y), (right, branch_y), color=MUTED, lw=0.7)
    arrow((left, branch_y), (left, 37.5), lw=0.7)
    arrow((right, branch_y), (right, 37.5), lw=0.7)

    ax.add_patch(Rectangle((SIDE_PAD_PT, 1.5), card_width, 33,
                          facecolor='#EFF3FA', edgecolor='none'))
    rule((SIDE_PAD_PT, 34.5), (SIDE_PAD_PT + card_width, 34.5), color=ACCENT, lw=1.05)
    rule((width - SIDE_PAD_PT - card_width, 34.5),
         (width - SIDE_PAD_PT, 34.5), color=MUTED, lw=0.7)
    text(left, 28, 'HT-weighted', size=10.5, weight='bold', color=ACCENT)
    text(right, 28, 'Unweighted', size=10.5, weight='bold')
    text(left, 16, r'$w_e=1/\pi_e$', size=10, color=ACCENT)
    text(right, 16, r'$w_e=1$', size=10, color=MUTED)
    text(left, 5, 'Full-test estimates', size=9.5, color=ACCENT)
    text(right, 5, 'Subset diagnostics', size=9.5)

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
