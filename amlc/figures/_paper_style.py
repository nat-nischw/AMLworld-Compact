"""Shared visual theme for all paper figures.

Import this module FIRST in any plot script:

    from _paper_style import apply_style, COLORS, LLM_PALETTE, short_model
    from _paper_style import FIG_COL, FIG_TEXT  # render-matched figsize templates
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_COL)

Design choices:
- Okabe-Ito colorblind-safe palette as the primary qualitative scale
- Single serif font + sizes tuned for two-column ACL layout
- Figsize templates match the ACL columnwidth (77mm) and textwidth (160mm)
  so embedded fonts render at their native point size — no scaling, no
  font-size mismatch with body text.
- Semantic color roles (ml/llm/dt/ours/fail/partial) so the same model
  is the same colour across every figure
"""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

from .. import archive

# ── ACL EMNLP layout dimensions (inches) ────────────────────────────────
COL_W   = 77 / 25.4    # ACL columnwidth, inches
TEXT_W  = 160 / 25.4   # ACL textwidth, inches (two columns + gutter)

# ── Figsize templates (use these instead of bare tuples) ────────────────
FIG_COL  = (COL_W,  2.4)        # single-column, default aspect
FIG_COL_TALL = (COL_W,  3.0)
FIG_TEXT = (TEXT_W, 3.0)        # textwidth (figure*), default aspect
FIG_TEXT_TALL = (TEXT_W, 3.6)

# ── Okabe-Ito (colorblind-safe) primitives ──────────────────────────────
OKABE = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "grey":       "#666666",
}

# ── Semantic colour roles (same across every figure) ────────────────────
COLORS = {
    # Method families
    "ml":          OKABE["grey"],        # ML baselines (LightGBM/XGBoost/Ensemble)
    "llm":         OKABE["orange"],      # LLM-alone
    "dt":          OKABE["blue"],        # Doubt Triage hybrid
    "ours":        OKABE["green"],       # the HT-Coreset
    # Pass/fail traffic light
    "fail":        OKABE["vermillion"],
    "partial":     OKABE["yellow"],
    "prior":       OKABE["skyblue"],     # closest prior baseline
    "ref_line":    "#95a5a6",            # neutral grey reference line
    # Error-transition groups (ML-vs-LLM analysis)
    "both_correct": OKABE["green"],
    "ml_only":      OKABE["blue"],
    "llm_only":     OKABE["orange"],
    "both_wrong":   OKABE["vermillion"],
    # Step rubric (heuristic vs DeepSeek annotators) — softer pair to avoid
    # neon clash next to numeric labels at the bar caps
    "annot_a":      "#4C72B0",   # steel blue
    "annot_b":      "#DD8452",   # terracotta
}

# Qualitative palette for up to 7 LLMs / 8 typology classes
LLM_PALETTE = [
    OKABE["blue"],       # GPT-OSS-20B
    OKABE["orange"],     # Nem-Nano-30B
    OKABE["green"],      # Qwen3.5-35B-A3B
    OKABE["vermillion"], # GPT-OSS-120B
    OKABE["purple"],     # Nem-Super-120B
    OKABE["skyblue"],    # Qwen3.5-397B
    OKABE["yellow"],     # Qwen3.5-27B
]

TYPOLOGY_PALETTE = [
    OKABE["blue"], OKABE["orange"], OKABE["green"], OKABE["vermillion"],
    OKABE["purple"], OKABE["skyblue"], OKABE["yellow"], OKABE["grey"],
]


FONT_FAMILY = "ACL Computer Modern"


def _register_roman_fonts():
    """Use Matplotlib's bundled Computer Modern, an ACL-approved alternative.

    A shared family name makes bold headings resolve to cmb10 instead of
    silently reusing the regular face. No system font installation is needed.
    """
    if any(entry.name == FONT_FAMILY for entry in font_manager.fontManager.ttflist):
        return
    directory = Path(mpl.get_data_path()) / "fonts" / "ttf"
    for file, style, weight in [("cmr10.ttf", "normal", "normal"),
                                ("cmb10.ttf", "normal", "bold"),
                                ("cmmi10.ttf", "italic", "normal")]:
        font_manager.fontManager.ttflist.append(font_manager.FontEntry(
            fname=str(directory / file), name=FONT_FAMILY, style=style,
            weight=weight, stretch="normal"))


def apply_style():
    """Set matplotlib rcParams for paper-grade output.

    Idempotent: safe to call multiple times.
    """
    _register_roman_fonts()
    mpl.rcParams.update({
        # Typography — serif, ACL-style; sizes match ACL body 10-11pt at
        # native render size (figures must be generated at COL_W or TEXT_W
        # inches; do not generate oversize and let LaTeX scale down).
        "font.family":       "serif",
        "font.serif":        [FONT_FAMILY],
        "mathtext.fontset":   "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "legend.fontsize":   9.5,
        "xtick.labelsize":   9.5,
        "ytick.labelsize":   9.5,
        # Figure & layout
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      None,
        "savefig.pad_inches": 0,
        # Axes styling — clean, no top/right spines
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#222222",
        "xtick.color":       "#333333",
        "ytick.color":       "#333333",
        # Grid (subtle)
        "axes.grid":         False,
        "grid.color":        "#cccccc",
        "grid.linestyle":    ":",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
        # Legend
        "legend.frameon":       False,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.4,
    })


def short_model(name: str) -> str:
    """Canonical short form of an LLM model name. Use everywhere."""
    return (name
            .replace("Nemotron-3-Super-120B", "Nem-Super-120B")
            .replace("Nemotron-3-Nano-30B",   "Nem-Nano-30B")
            .replace("Nemotron-3-Super",      "Nem-Super-120B")
            .replace("Nemotron-3-Nano",       "Nem-Nano-30B")
            .replace("Qwen3.5-397B-A17B",     "Qwen3.5-397B")
            .replace("Qwen3-5-397B-A17B",     "Qwen3.5-397B")
            .replace("Qwen3.5-35B-A3B",       "Qwen3.5-35B")
            .replace("Qwen3-5-35B-A3B",       "Qwen3.5-35B")
            .replace("Qwen3-5-27B",           "Qwen3.5-27B")
            .replace(archive.member_dir("GCPAL+GFP"), "GCPAL+GFP")
            .replace("GCPAL-knn-temporal",    "GCPAL")
            .replace("ML_ensemble",           "ML Ensemble")
            .replace("Ensemble_Soft",         "Ensemble")
            )
