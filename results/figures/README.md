# Figures

## README overview

The repository README and dataset card use the [PNG overview](readme_overview.png).
The [SVG version](readme_overview.svg) provides the vector artwork. This layout
uses sans-serif type and horizontal panels for reading on the web.

Regenerate it from the repository root with the `figures` extra installed:

```bash
python scripts/plot_readme_overview.py
```

## Manuscript figures

The current paper uses these five PDFs:

| File | Content |
| --- | --- |
| [evaluation_overview.pdf](evaluation_overview.pdf) | Panel (a) shows HT-Coreset target selection; panel (b) shows model predictions feeding HT-weighted full-test estimates and unweighted subset diagnostics. |
| [typology_f1_hi_small.pdf](typology_f1_hi_small.pdf) | Typology F1 on HI-Small. |
| [typology_anchor_heatmap_main.pdf](typology_anchor_heatmap_main.pdf) | Predicted labels on ground-truth-benign targets, with separate HI-Small and LI-Small panels. |
| [reduction_sweep.pdf](reduction_sweep.pdf) | Sampling-budget sweep. |
| [step_heatmap_per_model.pdf](step_heatmap_per_model.pdf) | Regex audit pass rates by model. |

Regenerate the manuscript figures from the code repository root (requires
the `figures` extra):

```bash
python scripts/plot_evaluation_overview.py
python scripts/plot_typology_anchor_main.py
python scripts/plot_reduction_sweep.py
python scripts/plot_typology_f1.py
python scripts/plot_step_heatmap.py
```

The overview is a schematic; its tokens are not individual experimental
observations. The other scripts read the released CSVs without training,
model calls, or new sampling draws. They support `--output` for another
destination.

Figures use embedded Computer Modern Roman fonts, with ordinary labels
at 9.5–11 pt at their intended print size. The overview and audit heatmap
are 77 mm wide; the other three figures are 160 mm wide. These match the
ACL column and text widths. Export uses a fixed canvas so cropping cannot
silently reduce the font sizes when LaTeX scales the PDF to those widths.

## Supporting analysis plots

These plots remain available as supporting results. They are omitted from the
manuscript PDF to avoid repeating its tables and text.

| File | Content |
| --- | --- |
| [ablation_illicit_retention.pdf](ablation_illicit_retention.pdf) | Illicit-target retention under the sampling baselines. |
| [ablation_variance_boxplot.pdf](ablation_variance_boxplot.pdf) | Variation over sampling draws for the construction ensemble. |
| [step_funnel.pdf](step_funnel.pdf) | Four-step trace audit, broken down by annotator. |

The sampling plots use the [draw-level results](../coreset/baselines_draws.csv)
and [aggregate results](../coreset/baselines_agg.csv). The trace-audit annotations
and summaries are in [results/audit](../audit/).

## Historical plots

The remaining PDFs in this directory are historical plots retained with the released
analysis outputs. They are not figures in the current manuscript and may use
earlier labels or scoring configurations. Use the current paper and the results
documented in the repository README when interpreting the reported experiments.
