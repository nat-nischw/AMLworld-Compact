# Manuscript figures

The current paper uses these five PDFs:

| File | Content |
| --- | --- |
| [evaluation_overview.pdf](evaluation_overview.pdf) | HT-Coreset selection and the two reports from the same predictions: HT-weighted full-test estimates and unweighted subset diagnostics. |
| [typology_f1_hi_small.pdf](typology_f1_hi_small.pdf) | Typology F1 on HI-Small. |
| [typology_anchor_heatmap_main.pdf](typology_anchor_heatmap_main.pdf) | Predicted labels on ground-truth-benign targets, with separate HI-Small and LI-Small panels. |
| [reduction_sweep.pdf](reduction_sweep.pdf) | Sampling-budget sweep. |
| [step_heatmap_per_model.pdf](step_heatmap_per_model.pdf) | Regex audit pass rates by model. |

Regenerate the overview from the code repository root with
`python scripts/plot_evaluation_overview.py` (requires the `figures` extra).
It is a schematic; its tokens are not individual experimental observations.

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
