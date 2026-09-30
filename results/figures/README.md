# Manuscript figures

The current paper uses these eight PDFs:

| File | Content |
| --- | --- |
| [evaluation_overview.pdf](evaluation_overview.pdf) | HT-Coreset selection and the two reports from the same predictions: HT-weighted full-test estimates and unweighted subset diagnostics. |
| [typology_f1_hi_small.pdf](typology_f1_hi_small.pdf) | Typology F1 on HI-Small. |
| [typology_anchor_heatmap_main.pdf](typology_anchor_heatmap_main.pdf) | Predicted labels on ground-truth-benign targets, with separate HI-Small and LI-Small panels. |
| [step_funnel.pdf](step_funnel.pdf) | Four-step trace audit, broken down by annotator. |
| [ablation_illicit_retention.pdf](ablation_illicit_retention.pdf) | Illicit-target retention under the sampling baselines. |
| [ablation_variance_boxplot.pdf](ablation_variance_boxplot.pdf) | Variation over sampling draws for the construction ensemble. |
| [reduction_sweep.pdf](reduction_sweep.pdf) | Sampling-budget sweep. |
| [step_heatmap_per_model.pdf](step_heatmap_per_model.pdf) | Regex audit pass rates by model. |

Regenerate the overview from the code repository root with
`python scripts/plot_evaluation_overview.py` (requires the `figures` extra).
It is a schematic; its tokens are not individual experimental observations.

The other PDFs in this directory are historical plots retained with the released
analysis outputs. They are not figures in the current manuscript and may use
earlier labels or scoring configurations. Use the current paper and the results
documented in the repository README when interpreting the reported experiments.
