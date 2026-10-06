# Sampling uncertainty from cached LLM predictions

This analysis estimates uncertainty introduced by sampling benign targets in
the released AMLworld-Compact evaluation set. It uses the existing predictions
for 7 models × 2 prompts × 2 splits × 5 archived inference runs. No model training,
API calls, new inference, or additional benchmark draws are required.

The files retain historical `seed` field names and run identifiers. The archived
requests did not pass these values to the model API; the current runner does.
The paper calls this variation run SD; cached files retain seed-based field names.

## Reproduce the analysis

Run from the repository root with the package dependencies installed. The
portable files in `inputs/` are sufficient for the numerical analysis:

```bash
python scripts/25_score_sampling_uncertainty.py \
  --inputs results/analysis/sampling_uncertainty/inputs \
  --out results/analysis/sampling_uncertainty
```

To regenerate the two manuscript tables as well, add
`--paper-tables /path/to/paper/tables`. This writes
`llm_sampling_uncertainty.tex` and `llm_sampling_contrasts.tex` in that directory.

Rebuilding the portable inputs requires the original local archive and staged
evaluation-set arrays. Replace the two example source paths below:

```bash
python scripts/24_prepare_sampling_uncertainty.py \
  --archive /path/to/archive/outputs \
  --coreset-root /path/to/amlcompact-dataset \
  --reference-csv results/analysis/llm_ht_weighted.csv \
  --out results/analysis/sampling_uncertainty/inputs
```

The archive directory contains `eval_subsets/`, `test_probs/`, and the model
output directories. `--coreset-root` accepts either a dataset directory with
`extras/` or the `extras/` directory itself. Input preparation stops on missing,
duplicate, out-of-range, or misaligned target records; it does not impute them.

## Portable inputs and provenance

Each split has three input files:

| File | Contents |
| --- | --- |
| `inputs/{split}.npz` | Binary predictions, labels, target IDs, stratum IDs, weights, cell names, and inference run identifiers. Loads with `allow_pickle=False`. |
| `inputs/{split}.json` | Stratum population and sample sizes, source SHA256s, preparation-script/output hashes, and per-run validation. |
| `inputs/{split}_strata.csv` | One row per target, with its case/center-edge IDs, full-test index, label, stratum, population/sample sizes, and weight. |

Prediction tensors have shape `(3753, 14, 5)` for HI-Small and `(2268, 14, 5)`
for LI-Small. Axes are released target position, model/prompting cell, and run.
Cells follow `LLM_MODELS` order, with `ICL-FS` then `ICL-ZS` for each model;
the `models`, `promptings`, and `seeds` arrays record the actual ordering.

Strata are reconstructed from the **original three-scorer construction
probabilities**, independently of the current two-booster task reference.
Reconstructed weights match the corrected released weights bit-for-bit; targets
match the archived draw bit-for-bit. The original uncorrected LI weight vector
is hashed for provenance but is not used for scoring. Equal weights do not
merge different strata.

All 421,470 archived prediction records have matching canonical center-edge IDs
and labels. Their binary verdicts reproduce all 140 existing per-seed HT and
compact results, plus both predict-all references, with maximum absolute
metric difference below `1.5e-14` percentage points. Preparation preserves the
archived `bool(record.get("illicit", False))` scoring convention; no missing,
null, or nonbinary illicit values occurred in these files. Portable inputs
contain no reasoning traces or raw response text.

## Estimand and uncertainty

The estimand is the **mean of five per-run metrics**, not a metric computed
after averaging predictions. The analysis conditions on the construction
strata, realized sample allocation, fixed per-target contexts, and the five
fixed prediction vectors. Within each sampled stratum, the conditional design
is simple random sampling without replacement. LI-Small's final allocation
includes its one residual-fill target in the first easy stratum.

All illicit targets and hard-negative targets are censused. Therefore only
the sampled easy-benign strata contribute target-sampling uncertainty to the
false-positive total. The calculation uses finite-population corrections and
within-stratum sample covariance. Ratio linearization gives sampling SEs for
precision, F1, and precision lift. Same-target covariance is retained across
all five runs and, for paired FS–ZS differences, across both prompts.

- **Run SD** is the descriptive standard deviation of the five metrics,
  using `ddof=1`.
- **Sampling SE** conditions on those five prediction vectors. It is reported
  separately from run SD; it is neither divided by an extra `sqrt(5)` nor
  combined with run SD into a total uncertainty estimate.
- **95% intervals** are conservative confidence envelopes from equal-tailed
  hypergeometric count inversion. Bonferroni allocation covers the 15
  run–stratum counts within each cell (`5 runs × 3 sampled strata`). Paired
  contrasts cover 30 counts across their two cells. Endpoints are transformed
  to metrics and averaged; contrasts subtract the joint endpoints.

The envelopes remain applicable when a sampled stratum contains only positive
or only negative predictions, where an empirical sampling SE can be zero.
They are not Wald intervals constructed as an estimate ± 1.96 SE. Contrast
SEs exploit pairing; conservative contrast envelopes do not. Coverage is at
least 95% **per cell or per contrast**, not simultaneous across all 28 cells or
all 14 contrasts in the tables. No table-wide ranking claim follows from them.

Recall has zero target-sampling SE under these conditions because illicit
targets are censused; its between-run variation remains. The deterministic
predict-all rule has known predictions throughout the population, so its
subtraction introduces no additional sampling uncertainty.

These results do not measure variation over future inference runs, changing
contexts, reconstructed coresets, future graphs, or real banking populations.
They do not replace direct full-versus-coreset LLM validation or experiments
on independent coreset draws.

## Outputs and units

| Artifact | Contents and units |
| --- | --- |
| `summary.csv` | 28 five-run means, run SDs, sampling SEs, confidence envelopes, and comparisons with predict-all. P/R/F1 are percentages; their SDs, SEs, and differences are percentage points. |
| `per_seed.csv` | 140 per-seed results. P/R/F1, their SEs, and endpoints are fractions; confusion totals and FP uncertainty are counts. Bounds use the allocation for the parent five-run cell. |
| `fs_zs_contrasts.csv` | 14 FS-minus-ZS comparisons with paired SEs and conservative envelopes. P/F1 differences and their uncertainty are percentage points. |
| `analysis.json` | Full results, covariance matrices, sampled-stratum positive counts, interval error allocations, and predict-all references. P/R/F1 use fractions; confusion quantities use counts. |
| Optional `.tex` tables | Manuscript tables with F1 as percentages and SD/SE in units of `10^-3` percentage points. Interval endpoints are rounded outwards. |

Precision lift and all of its uncertainties/differences are dimensionless in
every artifact. It divides precision by the exact full-test illicit prevalence.
The numerical implementation is
[`amlc/analysis/sampling_uncertainty.py`](../../../amlc/analysis/sampling_uncertainty.py).
