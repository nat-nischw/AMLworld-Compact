# Held-out GFP-tree sampling experiment

This CPU-only experiment evaluates transfer between two existing GFP-tree
families using cached full-test predictions. It does not validate sampling
accuracy for LLMs, new inference seeds, new contexts, or future graphs. Both
families use GFP features, so the family separation is a constrained check.
The frozen protocol is in `protocol.json`; it was saved before inspecting
held-out errors. All 36 configurations are reported without choosing a budget
or acceptance threshold from the outcomes.

## Design and target

For each dataset, construct using the mean probabilities of the five LightGBM
seeds and evaluate the five fixed XGBoost seeds, then swap families. The
constructor threshold is the mean of its five saved **validation** thresholds;
the hard cutoff is half that mean. Each held-out seed is evaluated at its own
saved validation threshold. No held-out prediction or threshold enters the
constructor, allocation, or budget selection. The population target is the
arithmetic mean of five per-seed metrics, not a metric computed from pooled
predictions or pooled confusion counts.

HI-Small has 1,015,669 cached test rows (1,251 illicit); LI-Small has 1,384,810
(756 illicit). Every illicit row is retained. Benign budgets are 2, 4, and 8
times the illicit count: respectively 2,502/5,004/10,008 on HI-Small and
1,512/3,024/6,048 on LI-Small. Each configuration has 500 independent draws.
All three samplers have the same exact budget at a given multiplier:

* `uniform_ht`: simple random sampling of benign rows with HT weights.
* `difficulty_ht`: benign difficulty strata, without a hard reserve, with HT
  weights. Difficulty is `1 - 2*abs(p_constructor - 0.5)`; cut points are the
  33rd and 67th percentiles, matching the released sampler.
* `hard_terciles_ht`: the released design's 30% hard reserve followed by
  difficulty strata over remaining benign rows, with HT weights. Each
  nonempty stratum has positive inclusion probability, even when hard rows
  exceed the hard budget. In these cached configurations the hard stratum is
  a census at every budget.

The comparison deliberately uses a fixed allocation: after the released
reserve and integer tercile allocations, residual slots are assigned in
proportion to residual stratum capacities by largest remainder, breaking ties
in fixed stratum order. This replaces the released sampler's **random residual
fill** and is not an exact recreation of that released draw. Independent
simple random sampling without replacement within each resulting stratum has
inclusion probability `n_h/N_h` and weight `N_h/n_h`. Empty strata are omitted;
budgets that cannot give every nonempty stratum a positive allocation are
rejected. The complete rule and deterministic RNG schedule are in the protocol.

## Estimation and uncertainty

TP and FN are exact because illicit rows form a census; FP is estimated by the
HT total. Precision and F1 are nonlinear ratio estimates and need not be
unbiased. Recall is exact for every seed and every draw. The five seeds share
the same sampled rows, preserving their sampling dependence; their individual
metrics are averaged for each draw before errors and intervals are summarized.

Intervals invert exact finite-population hypergeometric counts, with
Bonferroni alpha `0.05 / (5 * number_of_noncensus_benign_strata)`. Component
endpoints are summed into FP bounds, transformed monotonically to precision
and F1, and averaged across seeds. This matches `sampling_uncertainty.py`.
Intervals have at least 95% coverage for each fixed five-seed cell separately;
coverage is not simultaneous across all reported configurations. Per-seed
diagnostics use these same simultaneous component bounds. Repeated inversions
are cached. Observing zero sampled false positives in a stratum does not imply
that the unobserved population has zero false positives.

All CSV metrics, signed biases, MAEs, RMSEs, SDs, and interval widths are
**fractions**. Multiply by 100 for percentages or percentage points. SD is the
sample SD across 500 sampling draws (`ddof=1`), not variation over training
seeds. Coverage is the proportion of draws covering the exact population
target; `coverage_mcse` is its plug-in binomial Monte Carlo standard error.
At observed coverage 1 this plug-in error is zero and should not be read as
certainty about true coverage. Width is the average upper-minus-lower bound.
Tiny nonzero SDs for constant arrays can occur at floating-point precision.

## Primary 1:2 result

The following F1 errors and widths are in **percentage points**; coverage is
a proportion. LGB and XGB abbreviate LightGBM+GFP and XGBoost+GFP. Construction
family is left of the arrow. Full five-seed mean F1 is 67.6881% / 65.6813% on
HI for held-out XGB / LGB, and 28.5917% / 27.4937% on LI.

| Split and transfer | Sampler | Bias | MAE | RMSE | Coverage | Mean 95% width |
|---|---|---:|---:|---:|---:|---:|
| HI LGB → XGB | Uniform | 0.743 | 5.953 | 6.756 | 1.000 | 40.254 |
| HI LGB → XGB | Difficulty | 0.994 | 5.937 | 6.737 | 0.996 | 58.729 |
| HI LGB → XGB | Hard + terciles | 0.000 | 0.000 | 0.000 | 1.000 | 54.972 |
| HI XGB → LGB | Uniform | 0.859 | 6.697 | 7.410 | 0.998 | 40.375 |
| HI XGB → LGB | Difficulty | 0.546 | 6.733 | 7.597 | 0.996 | 58.260 |
| HI XGB → LGB | Hard + terciles | 0.197 | 0.731 | 1.595 | 1.000 | 53.234 |
| LI LGB → XGB | Uniform | 1.157 | 3.487 | 4.436 | 1.000 | 26.460 |
| LI LGB → XGB | Difficulty | 1.105 | 3.508 | 4.526 | 1.000 | 29.720 |
| LI LGB → XGB | Hard + terciles | 0.037 | 0.179 | 0.635 | 1.000 | 27.457 |
| LI XGB → LGB | Uniform | 1.668 | 5.346 | 6.023 | 0.998 | 27.087 |
| LI XGB → LGB | Difficulty | 2.141 | 5.170 | 5.757 | 1.000 | 30.379 |
| LI XGB → LGB | Hard + terciles | 0.552 | 1.624 | 2.580 | 1.000 | 27.345 |

Hard + terciles has the lowest empirical F1 RMSE in all 12 dataset/direction/
budget comparisons. Difficulty-only stratification does not consistently
improve on uniform sampling. The conservative envelopes remain broad, even
where the point estimates are exact; this experiment therefore supports
point-estimation transfer for these fixed GFP trees, without establishing
tight intervals at the released-size budgets. Across all seed-mean P/R/F1
cells, observed coverage is 0.996–1.000.

The perfect HI LGB → XGB point result occurs because all five held-out false
positive sets lie inside the constructor hard census. Transfer is incomplete
in the other directions. Full-population false positives outside the hard
stratum, in seed order 42/123/456/789/1011, are:

* HI LGB → XGB: 0/0/0/0/0 (hard population 407).
* HI XGB → LGB: 16/7/31/16/11 (hard population 309).
* LI LGB → XGB: 8/0/2/2/7 (hard population 288).
* LI XGB → LGB: 27/82/23/52/34 (hard population 235).

These counts are descriptive diagnostics computed after designs were fixed,
not inputs to allocation or tuning. Incomplete transfer leaves rare missed
false positives with large HT weights and can produce a skewed error
distribution. Access to full labels to census illicit rows remains an
evaluation-sampling assumption.

## Artifacts and reproduction

Run from the `amlcompact` package directory:

```bash
python scripts/26_evaluate_heldout_sampling.py \
  --inputs /path/to/amlcompact-dataset/ml_baselines
python -m pytest tests/test_heldout_sampling.py tests/test_sampling_uncertainty.py -q
```

The input default is the sibling `amlcompact-dataset/ml_baselines` directory.
No model load, training, inference, network access, or GPU is used. The exact
input arrays must retain the supplied row alignment with `test_labels.npy`.
The script can check shape and binary labels; it cannot independently prove
semantic row alignment from probability arrays alone.

* `protocol.json`: frozen choices, seed schedule, scope, and units.
* `input_manifest.json`: SHA-256 of all 42 input files, the protocol, and the
  executed source files, plus runtime versions. Input paths are relative to
  `--inputs`, and source paths declare their package or output base.
* `summary.csv`: 108 rows (36 configurations × three metrics), summarizing
  errors of the mean of five fixed-seed metrics over 500 draws.
* `per_seed_summary.csv`: the corresponding 540 per-seed diagnostic rows.
* `full_population_metrics.csv`: exact population metrics and TP/FP for each
  fixed held-out seed and its five-seed mean. The mean threshold is descriptive;
  it is never used to threshold the held-out predictions.
* `designs.csv`: every population/sample stratum size, inclusion probability,
  weight, constructor threshold, and component confidence level.
* `transfer_diagnostics.csv`: full false positive counts inside/outside the
  constructor hard stratum; these do not enter the sampling design.
* `seed_mean_draws.csv`: all 18,000 seed-mean point estimates and envelopes.
* `draws_D_C_B_S.npz`: compact raw per-seed `point`, `lower`, `upper` arrays
  of shape `(500, 5, 3)`, with metric order precision/recall/F1; `truth` has
  shape `(5, 3)`. Also contains `fp` `(500, 5)`, `full_fp` `(5,)`, and `seeds`.
  D is dataset index, C is direction index, B is benign multiplier, and S is
  sampler index, using the ordered lists in `protocol.json`.
* `run_metadata.json`: elapsed CPU-run wall time, 36 configurations/18,000
  draws, interval-cache statistics, and the protocol checksum.

Tests include exhaustive enumeration of 36 small-population draws, unbiased
HT counts and marginal inclusion, conservative mean-metric coverage, a direct
comparison with the existing uncertainty implementation, positive inclusion
when the hard population exceeds its reserve, census and tied-score designs,
separate construction/evaluation thresholds, and cached interval reuse.
