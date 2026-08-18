# Reproducing the paper

Three tiers, by what they need. Pick the lowest one that answers your question.

## Tier 1: score a model on the benchmark

Needs this repository and network access. No AMLworld, no GPU, no archive.

```bash
pip install -e ".[hub]"
python - <<'PY'
from amlc import hub
from amlc.triage.doubt_triage import ht_weighted_prf

d = hub.load_coreset("HI-Small")          # arrays, about 200 KB
t = hub.load_table("HI-Small")            # adds the prompt column, 20 MB
preds = your_model(t["typed_graph_text"])
print(ht_weighted_prf(preds, d["labels"], d["weights"]))
PY
```

Weighted metrics on the coreset estimate the full-split metrics. For the supervised ensemble
the reproduction is exact; for a predictor that flags far more edges than the ensemble it is
design-consistent with variance, which is the regime the paper's LLM results live in.

`make test` exercises this path, and needs `pip install -e ".[dev]"` rather than the `[hub]`
above: the suite walks every module in the package, and three of them import matplotlib,
networkx or tqdm at module level. `make install-dev` does it.

## Tier 2: reproduce a table from the released results

Needs this repository only. Everything under `results/` is the source of a published number,
and `docs/` explains which. The audit stages rerun from the shipped annotation CSVs:

```bash
make iaa            # Fleiss and pairwise kappa across the four judges
make human-rating   # the three-rater validation tables
make triage         # Doubt Triage, if the archive is available
```

## Tier 3: rebuild from AMLworld

Needs a Kaggle account (AMLworld is CDLA-Sharing-1.0, Elliptic is CC BY-NC-ND 4.0;
neither is redistributed here), roughly 2 TB of scratch, a GPU for GCPAL, and a vLLM-capable cluster
for the LLM stages. Budget several days.

Install the extras the stages need first: `pip install -e ".[data,ml,gnn,llm,figures]"`.

```bash
export AMLC_ARCHIVE=/path/to/outputs
make download gfp train-ml train-gcpal infer-gcpal ensemble
make coreset ablation serialize icl
# then one LLM at a time, see scripts/slurm/host_vllm/
make triage deferral rubric
```

## What will not reproduce bit-for-bit, and why

**The k-hop neighbourhood sample.** `extract_k_hop_subgraph` caps neighbours per hop with
`random.sample` under the process-global random state, and the multiprocessing workers are not
seeded individually. Two runs on the same input can select different neighbours for
high-degree nodes. The released serialisations are the ones that were evaluated; regenerating
them produces an equivalent but not identical set.

**LLM generations.** Every model was run in thinking mode at its provider-recommended sampling
parameters, none of them greedy. Verdicts are stable enough that per-seed detection F1 varies
under 0.5 pp on 26 of 28 cells, but individual traces will differ.

**Anything downstream of GFP.** The preprocessor streams edges in temporal order with
`batch_size=128`. A different batch size changes the features, and `batch_size <= 0` fits on
the whole edge set at once, which destroys temporal causality and silently inflates every
supervised number.

## What will reproduce exactly

- The coreset construction, given the same ensemble probabilities. The stratification is
  deterministic and the weights are a closed form over realised membership.
- Every Horvitz-Thompson weighted metric, given the same predictions.
- The prompt templates. `make verify` renders all of them and compares byte-for-byte with the
  prompts that were sent.
- The frontier-probe templates under `prompts/frontier_probe/`. `make verify-probe` renders the
  six reported variants against the nine pilot cases and compares byte-for-byte with the prompts
  the APIs received. It needs `AMLC_PILOT_DIR` pointing at the archived pilot tree, because those
  prompts embed AMLworld transactions and are therefore not redistributed here; without it the
  check prints why it skipped and exits clean. Verified 2026-08-18: `user 9/9  system exact` for
  all six.
- The submitted Doubt Triage numbers, via `--draw ablation-redraw --no-verify
  --archived-weights`, which is how the draw defect was confirmed.

## Provenance caveats

Read `docs/known_issues.md` before quoting a number. Two entries matter for anyone rerunning:
the operating thresholds were computed on the full test split rather than validation, and the
AMLworld coreset was drawn with `seed_rng=0` rather than the seed 42 the paper names.

## Per-row tolerance, and the path caveats

###

Per `docs/reproducibility.md`, three things reproduce exactly and three do not. Exact: the coreset construction given the same ensemble probabilities, because the stratification is deterministic and the weights are a closed form over realised membership; every Horvitz-Thompson weighted metric given the same predictions; and the prompt templates, byte-for-byte under `make verify`. That makes the repo-tier rows (Tables 22 to 27 and the three Appendix G.2 rows) reruns rather than re-measurements, and each was regenerated here from the shipped CSVs and diffed clean against the published copy. The submitted Doubt Triage numbers reproduce exactly only via `--draw ablation-redraw --no-verify --archived-weights`, which is how the draw defect was confirmed. Variation, in three places: the k-hop neighbourhood sample caps neighbours with `random.sample` under the process-global random state and the multiprocessing workers are not seeded individually, so rebuilding the serialisations gives an equivalent but not identical set and the released ones are the ones that were evaluated; LLM generations ran in thinking mode at provider-recommended sampling parameters, none greedy, and the doc quantifies the resulting spread as per-seed detection F1 under 0.5 pp on 26 of 28 cells, with individual traces differing every run, which is the tolerance that applies to Tables 2, 8, 13, 14, 15, 18, 31 and everything downstream of them; and anything downstream of GFP depends on `batch_size=128`, since a different batch size changes the features and `batch_size <= 0` destroys temporal causality and silently inflates every supervised number, which is the tolerance on Tables 10 and 11 and on the ensemble probabilities the coreset is built from. No tolerance is stated for the Elliptic row beyond the bootstrap `--n-boot 2000` seed behaviour, so treat Table 12 as exact given the same upstream artefacts.

**Path and provenance caveats.** `make triage` and `make triage-stats` write into `$AMLC_ARCHIVE/triage` and `results/triage` respectively, not the published paths, which is why those two rows are typed as scripts with `--out`. `scripts/elliptic/eval_ht_coreset.py` writes to its `--core-dir`, by default `outputs/elliptic/coreset/`; the released copy sits under `results/elliptic/`. The current deferral code writes one unsuffixed `score_deferral_summary.csv`, so the two shipped per-dataset summaries were produced by per-dataset runs and renamed. Baselines B6 and B7 of Tables 1 and 11 register only when AMLworld edge data is present: for those two rows type `python scripts/08_run_ablation.py --datasets HI-Small LI-Small --amlworld /path/to/amlworld`. Four printed cells disagree with the shipped files: HI B4 sigma (3.3 in the file, 3.4 printed), LightGBM HI precision (78.0 against 77.9), five Table 15 cells off by 0.1 pp from subtracting before rather than after rounding, and, more seriously, the whole shipped `error_transition_deep.csv` is a different draw from the submitted one (HI-Small GPT-OSS-120B `both_correct` reads 988 / 655 / 0.688 against the printed 820 / 732 / 0.87), with `AMLC_LEGACY_DRAW=1` documented as the switch that reloads the submitted draw.

### Objects with no command

- **Table 5** (`tab:subgraph_params`): not traced, it is a literature and configuration table, three of its five cells read off cited papers and the other two are pipeline constants; no results file holds it and no script writes one.
- **Table 6** (`tab:models`): not traced, it is a model and hardware inventory; names come from `amlc/config.yaml`, context lengths from the `--max-model-len` flags in `scripts/slurm/host_vllm/*.sh`, and Arch/Active only from free-text header comments there, one of which contradicts the printed value.
- **Table 7** (`tab:sampling_params`): not traced as an output, every value is a literal in `amlc/config.yaml` that `tests/test_config.py` pins and `make test` checks, so it is verified rather than produced and has no path under `results/`.
- **Table 10, GCPAL+GFP typology cells**: partially traced, the detection triple recomputes from `results/coreset/ht_subset_models.csv` but the typology macro-F1 goes to `results/ml/gcpal+gfp_summary.json`, which was not shipped.
- **Table 3, TF1 column of the three non-DT rows**: not traced, the deferral grid computes no typology at all and `dt_results.csv` carries only `dt_typ_f1`, so the caption's shared typology fusion has no shipped implementation.
- **Table 20** (`tab:failure_taxonomy`): not traced, only the Fmt and Over columns reproduce from shipped files; the taxonomy stage writes `results/audit/intervention_results.csv`, which is not in the release, and Cor, Und and Typo do not recompute because the shipped comparison CSV has already absorbed the unparseable share into the other classes.
- **Table 21** (`tab:anchor_stats`): not traced to a command, `results/analysis/anchor_chi_square.csv` holds the table exactly and recomputes from `results/analysis/per_run.csv`, but neither file has a producer anywhere in the checkout or in git history.
- **Table 28** (`tab:trace_length`): partially traced, the Trace Rate column is `valid_ratio` in `results/summary_all.csv` on all fourteen cells, but the mean and median character columns have no producer and appear in no result file.
- **Table 29** (`tab:trace_correctness`): not traced, nothing in the package joins trace length against prediction correctness, and reproducing it needs the archived seed-42 prediction JSONs plus a script that was not released.
- **Table 30** (`tab:self_correction`): not traced, the self-correction marker regex bank exists nowhere in the checkout and no file under `results/` holds the 28 percentages.
- **Algorithm 2** (`alg:dt`): not traced, it is pseudocode rather than a measurement; its implementation is `amlc/triage/doubt_triage.py`, whose three branches match it in order.
- **Appendix G.3, OR-rule 50.4% and AND-rule 65.6%**: not traced, no script mentions the one naive-hybrid artefact shipped and that file contains neither number; only the score-only gate half of G.3 traces.
- **Figure 1** (`fig:reduction_sweep`): not traced to a command, the plotted values are in `results/coreset/ht_coreset_<dataset>_sizes.csv` and `baselines_agg.csv`, but no code in the checkout draws the PDF.
- **Figure 2** (`fig:cost_reduction`): not traced, the token ratios are in `ht_coreset_<dataset>_sizes.csv` and the Naive arm in `naive_vs_ht_summary.json`, but no command produces the figure.
- **Figure 3** (`fig:anchor_heatmap`): not traced, its shares recompute from the `gt=benign` rows of `results/analysis/per_run.csv`, which itself has no producer, and no plotting code exists.
- **Figure 4** (`fig:step_rubric`): not traced, the funnel values are the `A_pass_pct` entries of `results/audit/rubric_n1000_summary.json` that `make iaa` regenerates, but nothing in the package renders the PDF.
- **Figure 5** (`fig:ablation_prf`): not traced, the three bars per method are the `delta_p_mean`, `delta_r_mean` and `delta_f1_mean` columns of `baselines_agg.csv`; the plot has no producer.
- **Figure 6** (`fig:illicit_retention`): not traced, the plotted quantity is `n_illicit_mean` in `baselines_agg.csv` (per draw in `baselines_draws.csv`); the plot has no producer.
- **Figure 7** (`fig:ablation_variance`): not traced, the boxplotted distribution is the `w_f1` column of `baselines_draws.csv`; the plot has no producer.
- **Figure 10** (`fig:dt_models`): not traced, the model-agnostic spread is in `results/triage/dt_summary.csv`, but no notebook or script in the tree draws `dt_model_agnostic.pdf`.
- **Figure 11** (`fig:step_heatmap`): not traced, the seven-by-four grid is the `per_model_A` block of `results/audit/rubric_n1000_summary.json` that `make iaa` regenerates, but the stage writes no image.

Of the 26 traced rows, 8 run from this checkout alone, 11 additionally need `AMLC_ARCHIVE`, and 7 need the full pipeline with AMLworld or Elliptic downloaded and a GPU or vLLM cluster.
