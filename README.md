# AMLCompact

[![CI](https://github.com/nat-nischw/AMLCompact/actions/workflows/ci.yml/badge.svg)](https://github.com/nat-nischw/AMLCompact/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-amlcompact--eval-yellow)](https://huggingface.co/datasets/natnitaract/amlcompact-eval)
[![Code MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Data CDLA-Sharing-1.0](https://img.shields.io/badge/data-CDLA--Sharing--1.0-orange)](https://cdla.dev/sharing-1-0/)

[Score a model](#score-a-model) ·
[Baselines](#baselines) ·
[Reproduce the paper](#reproduce-the-paper) ·
[Extend](#implementing-new-prompts-and-routers) ·
[Uses](#uses) ·
[Cite](#citation)

An anti-money-laundering benchmark that a language model can actually be run on. AMLworld's
test split is a million transaction edges and serialises to roughly 1B tokens per pass; this
is the same split compressed 271x and 611x, with importance weights that put the metrics back.

Code for *AMLCompact: Bias-Free Downsampling Makes AMLworld Tractable for LLM Evaluation*,
an ARR submission committed to EMNLP 2026 (Resources and Evaluation). The venue state and the
citation live in [`docs/citation.md`](docs/citation.md) and this page is updated when the
decision lands.

**Three names, one project.** *AMLCompact* is the paper and this repository, `amlc` is the
package you import, `amlcompact-eval` is the dataset on Hugging Face. Only `amlc/` is
importable code.

## Install

```bash
pip install -e ".[hub]"
```

## Score a model

The dataset downloads on first use and caches. You need neither AMLworld nor the archived run.

<!--quick-start-begin-->
```python
from amlc import hub
from amlc.triage.doubt_triage import ht_weighted_prf

d = hub.load_coreset("HI-Small")                 # arrays only, 0.9 MB
P, R, F1 = ht_weighted_prf(your_predictions, d["labels"], d["weights"])

table = hub.load_table("HI-Small")               # adds the prompt text, 20 MB
```
<!--quick-start-end-->

Each row of `table` carries `typed_graph_text`, the prompt as the evaluated models received it,
and asks for two things: is this edge illicit, and which of eight laundering typologies is it.
The [dataset card](https://huggingface.co/datasets/natnitaract/amlcompact-eval) is the column
dictionary.

**Pass the HT weights.** Unweighted numbers on a coreset describe the coreset and not the
population, and the whole point of the construction is that the weighted ones describe the
population exactly. `hub.load_coreset` asserts that the weights sum to the full test-split size
before handing them back, because a weight vector that does not is silently wrong rather than
loudly broken.

```bash
make selftest        # scores the shipped ensemble, compares against the published numbers
```

Ten seconds, and it exercises the download, the weight-sum assertion and the scorer in one go.

The evaluation set is **not vendored into this repository**. One published copy stays
authoritative; a copy committed here would drift from it, and the copy that drifts is always
the one nobody rebuilt. Point `AMLC_DATASET` at a fork, or `AMLC_CORESET_DIR` at a local build.

## Baselines

Every number is HT-weighted, so it describes the full million-edge split.

| | HI-Small | LI-Small |
|---|---:|---:|
| Supervised ensemble, detection F1 | 70.9 | 29.6 |
| Seven open-weight LLMs, detection F1 | 40.7 to 50.7 | 40.6 to 50.7 |
| Their precision | 31.5 to 34.2 | 28.3 to 34.0 |
| Their recall | 55.2 to 100.0 | 63.5 to 100.0 |
| Their typology macro-F1 | 5.6 to 18.4 | 2.5 to 13.0 |
| Doubt Triage, an ML and LLM hybrid | 82.1 to 87.4 | 71.8 to 84.9 |
| The same idea, implementable | 66.0 | 31.9 |

**LLMs recall almost everything and decide almost nothing.** Precision sits near a third, which
under the 1:2 design ratio is what predict-all-illicit earns. Restore the native prevalence by
weighting and all fourteen model-and-prompting cells land on that trivial baseline. Scale does
not help: typology macro-F1 tops out at 18.4 against the ensemble's 72.3.

**The failure is at the last step.** Of 1,000 audited reasoning traces, 538 parse the subgraph,
recall the typology inventory and match evidence against it, then emit the wrong verdict.

**The asymmetry is exploitable, but not by a bank.** Routing only the supervised model's
uncertain cases to an LLM reaches 87.4 and 84.9. That router reads the coreset stratum, a
property of how the evaluation set was drawn rather than of the edge, so it is an upper bound.
Gating on the supervised score alone, which a deployment could compute, reaches 66.0 and 31.9.
The gap between those two rows is the open problem.

Sources: `results/summary_all.csv`, `results/triage/dt_summary.csv`,
`results/triage/score_deferral_grid_*.csv`.

## Coreset statistics and weighting

| | HI-Small | LI-Small |
|---|---:|---:|
| Full temporal test split | 1,015,669 edges | 1,384,810 edges |
| Illicit in it | 1,251 (0.1232%) | 756 (0.0546%) |
| Coreset | 3,753 | 2,268 |
| Ensemble P / R / F1, weighted on the coreset | 93.4555 / 57.0743 / 70.8685 | 73.2984 / 18.5185 / 29.5671 |
| The same, computed on the full split | identical | identical |

Identical is literal, and it is a narrower claim than "the weighting is unbiased". The paper
separates three things in §3.3, *Estimation properties*, and the separation matters:

- The weighted TP, FP and FN **counts** are unbiased for their full-split values, for any
  predictor. That is the Horvitz-Thompson guarantee and it assumes nothing.
- **Precision and F1** are ratios of two such counts, so they are consistent and asymptotically
  unbiased but not unbiased in finite samples.
- **Recall is exact** for any predictor, because every illicit edge is retained at weight one,
  so its denominator is observed rather than sampled.

The row above is stronger than all three and is not an instance of the ratio guarantee: the
ensemble flags no down-sampled benign edge at its operating threshold, so every edge entering P,
R and F1 carries weight one and the two computations are the same arithmetic on the same edges.
The residual is below 1e-15, which is double precision and not a tolerance.

That exactness belongs to the scored predictor, not to the coreset alone. A predictor that does
flag down-sampled edges pays about 470x per false positive on HI-Small and falls back to the
ratio regime with real sampling variance, which is the LLM case above and the reason those rows
collapse onto the trivial floor. The dataset card carries the same statement in full.

## Reproduce the paper

Every row was traced by opening the producing script and the result file. **Needs** is `repo`
for this checkout alone, `archive` for `AMLC_ARCHIVE` pointing at the run directory, `full` for
AMLworld and a cluster. Of 32 traced objects, 9 run from a clone, 13 need the archive, 10 need
the full pipeline.

| Paper object | Command | Output | Needs |
|---|---|---|---|
| Tab 1 | `make ablation` | `results/coreset/baselines_agg.csv` | full |
| Tab 9 | `make ablation` | `results/coreset/baselines_agg.csv` | full |
| Tab 11 | `make ablation` | `results/coreset/baselines_permodel.csv` | full |
| Tab 12 | `python scripts/elliptic/eval_ht_coreset.py --n-boot 2000` | `results/elliptic/eval_summary.csv` | full |
| Tab 4 | `make analysis` | `results/error_analysis/typology_recall.csv` | archive |
| 2 | `make llm-eval MODEL=GPT-OSS-120B VLLM=http://host:18809/v1 DATASETS=HI-Small` | `results/summary_all.csv` | full |
| 13 | `make llm-eval MODEL=GPT-OSS-120B VLLM=http://host:18809/v1 DATASETS=LI-Small` | `results/summary_all.csv` | full |
| 14 | `make llm-ht` | `results/analysis/llm_ht_weighted.csv` | archive |
| 15 | `make llm-eval MODEL=GPT-OSS-120B VLLM=http://host:18809/v1` | `results/summary_all.csv` | full |
| 8 | `make llm-eval MODEL=GPT-OSS-120B VLLM=http://host:18809/v1` | `results/summary_all.csv` | full |
| 10 | `make train-ml train-gcpal infer-gcpal ensemble` | `results/summary_all.csv` | full |
| 18 | `make llm-eval MODEL=GPT-OSS-120B VLLM=http://host:18809/v1` | `results/summary_all.csv` | full |
| 19 | `make analysis` | `results/error_analysis/typology_recall.csv` | archive |
| 3 | `AMLC_ARCHIVE=/path/to/outputs make triage` | `results/triage/dt_summary.csv` | archive |
| 3 | `AMLC_ARCHIVE=/path/to/outputs make deferral` | `results/triage/score_deferral_grid_HI-Small_Qwen3.5-27B_ICL-ZS.csv` | archive |
| app:significance | `make triage-stats` | `results/stats/dt_pooled_stats.json` | repo |
| app:naive_hybrids | `AMLC_ARCHIVE=/path/to/outputs make deferral` | `results/triage/score_deferral_summary_HI-Small.csv` | archive |
| Table 23 | `make iaa` | `results/audit/rubric_n1000_summary.json` | repo |
| Table 24 | `make iaa` | `results/audit/multi_judge_iaa_n1000.csv` | repo |
| Table 25 | `make human-rating` | `results/human_rating/human_rating_summary.csv` | repo |
| Table 26 | `make human-rating` | `results/human_rating/human_rating_summary.csv` | repo |
| Table 27 | `make human-rating` | `results/human_rating/human_rating_by_outcome.csv` | repo |
| Table 16 | `AMLC_ARCHIVE=/path/to/run/outputs make analysis` | `results/error_analysis/error_transition_deep.csv` | archive |
| Table 17 | `AMLC_ARCHIVE=/path/to/run/outputs make analysis` | `results/error_analysis/error_transition_deep.csv` | archive |
| Figure 9 | `AMLC_ARCHIVE=/path/to/run/outputs AMLC_FIGURE_OUT=results/figures make analysis` | `results/error_analysis/error_transitions.csv` | archive |
| Figure 8 | `AMLC_ARCHIVE=/path/to/run/outputs AMLC_FIGURE_OUT=results/figures make analysis` | `results/error_analysis/typology_f1.csv` | archive |
| Table 22 | `python scripts/21_score_frontier_probe.py --convention benign` | `results/frontier_probe/predictions.csv` | repo |
| Table 31 | `AMLC_ARCHIVE=/path/to/run/outputs python scripts/22_compare_intervention.py` | `results/summary_all.csv` | archive |

Twenty objects have no command, listed as such in
[`docs/reproducibility.md`](docs/reproducibility.md) along with the per-row tolerance and four
cases where a printed cell disagrees with the shipped file. Most are configuration tables with
nothing to regenerate. Nine are figures whose plotting scripts live in the paper workspace
rather than here; the data behind all of them ships, so any figure can be redrawn.

Three things do not reproduce bit-for-bit: the k-hop neighbourhood sample, LLM generations, and
anything downstream of the Graph Feature Preprocessor's `batch_size=128`. The third is the one
that can bite you, because it fails quietly and in the direction that looks like success.
[`docs/reproducibility.md`](docs/reproducibility.md) has the mechanism for each.

## Implementing new prompts and routers

The paper closes on two open problems. Neither needs the archive.

### A different prompt

The prompts are Jinja templates, not strings buried in a runner: `prompts/icl_zs.j2`,
`icl_fs.j2`, `icl_v.j2`, and the six frontier-probe variants under `prompts/frontier_probe/`.
`scripts/verify_prompt_templates.py` proves they render byte-for-byte into what was actually
sent, so a new variant starts from a known baseline rather than an approximation.

The four-step audit scores Parse, Recall, Match and Conclude separately, so a prompt change
reports which step it moved:

```bash
python scripts/13_run_llm_eval.py --model <yours> --vllm-url <url>
python scripts/16_sample_traces.py                 # draw 1,000 traces, regex annotator
python scripts/19a_run_judges.py deepseek          # add an LLM judge
python scripts/19b_analyze_rubric.py               # per-step pass rates and the fail mode
```

`scripts/21_score_frontier_probe.py` reproduces all ninety of those numbers from
`results/frontier_probe/predictions.csv` with nothing but this checkout.

### A different router

Doubt Triage is one rule for deciding which edges to send to the LLM, and
`amlc.triage.doubt_triage.STRATEGIES` ships four of them (`dt`, `confidence`, `disagree`,
`union`). A new one needs the supervised scores, the labels and the weights, all of which are
one call away:

```python
probs = hub.load_test_probs("HI-Small", "LightGBM+GFP", seed=42)
d     = hub.load_coreset("HI-Small")
# route however you like, then score the fused predictions
P, R, F1 = ht_weighted_prf(fused, d["labels"], d["weights"])
```

Two reference points make the result interpretable. `scripts/17b_dt_stats.py` gives the paired
comparison against DT over all 140 cells rather than a single number.
`scripts/18_score_deferral_grid.py` gives the floor that matters: a grid-searched gate on the ML
score alone, which is what a deployed system can actually compute, since DT routes on the
coreset stratum and a bank has no stratum. That gate recovers 2.3 of DT's 55.3 pp on LI-Small.
The gap between those two numbers is the open problem.

## Repository and Hub contents

| | this repository | the dataset on Hugging Face |
|---|---|---|
| pipeline code, prompts, SLURM jobs | yes | |
| tuned hyperparameters, demonstration pool | yes | |
| every result table and figure | yes | |
| the evaluation set and its prompts | | yes |
| HT weights, labels, typologies, GFP tensors | | yes |
| supervised model weights and probabilities | | yes |

### Which file supports which claim

| Path | The claim it supports |
|---|---|
| `amlc/coreset/` | the coreset construction and the seven-baseline ablation |
| `amlc/triage/` | Doubt Triage, its paired statistics, the score-only deferral floor |
| `amlc/audit/` | the four-step rubric, the four judges, inter-annotator agreement |
| `amlc/llm/`, `prompts/` | the LLM evaluation and the prompts that were actually sent |
| `amlc/baselines/` | the three supervised members and the soft-vote ensemble |
| `results/` | every CSV and JSON behind a published number |
| `docs/` | [known issues](docs/known_issues.md), [reproduction tiers](docs/reproducibility.md), [another split](docs/another_split.md), [citation](docs/citation.md), [full layout](docs/repo_layout.md) |

`make help` lists every stage in pipeline order. `amlc/config.yaml` holds every constant the
paper reports, annotated, in one file.

## Uses

### Direct Use

Scoring a detector on a transaction-graph AML benchmark whose weighted
metrics stand in for a million-edge split; measuring where a prompt change acts, using the
four-step rubric rather than a single F1 number; and building selective ML-and-LLM routers
against a fixed evaluation design. It is a research benchmark and a measurement instrument.

### Out-of-Scope Use

**Not for deployment, and the paper's own numbers are the argument.** Restoring the native
prevalence by Horvitz-Thompson weighting puts every one of the fourteen evaluated LLM cells on
the trivial predict-all baseline: F1 between 0.23 and 0.25 on HI-Small against a floor of 0.25,
and between 0.09 and 0.11 on LI-Small against 0.11. The hybrid that does lift F1 routes on the
coreset stratum, which is a property of how the evaluation set was drawn and not of the edge, so
a bank cannot compute it; the implementable version, a grid-searched gate on the supervised score
alone, recovers 2.3 of Doubt Triage's 55.3 pp on LI-Small and falls below the ensemble on
HI-Small. Nothing here is evidence that an LLM should decide whether a transaction is
suspicious.

**Also out of scope.** Estimating what precision a detector would reach in production at the
native rate, which needs a deployment study and not a coreset. Any use that treats a flagged
edge as an accusation about a person or an institution. Training on the evaluation split, which
is what the released supervised probabilities are for.

**Personal and sensitive information.** AMLworld is agent-generated: accounts, amounts and
counterparties are synthetic and correspond to no real person or institution, which is why IBM
publishes it openly. That provenance is the basis for the claim. What we verify mechanically is
integrity rather than content: `scripts/00_download_amlworld.py` checks pinned SHA256 digests
before anything is derived. We did not run a content sweep over the 6,021 serialised prompts,
because they are built from that synthetic source by a deterministic serialiser.

## Licences

Named per asset class, because they are not the same and the difference has consequences.

| Asset | Licence | Consequence |
|---|---|---|
| This code | MIT | do what you like |
| Anything derived from AMLworld: coreset indices, HT weights, labels, typologies, GFP tensors, serialised prompts | **CDLA-Sharing-1.0** | copyleft. Publish a derivative and you are bound to the same terms |
| Supervised model weights and the aggregate metrics in `results/` | MIT | the agreement's carve-out for computational results, §3.5 |
| Elliptic | CC BY-NC-ND 4.0 | NoDerivatives, so we ship the pipeline and no derived data |

[`NOTICE.md`](NOTICE.md) carries the reasoning, the provenance and the prior work reused.

## Known issues

[`docs/known_issues.md`](docs/known_issues.md) records what was corrected while preparing this
release and which published numbers moved. The superseded coreset draw behind the
ARR-submission numbers is still runnable with `AMLC_LEGACY_DRAW=1`, so the difference can be
seen rather than guessed at.

## Maintenance and contact

**Frozen at camera-ready.** This repository records what the paper reports. Dependency rot and
correctness fixes land on a `post-camera-ready` branch; the tagged commit the paper cites does
not move. Correctness reports are welcome and feature requests are not, because a benchmark that
changes under the people comparing against it is not a benchmark.

If a number here disagrees with the paper, that is a bug and we want to know: open an issue with
the command you ran and the file you read. If a case looks mislabelled, raise it on the dataset's
Community tab quoting its `amlc_NNNNN` identifier.

Point of contact: `<NAME AND ADDRESS>`.

This is a research artefact and not a product. It is not audited, not supported, and not
suitable for use in a compliance decision.

## Citation

```bibtex
@misc{nitarach2026amlcompact,
  title  = {AMLCompact: Bias-Free Downsampling Makes AMLworld Tractable for LLM Evaluation},
  author = {Nitarach, Natapong and Ngampornsukswadi, Phume and
            Taveekitworachai, Pittawat and Nonesung, Surapon and
            Sirichotedumrong, Warit and Halverson, Duncan and
            Pipatanakul, Kunat},
  year   = {2026},
  url    = {https://openreview.net/forum?id=VouFZFf8Ph}
}
```

Cite AMLworld as well; nothing here exists without it, and its licence asks that the attribution
be preserved. The entry, the three venue states and what changes at each are in
[`docs/citation.md`](docs/citation.md), which `make check-citation` holds the other four copies
against.
