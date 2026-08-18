# What is where

The README keeps a short claim-keyed index. This is the full tree, for when you
need to find something specific rather than reproduce a number.

## The package

Everything importable is under `amlc/`. `import amlc`, nothing else.

```
amlc/
  config.py           loads config.yaml and re-exports every value
  config.yaml         every constant in one annotated file. Start here
  paths.py            every filesystem location; repo and archive kept apart
  archive.py          the only module allowed to name the original run directory
  hub.py              pulls the released dataset from the Hub and caches it
  selftest.py         scores the shipped ensemble against the published numbers
  case_ids.py         canonical `amlc_NNNNN` identifiers, with a verifier
  typology.py         the eight-class encoding, with a verifier
  coreset/            the HT-Coreset sampler, weight recomputation, the ablation
  triage/             Doubt Triage, its statistics, the score-only deferral grid
  llm/                clients, prompting methods, the evaluation runner
  serialize/          typed-graph serialisation and the demonstration pool
  baselines/          LightGBM, XGBoost, GCPAL, the soft-vote ensemble
  data/               AMLworld download and loading, the temporal split, GFP
  audit/              the four-step rubric, the judges, inter-annotator agreement
  analysis/           error analysis, and the HT-weighted LLM scoring of stage 23
  figures/            the plot style the paper's figures use
```

Two of these carry a rule rather than only code.

`archive.py` is the single place that knows the pre-release names of the
original 18 GB run directory. No other module may use one, and the rule is
enforced by a grep gate that must return nothing. It is why a reader of this
repository never meets a `v2_best` or an `LLM+ICL-AML`, and why the paper's
vocabulary is the only vocabulary here.

`config.yaml` separates two kinds of value. Split sizes, thresholds, seeds and
the per-model sampling parameters are facts about a run that already happened;
changing one does not reconfigure anything, it makes the code describe a run
that did not occur. Addresses are yours to change and each carries the name of
its environment override beside it.

## Everything else

```
scripts/              numbered entry points, one per pipeline stage
  slurm/              the same stages as cluster jobs, plus vLLM hosting
  elliptic/           the real-data check of Appendix D
prompts/              the executed prompts as Jinja templates
data/                 tuned hyperparameters and the demonstration pool
results/              every CSV and JSON behind a published number, plus figures/
  frontier_probe/     parsed predictions from the appendix probe, no model text
  human_rating/       the three-rater validation, raw per-rater ratings kept
docs/                 this file, known issues, reproduction tiers, citation
envs/                 conda environments for the cluster jobs
tests/                the smoke test a fresh clone must pass
```

`make help` lists every stage in pipeline order with its number, which is the
fastest way to find the command for a stage you can already name.

## The four numbered groups

The stage numbers in `scripts/` are the pipeline order, and they cluster.

| Stages | What they do | What they need |
|---|---|---|
| 00 to 06 | AMLworld, GFP features, the three supervised baselines, the ensemble | AMLworld, a GPU for GCPAL |
| 07 to 12 | the coreset, the ablation, serialisation, the demonstration pool | the above |
| 13 to 18 | the LLM evaluation, Doubt Triage, its statistics, the deferral grid | a vLLM server, or the archive to rescore |
| 19 to 23 | the rubric, the judges, agreement, error analysis, HT-weighted scoring | the archive, or the shipped annotations |

Most readers need none of them. Scoring your own model needs only the package
and the published dataset, and rebuilding a table from the released results
needs only this checkout.
