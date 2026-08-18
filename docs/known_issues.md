# Known issues

Everything here was found while preparing the release. Each entry says what is wrong, whether
a published number moves, and what a user should do about it. Nothing is hidden behind a
"future work" phrasing.

## Corrected in this release, and the published numbers move

### Doubt Triage read the wrong coreset draw

Two same-size draws exist. `ht-coreset` is the released one, the draw that was serialised into
prompts. `ablation-redraw` is a separate sample kept only to reproduce the submitted numbers.
They share 47.3% (HI-Small) and 49.4% (LI-Small) of their edges and match positionally on 3.0%
and 8.9%.

The pre-release Doubt Triage loaded the ablation re-draw while indexing LLM predictions by the
released draw's identifiers, so 25.6% and 32.4% of positions paired a verdict about one edge
with the label and ensemble probability of another. The submitted numbers reproduce exactly
under that pairing, which is how the defect was confirmed.

Corrected, the ICL-ZS panel's best cell moves from 87.1 to 87.4 on HI-Small and from 85.2 to
84.9 on LI-Small. Six of seven models move under 0.5 pp; Qwen3.5-27B moves 3.4 pp and 7.8 pp,
consistent with its 38.7% format-failure rate leaving the sparsest verdict array. Reproduce
either with `--draw ht-coreset` or `--draw ablation-redraw --no-verify --archived-weights`.

### The typology integer encoding was permuted in two consumers

The arrays on disk are encoded fan-out, fan-in, cycle, scatter-gather, gather-scatter, stack,
bipartite, random. The pre-release error analysis and Doubt Triage each hardcoded a different
permutation, which agrees with the ground-truth string column on 0.00% of typed rows.

It survived because the permutation was applied to both sides of every ML-versus-truth
comparison, and macro-F1 is invariant under a consistent relabelling: all four supervised rows
are bit-identical either way. It only bites where a correctly named LLM string meets a wrongly
decoded one. Table 5's typology column drops 2.9 to 14.7 pp.

The main LLM typology numbers are unaffected: those read a string column, not the integers.

### The shipped typology figure predated its own CSV

Separate from the permutation above, and found by regenerating the stage rather than reading it.
`results/figures/typology_f1_hi_small.pdf` showed LLM macro-F1 of 3.2 to 8.4, while
`results/error_analysis/typology_f1.csv` beside it says 6.6 to 15.8. The figure predates the
Unicode-aware typology re-extraction: GPT-OSS emits a non-breaking hyphen, U+2011, which the
original parser missed, so every typology it named was scored wrong. Patching that raised the
LLM numbers, the CSV was rebuilt and the figure was not.

The figure here is now regenerated and matches the CSV. Everything else the stage writes is
reproduced identically, including all of HI-Small, so this was the only stale artefact.

**The paper's copy is the stale one.** `paper/figures/typology_f1_hi_small.pdf` carries the same
pre-patch values, and its caption asserts that LLMs "score below 10%", which was true of those
values and is not true of the corrected ones. The caption also describes the figure as
"\fs zero-shot", and `\fs` expands to ICL-FS; the stage that produces it reads
`LLM+ICL-AML`, the few-shot condition. Both are camera-ready items.

### The Elliptic ablation

Three defects. B1 and B2 sampled from the benign pool only, so every draw held zero illicit
nodes and the reported bias was the full-set metric itself: that is the origin of the 80.63 pp
rows, and of a standard deviation of exactly 0.00 across 50 random draws. B5 computed
class-proportional allocation rather than Neyman, so it was the same estimator as B2 and,
under the shared random stream, drew byte-identical samples in 50 of 50 rounds. Degenerate
draws were scored silently as zero.

Corrected: B1 1.57, B2 1.62, B5 0.93. B3, B4, B6, B7 and the HT-Coreset row are unchanged.

### The four-judge IAA table predated its inputs

The shipped tables were computed 22 minutes before the annotation CSVs they summarise were
rewritten. Only the Conclude step moves: Fleiss 0.772 to 0.735, and the regex pass rate 20.4%
to 19.2%, which removes an appendix-versus-body inconsistency since the body already says 19%.
The 538-of-1,000 headline recomputes to exactly 538.

## Corrected, and no published number moves

### LI-Small Horvitz-Thompson weights double-counted the benign population

The generator assigned each stratum's weight when that stratum was drawn, then appended a
spare-fill block weighted by the whole leftover pool. By then the four strata already carried
the entire benign population, so the fill added almost all of it again: LI-Small summed to
2,767,353 against a population of 1,384,810, with one edge at 1,382,543 where the next largest
was 1,136. HI-Small needed no fill and was correct. The Elliptic coreset had the same defect,
summing to 1.805x its population.

No published number moves, for a structural reason: the inflated edge is easy-benign, the
ensemble never flags it, and Doubt Triage only ever flips inside the census stratum. What did
move are two families the camera-ready adds rather than corrects: the HT-weighted predict-all
baseline and HT-weighted LLM-only metrics, both exactly 2x wrong on LI-Small.

The repair recomputes `N_s / n_s` from realised membership; it reproduces the HI-Small vector
bit for bit, which is the regression test that the reconstruction matches the generator.

### The shipped LI-Small case index disagreed with the weights beside it

The first upload of the dataset carried an `extras/LI-Small/case_index.csv` whose `weight` column
summed to 2,767,353 against a population of 1,384,810, while `ht_weights.npy` and the parquet
`ht_weight` column beside it were both correct. The builder read the case index out of the
archive, where that column predates the spare-fill fix, rewrote the identifiers and copied
everything else through. Its guard checked the repaired weight vector, which passed, and never
looked at the duplicate.

No published number moves: nothing reads the CSV's weight column. It matters because the file is
the human-readable join table, so it is exactly what somebody would open first to check the
headline invariant, and it would have told them the invariant fails.

Fixed by giving each duplicated quantity one source. The builder now overwrites `subset_index`,
`label` and `weight` from the arrays it is about to ship, then asserts equality and the population
sum. `tests/smoke_test.py` checks the same three columns against the shipped `.npy` files.

### The Elliptic prompts contained the answer

`render_edge_list` printed the ground-truth label of every node including the focal one, so
each prompt read `*tx_136285 (t=35,ill)` with `*` marking the transaction under test. All
3,249 cases leaked; after masking every evaluated node, none do. No published number is
affected: the Elliptic LLM stage was never run and the paper makes no Elliptic LLM or Doubt
Triage claim. Note that the internal rebuttal notes cite an Elliptic Doubt Triage
generalisation result that has no backing run.

## Not fixed, disclosed

### GCPAL was fine-tuned on a random split

`gcpal.py` fine-tunes on a random 60/20/20 permutation, and `gcpal_infer.py` then scores the
temporal test split with those weights. 609,857 of 1,015,669 HI-Small and 830,310 of 1,384,810
temporal test edges were inside its fine-tuning set. GCPAL is one of three ensemble members,
so the reported 70.9 and 29.6 are optimistic to that extent. Retraining without
`--random-split` is the fix; it has not been done.

### Threshold and seed provenance disagree with the paper

The paper describes the 0.80 and 0.48 thresholds as tuned on validation. The code computes
them on the full test split. The paper says the artefact pins seed 42; the AMLworld
construction uses `seed_rng=0`, and only the Elliptic build uses 42. Both need either a text
correction or a rerun.

### The difficulty score does not peak where the text says

Difficulty is `1 - 2 |p - 0.5|`, which peaks at 0.5, while the operating thresholds are 0.80
and 0.48. The paper describes Stage 2 as peaking at the decision boundary.

### The appendix prompt inventory does not match what ran

Four discrepancies, detailed in `prompts/frontier_probe/README.md`: the `FS-Base` box
describes a variant with no results; a sixth executed variant is undocumented; the abstain box
describes `V4` while `V4p1` ran; and the claim that all six share the typed-graph serialiser is
false, since the pilot prompts use a condensed serialisation.

### The two datasets carry different, and restrictive, terms

AMLworld is CDLA-Sharing-1.0, which is copyleft: derived data must be published
under the same terms. Elliptic is CC BY-NC-ND 4.0, whose NoDerivatives clause forbids
publishing adapted material at all, which is why the Elliptic side of this repository is
code only and no Elliptic tensor, coreset or serialisation is committed. Both confirmed from
the Kaggle listings on 2026-08-10. See `NOTICE.md`.

### The frontier-probe table scores the typology answer, not the model's verdict

`results/frontier_probe/README.md` has the detail. Two points carry into the appendix text.

A probe case counts as predicted-suspicious when the model named a laundering typology, not when
it wrote the word suspicious; the model's own binary answer is in the archived responses and is
used by no published column. For `FS-TypFirst` that is the stated design. For the other five
variants the appendix does not say it.

The columns also disagree about an unparsed response. 411 of 3,564 scored responses parsed no
typology. Det and Typ-F1 treat those as benign predictions, Typ-Acc in the same row treats them
as non-answers. Adopting the first rule throughout moves seven Typ-Acc numbers by up to 6.1
points and leaves the body prose alone; adopting the second moves twelve Typ-F1 numbers and turns
the 30.3 quoted in the discussion into 31.0. Neither reverses any comparison in the table.

### The Makefile did not run

Five of its targets could not have worked as written, found by running each recipe's arguments
against the parser it calls rather than by reading them. `tune` and `train-gcpal` passed
`--datasets` to scripts that take a single `--dataset`, so both died in argparse before doing
anything; `judges` passed `--provider` where the provider is positional; `serialize` and `icl`
omitted `--loader`, which was a required argument with no shipped factory to give it. `figures`
ran stage 20 a second time rather than regenerating anything, and stage 20 wrote nine figures
into a directory it never created.

All fixed, and all 25 recipe lines now get past argument parsing. The `--loader` case needed
more than a flag: the serialiser injects the AMLworld graph so a clone that only reads the
released prompts never loads it, but no injectable factory shipped. `amlc.data.loader:graph_source`
is that factory and is now the default, so `make serialize` and `make icl` run with no argument.

`.gitignore` also did not cover AMLworld. `make download` writes 1.1 GB of CDLA-licensed
transaction CSVs straight into `data/`, and nothing stopped a later `git add -A` from committing
them. It does now, while tracking the SHA256 manifest the downloader verifies against, which was
missing from the release.

### The Elliptic pipeline shipped without five of its stages

`scripts/elliptic/` held three scripts and the two cluster jobs called eight. The three that
shipped had been renamed from the pre-release `v2` vocabulary without their callers being
updated, and five more were never copied at all: the downloader, the preprocessor, the ML
baselines, the deferral evaluation and the LLM runner. Appendix D is the paper's only real-data
generalisation result, and `NOTICE.md` asserted that this directory downloads and processes
Elliptic locally, so the licence position rested on code that was not there.

All five are now ported, renamed to the paper's vocabulary, and the two job scripts point at
them. `wasd_eval.py` became `doubt_triage.py` and `predict_llm_elliptic.py` became
`run_llm_eval.py`; the internal `v2_` variable names went with them. Every one of the eight
parses and answers `--help` with no archive present.

### Coverage limits

ICL-V is four files: two models, two datasets, seed 42. Reasoning traces are populated for
seed 42 only; the other four seeds carry the field empty. The rubric is 1,000 HI-Small traces
at seed 42, scored by four judges.

### The released prompts differ from the executed ones by one token

Case identifiers were renamed from `v2_NNNNN` to `amlc_NNNNN` everywhere, prompts and recorded
model output included, so the release speaks one vocabulary. The substitution is exactly that
token and reverses by prefix swap.

## Two gaps this release did not close

**Nine of the eleven paper figures have no producer here.** `make figures` renders the two that
stage 20 writes. The other nine are drawn by scripts that live in the paper workspace rather
than in this package: `plot_reduction_sweep.py`, `plot_dt_models.py`, `plot_step_funnel.py` and
`generate_figures.py`. The data behind every one of them ships, in the files the README's
reproduction table names, so a figure can be redrawn from the numbers. The `Makefile` help text
used to claim stage 20 wrote all of them; it now says two.

**The shipped error-transition files disagree with the printed tables, and the printed tables
disagree with each other.** For HI-Small against GPT-OSS-120B, both
`results/error_analysis/error_transition_deep.csv` and `error_transitions.csv` give the four
groups as 988 / 2,178 / 408 / 179 edges. Table 16 of the paper prints 820 / 2,163 / 503 / 267
and Table 17 prints 807 / 2,145 / 516 / 285 for what should be the same four groups. Neither
caption names a prompting condition or a seed, and neither shipped file carries a prompting
column, so the three cannot be reconciled from the artefacts alone.

`_assets/error_analysis_fix/` shows the shipped values are the corrected draw with the corrected
typology encoding, and that its `repro/` variant reproduces the *published CSVs* exactly, not
the printed tables. So the tables were not regenerated from these files at any point in the
chain we can see. This is recorded rather than fixed: correcting it changes two appendix tables
and a paragraph of Section 5.2, which is a decision for the authors and not a release task.
