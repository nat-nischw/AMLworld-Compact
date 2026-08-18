# Frontier-API context-engineering probe

Backs the eighteen-cell probe table in the appendix: three proprietary reasoning APIs
(DeepSeek-V4-Pro, Gemini 3.1 Pro, Claude Sonnet 4.6) across six prompt designs on a 198-case
pool. Score it with

```bash
python scripts/21_score_frontier_probe.py
```

which prints the table and asserts that all ninety published numbers come back.

## What is here

| path | what it is |
|---|---|
| `predictions.csv` | 4,356 rows, one per model x variant x case. Gold class and label, the parsed typology, the parsed binary verdict, token counts, whether the call errored |
| `metrics_archived/<model>/<variant>/metrics.json` | the twenty-two aggregates written at run time by the original analysis script |

## What is not here, and why

The archived responses carry the model's chain of thought, about seven thousand characters per
case, returned by proprietary APIs. That is the most terms-sensitive category in this release and
it is not redistributed; neither is the visible response text. `predictions.csv` is the parsed
result with both dropped, and every published number is recomputable from it, which is the point
of shipping it in that form.

Anyone holding the raw responses can regenerate the CSV and check the extraction:

```bash
AMLC_PROBE_RESPONSES=/path/to/responses python scripts/21_score_frontier_probe.py --extract
```

## The probe pool is a slice of the released coreset

The 198 cases are 22 per class across the nine classes, so 176 illicit against 22 benign. All 198
identifiers resolve inside the released HI-Small coreset and their labels agree with it on all
198, so the probe can be checked against the published dataset rather than against a private
pool. Identifiers were retagged from `v2_` to `amlc_` along with the rest of the release.

## Two things the table does that the appendix does not say

No generator for this table survives anywhere, so the scoring convention was recovered by fitting
candidate rules to the published numbers. Both findings below are camera-ready items and both are
checkable by running the scorer with a different `--convention`.

**The model's own binary answer is not what the detection columns measure.** A case counts as
predicted-suspicious when the model named a laundering typology, not when it wrote the word
suspicious. Scoring detection from the model's own verdict instead reproduces 31 of the 54
detection numbers and none of DeepSeek's six cells. For `FS-TypFirst` that is the stated design,
since its prompt derives the verdict from the typology. For the other five variants it is not
what the prompt asked for.

**The columns disagree about what an unparsed response means.** 411 of the 3,564 scored responses
parsed no typology at all. Det and Typ-F1 were computed treating those as confident benign
predictions; Typ-Acc in the same row was computed treating them as non-answers that match
nothing. Each rule is defensible on its own, but a reader compares Typ-F1 against Typ-Acc across
the row, and those two numbers were not produced the same way. The gap is widest for
`FS-CoT-Elim` on Claude Sonnet, where 95 of 198 responses failed to parse: the table prints
Typ-Acc 15.2, and the convention its own Typ-F1 column uses gives 21.2.

Adopting one rule for the whole table moves either seven numbers or twelve:

| rule for the whole table | numbers that change | effect |
|---|---|---|
| unparsed counts as benign | 7 of 90, all Typ-Acc, by up to +6.1 | Det and Typ-F1 already use it, so nothing in the body prose moves |
| unparsed counts as no answer | 12 of 90, all Typ-F1, between -1.5 and +0.6 | the best-cell Typ-F1 quoted in the discussion becomes 31.0 rather than 30.3 |

Neither reverses any comparison in the table: `FS-TypFirst` remains the best variant on every
model under both.

## Claude Opus 4.7 is in the CSV and not in the table

A fourth model was attempted on four variants and returned an error on all 198 cases of all four.
It carries no result and the paper reports three models. It stays in `predictions.csv` so the
failure is visible rather than silently absent, and `score()` excludes it.

## The archived aggregates disagree with the table, on purpose

`metrics_archived/` records what `analyze_vllm_pilot.py` computed at run time. Its
`detection_accuracy` uses the model's own verdict, so it does not match the published detection
columns. It ships as a record of what was computed then, not as a source for the table.
