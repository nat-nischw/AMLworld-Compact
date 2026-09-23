# Frontier-API context-engineering probe

Backs the eighteen-cell probe table in the paper's appendix: three proprietary
reasoning APIs (DeepSeek-V4-Pro, Gemini 3.1 Pro, Claude Sonnet 4.6) across six
prompt designs on a 198-case pool. Score it with

```bash
python scripts/21_score_frontier_probe.py
```

which prints the table and checks every published number against
`predictions.csv`.

## What is here

| path | what it is |
|---|---|
| `predictions.csv` | 4,356 rows, one per model x variant x case. Gold class and label, the parsed typology, the parsed binary verdict, token counts, whether the call errored |
| `metrics_archived/<model>/<variant>/metrics.json` | the twenty-two aggregates written at run time by the original analysis script |

## What is not here

The archived responses carry the model's chain of thought, about seven
thousand characters per case, returned by proprietary APIs. They are not
redistributed; neither is the visible response text. `predictions.csv` is the
parsed result with both dropped, and every published number is recomputable
from it.

Anyone holding the raw responses can regenerate the CSV and check the
extraction:

```bash
AMLC_PROBE_RESPONSES=/path/to/responses python scripts/21_score_frontier_probe.py --extract
```

## The probe pool

The 198 cases are 22 per class across the nine classes, so 176 illicit against
22 benign. All 198 identifiers resolve inside the released HI-Small coreset and
their labels agree with it on all 198, so the probe is checkable against the
published dataset rather than against a private pool.

## Scoring conventions

The predicted class is `parsed.observed_pattern`, with `legitimate` meaning the
negative class. The model's own binary answer, `parsed.conclusion`, is not part
of any published column: the probe's detection metric measures whether the
model named a laundering typology. For `FS-TypFirst` that is the stated design,
since its prompt derives the verdict from the typology.

Unparsed responses are scored per column. The Det and Typ-F1 columns treat a
parse failure as a confident benign prediction; the Typ-Acc column treats it as
a non-answer that matches nothing. The scorer's `--convention` flag applies
either rule to the whole table.

## Claude Opus 4.7

A fourth model was attempted on four variants and returned an error on all 198
cases of all four. It carries no result and the paper reports three models. It
stays in `predictions.csv` so the failure is visible rather than silently
absent, and `score()` excludes it.

## metrics_archived

`metrics_archived/` records what the original analysis script computed at run
time. Its `detection_accuracy` uses the model's own verdict, so it differs from
the published detection columns, which follow the conventions above. It ships
as a record of what was computed at the time; the scorer is the source for the
table.
