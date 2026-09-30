# Offline audit scoring sensitivity

This is a descriptive reanalysis of the frozen 1,000 model/case audit rows: HI-Small, ICL-FS, seed 42, seven models. The sample was stratified by model and prediction outcome and underrepresents correct benign predictions. Rates describe this audit slice; they are not benchmark prevalence estimates, population-weighted error rates, or causal evidence about reasoning stages. No new model calls or semantic judgements were made.

Reproduce all reports using only the released annotation CSVs and compact features:

```sh
python scripts/27_audit_scoring_sensitivity.py
```

Optionally recompute the compact features from the original local archive:

```sh
python scripts/27_audit_scoring_sensitivity.py --archive-root /path/to/outputs
```

The archive must contain `<model>/LLM+ICL-AML/HI-Small/seed_42.json`. The script reads archived reasoning_content, maps v2_ case keys to amlc_ keys, and does not change trace text, final-answer parsing, annotations, or judge scores. No raw reasoning text is distributed here. SHA-256 hashes cover source archives, each full/prefix trace, annotation inputs, scoring definitions, source code, and features.

## Definitions and denominators

- `strict_archived`: frozen heuristic Conclude column, denominator all rows in the group. Untyped illicit references are failures under that archived convention.
- `binary`: extracted suspicious/benign verdict equals the binary label; denominator all group rows.
- `exclude_illicit_reference_absent`: correct benign verdict or correct typed illicit verdict; denominator benign plus typed illicit rows. Untyped illicit rows are excluded, not relabelled as correct.
- `typed_illicit_only`: suspicious verdict and exact canonical typology match; denominator typed illicit rows only, including binary misses. Typed correctness is undefined for benign or missing-reference rows; their false indicator is always masked by eligibility.
- `full` and `prefix8000` reuse the same original Parse, Recall, and Match functions. Prefix means the first 8,000 Python characters, matching the archived judge input limit. Final prediction fields are held fixed at both lengths. Empty traces fail all three lexical steps.
- `marker_*`, `marker_count`, and `marker_ge2` count the original six regex marker categories regardless of the verdict. These are lexical presence indicators; negated, hypothetical, or incorrect statements can match. They do not establish factual graph correctness or entailment.
- Original Match automatically passes a nonempty trace with a benign prediction. `benign_match_bypass` marks that branch, and `match_without_two_markers` counts passes with fewer than two marker categories. `PR_and_two_marker_categories` removes that vacuity.
- Original Recall counts matched surface strings, so synonyms can count separately. `canonical_recall` counts distinct canonical classes using the already defined `amlc.llm.methods.PATTERN_MAP` with case-insensitive substring matching and no new aliases. This diagnostic changes the lexicon as well as collapsing aliases (e.g. mule is present in PATTERN_MAP, cyclic is not). It is not a pure deduplication experiment or semantic Recall score. No judge or ground truth was used to choose the mapping. The final-answer parser remains frozen.
- A conjunction is the stated upstream lexical rule AND failure under the stated correctness definition. Its denominator is all eligible rows in that group, not only upstream passes or errors. `upstream_count` and `incorrect_count` give those alternate denominators explicitly. This co-occurrence is not an attribution of an error to Conclude or evidence of stage order.
- `judge_discordance.csv` is a 2-by-2 comparison of each archived API Conclude judgement with each fixed parser-based definition. `all_rows` retains the original convention: empty traces and recorded judge errors have all-failure labels, and stay in the denominator. `scored_only` excludes recorded error rows and gives their count. In these tables the 148 error rows for each judge all represent empty traces, not failed API calls. The archived prompts exposed missing illicit typology as the string nan; the sensitivity analysis changes eligibility masks only and cannot remove that influence from judge labels. Judge PRM/Conclude conjunctions are reported under each judge's own unchanged labels. Disagreement is not a corrected semantic accuracy estimate.
- Every rate CSV row includes its integer numerator and denominator. A zero denominator has an empty rate. `ALL` rows pool the selected audit cases without weighting. `outcome` preserves archived classes; `error_class` splits missing-reference flagged/missed cases from typology errors and under-predictions; `reference_group` separates reference availability. `binary_outcome` conditions on binary correctness (470 correct, 530 incorrect). Every grouping is also reported separately by model. These overlapping grouping tables must not be summed.

## Overall checks

Rows: 1000. Binary-correct: 470/1000. Typed-correct: 183/456. Illicit reference absent: 240/1000. Truncated at 8,000 characters: 400/1000. Empty reasoning traces: 148/1000.

| Window | Lexical rule | Correctness convention | Conjunction / eligible rows |
|---|---|---|---|
| archived | original_PRM | strict_archived | 538/1000 |
| archived | original_PRM | binary | 303/1000 |
| archived | original_PRM | exclude_illicit_reference_absent | 421/760 |
| archived | original_PRM | typed_illicit_only | 184/456 |
| full | original_PRM | strict_archived | 538/1000 |
| full | original_PRM | binary | 303/1000 |
| full | original_PRM | exclude_illicit_reference_absent | 421/760 |
| full | original_PRM | typed_illicit_only | 184/456 |
| full | PR_and_two_marker_categories | strict_archived | 533/1000 |
| full | PR_and_two_marker_categories | binary | 298/1000 |
| full | PR_and_two_marker_categories | exclude_illicit_reference_absent | 417/760 |
| full | PR_and_two_marker_categories | typed_illicit_only | 180/456 |
| full | P_canonical_R_and_two_marker_categories | strict_archived | 533/1000 |
| full | P_canonical_R_and_two_marker_categories | binary | 298/1000 |
| full | P_canonical_R_and_two_marker_categories | exclude_illicit_reference_absent | 417/760 |
| full | P_canonical_R_and_two_marker_categories | typed_illicit_only | 180/456 |
| prefix8000 | original_PRM | strict_archived | 531/1000 |
| prefix8000 | original_PRM | binary | 301/1000 |
| prefix8000 | original_PRM | exclude_illicit_reference_absent | 417/760 |
| prefix8000 | original_PRM | typed_illicit_only | 182/456 |
| prefix8000 | PR_and_two_marker_categories | strict_archived | 525/1000 |
| prefix8000 | PR_and_two_marker_categories | binary | 295/1000 |
| prefix8000 | PR_and_two_marker_categories | exclude_illicit_reference_absent | 412/760 |
| prefix8000 | PR_and_two_marker_categories | typed_illicit_only | 177/456 |
| prefix8000 | P_canonical_R_and_two_marker_categories | strict_archived | 525/1000 |
| prefix8000 | P_canonical_R_and_two_marker_categories | binary | 295/1000 |
| prefix8000 | P_canonical_R_and_two_marker_categories | exclude_illicit_reference_absent | 412/760 |
| prefix8000 | P_canonical_R_and_two_marker_categories | typed_illicit_only | 177/456 |
| archived_deepseek_prefix8000 | judge_PRM_all_rows | judge_conclude | 527/1000 |
| archived_deepseek_prefix8000 | judge_PRM_scored_only | judge_conclude | 527/852 |
| archived_opus_prefix8000 | judge_PRM_all_rows | judge_conclude | 365/1000 |
| archived_opus_prefix8000 | judge_PRM_scored_only | judge_conclude | 365/852 |
| archived_gemini_prefix8000 | judge_PRM_all_rows | judge_conclude | 512/1000 |
| archived_gemini_prefix8000 | judge_PRM_scored_only | judge_conclude | 512/852 |

| Step | Full passes | Prefix passes | Full-only | Prefix-only |
|---|---:|---:|---:|---:|
| parse | 804 | 803 | 1 | 0 |
| recall | 828 | 827 | 1 | 0 |
| match | 757 | 747 | 10 | 0 |
| marker_ge2 | 729 | 717 | 12 | 0 |
| canonical_recall | 828 | 827 | 1 | 0 |

## Files

`inputs/trace_features.csv` and `inputs/provenance.json` are the portable archived-text derivatives and their provenance. `items.csv` joins those features to fixed labels, eligibility indicators, and reference groups. `rates.csv` contains outcome-conditioned lexical marker and correctness rates; `conjunctions.csv` contains joint descriptive counts; `truncation.csv` compares paired full/prefix scores; `judge_discordance.csv` contains parser/judge contingency counts. `analysis.json` records hashes, integrity checks, and overall tables. Portable reproduction verifies input hashes before reporting.
