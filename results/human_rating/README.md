# Human-rating pack

Three annotators independently scored the same 28-trace stratified slice with the four-step
rubric: one NLP researcher and two AML practitioners, one junior and one senior. The slice is
7 models x 4 outcome classes (correct-illicit, over-prediction, typology-error,
under-prediction), one trace per cell, all HI-Small seed 42.

| file | rater |
|---|---|
| `rater_nlp.csv` | NLP researcher |
| `rater_aml_practitioner_1.csv` | AML practitioner, junior |
| `rater_aml_practitioner_2.csv` | AML practitioner, senior |
| `per_trace_NN_amlc_NNNNN.txt` | the trace each rater read, one file per case |

All three report a Conclude pass rate of 21.4% and agree unanimously on Conclude for all 28
traces, which is the Fleiss kappa of 1.00 the appendix cites.

Conclude is binarised under a uniform parser-extraction rule: 1 if the extraction pipeline's
final prediction matches ground truth on both the binary verdict and the typology label where
one exists, 0 otherwise.

## Sheets not included

The senior practitioner rescored after a calibration pass. The first pass gave a Conclude
rate of 32.7%; the adopted sheet gives 21.4%. Only the adopted sheet ships. The superseded
sheets, the blank template and the sheet builder are retained in the authors' archive rather
than the release, since publishing several conflicting versions of the same rater's judgement
invites the wrong reading. The rescore is disclosed here because the difference is material.
