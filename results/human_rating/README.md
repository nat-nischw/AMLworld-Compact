# Human-rating pack

Three annotators independently rated the same 28-trace stratified slice:
one NLP researcher and two AML practitioners, one junior and one senior.
The slice contains one HI-Small few-shot trace from run 42 per cell of
7 models × 4 outcome classes (correct-illicit, over-prediction,
typology-error, under-prediction). The archived API requests did not set
this identifier as a sampling seed.

| file | rater |
|---|---|
| `rater_nlp.csv` | NLP researcher |
| `rater_aml_practitioner_1.csv` | AML practitioner, junior |
| `rater_aml_practitioner_2.csv` | AML practitioner, senior |
| `per_trace_NN_amlc_NNNNN.txt` | the trace each rater read, one file per case |

Parse, Recall, and Match are human ratings. In the adopted sheets, Conclude
is replaced with the same parser-derived score for all three raters: one for
a correct benign verdict, or an illicit verdict with a present, matching
reference typology; zero otherwise. Missing illicit references count as
failures under this archived convention.

Conclude therefore has the same 21.4% pass rate and Fleiss kappa of 1.00 in
all three sheets. This agreement follows from the shared score and is not
independent human validation of the answer. Human Match agreement is low
(Fleiss kappa = 0.10).

## Sheets not included

The senior practitioner rescored after a calibration pass. Only the adopted
sheets ship. Superseded sheets, the blank template, and the sheet builder
remain in the authors' archive.
