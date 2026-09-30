# Archived predictions by target presence

Reproduce the descriptive tables without model calls:

```bash
python scripts/30_score_target_presence.py
```

Inputs are the shipped target-integrity `cases.csv` and the portable five-run
prediction tensors used by the sampling-uncertainty analysis. Every case is
joined by dataset and canonical case ID, with transaction-ID equality checked.

`per_run.csv` reports benign flag rate (FP / benign cases) and illicit recall
(TP / illicit cases), separately for target-present and target-absent inputs.
`summary.csv` averages each rate across five fixed runs and reports run SD.
All metric values are fractions. No requests are removed because their outputs
were empty or unscorable; the historical binary prediction rule is unchanged.

These are unweighted subset diagnostics, with a separate denominator for each
class and presence group. Differences may reflect graph size, input length,
case difficulty, or request rejection. Target-present inputs still leave the
target unmarked. The tables therefore do not estimate a causal effect of target
omission or performance on the repaired input version. No primary score is
replaced by a present-only score.
