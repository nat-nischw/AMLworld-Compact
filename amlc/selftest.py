"""One command that proves a clone can score the benchmark correctly.

SWE-bench's install check is the model: run the shipped gold predictions, score
them, and compare against a published number. If that passes, the environment,
the download, the loader and the scorer are all working, and the user has not
spent an afternoon discovering otherwise on their own model.

Here the gold predictions are the supervised ensemble probabilities that ship
with the dataset, thresholded at the operating point the paper reports.

    python -m amlc.selftest          # both splits
    make selftest

The targets and HT weights were selected with a separate, frozen construction
scorer. This check validates the two-booster evaluation export at its inherited
thresholds; it does not re-estimate the sampling design or establish unbiased
precision/F1 ratios. HT weights are required when evaluating other predictors.
"""

from __future__ import annotations

import sys

import numpy as np

from . import config, hub
from .triage.doubt_triage import ht_weighted_prf

#: Two-booster evaluation on the frozen released targets; no coreset rebuild.
EXPECTED = {
    "HI-Small": (84.8449, 56.8345, 68.0708),
    "LI-Small": (60.1695, 18.7831, 28.6290),
}

#: Absolute tolerance in percentage points. The arithmetic is deterministic, so
#: this is a float-formatting allowance and not an experimental one.
TOL = 1e-3


def check(dataset: str) -> tuple[bool, str]:
    """Score the shipped ensemble on one split and compare with the paper."""
    d = hub.load_coreset(dataset)

    n_full = config.N_TEST_FULL[dataset]
    total = d["weights"].sum()
    if not np.isclose(total, n_full, rtol=0, atol=1e-6):
        return False, (f"{dataset}: weights sum to {total:,.1f}, expected "
                       f"{n_full:,}. Weighted metrics from this vector are wrong.")

    preds = (d["ensemble_probs"] >= d["ml_threshold"]).astype(int)
    got = tuple(100 * v for v in ht_weighted_prf(preds, d["labels"], d["weights"]))
    want = EXPECTED[dataset]

    worst = max(abs(a - b) for a, b in zip(got, want))
    line = (f"{dataset}: n={d['n']:,}  tau={d['ml_threshold']}  "
            f"P/R/F1 = {got[0]:.4f} / {got[1]:.4f} / {got[2]:.4f}")
    if worst > TOL:
        return False, (f"{line}\n    expected {want[0]:.4f} / {want[1]:.4f} / "
                       f"{want[2]:.4f}, off by {worst:.4f} pp")
    return True, line


def main(argv=None) -> int:
    datasets = list(argv) if argv else list(config.DATASETS)
    ok = True
    for dataset in datasets:
        try:
            passed, line = check(dataset)
        except Exception as exc:  # broad on purpose: reported below, not swallowed
            passed, line = False, f"{dataset}: {type(exc).__name__}: {exc}"
        print(f"  {'PASS' if passed else 'FAIL'}  {line}")
        ok &= passed

    if ok:
        print("\nself-test passed: loaded arrays, weight sums, and ensemble metrics match.")
        print("Use the original HT weights to estimate full-split counts; the")
        print("two-booster evaluator does not redefine the sampling design.")
    else:
        print("\nself-test failed. Check the reported mismatch and the dataset files")
        print("selected by AMLC_CORESET_DIR or AMLC_DATASET.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
