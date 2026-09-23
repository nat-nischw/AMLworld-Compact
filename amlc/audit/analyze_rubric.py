"""Pass rates, Wilson intervals and attribution for the four-step audit slice.

This is the stage that turns two annotators' step scores into the numbers the
paper quotes: the per-step pass rate with a 95% interval, the count of traces
that clear Parse, Recall and Match yet fail Conclude, the per-model heatmap
values, and which step each outcome class fails first.

Intervals are Wilson score intervals rather than normal approximations. At
n = 1,000 the difference is small, but the Conclude rate sits near 20% and the
per-outcome cells go to 0%, where the normal interval runs below zero and the
Wilson interval does not.

Reads
    ``results/audit/trace_4step_annotations_n1000.csv`` (annotator A, the regex
    heuristic) and one judge table, by default
    ``results/audit/trace_4step_deepseek_n1000.csv`` (annotator B). Both are
    produced upstream by :mod:`amlc.audit.sample_traces` and
    :mod:`amlc.audit.judges`.

Writes
    ``results/audit/rubric_n1000_summary.json``, and prints the same numbers.

Outcome vocabulary
------------------
The classes come from
:data:`amlc.audit.sample_traces.OUTCOMES`, and the older ``correct_benign``
spelling is mapped onto the canonical ``correct_legit`` on read, so an older
CSV still reports.

The judge column prefix (``ds_``, ``op_``, ``gm_``) is resolved from the file,
so any of the three judges can stand in as annotator B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .iaa import cohen_kappa
from .sample_traces import OUTCOMES, normalise_outcome

#: 95% two-sided normal quantile.
Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as a fraction."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def judge_prefix(frame: pd.DataFrame) -> str:
    """Column prefix a judge table uses, or "" for the regex annotator."""
    for prefix in ("", "ds_", "op_", "gm_"):
        if all(f"{prefix}{step}" in frame.columns for step in config.RUBRIC_STEPS):
            return prefix
    raise KeyError(f"no rubric step columns in {list(frame.columns)}")


def steps_as_int(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """The four step columns as 0/1 integers, under their bare names."""
    return pd.DataFrame(
        {step: frame[f"{prefix}{step}"].astype(int).values
         for step in config.RUBRIC_STEPS},
        index=frame.index)


def first_fail(scores: pd.DataFrame) -> pd.Series:
    """First failing step per row, in pipeline order, or ``all_pass``."""
    out = pd.Series("all_pass", index=scores.index, dtype=object)
    for step in reversed(config.RUBRIC_STEPS):
        out[scores[step] == 0] = step
    return out


def analyse(a_frame: pd.DataFrame, b_frame: pd.DataFrame,
            b_name: str = "deepseek") -> dict:
    """Compare two annotators over the traces they both scored."""
    b_prefix = judge_prefix(b_frame)
    key = ["model", "case_id"]
    merged = a_frame.merge(b_frame, on=key, suffixes=("_a", "_b"))
    n = len(merged)

    a = steps_as_int(merged)
    b = steps_as_int(merged, b_prefix)

    summary = {"n": int(n), "annotator_a": "regex", "annotator_b": b_name,
               "per_step": {}}
    print(f"Aligned {n} traces scored by both annotators\n")
    print(f"  {'step':<10s} {'A (regex)':<22s} {f'B ({b_name})':<22s} "
          f"{'A and B':<22s} {'kappa':>8s}")
    print("  " + "-" * 76)

    for step in config.RUBRIC_STEPS:
        k_a, k_b = int(a[step].sum()), int(b[step].sum())
        k_and = int((a[step] & b[step]).sum())
        lo_a, hi_a = wilson_ci(k_a, n)
        lo_b, hi_b = wilson_ci(k_b, n)
        lo_and, hi_and = wilson_ci(k_and, n)
        kappa = cohen_kappa(a[step].values, b[step].values)

        print(f"  {step:<10s} "
              f"{f'{k_a / n * 100:5.1f}% [{lo_a * 100:4.1f},{hi_a * 100:4.1f}]':<22s} "
              f"{f'{k_b / n * 100:5.1f}% [{lo_b * 100:4.1f},{hi_b * 100:4.1f}]':<22s} "
              f"{f'{k_and / n * 100:5.1f}% [{lo_and * 100:4.1f},{hi_and * 100:4.1f}]':<22s} "
              f"{kappa:+8.3f}")

        summary["per_step"][step] = {
            "A_pass": k_a, "A_pass_pct": k_a / n * 100,
            "A_ci": [lo_a * 100, hi_a * 100],
            "B_pass": k_b, "B_pass_pct": k_b / n * 100,
            "B_ci": [lo_b * 100, hi_b * 100],
            "AND_pass": k_and, "AND_pass_pct": k_and / n * 100,
            "AND_ci": [lo_and * 100, hi_and * 100],
            "cohen_kappa": float(kappa),
        }

    # The headline: procedurally complete traces that still get the verdict
    # wrong. This is the "vocabulary without discrimination" count.
    print("\n  Pass Parse, Recall and Match yet fail Conclude")
    for label, scores in (("A (regex)", a), (f"B ({b_name})", b)):
        complete = (scores["parse"] == 1) & (scores["recall"] == 1) & (scores["match"] == 1)
        wrong = complete & (scores["conclude"] == 0)
        count = int(wrong.sum())
        lo, hi = wilson_ci(count, n)
        print(f"    {label:<16s} {count}/{n}  "
              f"({count / n * 100:.1f}%  [{lo * 100:.1f}, {hi * 100:.1f}])")
        side = "a" if scores is a else "b"
        summary[f"prm_yet_fail_conclude_{side}"] = {
            "count": count, "pct": count / n * 100, "ci": [lo * 100, hi * 100]}

    per_model = pd.concat([merged["model"], a], axis=1).groupby("model").mean() * 100
    print("\n  Per-model pass rate (annotator A)")
    print(per_model.round(1).to_string())
    summary["per_model_A"] = per_model.round(2).to_dict(orient="index")

    outcome = (merged["outcome_a"] if "outcome_a" in merged.columns
               else merged["outcome"]).map(normalise_outcome)
    fails = first_fail(a)
    print("\n  First failing step by outcome class (annotator A)")
    summary["failmode_step_A"] = {}
    for name in OUTCOMES:
        rows = outcome == name
        if not rows.any():
            continue
        counts = fails[rows].value_counts().to_dict()
        print(f"    {name:<20s} (n={int(rows.sum())}): {counts}")
        summary["failmode_step_A"][name] = {"n": int(rows.sum()),
                                            "first_fail_counts": counts}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit-dir", type=Path, default=None,
                    help="directory holding the annotation CSVs "
                         "(default: <repo>/results/audit)")
    ap.add_argument("--judge", default="deepseek",
                    help="which judge table stands in as annotator B")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from ..paths import ensure, results

    audit = Path(args.audit_dir) if args.audit_dir else ensure(results() / "audit")
    n = config.RUBRIC_N_TRACES
    a_frame = pd.read_csv(audit / f"trace_4step_annotations_n{n}.csv")
    b_frame = pd.read_csv(audit / f"trace_4step_{args.judge}_n{n}.csv")

    summary = analyse(a_frame, b_frame, args.judge)

    path = args.out or audit / f"rubric_n{n}_summary.json"
    Path(path).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
