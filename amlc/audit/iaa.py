#!/usr/bin/env python3
"""Fleiss' kappa across all four rubric annotators, plus the six pairwise
Cohen's kappas, on the HI-Small trace sample.

The annotators are the regex heuristic and the three frontier judges named in
``config.JUDGE_PROVIDERS``. Sample size is ``config.RUBRIC_N_TRACES``.

Outputs, under ``results/audit/`` unless ``AMLC_AUDIT_OUT`` says otherwise:
  multi_judge_iaa_n<N>.csv
    columns: step, n_judges, n_traces, fleiss_kappa, mean_pair_kappa,
             min_pair_kappa, max_pair_kappa
  multi_judge_pairwise_n<N>.csv
    columns: step, judge_a, judge_b, kappa, raw_agreement_pct
"""
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    JUDGE_COLUMN_PREFIX, JUDGE_PROVIDERS, RUBRIC_N_TRACES, RUBRIC_STEPS,
)

# Input and output directories are separate so a recomputation never silently
# overwrites the annotation files it read, which is how the shipped IAA table
# came to be stamped 22 minutes older than the CSVs it summarises.
REPO_ROOT = Path(__file__).resolve().parents[2]
STATS = Path(os.environ.get("AMLC_AUDIT_IN", REPO_ROOT / "results" / "audit"))
OUT = Path(os.environ.get("AMLC_AUDIT_OUT", STATS))
OUT.mkdir(parents=True, exist_ok=True)

_N = RUBRIC_N_TRACES

# The regex heuristic writes the unprefixed columns; each frontier judge writes
# its own prefix. Both the filenames and the prefixes come from config so that
# adding a judge there is enough.
JUDGE_FILES = {
    "regex": (f"trace_4step_annotations_n{_N}.csv", ""),
    **{p: (f"trace_4step_{p}_n{_N}.csv", JUDGE_COLUMN_PREFIX[p])
       for p in JUDGE_PROVIDERS},
}

STEPS = list(RUBRIC_STEPS)


def cohen_kappa(a, b):
    a = np.asarray(a).astype(int); b = np.asarray(b).astype(int)
    p_o = float((a == b).mean())
    p_a1 = a.mean(); p_b1 = b.mean()
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(M):
    """Fleiss' kappa.  M is N x k where M[i,j] is the count of raters
    assigning item i to category j.  Total raters per item = n (constant)."""
    N = M.shape[0]
    n = M.sum(axis=1)[0]
    p_j = M.sum(axis=0) / (N * n)
    P_e = float((p_j ** 2).sum())
    P_i = (M * (M - 1)).sum(axis=1) / (n * (n - 1))
    P_bar = float(P_i.mean())
    if P_e >= 1.0:
        return 1.0 if P_bar >= 1.0 else 0.0
    return (P_bar - P_e) / (1 - P_e)


def load_judge(name: str) -> pd.DataFrame:
    fname, prefix = JUDGE_FILES[name]
    path = STATS / fname
    df = pd.read_csv(path)
    cols = {}
    for s in STEPS:
        col = f"{prefix}{s}"
        if col not in df.columns:
            raise KeyError(f"{name}: missing column {col}")
        cols[col] = f"{name}_{s}"
    keep = ["model", "case_id"] + list(cols.keys())
    return df[keep].rename(columns=cols)


def main():
    judges = list(JUDGE_FILES.keys())
    print(f"Loading {len(judges)} judges...")
    frames = [load_judge(j) for j in judges]
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["model", "case_id"], how="inner")
    print(f"Merged n={len(merged)} traces across all {len(judges)} judges")

    pair_rows = []
    for a, b in itertools.combinations(judges, 2):
        for s in STEPS:
            ka = merged[f"{a}_{s}"].astype(int).values
            kb = merged[f"{b}_{s}"].astype(int).values
            kappa = cohen_kappa(ka, kb)
            agree = float((ka == kb).mean()) * 100
            pair_rows.append({"step": s, "judge_a": a, "judge_b": b,
                              "kappa": round(kappa, 4),
                              "raw_agreement_pct": round(agree, 2)})
    pair_df = pd.DataFrame(pair_rows)
    pair_path = OUT / f"multi_judge_pairwise_n{_N}.csv"
    pair_df.to_csv(pair_path, index=False)
    print(f"\nSaved pairwise IAA -> {pair_path}")

    fleiss_rows = []
    for s in STEPS:
        cols = [f"{j}_{s}" for j in judges]
        v = merged[cols].astype(int).values
        n = len(judges)
        N = v.shape[0]
        cnt1 = v.sum(axis=1)
        cnt0 = n - cnt1
        M = np.column_stack([cnt0, cnt1])
        fk = fleiss_kappa(M)
        pair_subset = pair_df[pair_df["step"] == s]
        mean_k = float(pair_subset["kappa"].mean())
        min_k = float(pair_subset["kappa"].min())
        max_k = float(pair_subset["kappa"].max())
        fleiss_rows.append({
            "step": s, "n_judges": n, "n_traces": N,
            "fleiss_kappa": round(fk, 4),
            "mean_pair_kappa": round(mean_k, 4),
            "min_pair_kappa": round(min_k, 4),
            "max_pair_kappa": round(max_k, 4),
        })

    fleiss_df = pd.DataFrame(fleiss_rows)
    fleiss_path = OUT / f"multi_judge_iaa_n{_N}.csv"
    fleiss_df.to_csv(fleiss_path, index=False)
    print(f"\n{'='*72}")
    print(f"  {len(judges)}-Judge IAA Summary  (n={len(merged)} traces)")
    print(f"{'='*72}")
    print(f"  judges: {', '.join(judges)}")
    print()
    print(f"  {'Step':10s}  {'Fleiss κ':>9s}  {'mean pair κ':>11s}  {'min pair κ':>10s}  {'max pair κ':>10s}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*11}  {'-'*10}  {'-'*10}")
    for r in fleiss_rows:
        print(f"  {r['step']:10s}  {r['fleiss_kappa']:9.3f}  {r['mean_pair_kappa']:11.3f}  {r['min_pair_kappa']:10.3f}  {r['max_pair_kappa']:10.3f}")
    print(f"\nSaved Fleiss summary -> {fleiss_path}")

    print(f"\n{'='*72}")
    print(f"  Pass-rate replication across {len(judges)} judges")
    print(f"{'='*72}")
    # Header and row widths follow `judges`, so a fifth annotator prints without
    # editing the format strings.
    print("  " + f"{'Step':10s}" + "".join(f"  {j:>9s}" for j in judges))
    print("  " + "-" * 10 + "".join("  " + "-" * 9 for _ in judges))
    for s in STEPS:
        rates = [merged[f"{j}_{s}"].astype(int).mean() * 100 for j in judges]
        print("  " + f"{s:10s}" + "".join(f"  {r:8.1f}%" for r in rates))


if __name__ == "__main__":
    main()
